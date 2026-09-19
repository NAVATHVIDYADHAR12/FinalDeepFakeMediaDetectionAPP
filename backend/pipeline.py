"""Image analysis orchestration - the path a single still image takes.

    load -> detect faces -> classify each face -> heatmap -> forensics -> report

Kept separate from the web layer so it can be run and tested without a server.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import config as cfg
import forensics
from detector import colorize_heatmap


def _b64_jpeg(image_bgr: np.ndarray, quality: int = 85, max_side: int = 512) -> str:
    """Encode a preview as a data URI, downscaled to a sensible display size.

    Previews are embedded directly in the JSON report, so a full-resolution
    frame would bloat every response - a 1536px full-frame crop encodes to
    ~158 KB, and a report can carry two previews per face.
    """
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def analyze_image(path: Path, detector, face_analyzer, want_previews: bool = True,
                  ai_detector=None) -> dict:
    """Full analysis of one image file. Returns the report dict the UI renders."""
    started = time.perf_counter()
    timeline: list[dict] = []

    def mark(stage: str) -> None:
        timeline.append({
            "stage": stage,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        })

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path.name}")
    mark("File loaded")

    height, width = image.shape[:2]

    faces = face_analyzer.detect(image)
    mark("Face detection")

    face_reports: list[dict] = []
    # A face-manipulation model is out of domain on a full scene. Treating its
    # softmax output as evidence made AI-generated scenes look "95% authentic".
    # Only weights explicitly trained for AI generation may score a no-face
    # full frame.
    full_frame_detector = (
        ai_detector if ai_detector is not None and ai_detector.ready
        and ai_detector.supports("ai_generation")
        else detector if detector.supports("ai_generation") else None
    )
    can_score_full_frame = full_frame_detector is not None
    targets = (
        [(f, face_analyzer.crop(image, f["bbox"])) for f in faces]
        if faces else ([(None, image)] if can_score_full_frame else [])
    )

    generation_result = None
    for idx, (face, crop) in enumerate(targets):
        active_detector = full_frame_detector if face is None else detector
        result = active_detector.predict(crop, want_heatmap=True)
        heatmap = result.pop("heatmap", None)
        if face is None and active_detector is full_frame_detector:
            # Keep a clean classifier result for the dedicated generation
            # section; the face entry below adds UI-only crop/bbox fields.
            generation_result = dict(result)

        entry = {
            "face_id": idx + 1,
            "is_full_frame": face is None,
            "bbox": face["bbox"] if face else [0, 0, float(width), float(height)],
            "detection_score": round(face["score"], 4) if face else None,
            "landmarks": face["landmarks"] if face else None,
            **result,
        }

        if want_previews:
            entry["crop_preview"] = _b64_jpeg(crop)
            if heatmap is not None:
                entry["heatmap_preview"] = _b64_jpeg(colorize_heatmap(heatmap, crop))

        if face is not None:
            emb = face_analyzer.embed(image, face["_raw"])
            entry["has_embedding"] = emb is not None
            entry["_embedding"] = emb.tolist() if emb is not None else None

        face_reports.append(entry)

    if not faces and not can_score_full_frame:
        entry = {
            "face_id": 1,
            "is_full_frame": True,
            "bbox": [0, 0, float(width), float(height)],
            "detection_score": None,
            "landmarks": None,
            "fake_probability": None,
            "authenticity_score": None,
            "verdict": "UNVERIFIED",
            "risk_level": "UNKNOWN",
            "confidence": None,
            "model_agreement": None,
            "models": [],
        }
        if want_previews:
            entry["crop_preview"] = _b64_jpeg(image)
        face_reports.append(entry)

    mark("Model inference")

    # Overall verdict: the most suspicious face drives the result. One forged
    # face in a group photo makes the whole image manipulated.
    probs = [f["fake_probability"] for f in face_reports
             if f.get("fake_probability") is not None]

    # A dedicated AI-generation model examines the whole image even when faces
    # are present. This catches synthetic portraits while the face bank remains
    # responsible for swaps and reenactment artifacts.
    if faces and full_frame_detector is not None and full_frame_detector is not detector:
        generation_result = full_frame_detector.predict(image, want_heatmap=False)
        probs.append(generation_result["fake_probability"])
    overall = float(max(probs)) if probs else None

    face_probs = [f["fake_probability"] for f in face_reports
                  if not f.get("is_full_frame") and f.get("fake_probability") is not None]
    generation_wins = (
        generation_result is not None
        and generation_result.get("fake_probability") is not None
        and (not face_probs or generation_result["fake_probability"] >= max(face_probs))
    )
    if overall is None:
        overall_confidence_kind = None
        overall_confidence = None
    elif generation_wins:
        overall_confidence_kind = generation_result.get("confidence_kind")
        overall_confidence = generation_result.get("confidence")
    else:
        winning_face = next(
            (f for f in face_reports
             if not f.get("is_full_frame") and f.get("fake_probability") == overall),
            None,
        )
        overall_confidence_kind = (
            winning_face.get("confidence_kind", "uncalibrated_model_certainty")
            if winning_face else "uncalibrated_model_certainty"
        )
        overall_confidence = winning_face.get("confidence") if winning_face else None

    metadata = forensics.read_metadata(path)
    mark("Metadata extraction")

    c2pa = forensics.check_c2pa(path)
    mark("Provenance check")

    ela = forensics.error_level_analysis(path)
    mark("Error level analysis")

    input_quality = forensics.assess_input_quality(path, image)
    mark("Input quality assessment")
    quality_limited = input_quality["reliability"] == "LIMITED"

    agreements = [f["model_agreement"] for f in face_reports
                  if f.get("model_agreement") is not None]
    if generation_result is not None and generation_result.get("model_agreement") is not None:
        agreements.append(generation_result["model_agreement"])

    findings = forensics.build_findings(
        {"fake_probability": overall,
         "model_agreement": min(agreements, default=None),
         "supported": overall is not None,
         "signal": "ai_generation" if generation_wins else "face_manipulation"},
        metadata, c2pa, ela, len(faces), input_quality,
    )
    mark("Report generated")

    # Per-model rollup for the dashboard's comparison panel
    model_rollup: list[dict] = []
    scored_faces = [f for f in face_reports
                    if not f.get("is_full_frame") and f.get("models")]
    if scored_faces:
        for i, m in enumerate(scored_faces[0]["models"]):
            worst = max(f["models"][i]["fake_probability"] for f in scored_faces)
            model_rollup.append({
                "arch": m["arch"],
                "name": m["name"],
                "fake_probability": round(worst, 4),
                "verdict": cfg.verdict_from_score(worst),
                "confidence": round(abs(worst - 0.5) * 2, 4),
                "test_accuracy": m.get("test_accuracy"),
                "test_auc": m.get("test_auc"),
            })
    if generation_result is not None:
        for m in generation_result["models"]:
            model_rollup.append({
                **m,
                "arch": f"ai:{m['arch']}",
                "name": f"{m['name']} (AI generation)",
            })

    return {
        "scan_id": f"SCN{uuid.uuid4().hex[:6].upper()}",
        "media_type": "image",
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "dimensions": f"{width}x{height}",
        "width": width,
        "height": height,
        "fake_probability": round(overall, 4) if overall is not None else None,
        "authenticity_score": round((1 - overall) * 100, 1) if overall is not None else None,
        "verdict": (
            "UNVERIFIED" if overall is None or quality_limited
            else cfg.verdict_from_score(overall)
        ),
        "risk_level": (
            "UNKNOWN" if overall is None or quality_limited
            else cfg.risk_from_score(overall)
        ),
        "decision_reliability": input_quality["reliability"],
        "confidence": overall_confidence,
        "confidence_kind": overall_confidence_kind,
        "analysis_scope": [
            *([detector.task] if faces else []),
            *(["ai_generation"] if full_frame_detector is not None else []),
        ],
        "generation_analysis": generation_result,
        "faces_detected": len(faces),
        "faces": face_reports,
        "models": model_rollup,
        "metadata": metadata,
        "c2pa": c2pa,
        "ela": ela,
        "input_quality": input_quality,
        "findings": findings,
        "timeline": timeline,
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
    }
