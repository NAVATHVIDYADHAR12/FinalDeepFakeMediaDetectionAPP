"""Classical (non-neural) forensic signals.

These do not use the trained model at all. They are cheap, independent checks
that either corroborate the network or flag things it cannot see - a file that
has had its metadata scrubbed, or that carries no provenance credential.

Deliberately conservative: each check reports what it observed, and absence of
evidence is reported as "not found" rather than as guilt.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageChops, ExifTags

import config as cfg

# JUMBF/C2PA content-credential markers, as embedded by Adobe / camera vendors
_C2PA_MARKERS = (b"c2pa", b"jumbf", b"urn:uuid:c2pa", b"contentauth")

_INTERESTING_EXIF = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "GPSInfo", "Artist", "Copyright",
}


def read_metadata(path: Path) -> dict:
    """EXIF summary plus a judgement on whether metadata looks stripped.

    Cameras and phones write a rich EXIF block. Generative models write none,
    and re-encoding through most editors or social platforms destroys it. So
    "no metadata" is a weak signal on its own but meaningful alongside others.
    """
    out: dict = {
        "has_exif": False,
        "metadata_stripped": True,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime": None,
        "has_gps": False,
        "fields": {},
    }

    try:
        with Image.open(path) as img:
            out["format"] = img.format
            out["dimensions"] = f"{img.width}x{img.height}"
            out["mode"] = img.mode

            exif = img.getexif()
            if not exif:
                return out

            tags = {}
            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if name not in _INTERESTING_EXIF:
                    continue
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                tags[name] = str(value)[:200]

            out["has_exif"] = True
            out["fields"] = tags
            out["camera_make"] = tags.get("Make")
            out["camera_model"] = tags.get("Model")
            out["software"] = tags.get("Software")
            out["datetime"] = tags.get("DateTimeOriginal") or tags.get("DateTime")
            out["has_gps"] = "GPSInfo" in tags
            # Real capture data present -> metadata was not scrubbed
            out["metadata_stripped"] = not (out["camera_make"] or out["datetime"])
    except Exception as exc:                               # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out


def check_c2pa(path: Path) -> dict:
    """Look for a C2PA / Content Credentials provenance manifest.

    A full C2PA verification requires cryptographic validation against a trust
    list. This checks only for the manifest's presence, and says so - it is
    honest about being a presence check, not a signature check.
    """
    try:
        blob = path.read_bytes()[:2_000_000]
        lowered = blob.lower()
        found = [m.decode() for m in _C2PA_MARKERS if m in lowered]
        present = bool(found)
        return {
            "signature_found": present,
            "markers": found,
            "status": "PRESENT" if present else "NOT_FOUND",
            "note": (
                "A provenance manifest is embedded. Cryptographic validation "
                "against a trust list is not performed."
                if present else
                "No content credential found. Most cameras and editors do not "
                "yet write one, so this is not evidence of manipulation."
            ),
        }
    except Exception as exc:                               # noqa: BLE001
        return {"signature_found": False, "status": "ERROR", "note": str(exc)}


def error_level_analysis(path: Path, quality: int = 92) -> dict:
    """Error Level Analysis.

    Re-saves the image at a known JPEG quality and measures how much each region
    changes. Untouched regions sit near their compression fixed point and barely
    move; regions that were pasted, generated, or edited have a different
    compression history and move more. High variance across the frame therefore
    hints at splicing.

    Weak on PNGs and on heavily re-compressed images - reported, not hidden.
    """
    try:
        with Image.open(path) as img:
            original = img.convert("RGB")

            buf = io.BytesIO()
            original.save(buf, "JPEG", quality=quality)
            buf.seek(0)
            with Image.open(buf) as resaved:
                diff = ImageChops.difference(original, resaved)

            arr = np.asarray(diff, dtype=np.float32)
            peak = float(arr.max()) or 1.0
            norm = arr / peak

            mean_err = float(norm.mean())
            std_err = float(norm.std())
            # Block-level spread: splices show up as a few hot tiles
            gray = norm.mean(axis=2)
            h, w = gray.shape
            bh, bw = max(1, h // 8), max(1, w // 8)
            blocks = [
                float(gray[y:y + bh, x:x + bw].mean())
                for y in range(0, h - bh + 1, bh)
                for x in range(0, w - bw + 1, bw)
            ]
            block_spread = float(np.std(blocks)) if blocks else 0.0

            suspicious = block_spread > 0.055
            return {
                "mean_error": round(mean_err, 5),
                "std_error": round(std_err, 5),
                "block_spread": round(block_spread, 5),
                "suspicious_regions": suspicious,
                "note": (
                    "Uneven compression response across the frame, consistent "
                    "with a spliced or edited region."
                    if suspicious else
                    "Compression response is uniform across the frame."
                ),
            }
    except Exception as exc:                               # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def assess_input_quality(path: Path, image_bgr: np.ndarray) -> dict:
    """Flag inputs outside the detector's comfortable operating range.

    These are warning heuristics, not authenticity signals. In particular,
    compression and blur must never be counted as evidence that an image is AI.
    They only tell the UI that the neural probability is less dependable.
    """
    height, width = image_bgr.shape[:2]
    short_edge = min(width, height)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bits_per_pixel = path.stat().st_size * 8 / max(1, width * height)
    suffix = path.suffix.lower()

    limitations: list[str] = []
    if short_edge < 384:
        limitations.append(
            f"short edge is {short_edge}px; the detector must upscale it to 384px"
        )
    if blur_variance < 20:
        limitations.append("image is heavily blurred or contains very little fine detail")
    if suffix in {".jpg", ".jpeg", ".jfif"} and bits_per_pixel < 0.45:
        limitations.append("JPEG is extremely compressed for its dimensions")

    return {
        "reliability": "LIMITED" if limitations else "STANDARD",
        "short_edge_px": short_edge,
        "blur_variance": round(blur_variance, 2),
        "bits_per_pixel": round(bits_per_pixel, 3),
        "limitations": limitations,
        "note": (
            "Input quality may make the neural probability unstable. Compression, "
            "blur, and low resolution are not evidence of AI generation."
            if limitations else
            "No obvious resolution, blur, or extreme-compression limitation detected."
        ),
    }


def build_findings(detection: dict, metadata: dict, c2pa: dict,
                   ela: dict, face_count: int, quality: dict | None = None) -> list[dict]:
    """Assemble the 'Key Findings' bullets shown on the report page.

    Severity drives the dot colour in the UI: high = red, medium = amber,
    info = blue.
    """
    findings: list[dict] = []
    fake_p = detection.get("fake_probability")

    if not detection.get("supported", True) or fake_p is None:
        findings.append({
            "severity": "medium",
            "text": ("Neural verdict unavailable: the loaded models detect face swaps, "
                     "not full-scene AI generation"),
        })
    elif fake_p >= cfg.FAKE_THRESHOLD:
        label = ("AI-generation signal" if detection.get("signal") == "ai_generation"
                 else "Face manipulation")
        findings.append({
            "severity": "high",
            "text": f"{label} detected ({fake_p * 100:.0f}% model probability)",
        })
    elif fake_p >= cfg.SUSPICIOUS_THRESHOLD:
        findings.append({
            "severity": "medium",
            "text": f"Possible manipulation ({fake_p * 100:.0f}% fake probability)",
        })
    else:
        findings.append({
            "severity": "info",
            "text": ("Model score is below the detection threshold; this does not "
                     "prove that the image came from a camera"),
        })

    if quality and quality.get("reliability") == "LIMITED":
        findings.append({
            "severity": "medium",
            "text": "Input-quality warning: " + "; ".join(quality["limitations"]),
        })

    agreement = detection.get("model_agreement")
    if agreement is not None and agreement < 0.6:
        findings.append({
            "severity": "medium",
            "text": "Models disagree on this sample - treat the verdict as uncertain",
        })

    if face_count == 0:
        findings.append({
            "severity": "info",
            "text": "No face detected; face-manipulation models were not applied",
        })
    elif face_count > 1:
        findings.append({
            "severity": "info",
            "text": f"{face_count} faces analysed independently",
        })

    if metadata.get("metadata_stripped"):
        findings.append({"severity": "medium", "text": "Metadata missing or removed"})
    elif metadata.get("camera_make"):
        findings.append({
            "severity": "info",
            "text": f"Camera metadata present ({metadata['camera_make']} "
                    f"{metadata.get('camera_model') or ''})".strip(),
        })

    if metadata.get("software"):
        findings.append({
            "severity": "medium",
            "text": f"Processed by editing software: {metadata['software']}",
        })

    if not c2pa.get("signature_found"):
        findings.append({"severity": "medium", "text": "No C2PA signature found"})
    else:
        findings.append({"severity": "info", "text": "C2PA content credential present"})

    if ela.get("suspicious_regions"):
        findings.append({
            "severity": "high",
            "text": "Error Level Analysis shows inconsistent compression regions",
        })

    return findings
