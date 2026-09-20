"""Face detection, identity embedding, and cross-frame tracking.

Uses the two models bundled with OpenCV Zoo:
  * YuNet  - fast CPU face detector, returns a box plus 5 landmarks
  * SFace  - 128-d identity embedding, used both for "who is this" and for
             following the same person across video frames

Neither needs PyTorch, which is why the app runs on a CPU-only laptop.
"""

from __future__ import annotations

import gc
import numpy as np
import cv2

import config as cfg


class FaceAnalyzer:
    def __init__(self) -> None:
        if not cfg.YUNET_PATH.exists():
            raise FileNotFoundError(
                f"Face detector missing: {cfg.YUNET_PATH}\n"
                "Run setup again to download it."
            )
        self._detector = cv2.FaceDetectorYN.create(
            str(cfg.YUNET_PATH), "", (320, 320),
            cfg.FACE_SCORE_THRESHOLD, cfg.FACE_NMS_THRESHOLD, 5000,
        )
        self._recognizer = None
        if cfg.SFACE_PATH.exists() and not cfg.LOW_MEMORY_MODE:
            try:
                self._recognizer = cv2.FaceRecognizerSF.create(str(cfg.SFACE_PATH), "")
            except cv2.error:
                self._recognizer = None

    # ------------------------------------------------------------------ detect
    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """Return every face found, largest first."""
        h, w = image_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, raw = self._detector.detect(image_bgr)
        if raw is None:
            return []

        faces = []
        for row in raw:
            x, y, bw, bh = row[:4].astype(float)
            if bw < cfg.MIN_FACE_PIXELS or bh < cfg.MIN_FACE_PIXELS:
                continue
            faces.append({
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "score": float(row[-1]),
                # right eye, left eye, nose, right mouth, left mouth
                "landmarks": row[4:14].reshape(5, 2).astype(float).tolist(),
                "_raw": row,
            })

        faces.sort(key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
        return faces

    # -------------------------------------------------------------------- crop
    @staticmethod
    def crop(image_bgr: np.ndarray, bbox, margin: float | None = None) -> np.ndarray:
        """Crop a face with margin. The blend seam of a face-swap sits just
        outside the detector's box, so the margin matters for accuracy."""
        margin = cfg.FACE_MARGIN if margin is None else margin
        h, w = image_bgr.shape[:2]
        x, y, bw, bh = bbox
        mx, my = bw * margin, bh * margin
        x0 = max(0, int(x - mx))
        y0 = max(0, int(y - my))
        x1 = min(w, int(x + bw + mx))
        y1 = min(h, int(y + bh + my))
        if x1 <= x0 or y1 <= y0:
            return image_bgr
        return image_bgr[y0:y1, x0:x1]

    # --------------------------------------------------------------- embedding
    def embed(self, image_bgr: np.ndarray, raw_row: np.ndarray) -> np.ndarray | None:
        """128-d identity vector for one detected face, or None if unavailable."""
        recognizer = self._recognizer
        temporary = False
        if recognizer is None and cfg.LOW_MEMORY_MODE and cfg.SFACE_PATH.exists():
            try:
                recognizer = cv2.FaceRecognizerSF.create(str(cfg.SFACE_PATH), "")
                temporary = True
            except cv2.error:
                recognizer = None
        if recognizer is None:
            return None
        try:
            aligned = recognizer.alignCrop(image_bgr, raw_row)
            feat = recognizer.feature(aligned)
            return np.asarray(feat, dtype=np.float32).flatten()
        except cv2.error:
            return None
        finally:
            if temporary:
                del recognizer
                gc.collect()

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    def match(self, probe: np.ndarray, gallery: dict[str, np.ndarray],
              threshold: float | None = None) -> tuple[str | None, float]:
        """Best identity match for a probe embedding. Returns (name, similarity)."""
        threshold = cfg.IDENTITY_MATCH_THRESHOLD if threshold is None else threshold
        best_name, best_sim = None, -1.0
        for name, vec in gallery.items():
            sim = self.cosine(probe, vec)
            if sim > best_sim:
                best_name, best_sim = name, sim
        if best_sim < threshold:
            return None, max(best_sim, 0.0)
        return best_name, best_sim


# ---------------------------------------------------------------------- tracking
def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


class FaceTracker:
    """Follows people across video frames so each person gets their own
    authenticity timeline rather than one blended score for the whole clip.

    Matches on identity embedding first (survives movement and occlusion) and
    falls back to box overlap when no embedding is available.
    """

    def __init__(self) -> None:
        self.tracks: list[dict] = []

    def update(self, frame_idx: int, detections: list[dict]) -> None:
        for det in detections:
            best, best_score = None, 0.0

            for track in self.tracks:
                emb, temb = det.get("embedding"), track.get("embedding")
                if emb is not None and temb is not None:
                    score = FaceAnalyzer.cosine(np.asarray(emb), np.asarray(temb))
                    thresh = cfg.TRACK_MATCH_THRESHOLD
                else:
                    score = _iou(det["bbox"], track["last_bbox"])
                    thresh = cfg.TRACK_IOU_THRESHOLD
                if score > best_score and score >= thresh:
                    best, best_score = track, score

            if best is None:
                best = {
                    "track_id": len(self.tracks) + 1,
                    "embedding": det.get("embedding"),
                    "last_bbox": det["bbox"],
                    "frames": [],
                    "scores": [],
                }
                self.tracks.append(best)

            best["last_bbox"] = det["bbox"]
            if best.get("embedding") is None and det.get("embedding") is not None:
                best["embedding"] = det["embedding"]
            best["frames"].append(frame_idx)
            best["scores"].append(float(det["fake_probability"]))

    def summary(self) -> list[dict]:
        out = []
        for t in sorted(self.tracks, key=lambda t: len(t["frames"]), reverse=True):
            scores = np.asarray(t["scores"], dtype=np.float32)
            mean = float(scores.mean())
            std = float(scores.std()) if scores.size > 1 else 0.0
            out.append({
                "track_id": t["track_id"],
                "frames_seen": len(t["frames"]),
                "first_frame": int(min(t["frames"])),
                "last_frame": int(max(t["frames"])),
                "mean_fake_probability": round(mean, 4),
                "max_fake_probability": round(float(scores.max()), 4),
                "score_std": round(std, 4),
                "temporally_inconsistent": bool(std > cfg.TEMPORAL_INCONSISTENCY_THRESHOLD),
                "verdict": cfg.verdict_from_score(mean),
            })
        return out
