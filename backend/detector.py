"""Deepfake classification: ONNX model ensemble plus explainability heatmaps.

Each model exported by the training notebook has two outputs:
    logits    (1, 2)          -> [real, fake] scores
    features  (1, C, H, W)    -> final convolutional feature map

Having the feature map lets us compute a Class Activation Map (CAM) with nothing
but numpy. CAM is the weighted sum of the feature channels, weighted by the
classifier's own "fake" weights:

    cam(y, x) = sum_c  W[FAKE, c] * features[c, y, x]

Because the network ends in global-average-pool -> linear, this is exactly the
spatial decomposition of the fake logit. Grad-CAM would need backpropagation,
which ONNX Runtime does not do; CAM needs only the forward pass we already ran.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

import config as cfg

# The face models live in the same folder but are not deepfake classifiers.
_EXCLUDED = {"face_detection_yunet.onnx", "face_recognition_sface.onnx"}

# Human-readable names for the dashboard's model comparison table
DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "legacy_xception": "XceptionNet",
    "mobilenetv3_large_100": "MobileNetV3",
    "community_forensics_vit_s": "Community Forensics ViT-S",
    "community_forensics_v13_fp32": "Community Forensics ViT-S v13",
}


def _preprocess(image_bgr: np.ndarray, manifest: dict | None = None) -> np.ndarray:
    """BGR uint8 image -> the tensor contract declared by the model manifest."""
    manifest = manifest or {}
    size = int(manifest.get("img_size") or cfg.CLASSIFIER_INPUT_SIZE)
    normalization = manifest.get("normalization", {})
    mean = np.asarray(normalization.get("mean", cfg.IMAGENET_MEAN), dtype=np.float32)
    std = np.asarray(normalization.get("std", cfg.IMAGENET_STD), dtype=np.float32)

    resize_size = manifest.get("resize_size")
    if resize_size:
        # Community Forensics uses PIL bicubic: shortest edge -> resize_size,
        # then an exact center crop.  This is part of the trained contract.
        rgb_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        width, height = rgb_image.size
        scale = float(resize_size) / min(width, height)
        resized = rgb_image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
        )
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        rgb = np.asarray(resized.crop((left, top, left + size, top + size)),
                         dtype=np.float32) / 255.0
    else:
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    chw = np.transpose(rgb, (2, 0, 1))
    return ((chw - mean[:, None, None]) / std[:, None, None])[None, ...].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class DeepfakeDetector:
    """Loads every trained ONNX classifier it finds and runs them as an ensemble."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models: list[dict] = []
        self.manifest: dict = {}
        self.models_dir = Path(models_dir or cfg.MODELS_DIR)
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        manifest_path = self.models_dir / "manifest.json"
        if manifest_path.exists():
            try:
                self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.manifest = {}

        # The repository's tiny dummy network exists only to exercise the
        # pipeline in tests. It produces plausible-shaped numbers but has
        # never learned to distinguish real from manipulated media. Never
        # expose those numbers as scan verdicts in a normal app run.
        allow_dummy = os.getenv("OMNIGUARD_ALLOW_DUMMY_MODELS", "").lower() in {
            "1", "true", "yes",
        }
        dummy_arches = {
            item.get("arch")
            for item in self.manifest.get("models", [])
            if item.get("dummy") is True
        }
        if (not allow_dummy and self.manifest.get("dummy") is True):
            print("  ! ignoring untrained dummy classifier manifest")
            return

        metrics_by_arch = {
            m["arch"]: m.get("metrics", {})
            for m in self.manifest.get("models", [])
        }

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 2 physical cores on the target machine; more threads only adds contention
        opts.intra_op_num_threads = 2

        for path in sorted(self.models_dir.glob("*.onnx")):
            if path.name in _EXCLUDED:
                continue
            if not allow_dummy and (
                path.stem in dummy_arches or path.stem.startswith("dummy_")
            ):
                print(f"  ! ignoring untrained classifier {path.name}")
                continue
            try:
                sess = ort.InferenceSession(
                    str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                )
            except Exception as exc:                      # noqa: BLE001
                print(f"  ! could not load {path.name}: {exc}")
                continue

            arch = path.stem
            weight_path = self.models_dir / f"{arch}_classifier_w.npy"
            weights = None
            if weight_path.exists():
                try:
                    weights = np.load(weight_path)
                except Exception:                          # noqa: BLE001
                    weights = None

            self.models.append({
                "arch": arch,
                "name": DISPLAY_NAMES.get(arch, arch.replace("_", " ").title()),
                "session": sess,
                "input_name": sess.get_inputs()[0].name,
                "output_names": [o.name for o in sess.get_outputs()],
                "classifier_weights": weights,
                "metrics": metrics_by_arch.get(arch, {}),
            })
            print(f"  + loaded {arch}")

    @property
    def ready(self) -> bool:
        return len(self.models) > 0

    @property
    def task(self) -> str:
        """Return the visual task these weights were trained to solve.

        Old manifests predate this field. Those models used the Sowaiba
        face-swap corpus, so the safe backwards-compatible default is face
        manipulation -- never generic AI-image detection.
        """
        return str(self.manifest.get("task") or "face_manipulation")

    def supports(self, task: str) -> bool:
        return self.task in {task, "universal_media_authenticity"}

    def info(self) -> dict:
        return {
            "ready": self.ready,
            "task": self.task,
            "model_count": len(self.models),
            "models": [
                {
                    "arch": m["arch"],
                    "name": m["name"],
                    "metrics": m["metrics"],
                }
                for m in self.models
            ],
            "trained_on": self.manifest.get("source_dataset"),
            "test_set_size": self.manifest.get("n_test_images"),
            "ensemble_metrics": self.manifest.get("metrics", {}).get("ENSEMBLE", {}),
        }

    # ---------------------------------------------------------------- inference
    def predict(self, face_bgr: np.ndarray, want_heatmap: bool = True) -> dict:
        """Classify one face crop.

        Returns per-model probabilities, the ensemble probability, and optionally
        a CAM heatmap resized to the input crop.
        """
        if not self.ready:
            raise RuntimeError(
                "No trained deepfake models are loaded. "
                "Run the Colab notebook and place the .onnx files in backend/models/."
            )

        batch = _preprocess(face_bgr, self.manifest)
        per_model, probs = [], []
        best_cam, best_conf = None, -1.0

        for m in self.models:
            outputs = m["session"].run(None, {m["input_name"]: batch})
            logits = np.asarray(outputs[0], dtype=np.float32)
            if self.manifest.get("output_type") == "sigmoid_logit":
                logit = float(logits.reshape(-1)[0])
                prob_fake = float(1.0 / (1.0 + np.exp(-np.clip(logit, -80, 80))))
            else:
                prob_fake = float(_softmax(logits)[0, cfg.FAKE_IDX])
            probs.append(prob_fake)

            confidence_kind = self.manifest.get(
                "confidence_kind", "uncalibrated_model_certainty"
            )
            display_score = (
                prob_fake if confidence_kind == "threshold_calibrated_model_score"
                else abs(prob_fake - 0.5) * 2
            )
            per_model.append({
                "arch": m["arch"],
                "name": m["name"],
                "fake_probability": round(prob_fake, 4),
                "verdict": cfg.verdict_from_score(prob_fake),
                "confidence": round(display_score, 4),
                "test_accuracy": m["metrics"].get("accuracy"),
                "test_auc": m["metrics"].get("roc_auc"),
            })

            # Build the heatmap from whichever model is most certain
            if want_heatmap and len(outputs) > 1:
                certainty = abs(prob_fake - 0.5)
                if certainty > best_conf:
                    cam = self._cam(outputs[1], m["classifier_weights"])
                    if cam is not None:
                        best_cam, best_conf = cam, certainty

        ensemble = float(np.mean(probs))
        # Agreement: 1.0 when every model says the same thing.
        spread = float(np.std(probs)) if len(probs) > 1 else 0.0

        confidence_kind = self.manifest.get(
            "confidence_kind", "uncalibrated_model_certainty"
        )
        display_score = (
            ensemble if confidence_kind == "threshold_calibrated_model_score"
            else abs(ensemble - 0.5) * 2
        )
        result = {
            "fake_probability": round(ensemble, 4),
            "authenticity_score": round((1.0 - ensemble) * 100, 1),
            "verdict": cfg.verdict_from_score(ensemble),
            "risk_level": cfg.risk_from_score(ensemble),
            "confidence": round(display_score, 4),
            "confidence_kind": confidence_kind,
            "decision_threshold": self.manifest.get(
                "decision_threshold", cfg.FAKE_THRESHOLD
            ),
            "model_agreement": round(max(0.0, 1.0 - spread * 2), 4),
            "models": per_model,
        }
        if best_cam is not None:
            upscaled = cv2.resize(
                best_cam, (face_bgr.shape[1], face_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            # Cubic interpolation overshoots at sharp edges, which would push
            # values outside the 0..1 range the heatmap contract promises.
            result["heatmap"] = np.clip(upscaled, 0.0, 1.0)
        return result

    # ------------------------------------------------------------------ heatmap
    @staticmethod
    def _cam(features: np.ndarray, weights: np.ndarray | None) -> np.ndarray | None:
        """Class activation map, normalized to 0..1. Pure numpy, forward-only."""
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim != 4:
            return None
        feats = feats[0]                                   # (C, H, W)
        channels = feats.shape[0]

        if weights is not None and weights.ndim == 2 and weights.shape[1] == channels:
            # Exact CAM: weight each channel by its contribution to the fake logit
            w = weights[cfg.FAKE_IDX].astype(np.float32)
            cam = np.tensordot(w, feats, axes=([0], [0]))
        else:
            # Architectures whose head reshapes the channels (e.g. MobileNetV3's
            # post-pool conv) break the exact form; mean activation still shows
            # which regions the network responded to.
            cam = feats.mean(axis=0)

        cam = np.maximum(cam, 0)                           # ReLU
        peak = float(cam.max())
        return cam / peak if peak > 1e-8 else np.zeros_like(cam)


def colorize_heatmap(heatmap: np.ndarray, face_bgr: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Overlay a 0..1 heatmap on the face crop, as shown in the dashboard."""
    hm = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    if colored.shape[:2] != face_bgr.shape[:2]:
        colored = cv2.resize(colored, (face_bgr.shape[1], face_bgr.shape[0]))
    return cv2.addWeighted(colored, alpha, face_bgr, 1 - alpha, 0)
