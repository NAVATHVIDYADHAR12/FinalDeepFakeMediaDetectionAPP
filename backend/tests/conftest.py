"""Shared pytest fixtures.

Every test runs against a temporary database so the real scan history is never
touched, and against the placeholder ONNX model so the suite passes with or
without the Colab-trained weights present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import config as cfg          # noqa: E402
import database as db         # noqa: E402
from detector import DeepfakeDetector   # noqa: E402
from faces import FaceAnalyzer          # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _ensure_placeholder_model():
    """Allow the untrained fixture only inside tests of the inference plumbing."""
    import os
    previous = os.environ.get("OMNIGUARD_ALLOW_DUMMY_MODELS")
    os.environ["OMNIGUARD_ALLOW_DUMMY_MODELS"] = "1"
    has_classifier = any(
        p.name not in {"face_detection_yunet.onnx", "face_recognition_sface.onnx"}
        for p in cfg.MODELS_DIR.glob("*.onnx")
    )
    if not has_classifier:
        sys.path.insert(0, str(BACKEND / "tools"))
        import make_dummy_model
        make_dummy_model.install()
    yield
    if previous is None:
        os.environ.pop("OMNIGUARD_ALLOW_DUMMY_MODELS", None)
    else:
        os.environ["OMNIGUARD_ALLOW_DUMMY_MODELS"] = previous


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the database at a throwaway file for the duration of each test."""
    monkeypatch.setattr(cfg, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db.cfg, "DB_PATH", tmp_path / "test.db")
    db.init()
    yield
    # The connection is shared and long-lived, so it must be released before
    # pytest removes the temporary directory it points at.
    db.close()


@pytest.fixture(scope="session")
def face_analyzer():
    return FaceAnalyzer()


@pytest.fixture(scope="session")
def detector():
    return DeepfakeDetector()


@pytest.fixture(scope="session")
def face_image_path() -> Path:
    """A real photograph containing exactly one detectable face."""
    path = FIXTURES / "face.jpg"
    if not path.exists():
        pytest.skip("fixtures/face.jpg missing")
    return path


@pytest.fixture(scope="session")
def face_image(face_image_path) -> np.ndarray:
    return cv2.imread(str(face_image_path))


@pytest.fixture
def noise_image_path(tmp_path) -> Path:
    """Random noise: valid image, no face. Exercises the full-frame fallback."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "noise.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="session")
def video_path(tmp_path_factory, face_image) -> Path:
    """Short clip built from the face fixture, with the face drifting so the
    tracker and the per-frame sampler both have something to do."""
    out = tmp_path_factory.mktemp("video") / "clip.mp4"
    h, w = face_image.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available in this OpenCV build")

    for i in range(30):
        shifted = np.roll(face_image, shift=i % 12, axis=1)
        writer.write(shifted)
    writer.release()

    if not out.exists() or out.stat().st_size == 0:
        pytest.skip("video encoding produced no output")
    return out
