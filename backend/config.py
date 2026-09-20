"""Central configuration. Every tunable number in the system lives here.

Deployment-dependent values read from the environment, with defaults that keep
local behaviour exactly as it was. Nothing here needs setting to run locally.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
AI_MODELS_DIR = MODELS_DIR / "ai_generation"

# Writable state. Containers often mount a volume elsewhere, or have a
# read-only image with only /tmp writable, so the location is overridable.
DATA_DIR = Path(os.getenv("OMNIGUARD_DATA_DIR", str(BACKEND_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
DB_PATH = DATA_DIR / "omniguard.db"

for _d in (DATA_DIR, UPLOAD_DIR, EVIDENCE_DIR, MODELS_DIR, AI_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- deployment ---
HOST = os.getenv("OMNIGUARD_HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", os.getenv("OMNIGUARD_PORT", "8000")))


def _cgroup_memory_limit_mb() -> int | None:
    """Best-effort container RAM limit detection (cgroup v2, then v1)."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw and raw != "max":
                limit = int(raw) // (1024 * 1024)
                # Some unrestricted cgroup-v1 hosts expose a sentinel near 2^63.
                if 0 < limit < 1_000_000:
                    return limit
        except (OSError, ValueError):
            continue
    return None


_LOW_MEMORY_ENV = os.getenv("OMNIGUARD_LOW_MEMORY", "auto").lower()
_CGROUP_MEMORY_MB = _cgroup_memory_limit_mb()
LOW_MEMORY_MODE = (
    _LOW_MEMORY_ENV in {"1", "true", "yes"}
    or (_LOW_MEMORY_ENV == "auto" and _CGROUP_MEMORY_MB is not None
        and _CGROUP_MEMORY_MB <= 768)
)

# Large phone photos can consume hundreds of MB across OpenCV, Pillow and ELA
# buffers. Neural inputs are only 224/384px, so bounding the working copy does
# not discard useful classifier detail and keeps 512MB containers alive.
MAX_ANALYSIS_SIDE = int(os.getenv(
    "OMNIGUARD_MAX_ANALYSIS_SIDE", "1600" if LOW_MEMORY_MODE else "2560"
))
FORENSICS_MAX_SIDE = int(os.getenv(
    "OMNIGUARD_FORENSICS_MAX_SIDE", "1400" if LOW_MEMORY_MODE else "2048"
))

# Origins allowed to call the API. Local dev origins are always permitted; a
# hosted frontend (Vercel, etc.) is added via the environment.
#   OMNIGUARD_ALLOWED_ORIGINS="https://your-app.vercel.app,https://www.example.com"
_LOCAL_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8000", "http://127.0.0.1:8000",
]
ALLOWED_ORIGINS = _LOCAL_ORIGINS + [
    o.strip().rstrip("/")
    for o in os.getenv("OMNIGUARD_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

# When the frontend is on a different site to the API, the session cookie must
# be SameSite=None and Secure or the browser will not send it. That combination
# requires HTTPS, so it is opt-in rather than the local default.
CROSS_SITE_COOKIES = os.getenv("OMNIGUARD_CROSS_SITE", "").lower() in {"1", "true", "yes"}

# Face models are ~38 MB and are not committed. A fresh container fetches them
# on first boot; set to "0" if the image already bundles them.
AUTO_DOWNLOAD_FACE_MODELS = os.getenv("OMNIGUARD_AUTO_DOWNLOAD", "1").lower() not in {"0", "false", "no"}

FACE_MODEL_URLS = {
    "face_detection_yunet.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# --- face detection / recognition (OpenCV Zoo models, shipped in models/) ---
YUNET_PATH = MODELS_DIR / "face_detection_yunet.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface.onnx"

FACE_SCORE_THRESHOLD = 0.75   # below this, a detection is discarded
FACE_NMS_THRESHOLD = 0.3
FACE_MARGIN = 0.20            # expand the crop 20% beyond the box; forgery
                              # artifacts concentrate at the blend boundary
MIN_FACE_PIXELS = 48          # faces smaller than this are too low-res to judge

# --- deepfake classifier ---
CLASSIFIER_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class index convention, must match the training notebook: 0 = REAL, 1 = FAKE
REAL_IDX, FAKE_IDX = 0, 1

# Verdict bands, applied to the ensemble probability of "fake"
SUSPICIOUS_THRESHOLD = 0.40
FAKE_THRESHOLD = 0.65

# --- video ---
VIDEO_MAX_FRAMES = 32         # evenly sampled across the clip
VIDEO_MAX_SECONDS = 300
TEMPORAL_INCONSISTENCY_THRESHOLD = 0.18   # std-dev of per-frame scores

# --- identity matching (SFace cosine similarity) ---
IDENTITY_MATCH_THRESHOLD = 0.363   # OpenCV's recommended cosine threshold
TRACK_MATCH_THRESHOLD = 0.30       # looser, for following a face across frames
TRACK_IOU_THRESHOLD = 0.30

# --- uploads ---
MAX_UPLOAD_MB = 200
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tiff"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def verdict_from_score(fake_prob: float) -> str:
    """Map an ensemble fake-probability onto the dashboard's three verdict bands."""
    if fake_prob >= FAKE_THRESHOLD:
        return "FAKE"
    if fake_prob >= SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "AUTHENTIC"


def risk_from_score(fake_prob: float) -> str:
    if fake_prob >= 0.80:
        return "HIGH"
    if fake_prob >= FAKE_THRESHOLD:
        return "MEDIUM"
    if fake_prob >= SUSPICIOUS_THRESHOLD:
        return "LOW"
    return "MINIMAL"
