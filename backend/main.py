"""OmniGuard AI - FastAPI application.

Run directly:      python backend/main.py
Or via uvicorn:    uvicorn main:app --app-dir backend --reload
"""

from __future__ import annotations

import mimetypes
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assistant              # noqa: E402
import bootstrap              # noqa: E402
import auth                   # noqa: E402
import config as cfg          # noqa: E402
import database as db         # noqa: E402
import pipeline               # noqa: E402
import textcheck              # noqa: E402
import video as video_mod     # noqa: E402
from detector import DeepfakeDetector   # noqa: E402
from faces import FaceAnalyzer          # noqa: E402

STATE: dict = {"detector": None, "ai_detector": None, "faces": None, "load_error": None}

# Avoid large OpenCV thread pools and their stacks in constrained containers.
if cfg.LOW_MEMORY_MODE:
    cv2.setNumThreads(1)

# Python's mimetype database predates woff2, so self-hosted fonts would be
# served as application/octet-stream without this.
mimetypes.add_type("font/woff2", ".woff2")
# Serve the plain-text documentation inline in the browser rather than as a
# download, and label markdown correctly.
mimetypes.add_type("text/plain; charset=utf-8", ".txt")
mimetypes.add_type("text/plain; charset=utf-8", ".md")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("\n" + "=" * 62)
    print("  OmniGuard AI - starting")
    print("=" * 62)

    quarantined = db.init()
    if quarantined:
        print(f"  ! previous database was unreadable and was moved aside as")
        print(f"    {quarantined} — starting with a fresh one")
    auth.init()
    expired = auth.purge_expired()
    print(f"  database ready ({auth.user_count()} account(s)"
          + (f", {expired} expired session(s) purged" if expired else "") + ")")

    fetched = bootstrap.ensure_face_models()
    if fetched:
        print(f"  fetched {len(fetched)} face model(s) on first boot")

    try:
        STATE["faces"] = FaceAnalyzer()
        print("  face analyzer ready (YuNet + SFace)")
    except Exception as exc:                               # noqa: BLE001
        STATE["load_error"] = str(exc)
        print(f"  ! face analyzer failed: {exc}")

    print("  loading deepfake classifiers:")
    detector = DeepfakeDetector()
    STATE["detector"] = detector
    if detector.ready:
        print(f"  {len(detector.models)} classifier(s) ready")
    else:
        print("  ! NO TRAINED MODELS FOUND in backend/models/")
        print("    Run notebooks/OmniGuard_Training.ipynb on Colab,")
        print("    then unzip omniguard_models.zip into backend/models/.")
        print("    The server still runs; scan endpoints will return 503.")

    print("  loading AI-generation classifiers:")
    ai_detector = DeepfakeDetector(cfg.AI_MODELS_DIR)
    STATE["ai_detector"] = ai_detector
    if ai_detector.ready and ai_detector.supports("ai_generation"):
        print(f"  {len(ai_detector.models)} AI-image classifier(s) ready")
    elif ai_detector.ready:
        print("  ! ignored AI model bank: manifest task is not ai_generation")
        ai_detector.models.clear()
    else:
        print("  ! no AI-generation model bank yet; no-face images stay UNVERIFIED")

    print(f"  {bootstrap.describe_environment()}")
    print(f"  memory_profile={'low' if cfg.LOW_MEMORY_MODE else 'standard'} "
          f"analysis_max_side={cfg.MAX_ANALYSIS_SIDE}px")
    print("=" * 62)
    print(f"  listening on {cfg.HOST}:{cfg.PORT}     (API docs at /docs)")
    print("=" * 62 + "\n")
    yield


app = FastAPI(
    title="OmniGuard AI",
    description="Deepfake and AI-generated media detection",
    version="1.0.0",
    lifespan=lifespan,
)

# allow_origins must be an explicit list, never "*": a wildcard is invalid
# alongside allow_credentials, and the browser silently drops the response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- helpers
def _require_detector() -> DeepfakeDetector:
    det = STATE["detector"]
    if det is None or not det.ready:
        raise HTTPException(
            status_code=503,
            detail=("No trained models loaded. Run the Colab training notebook, "
                    "then place the .onnx files in backend/models/."),
        )
    return det


def _require_faces() -> FaceAnalyzer:
    fa = STATE["faces"]
    if fa is None:
        raise HTTPException(status_code=503,
                            detail=f"Face analyzer unavailable: {STATE['load_error']}")
    return fa


def _save_upload(upload: UploadFile, allowed: set[str]) -> Path:
    suffix = Path(upload.filename or "upload").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    dest = cfg.UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh, length=1024 * 1024)

    size_mb = dest.stat().st_size / 1e6
    if size_mb > cfg.MAX_UPLOAD_MB:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.0f} MB; limit is {cfg.MAX_UPLOAD_MB} MB.",
        )
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # keep the original name for display, without trusting it as a path
    dest_named = dest.with_name(dest.stem + "__" + Path(upload.filename or "file").name)
    try:
        dest.rename(dest_named)
        return dest_named
    except OSError:
        return dest


def _strip_private(report: dict) -> dict:
    for face in report.get("faces", []):
        face.pop("_embedding", None)
    return report


# ---------------------------------------------------------------------- system
@app.get("/api/health")
def health():
    det = STATE["detector"]
    ai_det = STATE["ai_detector"]
    return {
        "status": "ok",
        "models_loaded": bool(det and det.ready),
        "model_count": len(det.models) if det else 0,
        "ai_models_loaded": bool(ai_det and ai_det.ready),
        "ai_model_count": len(ai_det.models) if ai_det else 0,
        "face_analyzer": STATE["faces"] is not None,
        "low_memory_mode": cfg.LOW_MEMORY_MODE,
    }


@app.get("/api/system/info")
def system_info():
    det = STATE["detector"]
    ai_det = STATE["ai_detector"]
    return {
        "detector": det.info() if det else {"ready": False},
        "ai_detector": ai_det.info() if ai_det else {"ready": False},
        "face_analyzer_ready": STATE["faces"] is not None,
        "low_memory_mode": cfg.LOW_MEMORY_MODE,
        "thresholds": {
            "suspicious": cfg.SUSPICIOUS_THRESHOLD,
            "fake": cfg.FAKE_THRESHOLD,
        },
        "limits": {
            "max_upload_mb": cfg.MAX_UPLOAD_MB,
            "video_max_frames": cfg.VIDEO_MAX_FRAMES,
        },
        "supported": {
            "image": sorted(cfg.ALLOWED_IMAGE_EXT),
            "video": sorted(cfg.ALLOWED_VIDEO_EXT),
        },
    }


@app.get("/api/models")
def models():
    """Powers the Model Comparison panel. Numbers come from the held-out test
    set measured during training, not from anything invented at runtime."""
    det = STATE["detector"]
    if det is None or not det.ready:
        return {"ready": False, "models": []}
    info = det.info()
    return {
        "ready": True,
        "trained_on": info["trained_on"],
        "test_set_size": info["test_set_size"],
        "ensemble_metrics": info["ensemble_metrics"],
        "models": info["models"],
    }


# ----------------------------------------------------------------------- scans
#
# The analysis helpers below are plain functions, and every route calls one of
# them. Route handlers must never call each other directly: FastAPI resolves
# defaults like `Form(None)` only through dependency injection, so a direct call
# passes the raw `Form` object instead of the value and the handler fails deep
# inside the pipeline.

def _run_image_scan(file: UploadFile) -> dict:
    detector, faces = _require_detector(), _require_faces()
    path = _save_upload(file, cfg.ALLOWED_IMAGE_EXT)
    try:
        report = pipeline.analyze_image(path, detector, faces, ai_detector=STATE["ai_detector"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # Previews are embedded in the report, so the upload has served its
        # purpose. Keeping it would grow without bound across a long session.
        path.unlink(missing_ok=True)

    report["filename"] = file.filename or path.name
    db.save_scan(report)
    return _strip_private(report)


def _run_video_scan(file: UploadFile, max_frames: int | None) -> dict:
    detector, faces = _require_detector(), _require_faces()
    path = _save_upload(file, cfg.ALLOWED_VIDEO_EXT)
    try:
        report = video_mod.analyze_video(path, detector, faces, max_frames=max_frames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

    report["filename"] = file.filename or path.name
    db.save_scan(report)
    return report


@app.post("/api/scan/image")
async def scan_image(file: UploadFile = File(...)):
    return _run_image_scan(file)


@app.post("/api/scan/video")
async def scan_video(file: UploadFile = File(...), max_frames: int | None = Form(None)):
    return _run_video_scan(file, max_frames)


@app.post("/api/scan")
async def scan_auto(file: UploadFile = File(...)):
    """Single entry point for the drag-and-drop box; routes on file extension."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in cfg.ALLOWED_IMAGE_EXT:
        return _run_image_scan(file)
    if suffix in cfg.ALLOWED_VIDEO_EXT:
        return _run_video_scan(file, None)
    raise HTTPException(
        status_code=400,
        detail=(f"Unsupported file type '{suffix}'. "
                f"Images: {sorted(cfg.ALLOWED_IMAGE_EXT)}  "
                f"Videos: {sorted(cfg.ALLOWED_VIDEO_EXT)}"),
    )


@app.get("/api/scans")
def list_scans(limit: int = 20, offset: int = 0, verdict: str | None = None):
    return {"scans": db.recent_scans(limit=limit, offset=offset, verdict=verdict)}


@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: str):
    report = db.get_scan(scan_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No scan with id {scan_id}")
    return report


@app.delete("/api/scan/{scan_id}")
def remove_scan(scan_id: str):
    if not db.delete_scan(scan_id):
        raise HTTPException(status_code=404, detail=f"No scan with id {scan_id}")
    return {"deleted": scan_id}


@app.get("/api/stats")
def stats():
    return db.stats()


# ------------------------------------------------------------------ identities
@app.post("/api/identity/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...),
                 notes: str | None = Form(None)):
    faces = _require_faces()
    path = _save_upload(file, cfg.ALLOWED_IMAGE_EXT)
    try:
        image = cv2.imread(str(path))
    finally:
        # Only the embedding is kept; the photograph itself is not retained.
        path.unlink(missing_ok=True)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    found = faces.detect(image)
    if not found:
        raise HTTPException(status_code=422, detail="No face detected in that image.")

    emb = faces.embed(image, found[0]["_raw"])
    if emb is None:
        raise HTTPException(status_code=500, detail="Face recognition model unavailable.")

    return db.enroll_identity(name.strip(), emb, notes)


@app.post("/api/identity/match")
async def match(file: UploadFile = File(...)):
    """Identify every face in an image against the enrolled gallery, and report
    each one's deepfake verdict alongside - i.e. 'this claims to be X, and the
    face is manipulated'."""
    faces = _require_faces()
    gallery = db.load_gallery()
    path = _save_upload(file, cfg.ALLOWED_IMAGE_EXT)
    try:
        image = cv2.imread(str(path))
    finally:
        path.unlink(missing_ok=True)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    detector = STATE["detector"]
    results = []
    for i, face in enumerate(faces.detect(image)):
        emb = faces.embed(image, face["_raw"])
        name, sim = (None, 0.0)
        if emb is not None and gallery:
            name, sim = faces.match(emb, gallery)

        entry = {
            "face_id": i + 1,
            "bbox": face["bbox"],
            "identity": name,
            "similarity": round(float(sim), 4),
            "matched": name is not None,
        }
        if detector and detector.ready:
            verdict = detector.predict(faces.crop(image, face["bbox"]), want_heatmap=False)
            entry["fake_probability"] = verdict["fake_probability"]
            entry["verdict"] = verdict["verdict"]
        results.append(entry)

    return {
        "faces_detected": len(results),
        "gallery_size": len(gallery),
        "matches": results,
    }


# ------------------------------------------------------------------------ auth
#
# The session token is returned in the body and also set as an HttpOnly cookie.
# The cookie is what the browser actually uses, so the token is never readable
# from JavaScript and therefore not exposed to an XSS bug.

_COOKIE = "omniguard_session"


def _set_session_cookie(response: Response, token: str) -> None:
    # Same-site by default (local use). When the frontend is hosted on a
    # different domain to the API, the browser will only send the cookie if it
    # is SameSite=None AND Secure - and Secure requires HTTPS, which is why
    # this is opt-in rather than always on.
    response.set_cookie(
        _COOKIE, token,
        max_age=auth.SESSION_DAYS * 86400,
        httponly=True,
        samesite="none" if cfg.CROSS_SITE_COOKIES else "lax",
        secure=cfg.CROSS_SITE_COOKIES,
    )


@app.post("/api/auth/signup")
async def auth_signup(response: Response,
                      email: str = Form(...),
                      name: str = Form(...),
                      password: str = Form(...)):
    try:
        result = auth.sign_up(email, name, password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    _set_session_cookie(response, result["token"])
    return {"user": result["user"], "expires_at": result["expires_at"]}


@app.post("/api/auth/login")
async def auth_login(response: Response,
                     email: str = Form(...),
                     password: str = Form(...)):
    try:
        result = auth.sign_in(email, password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    _set_session_cookie(response, result["token"])
    return {"user": result["user"], "expires_at": result["expires_at"]}


@app.post("/api/auth/logout")
async def auth_logout(request: Request, response: Response):
    auth.sign_out(request.cookies.get(_COOKIE))
    response.delete_cookie(_COOKIE)
    return {"signed_out": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    user = auth.current_user(request.cookies.get(_COOKIE))
    return {"authenticated": user is not None, "user": user}


# ------------------------------------------------------------------------ text
@app.post("/api/text/analyze")
async def text_analyze(text: str = Form(...),
                       reference: str = Form(""),
                       check_plagiarism: bool = Form(True),
                       check_ai: bool = Form(True),
                       check_news: bool = Form(False)):
    """Plagiarism overlap, AI-generation indicators and news credibility signals.

    The three results are not equivalent in strength, and the response keeps
    them separate for that reason. Plagiarism overlap is an exact measurement
    against the supplied reference. The AI figure is a summary of stylistic
    statistics that no detector can turn into proof. The news figure is
    narrower still: it describes how a passage is written and cannot check a
    single claim against the world.
    """
    if len(text) > 200_000:
        raise HTTPException(status_code=413,
                            detail="Text is too long; limit is 200,000 characters.")
    try:
        return textcheck.analyze(text, reference,
                                 want_plagiarism=check_plagiarism,
                                 want_ai=check_ai,
                                 want_news=check_news)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ------------------------------------------------------------------- assistant
@app.post("/api/assistant")
async def assistant_ask(question: str = Form(...)):
    """In-app help chat. Answers from a curated knowledge base and can read
    live scan data, so figures it quotes are real rather than generated."""
    detector = STATE["detector"]
    return assistant.ask(question, detector.info() if detector else None)


@app.get("/api/assistant/suggestions")
def assistant_suggestions():
    return {"suggestions": assistant.DEFAULT_FOLLOWUPS}


@app.get("/api/identities")
def identities():
    return {"identities": db.list_identities()}


@app.delete("/api/identity/{name}")
def remove_identity(name: str):
    if not db.delete_identity(name):
        raise HTTPException(status_code=404, detail=f"No identity named {name}")
    return {"deleted": name}


# ------------------------------------------------------- frontend static files
_DIST = cfg.PROJECT_DIR / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Unknown API route")
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
else:
    @app.get("/")
    def root():
        return JSONResponse({
            "service": "OmniGuard AI",
            "note": "Frontend not built yet. Run: cd frontend && npm install && npm run build",
            "api_docs": "/docs",
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, log_level="info")
