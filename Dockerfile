# OmniGuard AI — the whole application in one container.
#
# FastAPI serves both the API and the built dashboard, so a single container is
# the complete app: landing page, dashboard, scanner, face recognition,
# accounts, assistant and documentation on one origin. No CORS, no separate
# frontend host, no split deployment.
#
# This is the way to run it for free on Hugging Face Spaces (2 vCPU, 16 GB RAM,
# 16 GB disk, no card required). It works unchanged on Render, Railway, Fly.io,
# Cloud Run or a local Docker daemon.
#
#   docker build -t omniguard .
#   docker run -p 8000:8000 omniguard
#
# Why not Vercel or Netlify: this image needs ~261 MB of Python dependencies
# (OpenCV alone is ~117 MB) against a 250 MB unzipped limit on Vercel and
# 50 MB zipped on Netlify.

# ---------------------------------------------------------------- stage 1 ----
# Build the dashboard. frontend/dist is gitignored, so it must be produced here
# rather than copied — a fresh clone has no build output.
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------- stage 2 ----
FROM python:3.12-slim

# libGL and libglib are what the headless OpenCV wheel still needs; curl is used
# by the healthcheck and by the first-boot model fetch if a model is missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000. Creating that user here means
# the same image behaves identically everywhere, and nothing runs as root.
RUN useradd -m -u 1000 omniguard

WORKDIR /app

# Dependencies first so a code change does not invalidate the layer.
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Backend, including the committed YuNet and SFace models.
COPY --chown=omniguard:omniguard backend/ ./backend/

# The dashboard built in stage 1.
COPY --from=frontend --chown=omniguard:omniguard /build/dist ./frontend/dist

# Writable state lives under the user's own home, which is writable on every
# host. A mounted volume can override it with OMNIGUARD_DATA_DIR.
RUN mkdir -p /home/omniguard/data && chown -R omniguard:omniguard /home/omniguard

USER omniguard

ENV OMNIGUARD_HOST=0.0.0.0 \
    OMNIGUARD_DATA_DIR=/home/omniguard/data \
    OMNIGUARD_LOW_MEMORY=1 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# start-period is generous: the first boot loads several ONNX models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

CMD ["python", "backend/main.py"]
