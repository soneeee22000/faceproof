# syntax=docker/dockerfile:1
# Single CPU-only container for GCP Cloud Run: the React UI plus the FastAPI service.

# --- Stage 1: build the React frontend ---
FROM node:24-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime ---
FROM python:3.10-slim AS runtime
WORKDIR /app

COPY pyproject.toml ./
COPY faceproof/ ./faceproof/

# System libraries for OpenCV / ONNX Runtime, plus a compiler for the InsightFace
# build. CPU-only Torch wheels keep the image small; the compiler is purged after.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libgomp1 \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[serve]" \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Bake in the model weights so there are no cold-start downloads on Cloud Run.
COPY models/ ./models/
RUN python models/download_silentface.py \
 && python -c "from faceproof.detection import _detector; from faceproof.embedding import _recognizer; _detector(); _recognizer()"

# The built React UI — FastAPI serves it at /.
COPY --from=frontend /build/dist ./frontend/dist

EXPOSE 8080
CMD ["sh", "-c", "uvicorn faceproof.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
