# FaceProof — single CPU-only container for GCP Cloud Run.
FROM python:3.10-slim

WORKDIR /app

# System libs for OpenCV (headless) and image handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY faceproof ./faceproof
RUN pip install --no-cache-dir ".[ml]"

# Model weights are baked in at build time (see models/README.md).
COPY models ./models

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn faceproof.api:app --host 0.0.0.0 --port ${PORT}"]
