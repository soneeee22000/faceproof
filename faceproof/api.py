"""FaceProof API — FastAPI application.

Verification endpoints (``/api/verify``, ``/api/match``, ``/api/liveness``) are added
test-first during the PRD build phases. Phase 0 ships only the health probe.
"""

from fastapi import FastAPI

from faceproof.config import settings

app = FastAPI(title="FaceProof", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe."""
    return {"status": "ok", "service": settings.service_name}
