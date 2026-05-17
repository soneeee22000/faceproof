"""FaceProof API — FastAPI application.

Exposes the verification pipeline over HTTP. Every response uses one envelope:
success is ``{"data": <result>, "error": null}``; failure is
``{"data": null, "error": {"code", "message"}}``. Endpoints are synchronous —
the CV inference is blocking CPU work, so FastAPI runs them in a threadpool.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from faceproof.config import settings
from faceproof.decision import verify
from faceproof.errors import (
    FaceProofError,
    ImageTooLargeError,
    InvalidImageError,
    LivenessModelMissingError,
    NoFaceDetectedError,
)
from faceproof.images import decode_image
from faceproof.liveness import detect_liveness
from faceproof.matching import match_faces
from faceproof.schemas import ApiResponse, FaceMatchOut, LivenessOut, VerifyOut

app = FastAPI(title="FaceProof", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ERROR_CODES: dict[type[Exception], tuple[str, int]] = {
    InvalidImageError: ("INVALID_IMAGE", 415),
    ImageTooLargeError: ("IMAGE_TOO_LARGE", 413),
    NoFaceDetectedError: ("NO_FACE_DETECTED", 422),
    LivenessModelMissingError: ("LIVENESS_MODEL_MISSING", 503),
}


def _error_response(code: str, message: str, status_code: int) -> JSONResponse:
    """Build the structured error envelope as a JSON response."""
    return JSONResponse(
        status_code=status_code,
        content={"data": None, "error": {"code": code, "message": message}},
    )


@app.exception_handler(FaceProofError)
def _handle_faceproof_error(request: Request, exc: Exception) -> JSONResponse:
    """Map a domain error to the structured error envelope."""
    code, status_code = _ERROR_CODES.get(type(exc), ("INTERNAL_ERROR", 500))
    return _error_response(code, str(exc), status_code)


@app.exception_handler(RequestValidationError)
def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Map a request-validation failure (e.g. a missing upload) to the envelope."""
    return _error_response("INVALID_REQUEST", str(exc), 422)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe."""
    return {"status": "ok", "service": settings.service_name}


@app.post("/api/verify")
def verify_endpoint(id_portrait: UploadFile, selfie: UploadFile) -> ApiResponse[VerifyOut]:
    """Verify a selfie against an ID portrait — face match plus liveness."""
    result = verify(
        decode_image(id_portrait.file.read()),
        decode_image(selfie.file.read()),
        settings.face_match_threshold,
    )
    return ApiResponse[VerifyOut](data=VerifyOut.model_validate(result))


@app.post("/api/match")
def match_endpoint(image_a: UploadFile, image_b: UploadFile) -> ApiResponse[FaceMatchOut]:
    """Compare two faces — match only, no liveness."""
    result = match_faces(
        decode_image(image_a.file.read()),
        decode_image(image_b.file.read()),
        settings.face_match_threshold,
    )
    return ApiResponse[FaceMatchOut](data=FaceMatchOut.model_validate(result))


@app.post("/api/liveness")
def liveness_endpoint(selfie: UploadFile) -> ApiResponse[LivenessOut]:
    """Assess whether a selfie is a live face or a presentation attack."""
    result = detect_liveness(decode_image(selfie.file.read()))
    return ApiResponse[LivenessOut](data=LivenessOut.model_validate(result))


# Serve the built React frontend when present, so the demo ships as one container.
# Registered after the API routes, which therefore take precedence.
_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
