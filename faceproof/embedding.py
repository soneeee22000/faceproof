"""ArcFace face embeddings using InsightFace.

The ArcFace recognition model (``w600k_r50`` from the InsightFace ``buffalo_l``
pack) maps an aligned 112x112 face crop to a 512-dimensional embedding. Two
faces are the same person when their embeddings are close; FaceProof compares
them with cosine similarity in the matching module.

Embeddings are L2-normalized here so that cosine similarity reduces to a plain
dot product downstream. The model is loaded lazily, so importing this module
does not require the optional ``[ml]`` extra (CI).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from faceproof.config import settings
from faceproof.detection import align_face, detect_primary_face

if TYPE_CHECKING:
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX

EMBEDDING_DIM = 512
"""Dimensionality of an ArcFace embedding vector."""

_MODEL_PACK = "buffalo_l"
_RECOGNITION_MODEL = "w600k_r50.onnx"
_INSIGHTFACE_ROOT = "~/.insightface"


@lru_cache(maxsize=1)
def _recognizer() -> ArcFaceONNX:
    """Return the ArcFace recognition model, loaded once on first call.

    The ``buffalo_l`` pack is downloaded on first use if absent. The model is
    loaded directly via ``model_zoo`` — ``FaceAnalysis`` cannot be used here
    because its constructor mandates a detection model.
    """
    from insightface.model_zoo import get_model
    from insightface.utils.storage import ensure_available

    model_dir = ensure_available("models", _MODEL_PACK, root=_INSIGHTFACE_ROOT)
    recognizer = get_model(
        os.path.join(model_dir, _RECOGNITION_MODEL),
        providers=settings.onnx_provider_list,
    )
    recognizer.prepare(ctx_id=settings.onnx_ctx_id)
    return recognizer


def l2_normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale a vector to unit Euclidean norm; a zero vector is left unchanged."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)


def embed_aligned_face(aligned_face: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Embed an aligned 112x112 BGR face crop into a unit-norm 512-d vector."""
    raw = np.asarray(_recognizer().get_feat(aligned_face)[0], dtype=np.float32)
    return l2_normalize(raw)


def embed_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Detect the primary face in an image and return its embedding.

    Raises:
        NoFaceDetectedError: If no face is detected.
    """
    face = detect_primary_face(image)
    aligned = align_face(image, face.landmarks)
    return embed_aligned_face(aligned)
