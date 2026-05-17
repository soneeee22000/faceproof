"""Face matching — cosine similarity and the same-person decision.

Two faces are compared by the cosine similarity of their ArcFace embeddings.
The match decision is ``similarity >= threshold``; the boundary is inclusive.

The threshold is **not** defined here. Per the project rule "calibrate, don't
guess", it is selected from the LFW ROC curve (evaluation phase) and supplied
by the caller (service config). This module is a pure, threshold-parameterized
scorer so the calibrated value has exactly one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from faceproof.embedding import embed_image


@dataclass(frozen=True)
class FaceMatch:
    """The outcome of comparing two faces.

    Attributes:
        similarity: Cosine similarity of the two embeddings, in ``[-1, 1]``.
        threshold: The decision threshold the similarity was compared against.
        is_match: ``True`` when ``similarity >= threshold``.
    """

    similarity: float
    threshold: float
    is_match: bool


def cosine_similarity(
    embedding_a: NDArray[np.float32], embedding_b: NDArray[np.float32]
) -> float:
    """Return the cosine similarity of two vectors; 0.0 if either is a zero vector."""
    norm_a = float(np.linalg.norm(embedding_a))
    norm_b = float(np.linalg.norm(embedding_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def match_embeddings(
    embedding_a: NDArray[np.float32],
    embedding_b: NDArray[np.float32],
    threshold: float,
) -> FaceMatch:
    """Decide whether two embeddings are the same person at a given threshold."""
    similarity = cosine_similarity(embedding_a, embedding_b)
    return FaceMatch(
        similarity=similarity,
        threshold=threshold,
        is_match=similarity >= threshold,
    )


def match_faces(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    threshold: float,
) -> FaceMatch:
    """Detect, embed and compare the primary face in each of two images.

    Raises:
        NoFaceDetectedError: If no face is detected in either image.
    """
    return match_embeddings(
        embed_image(image_a), embed_image(image_b), threshold
    )
