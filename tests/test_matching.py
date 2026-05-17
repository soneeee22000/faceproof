"""Tests for the face matching module.

Matching is pure FaceProof logic — cosine similarity and a threshold decision —
so most tests need no model. One integration test runs the real pipeline on
InsightFace's bundled sample image and is skipped without the ``[ml]`` extra.
"""

import numpy as np
import pytest

from faceproof.matching import (
    FaceMatch,
    cosine_similarity,
    match_embeddings,
    match_faces,
)


def _vec(*values: float) -> np.ndarray:
    """Build a float32 vector for similarity tests."""
    return np.array(values, dtype=np.float32)


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    """Identical direction yields a similarity of 1.0."""
    assert cosine_similarity(_vec(1.0, 2.0, 3.0), _vec(1.0, 2.0, 3.0)) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    """Perpendicular vectors yield a similarity of 0.0."""
    assert cosine_similarity(_vec(1.0, 0.0), _vec(0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    """Opposite directions yield a similarity of -1.0."""
    assert cosine_similarity(_vec(1.0, 0.0), _vec(-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_similarity_with_zero_vector_is_zero() -> None:
    """A zero vector yields 0.0 rather than dividing by zero."""
    assert cosine_similarity(_vec(0.0, 0.0), _vec(1.0, 1.0)) == 0.0


def test_match_embeddings_above_threshold_is_a_match() -> None:
    """Similarity above the threshold decides a match."""
    result = match_embeddings(_vec(1.0, 0.0), _vec(1.0, 0.0), threshold=0.5)
    assert result.is_match is True


def test_match_embeddings_below_threshold_is_not_a_match() -> None:
    """Similarity below the threshold decides no match."""
    result = match_embeddings(_vec(1.0, 0.0), _vec(0.0, 1.0), threshold=0.5)
    assert result.is_match is False


def test_match_embeddings_at_threshold_is_a_match() -> None:
    """The decision boundary is inclusive: similarity == threshold matches."""
    result = match_embeddings(_vec(1.0, 0.0), _vec(0.0, 1.0), threshold=0.0)
    assert result.is_match is True


def test_face_match_exposes_similarity_and_threshold() -> None:
    """The result carries the sub-scores, never just a boolean (explainability)."""
    result = match_embeddings(_vec(1.0, 0.0), _vec(1.0, 0.0), threshold=0.42)
    assert isinstance(result, FaceMatch)
    assert result.similarity == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.42)


def test_match_faces_same_image_is_a_match() -> None:
    """The full pipeline matches an image against itself."""
    pytest.importorskip("insightface")
    from insightface.data import get_image

    image = get_image("t1")
    result = match_faces(image, image, threshold=0.5)
    assert result.is_match is True
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
