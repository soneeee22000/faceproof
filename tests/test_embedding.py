"""Tests for the ArcFace embedding module.

The ArcFace recognition model is a trusted third-party dependency, so the unit
tests cover FaceProof's own logic — L2 normalization and the embedding
contract. Integration tests run the real model on InsightFace's bundled sample
image and are skipped wherever the optional ``[ml]`` extra is absent (e.g. CI).
"""

import numpy as np
import pytest

from faceproof.embedding import EMBEDDING_DIM, embed_image, l2_normalize


def test_embedding_dim_is_arcface_standard() -> None:
    """ArcFace (w600k_r50) embeddings are 512-dimensional."""
    assert EMBEDDING_DIM == 512


def test_l2_normalize_produces_unit_vector() -> None:
    """A normalized vector has unit Euclidean norm."""
    normed = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.linalg.norm(normed) == pytest.approx(1.0)


def test_l2_normalize_preserves_dtype() -> None:
    """Normalization keeps the float32 dtype embeddings use."""
    normed = l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert normed.dtype == np.float32


def test_l2_normalize_returns_zero_vector_unchanged() -> None:
    """A zero vector is returned as-is rather than dividing by zero."""
    zero = np.zeros(4, dtype=np.float32)
    assert np.array_equal(l2_normalize(zero), zero)


def test_embed_image_returns_unit_512_vector() -> None:
    """Embedding a real photo yields a 512-d, L2-normalized float32 vector."""
    pytest.importorskip("insightface")
    from insightface.data import get_image

    embedding = embed_image(get_image("t1"))
    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic() -> None:
    """The same image embeds to the same vector (cosine similarity ~ 1.0)."""
    pytest.importorskip("insightface")
    from insightface.data import get_image

    image = get_image("t1")
    first = embed_image(image)
    second = embed_image(image)
    assert float(first @ second) == pytest.approx(1.0, abs=1e-5)
