"""Tests for the face detection + alignment module.

The InsightFace SCRFD model itself is a trusted third-party dependency (per the
PRD testing strategy), so the unit tests here cover FaceProof's own logic —
primary-face selection and the no-face error. One integration test exercises
real detection on InsightFace's bundled sample image; it is skipped wherever the
optional ``[ml]`` extra is not installed (e.g. CI).
"""

import numpy as np
import pytest

from faceproof.detection import (
    ALIGNED_FACE_SIZE,
    DetectedFace,
    detect_primary_face,
    select_primary_face,
)
from faceproof.errors import NoFaceDetectedError


def _face(det_score: float) -> DetectedFace:
    """Build a DetectedFace with a given confidence and placeholder geometry."""
    return DetectedFace(
        bbox=np.array([0.0, 0.0, 10.0, 10.0]),
        landmarks=np.zeros((5, 2)),
        det_score=det_score,
    )


def test_select_primary_face_picks_highest_confidence() -> None:
    """With several faces, the highest-confidence one is chosen (edge case P1)."""
    primary = select_primary_face([_face(0.62), _face(0.97), _face(0.81)])
    assert primary.det_score == pytest.approx(0.97)


def test_select_primary_face_raises_when_no_faces() -> None:
    """An empty detection list raises NoFaceDetectedError (edge case P0)."""
    with pytest.raises(NoFaceDetectedError):
        select_primary_face([])


def test_aligned_face_size_is_arcface_canonical() -> None:
    """Alignment targets the 112x112 crop ArcFace expects."""
    assert ALIGNED_FACE_SIZE == 112


def test_align_face_returns_canonical_crop() -> None:
    """align_face warps an image to a 112x112x3 crop from 5-point landmarks."""
    pytest.importorskip("insightface")
    from faceproof.detection import align_face

    image = np.zeros((640, 480, 3), dtype=np.uint8)
    landmarks = np.array(
        [[180, 240], [300, 240], [240, 300], [195, 360], [285, 360]],
        dtype=np.float32,
    )
    aligned = align_face(image, landmarks)
    assert aligned.shape == (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE, 3)
    assert aligned.dtype == np.uint8


def test_detect_primary_face_on_sample_image() -> None:
    """Real SCRFD detection finds a face in InsightFace's bundled sample image.

    Downloads the model pack on first run; skipped without the ``[ml]`` extra.
    """
    pytest.importorskip("insightface")
    from insightface.data import get_image

    face = detect_primary_face(get_image("t1"))
    assert face.det_score > 0.5
    assert face.landmarks.shape == (5, 2)
    assert face.bbox.shape == (4,)
