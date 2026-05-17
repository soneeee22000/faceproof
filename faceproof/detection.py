"""Face detection and alignment using InsightFace SCRFD.

Detection produces a bounding box, five facial landmarks, and a confidence
score for every face in an image. Alignment warps a face to the canonical
112x112 ArcFace crop from those landmarks, ready for embedding.

Input images follow the OpenCV/InsightFace convention: ``uint8`` arrays of
shape ``(H, W, 3)`` in BGR channel order — the same convention the embedding
stage expects, so the whole pipeline stays color-consistent.

The InsightFace model is loaded lazily on first use, so importing this module
(e.g. in CI without the optional ``[ml]`` extra) does not pull in the CV stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from faceproof.errors import NoFaceDetectedError

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

ALIGNED_FACE_SIZE = 112
"""Side length of the canonical ArcFace crop, in pixels."""

_MODEL_PACK = "buffalo_l"
_DETECTION_SIZE = (640, 640)
_CPU_CTX_ID = -1
_CPU_PROVIDERS = ["CPUExecutionProvider"]


@dataclass(frozen=True)
class DetectedFace:
    """A single detected face.

    Attributes:
        bbox: Bounding box ``[x1, y1, x2, y2]``, shape ``(4,)``.
        landmarks: Five-point landmarks (eyes, nose, mouth corners), shape ``(5, 2)``.
        det_score: Detection confidence in ``[0, 1]``.
    """

    bbox: NDArray[np.float32]
    landmarks: NDArray[np.float32]
    det_score: float


@lru_cache(maxsize=1)
def _detector() -> FaceAnalysis:
    """Return the SCRFD detector, loading the model pack once on first call."""
    from insightface.app import FaceAnalysis

    detector = FaceAnalysis(
        name=_MODEL_PACK,
        allowed_modules=["detection"],
        providers=_CPU_PROVIDERS,
    )
    detector.prepare(ctx_id=_CPU_CTX_ID, det_size=_DETECTION_SIZE)
    return detector


def detect_faces(image: NDArray[np.uint8]) -> list[DetectedFace]:
    """Detect every face in a BGR ``uint8`` image."""
    return [
        DetectedFace(
            bbox=np.asarray(face.bbox, dtype=np.float32),
            landmarks=np.asarray(face.kps, dtype=np.float32),
            det_score=float(face.det_score),
        )
        for face in _detector().get(image)
    ]


def select_primary_face(faces: list[DetectedFace]) -> DetectedFace:
    """Return the highest-confidence face.

    Raises:
        NoFaceDetectedError: If ``faces`` is empty.
    """
    if not faces:
        raise NoFaceDetectedError("No face detected in the image.")
    return max(faces, key=lambda face: face.det_score)


def detect_primary_face(image: NDArray[np.uint8]) -> DetectedFace:
    """Detect faces in an image and return the highest-confidence one.

    Raises:
        NoFaceDetectedError: If no face is detected.
    """
    return select_primary_face(detect_faces(image))


def align_face(
    image: NDArray[np.uint8], landmarks: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """Warp a face to the canonical 112x112 ArcFace crop from its landmarks."""
    from insightface.utils import face_align

    aligned: NDArray[np.uint8] = face_align.norm_crop(
        image, landmarks, image_size=ALIGNED_FACE_SIZE
    )
    return aligned
