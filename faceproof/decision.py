"""Verification decision — combine face matching and liveness into one verdict.

A pair is *verified* only when the selfie matches the ID portrait **and** the
selfie is a live capture. The result carries every sub-score and a list of
human-readable reasons, so the decision is never an unexplained boolean
(PRD stories 1 and 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from faceproof.liveness import LivenessResult
from faceproof.matching import FaceMatch


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of verifying a selfie against an ID portrait.

    Attributes:
        verified: ``True`` only when the faces match and the selfie is live.
        face_match: The face-match sub-result (similarity, threshold, is_match).
        liveness: The liveness sub-result (score, label, is_live).
        reasons: Human-readable explanation lines covering both checks.
    """

    verified: bool
    face_match: FaceMatch
    liveness: LivenessResult
    reasons: list[str]


def decide(face_match: FaceMatch, liveness: LivenessResult) -> VerificationResult:
    """Combine a face-match and a liveness result into an explainable verdict."""
    if face_match.is_match:
        match_reason = (
            f"Face match confirmed (similarity {face_match.similarity:.3f} "
            f">= threshold {face_match.threshold:.3f})"
        )
    else:
        match_reason = (
            f"Face mismatch (similarity {face_match.similarity:.3f} "
            f"< threshold {face_match.threshold:.3f})"
        )
    if liveness.is_live:
        liveness_reason = f"Live face confirmed (liveness score {liveness.score:.3f})"
    else:
        liveness_reason = (
            f"Presentation attack detected — {liveness.label} "
            f"(liveness score {liveness.score:.3f})"
        )
    return VerificationResult(
        verified=face_match.is_match and liveness.is_live,
        face_match=face_match,
        liveness=liveness,
        reasons=[match_reason, liveness_reason],
    )


def verify(
    id_portrait: NDArray[np.uint8],
    selfie: NDArray[np.uint8],
    match_threshold: float,
) -> VerificationResult:
    """Run the full pipeline: match the selfie to the ID portrait, check the selfie
    for liveness, and combine both into a verification decision.

    Raises:
        NoFaceDetectedError: If no face is detected in either image.
        LivenessModelMissingError: If the anti-spoofing weights are not present.
    """
    from faceproof.liveness import detect_liveness
    from faceproof.matching import match_faces

    face_match = match_faces(id_portrait, selfie, match_threshold)
    liveness = detect_liveness(selfie)
    return decide(face_match, liveness)
