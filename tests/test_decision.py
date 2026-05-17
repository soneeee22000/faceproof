"""Tests for the verification decision module.

The decision logic is pure FaceProof code, so the unit tests use plain
FaceMatch / LivenessResult objects and no model. One integration test runs the
full verify pipeline and is skipped without the optional ``[ml]`` extra.
"""

import pytest

from faceproof.decision import VerificationResult, decide, verify
from faceproof.liveness import LivenessResult
from faceproof.matching import FaceMatch


def _match(is_match: bool) -> FaceMatch:
    """Build a FaceMatch with a given match outcome."""
    return FaceMatch(
        similarity=0.62 if is_match else 0.11, threshold=0.25, is_match=is_match
    )


def _liveness(is_live: bool) -> LivenessResult:
    """Build a LivenessResult with a given liveness outcome."""
    return LivenessResult(
        score=0.95 if is_live else 0.04,
        is_live=is_live,
        label="live" if is_live else "spoof",
    )


def test_verified_when_face_matches_and_is_live() -> None:
    """A matching, live pair is verified."""
    assert decide(_match(True), _liveness(True)).verified is True


def test_rejected_when_face_does_not_match() -> None:
    """A face mismatch rejects, and a reason names the mismatch."""
    result = decide(_match(False), _liveness(True))
    assert result.verified is False
    assert any("mismatch" in reason.lower() for reason in result.reasons)


def test_rejected_when_selfie_is_a_spoof() -> None:
    """A presentation attack rejects, and a reason names the spoof."""
    result = decide(_match(True), _liveness(False))
    assert result.verified is False
    assert any("attack" in reason.lower() for reason in result.reasons)


def test_rejected_when_both_checks_fail() -> None:
    """Failing both checks rejects with a reason for each."""
    result = decide(_match(False), _liveness(False))
    assert result.verified is False
    assert len(result.reasons) >= 2


def test_result_exposes_every_sub_score() -> None:
    """The result carries the full face-match and liveness sub-scores."""
    result = decide(_match(True), _liveness(True))
    assert isinstance(result, VerificationResult)
    assert result.face_match.similarity == pytest.approx(0.62)
    assert result.liveness.score == pytest.approx(0.95)
    assert len(result.reasons) >= 2


def test_verify_runs_the_full_pipeline_on_a_sample_image() -> None:
    """End-to-end verify chains matching, liveness and the decision."""
    pytest.importorskip("insightface")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from insightface.data import get_image

    from faceproof.errors import LivenessModelMissingError

    image = get_image("t1")
    try:
        result = verify(image, image, match_threshold=0.5)
    except LivenessModelMissingError:
        pytest.skip("Silent-Face weights not downloaded")
    assert isinstance(result, VerificationResult)
    assert result.face_match.is_match is True
    assert isinstance(result.verified, bool)
