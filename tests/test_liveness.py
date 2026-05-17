"""Tests for the Silent-Face liveness module.

The MiniFASNet network is trusted vendored third-party code, so the unit tests
cover FaceProof's own logic — model-name parsing, the crop box, and score
fusion. One integration test runs the real models; it is skipped when the
optional ``[ml]`` extra or the pretrained weights are absent.
"""

import numpy as np
import pytest

from faceproof.errors import LivenessModelMissingError
from faceproof.liveness import (
    LivenessResult,
    _crop_box,
    _fuse,
    _kernel_size,
    _parse_model_name,
    detect_liveness,
)


def test_parse_model_name_v2() -> None:
    """A scale-2.7 MiniFASNetV2 filename parses into its components."""
    assert _parse_model_name("2.7_80x80_MiniFASNetV2.pth") == (80, 80, "MiniFASNetV2", 2.7)


def test_parse_model_name_v1se() -> None:
    """A scale-4.0 MiniFASNetV1SE filename parses into its components."""
    assert _parse_model_name("4_0_0_80x80_MiniFASNetV1SE.pth") == (80, 80, "MiniFASNetV1SE", 4.0)


def test_kernel_size_for_80px_input() -> None:
    """The final-conv kernel for an 80x80 input is 5x5."""
    assert _kernel_size(80, 80) == (5, 5)


def test_crop_box_stays_within_image() -> None:
    """A scaled crop box is clamped inside the image bounds."""
    x1, y1, x2, y2 = _crop_box(640, 480, (300.0, 200.0, 120.0, 120.0), 2.7)
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480


def test_fuse_predicts_live_when_real_class_dominates() -> None:
    """Summed probabilities favouring the real class decide a live face."""
    probabilities = [np.array([0.1, 0.8, 0.1]), np.array([0.2, 0.7, 0.1])]
    score, is_live = _fuse(probabilities)
    assert is_live is True
    assert score == pytest.approx(0.75)


def test_fuse_predicts_spoof_when_attack_class_dominates() -> None:
    """Summed probabilities favouring an attack class decide a spoof."""
    probabilities = [np.array([0.8, 0.1, 0.1]), np.array([0.7, 0.2, 0.1])]
    score, is_live = _fuse(probabilities)
    assert is_live is False
    assert score == pytest.approx(0.15)


def test_detect_liveness_on_sample_image() -> None:
    """The full liveness pipeline returns a well-formed result on a real photo."""
    pytest.importorskip("torch")
    from insightface.data import get_image

    try:
        result = detect_liveness(get_image("t1"))
    except LivenessModelMissingError:
        pytest.skip("Silent-Face weights not downloaded — run models/download_silentface.py")
    assert isinstance(result, LivenessResult)
    assert 0.0 <= result.score <= 1.0
    assert result.label in {"live", "spoof"}
