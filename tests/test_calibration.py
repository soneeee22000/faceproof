"""Tests for the threshold-calibration logic.

Calibration is pure FaceProof logic — ROC, AUC, EER and operating-threshold
selection computed in NumPy — so every test runs without the CV stack and on
deterministic synthetic data.
"""

import numpy as np
import pytest

from evaluation.calibration import calibrate


def test_calibrate_perfect_separation() -> None:
    """Cleanly separable scores give AUC 1.0 and a threshold between the classes."""
    result = calibrate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert result.auc == pytest.approx(1.0)
    assert result.accuracy == pytest.approx(1.0)
    assert result.eer == pytest.approx(0.0)
    assert result.far == pytest.approx(0.0)
    assert result.frr == pytest.approx(0.0)
    assert 0.2 < result.threshold <= 0.8


def test_calibrate_inverted_scores_give_zero_auc() -> None:
    """Scores that rank impostors above genuine pairs give AUC 0.0."""
    result = calibrate(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1]))
    assert result.auc == pytest.approx(0.0)


def test_calibrate_roc_runs_from_origin_to_corner() -> None:
    """The ROC curve starts at (0, 0), ends at (1, 1) and is monotonic."""
    result = calibrate(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1]))
    assert result.fpr[0] == 0.0
    assert result.tpr[0] == 0.0
    assert result.fpr[-1] == pytest.approx(1.0)
    assert result.tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.fpr) >= 0)
    assert np.all(np.diff(result.tpr) >= 0)


def test_calibrate_threshold_reproduces_reported_accuracy() -> None:
    """Applying ``score >= threshold`` reproduces the reported accuracy."""
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.55, 0.2])
    labels = np.array([0, 1, 0, 1, 1, 0])
    result = calibrate(scores, labels)
    predicted = scores >= result.threshold
    accuracy = float((predicted == (labels == 1)).mean())
    assert accuracy == pytest.approx(result.accuracy)


def test_calibrate_metrics_stay_in_unit_range() -> None:
    """Every reported rate is a valid probability."""
    result = calibrate(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 1, 0, 1]))
    for value in (result.auc, result.eer, result.accuracy, result.far, result.frr):
        assert 0.0 <= value <= 1.0


def test_calibrate_requires_both_classes() -> None:
    """A single-class label set cannot define a verification ROC."""
    with pytest.raises(ValueError, match="both classes"):
        calibrate(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))


def test_calibrate_rejects_mismatched_shapes() -> None:
    """Scores and labels must be aligned."""
    with pytest.raises(ValueError, match="same shape"):
        calibrate(np.array([0.1, 0.2]), np.array([1, 0, 1]))
