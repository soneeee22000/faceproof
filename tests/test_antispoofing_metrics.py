"""Tests for the presentation-attack-detection metrics.

APCER / BPCER / ACER are pure NumPy logic, so every test runs on deterministic
synthetic scores with no model.
"""

import pytest

from evaluation.antispoofing_metrics import metrics_at_threshold, select_threshold

# Label convention: 0 = live (bona fide), 1 = spoof (attack).
_LABELS = [0, 0, 1, 1]


def test_metrics_zero_when_scores_separate_the_classes() -> None:
    """Live faces scored high and spoofs scored low give zero error."""
    result = metrics_at_threshold([0.9, 0.8, 0.1, 0.2], _LABELS, threshold=0.5)
    assert result.apcer == 0.0
    assert result.bpcer == 0.0
    assert result.acer == 0.0


def test_apcer_counts_attacks_accepted_as_live() -> None:
    """Every spoof scoring above the threshold counts toward APCER."""
    result = metrics_at_threshold([0.9, 0.9, 0.7, 0.8], _LABELS, threshold=0.5)
    assert result.apcer == pytest.approx(1.0)
    assert result.bpcer == pytest.approx(0.0)


def test_bpcer_counts_bona_fide_rejected_as_spoof() -> None:
    """Every live face scoring below the threshold counts toward BPCER."""
    result = metrics_at_threshold([0.2, 0.1, 0.1, 0.2], _LABELS, threshold=0.5)
    assert result.bpcer == pytest.approx(1.0)
    assert result.apcer == pytest.approx(0.0)


def test_acer_is_the_mean_of_apcer_and_bpcer() -> None:
    """ACER averages the two error rates."""
    result = metrics_at_threshold([0.9, 0.4, 0.6, 0.2], _LABELS, threshold=0.5)
    assert result.apcer == pytest.approx(0.5)
    assert result.bpcer == pytest.approx(0.5)
    assert result.acer == pytest.approx(0.5)


def test_select_threshold_minimizes_acer() -> None:
    """The selected threshold drives ACER to its minimum."""
    scores = [0.9, 0.8, 0.1, 0.2]
    threshold = select_threshold(scores, _LABELS)
    assert metrics_at_threshold(scores, _LABELS, threshold).acer == 0.0
    assert 0.2 < threshold <= 0.8
