"""Threshold calibration for face verification.

Given the cosine similarities of a set of labelled face pairs (1 = genuine,
0 = impostor), this module computes the ROC curve, the area under it (AUC),
the equal-error rate (EER), and selects the operating threshold that maximizes
verification accuracy — the data-calibrated value the matching module compares
against. Everything is pure NumPy so it is fully unit-testable and dependency-
light: no model, no scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of calibrating a verification threshold.

    Attributes:
        threshold: Operating threshold; pairs with ``similarity >= threshold`` match.
        auc: Area under the ROC curve, in ``[0, 1]``.
        eer: Equal-error rate — the rate where false accepts equal false rejects.
        accuracy: Verification accuracy at ``threshold``.
        far: False-accept rate at ``threshold`` (impostors wrongly accepted).
        frr: False-reject rate at ``threshold`` (genuine pairs wrongly rejected).
        fpr: ROC false-positive-rate axis, ascending.
        tpr: ROC true-positive-rate axis, ascending.
        num_pairs: Number of pairs the calibration was computed over.
    """

    threshold: float
    auc: float
    eer: float
    accuracy: float
    far: float
    frr: float
    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    num_pairs: int


def _roc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> tuple[
    NDArray[np.float64], NDArray[np.float64]
]:
    """Return the ROC ``(fpr, tpr)`` curve, each starting at 0 and ending at 1."""
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    cumulative_tp = np.cumsum(ranked)
    cumulative_fp = np.cumsum(1 - ranked)
    tpr = np.concatenate(([0.0], cumulative_tp / cumulative_tp[-1]))
    fpr = np.concatenate(([0.0], cumulative_fp / cumulative_fp[-1]))
    return fpr, tpr


def _auc(fpr: NDArray[np.float64], tpr: NDArray[np.float64]) -> float:
    """Integrate the ROC curve with the trapezoidal rule."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))  # numpy < 2.0


def _equal_error_rate(
    fpr: NDArray[np.float64], tpr: NDArray[np.float64]
) -> float:
    """Return the rate where the false-accept and false-reject rates meet."""
    fnr = 1.0 - tpr
    crossing = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[crossing] + fnr[crossing]) / 2.0)


def _select_threshold(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[float, float]:
    """Return the accuracy-maximizing ``(threshold, accuracy)`` over the scores."""
    candidates = np.unique(scores)
    predictions = scores[None, :] >= candidates[:, None]
    genuine = labels.astype(bool)
    accuracies = (predictions == genuine[None, :]).mean(axis=1)
    best = int(np.argmax(accuracies))
    return float(candidates[best]), float(accuracies[best])


def _error_rates(
    scores: NDArray[np.float64], labels: NDArray[np.int64], threshold: float
) -> tuple[float, float]:
    """Return ``(far, frr)`` at a given decision threshold."""
    accepted = scores >= threshold
    impostors = labels == 0
    genuine = labels == 1
    far = float(accepted[impostors].mean())
    frr = float((~accepted[genuine]).mean())
    return far, frr


def calibrate(similarities: ArrayLike, labels: ArrayLike) -> CalibrationResult:
    """Calibrate a verification threshold from labelled pair similarities.

    Args:
        similarities: Cosine similarity of each face pair.
        labels: Binary labels — 1 for genuine pairs, 0 for impostor pairs.

    Raises:
        ValueError: If the inputs differ in shape, or do not contain both classes.
    """
    scores = np.asarray(similarities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.shape != targets.shape:
        raise ValueError("similarities and labels must have the same shape")
    if not np.array_equal(np.unique(targets), np.array([0, 1])):
        raise ValueError("labels must contain both classes: 0 (impostor) and 1 (genuine)")

    fpr, tpr = _roc(scores, targets)
    threshold, accuracy = _select_threshold(scores, targets)
    far, frr = _error_rates(scores, targets, threshold)
    return CalibrationResult(
        threshold=threshold,
        auc=_auc(fpr, tpr),
        eer=_equal_error_rate(fpr, tpr),
        accuracy=accuracy,
        far=far,
        frr=frr,
        fpr=fpr,
        tpr=tpr,
        num_pairs=int(scores.size),
    )
