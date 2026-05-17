"""Tests for the CelebA-Spoof subset loader.

The Hugging Face download is an external dependency exercised on Colab during
training; the unit tests here cover FaceProof's own logic — the stratified
train / validation / test split.
"""

import numpy as np
import pytest

from training.celeba_spoof import stratified_split


def _labels() -> np.ndarray:
    """100 live (0) and 60 spoof (1) labels."""
    return np.array([0] * 100 + [1] * 60, dtype=np.int64)


def test_split_is_disjoint_and_complete() -> None:
    """The three splits partition every sample exactly once."""
    train, val, test = stratified_split(_labels(), 0.7, 0.15, seed=1)
    combined = np.concatenate([train, val, test])
    assert len(combined) == 160
    assert len(np.unique(combined)) == 160


def test_split_preserves_class_proportions() -> None:
    """Each split keeps the dataset's 100:60 live:spoof ratio."""
    labels = _labels()
    for part in stratified_split(labels, 0.7, 0.15, seed=1):
        live_fraction = float(np.mean(labels[part] == 0))
        assert live_fraction == pytest.approx(100 / 160, abs=0.05)


def test_split_is_deterministic() -> None:
    """The same seed yields the same split."""
    first = stratified_split(_labels(), 0.7, 0.15, seed=7)
    second = stratified_split(_labels(), 0.7, 0.15, seed=7)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_split_rejects_fractions_that_leave_no_test_set() -> None:
    """Train + validation fractions must leave room for a test split."""
    with pytest.raises(ValueError, match="test split"):
        stratified_split(_labels(), 0.8, 0.2, seed=1)
