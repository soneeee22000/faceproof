"""CelebA-Spoof subset loader for anti-spoofing training.

Uses a public, ungated Hugging Face mirror of CelebA-Spoof — 67,170 pre-cropped
face images labelled live (0) or spoof (1). The Hugging Face ``datasets`` library
downloads and caches it on first use (~5 GB); no credentials are required. Those
images are re-split here into stratified train / validation / test sets.

The full 625k-image CelebA-Spoof dataset is non-commercial, no-redistribution;
this mirror is used locally for training and never re-committed (see
``data/README.md``). ``datasets`` is imported lazily so this module imports
without the optional ``[ml]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_DATASET_ID = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"

LIVE_LABEL = 0
SPOOF_LABEL = 1


@dataclass(frozen=True)
class CelebASpoofSubset:
    """A stratified train / validation / test split of CelebA-Spoof face crops.

    Each field is a Hugging Face ``Dataset`` with a ``cropped_image`` column and
    an integer ``labels`` column (0 = live, 1 = spoof).
    """

    train: Any
    validation: Any
    test: Any


def stratified_split(
    labels: NDArray[np.int64],
    train_fraction: float,
    val_fraction: float,
    seed: int = 42,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Split sample indices into stratified train / validation / test index arrays.

    Whatever fraction remains after train and validation becomes the test split.
    Each class keeps its proportion in every split; the result is deterministic
    given ``seed``.

    Raises:
        ValueError: If the fractions are out of range or leave no test split.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("train_fraction and val_fraction must lie within (0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave room for a test split")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_parts: list[NDArray[np.int64]] = []
    val_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    for class_value in np.unique(labels):
        indices = np.where(labels == class_value)[0]
        rng.shuffle(indices)
        n_train = int(len(indices) * train_fraction)
        n_val = int(len(indices) * val_fraction)
        train_parts.append(indices[:n_train])
        val_parts.append(indices[n_train : n_train + n_val])
        test_parts.append(indices[n_train + n_val :])
    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(val_parts)),
        np.sort(np.concatenate(test_parts)),
    )


def load_celeba_spoof_subset(
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    subset_size: int | None = None,
    seed: int = 42,
) -> CelebASpoofSubset:
    """Download (cached) the CelebA-Spoof mirror and split it for training.

    Args:
        train_fraction: Share of images for the training split.
        val_fraction: Share for validation; the remainder becomes the test split.
        subset_size: If set, first stratified-subsample to this many images —
            useful for a fast iteration before a full run.
        seed: Seed for the optional subsample and the split.
    """
    from datasets import load_dataset

    dataset = load_dataset(_DATASET_ID, split="test")
    labels = np.asarray(dataset["labels"], dtype=np.int64)

    if subset_size is not None and subset_size < len(labels):
        keep, _, _ = stratified_split(labels, subset_size / len(labels), 0.0, seed)
        dataset = dataset.select(keep)
        labels = labels[keep]

    train_idx, val_idx, test_idx = stratified_split(labels, train_fraction, val_fraction, seed)
    return CelebASpoofSubset(
        train=dataset.select(train_idx),
        validation=dataset.select(val_idx),
        test=dataset.select(test_idx),
    )
