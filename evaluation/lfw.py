"""LFW pair loading for face-verification evaluation.

The Labeled Faces in the Wild verification benchmark is downloaded on first use
via scikit-learn into ``data/lfw/`` (gitignored — see ``data/README.md``).
Pairs are returned as uint8 BGR images — the convention the FaceProof pipeline
expects — each with a binary label: 1 = same person, 0 = different.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_DATA_HOME = Path(__file__).resolve().parents[1] / "data" / "lfw"
_RESIZE = 0.5


@dataclass(frozen=True)
class LFWPairs:
    """A loaded set of LFW verification pairs.

    Attributes:
        images_a: First image of each pair (uint8 BGR).
        images_b: Second image of each pair (uint8 BGR).
        labels: Binary label per pair — 1 = genuine, 0 = impostor.
    """

    images_a: list[NDArray[np.uint8]]
    images_b: list[NDArray[np.uint8]]
    labels: NDArray[np.int64]

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.labels)


def _to_bgr_uint8(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Convert a scikit-learn LFW image (RGB float) to a uint8 BGR array."""
    array = np.asarray(image, dtype=np.float64)
    if float(array.max()) <= 1.0:
        array = array * 255.0
    rgb = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def load_lfw_pairs(subset: str = "10_folds") -> LFWPairs:
    """Download (if needed) and load LFW verification pairs.

    Args:
        subset: One of ``"train"``, ``"test"``, or ``"10_folds"`` — the full
            6000-pair standard verification protocol.
    """
    from sklearn.datasets import fetch_lfw_pairs

    _DATA_HOME.mkdir(parents=True, exist_ok=True)
    dataset = fetch_lfw_pairs(
        subset=subset,
        funneled=True,
        color=True,
        resize=_RESIZE,
        slice_=None,
        data_home=str(_DATA_HOME),
        download_if_missing=True,
    )
    images_a = [_to_bgr_uint8(pair[0]) for pair in dataset.pairs]
    images_b = [_to_bgr_uint8(pair[1]) for pair in dataset.pairs]
    return LFWPairs(
        images_a=images_a,
        images_b=images_b,
        labels=np.asarray(dataset.target, dtype=np.int64),
    )
