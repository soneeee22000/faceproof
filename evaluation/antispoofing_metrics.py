"""Presentation-attack-detection metrics (ISO/IEC 30107-3).

- **APCER** — Attack Presentation Classification Error Rate: spoof presentations
  wrongly accepted as live.
- **BPCER** — Bona Fide Presentation Classification Error Rate: live presentations
  wrongly rejected as spoof.
- **ACER** — Average Classification Error Rate: the mean of APCER and BPCER.

Labels follow the CelebA-Spoof convention — 0 = live (bona fide), 1 = spoof
(attack). A higher live-score means more "live"; a face is accepted as live when
its score is ``>=`` the decision threshold. Everything here is pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

_LIVE = 0
_SPOOF = 1


@dataclass(frozen=True)
class AntiSpoofingMetrics:
    """Presentation-attack-detection error rates at one decision threshold.

    Attributes:
        apcer: Fraction of attack presentations accepted as live.
        bpcer: Fraction of bona fide presentations rejected as spoof.
        acer: Mean of ``apcer`` and ``bpcer``.
        threshold: The live-score threshold these rates were measured at.
    """

    apcer: float
    bpcer: float
    acer: float
    threshold: float


def metrics_at_threshold(
    live_scores: ArrayLike, labels: ArrayLike, threshold: float
) -> AntiSpoofingMetrics:
    """Compute APCER / BPCER / ACER at a given live-score threshold."""
    scores = np.asarray(live_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    accepted = scores >= threshold
    attacks = targets == _SPOOF
    bona_fide = targets == _LIVE
    apcer = float(accepted[attacks].mean()) if attacks.any() else 0.0
    bpcer = float((~accepted[bona_fide]).mean()) if bona_fide.any() else 0.0
    return AntiSpoofingMetrics(
        apcer=apcer, bpcer=bpcer, acer=(apcer + bpcer) / 2.0, threshold=threshold
    )


def select_threshold(live_scores: ArrayLike, labels: ArrayLike) -> float:
    """Return the live-score threshold that minimizes ACER."""
    scores = np.asarray(live_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(scores)
    accepted = scores[None, :] >= candidates[:, None]
    attacks = targets == _SPOOF
    bona_fide = targets == _LIVE
    apcer = accepted[:, attacks].mean(axis=1)
    bpcer = (~accepted[:, bona_fide]).mean(axis=1)
    return float(candidates[int(np.argmin((apcer + bpcer) / 2.0))])
