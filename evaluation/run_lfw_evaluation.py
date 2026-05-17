"""Run the LFW face-verification evaluation and calibrate the match threshold.

Downloads LFW on first use, runs the FaceProof detect -> embed -> match
pipeline over every pair, computes the ROC / AUC / EER, and selects the
operating threshold. Writes a markdown report, an ROC plot, and the raw scores
to ``evaluation/results/``.

Usage (from the repository root):
    python -m evaluation.run_lfw_evaluation [--subset {train,test,10_folds}]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluation.calibration import CalibrationResult, calibrate
from evaluation.lfw import LFWPairs, load_lfw_pairs
from faceproof.embedding import embed_image
from faceproof.errors import NoFaceDetectedError
from faceproof.matching import cosine_similarity

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_PROGRESS_EVERY = 200


def _score_pairs(
    pairs: LFWPairs,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Score every pair by cosine similarity; count pairs with no detectable face."""
    similarities: list[float] = []
    labels: list[int] = []
    skipped = 0
    total = len(pairs)
    for index in range(total):
        try:
            embedding_a = embed_image(pairs.images_a[index])
            embedding_b = embed_image(pairs.images_b[index])
        except NoFaceDetectedError:
            skipped += 1
            continue
        similarities.append(cosine_similarity(embedding_a, embedding_b))
        labels.append(int(pairs.labels[index]))
        if (index + 1) % _PROGRESS_EVERY == 0:
            print(f"  scored {index + 1}/{total} pairs ({skipped} skipped)", flush=True)
    return (
        np.asarray(similarities, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        skipped,
    )


def _plot_roc(result: CalibrationResult, destination: Path) -> None:
    """Save the ROC curve, with the operating point marked, as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot(
        result.fpr,
        result.tpr,
        color="#1f5fbf",
        label=f"ArcFace (AUC = {result.auc:.4f})",
    )
    axes.plot([0, 1], [0, 1], linestyle="--", color="#9aa0a6")
    axes.scatter(
        [result.far],
        [1.0 - result.frr],
        color="#d93025",
        zorder=5,
        label=f"Operating point (thr = {result.threshold:.4f})",
    )
    axes.set_xlabel("False Accept Rate")
    axes.set_ylabel("True Accept Rate")
    axes.set_title("FaceProof - LFW verification ROC")
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.legend(loc="lower right")
    figure.savefig(destination, dpi=150, bbox_inches="tight")


def _write_report(
    result: CalibrationResult,
    subset: str,
    skipped: int,
    elapsed_seconds: float,
    destination: Path,
) -> None:
    """Write the human-readable evaluation report."""
    lines = [
        "# LFW Face-Verification Evaluation",
        "",
        f"- Subset: `{subset}`",
        f"- Pairs scored: {result.num_pairs}",
        f"- Pairs skipped (no face detected): {skipped}",
        "- Pipeline: SCRFD detection -> ArcFace `w600k_r50` -> cosine similarity",
        f"- Runtime: {elapsed_seconds / 60:.1f} min (CPU)",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| ROC AUC | {result.auc:.4f} |",
        f"| Equal Error Rate | {result.eer:.4f} |",
        f"| Operating threshold | {result.threshold:.4f} |",
        f"| Accuracy @ threshold | {result.accuracy:.4f} |",
        f"| False Accept Rate @ threshold | {result.far:.4f} |",
        f"| False Reject Rate @ threshold | {result.frr:.4f} |",
        "",
        "The operating threshold maximizes verification accuracy on the LFW",
        "pairs — the data-calibrated value `faceproof.matching` compares cosine",
        "similarity against. ROC curve: `lfw_roc.png`. Raw scores: `lfw_scores.npz`",
        "(reproduce every metric with `evaluation.calibration.calibrate`).",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Load LFW, score the pipeline, calibrate, and write the report."""
    parser = argparse.ArgumentParser(description="LFW verification evaluation")
    parser.add_argument(
        "--subset",
        default="10_folds",
        choices=["train", "test", "10_folds"],
        help="LFW subset to evaluate (default: the full 6000-pair protocol)",
    )
    args = parser.parse_args()

    print(f"Loading LFW pairs (subset={args.subset})...", flush=True)
    pairs = load_lfw_pairs(subset=args.subset)
    print(f"Loaded {len(pairs)} pairs. Scoring on CPU...", flush=True)

    started = time.perf_counter()
    similarities, labels, skipped = _score_pairs(pairs)
    elapsed_seconds = time.perf_counter() - started

    result = calibrate(similarities, labels)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _RESULTS_DIR / "lfw_scores.npz", similarities=similarities, labels=labels
    )
    _plot_roc(result, _RESULTS_DIR / "lfw_roc.png")
    _write_report(
        result, args.subset, skipped, elapsed_seconds, _RESULTS_DIR / "lfw_report.md"
    )

    print(
        f"\nDone. AUC={result.auc:.4f} EER={result.eer:.4f} "
        f"threshold={result.threshold:.4f} accuracy={result.accuracy:.4f}",
        flush=True,
    )
    print(f"Results written to {_RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
