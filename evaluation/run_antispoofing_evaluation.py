"""Evaluate liveness on the held-out CelebA-Spoof test split.

Scores every held-out face with the trained MobileNetV2 and with the Silent-Face
baseline, then reports APCER / BPCER / ACER for each — a head-to-head benchmark.
The Silent-Face baseline is zero-shot (never trained on CelebA-Spoof), so the
comparison measures what task-specific fine-tuning buys.

The test split is loaded with the same seed the training used, so it is exactly
the data the trained model never saw.

Usage (from the repository root, ideally on a GPU):
    python -m evaluation.run_antispoofing_evaluation [--limit N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from evaluation.antispoofing_metrics import (
    AntiSpoofingMetrics,
    metrics_at_threshold,
    select_threshold,
)
from training.celeba_spoof import load_celeba_spoof_subset
from training.model import LIVE_CLASS, inference_transform, load_antispoofing_model

_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "antispoofing_mobilenetv2.pth"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_PROGRESS_EVERY = 500


def _score_test_split(
    test_split: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Return ``(trained live-scores, baseline live-scores, labels)`` over the split."""
    import cv2
    import torch

    from faceproof.liveness import assess_liveness

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = load_antispoofing_model(_WEIGHTS_PATH, device)
    transform = inference_transform()

    trained: list[float] = []
    baseline: list[float] = []
    labels: list[int] = []
    total = len(test_split)  # type: ignore[arg-type]
    for index in range(total):
        record = test_split[index]  # type: ignore[index]
        pil_image = record["cropped_image"].convert("RGB")
        labels.append(int(record["labels"]))

        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        trained.append(float(probabilities[LIVE_CLASS]))

        bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        whole_image = np.array([0.0, 0.0, width, height], dtype=np.float32)
        baseline.append(assess_liveness(bgr, whole_image).score)

        if (index + 1) % _PROGRESS_EVERY == 0:
            print(f"  scored {index + 1}/{total}", flush=True)
    return (
        np.asarray(trained, dtype=np.float64),
        np.asarray(baseline, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _plot_comparison(
    trained: AntiSpoofingMetrics, baseline: AntiSpoofingMetrics, destination: Path
) -> None:
    """Save a grouped bar chart comparing the two models' error rates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["APCER", "BPCER", "ACER"]
    positions = np.arange(len(names))
    width = 0.36
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.bar(positions - width / 2, [trained.apcer, trained.bpcer, trained.acer],
             width, label="MobileNetV2 (trained)", color="#1f5fbf")
    axes.bar(positions + width / 2, [baseline.apcer, baseline.bpcer, baseline.acer],
             width, label="Silent-Face (baseline)", color="#9aa0a6")
    axes.set_xticks(positions)
    axes.set_xticklabels(names)
    axes.set_ylabel("Error rate (lower is better)")
    axes.set_title("FaceProof - anti-spoofing on CelebA-Spoof")
    axes.legend()
    figure.savefig(destination, dpi=150, bbox_inches="tight")


def _write_report(
    trained: AntiSpoofingMetrics,
    baseline: AntiSpoofingMetrics,
    num_images: int,
    elapsed_seconds: float,
    destination: Path,
) -> None:
    """Write the human-readable trained-vs-baseline comparison report."""
    lines = [
        "# Anti-Spoofing Evaluation — CelebA-Spoof held-out test split",
        "",
        f"- Test images: {num_images}",
        "- Trained model: MobileNetV2 fine-tuned on the CelebA-Spoof train split",
        "- Baseline: Silent-Face / MiniFASNet (Apache-2.0), zero-shot",
        f"- Wall-clock runtime: {elapsed_seconds / 60:.1f} min",
        "",
        "## Metrics (ISO/IEC 30107-3, at each model's ACER-minimizing threshold)",
        "",
        "| Model | APCER | BPCER | ACER | Threshold |",
        "| --- | --- | --- | --- | --- |",
        f"| MobileNetV2 (trained) | {trained.apcer:.4f} | {trained.bpcer:.4f} "
        f"| {trained.acer:.4f} | {trained.threshold:.4f} |",
        f"| Silent-Face (baseline) | {baseline.apcer:.4f} | {baseline.bpcer:.4f} "
        f"| {baseline.acer:.4f} | {baseline.threshold:.4f} |",
        "",
        "Lower is better. APCER = spoofs accepted as live; BPCER = live faces rejected;",
        "ACER = their mean. The trained model is fine-tuned on this dataset; the",
        "Silent-Face baseline is zero-shot, so this measures the value of fine-tuning.",
        "Comparison chart: `antispoofing_comparison.png`. Raw scores: `antispoofing_scores.npz`.",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Score the held-out test split with both models and write the report."""
    parser = argparse.ArgumentParser(description="Anti-spoofing evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N test images (quick run)")
    args = parser.parse_args()

    print("Loading CelebA-Spoof held-out test split...", flush=True)
    test_split = load_celeba_spoof_subset(seed=42).test
    if args.limit is not None:
        test_split = test_split.select(range(min(args.limit, len(test_split))))
    print(f"Scoring {len(test_split)} test images...", flush=True)

    started = time.perf_counter()
    trained_scores, baseline_scores, labels = _score_test_split(test_split)
    elapsed_seconds = time.perf_counter() - started

    trained = metrics_at_threshold(trained_scores, labels, select_threshold(trained_scores, labels))
    baseline = metrics_at_threshold(
        baseline_scores, labels, select_threshold(baseline_scores, labels)
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _RESULTS_DIR / "antispoofing_scores.npz",
        trained=trained_scores, baseline=baseline_scores, labels=labels,
    )
    _plot_comparison(trained, baseline, _RESULTS_DIR / "antispoofing_comparison.png")
    _write_report(
        trained, baseline, len(labels), elapsed_seconds,
        _RESULTS_DIR / "antispoofing_report.md",
    )

    print(
        f"\nTrained MobileNetV2  ACER={trained.acer:.4f}  "
        f"(APCER={trained.apcer:.4f} BPCER={trained.bpcer:.4f})",
        flush=True,
    )
    print(
        f"Silent-Face baseline ACER={baseline.acer:.4f}  "
        f"(APCER={baseline.apcer:.4f} BPCER={baseline.bpcer:.4f})",
        flush=True,
    )
    print(f"Results written to {_RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
