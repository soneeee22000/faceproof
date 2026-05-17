# Anti-Spoofing Evaluation — CelebA-Spoof held-out test split

- Test images: 10020
- Trained model: MobileNetV2 fine-tuned on the CelebA-Spoof train split
- Baseline: Silent-Face / MiniFASNet (Apache-2.0), zero-shot
- Wall-clock runtime: 5.7 min

## Metrics (ISO/IEC 30107-3, at each model's ACER-minimizing threshold)

| Model | APCER | BPCER | ACER | Threshold |
| --- | --- | --- | --- | --- |
| MobileNetV2 (trained) | 0.0004 | 0.0000 | 0.0002 | 0.1058 |
| Silent-Face (baseline) | 0.9682 | 0.0239 | 0.4960 | 0.0139 |

Lower is better. APCER = spoofs accepted as live; BPCER = live faces rejected;
ACER = their mean. The trained model is fine-tuned on this dataset, so it holds a
home-field advantage; the Silent-Face baseline is zero-shot **and** receives the
pre-cropped CelebA-Spoof faces, while it is designed to crop a face from a wider
frame with surrounding context — so it runs outside its intended input conditions.
The comparison shows the value of task-specific fine-tuning, not a like-for-like
baseline benchmark. Comparison chart: `antispoofing_comparison.png`.
