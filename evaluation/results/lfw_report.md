# LFW Face-Verification Evaluation

- Subset: `10_folds`
- Pairs scored: 5954
- Pairs skipped (no face detected): 46
- Pipeline: SCRFD detection -> ArcFace `w600k_r50` -> cosine similarity
- Wall-clock runtime: 4.4 min

## Metrics

| Metric | Value |
| --- | --- |
| ROC AUC | 0.9903 |
| Equal Error Rate | 0.0222 |
| Operating threshold | 0.2528 |
| Accuracy @ threshold | 0.9881 |
| False Accept Rate @ threshold | 0.0000 |
| False Reject Rate @ threshold | 0.0238 |

The operating threshold maximizes verification accuracy on the LFW
pairs — the data-calibrated value `faceproof.matching` compares cosine
similarity against. ROC curve: `lfw_roc.png`. Raw scores: `lfw_scores.npz`
(reproduce every metric with `evaluation.calibration.calibrate`).
