# data/

Datasets live here at evaluation/training time. They are **never committed** — licence terms
forbid redistribution, and `.gitignore` blocks the image files.

- **LFW** (Labeled Faces in the Wild) — face-verification evaluation (pairs protocol → ROC,
  FAR/FRR). Downloaded automatically by `evaluation/run_lfw_evaluation.py` via scikit-learn.
- **CelebA-Spoof** — anti-spoofing training + evaluation. Licence: **non-commercial research
  only, no redistribution**. `training/celeba_spoof.py` loads a public, ungated Hugging Face
  mirror (67k pre-cropped face crops, ~5 GB) via the `datasets` library — cached here, never
  re-committed.

Model weights (InsightFace, Silent-Face) download similarly — see `models/`.
