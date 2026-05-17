# data/

Datasets live here at evaluation/training time. They are **never committed** — licence terms
forbid redistribution, and `.gitignore` blocks the image files.

- **LFW** (Labeled Faces in the Wild) — face-verification evaluation (pairs protocol → ROC,
  FAR/FRR). Publicly available; download via the script added in the face-verification phase.
- **CelebA-Spoof** — anti-spoofing training + evaluation. Licence: **non-commercial research
  only, no redistribution**. Used locally to train/evaluate the liveness classifier; cited, never
  redistributed.

A `download_datasets.py` script is added in the build phase that first needs the data.
