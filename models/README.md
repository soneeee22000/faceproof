# models/

Model weights live here. They are **downloaded by scripts, never committed** (see `.gitignore`).

- **Face detection + ArcFace embeddings** — InsightFace model pack (auto-downloaded by the
  `insightface` package on first use). Code: MIT. Pretrained weights: non-commercial research
  use — FaceProof is a non-commercial portfolio piece.
- **Liveness baseline** — Silent-Face / MiniFASNet pretrained weights (Apache 2.0).
- **Liveness (trained)** — the anti-spoofing classifier trained on CelebA-Spoof in `training/`.

A `download_models.py` script is added in the build phase that first needs the weights.
For Docker builds, weights are fetched during image build so the container is self-contained.
