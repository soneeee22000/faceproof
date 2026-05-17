# Changelog

All notable changes to FaceProof are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — Unreleased

Phases 1–4 — face verification, liveness, the service + UI, and deployment.

### Added

- Face detection and 5-point landmark alignment using InsightFace SCRFD.
- ArcFace 512-d face embeddings (`w600k_r50`), L2-normalized.
- Cosine-similarity face matching with a data-calibrated decision threshold.
- Pure-NumPy calibration: ROC curve, AUC, equal-error rate, and accuracy-maximizing
  operating-threshold selection — fully unit-tested.
- LFW verification evaluation harness, plus a Colab notebook for GPU runs.
- LFW results — ROC AUC 0.9903, EER 2.22%, 98.81% accuracy at cosine threshold 0.2528 —
  with the report, ROC curve, and raw scores committed under `evaluation/results/`.
- Silent-Face / MiniFASNet liveness baseline — Apache-2.0 architecture vendored, weights
  fetched by a download script.
- CelebA-Spoof subset loader and a MobileNetV2 anti-spoofing classifier — 99.90%
  validation accuracy, transfer-learned on Colab GPU.
- Anti-spoofing evaluation harness — APCER / BPCER / ACER metrics (ISO/IEC 30107-3),
  trained model benchmarked against the Silent-Face baseline.
- Image upload decoding and validation — EXIF orientation, RGB normalization, size
  and type checks.
- FastAPI verification endpoints — `/api/verify`, `/api/match`, `/api/liveness` — with a
  uniform `{data, error}` response envelope and domain-error-to-HTTP-status mapping.
- React + Vite + TypeScript upload/result UI in `frontend/`, served by FastAPI in
  production so the demo ships as a single container.
- The trained MobileNetV2 wired into the production liveness path, with the Silent-Face
  baseline as fallback.
- Multi-stage production Dockerfile and a Cloud Run deployment — live at
  https://faceproof-102991984200.europe-west1.run.app.
- Project scaffold: FastAPI application, Dockerfile, GitHub Actions CI (ruff + pytest),
  and PRD-driven documentation.

### Changed

- The ONNX Runtime execution provider is configurable via `FACEPROOF_ONNX_PROVIDERS`
  (defaults to CPU, the Cloud Run deployment target).

[0.1.0]: https://github.com/soneeee22000/faceproof
