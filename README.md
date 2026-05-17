# FaceProof

Face verification + liveness detection — an open reference implementation of the core
computer-vision subsystem behind identity verification.

Given an ID-style portrait and a selfie, FaceProof decides **whether they are the same person**
and **whether the selfie is a live face** (not a printout or a screen replay) — and ships a
reproducible evaluation report behind every claim.

> Non-commercial portfolio / reference-implementation project. Source of truth: `docs/PRD.md`.

## What it does

- **Face verification** — detection → alignment → ArcFace embeddings → cosine similarity against a
  threshold **calibrated on LFW** (not guessed).
- **Liveness / anti-spoofing** — a CNN classifier trained on CelebA-Spoof, benchmarked against the
  Apache-2.0 Silent-Face baseline; rejects print and replay attacks.
- **Honest evaluation** — ROC / FAR / FRR on LFW; APCER / BPCER / ACER on CelebA-Spoof —
  reproducible from `evaluation/`.

## Stack

Python 3.10 · FastAPI · InsightFace (SCRFD + ArcFace) · PyTorch · React · Docker · GCP Cloud Run.
Stateless service — no database.

## Setup

```bash
git clone <repo-url>
cd faceproof
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"              # core + dev (lint/test)
pip install -e ".[ml]"               # CV/ML stack — needed from the face-verification phase on
cp .env.example .env
```

## Commands

```bash
uvicorn faceproof.api:app --reload   # run the API
pytest -q                            # run tests
ruff check .                         # lint
mypy faceproof                       # type-check
```

## Architecture

See `docs/ARCHITECTURE.md`. A stateless FastAPI service runs the verification pipeline
(detect → embed → match → liveness → decision); no database; deployed as one CPU-only container
on Cloud Run. The React frontend is a thin upload/result UI.

## Project status

Built in phases (see `docs/PRD.md`): scaffold → face verification → liveness → service + UI →
deploy. v1 ships when the six Definition-of-Done items are green.

## Licensing

Project code: MIT. InsightFace **code** is MIT; its **pretrained weights** are non-commercial
research only — FaceProof is non-commercial. Silent-Face / MiniFASNet weights are Apache 2.0.
Datasets (LFW, CelebA-Spoof) are not redistributed — see `data/README.md`.
