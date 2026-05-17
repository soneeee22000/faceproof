# Project: FaceProof

## Quick Reference

- **Stack:** Python 3.10 + FastAPI + InsightFace (SCRFD + ArcFace) + PyTorch + React
- **Python version:** 3.10
- **Package manager:** pip (`pip install -e ".[dev]"`, plus `".[ml]"` for the CV stack)
- **PRD:** See `docs/PRD.md` — the source of truth for all features
- **Architecture:** See `docs/ARCHITECTURE.md`

## Commands

```bash
uvicorn faceproof.api:app --reload   # start dev server
pytest -q                            # run tests
ruff check .                         # lint
mypy faceproof                       # type check
docker build -t faceproof .          # production build
```

## Project Structure

```
faceproof/
├── faceproof/          # inference package (detection, embedding, matching, liveness, pipeline, api)
├── training/           # anti-spoofing training on CelebA-Spoof
├── evaluation/         # LFW + CelebA-Spoof evaluation notebooks
├── models/             # weights — downloaded, never committed
├── data/               # datasets — downloaded, never committed
├── frontend/           # React upload/result UI
├── tests/              # pytest — mirrors faceproof/
└── docs/               # PRD.md (source of truth), ARCHITECTURE.md
```

No `docker-compose.yml` — the service is stateless with no backing services.

---

## BUILD RULES (Claude MUST follow these)

### Rule 1: PRD Is Law

- Every feature, endpoint, and component MUST trace back to a user story in `docs/PRD.md`.
- If a feature is not in the PRD, DO NOT build it — ask first. v2 (document/MRZ) is out of scope.
- If requirements are ambiguous, STOP and ask before writing code.

### Rule 2: Build in Order

- Follow the milestone phases in the PRD strictly. Complete Phase N before Phase N+1.
- Each phase has a quality gate — ALL gate criteria must pass before proceeding.

### Rule 3: Test Before Implement

- Write the test file FIRST for every new module. It should fail (red), then implement to green.
- No exceptions — untested code is unshipped code.

### Rule 4: One Thing at a Time

- Implement ONE user story at a time: test → implement → verify → commit.
- Commit after each completed story with a conventional commit message.

### Rule 5: Error Handling Is Not Optional

- Every endpoint handles: validation errors, no-face-detected, oversized/invalid uploads, server errors.
- Every frontend action handles: loading, success, error, empty states.
- Use the edge-case table in the PRD as a checklist.

### Rule 6: No Dead Code, No TODOs in Commits

- Remove all `print()` debug statements before committing.
- No `# TODO` comments — implement it now or add it to the PRD. No commented-out code.

### Rule 7: Types Are Mandatory

- All functions have type hints (strict mypy compliance).
- All API request/response bodies have Pydantic schemas.

---

## PROJECT-SPECIFIC RULES

- **Generic framing only.** This is a public, generic identity-verification reference. Do NOT
  reference any specific employer, recruiter, or hiring context anywhere in the repo.
- **Never commit datasets or model weights.** `data/` and `models/` ship download scripts +
  citations only. Verify `.gitignore` before every commit.
- **Honest evaluation.** Every metric in the README or docs must be reproducible from
  `evaluation/`. No hand-picked or self-graded numbers.
- **Calibrate, don't guess.** Decision thresholds (face match, liveness) are selected from data
  (ROC curves), never hardcoded by guess.
- **Scope is frozen.** v1 = the 6 Definition-of-Done items in the PRD. Document/MRZ is v2, a
  separate branch started only after v1 ships. Extra time buys polish, never a 7th feature.

## API Response Pattern

- Success: `{ "data": <result>, "error": null }`
- Error: `{ "data": null, "error": { "code": "NO_FACE_DETECTED", "message": "..." } }`
- Always the same shape — the frontend never inspects response structure to branch.

## QUALITY GATES

### Before Each Commit

- [ ] All tests pass · new tests written for new code
- [ ] No lint errors (`ruff check .`) · no type errors (`mypy faceproof`)
- [ ] No debug `print()` statements
- [ ] Conventional commit message (feat/fix/refactor/test/docs/chore)

### Before Phase Transition

- [ ] All stories in the phase have passing tests
- [ ] The phase's PRD quality gate is satisfied

### Before Shipping (v1)

- [ ] All 6 Definition-of-Done items green
- [ ] 80%+ coverage on pipeline business logic
- [ ] Evaluation report reproducible
- [ ] `.env.example` current · README setup steps verified
- [ ] Deployed to a live public URL

## ENVIRONMENT VARIABLES

```bash
# Copy from .env.example. NEVER commit .env.
FACEPROOF_SERVICE_NAME=faceproof
FACEPROOF_CORS_ORIGINS=http://localhost:5173
FACEPROOF_MAX_UPLOAD_BYTES=8388608
```

## KNOWN PATTERNS & DECISIONS

- Stateless service, no database (ADR-003).
- Anti-spoofing: trained CNN + Silent-Face baseline (ADR-001).
- CPU-only deployment on Cloud Run (ADR-004).
