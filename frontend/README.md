# FaceProof frontend

A thin React + Vite + TypeScript UI for the FaceProof verification API — upload
an ID portrait and a selfie, get an explainable verified/rejected result.

## Development

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 — proxies /api to the backend on :8000
```

Run the API alongside it from the repository root:

```bash
uvicorn faceproof.api:app --reload
```

## Production build

```bash
npm run build        # type-checks, then bundles to frontend/dist/
```

When `frontend/dist/` exists, the FastAPI app serves it directly, so the whole
demo ships as a single container.
