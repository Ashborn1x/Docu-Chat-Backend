# Backend

Phase 1 backend for the React frontend.

## What It Includes

- FastAPI app under `app/`
- `POST /api/chat`
- `GET /health`
- CORS enabled for local React development
- Reuses the parent project's Chroma databases from:
  - `../dbv2/chroma_db`
  - `../dbv1/chroma_db`
  - `../db/chroma_db`

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Environment

The backend loads environment variables from:

1. `backend/.env`
2. root `.env`

Copy `.env.example` to `backend/.env` if you want backend-specific values.

The Phase 1 backend is configured for Groq:

- `GROQ_API_KEY`
- `GROQ_MODEL_NAME=groq/compound-mini`
