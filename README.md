# Backend

Phase 1 backend for the React frontend.

## What It Includes

- FastAPI app under `app/`
- `POST /api/chat`
- `GET /health`
- CORS enabled for local React development
- Provider profiles for `groq` and `gemini`
- API key failover for chat and embedding requests
- Reuses the parent project's Chroma databases from:
  - `../dbv2/chroma_db`
  - `../dbv1/chroma_db`
  - `../db/chroma_db`

## Run

From the `python-backend` directory:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API root: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Environment

The backend loads environment variables from:

1. `python-backend/.env`
2. project root `.env`

Create `python-backend/.env` if you want backend-specific values. You can start from `python-backend/.env.example`.

### General app settings

- `APP_ENV=development`
- `API_TITLE=RAG Backend API`
- `ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173`
- `DEFAULT_TOP_K=4`

### Auth and security

- `REQUIRE_AUTH=false`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SECRET_KEY`
- `SUPABASE_STORAGE_BUCKET=documents`
- `USE_SUPABASE_VECTORS=false`
- `SUPABASE_VECTOR_RPC_NAME=match_document_chunks`
- `RATE_LIMIT_PER_MINUTE=30`
- `MAX_UPLOAD_SIZE_MB=20`
- `ALLOWED_UPLOAD_EXTENSIONS=pdf,csv,docx,txt`

With `REQUIRE_AUTH=true`, protected routes require a Bearer token and the backend expects a confirmed email on the authenticated Supabase user.
Set `RATE_LIMIT_PER_MINUTE=0` to disable rate limiting entirely.

To migrate production retrieval away from local Chroma, run the SQL in `python-backend/supabase_pgvector_migration.sql` and then set `USE_SUPABASE_VECTORS=true`.

To persist document partition metrics such as text sections, tables, and atomic elements, run the SQL in `python-backend/supabase_document_metrics_migration.sql`.

To scope uploaded documents and retrieval to an individual chat session, run the SQL in `python-backend/supabase_chat_scoped_kb_migration.sql`.

### Groq profile

- `GROQ_API_KEY`
- `GROQ_API_KEYS`
- `GROQ_MODEL_NAME=groq/compound-mini`

With `AI_PROVIDER=groq`, the backend uses:

- Groq for chat generation
- local HuggingFace embeddings for retrieval

### Gemini profile

- `AI_PROVIDER=gemini`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `GOOGLE_API_KEYS` or `GEMINI_API_KEYS`
- `GEMINI_MODEL_NAME`
- `GEMINI_EMBEDDING_MODEL_NAME`

With `AI_PROVIDER=gemini`, the backend uses Gemini for both chat and embeddings.

## Per-Request Provider Selection

The backend default provider still comes from `AI_PROVIDER`, but clients can override it per request.

Example chat payload:

```json
{
  "question": "Summarize the policy",
  "history": [],
  "top_k": 4,
  "provider": "gemini"
}
```

Valid values:

- `groq`
- `gemini`

You can also inspect a specific provider profile through health:

```text
GET /health?provider=groq
GET /health?provider=gemini
```

## Multi-Key Failover

You can provide multiple API keys as a comma-separated list:

- `GROQ_API_KEYS=key1,key2,key3`
- `GOOGLE_API_KEYS=key1,key2,key3`

When a request fails with one key, the backend automatically retries with the next configured key.

## Important Note About Chroma

Switching from HuggingFace embeddings to Gemini embeddings changes the vector space. That means an existing Chroma collection created for the Groq profile cannot be reused safely for the Gemini profile.

If you switch to `AI_PROVIDER=gemini`, re-index the documents into a Gemini-backed collection. By default the app uses:

- `langchain` for the Groq profile
- `rag-gemini` for the Gemini profile

You can override this with `CHROMA_COLLECTION_NAME`.
