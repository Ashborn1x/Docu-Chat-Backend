import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for env_file in (BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"):
    if env_file.exists():
        load_dotenv(env_file, override=False)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "groq/compound-mini")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
API_TITLE = os.getenv("API_TITLE", "RAG Backend API")


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def get_db_candidates() -> list[Path]:
    override = os.getenv("CHROMA_DB_PATH")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (BACKEND_ROOT / candidate).resolve()
        return [candidate]

    return [
        (PROJECT_ROOT / "dbv2" / "chroma_db").resolve(),
        (PROJECT_ROOT / "dbv1" / "chroma_db").resolve(),
        (PROJECT_ROOT / "db" / "chroma_db").resolve(),
    ]


def resolve_embedding_model() -> tuple[str, bool]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots_root = (
        cache_root / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    )
    if snapshots_root.exists():
        snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1]), True

    return EMBEDDING_MODEL_NAME, False
