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
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").strip().lower()
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").strip().lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
SUPABASE_JWT_ALGORITHM = os.getenv("SUPABASE_JWT_ALGORITHM", "HS256").strip()
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "groq/compound-mini")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL_NAME = os.getenv(
    "GEMINI_EMBEDDING_MODEL_NAME", "models/gemini-embedding-001"
)
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
API_TITLE = os.getenv("API_TITLE", "RAG Backend API")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
ALLOWED_UPLOAD_EXTENSIONS = {
    extension.strip().lower()
    for extension in os.getenv("ALLOWED_UPLOAD_EXTENSIONS", "pdf,csv,docx,txt").split(",")
    if extension.strip()
}


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def _split_env_values(*names: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    for name in names:
        raw_value = os.getenv(name, "")
        if not raw_value:
            continue

        normalized = raw_value.replace("\n", ",")
        for item in normalized.split(","):
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            values.append(candidate)

    return values


def normalize_ai_provider(provider: str | None = None) -> str:
    provider = (provider or AI_PROVIDER).strip().lower()
    if provider not in {"groq", "gemini"}:
        raise ValueError(
            "AI_PROVIDER must be either 'groq' or 'gemini'."
        )
    return provider


def get_ai_provider() -> str:
    return normalize_ai_provider()


def get_groq_api_keys() -> list[str]:
    return _split_env_values("GROQ_API_KEYS", "GROQ_API_KEY")


def get_gemini_api_keys() -> list[str]:
    return _split_env_values(
        "GOOGLE_API_KEYS",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
    )


def get_chroma_collection_name(provider: str | None = None) -> str:
    explicit = os.getenv("CHROMA_COLLECTION_NAME")
    if explicit:
        return explicit

    provider_name = provider or get_ai_provider()
    if provider_name == "gemini":
        return "rag-gemini"

    return "langchain"


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


def get_primary_db_path() -> Path:
    candidates = get_db_candidates()
    primary = candidates[0]
    primary.mkdir(parents=True, exist_ok=True)
    return primary


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


def get_provider_summary() -> dict[str, str]:
    provider = get_ai_provider()
    if provider == "gemini":
        return {
            "provider": "gemini",
            "chat_model": GEMINI_MODEL_NAME,
            "embedding_provider": "gemini",
            "embedding_model": GEMINI_EMBEDDING_MODEL_NAME,
            "collection_name": get_chroma_collection_name(provider),
        }

    embedding_model_name, _ = resolve_embedding_model()
    return {
        "provider": "groq",
        "chat_model": GROQ_MODEL_NAME,
        "embedding_provider": "huggingface",
        "embedding_model": embedding_model_name,
        "collection_name": get_chroma_collection_name(provider),
    }


def get_provider_summary_for(provider: str | None) -> dict[str, str]:
    resolved = normalize_ai_provider(provider)
    if resolved == "gemini":
        return {
            "provider": "gemini",
            "chat_model": GEMINI_MODEL_NAME,
            "embedding_provider": "gemini",
            "embedding_model": GEMINI_EMBEDDING_MODEL_NAME,
            "collection_name": get_chroma_collection_name(resolved),
        }

    embedding_model_name, _ = resolve_embedding_model()
    return {
        "provider": "groq",
        "chat_model": GROQ_MODEL_NAME,
        "embedding_provider": "huggingface",
        "embedding_model": embedding_model_name,
        "collection_name": get_chroma_collection_name(resolved),
    }
