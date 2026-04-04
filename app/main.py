from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from .auth import CurrentUser, get_current_user
from .config import (
    API_TITLE,
    ALLOWED_UPLOAD_EXTENSIONS,
    DEFAULT_TOP_K,
    MAX_UPLOAD_SIZE_MB,
    get_allowed_origins,
    get_provider_summary_for,
    normalize_ai_provider,
)
from .models import (
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionRename,
    DocumentChunkList,
    DocumentPipeline,
    HealthResponse,
)
from .services.chat_service import get_chat_persistence_service
from .services.ingestion_service import get_ingestion_service
from .services.rag_service import ask_async, get_rag_service
from .services.supabase_service import get_supabase_service
from .security import build_rate_limit_key, rate_limiter

app = FastAPI(title=API_TITLE, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enforce_rate_limit(request: Request, user: CurrentUser, bucket: str) -> None:
    rate_limiter.check(build_rate_limit_key(request, user.id, bucket))


def _validate_upload(file: UploadFile) -> None:
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {allowed}.",
        )

    content_length = file.size or 0
    if content_length and content_length > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit.",
        )


@app.get("/")
def root():
    return {
        "name": API_TITLE,
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health(provider: str | None = Query(default=None)):
    try:
        normalized_provider = normalize_ai_provider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    provider_summary = get_provider_summary_for(normalized_provider)
    try:
        service = get_rag_service(normalized_provider)
    except Exception as exc:
        return HealthResponse(
            ok=False,
            detail=str(exc),
            db_path=None,
            top_k=DEFAULT_TOP_K,
            model=provider_summary["chat_model"],
            provider=provider_summary["provider"],
            embedding_provider=provider_summary["embedding_provider"],
            embedding_model=provider_summary["embedding_model"],
            collection_name=provider_summary["collection_name"],
        )

    return HealthResponse(
        ok=True,
        detail="ready",
        db_path=str(service.db_path),
        top_k=DEFAULT_TOP_K,
        model=service.chat_model_name,
        provider=service.provider,
        embedding_provider=service.embedding_provider,
        embedding_model=service.embedding_model_name,
        collection_name=service.collection_name,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        _enforce_rate_limit(request, current_user, "chat")
        provider = normalize_ai_provider(payload.provider)
        chat_store = get_chat_persistence_service()
        session_id, history = chat_store.get_history_for_session(
            payload.session_id,
            current_user,
            [item.model_dump() for item in payload.history],
        )
        result = await ask_async(
            question=payload.question,
            history=history,
            top_k=payload.top_k,
            provider=provider,
            user_id=current_user.id,
        )
        persisted_session_id = chat_store.persist_chat_round(
            session_id=session_id,
            current_user=current_user,
            provider=provider,
            question=payload.question.strip(),
            answer=result.answer,
            rewritten_query=result.rewritten_query,
            model_name=get_rag_service(provider).chat_model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EnvironmentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected chat error.") from exc

    return ChatResponse(
        session_id=persisted_session_id,
        answer=result.answer,
        rewritten_query=result.rewritten_query,
        sources=result.sources,
    )


@app.get("/api/chats", response_model=list[ChatSession])
def list_chats(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    _enforce_rate_limit(request, current_user, "chats-list")
    return get_chat_persistence_service().list_chats(current_user)


@app.delete("/api/chats/{session_id}", status_code=204)
def delete_chat(
    session_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    _enforce_rate_limit(request, current_user, "chats-delete")
    get_chat_persistence_service().delete_chat(session_id, current_user)


@app.patch("/api/chats/{session_id}", status_code=204)
def rename_chat(
    session_id: str,
    payload: ChatSessionRename,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    _enforce_rate_limit(request, current_user, "chats-rename")
    get_chat_persistence_service().rename_chat(session_id, current_user, payload.title)


@app.get("/api/documents", response_model=list[DocumentPipeline])
def list_documents(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    _enforce_rate_limit(request, current_user, "documents-list")
    get_supabase_service().ensure_user(current_user.id, current_user.email)
    return get_ingestion_service().list_documents_for_user(current_user.id)


@app.get("/api/documents/{document_id}", response_model=DocumentPipeline)
def get_document(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        _enforce_rate_limit(request, current_user, "documents-detail")
        return get_ingestion_service().get_document(document_id, current_user.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.get("/api/documents/{document_id}/chunks", response_model=DocumentChunkList)
def get_document_chunks(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        _enforce_rate_limit(request, current_user, "documents-chunks")
        return get_ingestion_service().get_chunks(document_id, current_user.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        _enforce_rate_limit(request, current_user, "documents-delete")
        get_ingestion_service().delete_document(document_id, current_user.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.post("/api/documents/upload", response_model=DocumentPipeline)
async def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    provider: str | None = Form(default=None),
    current_user: CurrentUser = Depends(get_current_user),
):
    try:
        _enforce_rate_limit(request, current_user, "documents-upload")
        _validate_upload(file)
        normalized_provider = normalize_ai_provider(provider)
        get_supabase_service().ensure_user(current_user.id, current_user.email)
        document = get_ingestion_service().create_document(
            file,
            normalized_provider,
            current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Upload failed.") from exc
    finally:
        await file.close()

    background_tasks.add_task(
        get_ingestion_service().process_document,
        document.id,
    )
    return document
