from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    API_TITLE,
    DEFAULT_TOP_K,
    get_allowed_origins,
    get_provider_summary_for,
    normalize_ai_provider,
)
from .models import (
    ChatRequest,
    ChatResponse,
    DocumentChunkList,
    DocumentPipeline,
    HealthResponse,
)
from .services.ingestion_service import get_ingestion_service
from .services.rag_service import ask_async, get_rag_service

app = FastAPI(title=API_TITLE, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
async def chat(payload: ChatRequest):
    try:
        provider = normalize_ai_provider(payload.provider)
        result = await ask_async(
            question=payload.question,
            history=[item.model_dump() for item in payload.history],
            top_k=payload.top_k,
            provider=provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EnvironmentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected chat error.") from exc

    return ChatResponse(
        answer=result.answer,
        rewritten_query=result.rewritten_query,
        sources=result.sources,
    )


@app.get("/api/documents", response_model=list[DocumentPipeline])
def list_documents():
    return get_ingestion_service().list_documents()


@app.get("/api/documents/{document_id}", response_model=DocumentPipeline)
def get_document(document_id: str):
    try:
        return get_ingestion_service().get_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.get("/api/documents/{document_id}/chunks", response_model=DocumentChunkList)
def get_document_chunks(document_id: str):
    try:
        return get_ingestion_service().get_chunks(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    try:
        get_ingestion_service().delete_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc


@app.post("/api/documents/upload", response_model=DocumentPipeline)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str | None = Form(default=None),
):
    try:
        normalized_provider = normalize_ai_provider(provider)
        document = get_ingestion_service().create_document(file, normalized_provider)
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
