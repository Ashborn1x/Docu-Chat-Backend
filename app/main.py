from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_TITLE, DEFAULT_TOP_K, GROQ_MODEL_NAME, get_allowed_origins
from .models import ChatRequest, ChatResponse, HealthResponse
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
def health():
    service = get_rag_service()
    return HealthResponse(
        ok=True,
        db_path=str(service.db_path),
        top_k=DEFAULT_TOP_K,
        model=GROQ_MODEL_NAME,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    result = await ask_async(
        question=payload.question,
        history=[item.model_dump() for item in payload.history],
        top_k=payload.top_k,
    )
    return ChatResponse(
        answer=result.answer,
        rewritten_query=result.rewritten_query,
        sources=result.sources,
    )
