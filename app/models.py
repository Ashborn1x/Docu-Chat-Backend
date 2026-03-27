from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=8)


class SourceItem(BaseModel):
    id: int
    source: str
    preview: str


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    ok: bool
    db_path: str
    top_k: int
    model: str
