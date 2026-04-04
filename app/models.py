from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=4, ge=1, le=8)
    provider: str | None = Field(default=None)
    session_id: str | None = Field(default=None)


class ChatSessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class SourceItem(BaseModel):
    id: int
    source: str
    preview: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    sources: list[SourceItem]


class ChatSessionMessage(BaseModel):
    id: str
    session_id: str
    user_id: str
    role: str
    content: str
    rewritten_query: str | None = None
    message_order: int
    model_name: str | None = None
    created_at: str


class ChatSession(BaseModel):
    id: str
    user_id: str
    title: str | None = None
    provider: str
    last_message_at: str
    created_at: str
    updated_at: str
    messages: list[ChatSessionMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    ok: bool
    detail: str | None = None
    db_path: str | None = None
    top_k: int
    model: str | None = None
    provider: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None


class ProcessingStage(BaseModel):
    key: str
    label: str
    status: str
    detail: str | None = None
    progress_current: int = 0
    progress_total: int = 0


class PartitionCounts(BaseModel):
    text_sections: int = 0
    tables: int = 0
    images: int = 0
    titles_headers: int = 0
    other_elements: int = 0


class DocumentPipeline(BaseModel):
    id: str
    user_id: str
    filename: str
    provider: str
    status: str
    current_stage: str
    file_size: int
    created_at: str
    updated_at: str
    error: str | None = None
    stages: list[ProcessingStage] = Field(default_factory=list)
    partition_counts: PartitionCounts = Field(default_factory=PartitionCounts)
    atomic_elements: int = 0
    chunk_count: int = 0
    summary_count: int = 0
    vectorized_count: int = 0
    detail_log: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    id: str
    chunk_index: int
    kind: str
    page: int | None = None
    char_count: int
    content: str
    summary: str | None = None


class DocumentChunkList(BaseModel):
    document_id: str
    filename: str
    chunks: list[DocumentChunk] = Field(default_factory=list)
