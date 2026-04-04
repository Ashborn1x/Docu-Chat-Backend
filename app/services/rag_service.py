import asyncio
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import (
    DEFAULT_TOP_K,
    GEMINI_EMBEDDING_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GROQ_MODEL_NAME,
    USE_SUPABASE_VECTORS,
    get_chroma_collection_name,
    get_db_candidates,
    get_gemini_api_keys,
    get_groq_api_keys,
    normalize_ai_provider,
    resolve_embedding_model,
)
from .supabase_service import get_supabase_service


@dataclass
class ChatResult:
    answer: str
    rewritten_query: str
    sources: list[dict[str, Any]]


def _extract_document_text(doc: Document) -> str:
    original_content = doc.metadata.get("original_content")
    if not original_content:
        return doc.page_content

    try:
        data = json.loads(original_content)
    except json.JSONDecodeError:
        return doc.page_content

    parts: list[str] = []
    raw_text = (data.get("raw_text") or "").strip()
    if raw_text:
        parts.append(raw_text)

    tables_html = data.get("tables_html") or []
    for index, table in enumerate(tables_html, start=1):
        table_text = str(table).strip()
        if table_text:
            parts.append(f"Table {index}:\n{table_text}")

    return "\n\n".join(parts).strip() or doc.page_content


def _build_sources(docs: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        content = _extract_document_text(doc)
        preview = " ".join(content.split())[:320]
        sources.append(
            {
                "id": index,
                "source": doc.metadata.get("source", "Indexed document"),
                "preview": preview,
            }
        )
    return sources


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


class RotatingChatModel:
    def __init__(self, models: list[Any], provider: str) -> None:
        self._models = models
        self.provider = provider

    def invoke(self, messages: list[Any]) -> str:
        errors: list[str] = []
        for index, model in enumerate(self._models, start=1):
            try:
                result = model.invoke(messages)
                text = _content_to_text(getattr(result, "content", result))
                if text:
                    return text
                raise RuntimeError("Model returned an empty response.")
            except Exception as exc:
                errors.append(f"key {index}: {exc}")

        joined = "; ".join(errors) if errors else "no configured API keys"
        raise RuntimeError(
            f"All {self.provider} API keys failed during generation: {joined}"
        )


class RotatingEmbeddings:
    def __init__(self, clients: list[Any], provider: str, model_name: str) -> None:
        self._clients = clients
        self.provider = provider
        self.model_name = model_name

    def _invoke(self, method_name: str, payload: Any) -> Any:
        errors: list[str] = []
        for index, client in enumerate(self._clients, start=1):
            try:
                return getattr(client, method_name)(payload)
            except Exception as exc:
                errors.append(f"key {index}: {exc}")

        joined = "; ".join(errors) if errors else "no configured API keys"
        raise RuntimeError(
            f"All {self.provider} API keys failed during embedding: {joined}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._invoke("embed_documents", texts)

    def embed_query(self, text: str) -> list[float]:
        return self._invoke("embed_query", text)


def build_embeddings_for_provider(provider: str | None = None) -> tuple[Any, str, str]:
    resolved_provider = normalize_ai_provider(provider)
    if resolved_provider == "gemini":
        api_keys = get_gemini_api_keys()
        if not api_keys:
            raise EnvironmentError(
                "Missing Gemini API keys. Set GOOGLE_API_KEY, GEMINI_API_KEY, "
                "GOOGLE_API_KEYS, or GEMINI_API_KEYS."
            )

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        clients = [
            GoogleGenerativeAIEmbeddings(
                model=GEMINI_EMBEDDING_MODEL_NAME,
                google_api_key=api_key,
            )
            for api_key in api_keys
        ]
        return (
            RotatingEmbeddings(
                clients=clients,
                provider="gemini",
                model_name=GEMINI_EMBEDDING_MODEL_NAME,
            ),
            GEMINI_EMBEDDING_MODEL_NAME,
            "gemini",
        )

    embedding_model_name, local_only = resolve_embedding_model()
    return (
        HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        ),
        embedding_model_name,
        "huggingface",
    )


class RagService:
    def __init__(self, provider: str | None = None) -> None:
        self.provider = normalize_ai_provider(provider)
        self.collection_name = get_chroma_collection_name(self.provider)
        (
            self.embedding_model,
            self.embedding_model_name,
            self.embedding_provider,
        ) = build_embeddings_for_provider(self.provider)
        self.db_path = None
        self.db = None
        if not USE_SUPABASE_VECTORS:
            self.db_path = self._resolve_db_path()
            self.db = self._open_db(self.db_path)
        self.llm, self.chat_model_name = self._build_chat_model()

    def _build_embeddings(self) -> tuple[Any, str]:
        embedding_model, embedding_model_name, _ = build_embeddings_for_provider(
            self.provider
        )
        return embedding_model, embedding_model_name

    def _build_chat_model(self) -> tuple[RotatingChatModel, str]:
        if self.provider == "gemini":
            api_keys = get_gemini_api_keys()
            if not api_keys:
                raise EnvironmentError(
                    "Missing Gemini API keys. Set GOOGLE_API_KEY, GEMINI_API_KEY, "
                    "GOOGLE_API_KEYS, or GEMINI_API_KEYS."
                )

            from langchain_google_genai import ChatGoogleGenerativeAI

            models = [
                ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_NAME,
                    temperature=0,
                    google_api_key=api_key,
                )
                for api_key in api_keys
            ]
            return RotatingChatModel(models=models, provider="gemini"), GEMINI_MODEL_NAME

        api_keys = get_groq_api_keys()
        if not api_keys:
            raise EnvironmentError(
                "Missing GROQ API keys. Set GROQ_API_KEY or GROQ_API_KEYS."
            )

        models = [
            ChatGroq(model=GROQ_MODEL_NAME, temperature=0, api_key=api_key)
            for api_key in api_keys
        ]
        return RotatingChatModel(models=models, provider="groq"), GROQ_MODEL_NAME

    def _open_db(self, candidate):
        return Chroma(
            persist_directory=str(candidate),
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _resolve_db_path(self):
        for candidate in get_db_candidates():
            if not candidate.exists():
                continue

            db = self._open_db(candidate)
            if db._collection.count() > 0:
                return candidate

        searched = ", ".join(str(path) for path in get_db_candidates())
        raise FileNotFoundError(
            "No populated Chroma database found for "
            f"collection '{self.collection_name}'. Checked: {searched}. "
            "If you switched embedding providers, re-index the documents for the new collection."
        )

    def _rewrite_question(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question

        messages: list[Any] = [
            SystemMessage(
                content=(
                    "Rewrite the latest user question into a standalone search query. "
                    "Return only the rewritten query."
                )
            )
        ]

        for item in history[-8:]:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=question))
        try:
            rewritten = self.llm.invoke(messages).strip()
            return rewritten or question
        except Exception:
            return question

    def _answer_prompt(self, question: str, docs: list[Document]) -> str:
        context_blocks: list[str] = []
        for index, doc in enumerate(docs, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"Document {index}",
                        f"Source: {doc.metadata.get('source', 'Indexed document')}",
                        _extract_document_text(doc),
                    ]
                )
            )

        context = "\n\n".join(context_blocks)
        return (
            "Answer the user's question using only the retrieved document context.\n"
            "If the answer is not supported by the context, say you do not have enough information.\n"
            "Be concise and cite sources in plain text like [1], [2] when relevant.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )

    def _retrieve_documents(
        self,
        query: str,
        *,
        top_k: int,
        user_id: str | None,
    ) -> list[Document]:
        if not USE_SUPABASE_VECTORS:
            return self.db.similarity_search(query, k=max(1, min(top_k, 8)))

        if not user_id:
            raise ValueError("User id is required when using Supabase vector retrieval.")

        query_embedding = self.embedding_model.embed_query(query)
        rows = get_supabase_service().match_document_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            provider=self.provider,
            match_count=max(1, min(top_k, 8)),
        )
        docs: list[Document] = []
        for row in rows:
            raw_payload = json.dumps(
                {
                    "raw_text": row.get("content", ""),
                    "summary": row.get("summary"),
                }
            )
            docs.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata={
                        "source": row.get("source", "Indexed document"),
                        "page": row.get("page_number"),
                        "chunk_index": row.get("chunk_index"),
                        "kind": row.get("kind"),
                        "original_content": raw_payload,
                    },
                )
            )
        return docs

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        user_id: str | None = None,
    ) -> ChatResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        history = history or []
        rewritten_query = self._rewrite_question(clean_question, history)
        docs = self._retrieve_documents(
            rewritten_query,
            top_k=top_k,
            user_id=user_id,
        )
        sources = _build_sources(docs)

        prompt = self._answer_prompt(clean_question, docs)
        try:
            answer = self.llm.invoke([HumanMessage(content=prompt)])
        except Exception:
            fallback_lines = [
                "Generation is currently unavailable, so here are the most relevant retrieved passages:",
            ]
            for source in sources:
                fallback_lines.append(
                    f"[{source['id']}] {source['source']}: {source['preview']}"
                )
            answer = "\n\n".join(fallback_lines)

        return ChatResult(
            answer=(answer or "").strip(),
            rewritten_query=rewritten_query,
            sources=sources,
        )


@lru_cache(maxsize=4)
def get_rag_service(provider: str | None = None) -> RagService:
    return RagService(provider=provider)


async def ask_async(
    question: str,
    history: list[dict[str, str]] | None = None,
    top_k: int = DEFAULT_TOP_K,
    provider: str | None = None,
    user_id: str | None = None,
) -> ChatResult:
    return await asyncio.to_thread(
        get_rag_service(provider).ask,
        question,
        history,
        top_k,
        user_id,
    )
