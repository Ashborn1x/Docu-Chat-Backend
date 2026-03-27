import asyncio
import json
import os
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
    GROQ_MODEL_NAME,
    get_db_candidates,
    resolve_embedding_model,
)


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


class RagService:
    def __init__(self) -> None:
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("Missing GROQ_API_KEY in environment.")

        embedding_model_name, local_only = resolve_embedding_model()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu", "local_files_only": local_only},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db_path = self._resolve_db_path()
        self.db = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0)

    def _resolve_db_path(self):
        for candidate in get_db_candidates():
            if not candidate.exists():
                continue

            db = Chroma(
                persist_directory=str(candidate),
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
            )
            if db._collection.count() > 0:
                return candidate

        searched = ", ".join(str(path) for path in get_db_candidates())
        raise FileNotFoundError(
            f"No populated Chroma database found. Checked: {searched}"
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
            result = self.llm.invoke(messages)
            rewritten = (result.content or "").strip()
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

    def ask(
        self, question: str, history: list[dict[str, str]] | None = None, top_k: int = DEFAULT_TOP_K
    ) -> ChatResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        history = history or []
        rewritten_query = self._rewrite_question(clean_question, history)
        docs = self.db.similarity_search(rewritten_query, k=max(1, min(top_k, 8)))
        sources = _build_sources(docs)

        prompt = self._answer_prompt(clean_question, docs)
        try:
            answer = self.llm.invoke([HumanMessage(content=prompt)]).content
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


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService()


async def ask_async(
    question: str, history: list[dict[str, str]] | None = None, top_k: int = DEFAULT_TOP_K
) -> ChatResult:
    return await asyncio.to_thread(get_rag_service().ask, question, history, top_k)
