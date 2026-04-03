import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import BACKEND_ROOT, get_chroma_collection_name, get_primary_db_path
from ..models import DocumentChunk, DocumentChunkList, DocumentPipeline, PartitionCounts
from .rag_service import build_embeddings_for_provider, get_rag_service


STAGE_DEFINITIONS = [
    ("upload", "Upload to S3"),
    ("queued", "Queued"),
    ("partitioning", "Partitioning"),
    ("chunking", "Chunking"),
    ("summarisation", "Summarisation"),
    ("vectorization", "Vectorization & Storage"),
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stage_dict(key: str, label: str, status: str = "pending") -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "detail": None,
        "progress_current": 0,
        "progress_total": 0,
    }


def _slugify_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-")
    return cleaned or "document"


def _extract_paragraphs(text: str, page: int | None = None) -> list[dict[str, Any]]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]

    elements: list[dict[str, Any]] = []
    for block in blocks:
        kind = "header" if len(block) <= 80 and "\n" not in block else "text"
        elements.append({"kind": kind, "content": block, "page": page})
    return elements


def _summarize_text(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 180:
        return normalized

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    summary = " ".join(sentences[:2]).strip()
    if summary and len(summary) >= 80:
        return summary[:220].rstrip()

    return normalized[:220].rstrip()


class IngestionService:
    def __init__(self) -> None:
        self._lock = RLock()
        self._documents: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, list[dict[str, Any]]] = {}
        self.upload_root = BACKEND_ROOT / "data" / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def create_document(
        self, upload: UploadFile, provider: str, user_id: str
    ) -> dict[str, Any]:
        document_id = uuid4().hex
        safe_name = _slugify_filename(upload.filename or "document")
        target_path = self.upload_root / f"{document_id}-{safe_name}"
        file_bytes = upload.file.read()
        target_path.write_bytes(file_bytes)

        created_at = _utc_now()
        document = {
            "id": document_id,
            "user_id": user_id,
            "filename": upload.filename or safe_name,
            "stored_path": str(target_path),
            "provider": provider,
            "status": "queued",
            "current_stage": "queued",
            "file_size": len(file_bytes),
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "stages": [
                _stage_dict("upload", "Upload to S3", "completed"),
                _stage_dict("queued", "Queued", "running"),
                _stage_dict("partitioning", "Partitioning"),
                _stage_dict("chunking", "Chunking"),
                _stage_dict("summarisation", "Summarisation"),
                _stage_dict("vectorization", "Vectorization & Storage"),
            ],
            "partition_counts": {
                "text_sections": 0,
                "tables": 0,
                "images": 0,
                "titles_headers": 0,
                "other_elements": 0,
            },
            "atomic_elements": 0,
            "chunk_count": 0,
            "summary_count": 0,
            "vectorized_count": 0,
            "detail_log": [
                f"[{created_at}] Uploaded {upload.filename or safe_name}",
                f"[{created_at}] Stored at {target_path}",
            ],
        }

        with self._lock:
            self._documents[document_id] = document
            self._chunks[document_id] = []

        return self._serialize_document(document)

    def list_documents(self) -> list[DocumentPipeline]:
        return self.list_documents_for_user(None)

    def list_documents_for_user(self, user_id: str | None) -> list[DocumentPipeline]:
        with self._lock:
            documents = sorted(
                (
                    item
                    for item in self._documents.values()
                    if user_id is None or item["user_id"] == user_id
                ),
                key=lambda item: item["created_at"],
                reverse=True,
            )
            return [self._serialize_document(item) for item in documents]

    def get_document(self, document_id: str, user_id: str | None = None) -> DocumentPipeline:
        with self._lock:
            document = self._documents.get(document_id)
            if not document or (user_id is not None and document["user_id"] != user_id):
                raise KeyError(document_id)
            return self._serialize_document(document)

    def get_chunks(
        self, document_id: str, user_id: str | None = None
    ) -> DocumentChunkList:
        with self._lock:
            document = self._documents.get(document_id)
            chunks = self._chunks.get(document_id)
            if (
                not document
                or chunks is None
                or (user_id is not None and document["user_id"] != user_id)
            ):
                raise KeyError(document_id)

            return DocumentChunkList(
                document_id=document_id,
                filename=document["filename"],
                chunks=[DocumentChunk(**chunk) for chunk in chunks],
            )

    def delete_document(self, document_id: str, user_id: str | None = None) -> None:
        with self._lock:
            document = self._documents.get(document_id)
            chunks = list(self._chunks.get(document_id, []))
            if not document or (user_id is not None and document["user_id"] != user_id):
                raise KeyError(document_id)

            provider = document["provider"]
            stored_path = Path(document["stored_path"])
            chunk_ids = [f"{document_id}:{chunk['chunk_index']}" for chunk in chunks]

            self._documents.pop(document_id, None)
            self._chunks.pop(document_id, None)

        if chunk_ids:
            embeddings, _, _ = build_embeddings_for_provider(provider)
            vector_store = Chroma(
                persist_directory=str(get_primary_db_path()),
                embedding_function=embeddings,
                collection_name=get_chroma_collection_name(provider),
                collection_metadata={"hnsw:space": "cosine"},
            )
            vector_store.delete(ids=chunk_ids)

        if stored_path.exists():
            stored_path.unlink()

        get_rag_service.cache_clear()

    def process_document(self, document_id: str) -> None:
        try:
            self._mark_stage(document_id, "queued", "completed", "Waiting slot cleared")
            self._mark_stage(document_id, "partitioning", "running", "Processing and extracting text, images, and tables")
            elements, counts = self._partition_document(document_id)
            self._finalize_partitioning(document_id, counts, len(elements))

            self._mark_stage(document_id, "chunking", "running", "Creating semantic chunks")
            chunks = self._chunk_elements(document_id, elements)
            self._finalize_chunking(document_id, len(elements), len(chunks))

            self._mark_stage(document_id, "summarisation", "running", "Creating chunk summaries")
            self._summarize_chunks(document_id)

            self._mark_stage(document_id, "vectorization", "running", "Embedding and storing chunks")
            self._vectorize_document(document_id)

            with self._lock:
                document = self._documents[document_id]
                document["status"] = "ready"
                document["current_stage"] = "view_chunks"
                document["updated_at"] = _utc_now()
                document["detail_log"].append(
                    f"[{document['updated_at']}] Pipeline completed successfully"
                )
        except Exception as exc:
            with self._lock:
                document = self._documents[document_id]
                document["status"] = "failed"
                document["error"] = str(exc)
                document["updated_at"] = _utc_now()
                document["detail_log"].append(
                    f"[{document['updated_at']}] Pipeline failed: {exc}"
                )
                current_key = document["current_stage"]
                self._set_stage_locked(
                    document,
                    current_key,
                    status="failed",
                    detail=str(exc),
                )

    def _partition_document(
        self, document_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        with self._lock:
            document = self._documents[document_id]
            path = Path(document["stored_path"])

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            elements, counts = self._partition_pdf(path)
        elif suffix == ".csv":
            elements, counts = self._partition_csv(path)
        elif suffix == ".docx":
            elements, counts = self._partition_docx(path)
        else:
            elements, counts = self._partition_text(path)

        return elements, counts

    def _partition_pdf(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        elements: list[dict[str, Any]] = []
        counts = PartitionCounts()
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            page_elements = _extract_paragraphs(text, page=page_index)
            elements.extend(page_elements)
            counts.text_sections += sum(
                1 for item in page_elements if item["kind"] == "text"
            )
            counts.titles_headers += sum(
                1 for item in page_elements if item["kind"] == "header"
            )

        return elements, counts.model_dump()

    def _partition_text(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        elements = _extract_paragraphs(text)
        counts = PartitionCounts(
            text_sections=sum(1 for item in elements if item["kind"] == "text"),
            titles_headers=sum(1 for item in elements if item["kind"] == "header"),
        )
        return elements, counts.model_dump()

    def _partition_csv(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append(" | ".join(cell.strip() for cell in row))

        content = "\n".join(rows)
        elements = [{"kind": "table", "content": content, "page": None}] if content else []
        counts = PartitionCounts(tables=1 if elements else 0)
        return elements, counts.model_dump()

    def _partition_docx(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        from docx import Document as DocxDocument

        document = DocxDocument(str(path))
        text = "\n\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
        elements = _extract_paragraphs(text)
        counts = PartitionCounts(
            text_sections=sum(1 for item in elements if item["kind"] == "text"),
            titles_headers=sum(1 for item in elements if item["kind"] == "header"),
        )
        return elements, counts.model_dump()

    def _finalize_partitioning(
        self, document_id: str, counts: dict[str, int], atomic_elements: int
    ) -> None:
        with self._lock:
            document = self._documents[document_id]
            document["partition_counts"] = counts
            document["atomic_elements"] = atomic_elements
            document["updated_at"] = _utc_now()
            document["detail_log"].append(
                f"[{document['updated_at']}] Partitioned into {atomic_elements} atomic elements"
            )
            self._set_stage_locked(
                document,
                "partitioning",
                status="completed",
                detail="Step completed successfully",
                progress_current=atomic_elements,
                progress_total=atomic_elements,
            )

    def _chunk_elements(
        self, document_id: str, elements: list[dict[str, Any]], target_size: int = 1400
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        current_parts: list[dict[str, Any]] = []
        current_length = 0

        for element in elements:
            content = element["content"].strip()
            if not content:
                continue

            projected = current_length + len(content) + 2
            if current_parts and projected > target_size:
                chunks.append(self._build_chunk(chunks, current_parts))
                current_parts = []
                current_length = 0

            current_parts.append(element)
            current_length += len(content) + 2

        if current_parts:
            chunks.append(self._build_chunk(chunks, current_parts))

        with self._lock:
            self._chunks[document_id] = chunks

        return chunks

    def _build_chunk(
        self, existing_chunks: list[dict[str, Any]], elements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        content = "\n\n".join(item["content"] for item in elements)
        first_page = next((item["page"] for item in elements if item["page"]), None)
        kind = "table" if any(item["kind"] == "table" for item in elements) else "text"
        chunk_index = len(existing_chunks)
        return {
            "id": f"chunk-{chunk_index + 1}",
            "chunk_index": chunk_index,
            "kind": kind,
            "page": first_page,
            "char_count": len(content),
            "content": content,
            "summary": None,
        }

    def _finalize_chunking(
        self, document_id: str, atomic_elements: int, chunk_count: int
    ) -> None:
        with self._lock:
            document = self._documents[document_id]
            document["chunk_count"] = chunk_count
            document["updated_at"] = _utc_now()
            document["detail_log"].append(
                f"[{document['updated_at']}] Chunked {atomic_elements} elements into {chunk_count} chunks"
            )
            self._set_stage_locked(
                document,
                "chunking",
                status="completed",
                detail="Step completed successfully",
                progress_current=chunk_count,
                progress_total=chunk_count,
            )

    def _summarize_chunks(self, document_id: str) -> None:
        with self._lock:
            chunks = self._chunks[document_id]
            document = self._documents[document_id]
            total = len(chunks)
            self._set_stage_locked(
                document,
                "summarisation",
                progress_current=0,
                progress_total=total,
            )

        for index, chunk in enumerate(chunks, start=1):
            chunk["summary"] = _summarize_text(chunk["content"])
            with self._lock:
                document = self._documents[document_id]
                self._set_stage_locked(
                    document,
                    "summarisation",
                    progress_current=index,
                    progress_total=total,
                    detail="Processing chunks and creating concise summaries",
                )

        with self._lock:
            document = self._documents[document_id]
            document["summary_count"] = len(chunks)
            document["updated_at"] = _utc_now()
            document["detail_log"].append(
                f"[{document['updated_at']}] Summarised {len(chunks)} chunks"
            )
            self._set_stage_locked(
                document,
                "summarisation",
                status="completed",
                detail="Step completed successfully",
                progress_current=len(chunks),
                progress_total=len(chunks),
            )

    def _vectorize_document(self, document_id: str) -> None:
        with self._lock:
            document = self._documents[document_id]
            provider = document["provider"]
            filename = document["filename"]
            chunks = list(self._chunks[document_id])
            total = len(chunks)
            self._set_stage_locked(
                document,
                "vectorization",
                progress_current=0,
                progress_total=total,
            )

        embeddings, _, _ = build_embeddings_for_provider(provider)
        db_path = get_primary_db_path()
        collection_name = get_chroma_collection_name(provider)
        vector_store = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        documents: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            raw_payload = json.dumps(
                {
                    "raw_text": chunk["content"],
                    "summary": chunk["summary"],
                }
            )
            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": filename,
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "kind": chunk["kind"],
                        "original_content": raw_payload,
                    },
                )
            )
            ids.append(f"{document_id}:{chunk['chunk_index']}")

        if documents:
            vector_store.add_documents(documents=documents, ids=ids)

        get_rag_service.cache_clear()

        with self._lock:
            document = self._documents[document_id]
            document["vectorized_count"] = len(documents)
            document["updated_at"] = _utc_now()
            document["detail_log"].append(
                f"[{document['updated_at']}] Stored {len(documents)} chunks in collection '{collection_name}'"
            )
            self._set_stage_locked(
                document,
                "vectorization",
                status="completed",
                detail="Step completed successfully",
                progress_current=len(documents),
                progress_total=len(documents),
            )

    def _mark_stage(
        self, document_id: str, key: str, status: str, detail: str | None = None
    ) -> None:
        with self._lock:
            document = self._documents[document_id]
            document["current_stage"] = key
            document["updated_at"] = _utc_now()
            self._set_stage_locked(document, key, status=status, detail=detail)

    def _set_stage_locked(
        self,
        document: dict[str, Any],
        key: str,
        status: str | None = None,
        detail: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
    ) -> None:
        for stage in document["stages"]:
            if stage["key"] != key:
                continue
            if status is not None:
                stage["status"] = status
            if detail is not None:
                stage["detail"] = detail
            if progress_current is not None:
                stage["progress_current"] = progress_current
            if progress_total is not None:
                stage["progress_total"] = progress_total
            break

    def _serialize_document(self, document: dict[str, Any]) -> DocumentPipeline:
        return DocumentPipeline(**document)


_SERVICE = IngestionService()


def get_ingestion_service() -> IngestionService:
    return _SERVICE
