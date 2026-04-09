import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from ..config import (
    SUPABASE_ANON_KEY,
    SUPABASE_SECRET_KEY,
    SUPABASE_STORAGE_BUCKET,
    SUPABASE_URL,
    SUPABASE_VECTOR_RPC_NAME,
)
from ..models import ChatSession, ChatSessionMessage, DocumentChunk, DocumentChunkList, DocumentPipeline, PartitionCounts, ProcessingStage


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_uuid() -> str:
    return str(uuid4())


def _json_headers(*extra_preferences: str) -> dict[str, str]:
    api_key = SUPABASE_SECRET_KEY or SUPABASE_ANON_KEY
    if not SUPABASE_URL or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase persistence is not configured.",
        )

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_preferences:
        headers["Prefer"] = ",".join(extra_preferences)
    return headers


def _storage_headers(content_type: str) -> dict[str, str]:
    headers = _json_headers()
    headers["Content-Type"] = content_type
    headers["x-upsert"] = "false"
    headers.pop("Prefer", None)
    return headers


def _encode_query(params: dict[str, str | None]) -> str:
    filtered = {key: value for key, value in params.items() if value is not None}
    if not filtered:
        return ""
    return f"?{urlencode(filtered)}"


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any | None = None,
    expected_statuses: tuple[int, ...] = (200, 201, 204),
) -> Any:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            if response.status not in expected_statuses:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unexpected Supabase response: {response.status}",
                )
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Supabase request failed: {detail or exc.reason}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach Supabase.",
        ) from exc

    if not payload:
        return None

    try:
        return json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return payload.decode("utf-8", errors="ignore")


def _request_bytes(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    expected_statuses: tuple[int, ...] = (200, 201, 204),
) -> bytes:
    request = Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            if response.status not in expected_statuses:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unexpected Supabase response: {response.status}",
                )
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Supabase storage request failed: {detail or exc.reason}",
        ) from exc
    except URLError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to reach Supabase storage.",
        ) from exc


@dataclass
class SupabaseService:
    base_url: str

    @property
    def rest_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/rest/v1"

    @property
    def storage_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/storage/v1/object"

    @property
    def bucket(self) -> str:
        return SUPABASE_STORAGE_BUCKET

    def ensure_user(self, user_id: str, email: str | None, auth_provider: str = "email") -> None:
        existing = _request_json(
            "GET",
            f"{self.rest_url}/users{_encode_query({'select': 'id,email', 'id': f'eq.{user_id}', 'limit': '1'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []
        if existing:
            return

        now = utc_now()
        record = {
            "id": user_id,
            "email": email,
            "auth_provider": auth_provider,
            "is_active": True,
            "updated_at": now,
            "created_at": now,
        }
        _request_json(
            "POST",
            f"{self.rest_url}/users{_encode_query({'on_conflict': 'id'})}",
            headers=_json_headers("resolution=merge-duplicates", "return=minimal"),
            body=[record],
        )

    def upload_file(self, storage_key: str, content: bytes, content_type: str) -> None:
        url = f"{self.storage_url}/{quote(self.bucket, safe='')}/{quote(storage_key, safe='/')}"
        _request_bytes(
            "POST",
            url,
            headers=_storage_headers(content_type),
            body=content,
        )

    def download_file(self, storage_key: str) -> bytes:
        url = f"{self.storage_url}/{quote(self.bucket, safe='')}/{quote(storage_key, safe='/')}"
        headers = _json_headers()
        headers.pop("Content-Type", None)
        return _request_bytes("GET", url, headers=headers, expected_statuses=(200,))

    def delete_file(self, storage_key: str) -> None:
        url = f"{self.storage_url}/{quote(self.bucket, safe='')}/{quote(storage_key, safe='/')}"
        headers = _json_headers()
        headers.pop("Content-Type", None)
        _request_bytes("DELETE", url, headers=headers, expected_statuses=(200, 204))

    def insert_chat_session(self, user_id: str, title: str, provider: str) -> dict[str, Any]:
        now = utc_now()
        record = {
            "id": new_uuid(),
            "user_id": user_id,
            "title": title,
            "provider": provider,
            "last_message_at": now,
            "created_at": now,
            "updated_at": now,
        }
        result = _request_json(
            "POST",
            f"{self.rest_url}/chat_sessions",
            headers=_json_headers("return=representation"),
            body=[record],
        )
        return result[0]

    def update_chat_session(self, session_id: str, user_id: str, fields: dict[str, Any]) -> None:
        fields = {**fields, "updated_at": utc_now()}
        _request_json(
            "PATCH",
            f"{self.rest_url}/chat_sessions{_encode_query({'id': f'eq.{session_id}', 'user_id': f'eq.{user_id}'})}",
            headers=_json_headers("return=minimal"),
            body=fields,
        )

    def insert_chat_messages(self, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        _request_json(
            "POST",
            f"{self.rest_url}/chat_messages",
            headers=_json_headers("return=minimal"),
            body=messages,
        )

    def list_chat_sessions(self, user_id: str) -> list[ChatSession]:
        sessions = _request_json(
            "GET",
            f"{self.rest_url}/chat_sessions{_encode_query({'select': '*', 'user_id': f'eq.{user_id}', 'order': 'updated_at.desc'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

        if not sessions:
            return []

        session_ids = ",".join(session["id"] for session in sessions)
        messages = _request_json(
            "GET",
            f"{self.rest_url}/chat_messages{_encode_query({'select': '*', 'user_id': f'eq.{user_id}', 'session_id': f'in.({session_ids})', 'order': 'message_order.asc'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

        grouped_messages: dict[str, list[ChatSessionMessage]] = defaultdict(list)
        for item in messages:
            grouped_messages[item["session_id"]].append(ChatSessionMessage(**item))

        return [
            ChatSession(
                **session,
                messages=grouped_messages.get(session["id"], []),
            )
            for session in sessions
        ]

    def get_chat_session(self, session_id: str, user_id: str) -> ChatSession | None:
        sessions = self.list_chat_sessions(user_id)
        for session in sessions:
            if session.id == session_id:
                return session
        return None

    def delete_chat_session(self, session_id: str, user_id: str) -> None:
        _request_json(
            "DELETE",
            f"{self.rest_url}/chat_messages{_encode_query({'session_id': f'eq.{session_id}', 'user_id': f'eq.{user_id}'})}",
            headers=_json_headers("return=minimal"),
        )
        _request_json(
            "DELETE",
            f"{self.rest_url}/chat_sessions{_encode_query({'id': f'eq.{session_id}', 'user_id': f'eq.{user_id}'})}",
            headers=_json_headers("return=minimal"),
        )

    def insert_document(self, record: dict[str, Any]) -> dict[str, Any]:
        result = _request_json(
            "POST",
            f"{self.rest_url}/documents",
            headers=_json_headers("return=representation"),
            body=[record],
        )
        return result[0]

    def update_document(self, document_id: str, user_id: str, fields: dict[str, Any]) -> None:
        fields = {**fields, "updated_at": utc_now()}
        _request_json(
            "PATCH",
            f"{self.rest_url}/documents{_encode_query({'id': f'eq.{document_id}', 'user_id': f'eq.{user_id}'})}",
            headers=_json_headers("return=minimal"),
            body=fields,
        )

    def list_documents(self, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
        params = {
            "select": "*",
            "user_id": f"eq.{user_id}",
            "order": "created_at.desc",
        }
        if session_id:
            params["chat_session_id"] = f"eq.{session_id}"
        return _request_json(
            "GET",
            f"{self.rest_url}/documents{_encode_query(params)}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

    def get_document(self, document_id: str, user_id: str) -> dict[str, Any] | None:
        result = _request_json(
            "GET",
            f"{self.rest_url}/documents{_encode_query({'select': '*', 'id': f'eq.{document_id}', 'user_id': f'eq.{user_id}', 'limit': '1'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []
        return result[0] if result else None

    def list_documents_for_processing(self, document_id: str) -> list[dict[str, Any]]:
        return _request_json(
            "GET",
            f"{self.rest_url}/documents{_encode_query({'select': '*', 'id': f'eq.{document_id}', 'limit': '1'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

    def delete_document(self, document_id: str, user_id: str) -> None:
        _request_json(
            "DELETE",
            f"{self.rest_url}/document_processing_events{_encode_query({'document_id': f'eq.{document_id}'})}",
            headers=_json_headers("return=minimal"),
        )
        _request_json(
            "DELETE",
            f"{self.rest_url}/document_chunks{_encode_query({'document_id': f'eq.{document_id}'})}",
            headers=_json_headers("return=minimal"),
        )
        _request_json(
            "DELETE",
            f"{self.rest_url}/documents{_encode_query({'id': f'eq.{document_id}', 'user_id': f'eq.{user_id}'})}",
            headers=_json_headers("return=minimal"),
        )

    def insert_document_event(
        self,
        *,
        document_id: str,
        stage_key: str,
        stage_label: str,
        status: str,
        detail: str | None = None,
        progress_current: int = 0,
        progress_total: int = 0,
    ) -> None:
        _request_json(
            "POST",
            f"{self.rest_url}/document_processing_events",
            headers=_json_headers("return=minimal"),
            body=[
                {
                    "id": new_uuid(),
                    "document_id": document_id,
                    "stage_key": stage_key,
                    "stage_label": stage_label,
                    "status": status,
                    "detail": detail,
                    "progress_current": progress_current,
                    "progress_total": progress_total,
                    "created_at": utc_now(),
                }
            ],
        )

    def list_document_events(self, document_ids: list[str]) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        joined_ids = ",".join(document_ids)
        return _request_json(
            "GET",
            f"{self.rest_url}/document_processing_events{_encode_query({'select': '*', 'document_id': f'in.({joined_ids})', 'order': 'created_at.asc'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

    def replace_document_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        _request_json(
            "DELETE",
            f"{self.rest_url}/document_chunks{_encode_query({'document_id': f'eq.{document_id}'})}",
            headers=_json_headers("return=minimal"),
        )
        if not chunks:
            return
        _request_json(
            "POST",
            f"{self.rest_url}/document_chunks",
            headers=_json_headers("return=minimal"),
            body=chunks,
        )

    def list_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        return _request_json(
            "GET",
            f"{self.rest_url}/document_chunks{_encode_query({'select': '*', 'document_id': f'eq.{document_id}', 'order': 'chunk_index.asc'})}",
            headers=_json_headers(),
            expected_statuses=(200,),
        ) or []

    def match_document_chunks(
        self,
        *,
        query_embedding: list[float],
        user_id: str,
        provider: str,
        match_count: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        body = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_user_id": user_id,
            "filter_provider": provider,
        }
        if session_id:
            body["filter_session_id"] = session_id
        return _request_json(
            "POST",
            f"{self.rest_url}/rpc/{SUPABASE_VECTOR_RPC_NAME}",
            headers=_json_headers(),
            body=body,
            expected_statuses=(200,),
        ) or []

    def build_document_pipeline(self, document: dict[str, Any], events: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> DocumentPipeline:
        stages_by_key: dict[str, ProcessingStage] = {
            "upload": ProcessingStage(key="upload", label="Upload to S3", status="pending"),
            "queued": ProcessingStage(key="queued", label="Queued", status="pending"),
            "partitioning": ProcessingStage(key="partitioning", label="Partitioning", status="pending"),
            "chunking": ProcessingStage(key="chunking", label="Chunking", status="pending"),
            "summarisation": ProcessingStage(key="summarisation", label="Summarisation", status="pending"),
            "vectorization": ProcessingStage(key="vectorization", label="Vectorization & Storage", status="pending"),
        }
        detail_log: list[str] = []
        for event in events:
            stages_by_key[event["stage_key"]] = ProcessingStage(
                key=event["stage_key"],
                label=event["stage_label"],
                status=event["status"],
                detail=event.get("detail"),
                progress_current=event.get("progress_current") or 0,
                progress_total=event.get("progress_total") or 0,
            )
            detail = event.get("detail") or event["status"]
            detail_log.append(f"[{event['created_at']}] {event['stage_label']}: {detail}")

        chunk_models = [DocumentChunk(**_map_chunk_row(chunk)) for chunk in chunks]
        chunk_count = len(chunk_models)
        summary_count = sum(1 for chunk in chunk_models if chunk.summary)
        vectorized_count = chunk_count if stages_by_key["vectorization"].status == "completed" else 0

        return DocumentPipeline(
            id=document["id"],
            user_id=document["user_id"],
            chat_session_id=document.get("chat_session_id"),
            filename=document["filename"],
            provider=document["provider"],
            status=document["status"],
            current_stage=document["current_stage"],
            file_size=document["file_size"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            error=document.get("error_message"),
            stages=list(stages_by_key.values()),
            partition_counts=PartitionCounts(
                text_sections=document.get("text_sections") or 0,
                tables=document.get("tables") or 0,
                images=document.get("images") or 0,
                titles_headers=document.get("titles_headers") or 0,
                other_elements=document.get("other_elements") or 0,
            ),
            atomic_elements=document.get("atomic_elements") or 0,
            chunk_count=chunk_count,
            summary_count=summary_count,
            vectorized_count=vectorized_count,
            detail_log=detail_log,
        )

    def build_document_chunk_list(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> DocumentChunkList:
        return DocumentChunkList(
            document_id=document["id"],
            filename=document["filename"],
            chunks=[DocumentChunk(**_map_chunk_row(chunk)) for chunk in chunks],
        )


def _map_chunk_row(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "chunk_index": chunk["chunk_index"],
        "kind": chunk["kind"],
        "page": chunk.get("page_number"),
        "char_count": chunk["char_count"],
        "content": chunk["content"],
        "summary": chunk.get("summary"),
    }


_SERVICE = SupabaseService(base_url=SUPABASE_URL)


def get_supabase_service() -> SupabaseService:
    return _SERVICE
