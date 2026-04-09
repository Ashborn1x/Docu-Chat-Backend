from typing import Any

from fastapi import HTTPException, status

from ..auth import CurrentUser
from ..models import ChatSession
from .supabase_service import get_supabase_service, new_uuid, utc_now


def _make_title(question: str) -> str:
    title = " ".join(question.split()).strip()
    return title[:80] or "New Chat"


class ChatPersistenceService:
    def __init__(self) -> None:
        self.supabase = get_supabase_service()

    def ensure_user(self, current_user: CurrentUser) -> None:
        self.supabase.ensure_user(current_user.id, current_user.email)

    def list_chats(self, current_user: CurrentUser) -> list[ChatSession]:
        self.ensure_user(current_user)
        return self.supabase.list_chat_sessions(current_user.id)

    def get_chat(self, session_id: str, current_user: CurrentUser) -> ChatSession:
        self.ensure_user(current_user)
        session = self.supabase.get_chat_session(session_id, current_user.id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
        return session

    def delete_chat(self, session_id: str, current_user: CurrentUser) -> None:
        self.ensure_user(current_user)
        self.supabase.delete_chat_session(session_id, current_user.id)

    def rename_chat(self, session_id: str, current_user: CurrentUser, title: str) -> None:
        self.ensure_user(current_user)
        self.supabase.update_chat_session(session_id, current_user.id, {"title": title})

    def create_chat(self, current_user: CurrentUser, provider: str, title: str = "New Chat") -> ChatSession:
        self.ensure_user(current_user)
        session_row = self.supabase.insert_chat_session(current_user.id, title, provider)
        return ChatSession(**session_row, messages=[])

    def persist_chat_round(
        self,
        *,
        session_id: str | None,
        current_user: CurrentUser,
        provider: str,
        question: str,
        answer: str,
        rewritten_query: str,
        model_name: str | None = None,
    ) -> str:
        self.ensure_user(current_user)
        session: ChatSession | None = None
        if session_id:
            session = self.supabase.get_chat_session(session_id, current_user.id)
            if not session:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
        if session is None:
            session_row = self.supabase.insert_chat_session(
                current_user.id,
                _make_title(question),
                provider,
            )
            session_id = session_row["id"]
            current_order = 0
        else:
            current_order = len(session.messages)
            if current_order == 0 and (session.title or "").strip() in {"", "New Chat"}:
                self.supabase.update_chat_session(
                    session.id,
                    current_user.id,
                    {"title": _make_title(question), "provider": provider, "last_message_at": utc_now()},
                )

        now = utc_now()
        self.supabase.insert_chat_messages(
            [
                {
                    "id": new_uuid(),
                    "session_id": session_id,
                    "user_id": current_user.id,
                    "role": "user",
                    "content": question,
                    "rewritten_query": rewritten_query,
                    "message_order": current_order + 1,
                    "model_name": model_name,
                    "created_at": now,
                },
                {
                    "id": new_uuid(),
                    "session_id": session_id,
                    "user_id": current_user.id,
                    "role": "assistant",
                    "content": answer,
                    "rewritten_query": rewritten_query,
                    "message_order": current_order + 2,
                    "model_name": model_name,
                    "created_at": now,
                },
            ]
        )
        self.supabase.update_chat_session(
            session_id,
            current_user.id,
            {
                "provider": provider,
                "last_message_at": now,
            },
        )
        return session_id

    def get_history_for_session(
        self,
        session_id: str | None,
        current_user: CurrentUser,
        fallback_history: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        if not session_id:
            return None, fallback_history
        session = self.supabase.get_chat_session(session_id, current_user.id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
        history = [
            {"role": message.role, "content": message.content}
            for message in session.messages
        ]
        return session.id, history


_SERVICE = ChatPersistenceService()


def get_chat_persistence_service() -> ChatPersistenceService:
    return _SERVICE
