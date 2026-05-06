"""Session management for a single local user."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    model_id: str
    messages: list[Message] = field(default_factory=list)
    pinned_prefix_ids: list[int] = field(default_factory=list)
    active_sequence_id: str | None = None
    last_access_ts: float = field(default_factory=time.time)
    metadata: dict[str, object] = field(default_factory=dict)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, model_id: str, session_id: str | None = None) -> Session:
        resolved_id = session_id or f"sess_{uuid.uuid4().hex}"
        session = self._sessions.get(resolved_id)
        if session is None:
            session = Session(session_id=resolved_id, model_id=model_id)
            self._sessions[resolved_id] = session
        session.last_access_ts = time.time()
        return session

    def append_messages(self, session_id: str, messages: list[Message]) -> None:
        session = self._sessions[session_id]
        session.messages.extend(messages)
        session.last_access_ts = time.time()

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def snapshot(self) -> dict[str, int]:
        active = sum(1 for session in self._sessions.values() if session.active_sequence_id)
        return {
            "sessions": len(self._sessions),
            "active_sessions": active,
        }
