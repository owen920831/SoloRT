"""OpenAI-compatible API schemas for the MVP."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solort.core.session import Message

ChatRole = Literal["system", "user", "assistant", "tool"]
TaskPriority = Literal["foreground", "background", "branch"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)

    model_config = ConfigDict(extra="allow")

    def to_core_message(self) -> Message:
        return Message(role=self.role, content=self.content)


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen3-4B"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    max_repeated_token_run: int | None = Field(default=None, ge=0, le=1024)
    enable_thinking: bool = False
    session_id: str | None = None
    priority: TaskPriority = "foreground"

    model_config = ConfigDict(extra="allow")

    def core_messages(self) -> list[Message]:
        return [message.to_core_message() for message in self.messages]


class CancelRequest(BaseModel):
    request_id: str | None = None
    session_id: str | None = None

    @model_validator(mode="after")
    def require_target(self) -> CancelRequest:
        if self.request_id is None and self.session_id is None:
            raise ValueError("request_id or session_id is required")
        return self
