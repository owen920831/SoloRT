from __future__ import annotations

import pytest
from pydantic import ValidationError

from solort.api.schemas import CancelRequest, ChatCompletionRequest


def test_chat_completion_schema_accepts_openai_style_request() -> None:
    request = ChatCompletionRequest(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        max_tokens=4,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.08,
        max_repeated_token_run=16,
        enable_thinking=False,
        session_id="sess_1",
    )

    assert request.model == "Qwen/Qwen3-0.6B"
    assert request.top_p == 0.8
    assert request.top_k == 20
    assert request.repetition_penalty == 1.08
    assert request.max_repeated_token_run == 16
    assert request.core_messages()[0].content == "hello"


def test_chat_completion_schema_rejects_multimodal_content_for_mvp() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "not yet"}],
                }
            ],
        )


def test_cancel_request_requires_target() -> None:
    with pytest.raises(ValidationError):
        CancelRequest()
