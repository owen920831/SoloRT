from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("httpx")

from solort.api.server import create_app
from solort.api.streaming import sse_event
from solort.core.runtime import RuntimeCore
from solort.core.session import Message


def test_non_streaming_chat_completion_endpoint() -> None:
    client = TestClient(create_app(RuntimeCore()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "SoloRT keeps"
    assert body["usage"]["completion_tokens"] == 2
    assert body["session_id"].startswith("sess_")


def test_streaming_chat_completion_endpoint() -> None:
    client = TestClient(create_app(RuntimeCore()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
        },
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "chat.completion.chunk" in body
    assert "SoloRT" in body
    assert "data: [DONE]" in body


def test_sse_event_preserves_utf8_text() -> None:
    event = sse_event({"choices": [{"delta": {"content": "你好 SoloRT"}}]})

    assert "你好 SoloRT" in event
    assert "\\u4f60" not in event


def test_cancel_endpoint() -> None:
    runtime = RuntimeCore()
    sequence = runtime.add_request(
        model_id="mock",
        messages=[Message(role="user", content="cancel through api")],
        max_new_tokens=4,
    )
    client = TestClient(create_app(runtime))

    response = client.post("/v1/cancel", json={"request_id": sequence.seq_id})

    assert response.status_code == 200
    assert response.json() == {"cancelled": True}


def test_metrics_endpoint_reflects_completed_request() -> None:
    client = TestClient(create_app(RuntimeCore()))
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "metrics"}],
            "max_tokens": 1,
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["runtime"]["tokens_generated"] == 1
