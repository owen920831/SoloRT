"""SSE helpers for OpenAI-style streaming responses."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from solort.core.runtime import RuntimeCore
from solort.core.sequence import Sequence


def sse_event(data: dict[str, object] | str) -> str:
    payload = (
        data
        if isinstance(data, str)
        else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    )
    return f"data: {payload}\n\n"


async def chat_completion_events(
    runtime: RuntimeCore,
    sequence: Sequence,
    model: str,
) -> AsyncIterator[str]:
    created = int(time.time())
    async for token_text in runtime.stream_request(sequence):
        yield sse_event(
            {
                "id": sequence.seq_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield sse_event(
        {
            "id": sequence.seq_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield sse_event("[DONE]")
