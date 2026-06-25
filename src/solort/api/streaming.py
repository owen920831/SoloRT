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

    def chunk(delta: dict[str, object], finish_reason: str | None) -> str:
        return sse_event(
            {
                "id": sequence.seq_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    async for token_text in runtime.stream_request(sequence):
        yield chunk({"content": token_text}, None)

    yield chunk({}, "stop")
    yield sse_event("[DONE]")
