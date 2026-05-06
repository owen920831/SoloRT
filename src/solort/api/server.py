"""FastAPI app for SoloRT serving."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from solort.api.schemas import CancelRequest, ChatCompletionRequest
from solort.api.streaming import chat_completion_events
from solort.core.runtime import RuntimeCore, build_default_runtime
from solort.core.sequence import TaskKind


def create_app(runtime: RuntimeCore | None = None) -> FastAPI:
    app = FastAPI(title="SoloRT", version="0.1.0")
    app.state.runtime = runtime or build_default_runtime()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        core: RuntimeCore = app.state.runtime
        return core.metrics_snapshot()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> object:
        core: RuntimeCore = app.state.runtime
        sequence = core.add_request(
            model_id=request.model,
            messages=request.core_messages(),
            max_new_tokens=request.max_tokens,
            session_id=request.session_id,
            task_kind=TaskKind(request.priority),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_repeated_token_run=request.max_repeated_token_run,
            enable_thinking=request.enable_thinking,
        )

        if request.stream:
            return StreamingResponse(
                chat_completion_events(core, sequence, request.model),
                media_type="text/event-stream; charset=utf-8",
            )

        content = await core.complete_request(sequence)
        return _chat_completion_response(
            request_id=sequence.seq_id,
            session_id=sequence.session_id,
            model=request.model,
            content=content,
            prompt_tokens=sequence.num_prompt_tokens,
            completion_tokens=sequence.generated_tokens,
        )

    @app.post("/v1/cancel")
    async def cancel(request: CancelRequest) -> dict[str, bool]:
        core: RuntimeCore = app.state.runtime
        cancelled = core.cancel_request(
            request_id=request.request_id,
            session_id=request.session_id,
        )
        return {"cancelled": cancelled}

    @app.post("/v1/responses")
    async def responses_stub() -> JSONResponse:
        return _not_implemented("The Responses API is reserved for a later SoloRT milestone.")

    @app.post("/v1/embeddings")
    async def embeddings_stub() -> JSONResponse:
        return _not_implemented("Embeddings are outside the text-generation MVP.")

    return app


def _chat_completion_response(
    *,
    request_id: str,
    session_id: str,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "session_id": session_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _not_implemented(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "detail": detail,
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
