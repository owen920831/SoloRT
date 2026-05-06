from __future__ import annotations

import asyncio

from solort.cache.kv_cache import KVCacheConfig, PagedKVCache
from solort.cache.prefix_cache import PrefixCache
from solort.core.runtime import RuntimeCore
from solort.core.session import Message
from solort.model.sampler import SampleResult


class BurstDecodeExecutor:
    name = "burst"
    supports_prefix_cache = False

    def forward_prefill(self, batch: object) -> None:
        del batch

    def forward_decode(self, batch: object) -> list[SampleResult]:
        del batch
        return [
            SampleResult(token_id=1, text="A"),
            SampleResult(token_id=2, text="B"),
            SampleResult(token_id=3, text="C"),
        ]


def test_runtime_non_streaming_completion_and_metrics() -> None:
    runtime = RuntimeCore(kv_cache=PagedKVCache(KVCacheConfig(num_pages=64, page_size=4)))
    sequence = runtime.add_request(
        model_id="mock",
        messages=[Message(role="user", content="say hello")],
        max_new_tokens=3,
    )

    content = asyncio.run(runtime.complete_request(sequence))
    metrics = runtime.metrics_snapshot()

    assert content == "SoloRT keeps foreground"
    assert metrics["runtime"]["tokens_generated"] == 3
    assert metrics["executor"] == "mock"
    assert metrics["kv_cache"]["used_pages"] >= 1


def test_runtime_accepts_multi_token_decode_results() -> None:
    runtime = RuntimeCore(
        kv_cache=PagedKVCache(KVCacheConfig(num_pages=64, page_size=4)),
        executor=BurstDecodeExecutor(),
    )
    sequence = runtime.add_request(
        model_id="burst",
        messages=[Message(role="user", content="burst please")],
        max_new_tokens=3,
    )

    content = asyncio.run(runtime.complete_request(sequence))
    metrics = runtime.metrics_snapshot()

    assert content == "ABC"
    assert sequence.output_ids == [1, 2, 3]
    assert metrics["runtime"]["tokens_generated"] == 3


def test_runtime_stores_real_model_request_metadata() -> None:
    runtime = RuntimeCore()
    sequence = runtime.add_request(
        model_id="Qwen/Qwen3-0.6B",
        messages=[Message(role="user", content="metadata check")],
        max_new_tokens=3,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.08,
        max_repeated_token_run=16,
        enable_thinking=False,
    )

    assert sequence.metadata["model_id"] == "Qwen/Qwen3-0.6B"
    assert sequence.metadata["messages"] == [{"role": "user", "content": "metadata check"}]
    assert sequence.metadata["request_messages"] == [
        {"role": "user", "content": "metadata check"}
    ]
    assert sequence.metadata["temperature"] == 0.7
    assert sequence.metadata["top_p"] == 0.8
    assert sequence.metadata["top_k"] == 20
    assert sequence.metadata["repetition_penalty"] == 1.08
    assert sequence.metadata["max_repeated_token_run"] == 16
    assert sequence.metadata["enable_thinking"] is False


def test_runtime_prefix_cache_hit_on_repeated_prompt() -> None:
    runtime = RuntimeCore(
        kv_cache=PagedKVCache(KVCacheConfig(num_pages=64, page_size=4)),
        prefix_cache=PrefixCache(block_size=2, max_entries=8),
    )
    messages = [Message(role="user", content="repeat this prompt")]

    first = runtime.add_request(model_id="mock", messages=messages, max_new_tokens=1)
    asyncio.run(runtime.complete_request(first))

    second = runtime.add_request(model_id="mock", messages=messages, max_new_tokens=1)
    asyncio.run(runtime.complete_request(second))

    assert runtime.metrics_snapshot()["prefix_cache"]["hits"] >= 1


def test_runtime_uses_session_history_as_chat_context() -> None:
    runtime = RuntimeCore(kv_cache=PagedKVCache(KVCacheConfig(num_pages=64, page_size=4)))
    first = runtime.add_request(
        model_id="mock",
        session_id="chat",
        messages=[Message(role="user", content="first question")],
        max_new_tokens=2,
    )

    first_content = asyncio.run(runtime.complete_request(first))
    session = runtime.session_manager.get("chat")

    assert first_content == "SoloRT keeps"
    assert session is not None
    assert [(message.role, message.content) for message in session.messages] == [
        ("user", "first question"),
        ("assistant", "SoloRT keeps"),
    ]

    second = runtime.add_request(
        model_id="mock",
        session_id="chat",
        messages=[Message(role="user", content="second question")],
        max_new_tokens=1,
    )

    assert second.metadata["messages"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "SoloRT keeps"},
        {"role": "user", "content": "second question"},
    ]
    assert second.metadata["request_messages"] == [
        {"role": "user", "content": "second question"}
    ]


def test_runtime_cancels_active_request() -> None:
    runtime = RuntimeCore()
    sequence = runtime.add_request(
        model_id="mock",
        messages=[Message(role="user", content="cancel me")],
        max_new_tokens=8,
    )

    assert runtime.cancel_request(request_id=sequence.seq_id) is True
    assert runtime.cancel_request(request_id=sequence.seq_id) is False
