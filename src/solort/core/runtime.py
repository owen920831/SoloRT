"""RuntimeCore wires sessions, scheduling, cache metadata, and model execution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator, Iterable

from solort.cache.kv_cache import KVCacheConfig, PagedKVCache
from solort.cache.prefix_cache import PrefixCache, PrefixMatch
from solort.cache.vram_manager import VRAMBudgetManager
from solort.core.batch import Batch
from solort.core.metrics import RuntimeMetrics
from solort.core.scheduler import InteractiveScheduler
from solort.core.sequence import BatchPhase, Sequence, SequenceStatus, TaskKind
from solort.core.session import Message, SessionManager
from solort.model.executor import (
    ModelExecutor,
    PagedQwenExecutor,
    TransformersGenerationConfig,
    TransformersTextExecutor,
    messages_to_metadata,
)
from solort.model.sampler import SampleResult


class RuntimeCore:
    """Python MVP runtime core.

    The public shape is intentionally stable enough to wrap a future C++ runtime core behind the
    same API server.
    """

    def __init__(
        self,
        scheduler: InteractiveScheduler | None = None,
        kv_cache: PagedKVCache | None = None,
        prefix_cache: PrefixCache | None = None,
        executor: ModelExecutor | None = None,
        session_manager: SessionManager | None = None,
        metrics: RuntimeMetrics | None = None,
        vram_manager: VRAMBudgetManager | None = None,
    ) -> None:
        self.scheduler = scheduler or InteractiveScheduler()
        self.kv_cache = kv_cache or PagedKVCache(KVCacheConfig(num_pages=1024, page_size=16))
        self.prefix_cache = prefix_cache or PrefixCache(block_size=16, max_entries=128)
        if executor is None:
            raise ValueError("RuntimeCore requires an executor; use build_default_runtime()")
        self.executor = executor
        # Executors that own their KV (e.g. the cudagraph runner) opt out of SoloRT's paged-KV
        # bookkeeping so the decode hot path does no unused per-token work.
        self._uses_paged_kv = bool(getattr(executor, "uses_paged_kv", True))
        attach_kv_cache = getattr(self.executor, "attach_kv_cache", None)
        if callable(attach_kv_cache):
            attach_kv_cache(self.kv_cache)
        self.session_manager = session_manager or SessionManager()
        self.metrics = metrics or RuntimeMetrics()
        self.vram_manager = vram_manager or VRAMBudgetManager()
        self._active_by_session: dict[str, str] = {}

    def add_request(
        self,
        *,
        model_id: str,
        messages: Iterable[Message],
        max_new_tokens: int,
        session_id: str | None = None,
        task_kind: TaskKind = TaskKind.FOREGROUND,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_repeated_token_run: int | None = None,
        enable_thinking: bool | None = None,
    ) -> Sequence:
        message_list = list(messages)
        session = self.session_manager.get_or_create(model_id=model_id, session_id=session_id)
        self.session_manager.append_messages(session.session_id, message_list)
        context_messages = list(session.messages)

        metadata = {
            "model_id": model_id,
            "messages": messages_to_metadata(context_messages),
            "request_messages": messages_to_metadata(message_list),
            "response_text": "",
            "response_appended": False,
            "temperature": temperature,
            "top_p": top_p if top_p is not None else 0.8,
            "top_k": top_k if top_k is not None else 20,
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else 1.08,
            "max_repeated_token_run": (
                max_repeated_token_run if max_repeated_token_run is not None else 16
            ),
            "enable_thinking": bool(enable_thinking) if enable_thinking is not None else False,
        }
        # Todo: can reuse history kvcache?
        input_ids = self._tokenize_for_executor(
            context_messages,
            enable_thinking=bool(metadata["enable_thinking"]),
        )
        # Prefix cache is deliberately queried after chat-template tokenization. That keeps cache
        # keys aligned with the exact token stream seen by the model.
        prefix_match = (
            self.prefix_cache.match(input_ids)
            if self._supports_prefix_cache()
            else PrefixMatch(matched_tokens=0, pages=[])
        )

        seq_id = f"req_{uuid.uuid4().hex}"
        sequence = Sequence(
            seq_id=seq_id,
            session_id=session.session_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            task_kind=task_kind,
            num_cached_tokens=prefix_match.matched_tokens,
            block_table=list(prefix_match.pages),
            cache_handle=prefix_match.entry,
            metadata=metadata,
        )
        session.active_sequence_id = seq_id
        self._active_by_session[session.session_id] = seq_id
        self.scheduler.add_sequence(sequence)
        self.metrics.start_request(seq_id)
        return sequence

    async def stream_request(self, sequence: Sequence) -> AsyncIterator[str]:
        """Yield decoded token text for a request."""

        while not sequence.is_terminal:
            batch = self.scheduler.build_next_batch()
            if batch is None:
                await asyncio.sleep(0)
                continue
            if batch.seqs[0].seq_id != sequence.seq_id:
                self._execute_batch(batch)
                await asyncio.sleep(0)
                continue

            token_texts = self._execute_batch(batch)
            for token_text in token_texts:
                yield token_text
            await asyncio.sleep(0)

        self._finish_sequence(sequence)

    async def complete_request(self, sequence: Sequence) -> str:
        chunks = []
        async for token_text in self.stream_request(sequence):
            chunks.append(token_text)
        return "".join(chunks)

    def cancel_request(self, request_id: str | None = None, session_id: str | None = None) -> bool:
        seq_id = request_id
        if seq_id is None and session_id is not None:
            seq_id = self._active_by_session.get(session_id)
        if seq_id is None:
            return False
        sequence = self.scheduler.get(seq_id)
        cancelled = self.scheduler.cancel(seq_id)
        if cancelled and sequence is not None:
            self._finish_sequence(sequence)
        return cancelled

    def metrics_snapshot(self) -> dict[str, object]:
        return {
            "runtime": self.metrics.snapshot(),
            "executor": getattr(self.executor, "name", self.executor.__class__.__name__),
            "executor_stats": self._executor_snapshot(),
            "scheduler": self.scheduler.snapshot(),
            "sessions": self.session_manager.snapshot(),
            "kv_cache": self.kv_cache.snapshot(),
            "prefix_cache": self.prefix_cache.snapshot(),
            "vram": self.vram_manager.snapshot(),
        }

    def _execute_batch(self, batch: Batch) -> list[str]:
        sequence = batch.seqs[0]
        if sequence.status == SequenceStatus.CANCELLED:
            return []

        if batch.phase == BatchPhase.PREFILL:
            total_tokens_after_chunk = (
                sequence.num_cached_tokens + sequence.num_scheduled_tokens + len(batch.input_ids)
            )
            # Build paged metadata only for executors that consume it. The cudagraph executor owns
            # its own KV, so skipping this avoids wasted per-step Python and unused page allocs.
            if self._uses_paged_kv:
                self.kv_cache.ensure_capacity(sequence.block_table, total_tokens_after_chunk)
                self._attach_kv_metadata(batch, token_count=total_tokens_after_chunk)
            self.executor.forward_prefill(batch)
            self.scheduler.postprocess_batch(batch)
            if (
                sequence.is_prefill_complete
                and sequence.cache_handle is None
                and self._supports_prefix_cache()
            ):
                entry = self.prefix_cache.insert(
                    sequence.input_ids,
                    sequence.block_table,
                    pinned=True,
                )
                sequence.cache_handle = entry
            return []

        total_tokens_after_decode = sequence.num_prompt_tokens + len(sequence.output_ids) + 1
        # Decode allocates for exactly one new logical token. Speculative decode may return several
        # accepted tokens, but the current HF bridge keeps their dense cache internally.
        if self._uses_paged_kv:
            self.kv_cache.ensure_capacity(sequence.block_table, total_tokens_after_decode)
            self._attach_kv_metadata(batch, token_count=total_tokens_after_decode)
        samples = self._normalize_samples(self.executor.forward_decode(batch))
        token_texts: list[str] = []
        for sample in samples:
            if sequence.is_terminal:
                break
            sequence.output_ids.append(sample.token_id)
            if sample.text:
                sequence.metadata["response_text"] = (
                    str(sequence.metadata.get("response_text", "")) + sample.text
                )
                token_texts.append(sample.text)
            self.metrics.record_token(sequence.seq_id)
            if (
                sequence.generated_tokens >= sequence.max_new_tokens
                or sample.token_id in sequence.stop_token_ids
                or sample.finished
            ):
                sequence.mark_finished()
                break
        return token_texts

    def _finish_sequence(self, sequence: Sequence) -> None:
        session = self.session_manager.get(sequence.session_id)
        if session is not None and session.active_sequence_id == sequence.seq_id:
            session.active_sequence_id = None
        self._active_by_session.pop(sequence.session_id, None)
        if sequence.status == SequenceStatus.FINISHED:
            self.metrics.finish_request(sequence.seq_id)
            self._append_assistant_response(sequence)
        if sequence.cache_handle is not None:
            self.prefix_cache.release(sequence.cache_handle)

    def _supports_prefix_cache(self) -> bool:
        return bool(getattr(self.executor, "supports_prefix_cache", True))

    def _append_assistant_response(self, sequence: Sequence) -> None:
        if sequence.metadata.get("response_appended"):
            return
        content = str(sequence.metadata.get("response_text", ""))
        if not content:
            return
        session = self.session_manager.get(sequence.session_id)
        if session is None:
            return
        self.session_manager.append_messages(
            sequence.session_id,
            [Message(role="assistant", content=content)],
        )
        sequence.metadata["response_appended"] = True

    @staticmethod
    def _normalize_samples(sample_or_samples: object) -> list[SampleResult]:
        if isinstance(sample_or_samples, SampleResult):
            return [sample_or_samples]
        if isinstance(sample_or_samples, list):
            return sample_or_samples
        raise TypeError("executor.forward_decode must return SampleResult or list[SampleResult]")

    def _attach_kv_metadata(self, batch: Batch, *, token_count: int) -> None:
        sequence = batch.seqs[0]
        # These four fields are the handoff contract between scheduling/cache policy and attention:
        # logical positions are mapped to physical KV slots, while page arrays describe each
        # sequence's active prefix.
        metadata = self.kv_cache.metadata_for(
            sequence.block_table,
            token_count=token_count,
            positions=batch.positions,
        )
        batch.page_indptr = metadata.page_indptr
        batch.page_indices = metadata.page_indices
        batch.last_page_len = metadata.last_page_len
        batch.slot_mapping = metadata.slot_mapping

    def _executor_snapshot(self) -> dict[str, object]:
        snapshot = getattr(self.executor, "snapshot", None)
        if callable(snapshot):
            return dict(snapshot())
        return {}

    @staticmethod
    def tokenize_messages(messages: Iterable[Message]) -> list[int]:
        parts: list[str] = []
        for message in messages:
            parts.append(f"{message.role}: {message.content}")
        text = "\n".join(parts).strip()
        if not text:
            return [0]
        tokens = text.split()
        return [RuntimeCore._stable_token_id(token) for token in tokens]

    @staticmethod
    def _stable_token_id(text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "little") % 262_144

    def _tokenize_for_executor(
        self,
        messages: Iterable[Message],
        *,
        enable_thinking: bool,
    ) -> list[int]:
        tokenizer = getattr(self.executor, "tokenize_messages", None)
        if callable(tokenizer):
            return list(tokenizer(messages, enable_thinking=enable_thinking))
        return self.tokenize_messages(messages)


def build_default_runtime() -> RuntimeCore:
    executor_name = os.getenv("SOLORT_EXECUTOR", "paged").strip().lower()
    if executor_name in {"paged", "paged-qwen", "qwen-paged"}:
        executor_cls: type[ModelExecutor] = PagedQwenExecutor
        default_backend = "flashinfer"
    elif executor_name in {"transformers", "hf", "qwen"}:
        executor_cls = TransformersTextExecutor
        default_backend = "auto"
    elif executor_name in {"cudagraph", "cuda-graph", "graph"}:
        from solort.model.cuda_graph_executor import (
            CudaGraphQwen3Executor,
            SpecCudaGraphQwen3Executor,
        )

        if _env_int("SOLORT_SPECULATIVE_TOKENS", default=0) > 0:
            executor_cls = SpecCudaGraphQwen3Executor  # cudagraph + speculative decoding
        else:
            executor_cls = CudaGraphQwen3Executor
        default_backend = "sdpa"
    else:
        raise ValueError(f"unknown SOLORT_EXECUTOR={executor_name!r}")

    speculative_tokens = _env_int("SOLORT_SPECULATIVE_TOKENS", default=0)

    executor = executor_cls(
        TransformersGenerationConfig(
            model_id=os.getenv("SOLORT_MODEL_ID", "Qwen/Qwen3-4B"),
            device_map=os.getenv("SOLORT_DEVICE_MAP", "auto"),
            torch_dtype=os.getenv("SOLORT_TORCH_DTYPE", "auto"),
            enable_thinking=_env_bool("SOLORT_ENABLE_THINKING", default=False),
            trust_remote_code=_env_bool("SOLORT_TRUST_REMOTE_CODE", default=False),
            speculative_draft_model_id=os.getenv(
                "SOLORT_SPECULATIVE_DRAFT_MODEL_ID",
                "Qwen/Qwen3-0.6B",
            ),
            speculative_tokens=speculative_tokens,
            speculative_draft_device_map=os.getenv("SOLORT_SPECULATIVE_DRAFT_DEVICE_MAP"),
            attention_backend=os.getenv("SOLORT_ATTENTION_BACKEND", default_backend),
            graph_max_len=_env_int("SOLORT_GRAPH_MAX_LEN", default=1024),
        )
    )
    kv_cache = _build_runtime_kv_cache(executor)
    return RuntimeCore(executor=executor, kv_cache=kv_cache)


def _build_runtime_kv_cache(executor: ModelExecutor) -> PagedKVCache:
    num_pages = _env_int("SOLORT_KV_NUM_PAGES", default=1024)
    page_size = _env_int("SOLORT_KV_PAGE_SIZE", default=16)
    allocate_tensors = _env_bool("SOLORT_KV_TENSOR_STORAGE", default=False)
    config_factory = getattr(executor, "kv_cache_config", None)
    if callable(config_factory):
        return PagedKVCache(
            config_factory(
                num_pages=num_pages,
                page_size=page_size,
                allocate_tensors=allocate_tensors,
            )
        )
    return PagedKVCache(
        KVCacheConfig(
            num_pages=num_pages,
            page_size=page_size,
            allocate_tensors=allocate_tensors,
        )
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)
