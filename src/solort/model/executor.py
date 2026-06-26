"""Model executor interfaces and implementations."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

from solort.cache.kv_cache import KVCacheConfig, PagedKVCache
from solort.core.batch import Batch
from solort.core.sequence import Sequence
from solort.core.session import Message
from solort.model.sampler import SampleResult

logger = logging.getLogger(__name__)


class ModelExecutor(Protocol):
    name: str

    def forward_prefill(self, batch: Batch) -> None:
        """Run a prefill chunk."""

    def forward_decode(self, batch: Batch) -> SampleResult | list[SampleResult]:
        """Run one decode step and return one or more sampled tokens."""


@dataclass
class TransformersGenerationConfig:
    model_id: str = "Qwen/Qwen3-4B"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    enable_thinking: bool = False
    trust_remote_code: bool = False
    default_temperature: float = 0.7
    default_top_p: float = 0.8
    default_top_k: int = 20
    default_repetition_penalty: float = 1.08
    default_max_repeated_token_run: int = 16
    speculative_draft_model_id: str | None = "Qwen/Qwen3-0.6B"
    # Speculative decoding is opt-in (0 = off). The GPU experiment in records.md shows that for the
    # Qwen3-0.6B->4B HF+FlashInfer bridge it is currently a net latency loss and not output-exact;
    # set SOLORT_SPECULATIVE_TOKENS>0 to enable it.
    speculative_tokens: int = 0
    speculative_draft_device_map: str | None = None
    attention_backend: str = "auto"
    # Preallocated StaticCache + sdpa for prefill/decode instead of a growing DynamicCache. In an
    # isolated tight loop this is ~2.3x faster, but through the full serving stack the per-token
    # gaps (async/HTTP/detokenize) dominate and a measured A/B showed no end-to-end win yet, so it
    # is OFF by default. It is kept as the precondition for CUDA-graph capture (fixed-shape KV),
    # which is the real lever. Incompatible with the FlashInfer bridge and speculative decoding.
    use_static_cache: bool = False
    # Max sequence length (prompt + generation) for the CUDA-graph executor's static KV buffers.
    graph_max_len: int = 2048


@dataclass
class _ServingState:
    past_key_values: object | None
    prompt_token_count: int
    prefilled_token_count: int
    pending_token_id: int | None
    generated_token_ids: list[int]
    decoded_text: str = ""
    finished: bool = False
    # Incremental detokenization offsets (HF/vLLM style): decode only a bounded suffix window per
    # token instead of the whole sequence (the old O(n^2) re-decode).
    detok_prefix_offset: int = 0
    detok_read_offset: int = 0
    # Incremental draft-model KV cache + the number of committed tokens it covers. The cache holds
    # only the accepted prefix between speculative rounds; proposed tokens are cropped off because
    # the target may reject them.
    draft_past_key_values: object | None = None
    draft_cached_len: int = 0


@dataclass(frozen=True)
class _KVWritePlan:
    """Physical slots for a model forward that appends K/V into SoloRT pages."""

    slot_mapping: list[int]
    transaction: object | None = None


@dataclass(frozen=True)
class _TargetForwardResult:
    outputs: object
    transaction: object | None = None


class TransformersTextExecutor:
    """Hugging Face executor with explicit prefill and one-token decode.

    This is the compatibility bridge. It uses Hugging Face `past_key_values` internally while the
    SoloRT scheduler, page tables, metrics, and streaming API exercise the same phase boundaries as
    the paged runtime.
    """

    name = "transformers"
    supports_prefix_cache = False

    def __init__(self, config: TransformersGenerationConfig | None = None) -> None:
        self.config = config or TransformersGenerationConfig()
        self._use_static_cache = bool(self.config.use_static_cache)
        # StaticCache needs a mask-respecting attention; force sdpa over the FlashInfer bridge.
        self._attn = (
            "sdpa" if self._use_static_cache else self.config.attention_backend.strip().lower()
        )
        self._states: dict[str, _ServingState] = {}
        self.speculative_proposed_tokens = 0
        self.speculative_accepted_tokens = 0
        self.speculative_rejected_tokens = 0
        # Total tokens fed through the draft model. With the incremental draft KV cache this grows
        # by ~accepted tokens per round instead of ~K x prefix_len, so it doubles as an efficiency
        # signal for the speculative path.
        self.speculative_draft_forward_tokens = 0
        self.kv_cache: PagedKVCache | None = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                from transformers import StaticCache
            except ImportError:
                from transformers.cache_utils import StaticCache
        except ImportError as exc:
            raise RuntimeError(
                "TransformersTextExecutor requires the model extra: "
                'install with `python -m pip install -e ".[model]"`.'
            ) from exc

        self._torch = torch
        self._StaticCache = StaticCache
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        model_kwargs = self._model_kwargs()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                **model_kwargs,
            )
        except TypeError:
            if "dtype" in model_kwargs:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                **model_kwargs,
            )
        if self.config.device_map == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.draft_model = self._load_draft_model()

    def attach_kv_cache(self, kv_cache: PagedKVCache) -> None:
        self.kv_cache = kv_cache

    def kv_cache_config(
        self,
        *,
        num_pages: int,
        page_size: int,
        allocate_tensors: bool,
    ) -> KVCacheConfig:
        model_config = self.model.config
        head_dim = getattr(
            model_config,
            "head_dim",
            model_config.hidden_size // model_config.num_attention_heads,
        )
        return KVCacheConfig(
            num_layers=int(model_config.num_hidden_layers),
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=int(model_config.num_key_value_heads),
            head_dim=int(head_dim),
            dtype=_dtype_name(getattr(self.model, "dtype", None)),
            device=str(self._model_device()),
            allocate_tensors=allocate_tensors,
        )

    def snapshot(self) -> dict[str, object]:
        acceptance_rate = (
            self.speculative_accepted_tokens / self.speculative_proposed_tokens
            if self.speculative_proposed_tokens
            else None
        )
        return {
            "attention_backend": self._attn,
            "kv_cache_type": "static" if self._use_static_cache else "dynamic",
            "speculative_enabled": self.draft_model is not None and not self._use_static_cache,
            "speculative_draft_model_id": self.config.speculative_draft_model_id,
            "speculative_tokens": self.config.speculative_tokens,
            "speculative_proposed_tokens": self.speculative_proposed_tokens,
            "speculative_accepted_tokens": self.speculative_accepted_tokens,
            "speculative_rejected_tokens": self.speculative_rejected_tokens,
            "speculative_acceptance_rate": acceptance_rate,
            "speculative_draft_forward_tokens": self.speculative_draft_forward_tokens,
            **self._attention_backend_snapshot(),
        }

    def tokenize_messages(
        self,
        messages: Iterable[Message],
        *,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        rendered_messages = messages_to_metadata(messages)
        return self._chat_token_ids(rendered_messages, enable_thinking=enable_thinking)

    def forward_prefill(self, batch: Batch) -> None:
        sequence = batch.seqs[0]
        if not batch.input_ids:
            return
        state = self._ensure_state(sequence)
        device = self._model_device()
        input_ids = self._torch.tensor([batch.input_ids], dtype=self._torch.long, device=device)

        if self._use_static_cache:
            if state.past_key_values is None:
                state.past_key_values = self._new_static_cache(sequence, device)
            start = state.prefilled_token_count
            cache_position = self._torch.arange(
                start, start + len(batch.input_ids), device=device
            )
            with self._torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    cache_position=cache_position,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )
        else:
            attention_mask = self._torch.ones(
                (1, state.prefilled_token_count + len(batch.input_ids)),
                dtype=self._torch.long,
                device=device,
            )
            with self._torch.inference_mode(), self._kv_write_context(batch):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )

        state.past_key_values = outputs.past_key_values
        state.prefilled_token_count += len(batch.input_ids)
        state.prompt_token_count = max(state.prompt_token_count, state.prefilled_token_count)
        # The logits from the final prefill chunk already contain the first decode decision, so we
        # cache it as `pending_token_id` instead of running an extra one-token forward.
        if state.prefilled_token_count >= sequence.num_prompt_tokens:
            state.pending_token_id = self._sample_token(outputs.logits[:, -1, :], sequence)

    def forward_decode(self, batch: Batch) -> SampleResult | list[SampleResult]:
        sequence = batch.seqs[0]
        state = self._ensure_state(sequence)
        if state.finished:
            return SampleResult(token_id=self._eos_token_id(), text="", finished=True)
        if state.pending_token_id is None:
            if self._speculative_enabled(sequence, state):
                return self._speculative_decode(sequence, state)
            self._decode_next(sequence, state)
        token_id = int(state.pending_token_id if state.pending_token_id is not None else 0)
        state.pending_token_id = None
        return self._append_token_result(sequence, state, token_id)

    def _append_token_result(
        self,
        sequence: Sequence,
        state: _ServingState,
        token_id: int,
    ) -> SampleResult:
        state.generated_token_ids.append(token_id)
        text = self._decode_delta(state, token_id)

        if (
            len(state.generated_token_ids) >= sequence.max_new_tokens
            or token_id == self._eos_token_id()
            or token_id in sequence.stop_token_ids
            or self._hit_repeated_token_run(sequence, state.generated_token_ids)
        ):
            state.finished = True
            self._states.pop(sequence.seq_id, None)
        return SampleResult(token_id=token_id, text=text, finished=state.finished)

    def _ensure_state(self, sequence: Sequence) -> _ServingState:
        state = self._states.get(sequence.seq_id)
        if state is not None:
            return state

        state = _ServingState(
            past_key_values=None,
            prompt_token_count=sequence.num_prompt_tokens,
            prefilled_token_count=0,
            pending_token_id=None,
            generated_token_ids=[],
        )
        self._states[sequence.seq_id] = state
        return state

    def _new_static_cache(self, sequence: Sequence, device: object) -> object:
        # Size the preallocated buffer to the full prompt + generation budget for this sequence.
        max_cache_len = sequence.num_prompt_tokens + sequence.max_new_tokens + 1
        return self._StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=self.model.dtype,
        )

    def _decode_next(self, sequence: Sequence, state: _ServingState) -> None:
        if not state.generated_token_ids:
            state.finished = True
            state.pending_token_id = self._eos_token_id()
            return

        device = self._model_device()
        input_ids = self._torch.tensor([[state.generated_token_ids[-1]]], device=device)

        if self._use_static_cache:
            position = state.prompt_token_count + len(state.generated_token_ids)
            cache_position = self._torch.tensor([position], dtype=self._torch.long, device=device)
            with self._torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    cache_position=cache_position,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )
        else:
            attention_mask = self._torch.ones(
                (1, state.prompt_token_count + len(state.generated_token_ids)),
                dtype=self._torch.long,
                device=device,
            )
            write_plan = self._kv_write_plan(
                sequence,
                start_position=state.prompt_token_count + len(state.generated_token_ids) - 1,
                token_count=1,
            )
            with self._torch.inference_mode(), self._kv_write_context(write_plan.slot_mapping):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )
        state.past_key_values = outputs.past_key_values
        state.pending_token_id = self._sample_token(outputs.logits[:, -1, :], sequence)

    def _speculative_decode(
        self,
        sequence: Sequence,
        state: _ServingState,
    ) -> list[SampleResult]:
        remaining = sequence.max_new_tokens - len(state.generated_token_ids)
        draft_count = min(self.config.speculative_tokens, remaining)
        draft_tokens = self._draft_tokens(sequence, state, draft_count)
        if not draft_tokens:
            self._decode_next(sequence, state)
            token_id = int(state.pending_token_id if state.pending_token_id is not None else 0)
            state.pending_token_id = None
            return [self._append_token_result(sequence, state, token_id)]

        original_past = state.past_key_values
        # Target validation asks the 4B model to score the previous accepted token plus all draft
        # candidates in one pass. Matching logits accept draft tokens; the first mismatch becomes
        # the correction token.
        validate_outputs = self._target_forward_from_state(
            sequence,
            state,
            [state.generated_token_ids[-1], *draft_tokens],
            provisional=True,
        )
        target_tokens = [
            self._greedy_token(validate_outputs.outputs.logits[:, index, :])
            for index in range(len(draft_tokens))
        ]

        accepted = 0
        for draft_token, target_token in zip(draft_tokens, target_tokens, strict=False):
            if draft_token != target_token:
                break
            accepted += 1

        self.speculative_proposed_tokens += len(draft_tokens)
        self.speculative_accepted_tokens += accepted
        self.speculative_rejected_tokens += len(draft_tokens) - accepted

        results: list[SampleResult] = []
        if accepted == len(draft_tokens):
            # Full acceptance can keep the target forward pass cache because it already contains
            # every draft token. The extra recovery token is emitted but not cached until the next
            # one-token decode step, matching the non-speculative path.
            if validate_outputs.transaction is not None and self.kv_cache is not None:
                self.kv_cache.commit_transaction(validate_outputs.transaction)
            state.past_key_values = validate_outputs.outputs.past_key_values
            for token_id in draft_tokens:
                results.append(self._append_token_result(sequence, state, token_id))
                if state.finished:
                    return results

            if len(state.generated_token_ids) < sequence.max_new_tokens:
                recovery_token = self._sample_token(
                    validate_outputs.outputs.logits[:, -1, :],
                    sequence,
                )
                results.append(self._append_token_result(sequence, state, recovery_token))
            return results

        accepted_prefix = draft_tokens[:accepted]
        # Partial rejection is transactional: roll back the target cache to the original prefix,
        # replay only accepted draft tokens, then emit the target correction token.
        if validate_outputs.transaction is not None and self.kv_cache is not None:
            self.kv_cache.rollback_transaction(sequence.block_table, validate_outputs.transaction)
        state.past_key_values = original_past
        prefix_outputs = self._target_forward_from_state(
            sequence,
            state,
            [state.generated_token_ids[-1], *accepted_prefix],
            provisional=False,
        )
        state.past_key_values = prefix_outputs.outputs.past_key_values

        for token_id in accepted_prefix:
            results.append(self._append_token_result(sequence, state, token_id))
            if state.finished:
                return results

        correction_token = target_tokens[accepted]
        results.append(self._append_token_result(sequence, state, correction_token))
        return results

    def _target_forward_from_state(
        self,
        sequence: Sequence,
        state: _ServingState,
        token_ids: list[int],
        *,
        provisional: bool,
    ) -> _TargetForwardResult:
        device = self._model_device()
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=device)
        attention_mask = self._torch.ones(
            (1, state.prompt_token_count + len(state.generated_token_ids) + len(token_ids) - 1),
            dtype=self._torch.long,
            device=device,
        )
        current_token_count = state.prompt_token_count + len(state.generated_token_ids)
        write_plan = self._kv_write_plan(
            sequence,
            start_position=current_token_count - 1,
            token_count=len(token_ids),
            current_token_count=current_token_count,
            provisional_append_tokens=max(0, len(token_ids) - 1) if provisional else 0,
        )
        with self._torch.inference_mode(), self._kv_write_context(write_plan.slot_mapping):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        return _TargetForwardResult(outputs=outputs, transaction=write_plan.transaction)

    def _draft_tokens(
        self,
        sequence: Sequence,
        state: _ServingState,
        draft_count: int,
    ) -> list[int]:
        if self.draft_model is None or draft_count <= 0:
            return []

        device = self._draft_device()
        committed = list(sequence.input_ids) + list(state.generated_token_ids)
        committed_len = len(committed)
        if committed_len == 0:
            return []

        proposed: list[int] = []
        with self._torch.inference_mode():
            # Sync the draft cache to the accepted prefix, then propose K tokens one incremental
            # forward at a time. The prefix is processed at most once per round (only the newly
            # accepted tail), so a proposal round costs ~accepted tokens, not ~K x prefix_len.
            logits = self._advance_draft_cache(state, committed, device)
            for index in range(draft_count):
                token_id = self._greedy_token(logits)
                proposed.append(token_id)
                if token_id == self._eos_token_id() or token_id in sequence.stop_token_ids:
                    break
                # Skip the forward after the final proposal: its logits would never be used.
                if index + 1 >= draft_count:
                    break
                logits = self._draft_forward([token_id], state, device).logits[:, -1, :]
        # Proposed tokens are provisional: the target may reject them, so they must not persist in
        # the draft cache across rounds. Keep only the accepted committed prefix.
        self._crop_draft_cache(state, committed_len)
        return proposed

    def _advance_draft_cache(
        self,
        state: _ServingState,
        committed: list[int],
        device: object,
    ) -> object:
        """Extend the draft cache to cover ``committed`` and return next-token logits."""

        committed_len = len(committed)
        # Drop any stale speculative tail left over from a previous round.
        if state.draft_cached_len > committed_len:
            self._crop_draft_cache(state, committed_len)
        tail = committed[state.draft_cached_len :]
        if not tail:
            # The cache already covers the full committed prefix but we need fresh logits for the
            # next token; re-feed only the last committed token without duplicating earlier K/V.
            self._crop_draft_cache(state, committed_len - 1)
            tail = committed[committed_len - 1 :]
        return self._draft_forward(tail, state, device).logits[:, -1, :]

    def _draft_forward(
        self,
        token_ids: list[int],
        state: _ServingState,
        device: object,
    ) -> object:
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=device)
        outputs = self.draft_model(
            input_ids=input_ids,
            past_key_values=state.draft_past_key_values,
            use_cache=True,
        )
        state.draft_past_key_values = outputs.past_key_values
        state.draft_cached_len += len(token_ids)
        self.speculative_draft_forward_tokens += len(token_ids)
        return outputs

    def _crop_draft_cache(self, state: _ServingState, length: int) -> None:
        past = state.draft_past_key_values
        if past is None:
            state.draft_cached_len = 0
            return
        if length <= 0:
            state.draft_past_key_values = None
            state.draft_cached_len = 0
            return
        crop = getattr(past, "crop", None)
        if callable(crop):
            cropped = crop(length)
            if cropped is not None:
                state.draft_past_key_values = cropped
            state.draft_cached_len = min(state.draft_cached_len, length)
        else:
            # The cache type cannot be truncated safely; rebuild from scratch next round so the
            # draft proposals stay correct (at the cost of one full-prefix draft pass).
            state.draft_past_key_values = None
            state.draft_cached_len = 0

    def _speculative_enabled(self, sequence: Sequence, state: _ServingState) -> bool:
        # The StaticCache fast path does not support the speculative rollback/replay logic; enable
        # speculation only on the dynamic-cache path (SOLORT_STATIC_CACHE=0).
        if self._use_static_cache:
            return False
        if self.draft_model is None or self.config.speculative_tokens <= 0:
            return False
        if not state.generated_token_ids:
            return False
        if sequence.max_new_tokens - len(state.generated_token_ids) <= 1:
            return False
        repetition_penalty = float(
            sequence.metadata.get(
                "repetition_penalty",
                self.config.default_repetition_penalty,
            )
        )
        if repetition_penalty != 1.0:
            return False
        temperature = float(sequence.metadata.get("temperature", self.config.default_temperature))
        # v1 uses exact greedy speculative decoding. Sampling needs stochastic acceptance math and
        # is intentionally disabled here.
        return temperature <= 0

    def _sample_token(self, logits: object, sequence: Sequence) -> int:
        torch = self._torch
        scores = logits[0].float()
        temperature = float(sequence.metadata.get("temperature", self.config.default_temperature))
        top_p = float(sequence.metadata.get("top_p", self.config.default_top_p))
        top_k = int(sequence.metadata.get("top_k", self.config.default_top_k))
        repetition_penalty = float(
            sequence.metadata.get(
                "repetition_penalty",
                self.config.default_repetition_penalty,
            )
        )
        scores = self._apply_repetition_penalty(scores, sequence, repetition_penalty)

        if temperature <= 0:
            return int(torch.argmax(scores).item())

        scores = scores / temperature
        if top_k > 0 and top_k < scores.numel():
            threshold = torch.topk(scores, top_k).values[-1]
            scores = torch.where(
                scores < threshold,
                torch.full_like(scores, float("-inf")),
                scores,
            )
        if 0 < top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scores[indices_to_remove] = float("-inf")

        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _greedy_token(self, logits: object) -> int:
        return int(self._torch.argmax(logits[0].float()).item())

    def _apply_repetition_penalty(
        self,
        scores: object,
        sequence: Sequence,
        repetition_penalty: float,
    ) -> object:
        if repetition_penalty <= 1.0 or not sequence.output_ids:
            return scores
        for token_id in set(sequence.output_ids):
            if token_id < 0 or token_id >= scores.numel():
                continue
            if scores[token_id] > 0:
                scores[token_id] /= repetition_penalty
            else:
                scores[token_id] *= repetition_penalty
        return scores

    def _hit_repeated_token_run(self, sequence: Sequence, token_ids: list[int]) -> bool:
        max_run = int(
            sequence.metadata.get(
                "max_repeated_token_run",
                self.config.default_max_repeated_token_run,
            )
        )
        if max_run <= 0 or len(token_ids) < max_run:
            return False
        last_token = token_ids[-1]
        return all(token_id == last_token for token_id in token_ids[-max_run:])

    def _chat_token_ids(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool | None,
    ) -> list[int]:
        thinking = self.config.enable_thinking if enable_thinking is None else enable_thinking
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except ValueError:
            text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(token_id) for token_id in token_ids]

    def _decode_delta(self, state: _ServingState, token_id: int) -> str:
        if token_id == self._eos_token_id():
            return ""

        # Incremental detokenization: decode only the bounded suffix window [prefix_offset:] and
        # diff against [prefix_offset:read_offset]. The window stays small (offsets advance), so it
        # is O(window) per token instead of re-decoding the whole sequence. Defer emitting while the
        # decode ends in a replacement char (an incomplete multi-token character).
        tokens = state.generated_token_ids
        prefix_text = self.tokenizer.decode(
            tokens[state.detok_prefix_offset : state.detok_read_offset],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        new_text = self.tokenizer.decode(
            tokens[state.detok_prefix_offset :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
            return ""
        state.detok_prefix_offset = state.detok_read_offset
        state.detok_read_offset = len(tokens)
        return new_text[len(prefix_text) :]

    def _eos_token_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        return int(token_id if token_id is not None else 0)

    def _model_device(self) -> object:
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _draft_device(self) -> object:
        try:
            return self.draft_model.device
        except AttributeError:
            return next(self.draft_model.parameters()).device

    def _model_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        attention_backend = self._attn
        if attention_backend == "flashinfer":
            # Registering a Transformers attention implementation lets Qwen layers stay in HF while
            # their attention math dispatches to FlashInfer.
            from solort.backends.transformers_flashinfer import (
                ATTENTION_NAME,
                register_solort_flashinfer_attention,
            )

            register_solort_flashinfer_attention()
            kwargs["attn_implementation"] = ATTENTION_NAME
        elif attention_backend in {"eager", "sdpa", "flash_attention_2", "flex_attention"}:
            kwargs["attn_implementation"] = attention_backend

        if self.config.torch_dtype != "auto":
            kwargs["dtype"] = getattr(self._torch, self.config.torch_dtype)
        else:
            kwargs["dtype"] = "auto"
        if self.config.device_map != "cpu":
            kwargs["device_map"] = self.config.device_map
        return kwargs

    def _kv_write_context(self, batch_or_slots: Batch | list[int] | None) -> object:
        if self._attn != "flashinfer":
            return nullcontext()
        from solort.backends.transformers_flashinfer import solort_kv_write_context

        slot_mapping = (
            batch_or_slots.slot_mapping
            if isinstance(batch_or_slots, Batch)
            else batch_or_slots
        )
        return solort_kv_write_context(
            self.kv_cache,
            slot_mapping,
        )

    def _kv_write_plan(
        self,
        sequence: Sequence,
        *,
        start_position: int,
        token_count: int,
        current_token_count: int | None = None,
        provisional_append_tokens: int = 0,
    ) -> _KVWritePlan:
        if self.kv_cache is None or token_count <= 0:
            return _KVWritePlan(slot_mapping=[])
        end_position = start_position + token_count
        if provisional_append_tokens > 0:
            transaction = self.kv_cache.begin_append_transaction(
                sequence.block_table,
                current_token_count=current_token_count or start_position + 1,
                append_token_count=provisional_append_tokens,
            )
        else:
            transaction = None
            self.kv_cache.ensure_capacity(sequence.block_table, end_position)
        positions = list(range(start_position, end_position))
        return _KVWritePlan(
            slot_mapping=self.kv_cache.slot_mapping(sequence.block_table, positions),
            transaction=transaction,
        )

    def _attention_backend_snapshot(self) -> dict[str, object]:
        if self._attn != "flashinfer":
            return {}
        try:
            from solort.backends.transformers_flashinfer import flashinfer_attention_snapshot
        except ImportError:
            return {}
        return flashinfer_attention_snapshot()

    def _load_draft_model(self) -> object | None:
        if not self.config.speculative_draft_model_id or self.config.speculative_tokens <= 0:
            return None

        from transformers import AutoModelForCausalLM

        kwargs = self._model_kwargs()
        draft_device_map = self.config.speculative_draft_device_map
        if draft_device_map is not None:
            if draft_device_map == "cpu":
                kwargs.pop("device_map", None)
            else:
                kwargs["device_map"] = draft_device_map

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.speculative_draft_model_id,
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.speculative_draft_model_id,
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        if draft_device_map == "cpu":
            model.to("cpu")
        model.eval()
        if not self._draft_vocab_compatible(model):
            return None
        return model

    def _draft_vocab_compatible(self, draft_model: object) -> bool:
        """Greedy speculative decoding compares draft and target token ids directly, so the two
        models must share a vocabulary. A size mismatch means a misconfigured draft pairing that
        would otherwise emit silently wrong tokens, so disable speculation with a clear warning.
        """

        target_vocab = getattr(getattr(self.model, "config", None), "vocab_size", None)
        draft_vocab = getattr(getattr(draft_model, "config", None), "vocab_size", None)
        if target_vocab is None or draft_vocab is None or int(target_vocab) == int(draft_vocab):
            return True
        logger.warning(
            "Disabling speculative decoding: draft %s vocab_size=%s != target %s vocab_size=%s; "
            "use a draft model from the same tokenizer family.",
            self.config.speculative_draft_model_id,
            draft_vocab,
            self.config.model_id,
            target_vocab,
        )
        return False


class QwenTransformersExecutor(TransformersTextExecutor):
    """Convenience executor for `Qwen/Qwen3-0.6B`."""

    def __init__(self) -> None:
        super().__init__(TransformersGenerationConfig(model_id="Qwen/Qwen3-0.6B"))


class PagedQwenExecutor(TransformersTextExecutor):
    """Qwen serving bridge configured like the paged runtime.

    The class is intentionally explicit about being a bridge: SoloRT now builds real page metadata
    and KV transactions outside the executor, while this path still delegates layer execution to
    Transformers until the custom Qwen layer runner lands.
    """

    name = "paged-qwen-transformers-bridge"
    supports_prefix_cache = False

    def snapshot(self) -> dict[str, object]:
        data = super().snapshot()
        mirrored = self.kv_cache is not None and self.kv_cache.has_tensors
        data.update(
            {
                "target_model_id": self.config.model_id,
                "cache_boundary": (
                    "solort-page-metadata + solort-kv-mirror + hf-past-key-values"
                    if mirrored
                    else "solort-page-metadata + hf-past-key-values"
                ),
                "paged_executor_status": (
                    "FlashInfer HF attention bridge enabled; "
                    "SoloRT KV mirror enabled; tensor-backed paged runner pending"
                    if mirrored
                    else (
                        "FlashInfer HF attention bridge enabled; "
                        "tensor-backed paged runner pending"
                    )
                ),
            }
        )
        return data


def messages_to_metadata(messages: Iterable[Message]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _text_delta(previous_text: str, new_text: str) -> str:
    if new_text.startswith(previous_text):
        return new_text[len(previous_text) :]

    common = 0
    max_common = min(len(previous_text), len(new_text))
    while common < max_common and previous_text[common] == new_text[common]:
        common += 1
    return new_text[common:]


def _dtype_name(dtype: object | None) -> str:
    text = str(dtype or "float16").replace("torch.", "")
    if text == "float16":
        return "fp16"
    if text == "bfloat16":
        return "bf16"
    if text == "float32":
        return "fp32"
    return text
