"""Model executor interfaces and implementations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from solort.core.batch import Batch
from solort.core.sequence import Sequence
from solort.core.session import Message
from solort.model.sampler import SampleResult


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
    speculative_tokens: int = 4
    speculative_draft_device_map: str | None = None
    attention_backend: str = "auto"


@dataclass
class _ServingState:
    past_key_values: object | None
    prompt_token_count: int
    prefilled_token_count: int
    pending_token_id: int | None
    generated_token_ids: list[int]
    decoded_text: str = ""
    finished: bool = False


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
        self._states: dict[str, _ServingState] = {}
        self.speculative_proposed_tokens = 0
        self.speculative_accepted_tokens = 0
        self.speculative_rejected_tokens = 0

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "TransformersTextExecutor requires the model extra: "
                'install with `python -m pip install -e ".[model]"`.'
            ) from exc

        self._torch = torch
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

    def snapshot(self) -> dict[str, object]:
        acceptance_rate = (
            self.speculative_accepted_tokens / self.speculative_proposed_tokens
            if self.speculative_proposed_tokens
            else None
        )
        return {
            "attention_backend": self.config.attention_backend,
            "speculative_enabled": self.draft_model is not None,
            "speculative_draft_model_id": self.config.speculative_draft_model_id,
            "speculative_tokens": self.config.speculative_tokens,
            "speculative_proposed_tokens": self.speculative_proposed_tokens,
            "speculative_accepted_tokens": self.speculative_accepted_tokens,
            "speculative_rejected_tokens": self.speculative_rejected_tokens,
            "speculative_acceptance_rate": acceptance_rate,
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
        attention_mask = self._torch.ones(
            (1, state.prefilled_token_count + len(batch.input_ids)),
            dtype=self._torch.long,
            device=device,
        )

        with self._torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            )

        state.past_key_values = outputs.past_key_values
        state.prefilled_token_count += len(batch.input_ids)
        state.prompt_token_count = max(state.prompt_token_count, state.prefilled_token_count)
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

    def _decode_next(self, sequence: Sequence, state: _ServingState) -> None:
        if not state.generated_token_ids:
            state.finished = True
            state.pending_token_id = self._eos_token_id()
            return

        device = self._model_device()
        input_ids = self._torch.tensor([[state.generated_token_ids[-1]]], device=device)
        attention_mask = self._torch.ones(
            (1, state.prompt_token_count + len(state.generated_token_ids)),
            dtype=self._torch.long,
            device=device,
        )
        with self._torch.inference_mode():
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
        validate_outputs = self._target_forward_from_state(
            state,
            [state.generated_token_ids[-1], *draft_tokens],
        )
        target_tokens = [
            self._greedy_token(validate_outputs.logits[:, index, :])
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
            state.past_key_values = validate_outputs.past_key_values
            for token_id in draft_tokens:
                results.append(self._append_token_result(sequence, state, token_id))
                if state.finished:
                    return results

            if len(state.generated_token_ids) < sequence.max_new_tokens:
                recovery_token = self._sample_token(validate_outputs.logits[:, -1, :], sequence)
                results.append(self._append_token_result(sequence, state, recovery_token))
            return results

        accepted_prefix = draft_tokens[:accepted]
        # Partial rejection is transactional: roll back the target cache to the original prefix,
        # replay only accepted draft tokens, then emit the target correction token.
        state.past_key_values = original_past
        prefix_outputs = self._target_forward_from_state(
            state,
            [state.generated_token_ids[-1], *accepted_prefix],
        )
        state.past_key_values = prefix_outputs.past_key_values

        for token_id in accepted_prefix:
            results.append(self._append_token_result(sequence, state, token_id))
            if state.finished:
                return results

        correction_token = target_tokens[accepted]
        results.append(self._append_token_result(sequence, state, correction_token))
        return results

    def _target_forward_from_state(self, state: _ServingState, token_ids: list[int]) -> object:
        device = self._model_device()
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=device)
        attention_mask = self._torch.ones(
            (1, state.prompt_token_count + len(state.generated_token_ids) + len(token_ids) - 1),
            dtype=self._torch.long,
            device=device,
        )
        with self._torch.inference_mode():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            )

    def _draft_tokens(
        self,
        sequence: Sequence,
        state: _ServingState,
        draft_count: int,
    ) -> list[int]:
        if self.draft_model is None or draft_count <= 0:
            return []

        device = self._draft_device()
        draft_ids = list(sequence.input_ids) + list(state.generated_token_ids)
        proposed: list[int] = []
        with self._torch.inference_mode():
            for _ in range(draft_count):
                input_ids = self._torch.tensor([draft_ids], dtype=self._torch.long, device=device)
                outputs = self.draft_model(input_ids=input_ids, use_cache=False)
                token_id = self._greedy_token(outputs.logits[:, -1, :])
                proposed.append(token_id)
                draft_ids.append(token_id)
                if token_id == self._eos_token_id() or token_id in sequence.stop_token_ids:
                    break
        return proposed

    def _speculative_enabled(self, sequence: Sequence, state: _ServingState) -> bool:
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

        new_text = self.tokenizer.decode(
            state.generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        delta = _text_delta(state.decoded_text, new_text)
        state.decoded_text = new_text
        return delta

    def _eos_token_id(self) -> int:
        token_id = self.tokenizer.eos_token_id
        return int(token_id if token_id is not None else 0)

    def _model_device(self) -> object:
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _draft_device(self) -> object:
        if self.draft_model is None:
            return self._model_device()
        try:
            return self.draft_model.device
        except AttributeError:
            return next(self.draft_model.parameters()).device

    def _model_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        attention_backend = self.config.attention_backend.strip().lower()
        if attention_backend == "flashinfer":
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

    def _attention_backend_snapshot(self) -> dict[str, object]:
        if self.config.attention_backend.strip().lower() != "flashinfer":
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
        return model


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
        data.update(
            {
                "target_model_id": self.config.model_id,
                "cache_boundary": "solort-page-metadata + hf-past-key-values",
                "paged_executor_status": (
                    "FlashInfer HF attention bridge enabled; tensor-backed paged runner pending"
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
