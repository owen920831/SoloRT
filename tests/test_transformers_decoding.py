from __future__ import annotations

import contextlib
import types

from solort.core.sequence import Sequence
from solort.model.executor import (
    TransformersGenerationConfig,
    TransformersTextExecutor,
    _ServingState,
    _text_delta,
)


class _Pred:
    """Stand-in for a logits row whose argmax is a fixed token id."""

    def __init__(self, token_id: int) -> None:
        self._token_id = token_id

    def __getitem__(self, _key: object) -> _Pred:
        return self

    def float(self) -> _Pred:
        return self

    def item(self) -> int:
        return self._token_id


class _FakeLogits:
    def __init__(self, token_id: int) -> None:
        self._token_id = token_id

    def __getitem__(self, _key: object) -> _Pred:
        return _Pred(self._token_id)


class _FakeOutputs:
    def __init__(self, token_id: int, past_key_values: object) -> None:
        self.logits = _FakeLogits(token_id)
        self.past_key_values = past_key_values


class _FakeDraftCache:
    def __init__(self) -> None:
        self.length = 0

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)


class _FakeTensor:
    def __init__(self, data: list[list[int]]) -> None:
        self.data = data


class _FakeTorch:
    long = "long"

    def tensor(self, data: list[list[int]], dtype: object = None, device: object = None):
        return _FakeTensor(data)

    def inference_mode(self):
        return contextlib.nullcontext()

    def argmax(self, value: object) -> object:
        return value


class _FakeDraftModel:
    """Greedy draft model that predicts ``last_input_token + 1`` and records fed lengths."""

    device = "cpu"

    def __init__(self) -> None:
        self.fed_lengths: list[int] = []

    def __call__(
        self,
        input_ids: _FakeTensor,
        past_key_values: object = None,
        use_cache: bool = True,
    ) -> _FakeOutputs:
        tokens = input_ids.data[0]
        self.fed_lengths.append(len(tokens))
        cache = past_key_values or _FakeDraftCache()
        cache.length += len(tokens)
        return _FakeOutputs(token_id=tokens[-1] + 1, past_key_values=cache)


def _draft_executor() -> tuple[TransformersTextExecutor, _FakeDraftModel]:
    executor = TransformersTextExecutor.__new__(TransformersTextExecutor)
    executor.config = TransformersGenerationConfig(speculative_tokens=4)
    executor._torch = _FakeTorch()
    executor.draft_model = _FakeDraftModel()
    executor.tokenizer = types.SimpleNamespace(eos_token_id=999)
    executor.speculative_draft_forward_tokens = 0
    return executor, executor.draft_model


def _draft_sequence() -> Sequence:
    return Sequence(seq_id="req", session_id="sess", input_ids=[1, 2, 3], max_new_tokens=64)


def test_draft_cache_processes_prefix_once_then_decodes_incrementally() -> None:
    executor, draft_model = _draft_executor()
    sequence = _draft_sequence()
    state = _ServingState(
        past_key_values=None,
        prompt_token_count=3,
        prefilled_token_count=3,
        pending_token_id=None,
        generated_token_ids=[10],
    )

    proposed = executor._draft_tokens(sequence, state, 4)

    # Greedy draft predicts last + 1, so committed tail 10 -> 11, 12, 13, 14.
    assert proposed == [11, 12, 13, 14]
    # One forward over the 4-token committed prefix, then three single-token incremental forwards.
    assert draft_model.fed_lengths == [4, 1, 1, 1]
    assert executor.speculative_draft_forward_tokens == 7
    # The cache is cropped back to the committed prefix; proposals do not persist.
    assert state.draft_cached_len == 4


def test_draft_cache_reuses_prefix_across_rounds() -> None:
    executor, draft_model = _draft_executor()
    sequence = _draft_sequence()
    state = _ServingState(
        past_key_values=None,
        prompt_token_count=3,
        prefilled_token_count=3,
        pending_token_id=None,
        generated_token_ids=[10],
    )

    executor._draft_tokens(sequence, state, 4)
    draft_model.fed_lengths.clear()

    # Target accepted two drafts (11, 12) and appended a correction token (50): three new tokens.
    state.generated_token_ids = [10, 11, 12, 50]
    proposed = executor._draft_tokens(sequence, state, 4)

    assert proposed == [51, 52, 53, 54]
    # Round two re-feeds only the 3 newly committed tokens, NOT the full 7-token prefix, then
    # three incremental single-token forwards.
    assert draft_model.fed_lengths == [3, 1, 1, 1]
    assert state.draft_cached_len == 7


def test_text_delta_uses_simple_suffix_when_text_is_stable() -> None:
    assert _text_delta("你好", "你好 SoloRT") == " SoloRT"


def test_text_delta_recovers_when_tokenizer_rewrites_prefix() -> None:
    assert _text_delta("hello world", "hello, world!") == ", world!"


def test_repeated_token_run_guard_detects_degenerate_tail() -> None:
    executor = TransformersTextExecutor.__new__(TransformersTextExecutor)
    executor.config = TransformersGenerationConfig(default_max_repeated_token_run=4)
    sequence = Sequence(
        seq_id="req",
        session_id="sess",
        input_ids=[1],
        max_new_tokens=16,
    )

    assert executor._hit_repeated_token_run(sequence, [7, 7, 7]) is False
    assert executor._hit_repeated_token_run(sequence, [7, 7, 7, 7]) is True
