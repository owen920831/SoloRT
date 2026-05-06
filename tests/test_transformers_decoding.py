from __future__ import annotations

from solort.core.sequence import Sequence
from solort.model.executor import (
    TransformersGenerationConfig,
    TransformersTextExecutor,
    _text_delta,
)


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
