from __future__ import annotations

from solort.backends.transformers_flashinfer import (
    ATTENTION_NAME,
    flashinfer_attention_snapshot,
)


def test_flashinfer_attention_snapshot_is_import_safe() -> None:
    snapshot = flashinfer_attention_snapshot()

    assert snapshot["attention_backend"] == ATTENTION_NAME
    assert snapshot["flashinfer_prefill_calls"] >= 0
    assert snapshot["flashinfer_decode_calls"] >= 0
    assert snapshot["flashinfer_fallback_calls"] >= 0
