from __future__ import annotations

import pytest

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


def test_torch_fallback_uses_bottom_right_causal_for_cached_prefill() -> None:
    torch = pytest.importorskip("torch")
    from solort.backends.transformers_flashinfer import _torch_attention_fallback

    torch.manual_seed(0)
    b, h, d = 1, 2, 8
    q_len, kv_len = 3, 7  # query block sits at the END of a 7-token cache
    query = torch.randn(b, h, q_len, d)
    key = torch.randn(b, h, kv_len, d)
    value = torch.randn(b, h, kv_len, d)
    scale = 1.0 / (d**0.5)

    got = _torch_attention_fallback(query, key, value, scale, causal=True)  # [b, q_len, h, d]

    # Independent reference: bottom-right causal (query token i may attend kv 0..offset+i).
    offset = kv_len - q_len
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    qi = torch.arange(q_len).unsqueeze(1)
    ki = torch.arange(kv_len).unsqueeze(0)
    mask = ki <= (qi + offset)
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = torch.matmul(torch.softmax(scores, dim=-1), value).transpose(1, 2)
    assert torch.allclose(got, ref, atol=1e-5)

    # The previously-shipped top-left path (is_causal=True) gives a different, wrong result.
    buggy = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, scale=scale
    ).transpose(1, 2)
    assert not torch.allclose(got, buggy, atol=1e-3)
