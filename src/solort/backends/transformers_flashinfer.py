"""FlashInfer attention bridge for Hugging Face transformer layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ATTENTION_NAME = "solort_flashinfer"


@dataclass
class FlashInferAttentionStats:
    prefill_calls: int = 0
    decode_calls: int = 0


_STATS = FlashInferAttentionStats()
_REGISTERED = False


def register_solort_flashinfer_attention() -> None:
    """Register a FlashInfer attention implementation with Transformers.

    This bridge is intentionally narrower than the future paged executor: it receives dense K/V
    tensors from Hugging Face's cache object, then dispatches attention math to FlashInfer. That
    gives SoloRT a real FlashInfer execution path while the full tensor-backed paged runner is still
    being built.
    """

    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from transformers import AttentionInterface
    except ImportError:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, solort_flashinfer_attention_forward)
    _REGISTERED = True


def flashinfer_attention_snapshot() -> dict[str, int | str]:
    return {
        "attention_backend": ATTENTION_NAME,
        "flashinfer_prefill_calls": _STATS.prefill_calls,
        "flashinfer_decode_calls": _STATS.decode_calls,
    }


def solort_flashinfer_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    scaling: float,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Transformers AttentionInterface-compatible FlashInfer call."""

    del attention_mask, dropout, kwargs
    flashinfer = _load_flashinfer()

    if query.shape[0] != 1:
        raise RuntimeError("SoloRT FlashInfer bridge currently supports batch_size=1")
    if query.device.type != "cuda":
        raise RuntimeError("SoloRT FlashInfer bridge requires CUDA tensors")

    q = query[0].transpose(0, 1).contiguous()
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    window_left = int(sliding_window) if sliding_window else -1

    if q.shape[0] == 1:
        _STATS.decode_calls += 1
        output = flashinfer.decode.single_decode_with_kv_cache(
            q[0],
            k,
            v,
            kv_layout="NHD",
            window_left=window_left,
            sm_scale=float(scaling),
        )
        return output.unsqueeze(0).unsqueeze(0), None

    _STATS.prefill_calls += 1
    output = flashinfer.prefill.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=True,
        kv_layout="NHD",
        window_left=window_left,
        sm_scale=float(scaling),
    )
    return output.unsqueeze(0), None


def _load_flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError(
            "SOLORT_ATTENTION_BACKEND=flashinfer requires `flashinfer-python` in the image."
        ) from exc
    return flashinfer
