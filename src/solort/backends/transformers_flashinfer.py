"""FlashInfer attention bridge for Hugging Face transformer layers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

ATTENTION_NAME = "solort_flashinfer"


@dataclass
class FlashInferAttentionStats:
    prefill_calls: int = 0
    decode_calls: int = 0
    fallback_calls: int = 0


_STATS = FlashInferAttentionStats()
_REGISTERED = False
_KV_CONTEXT: dict[str, Any] | None = None


@contextmanager
def solort_kv_write_context(kv_cache: Any | None, slot_mapping: list[int] | None) -> Any:
    """Expose the current batch's SoloRT KV destination to HF attention callbacks."""

    global _KV_CONTEXT
    previous = _KV_CONTEXT
    _KV_CONTEXT = {"kv_cache": kv_cache, "slot_mapping": slot_mapping or []}
    try:
        yield
    finally:
        _KV_CONTEXT = previous


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
        "flashinfer_fallback_calls": _STATS.fallback_calls,
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
    if query.shape[0] != 1:
        raise RuntimeError("SoloRT FlashInfer bridge currently supports batch_size=1")
    if query.device.type != "cuda":
        raise RuntimeError("SoloRT FlashInfer bridge requires CUDA tensors")

    # Transformers gives [batch, heads, tokens, dim]. FlashInfer's dense kernels use NHD:
    # [tokens, heads, dim], so the bridge only changes layout and dispatches.
    q = query[0].transpose(0, 1).contiguous()
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    window_left = int(sliding_window) if sliding_window else -1
    _mirror_to_solort_kv(module, key, value)

    if q.shape[0] == 1:
        decode_fn, _ = _flashinfer_attention_fns()
        if decode_fn is not None:
            try:
                _STATS.decode_calls += 1
                output = decode_fn(
                    q[0],
                    k,
                    v,
                    kv_layout="NHD",
                    window_left=window_left,
                    sm_scale=float(scaling),
                )
                return output.unsqueeze(0).unsqueeze(0), None
            except Exception:
                _STATS.decode_calls -= 1
        return _torch_attention_fallback(query, key, value, scaling, causal=False), None

    _, prefill_fn = _flashinfer_attention_fns()
    if prefill_fn is not None:
        try:
            _STATS.prefill_calls += 1
            output = prefill_fn(
                q,
                k,
                v,
                causal=True,
                kv_layout="NHD",
                window_left=window_left,
                sm_scale=float(scaling),
            )
            return output.unsqueeze(0), None
        except Exception:
            _STATS.prefill_calls -= 1
    return _torch_attention_fallback(query, key, value, scaling, causal=True), None


def _flashinfer_attention_fns() -> tuple[Any | None, Any | None]:
    try:
        import flashinfer
    except Exception:
        return None, None

    decode_module = getattr(flashinfer, "decode", None)
    prefill_module = getattr(flashinfer, "prefill", None)
    decode_fn = getattr(decode_module, "single_decode_with_kv_cache", None) or getattr(
        flashinfer,
        "single_decode_with_kv_cache",
        None,
    )
    prefill_fn = getattr(prefill_module, "single_prefill_with_kv_cache", None) or getattr(
        flashinfer,
        "single_prefill_with_kv_cache",
        None,
    )
    return decode_fn, prefill_fn


def _torch_attention_fallback(
    query: Any,
    key: Any,
    value: Any,
    scaling: float,
    *,
    causal: bool,
) -> Any:
    _STATS.fallback_calls += 1
    import torch

    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise RuntimeError(
                f"cannot expand KV heads {key.shape[1]} to query heads {query.shape[1]}"
            )
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    scale = float(scaling)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
        scale=scale,
    )
    return output.transpose(1, 2).contiguous()


def _mirror_to_solort_kv(module: Any, key: Any, value: Any) -> None:
    if _KV_CONTEXT is None:
        return
    kv_cache = _KV_CONTEXT.get("kv_cache")
    if kv_cache is None:
        return
    store = getattr(kv_cache, "store_layer_tokens", None)
    if not callable(store):
        return
    store(
        layer_idx=getattr(module, "layer_idx", None),
        slot_mapping=list(_KV_CONTEXT.get("slot_mapping") or []),
        key=key,
        value=value,
    )
