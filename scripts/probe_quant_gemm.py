"""Go/no-go probe: is weight-only quant actually faster at batch-1 on Ada (sm_89)?

The 4B cudagraph decode is weight-memory-bound (~73% of bf16 roofline). int8/fp8 weight-only halves
weight bytes, but only helps if the quantized GEMM kernel is faster than bf16 at M=1 on this GPU.
torch 2.4 made it 2-5x SLOWER; this re-checks on torch 2.6 + torchao over the 4B's real decode GEMM
shapes, reporting per-token GEMM latency (summed over all layers) and the quantization error.

Run in the quant image:
  docker run --rm --gpus all -e HF_HOME=/root/.cache/huggingface \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $PWD/src:/app/src -v $PWD/scripts:/app/scripts solort:quant \
    python scripts/probe_quant_gemm.py Qwen/Qwen3-4B
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
from transformers import AutoConfig


def _bench(linear: nn.Module, x: torch.Tensor, iters: int = 200) -> float:
    for _ in range(10):
        linear(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        linear(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/call


def _quantized(make_linear, x_ref, recipe):
    """Clone a fresh bf16 linear, apply a torchao recipe, return (module, max_abs_err vs bf16)."""
    from torchao.quantization import quantize_

    lin = make_linear()
    ref = lin(x_ref).float()
    quantize_(lin, recipe())
    out = lin(x_ref).float()
    err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    return lin, err


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
    cfg = AutoConfig.from_pretrained(model_id)
    h = cfg.hidden_size
    inter = cfg.intermediate_size
    hd = getattr(cfg, "head_dim", h // cfg.num_attention_heads)
    qd = cfg.num_attention_heads * hd
    kvd = cfg.num_key_value_heads * hd
    nl = cfg.num_hidden_layers
    vocab = cfg.vocab_size
    dev = "cuda"

    # (name, out_features N, in_features K, count-per-token)
    gemms = [
        ("qkv_proj", qd + 2 * kvd, h, nl),
        ("o_proj", h, qd, nl),
        ("gate_up", 2 * inter, h, nl),
        ("down_proj", h, inter, nl),
        ("lm_head", vocab, h, 1),
    ]

    print(f"{model_id}  sm={torch.cuda.get_device_capability()}  torch={torch.__version__}")
    print(f"hidden={h} inter={inter} qd={qd} kvd={kvd} layers={nl} vocab={vocab}\n")

    from torchao.quantization import int4_weight_only, int8_weight_only

    # int4 tinygemm (_weight_int4pack_mm) is the gpt-fast batch-1 decode kernel (1/4 the bytes,
    # low-batch tuned) -- the most promising avenue after int8/fp8 GEMV lost.
    recipes = [
        ("int4_wo", lambda: int4_weight_only(group_size=128)),
        ("int8_wo", int8_weight_only),
    ]
    try:
        from torchao.quantization import float8_weight_only

        recipes.append(("fp8_wo", float8_weight_only))
    except ImportError:
        print("(float8_weight_only unavailable in this torchao)\n")

    totals = {"bf16": 0.0}
    errs: dict[str, float] = {}
    for rname, _ in recipes:
        totals[rname] = 0.0

    header = f"{'gemm':10} {'N':>7} {'K':>6} {'bf16 us':>9}"
    for rname, _ in recipes:
        header += f" {rname + ' us':>10} {'x':>5} {'err':>8}"
    print(header)

    for name, N, K, count in gemms:
        def make() -> nn.Module:
            return nn.Linear(K, N, bias=False).to(dev).to(torch.bfloat16)

        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        base = make()
        t_bf16 = _bench(base, x)
        totals["bf16"] += t_bf16 * count
        row = f"{name:10} {N:>7} {K:>6} {t_bf16 * 1000:>9.1f}"
        for rname, recipe in recipes:
            try:
                qlin, err = _quantized(make, x, recipe)
                t_q = _bench(qlin, x)
                totals[rname] += t_q * count
                errs[rname] = max(errs.get(rname, 0.0), err)
                row += f" {t_q * 1000:>10.1f} {t_bf16 / t_q:>5.2f} {err:>8.4f}"
            except Exception as exc:  # noqa: BLE001
                row += f" {'ERR':>10} {'-':>5} {str(exc)[:8]:>8}"
        print(row)

    print(f"\nper-token GEMM latency (all layers): bf16 {totals['bf16']:.3f} ms")
    for rname, _ in recipes:
        spd = totals["bf16"] / totals[rname] if totals[rname] else 0.0
        print(f"  {rname:8}: {totals[rname]:.3f} ms  ({spd:.2f}x vs bf16)  max_rel_err={errs.get(rname, 0):.4f}")
    print("\nVerdict: integrate only the recipes with >1.0x AND acceptable err.")


if __name__ == "__main__":
    main()
