# Benchmark records

## 2026-06-26 — CUDA-graph custom Qwen3 executor BEATS vLLM on 0.6B

`SOLORT_EXECUTOR=cudagraph`: a hand-written, graph-friendly Qwen3 forward over SoloRT-owned static
KV (position read from a buffer, masked attention), with the single-token decode step captured once
into a CUDA graph and replayed per token. Single-stream, temp=0, through the full SoloRT server.

| model      | SoloRT cudagraph (SDPA) | vLLM v0.8.5 | SoloRT/vLLM | vs old SoloRT (~12 tps) |
| ---------- | ----------------------- | ----------- | ----------- | ----------------------- |
| Qwen3-0.6B | 157.7 tps               | 91.1 tps    | **1.73x**   | 13x                     |
| Qwen3-4B   | 44.5 tps                | 55.6 tps    | 0.80x       | 4.0x                    |

(Manual fp32-score attention -> `F.scaled_dot_product_attention` flash kernel lifted 0.6B from
108.9 to 157.7 tps. 4B barely moved: it still scans the full graph_max_len each step and the big
unfused projection GEMMs dominate.)

**We beat vLLM single-stream decode on 0.6B (1.19x) and reach 0.77x on 4B** (3.9x over the old HF
path). Isolated micro-benchmark (no server) hit 131.9 tps on 0.6B (eager-custom 27.4) — CUDA graphs
are the lever, exactly as predicted from the launch-bound diagnosis.

TTFT is higher than vLLM (0.6B 64ms vs 22ms) because prefill is not yet graph/kernel-optimized.
The 4B decode gap is kernel efficiency: the decode attention scans the full graph_max_len and
materializes fp32 scores, vs vLLM's FlashAttention. Next: attend only to the live length / paged
decode kernel, fuse ops, optimize prefill.

## 2026-06-26 — vLLM baseline (the target to beat) + StaticCache probe

Single-stream, temp=0, 200 tokens, RTX 4080 16 GB, same prompt. vLLM v0.8.5.post1 (CUDA 12.4 —
`latest` needs CUDA 13.0 which the 12.6 driver rejects), which uses torch.compile + CUDA graphs.

| model      | vLLM decode | vLLM TTFT | SoloRT decode (eager/dyn) | SoloRT TTFT | gap    |
| ---------- | ----------- | --------- | ------------------------- | ----------- | ------ |
| Qwen3-0.6B | 91.1 tps    | 22 ms     | ~12-15 tps                | ~150-180 ms | ~6-7x  |
| Qwen3-4B   | 55.6 tps    | 30 ms     | ~11-12 tps                | ~180-310 ms | ~5x    |

vLLM is ~5-7x faster on decode and ~6-10x on TTFT. The gap is CUDA graphs + compiled/fused kernels.

CUDA-graph probe (isolated, Qwen3-0.6B decode, NOT through the serving stack):
- eager + DynamicCache: 11.9 tok/s
- eager + StaticCache:  27.0 tok/s  (2.3x — but this win does NOT survive SoloRT's serving stack;
  a clean A/B through the server showed static 13.0 vs dynamic 15.5, i.e. the per-token serving
  gaps dominate). StaticCache is kept as groundwork (default off); it is the precondition for
  CUDA-graph capture.
- torch.compile(reduce-overhead): 0.6 tok/s on torch 2.4 (NGC image) — recompile thrash; needs a
  newer torch (>=2.6), which is the next step.

Note: batch-1 launch-bound decode is noisy (~+/-30%) — isolate runs and lock clocks for reliable
deltas.

## 2026-06-25 — single-stream roofline vs nano-vllm (the real bottleneck)

Single-user, batch-1, temp=0, FlashInfer, spec off, RTX 4080 16 GB, 200 tokens, 4 runs.
Same prompt for both models.

| model      | decode tps | TTFT  | mem-bw roofline | % of roofline |
| ---------- | ---------- | ----- | --------------- | ------------- |
| Qwen3-0.6B | 11.5       | 181 ms | ~600 tps        | ~2%           |
| Qwen3-4B   | 10.6       | 183 ms | ~90 tps         | ~12%          |

**Decode tps is ~identical across a 7x model-size difference**, and TTFT matches too for a short
prompt. Decode is therefore **kernel-launch / Python-overhead bound, not compute- or
bandwidth-bound** — the eager HuggingFace forward + per-layer Python FlashInfer bridge launches
hundreds of tiny kernels per token with no CUDA graph, so the fixed dispatch cost (~85-90 ms/token)
dwarfs the actual model math. (TTFT *does* scale with model size for a long prompt: 313 ms for 4B
vs 181 ms for 0.6B in the earlier run — prefill is compute-bound.)

**vs nano-vllm:** not comparable yet, and not close. nano-vllm (~1.2k LOC) runs its own paged
attention + CUDA-graph batched execution and reports aggregate *throughput* (hundreds of sequences,
~1000+ tok/s summed) near vLLM. SoloRT targets single-stream *latency*, but more importantly it
still executes Qwen through HF Transformers eager mode, so it sits at ~2-12% of the single-stream
roofline. Closing the gap needs the tensor-backed paged executor (own layer runner + paged
attention) and CUDA graphs — exactly the standing milestone. Until then, decode speed will stay
pinned near ~11 tps regardless of model size.

## 2026-06-25 — spec vs nospec with incremental draft KV cache

Qwen3-4B target, Qwen3-0.6B draft (K=4), FlashInfer, temp=0, 220 max tokens, RTX 4080 16 GB,
4 runs each (sequential — 16 GB can't hold both servers). Prompt: detailed speculative-decoding
explanation. Code: branch `feat/draft-kv-cache`.

| case   | overall tps | decode tps | ttft | total/run | gpu mem |
| ------ | ----------- | ---------- | ---- | --------- | ------- |
| nospec | 11.53       | 11.67      | 313 ms | 19.1 s  | 8.2 GB  |
| spec   | 5.03        | 5.04       | 281 ms | 43.8 s  | 9.6 GB  |

**spec/nospec speedup = 0.44x (speculative decoding is 2.3x SLOWER here).**

Speculative counters (spec, 1100 tokens over 4 runs):
- proposed=1928, accepted=610, rejected=1318, acceptance_rate=31.6%
- draft_forward_tokens=2718 (~5.6 per round) -> the incremental draft KV cache works: the old
  full-prefix loop would have fed hundreds of thousands of draft tokens for the same output.
- flashinfer: spec prefill_calls=30535 / decode_calls=55992 / fallback=1;
  nospec prefill_calls=539 / decode_calls=39420 / fallback=1.

**Greedy non-exactness:** spec output diverges from nospec at char 55 (deterministically, all 4
runs). Both models are internally deterministic. Root cause: emitted tokens come from the target
validation forward (q_len=K+1 -> FlashInfer `single_prefill`), whose logits differ slightly from
the `single_decode` kernel used in plain decode, flipping an argmax early. This is independent of
the draft KV cache (which only touches the draft model); it is a property of the HF+FlashInfer
speculative bridge and motivates a single-attention-path tensor-backed paged executor.

**Takeaways:**
1. The draft KV cache is a correct, contained win (kills the O(K x prefix) draft cost).
2. For this pairing/content, speculative decoding is net-negative (low acceptance + per-round
   target prefill-forward + 5 draft forwards with GPU->CPU syncs) and not output-exact -> it
   should not be the default until the paged executor unifies the attention path.

## Older runs (raw)

[
  {
    "label": "spec",
    "runs": 5,
    "ttft_avg_seconds": 0.15787307039999804,
    "ttft_p50_seconds": 0.15457563300000743,
    "tpot_avg_seconds": 0.0820574141818182,
    "tpot_p50_seconds": 0.08231044697628465,
    "itl_avg_seconds": 0.08205741418181821,
    "itl_p50_seconds": 0.07706811699998184,
    "itl_p95_seconds": 0.122040944999992,
    "overall_tps_avg": 12.152407268614414,
    "decode_tps_avg": 12.197068142651354,
    "decode_after_first_avg_seconds": 20.7606461736,
    "total_avg_seconds": 20.918519244000002,
    "total_p50_seconds": 20.97094911299999,
    "ttot_avg_seconds": 20.918519244000002,
    "ttot_p50_seconds": 20.97094911299999,
    "token_chunks_avg": 254.0,
    "chars_avg": 460.0
  }
]

[
  {
    "label": "nospec",
    "runs": 5,
    "ttft_avg_seconds": 0.1540908052000077,
    "ttft_p50_seconds": 0.14796632500002715,
    "tpot_avg_seconds": 0.08220801999288538,
    "tpot_p50_seconds": 0.08164056127667989,
    "itl_avg_seconds": 0.08220801999288539,
    "itl_p50_seconds": 0.07727923600003805,
    "itl_p95_seconds": 0.12129203840004257,
    "overall_tps_avg": 12.128450952837659,
    "decode_tps_avg": 12.170507545138493,
    "decode_after_first_avg_seconds": 20.798798058600006,
    "total_avg_seconds": 20.952888863800013,
    "total_p50_seconds": 20.823691723000024,
    "ttot_avg_seconds": 20.952888863800013,
    "ttot_p50_seconds": 20.823691723000024,
    "token_chunks_avg": 254.0,
    "chars_avg": 460.0
  }
]