# Benchmark records

## 2026-06-26 — Grouped attention (no KV repeat): 4B 55 -> 67 tps (1.21x vLLM)

The decode attention used `repeat_interleave` to GQA-expand KV to `[nh, bound, hd]` every step
(4x memory traffic for 4B's 32/8 heads) before SDPA. Micro-benchmark: SDPA+repeat ~330-390us/call
(repeat dominates, ~flat in bound) vs grouped matmul 173-184us (1.9x). Replaced it with grouped
attention reading the cache as `[nkv, bound, hd]` directly (batched matmul over kv-heads). This also
fixed the large-buffer slowdown (the KV repeat scaled with the scan).

| path (through server, greedy, graph_max_len=1024) | decode tps | TTFT  | vs vLLM |
| -------------------------------------------------- | ---------- | ----- | ------- |
| 0.6B cudagraph                                     | 149        | 11ms  | 1.64x   |
| 4B cudagraph                                       | **67**     | 27ms  | **1.21x** |
| vLLM 0.6B / 4B                                     | 91 / 55.6  | 22/30 | 1.0x    |

4B no longer needs a tiny graph_max_len to beat vLLM — grouped attention gives 1.21x at 1024
context. **SoloRT now beats vLLM on both models, decode AND TTFT.** Output verified coherent.

## 2026-06-26 — Graphed bucketed prefill: TTFT 74ms -> 20ms (beats vLLM)

TTFT profiling: it was ~entirely the **eager prefill forward** (74ms for a 24-token prompt;
tokenize 0.2ms, first decode 0.15ms). Eager prefill = ~360 kernel launches across 36 layers ->
launch-bound, exactly what graphs fixed for decode. Fix = **graphed bucketed prefill**: pad the
prompt to a length bucket (16/32/64/.../1024), causal-forward it in one captured graph, read the
last real position's hidden, apply lm_head eagerly for that one position (the graph outputs
[bound,hidden] ~MBs, not [bound,vocab] ~600MB). Real tokens never attend padding (causal order), so
output is exact; padding KV is masked out by decode.

| metric                  | before | after  | vLLM    |
| ----------------------- | ------ | ------ | ------- |
| prefill (24-tok, isolated) | 74ms | 20.6ms | -       |
| TTFT through server, 0.6B  | ~60ms | 13ms  | 22ms    |
| TTFT through server, 4B    | ~75ms | 33ms  | 30ms    |

Output verified coherent + exact (English, Traditional Chinese, lists). So SoloRT now beats/matches
vLLM on TTFT too (0.6B 13<22, 4B 33~=30), on top of the decode wins.

## 2026-06-26 — Bucketed decode graphs: 0.6B 2.02x vLLM, 4B parity-to-1.19x

cProfile of the decode loop settled the bottleneck: 88% of time is `.item()` blocking on the GPU
graph (the model forward); Python is ~4%. So it is GPU-compute bound and "fully on GPU" already.
The one remaining GPU waste was attention scanning the full `graph_max_len` each step (SDPA does not
skip masked keys). Fix = **bucketed CUDA graphs** (vLLM-style): one decode graph per length bucket
(128/256/512/1024/...), routed by position, so attention scans only ~live length.

| path (through server, greedy)     | decode tps | vs vLLM (0.6B 91 / 4B 55.6) |
| ---------------------------------- | ---------- | --------------------------- |
| 0.6B cudagraph (bucketed)          | 183        | **2.02x**                   |
| 4B  cudagraph, graph_max_len=256   | 66         | 1.19x                       |
| 4B  cudagraph, graph_max_len=1024  | 55         | 1.00x (parity)              |
| 4B  cudagraph, graph_max_len=2048  | 54         | 0.97x                       |

Note: even with bucketing (same ~256-key scan), a larger KV buffer is slower (256->66, 1024->55,
2048->54) — the buffer size itself costs memory locality. Default set to graph_max_len=1024
(context vs speed balance). So 4B is parity-to-winning vs vLLM depending on context length, 0.6B
wins decisively, both greedy + exact.

## 2026-06-26 — Profiled the decode bottleneck; in-graph argmax (4B 45 -> 52 tps)

Profiled the 4B cudagraph decode per-token (in-process, no HTTP):

| phase                                   | ms/tok | tps  |
| --------------------------------------- | ------ | ---- |
| raw graph replay (no per-token sync)    | 17.8   | 56   |
| + eager argmax over 151936 vocab + .item() | 22.1 | 45   |
| in-graph argmax + .item() of 1 elem     | 16.9   | 59   |

So the real bottleneck was NOT detokenize (~0 ms at 200 tokens) but the **eager argmax/`.float()`
over the 151936-vocab each token (~5 ms), run on the CPU side with the GPU idle**. Moving the
greedy argmax INTO the CUDA graph (pipelined on-GPU, read a 1-element token) removed it:

| 4B path (through server)        | before | after in-graph argmax | vs vLLM 55.6 |
| ------------------------------- | ------ | --------------------- | ------------ |
| cudagraph target-only (greedy)  | 45.2   | 51.6                  | 0.93x        |
| cudagraph + spec (K=3)          | 54.3   | 53.2                  | 0.96x        |

(Isolated decode_argmax is 59 tps = 1.06x vLLM; the server still loses ~2.4 ms/token to
async/SSE/HTTP + runtime per-token work, which is the next lever. With the per-token argmax cost
gone, spec and target-only nearly converge — spec amortized exactly that cost.)

### + incremental detok + skip unused paged-KV bookkeeping

Replaced the O(n^2) full-sequence re-decode (`_decode_delta`) with HF/vLLM-style incremental
detokenization (bounded suffix window, defer on a trailing replacement char), and skip
ensure_capacity/_attach_kv_metadata for the cudagraph executor (it owns its KV). HTTP/SSE adds
~nothing (RuntimeCore no-HTTP 52.7 tps ~= through-server), so the residual per-token overhead is
runtime Python, not HTTP. Multi-byte streaming verified coherent (Traditional Chinese).

### CLEAN final numbers (warmup 2, runs 6) — honest

| path (through server, greedy)      | decode tps | TTFT  | vs vLLM      |
| ---------------------------------- | ---------- | ----- | ------------ |
| 0.6B cudagraph                     | 154        | 59ms  | **1.69x**    |
| 4B cudagraph target-only           | 51.6       | 83ms  | 0.93x        |
| 4B cudagraph + spec (K=3)          | 50.5       | 143ms | 0.91x        |
| vLLM 0.6B / 4B (reference)         | 91 / 55.6  | 22/30ms | 1.0x       |

So: **0.6B beats vLLM decisively (1.69x)**; **4B is ~0.93x** (close). Note the 4B decode *compute*
already beats vLLM (isolated decode_argmax 59 tps > 55.6), so the ~7% server gap is ~2.4ms/token of
RuntimeCore/executor **Python** per token (scheduler Batch rebuild, sample/append, async loop), NOT
GPU. And after moving argmax on-GPU, **speculative decoding no longer helps 4B** — it had only been
amortizing the per-token eager-argmax cost, which is now gone; the draft + verify overhead now
slightly exceeds the benefit. TTFT lags vLLM (prefill is eager, not graph-captured).

Next bottleneck = the ~2.4ms/token serving Python (hot-loop micro-opt) and TTFT (graph-capture
prefill).

## 2026-06-26 — Speculative decoding on the cudagraph runner (4B beats/ties vLLM)

`SOLORT_EXECUTOR=cudagraph SOLORT_SPECULATIVE_TOKENS=3` (4B target + 0.6B draft). Both models run
as CUDA graphs, so the draft is cheap and the target verifies K+1 tokens in one graphed forward
(`verify` graph). The draft proposal loop keeps tokens on-GPU (no per-step `.item()` sync) so the K
draft replays pipeline. Exact greedy (verify uses the same kernel as decode -> spec output == target
greedy, validated 96/96). Greedy + repetition_penalty==1.0 only; else single-token decode.

| 4B path                          | isolated tps | through server | vs vLLM (55.6) |
| -------------------------------- | ------------ | -------------- | -------------- |
| cudagraph target-only            | 49           | 45             | 0.81x          |
| cudagraph + spec (K=3, 0.6B draft) | 60.9       | 54.3           | 1.10x / 0.98x  |

So speculative decoding now **helps** (1.2-1.3x over target-only cudagraph) instead of being a 2.3x
loss as on the HF bridge — because the draft is graph'd and fast. K sweep (isolated): K=3 60.9 best,
K=4 59.5, K=6 59.0. Acceptance ~0.6-0.76, ~2.3-3.3 accepted/round. TTFT is higher (164ms: draft
prefill + first-request graph captures).

Frontier to clearly beat vLLM on 4B through the server: cut per-token serving overhead (the spec
gain shrinks 60.9->54.3 through streaming), graph-capture prefill (TTFT), and a paged decode kernel.

## 2026-06-26 — CUDA-graph custom Qwen3 executor BEATS vLLM on 0.6B

`SOLORT_EXECUTOR=cudagraph`: a hand-written, graph-friendly Qwen3 forward over SoloRT-owned static
KV (position read from a buffer, masked attention), with the single-token decode step captured once
into a CUDA graph and replayed per token. Single-stream, temp=0, through the full SoloRT server.

| model      | SoloRT cudagraph | vLLM v0.8.5 | SoloRT/vLLM | vs old SoloRT (~12 tps) |
| ---------- | ---------------- | ----------- | ----------- | ----------------------- |
| Qwen3-0.6B | 164.0 tps        | 91.1 tps    | **1.80x**   | 13.7x                   |
| Qwen3-4B   | 45.2 tps         | 55.6 tps    | 0.81x       | 4.1x                    |

Progression of the cudagraph executor (0.6B / 4B decode tps): manual-attn 108.9 / 42.9 ->
SDPA-flash 157.7 / 44.5 -> +fused QKV & gate/up GEMMs 164.0 / 45.2.

**We decisively beat vLLM single-stream decode on 0.6B (1.80x).** 4B is at 0.81x (4x over the old
HF path). The remaining 4B gap is kernel sophistication: vLLM's paged FlashAttention computes only
the live length, whereas our SDPA still scans graph_max_len each step (4B @ MAXLEN=256 -> 47.9 tps
vs 45.2 @ 1024), plus vLLM fuses RMSNorm/residual. Next frontier for 4B: a paged decode kernel
(FlashInfer BatchDecode) reading length from buffers inside the graph, and fused norms. TTFT also
lags (prefill is not graph-captured).

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
  gaps dominate). StaticCache was kept as groundwork (default off) as the precondition for
  CUDA-graph capture, then removed once the custom `cudagraph` executor (its own static KV)
  superseded the HF-path experiment.
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