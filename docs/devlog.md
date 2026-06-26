# SoloRT Dev Log

A running log of backend work on the single-GPU interactive runtime. Newest entries on top.

## Goal

Single-machine inference for consumer NVIDIA GPUs (RTX 4080 16 GB), Qwen3-4B target +
Qwen3-0.6B greedy speculative draft. Keep improving the backend, push periodically, and run
GPU experiments while the card is idle.

## Standing facts (environment)

- Host conda PyTorch is built for CUDA 13 but the driver is 12.6, so `torch.cuda` is unavailable
  on the host. **GPU work runs inside the NGC image** (`solort:qwen3-4b-spec-ngc`).
- CPU unit tests + lint run on the host-mountable `solort:dev` image:
  `make docker-test` / `make docker-lint`.
- The Makefile live-mounts `src/`, `docs/`, `scripts/` into the serving containers, so Python
  changes do **not** require rebuilding the 20 GB NGC image — restart the container instead.
- GPU experiments: `make docker-ngc-up` (spec, port 8000) + `make docker-ngc-up-nospec`
  (baseline) then `benchmarks/bench_serving.py`.

## Active goal (2026-06-26): beat vLLM on single-stream, finish the paged/fast Qwen path

Target: single-user, single-stream decode latency that beats vLLM for Qwen3 on one RTX 4080, then
keep iterating. Established that we are **launch-bound** (~11 tps regardless of model size), so the
primary lever is **CUDA graphs**, not paged memory management.

Plan (iterative, measure every step against the vLLM baseline):
1. Baseline: run vLLM single-stream on the same 4080 + same bench (the number to beat).
2. Probe: quantify CUDA-graph upside — eager vs `torch.compile(reduce-overhead)` + StaticCache.
3. Build a CUDA-graph decode path in SoloRT (compilable attention + static cache + captured graph),
   measure vs vLLM, iterate.
4. Layer the SoloRT paged KV (the "my direction" executor) under the fast forward for
   memory/long-context, keeping the speed win.
5. Fused kernels / further wins as needed.

## Backlog / roadmap (ranked by measured value)

1. **Draft KV cache** (IN PROGRESS) — `_draft_tokens` re-runs the full prefix every draft step
   (`use_cache=False`), so proposing K tokens costs ~K x prefix_len. This is why speculative
   decoding shows ~no speedup (spec 12.15 vs nospec 12.13 tps). Give the draft model an
   incremental KV cache so a proposal round costs ~accepted tokens, not the whole prefix.
2. KV-mirror numerical validation — (read path DONE: `gather_layer_tokens` + store/gather
   round-trip test). Remaining: GPU check that mirrored paged K/V matches HF `past_key_values`.
3. Tensor-backed paged executor — replace the HF `past_key_values` bridge with a Qwen layer
   runner that reads/writes SoloRT pages directly (the long-term Phase 3 milestone). IN PROGRESS.

## Log

### 2026-06-26 — Optimization pass 2: prefill graph, grouped attention; quant frontier

- **Graphed bucketed prefill**: TTFT 74ms -> 20ms (prefill was eager / launch-bound). Beats vLLM.
- **Grouped decode attention** (drop the GQA repeat_interleave; read cache as [nkv,bound,hd]):
  4B decode 55 -> 67 tps (1.21x vLLM) at 1024 context; also fixes the large-KV-buffer slowdown.
- Exact state now: 0.6B ~150-180 tps & 11-13ms TTFT (1.6-2.0x vLLM); 4B 67 tps & 27ms TTFT (1.21x).
  SoloRT beats vLLM on both models, decode AND TTFT, exact greedy.

**Quantization frontier (thoroughly investigated, no fast path on torch 2.4 + Ada):** batch-1 decode
is weight-memory bound (~73% of the bf16 roofline) so smaller weights are the next big lever. Tried
all three avenues, all blocked:
- `torch._weight_int8pack_mm`: CPU-only (no CUDA kernel in 2.4).
- `torch._scaled_mm` fp8: per-tensor scalar scales only in 2.4 (per-row needs >=2.5) -> ~94% error.
- **torchao int8_weight_only**: accurate (0.5% error) but **2-5x SLOWER** on the 4080 (no tuned
  Ada/torch-2.4 kernel; it doesn't realize the memory-bandwidth saving here).

So there is no working *fast* quantization kernel on this hardware/software, and the exact bf16 path
(which already beats vLLM) is the practical optimum here. Exceeding the bf16 roofline would need a
newer-torch image with a tuned Ada int8/fp8 decode kernel (e.g. Marlin/Machete-style) -- a major
infra project, non-exact, and with uncertain payoff on Ada. Conclusion: the exact single-stream
runtime is optimized to its practical limit on this setup.

### 2026-06-26 — Speculative decoding on the cudagraph runner (4B)

Added exact greedy speculative decoding to the cudagraph executor (`SpecCudaGraphQwen3Executor`,
`SOLORT_EXECUTOR=cudagraph SOLORT_SPECULATIVE_TOKENS=K`): 0.6B draft + 4B target, both CUDA-graphed.
The draft proposes K tokens (graph decode, kept on-GPU to avoid per-step sync), the target verifies
K+1 tokens in one graphed `verify` forward, accept the longest matching prefix + correction/bonus.

Because verify uses the same kernel as decode, spec output is EXACT vs target greedy (96/96) — the
HF bridge couldn't do this (prefill vs decode kernels diverged). And because the draft is graph'd
(fast), spec now HELPS instead of being a 2.3x loss:
- 4B isolated: target-only 49 -> spec(K=3) 60.9 tps (1.10x vLLM, 1.27x target-only).
- 4B through server: 45 -> 54.3 tps (0.98x vLLM, 1.20x target-only). Exact, coherent.

Greedy + rep_penalty==1.0 only (else single-token). Frontier: serving-overhead per token (spec
gain shrinks through streaming), TTFT (graph-capture prefill), paged decode to beat 4B cleanly.

### 2026-06-26 — CUDA-graph custom Qwen3 executor BEATS vLLM on 0.6B

Built the custom runner (`SOLORT_EXECUTOR=cudagraph`, `src/solort/model/cuda_graph_executor.py`):
a hand-written graph-friendly Qwen3 forward over SoloRT-owned static KV (position from a buffer,
SDPA attention, fused QKV + gate/up GEMMs), with the single-token decode captured once into a CUDA
graph and replayed per token. Validated numerically correct (greedy == HF to bf16 precision).

Single-stream decode, through the full server (records.md):
- **Qwen3-0.6B: 164 tps vs vLLM 91 -> 1.80x (BEATS vLLM), ~13.7x the old HF path.**
- Qwen3-4B: 45 tps vs vLLM 56 -> 0.81x, 4.1x old path.

This realizes the launch-bound diagnosis: CUDA graphs eliminate the per-token kernel-launch cost
that capped the HF bridge at ~12 tps and that torch.compile-on-HF could not fix. Iteration that got
here: manual-attn 108.9/42.9 -> SDPA 157.7/44.5 -> fused GEMMs 164.0/45.2.

Open frontier: 4B (needs paged FlashAttention computing only the live length + fused norms to pass
vLLM), TTFT (prefill not yet graph-captured), and multi-sequence (currently single active sequence
= the single-user target). Enable with `SOLORT_EXECUTOR=cudagraph SOLORT_GRAPH_MAX_LEN=...`.

### 2026-06-26 — torch.compile on HF Qwen3 is a dead end; custom runner required

Tested torch.compile every way to get CUDA graphs onto the HF model:
- torch 2.4 (NGC): reduce-overhead recompile-thrashes -> 0.6 tok/s.
- torch 2.6 (cu124) + transformers latest: `NameError` in transformers `output_capturing.py` under
  dynamo (a transformers+compile bug).
- torch 2.6 + transformers==4.51.3 (no error), all modes SLOWER than eager: reduce-overhead 5.1,
  default 3.8, max-autotune-no-cudagraphs 4.0 tok/s vs eager ~15-27.

Root cause: HF's modeling code is data-dependent (`.item()` on `cache_position`, dynamic mask
build), so dynamo guards on the changing position and re-captures/recompiles every decode step.
The same data-dependence would bake the write position into a hand-captured `torch.cuda.CUDAGraph`,
so manual capture of the HF forward fails too.

**Conclusion:** vLLM/nano-vllm are fast because they run their OWN graph-friendly model
implementation (static shapes, position read from a buffer, paged-attention kernel) and CUDA-graph
it. To match/beat vLLM, SoloRT needs the same: a custom Qwen3 decode forward (the Phase 3/4
tensor-backed paged executor), not the HF bridge. Reverted the `use_compile` experiment (disproven).
This is the next build; first milestone = a custom decode forward whose logits match HF for one
step, then paged attention + manual CUDA-graph capture.

### 2026-06-26 — Toward beating vLLM: diagnosis + StaticCache groundwork

Goal: beat vLLM single-stream for Qwen3 on one 4080, finish the fast Qwen path, keep iterating.

Findings this iteration (all measured on the 4080):
- **CUDA-graph probe** (isolated, Qwen3-0.6B decode): eager+DynamicCache 11.9 tok/s; eager+
  StaticCache **27 tok/s** (2.3x); `torch.compile(reduce-overhead)` 0.6 tok/s — torch 2.4 (the NGC
  image) **recompile-thrashes**, so torch.compile is a dead end on that image.
- **StaticCache through SoloRT's serving stack does NOT help** (clean isolated A/B: 0.6B dynamic
  15.5 vs static 13.0; 4B dynamic 12.1 vs static 10.9). The isolated 2.3x evaporates because the
  per-token serving gaps (async/HTTP/detokenize) dominate and a faster forward just leaves the GPU
  idle/downclocked longer. So StaticCache is kept as **groundwork (default OFF)** — it is the
  precondition for CUDA-graph capture, not a standalone win.
  *(Update: superseded and removed — the custom `cudagraph` executor owns its own static KV, so the
  HF-path StaticCache experiment was deleted rather than carried as dead groundwork.)*
- **Benchmark noise**: batch-1 launch-bound decode swings ~±30% (the same "dynamic" measured
  11.5-15.5 across runs) because it is sensitive to GPU clock state and CPU contention. Lesson:
  isolate runs (nothing else on CPU/GPU), warm up, and prefer locked clocks; small deltas are
  noise.
- **vLLM works on this driver** (v0.8.5.post1, CUDA 12.4 — `latest` needs CUDA 13.0 which the 12.6
  driver rejects). Its speed is `torch.compile` + **CUDA graphs** (`use_cudagraph:true`), which need
  a newer torch than the NGC image's 2.4.

Conclusion / next lever: the only way to close the launch-bound gap is CUDA graphs. Since torch 2.4
thrashes, the next iteration builds a SoloRT image on torch >=2.6 (cu124, driver-compatible) and
enables `torch.compile(reduce-overhead)` + StaticCache on the decode path, measured vs the vLLM
baseline.

### 2026-06-26 — Chunked greedy decode (0.6B +7%) + dead-code removal

Cleanup: removed the off-by-default StaticCache HF-path experiment (superseded by the cudagraph
executor's own static KV) and several zero-reference stubs (`flashattn_backend`, `attention_base`,
`loader`, `flashinfer_backend` + test, a dead prefix-cache field).

Optimization: with the eager argmax already on-GPU, the residual gap was pinned to per-token Python.
A runner microbench (`scripts/microbench_decode.py`) showed the per-token `.item()` sync is only
~10% (0.6B 217 vs 239 tps tight) — the rest is the **per-scheduler-tick** Python paid once per
`forward_decode`. So `SOLORT_DECODE_CHUNK=K` returns K greedy tokens per decode step, pipelining K
graph replays back-to-back on the stream (`decode_gpu_argmax`, on-GPU argmax, no inter-replay sync)
and syncing once. Measured: 0.6B 149 -> 160 tps at K=4 (+7%, peak; K=8 regresses), 4B neutral (19
ms/token GPU dwarfs the fixed cost). Exact greedy (bit-identical completions). Default K=4. Numbers
in [../records.md](../records.md).

### 2026-06-26 — Code simplification pass

Ran a fan-out review (one finder per source file) + synthesis; applied the 16 verified-safe,
behavior-preserving findings (net -32 source lines), all gated by the test suite + lint:

- Dead code removed: `_prefix_entries_by_seq`, `SchedulerSnapshot`, `NotImplementedPayload`,
  `Sampler` Protocol, `Session.{pinned_prefix_ids,metadata,last_access_ts}`,
  `SequenceStatus.PAUSED`, `Sequence.kv_precision`, `Batch.padded_batch_size`,
  hand-rolled `_null_context` (use stdlib `nullcontext`), unreachable `_draft_device` None branch.
- Dedup: `build_default_runtime` branches (pick class + default backend, build config once),
  the SSE chunk envelope in `streaming.py`, the FlashInfer fn lookup (call once), and the
  `avg_tpot`/`avg_itl` computation in metrics.
- `_attach_kv_metadata` returns None (callers ignored it); FlashInfer counters now count on
  success instead of increment-then-decrement-on-failure.

Deferred to `consider` (riskier / judgment calls, not applied): deleting unused backend
scaffolding files (`flashattn_backend.py`, `attention_base.py`, parts of `flashinfer_backend.py`),
the public `QwenTransformersExecutor`, scheduler `token_budget`/priority-write cleanups, and
several cache micro-simplifications.

### 2026-06-25 — Single-stream roofline (vs nano-vllm): we are launch-bound

Benchmarked single-stream decode at two model sizes (records.md). Qwen3-0.6B and Qwen3-4B both
decode at ~11 tps with ~180 ms TTFT for the same prompt — **decode speed is independent of model
size**, so the runtime is **kernel-launch / Python-overhead bound**, not compute/bandwidth bound
(~2% of roofline on 0.6B, ~12% on 4B). The eager HF forward + per-layer Python FlashInfer bridge
launches hundreds of tiny kernels per token with no CUDA graph. So: **not at nano-vllm/vLLM class**
(they use paged attention + CUDA graphs + batched throughput). The fix is the tensor-backed paged
executor + CUDA graphs — the standing milestone. Decode tps will stay ~11 regardless of model until
then.

### 2026-06-25 — GPU validation of KV mirror + finish the spec-off default

Ran a GPU smoke test (spec off, `KV_TENSOR_STORAGE=1`) on the idle 4080:
- Server starts clean and serves a coherent answer (`finish_reason=stop`) — the fallback fix and
  per-layer KV tensor don't break real serving.
- KV mirror fires end-to-end: `mirrored_tokens=2124, mirror_skipped=0, tensor_storage=allocated`
  (~60 tokens x 36 layers). FlashInfer active (prefill=71, decode=1296, fallback=1).

The test also caught that the **Dockerfiles/compose baked `ENV SOLORT_SPECULATIVE_TOKENS=4`**,
which overrode the code default, so raw `docker run`/compose were still spec-on. Flipped those to
`0` (Dockerfile.ngc, Dockerfile x2, docker-compose.yml). `make docker-ngc-up` already passes
`-e ...=$(SPEC_TOKENS)=0` so the make path was already spec-off; raw `docker run` of an
already-built image needs a rebuild to pick up the new baked default.

### 2026-06-25 — Paged executor step 1: KV read path + store/gather round-trip

Toward the tensor-backed paged executor (one attention path for generate + validate). Added
`PagedKVCache.gather_layer_tokens` — the read counterpart to `store_layer_tokens` — returning paged
K/V in NHD `[tokens, heads, head_dim]` via a single `index_select` (slot == flat `[pages, page]`
index). A torch round-trip test proves `gather(store(x)) == x` and per-layer isolation. This is the
contract the paged executor needs: it can feed attention from SoloRT-owned KV instead of HF
`past_key_values`. Next: wire a forward that reads from the paged tensor and compare logits against
the HF path on GPU, then replace the `past_key_values` bridge layer by layer.

### 2026-06-25 — Decision: speculation off by default + fix SDPA fallback

User picked "paged executor + spec off" after the P2 result. Actions:

- **Speculation is now opt-in** (`speculative_tokens=0` in code/env defaults; `SPEC_TOKENS ?= 0`
  in the Makefile; `docker-ngc-up` follows it). Re-enable with `SPEC_TOKENS=4` or the
  `docker-ngc-up-qwen4b` demo target. README + records.md explain why.
- **Fixed a HIGH-severity bug** an adversarial review surfaced: the torch-SDPA fallback used
  `is_causal=True` (top-left aligned), which corrupts any forward over a populated KV cache
  (chunked-prefill 2nd+ chunks, speculative validation) on hosts without flashinfer. Now uses an
  explicit bottom-right mask, matching FlashInfer. The review confirmed the draft KV cache itself
  is correct.
- Next: the tensor-backed paged executor (one attention path for generate + validate). First
  concrete step is KV-mirror numerical validation — confirm the mirrored paged tensor matches HF
  `past_key_values` so the executor can trust SoloRT-owned KV.

### 2026-06-25 — P2: GPU experiment result (spec is net-negative here)

Ran spec (Qwen3-4B + 0.6B draft, K=4) vs nospec on the idle RTX 4080, temp=0, 220 tokens, 4 runs
each. Full numbers in `records.md`. Summary:

- **spec 5.03 tps vs nospec 11.53 tps -> spec is 2.3x SLOWER.**
- The incremental draft KV cache works: only 2718 draft-forward-tokens for 1100 generated tokens
  (~5.6/round). The old full-prefix loop would have fed orders of magnitude more. So P1 did its
  job; the loss is elsewhere.
- Acceptance is only 31.6% for this draft/content. Because an incremental cache yields proposals
  identical to the full-prefix loop, this is the inherent acceptance, not a cache bug.
- **spec output is NOT greedy-exact vs nospec** (diverges at char 55, deterministically). Emitted
  tokens come from the target validation forward (FlashInfer `single_prefill`, q_len=K+1), whose
  logits differ from the `single_decode` kernel used in plain decode -> early argmax flip. This is
  independent of the draft cache (target path unchanged) and is a property of the HF+FlashInfer
  bridge.

Why spec loses: each round costs 1 target prefill-forward (K+1 tokens) + ~5 draft forwards (each
with a GPU->CPU `.item()` sync) to emit ~2.3 tokens at 31.6% acceptance, vs nospec's 1 decode
forward per token. The draft is cheaper now, but the per-round target prefill-forward + Python/sync
overhead is not recovered.

**Recommendation / next decisions (surface to user):**
1. Disable speculation by default for this pairing (`SOLORT_SPECULATIVE_TOKENS=0`) — it currently
   hurts latency and exactness.
2. The real unlock is the tensor-backed paged executor: one paged-decode attention path for both
   generation and validation -> exactness + no HF `past_key_values` juggling + cheaper validation.
   This is the standing Phase 3/4 milestone and what the KV-mirror WIP is building toward.
3. Cheaper experiments worth trying meanwhile: smaller K (2), a stronger/closer draft, and removing
   per-draft-token GPU->CPU syncs.

### 2026-06-25 — Enable other LLMs

The runtime was already model-agnostic (target picked via `SOLORT_MODEL_ID`; the paged KV layout
is derived from `model.config`). The gaps were guardrails + ergonomics, not architecture:

- **Vocab guard**: greedy speculative decoding compares draft/target token ids directly, so a
  mismatched draft pair would emit silently wrong tokens. `_load_draft_model` now disables
  speculation (with a logged warning) when draft and target `vocab_size` differ.
- **Generic Make target**: `make docker-ngc-up-model MODEL=... DRAFT_MODEL=... SPEC_TOKENS=...
  ATTENTION_BACKEND=...` serves any HF causal LM without remembering Qwen-named vars.
- **Prefetch** honors the generic `MODEL`/`DRAFT_MODEL` knobs and skips empty entries.
- **README** "Running Other Models": env-var table + constraints (shared-vocab requirement for
  speculation; use `ATTENTION_BACKEND=sdpa` for Gemma2 soft-capping / DeepSeek MLA that the
  FlashInfer bridge does not model exactly).

### 2026-06-25 — P1: draft KV cache (in progress)

- Reviewed the runtime end to end; confirmed the WIP (KV mirror + FlashInfer torch fallback +
  bench sampling params + Makefile live-mounts) passes `make docker-test` (39 passed, 1 skipped)
  and `make docker-lint`. Checkpointed it on branch `feat/draft-kv-cache`.
- Next: incremental draft KV cache + CPU unit tests, then a GPU spec-vs-nospec experiment that
  also verifies greedy-exactness (temp=0 speculative output must be token-identical to baseline).
