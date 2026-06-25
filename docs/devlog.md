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

## Backlog / roadmap (ranked by measured value)

1. **Draft KV cache** (IN PROGRESS) — `_draft_tokens` re-runs the full prefix every draft step
   (`use_cache=False`), so proposing K tokens costs ~K x prefix_len. This is why speculative
   decoding shows ~no speedup (spec 12.15 vs nospec 12.13 tps). Give the draft model an
   incremental KV cache so a proposal round costs ~accepted tokens, not the whole prefix.
2. KV-mirror numerical validation — confirm the mirrored per-layer paged tensor matches HF
   `past_key_values` so the tensor-backed paged runner can later trust it.
3. Tensor-backed paged executor — replace the HF `past_key_values` bridge with a Qwen layer
   runner that reads/writes SoloRT pages directly (the long-term Phase 3 milestone).

## Log

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
