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

### 2026-06-25 — P1: draft KV cache (in progress)

- Reviewed the runtime end to end; confirmed the WIP (KV mirror + FlashInfer torch fallback +
  bench sampling params + Makefile live-mounts) passes `make docker-test` (39 passed, 1 skipped)
  and `make docker-lint`. Checkpointed it on branch `feat/draft-kv-cache`.
- Next: incremental draft KV cache + CPU unit tests, then a GPU spec-vs-nospec experiment that
  also verifies greedy-exactness (temp=0 speculative output must be token-identical to baseline).
