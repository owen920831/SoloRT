# SoloRT Architecture

SoloRT is a single-user, single-GPU runtime optimized for interactive local chat on consumer NVIDIA
GPUs. It has two execution paths behind one OpenAI-compatible API and scheduler:

- **`cudagraph` (fast path)** — a custom, CUDA-graph Qwen3 forward that beats vLLM single-stream
  (see below). Single active sequence, Qwen3-family + CUDA, exact greedy.
- **`paged` / `transformers` (general path)** — a HuggingFace-Transformers bridge with SoloRT
  scheduling, paged-KV metadata, prefix cache, and a FlashInfer attention option. Works for any HF
  causal LM. The rest of this document describes this path's KV/scheduling machinery, which the
  fast path bypasses with its own static KV.

## CUDA-Graph Fast Path

The interactive batch-1 decode is kernel-launch / weight-memory bound, not compute bound, so the
fast path removes per-token CPU and launch overhead:

```mermaid
flowchart TD
    P[Prompt] --> PF[Graphed bucketed prefill<br/>pad to length bucket, one captured causal graph]
    PF --> KV[SoloRT-owned static KV per layer]
    PF --> T0[first token]
    T0 --> DEC[Graphed bucketed decode<br/>scan only live length]
    DEC --> AM[On-GPU greedy argmax in-graph]
    AM --> TOK[token]
    TOK --> DEC
    KV --> DEC
```

Key techniques (`src/solort/model/cuda_graph_executor.py`): CUDA-graph capture of prefill and the
single-token decode (bucketed by length); on-GPU argmax inside the graph (no eager vocab argmax);
grouped-query attention without materializing the GQA-expanded KV (no `repeat_interleave`); fused
QKV / gate-up GEMMs; incremental detokenization. Numbers in [../records.md](../records.md).

## Serving Data Flow (general `paged` path)

```mermaid
flowchart TD
    A[OpenAI-style Request] --> B[Session Manager]
    B --> C[Sequence]
    C --> D[Interactive Scheduler]
    D -->|prefill chunk| E[Batch Builder]
    D -->|decode token| E
    E --> F[Paged KV Cache]
    F --> G[slot_mapping]
    F --> H[page_indptr / page_indices]
    F --> I[last_page_len]
    G --> J[Attention Backend]
    H --> J
    I --> J
    J --> K[Qwen3-4B Target Executor]
    K --> L[Sampler]
    L --> M[SSE Stream]
```

## Foreground-First Scheduling

```mermaid
flowchart LR
    A[Foreground Decode] --> S[Scheduler]
    B[Foreground Prefill] --> S
    C[Branch Decode] --> S
    D[Background Prefill] --> S
    E[Cache Maintenance] --> S

    S --> F{Priority}
    F -->|P0| G[decode one token]
    F -->|P1| H[prefill bounded chunk]
    F -->|P3| I[background leftover budget]
```

Decode is intentionally scheduled ahead of prefill. This protects inter-token latency when a long
background prompt is being prefetched.

## Paged KV Layout

```mermaid
flowchart LR
    A[Logical tokens] --> B[Logical pages]
    B --> C[Sequence block_table]
    C --> D[Physical pages]
    D --> E[KV tensor]

    subgraph Metadata
        F[slot_mapping]
        G[page_indptr]
        H[page_indices]
        I[last_page_len]
    end

    C --> F
    C --> G
    C --> H
    C --> I
```

SoloRT's tensor-backed layout is FlashInfer-friendly:

```text
kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
```

The `2` dimension stores key then value. The control plane works without tensor allocation so unit
tests can run on CPU-only machines.

## Greedy Speculative Decoding

```mermaid
sequenceDiagram
    participant R as Runtime
    participant D as Draft 0.6B
    participant KV as Paged KV
    participant T as Target 4B
    participant O as Output

    R->>D: generate up to K draft tokens
    D-->>R: candidates
    R->>KV: begin append transaction
    R->>T: validate candidates in target pass
    T-->>R: target greedy choices
    alt all accepted
        R->>KV: commit transaction
        R->>O: stream accepted tokens + recovery token
    else partial accepted
        R->>KV: rollback rejected provisional pages
        R->>O: stream accepted prefix + target correction
    end
```

v1 speculative decoding is enabled only for deterministic decoding (`temperature=0`). Sampling
support needs stochastic acceptance logic and is intentionally deferred.

## Benchmark Surface

```mermaid
flowchart TD
    A[bench_serving.py] --> B[GPU 4B baseline]
    A --> C[GPU 4B + 0.6B speculative]
    A --> D[CPU fallback]
    B --> E[TTFT]
    C --> E
    D --> E
    B --> F[ITL / TPOT / TPS / TTOT]
    C --> F
    D --> F
    C --> G[acceptance rate]
    C --> H[proposed / accepted / rejected tokens]
```

The metrics endpoint exposes runtime latency counters, page usage, prefix cache hit/miss counts,
and executor speculative counters.
