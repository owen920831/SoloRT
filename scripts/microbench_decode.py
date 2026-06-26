"""Isolate the cudagraph decode ceiling vs the per-token serving gap.

Measures runner-level decode three ways for Qwen3 on CUDA:
  A. per-token sync   -- decode_argmax + .item() every token (today's serving inner loop)
  B. chunked (k=4)    -- decode_gpu_argmax back-to-back, one .item() per chunk
  C. tight no-sync    -- N replays back-to-back, single .item() at the very end (GPU ceiling)

If C >> A, the per-token GPU<->CPU flush (and the serving gap it stands in for) is wasting GPU
headroom, and chunking the server's decode loop should recover it. If C ~= A, decode is already
GPU-saturated (memory-bound weights) and only quantization moves it.

Run inside the NGC image, e.g.:
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HF_HOME=/root/.cache/huggingface -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $PWD/src:/app/src -v $PWD/scripts:/app/scripts solort:qwen3-4b-spec-ngc \
    python scripts/microbench_decode.py Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from solort.model.cuda_graph_executor import _GraphQwen3Runner


def _time(fn, iters: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return iters / (time.perf_counter() - t0)


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    max_len = 1024

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to("cuda").eval()
    runner = _GraphQwen3Runner(model, max_len)
    dev = runner.dev

    prompt = tok("Explain how CUDA graphs reduce kernel launch overhead.", return_tensors="pt")
    ids = prompt.input_ids[0].to(dev)
    start = int(ids.shape[0])

    with torch.no_grad():
        runner.prefill(ids)
        first = int((runner._pre_hidden[next(b for b in runner.pre_buckets if b >= start)]
                     [start - 1:start] @ runner.lm_head.T).argmax(-1).item())

        # Warm: force-capture every decode bucket we will touch (capture cost must not pollute timing).
        cur = first
        for pos in range(start, start + n + k + 2):
            cur = runner.decode_argmax(cur, pos)

        def run_a() -> None:
            cur = first
            for pos in range(start, start + n):
                cur = runner.decode_argmax(cur, pos)

        def run_b() -> None:
            cur = torch.tensor([first], dtype=torch.long, device=dev)
            pos = start
            for _ in range(n // k):
                toks = []
                for i in range(k):
                    cur = runner.decode_gpu_argmax(cur, pos + i)
                    toks.append(cur)
                pos += k
                _ = torch.cat(toks).tolist()  # one sync per chunk
                cur = toks[-1]

        def run_c() -> None:
            cur = torch.tensor([first], dtype=torch.long, device=dev)
            for pos in range(start, start + n):
                cur = runner.decode_gpu_argmax(cur, pos)
            _ = cur.item()  # single sync at the very end

        a = _time(run_a, n)
        b = _time(run_b, n)
        c = _time(run_c, n)

    print(f"model={model_id}  n={n}  chunk={k}  start_pos={start}")
    print(f"A per-token sync   : {a:7.1f} tok/s  ({1000 / a:5.2f} ms/tok)")
    print(f"B chunked (k={k})    : {b:7.1f} tok/s  ({1000 / b:5.2f} ms/tok)   {b / a:.2f}x vs A")
    print(f"C tight no-sync    : {c:7.1f} tok/s  ({1000 / c:5.2f} ms/tok)   {c / a:.2f}x vs A")
    print(f"-> sync/gap headroom (C/A): {c / a:.2f}x   chunk recovers (B/A): {b / a:.2f}x")


if __name__ == "__main__":
    main()
