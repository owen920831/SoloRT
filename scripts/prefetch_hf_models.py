"""Prefetch SoloRT's default Hugging Face models into the mounted cache."""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Honor the generic MODEL/DRAFT_MODEL knobs first, falling back to the Qwen defaults, so the
    # same prefetch step works for any model pairing. Empty entries (e.g. spec disabled) are
    # skipped.
    models = [
        os.getenv("MODEL") or os.getenv("QWEN4B_MODEL", "Qwen/Qwen3-4B"),
        os.getenv("DRAFT_MODEL") or os.getenv("QWEN06B_MODEL", "Qwen/Qwen3-0.6B"),
    ]
    for model_id in dict.fromkeys(model for model in models if model):
        print(
            f"Prefetching {model_id} into HF_HOME={os.getenv('HF_HOME')} "
            f"(fast_transfer={os.getenv('HF_HUB_ENABLE_HF_TRANSFER')})",
            flush=True,
        )
        path = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            max_workers=int(os.getenv("HF_PREFETCH_WORKERS", "4")),
        )
        print(f"Cached {model_id} at {path}", flush=True)
    print("Hugging Face model cache is ready.")


if __name__ == "__main__":
    main()
