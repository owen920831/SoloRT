"""Model loader placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dtype: str = "auto"
    device: str = "cuda"
    text_only: bool = True


def load_model_config(model_id: str = "Qwen/Qwen3-0.6B") -> ModelConfig:
    return ModelConfig(model_id=model_id)
