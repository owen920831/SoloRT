"""Attention backend interfaces."""

from __future__ import annotations

from typing import Protocol

from solort.core.batch import Batch


class AttentionBackend(Protocol):
    name: str

    def prepare_metadata(self, batch: Batch, kv_meta: object | None = None) -> dict[str, object]:
        """Prepare backend-specific metadata for a batch."""

    def forward(self, q: object, k: object, v: object, metadata: dict[str, object]) -> object:
        """Run attention and return backend-specific output."""
