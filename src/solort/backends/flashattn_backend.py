"""FlashAttention backend placeholder."""

from __future__ import annotations

from solort.core.batch import Batch


class FlashAttentionBackend:
    name = "flashattn"

    def prepare_metadata(self, batch: Batch, kv_meta: object | None = None) -> dict[str, object]:
        raise NotImplementedError("FlashAttention integration is reserved for a later backend")

    def forward(self, q: object, k: object, v: object, metadata: dict[str, object]) -> object:
        raise NotImplementedError("FlashAttention integration is reserved for a later backend")
