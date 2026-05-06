"""Mock eager attention backend."""

from __future__ import annotations

from solort.core.batch import Batch


class EagerAttentionBackend:
    name = "eager"

    def prepare_metadata(self, batch: Batch, kv_meta: object | None = None) -> dict[str, object]:
        return {
            "batch_id": batch.batch_id,
            "phase": batch.phase.value,
            "positions": batch.positions,
            "kv_meta": kv_meta,
        }

    def forward(self, q: object, k: object, v: object, metadata: dict[str, object]) -> object:
        return {
            "q": q,
            "k": k,
            "v": v,
            "metadata": metadata,
        }
