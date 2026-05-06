"""FlashInfer backend placeholder."""

from __future__ import annotations

from solort.core.batch import Batch


class FlashInferBackend:
    name = "flashinfer"

    def prepare_metadata(self, batch: Batch, kv_meta: object | None = None) -> dict[str, object]:
        raise NotImplementedError("FlashInfer integration is reserved for the real-model path")

    def forward(self, q: object, k: object, v: object, metadata: dict[str, object]) -> object:
        raise NotImplementedError("FlashInfer integration is reserved for the real-model path")
