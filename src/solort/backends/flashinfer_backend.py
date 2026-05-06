"""FlashInfer attention backend adapter."""

from __future__ import annotations

from typing import Any

from solort.cache.kv_cache import KVCacheMetadata, PagedKVCache
from solort.core.batch import Batch


class FlashInferBackend:
    name = "flashinfer"

    def __init__(self) -> None:
        self._flashinfer: Any | None = None

    def prepare_metadata(
        self,
        batch: Batch,
        kv_meta: KVCacheMetadata | None = None,
    ) -> dict[str, object]:
        """Return page metadata as `int32` tensors when torch is available.

        FlashInfer page-table arrays must be int32. Keeping conversion here prevents the scheduler
        and batch builder from learning backend-specific dtype rules.
        """

        if kv_meta is None:
            kv_meta = KVCacheMetadata(
                page_indptr=batch.page_indptr or [0, len(batch.page_indices or [])],
                page_indices=batch.page_indices or [],
                last_page_len=batch.last_page_len or [],
                slot_mapping=batch.slot_mapping or [],
            )
        metadata: dict[str, object] = {
            "page_indptr": kv_meta.page_indptr,
            "page_indices": kv_meta.page_indices,
            "last_page_len": kv_meta.last_page_len,
            "slot_mapping": kv_meta.slot_mapping,
        }

        try:
            import torch
        except ImportError:
            return metadata

        metadata.update(
            {
                "page_indptr_tensor": torch.tensor(kv_meta.page_indptr, dtype=torch.int32),
                "page_indices_tensor": torch.tensor(kv_meta.page_indices, dtype=torch.int32),
                "last_page_len_tensor": torch.tensor(kv_meta.last_page_len, dtype=torch.int32),
                "slot_mapping_tensor": torch.tensor(kv_meta.slot_mapping, dtype=torch.int32),
            }
        )
        return metadata

    def forward(self, q: object, k: object, v: object, metadata: dict[str, object]) -> object:
        raise NotImplementedError(
            "FlashInfer kernels are wired at the metadata boundary; full Qwen layer execution "
            "lands in the paged executor milestone."
        )

    def build_decode_wrapper(
        self,
        kv_cache: PagedKVCache,
        *,
        workspace_bytes: int = 128 * 1024 * 1024,
        use_cuda_graph: bool = False,
    ) -> object:
        """Create FlashInfer's paged decode wrapper for tensor-backed serving."""

        flashinfer = self._load_flashinfer()
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("FlashInfer backend requires torch") from exc
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=kv_cache.config.device)
        return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout=kv_cache.config.layout,
            use_cuda_graph=use_cuda_graph,
        )

    def _load_flashinfer(self) -> object:
        if self._flashinfer is not None:
            return self._flashinfer
        try:
            import flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "FlashInfer backend requires `flashinfer-python`; install the `flashinfer` extra."
            ) from exc
        self._flashinfer = flashinfer
        return flashinfer
