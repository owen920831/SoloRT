"""Paged KV cache control plane and optional tensor storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solort.cache.page_allocator import PageAllocator


@dataclass(frozen=True)
class KVCacheConfig:
    num_layers: int = 42
    num_pages: int = 4096
    page_size: int = 16
    num_kv_heads: int = 8
    head_dim: int = 256
    dtype: str = "fp16"
    device: str = "cuda"
    layout: str = "NHD"
    allocate_tensors: bool = False


@dataclass(frozen=True)
class KVCacheMetadata:
    """FlashInfer-compatible page-table view for one small batch."""

    page_indptr: list[int]
    page_indices: list[int]
    last_page_len: list[int]
    slot_mapping: list[int]


@dataclass(frozen=True)
class KVCacheTransaction:
    """Pages reserved speculatively and either committed or rolled back.

    v1 transactions are page-granular: accepted tokens keep any newly allocated pages, while a
    rejected speculative branch releases only pages that were added for the candidate suffix.
    """

    pages: list[int]
    original_block_len: int


class PagedKVCache:
    """Paged KV cache with a real allocator and optional data-plane tensors.

    The allocator and metadata are always active. Tensor storage is allocated only when requested
    so CPU tests can exercise page semantics without importing torch or reserving GPU memory.
    """

    def __init__(self, config: KVCacheConfig | None = None) -> None:
        self.config = config or KVCacheConfig()
        self.allocator = PageAllocator(
            num_pages=self.config.num_pages,
            page_size=self.config.page_size,
        )
        self.kv_cache: Any | None = None
        self.k_cache: Any | None = None
        self.v_cache: Any | None = None
        if self.config.allocate_tensors:
            self.allocate_tensors()

    @property
    def has_tensors(self) -> bool:
        return self.kv_cache is not None or (self.k_cache is not None and self.v_cache is not None)

    def allocate_tensors(self) -> None:
        """Allocate a FlashInfer-friendly KV tensor.

        FlashInfer accepts either a single 5-D `[pages, 2, page, heads, dim]` tensor in NHD layout
        or separate K/V tensors. SoloRT keeps the single tensor by default because it is compact to
        pass through backend interfaces and mirrors the docs.
        """

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Tensor-backed KV cache requires torch") from exc

        dtype = self._torch_dtype(torch)
        if self.config.layout != "NHD":
            raise ValueError("SoloRT v1 tensor-backed KV cache supports only NHD layout")
        self.kv_cache = torch.empty(
            (
                self.config.num_pages,
                2,
                self.config.page_size,
                self.config.num_kv_heads,
                self.config.head_dim,
            ),
            dtype=dtype,
            device=self.config.device,
        )
        self.k_cache = self.kv_cache[:, 0]
        self.v_cache = self.kv_cache[:, 1]

    def pages_required(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        return (token_count + self.config.page_size - 1) // self.config.page_size

    def ensure_capacity(self, block_table: list[int], token_count: int) -> list[int]:
        required = self.pages_required(token_count)
        missing = required - len(block_table)
        if missing <= 0:
            return []
        pages = self.allocator.alloc(missing)
        block_table.extend(pages)
        return pages

    def metadata_for(
        self,
        block_table: list[int],
        *,
        token_count: int,
        positions: list[int] | None = None,
    ) -> KVCacheMetadata:
        """Build the page-table tensors/lists expected by paged attention backends."""

        active_pages = self.pages_required(token_count)
        page_indices = list(block_table[:active_pages])
        last_page_len = self.last_page_len(token_count)
        slot_mapping = self.slot_mapping(block_table, positions or [])
        return KVCacheMetadata(
            page_indptr=[0, len(page_indices)],
            page_indices=page_indices,
            last_page_len=[last_page_len],
            slot_mapping=slot_mapping,
        )

    def slot_mapping(self, block_table: list[int], positions: list[int]) -> list[int]:
        """Map logical token positions to flattened physical KV slots."""

        slots: list[int] = []
        for position in positions:
            logical_page = position // self.config.page_size
            if logical_page >= len(block_table):
                raise IndexError(
                    f"position {position} needs logical page {logical_page}, "
                    f"but block table has {len(block_table)} pages"
                )
            page_offset = position % self.config.page_size
            slots.append(block_table[logical_page] * self.config.page_size + page_offset)
        return slots

    def last_page_len(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        remainder = token_count % self.config.page_size
        return remainder if remainder else self.config.page_size

    def begin_append_transaction(
        self,
        block_table: list[int],
        *,
        current_token_count: int,
        append_token_count: int,
    ) -> KVCacheTransaction:
        """Reserve pages for a speculative append without making rollback ambiguous."""

        if append_token_count < 0:
            raise ValueError("append_token_count cannot be negative")
        original_len = len(block_table)
        self.ensure_capacity(block_table, current_token_count + append_token_count)
        return KVCacheTransaction(
            pages=list(block_table[original_len:]),
            original_block_len=original_len,
        )

    def commit_transaction(self, transaction: KVCacheTransaction) -> None:
        """Keep pages reserved by a speculative append."""

        del transaction

    def rollback_transaction(
        self,
        block_table: list[int],
        transaction: KVCacheTransaction,
    ) -> None:
        """Release pages added by a failed speculative branch."""

        if transaction.pages:
            del block_table[transaction.original_block_len :]
            self.allocator.free(transaction.pages)

    def snapshot(self) -> dict[str, int | str]:
        allocator = self.allocator.snapshot()
        return {
            **allocator,
            "num_layers": self.config.num_layers,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "layout": self.config.layout,
            "tensor_storage": "allocated" if self.has_tensors else "metadata_only",
        }

    def _torch_dtype(self, torch: Any) -> Any:
        aliases = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            return aliases[self.config.dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported KV cache dtype {self.config.dtype!r}") from exc
