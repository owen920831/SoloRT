"""Paged KV cache metadata for the Python MVP."""

from __future__ import annotations

from dataclasses import dataclass

from solort.cache.page_allocator import PageAllocator


@dataclass(frozen=True)
class KVCacheConfig:
    num_layers: int = 42
    num_pages: int = 4096
    page_size: int = 16
    num_kv_heads: int = 8
    head_dim: int = 256
    dtype: str = "fp16"


class PagedKVCache:
    """Metadata-only KV cache.

    The MVP keeps page tables real while omitting the tensor storage. A future backend can attach
    `k_cache` and `v_cache` tensors using the same page table semantics.
    """

    def __init__(self, config: KVCacheConfig | None = None) -> None:
        self.config = config or KVCacheConfig()
        self.allocator = PageAllocator(
            num_pages=self.config.num_pages,
            page_size=self.config.page_size,
        )

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

    def snapshot(self) -> dict[str, int | str]:
        allocator = self.allocator.snapshot()
        return {
            **allocator,
            "num_layers": self.config.num_layers,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
            "dtype": self.config.dtype,
        }
