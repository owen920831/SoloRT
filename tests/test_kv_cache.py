from __future__ import annotations

import pytest

from solort.cache.kv_cache import KVCacheConfig, PagedKVCache


def test_kv_cache_builds_flashinfer_style_metadata() -> None:
    cache = PagedKVCache(KVCacheConfig(num_pages=8, page_size=4))
    block_table: list[int] = []
    cache.ensure_capacity(block_table, 6)

    metadata = cache.metadata_for(block_table, token_count=6, positions=[0, 3, 4, 5])

    assert metadata.page_indptr == [0, 2]
    assert metadata.page_indices == [0, 1]
    assert metadata.last_page_len == [2]
    assert metadata.slot_mapping == [0, 3, 4, 5]


def test_kv_cache_transaction_commit_keeps_pages() -> None:
    cache = PagedKVCache(KVCacheConfig(num_pages=8, page_size=4))
    block_table = cache.ensure_capacity([], 4)

    transaction = cache.begin_append_transaction(
        block_table,
        current_token_count=4,
        append_token_count=4,
    )
    cache.commit_transaction(transaction)

    assert block_table == [0, 1]
    assert cache.snapshot()["used_pages"] == 2


def test_kv_cache_transaction_rollback_releases_new_pages() -> None:
    cache = PagedKVCache(KVCacheConfig(num_pages=8, page_size=4))
    block_table = cache.ensure_capacity([], 4)

    transaction = cache.begin_append_transaction(
        block_table,
        current_token_count=4,
        append_token_count=4,
    )
    cache.rollback_transaction(block_table, transaction)

    assert block_table == [0]
    assert cache.snapshot()["used_pages"] == 1
    assert cache.snapshot()["free_pages"] == 7


def test_kv_cache_slot_mapping_rejects_missing_page() -> None:
    cache = PagedKVCache(KVCacheConfig(num_pages=8, page_size=4))

    with pytest.raises(IndexError):
        cache.slot_mapping([0], [4])


def test_tensor_backed_kv_cache_mirrors_layer_tokens() -> None:
    torch = pytest.importorskip("torch")
    cache = PagedKVCache(
        KVCacheConfig(
            num_layers=2,
            num_pages=2,
            page_size=4,
            num_kv_heads=2,
            head_dim=4,
            dtype="fp32",
            device="cpu",
            allocate_tensors=True,
        )
    )
    key = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    value = key + 100

    cache.store_layer_tokens(layer_idx=1, slot_mapping=[0, 1, 5], key=key, value=value)

    assert cache.snapshot()["tensor_storage"] == "allocated"
    assert cache.snapshot()["mirrored_tokens"] == 3
    assert torch.equal(cache.k_cache[1, 0, 0], key[0, :, 0, :])
    assert torch.equal(cache.k_cache[1, 0, 1], key[0, :, 1, :])
    assert torch.equal(cache.v_cache[1, 1, 1], value[0, :, 2, :])
