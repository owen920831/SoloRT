from __future__ import annotations

from solort.cache.prefix_cache import PrefixCache


def test_prefix_cache_hit_miss_and_release() -> None:
    cache = PrefixCache(block_size=2, max_entries=4)

    miss = cache.match([1, 2, 3])
    assert miss.matched_tokens == 0

    entry = cache.insert([1, 2, 3], pages=[7, 8], pinned=True)
    hit = cache.match([1, 2, 3, 4])

    assert hit.matched_tokens == 3
    assert hit.pages == [7, 8]
    assert hit.entry is entry
    assert entry.ref_count == 1

    cache.release(hit.entry)
    assert entry.ref_count == 0
    assert cache.snapshot()["hits"] == 1
    assert cache.snapshot()["misses"] == 1


def test_prefix_cache_evicts_only_unpinned_idle_entries() -> None:
    cache = PrefixCache(block_size=2, max_entries=4)
    pinned = cache.insert([1, 2], pages=[1], pinned=True)
    idle = cache.insert([3, 4], pages=[2], pinned=False)
    busy = cache.insert([5, 6], pages=[3], pinned=False)
    busy.ref_count = 1

    removed = cache.evict(3)

    assert idle in removed
    assert pinned not in removed
    assert busy not in removed
    assert cache.snapshot()["entries"] == 2
