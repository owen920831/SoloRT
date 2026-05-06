"""Session-aware block-hash prefix cache."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


@dataclass
class PrefixCacheEntry:
    key: str
    token_hash_chain: tuple[int, ...]
    token_ids: tuple[int, ...]
    pages: list[int]
    token_count: int
    ref_count: int = 0
    pinned: bool = False
    last_access_ts: float = field(default_factory=time.time)


@dataclass
class PrefixMatch:
    matched_tokens: int
    pages: list[int]
    entry: PrefixCacheEntry | None = None


class PrefixCache:
    """Simple block-hash prefix cache.

    The cache stores exact block hash chains and returns the longest cached chain that is a prefix
    of the incoming token chain. A future radix cache can keep this public behavior while changing
    the implementation.
    """

    def __init__(self, block_size: int = 16, max_entries: int = 128) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.block_size = block_size
        self.max_entries = max_entries
        self._entries: dict[str, PrefixCacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def make_hash_chain(self, token_ids: list[int]) -> tuple[int, ...]:
        chain: list[int] = []
        for start in range(0, len(token_ids), self.block_size):
            block = token_ids[start : start + self.block_size]
            digest = hashlib.blake2b(digest_size=8)
            for token_id in block:
                digest.update(int(token_id).to_bytes(8, "little", signed=True))
            chain.append(int.from_bytes(digest.digest(), "little"))
        return tuple(chain)

    def make_key(self, token_ids: list[int]) -> str:
        chain = self.make_hash_chain(token_ids)
        return ":".join(str(item) for item in chain)

    def match(self, token_ids: list[int]) -> PrefixMatch:
        incoming_tokens = tuple(token_ids)
        best: PrefixCacheEntry | None = None

        for entry in self._entries.values():
            if entry.token_count > len(incoming_tokens):
                continue
            if entry.token_ids == incoming_tokens[: entry.token_count] and (
                best is None or entry.token_count > best.token_count
            ):
                best = entry

        if best is None:
            self._misses += 1
            return PrefixMatch(matched_tokens=0, pages=[])

        self._hits += 1
        best.ref_count += 1
        best.last_access_ts = time.time()
        return PrefixMatch(
            matched_tokens=min(best.token_count, len(token_ids)),
            pages=list(best.pages),
            entry=best,
        )

    def insert(
        self,
        token_ids: list[int],
        pages: list[int],
        pinned: bool = False,
    ) -> PrefixCacheEntry:
        key = self.make_key(token_ids)
        entry = PrefixCacheEntry(
            key=key,
            token_hash_chain=self.make_hash_chain(token_ids),
            token_ids=tuple(token_ids),
            pages=list(pages),
            token_count=len(token_ids),
            pinned=pinned,
        )
        self._entries[key] = entry
        self._evict_to_capacity()
        return entry

    def release(self, entry: PrefixCacheEntry | None) -> None:
        if entry is None:
            return
        entry.ref_count = max(0, entry.ref_count - 1)

    def evict(self, target_count: int = 1) -> list[PrefixCacheEntry]:
        if target_count <= 0:
            return []
        evictable = [
            entry
            for entry in self._entries.values()
            if not entry.pinned and entry.ref_count == 0
        ]
        evictable.sort(key=lambda item: item.last_access_ts)
        removed = evictable[:target_count]
        for entry in removed:
            self._entries.pop(entry.key, None)
        return removed

    def _evict_to_capacity(self) -> None:
        overflow = len(self._entries) - self.max_entries
        if overflow > 0:
            self.evict(overflow)

    def snapshot(self) -> dict[str, int]:
        pinned = sum(1 for entry in self._entries.values() if entry.pinned)
        return {
            "entries": len(self._entries),
            "pinned_entries": pinned,
            "hits": self._hits,
            "misses": self._misses,
        }
