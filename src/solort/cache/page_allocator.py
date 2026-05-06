"""Page allocator for the paged KV cache."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class PageAllocationError(RuntimeError):
    """Raised when the KV page allocator cannot satisfy a request."""


@dataclass
class PageAllocator:
    """Reference-counted allocator for fixed-size KV pages."""

    num_pages: int
    page_size: int
    free_pages: deque[int] = field(init=False)
    used_pages: set[int] = field(default_factory=set, init=False)
    ref_count: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        self.free_pages = deque(range(self.num_pages))
        self.ref_count = [0 for _ in range(self.num_pages)]

    def alloc(self, n: int) -> list[int]:
        """Allocate `n` pages and return their physical page ids."""

        if n < 0:
            raise ValueError("cannot allocate a negative number of pages")
        if n == 0:
            return []
        if n > len(self.free_pages):
            raise PageAllocationError(
                f"requested {n} pages but only {len(self.free_pages)} are free"
            )

        pages: list[int] = []
        for _ in range(n):
            page_id = self.free_pages.popleft()
            self.used_pages.add(page_id)
            self.ref_count[page_id] = 1
            pages.append(page_id)
        return pages

    def free(self, pages: list[int]) -> None:
        """Release one reference for each page."""

        for page_id in pages:
            self.decref(page_id)

    def incref(self, page_id: int) -> None:
        self._validate_allocated(page_id)
        self.ref_count[page_id] += 1

    def decref(self, page_id: int) -> None:
        self._validate_allocated(page_id)
        self.ref_count[page_id] -= 1
        if self.ref_count[page_id] == 0:
            self.used_pages.remove(page_id)
            self.free_pages.append(page_id)

    def _validate_allocated(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(f"page id {page_id} is out of range")
        if self.ref_count[page_id] <= 0 or page_id not in self.used_pages:
            raise ValueError(f"page id {page_id} is not allocated")

    def snapshot(self) -> dict[str, int]:
        return {
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "free_pages": len(self.free_pages),
            "used_pages": len(self.used_pages),
        }
