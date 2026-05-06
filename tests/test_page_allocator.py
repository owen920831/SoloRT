from __future__ import annotations

import pytest

from solort.cache.page_allocator import PageAllocationError, PageAllocator


def test_alloc_free_and_refcount() -> None:
    allocator = PageAllocator(num_pages=3, page_size=16)

    pages = allocator.alloc(2)
    assert pages == [0, 1]
    assert allocator.snapshot()["used_pages"] == 2

    allocator.incref(pages[0])
    allocator.free([pages[0]])
    assert allocator.ref_count[pages[0]] == 1
    assert pages[0] in allocator.used_pages

    allocator.free([pages[0], pages[1]])
    assert allocator.snapshot()["used_pages"] == 0
    assert allocator.snapshot()["free_pages"] == 3


def test_allocator_exhaustion() -> None:
    allocator = PageAllocator(num_pages=1, page_size=16)
    allocator.alloc(1)

    with pytest.raises(PageAllocationError):
        allocator.alloc(1)


def test_invalid_free_raises() -> None:
    allocator = PageAllocator(num_pages=1, page_size=16)

    with pytest.raises(ValueError):
        allocator.free([0])
