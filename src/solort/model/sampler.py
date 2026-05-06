"""Sampling interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from solort.core.sequence import Sequence


@dataclass(frozen=True)
class SampleResult:
    token_id: int
    text: str
    finished: bool = False


class Sampler(Protocol):
    def sample(self, logits: object, sequence: Sequence) -> SampleResult:
        """Sample a token from model logits."""
