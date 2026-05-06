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


class MockSampler:
    """Deterministic sampler for tests and local mock runs."""

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self.vocabulary = vocabulary or [
            "SoloRT",
            " keeps",
            " foreground",
            " tokens",
            " moving",
            " under",
            " tight",
            " VRAM",
            ".",
        ]

    def sample(self, logits: object, sequence: Sequence) -> SampleResult:
        del logits
        index = len(sequence.output_ids) % len(self.vocabulary)
        text = self.vocabulary[index]
        return SampleResult(token_id=10_000 + index, text=text)
