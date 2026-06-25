"""Sampling interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleResult:
    token_id: int
    text: str
    finished: bool = False
