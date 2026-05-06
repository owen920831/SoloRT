"""Runtime metrics for local serving."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SequenceTiming:
    created_ts: float
    first_token_ts: float | None = None
    finished_ts: float | None = None
    token_timestamps: list[float] = field(default_factory=list)


class RuntimeMetrics:
    def __init__(self) -> None:
        self.requests_started = 0
        self.requests_finished = 0
        self.tokens_generated = 0
        self._timings: dict[str, SequenceTiming] = {}

    def start_request(self, seq_id: str) -> None:
        self.requests_started += 1
        self._timings[seq_id] = SequenceTiming(created_ts=time.time())

    def record_token(self, seq_id: str) -> None:
        now = time.time()
        timing = self._timings.setdefault(seq_id, SequenceTiming(created_ts=now))
        if timing.first_token_ts is None:
            timing.first_token_ts = now
        timing.token_timestamps.append(now)
        self.tokens_generated += 1

    def finish_request(self, seq_id: str) -> None:
        self.requests_finished += 1
        timing = self._timings.get(seq_id)
        if timing is not None:
            timing.finished_ts = time.time()

    def snapshot(self) -> dict[str, float | int | None]:
        ttfts = [
            timing.first_token_ts - timing.created_ts
            for timing in self._timings.values()
            if timing.first_token_ts is not None
        ]
        tpots = []
        for timing in self._timings.values():
            if len(timing.token_timestamps) < 2:
                continue
            gaps = [
                later - earlier
                for earlier, later in zip(
                    timing.token_timestamps,
                    timing.token_timestamps[1:],
                    strict=False,
                )
            ]
            tpots.extend(gaps)

        return {
            "requests_started": self.requests_started,
            "requests_finished": self.requests_finished,
            "tokens_generated": self.tokens_generated,
            "avg_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else None,
            "avg_tpot_seconds": sum(tpots) / len(tpots) if tpots else None,
        }
