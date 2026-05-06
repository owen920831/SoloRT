"""Sequence state for SoloRT requests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum


class SequenceStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"


class BatchPhase(str, Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"


class TaskKind(str, Enum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    BRANCH = "branch"
    CACHE_MAINTENANCE = "cache_maintenance"


class SchedulerPriority(IntEnum):
    FOREGROUND_DECODE = 0
    FOREGROUND_PREFILL = 1
    BRANCH_DECODE = 2
    BACKGROUND_PREFILL = 3
    CACHE_MAINTENANCE = 4


@dataclass
class Sequence:
    seq_id: str
    session_id: str
    input_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    priority: SchedulerPriority = SchedulerPriority.FOREGROUND_PREFILL
    task_kind: TaskKind = TaskKind.FOREGROUND
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    stop_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    kv_precision: str = "fp16"
    cache_handle: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    last_active_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.num_prompt_tokens == 0:
            self.num_prompt_tokens = len(self.input_ids)

    @property
    def remaining_prefill_tokens(self) -> int:
        scheduled_or_cached = self.num_cached_tokens + self.num_scheduled_tokens
        return max(0, self.num_prompt_tokens - scheduled_or_cached)

    @property
    def generated_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def is_prefill_complete(self) -> bool:
        return self.remaining_prefill_tokens == 0

    @property
    def is_terminal(self) -> bool:
        return self.status in {SequenceStatus.FINISHED, SequenceStatus.CANCELLED}

    def mark_running(self) -> None:
        self.status = SequenceStatus.RUNNING
        self.last_active_ts = time.time()

    def mark_finished(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.last_active_ts = time.time()

    def mark_cancelled(self) -> None:
        self.status = SequenceStatus.CANCELLED
        self.last_active_ts = time.time()
