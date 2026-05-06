"""Foreground-first scheduler for the Python MVP."""

from __future__ import annotations

import time
from dataclasses import dataclass

from solort.core.batch import Batch, BatchBuilder
from solort.core.sequence import (
    BatchPhase,
    SchedulerPriority,
    Sequence,
    SequenceStatus,
    TaskKind,
)


@dataclass
class SchedulerSnapshot:
    waiting_sequences: int
    running_sequences: int
    finished_sequences: int


class InteractiveScheduler:
    """Scheduler that prioritizes foreground decode over all prefill work."""

    def __init__(self, max_prefill_chunk_tokens: int = 16) -> None:
        if max_prefill_chunk_tokens <= 0:
            raise ValueError("max_prefill_chunk_tokens must be positive")
        self.max_prefill_chunk_tokens = max_prefill_chunk_tokens
        self._builder = BatchBuilder()
        self._sequences: dict[str, Sequence] = {}
        self._order: list[str] = []

    def add_sequence(self, sequence: Sequence) -> None:
        self._sequences[sequence.seq_id] = sequence
        self._order.append(sequence.seq_id)

    def cancel(self, seq_id: str) -> bool:
        sequence = self._sequences.get(seq_id)
        if sequence is None or sequence.is_terminal:
            return False
        sequence.mark_cancelled()
        return True

    def get(self, seq_id: str) -> Sequence | None:
        return self._sequences.get(seq_id)

    def build_next_batch(self, token_budget: int | None = None) -> Batch | None:
        sequence = self._select_next_sequence()
        if sequence is None:
            return None

        sequence.mark_running()
        if sequence.is_prefill_complete:
            sequence.priority = self._effective_priority(sequence)
            return self._builder.build_decode(sequence)

        budget = token_budget or self.max_prefill_chunk_tokens
        chunk_size = min(sequence.remaining_prefill_tokens, budget, self.max_prefill_chunk_tokens)
        start = sequence.num_cached_tokens + sequence.num_scheduled_tokens
        end = start + chunk_size
        sequence.priority = self._effective_priority(sequence)
        return self._builder.build_prefill(sequence, sequence.input_ids[start:end])

    def postprocess_batch(self, batch: Batch) -> None:
        sequence = batch.seqs[0]
        if sequence.status == SequenceStatus.CANCELLED:
            return
        if batch.phase == BatchPhase.PREFILL:
            sequence.num_scheduled_tokens += len(batch.input_ids)
        sequence.last_active_ts = time.time()

    def mark_finished(self, seq_id: str) -> None:
        sequence = self._sequences.get(seq_id)
        if sequence is not None:
            sequence.mark_finished()

    def _select_next_sequence(self) -> Sequence | None:
        candidates = [
            self._sequences[seq_id]
            for seq_id in self._order
            if not self._sequences[seq_id].is_terminal
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda seq: (self._effective_priority(seq), seq.created_ts))
        return candidates[0]

    def _effective_priority(self, sequence: Sequence) -> SchedulerPriority:
        # Decode gets first claim on the GPU because interactive smoothness is governed by the
        # gap between streamed tokens, not aggregate prompt throughput.
        if sequence.task_kind == TaskKind.CACHE_MAINTENANCE:
            return SchedulerPriority.CACHE_MAINTENANCE
        if sequence.is_prefill_complete:
            if sequence.task_kind == TaskKind.BRANCH:
                return SchedulerPriority.BRANCH_DECODE
            return SchedulerPriority.FOREGROUND_DECODE
        if sequence.task_kind == TaskKind.BACKGROUND:
            return SchedulerPriority.BACKGROUND_PREFILL
        return SchedulerPriority.FOREGROUND_PREFILL

    def snapshot(self) -> dict[str, int]:
        waiting = 0
        running = 0
        finished = 0
        for sequence in self._sequences.values():
            if sequence.status == SequenceStatus.FINISHED:
                finished += 1
            elif sequence.status == SequenceStatus.RUNNING:
                running += 1
            elif sequence.status == SequenceStatus.WAITING:
                waiting += 1
        return {
            "waiting_sequences": waiting,
            "running_sequences": running,
            "finished_sequences": finished,
        }
