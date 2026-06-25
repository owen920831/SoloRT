"""Batch data structures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from solort.core.sequence import BatchPhase, Sequence


@dataclass
class Batch:
    batch_id: str
    phase: BatchPhase
    seqs: list[Sequence]
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int] | None = None
    page_indptr: list[int] | None = None
    page_indices: list[int] | None = None
    last_page_len: list[int] | None = None


class BatchBuilder:
    def build_prefill(self, sequence: Sequence, chunk_tokens: list[int]) -> Batch:
        start_pos = sequence.num_cached_tokens + sequence.num_scheduled_tokens
        positions = list(range(start_pos, start_pos + len(chunk_tokens)))
        return Batch(
            batch_id=f"batch_{uuid.uuid4().hex}",
            phase=BatchPhase.PREFILL,
            seqs=[sequence],
            input_ids=chunk_tokens,
            positions=positions,
            page_indices=list(sequence.block_table),
        )

    def build_decode(self, sequence: Sequence) -> Batch:
        position = sequence.num_prompt_tokens + len(sequence.output_ids)
        last_token = sequence.output_ids[-1] if sequence.output_ids else sequence.input_ids[-1]
        return Batch(
            batch_id=f"batch_{uuid.uuid4().hex}",
            phase=BatchPhase.DECODE,
            seqs=[sequence],
            input_ids=[last_token],
            positions=[position],
            page_indices=list(sequence.block_table),
        )
