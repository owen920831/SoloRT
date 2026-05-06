from __future__ import annotations

from solort.core.scheduler import InteractiveScheduler
from solort.core.sequence import BatchPhase, Sequence, TaskKind


def test_scheduler_chunks_prefill() -> None:
    scheduler = InteractiveScheduler(max_prefill_chunk_tokens=2)
    sequence = Sequence(
        seq_id="req_a",
        session_id="sess",
        input_ids=[1, 2, 3, 4, 5],
        max_new_tokens=1,
    )
    scheduler.add_sequence(sequence)

    batch = scheduler.build_next_batch()

    assert batch is not None
    assert batch.phase == BatchPhase.PREFILL
    assert batch.input_ids == [1, 2]

    scheduler.postprocess_batch(batch)
    assert sequence.remaining_prefill_tokens == 3


def test_foreground_decode_preempts_background_prefill() -> None:
    scheduler = InteractiveScheduler(max_prefill_chunk_tokens=4)
    background = Sequence(
        seq_id="req_bg",
        session_id="sess",
        input_ids=list(range(20)),
        max_new_tokens=1,
        task_kind=TaskKind.BACKGROUND,
    )
    foreground_decode = Sequence(
        seq_id="req_fg",
        session_id="sess",
        input_ids=[42],
        max_new_tokens=1,
        num_cached_tokens=1,
        task_kind=TaskKind.FOREGROUND,
    )

    scheduler.add_sequence(background)
    scheduler.add_sequence(foreground_decode)
    batch = scheduler.build_next_batch()

    assert batch is not None
    assert batch.phase == BatchPhase.DECODE
    assert batch.seqs[0].seq_id == "req_fg"
