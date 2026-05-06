from __future__ import annotations

from solort.backends.flashinfer_backend import FlashInferBackend
from solort.core.batch import Batch
from solort.core.sequence import BatchPhase, Sequence


def test_flashinfer_backend_preserves_page_metadata_lists() -> None:
    backend = FlashInferBackend()
    sequence = Sequence(
        seq_id="seq",
        session_id="sess",
        input_ids=[1],
        max_new_tokens=1,
    )
    batch = Batch(
        batch_id="batch",
        phase=BatchPhase.DECODE,
        seqs=[sequence],
        input_ids=[1],
        positions=[0],
        slot_mapping=[0],
        page_indptr=[0, 1],
        page_indices=[3],
        last_page_len=[1],
    )

    metadata = backend.prepare_metadata(batch)

    assert metadata["slot_mapping"] == [0]
    assert metadata["page_indptr"] == [0, 1]
    assert metadata["page_indices"] == [3]
    assert metadata["last_page_len"] == [1]
