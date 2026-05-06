from __future__ import annotations

from solort.model.sampler import SampleResult


class DeterministicExecutor:
    name = "deterministic-test"
    supports_prefix_cache = True

    def __init__(self) -> None:
        self.vocabulary = [
            "SoloRT",
            " keeps",
            " foreground",
            " tokens",
            " moving",
            ".",
        ]

    def forward_prefill(self, batch: object) -> None:
        del batch

    def forward_decode(self, batch: object) -> SampleResult:
        sequence = batch.seqs[0]
        index = len(sequence.output_ids) % len(self.vocabulary)
        return SampleResult(token_id=10_000 + index, text=self.vocabulary[index])
