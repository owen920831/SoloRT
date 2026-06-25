from __future__ import annotations

import pytest
from benchmarks.bench_serving import (
    ServingRun,
    _decode_tps,
    _mean_gap,
    _overall_tps,
    _percentile,
    iter_sse_payloads,
    normalize_sampling_args,
    parse_case,
    summarize_runs,
)


def test_iter_sse_payloads_extracts_data_events() -> None:
    lines = [
        b"data: {\"x\":1}\n",
        b"\n",
        b": ignored\n",
        b"data: [DONE]\n",
        b"\n",
    ]

    assert list(iter_sse_payloads(lines)) == ['{"x":1}', "[DONE]"]


def test_summarize_runs_reports_ttft_and_ttot() -> None:
    summary = summarize_runs(
        "gpu",
        [
            ServingRun(
                "gpu",
                "http://localhost:8000",
                0.1,
                0.2,
                [0.1, 0.3],
                6.0,
                5.0,
                0.5,
                4,
                3,
                10,
                "hello",
            ),
            ServingRun(
                "gpu",
                "http://localhost:8000",
                0.3,
                0.4,
                [0.3, 0.5],
                4.0,
                2.5,
                0.7,
                4,
                3,
                12,
                "world",
            ),
        ],
    )

    assert summary["label"] == "gpu"
    assert summary["runs"] == 2
    assert summary["ttft_avg_seconds"] == pytest.approx(0.2)
    assert summary["tpot_avg_seconds"] == pytest.approx(0.3)
    assert summary["itl_avg_seconds"] == pytest.approx(0.3)
    assert summary["itl_p50_seconds"] == pytest.approx(0.3)
    assert summary["itl_p95_seconds"] == pytest.approx(0.47)
    assert summary["overall_tps_avg"] == pytest.approx(5.0)
    assert summary["decode_tps_avg"] == pytest.approx(3.75)
    assert summary["decode_after_first_avg_seconds"] == pytest.approx(0.4)
    assert summary["total_avg_seconds"] == pytest.approx(0.6)
    assert summary["ttot_p50_seconds"] == pytest.approx(0.6)
    assert summary["token_chunks_avg"] == pytest.approx(3)


def test_mean_gap_uses_decode_chunk_arrival_gaps() -> None:
    assert _mean_gap([1.0, 1.2, 1.5]) == pytest.approx(0.25)
    assert _mean_gap([1.0]) is None


def test_percentile_and_tps_helpers() -> None:
    assert _percentile([0.1, 0.2, 0.3, 0.4], 95) == pytest.approx(0.385)
    assert _percentile([], 95) is None
    assert _overall_tps(16, 2.0) == pytest.approx(8.0)
    assert _decode_tps([0.1, 0.2]) == pytest.approx(6.6666667)
    assert _decode_tps([]) is None


def test_parse_case_requires_label_and_url() -> None:
    assert parse_case("cpu=http://127.0.0.1:8001") == ("cpu", "http://127.0.0.1:8001")


def test_normalize_sampling_args_uses_spec_friendly_greedy_defaults() -> None:
    class Args:
        temperature = 0.0
        top_p = None
        top_k = None
        repetition_penalty = None

    assert normalize_sampling_args(Args()) == (1.0, 0, 1.0)
