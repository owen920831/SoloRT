"""Measure TTFT, TPOT, ITL, TPS, and total time for SoloRT serving endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ServingRun:
    label: str
    url: str
    ttft_seconds: float | None
    tpot_seconds: float | None
    itl_seconds: list[float]
    overall_tps: float
    decode_tps: float | None
    ttot_seconds: float
    events: int
    token_chunks: int
    chars: int
    text: str


def iter_sse_payloads(lines: Iterable[bytes | str]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if data_lines:
        yield "\n".join(data_lines)


def run_streaming_request(
    *,
    label: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    timeout: float,
) -> ServingRun:
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    token_arrivals: list[float] = []
    chunks: list[str] = []
    events = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event_payload in iter_sse_payloads(response):
            now = time.perf_counter()
            if event_payload == "[DONE]":
                ttot = now - start
                itls = _gaps(token_arrivals)
                return ServingRun(
                    label=label,
                    url=base_url,
                    ttft_seconds=None if first_token_at is None else first_token_at - start,
                    tpot_seconds=_mean(itls),
                    itl_seconds=itls,
                    overall_tps=_overall_tps(len(token_arrivals), ttot),
                    decode_tps=_decode_tps(itls),
                    ttot_seconds=ttot,
                    events=events,
                    token_chunks=len(token_arrivals),
                    chars=sum(len(chunk) for chunk in chunks),
                    text="".join(chunks),
                )
            events += 1
            event = json.loads(event_payload)
            choices = event.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                if first_token_at is None:
                    first_token_at = now
                token_arrivals.append(now)
                chunks.append(content)

    raise RuntimeError(f"{label}: stream ended without [DONE]")


def summarize_runs(label: str, runs: Iterable[ServingRun]) -> dict[str, Any]:
    run_list = list(runs)
    ttfts = [run.ttft_seconds for run in run_list if run.ttft_seconds is not None]
    tpots = [run.tpot_seconds for run in run_list if run.tpot_seconds is not None]
    itls = [itl for run in run_list for itl in run.itl_seconds]
    overall_tps_values = [run.overall_tps for run in run_list]
    decode_tps_values = [run.decode_tps for run in run_list if run.decode_tps is not None]
    ttots = [run.ttot_seconds for run in run_list]
    decode_after_first = [
        run.ttot_seconds - run.ttft_seconds
        for run in run_list
        if run.ttft_seconds is not None
    ]
    return {
        "label": label,
        "runs": len(run_list),
        "ttft_avg_seconds": statistics.fmean(ttfts) if ttfts else None,
        "ttft_p50_seconds": statistics.median(ttfts) if ttfts else None,
        "tpot_avg_seconds": statistics.fmean(tpots) if tpots else None,
        "tpot_p50_seconds": statistics.median(tpots) if tpots else None,
        "itl_avg_seconds": statistics.fmean(itls) if itls else None,
        "itl_p50_seconds": statistics.median(itls) if itls else None,
        "itl_p95_seconds": _percentile(itls, 95),
        "overall_tps_avg": (
            statistics.fmean(overall_tps_values) if overall_tps_values else None
        ),
        "decode_tps_avg": (
            statistics.fmean(decode_tps_values) if decode_tps_values else None
        ),
        "decode_after_first_avg_seconds": (
            statistics.fmean(decode_after_first) if decode_after_first else None
        ),
        "total_avg_seconds": statistics.fmean(ttots) if ttots else None,
        "total_p50_seconds": statistics.median(ttots) if ttots else None,
        "ttot_avg_seconds": statistics.fmean(ttots) if ttots else None,
        "ttot_p50_seconds": statistics.median(ttots) if ttots else None,
        "token_chunks_avg": (
            statistics.fmean([run.token_chunks for run in run_list]) if run_list else None
        ),
        "chars_avg": statistics.fmean([run.chars for run in run_list]) if run_list else None,
    }


def _gaps(timestamps: list[float]) -> list[float]:
    return [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_gap(timestamps: list[float]) -> float | None:
    return _mean(_gaps(timestamps))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _overall_tps(token_count: int, total_seconds: float) -> float:
    if total_seconds <= 0:
        return 0.0
    return token_count / total_seconds


def _decode_tps(itls: list[float]) -> float | None:
    total_decode_seconds = sum(itls)
    if not itls or total_decode_seconds <= 0:
        return None
    return len(itls) / total_decode_seconds


def parse_case(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must be LABEL=URL, for example gpu=http://...")
    label, url = value.split("=", 1)
    if not label or not url:
        raise argparse.ArgumentTypeError("case label and URL must be non-empty")
    return label, url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=[],
        help="Endpoint to benchmark as LABEL=URL. Repeat for cpu/gpu comparison.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain SoloRT in one concise sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = args.case or [("gpu", "http://127.0.0.1:8000")]
    all_runs: dict[str, list[ServingRun]] = {label: [] for label, _ in cases}

    for label, url in cases:
        for _ in range(args.warmup):
            run_streaming_request(
                label=label,
                base_url=url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                timeout=args.timeout,
            )
        for index in range(args.runs):
            run = run_streaming_request(
                label=label,
                base_url=url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                timeout=args.timeout,
            )
            all_runs[label].append(run)
            ttft = "n/a" if run.ttft_seconds is None else f"{run.ttft_seconds:.3f}s"
            tpot = "n/a" if run.tpot_seconds is None else f"{run.tpot_seconds:.3f}s"
            itl_p95 = _percentile(run.itl_seconds, 95)
            itl_p95_text = "n/a" if itl_p95 is None else f"{itl_p95:.3f}s"
            decode_tps = "n/a" if run.decode_tps is None else f"{run.decode_tps:.2f}"
            decode_after_first = (
                "n/a"
                if run.ttft_seconds is None
                else f"{run.ttot_seconds - run.ttft_seconds:.3f}s"
            )
            print(
                f"{label} run {index + 1}/{args.runs}: "
                f"ttft={ttft} tpot={tpot} itl_p95={itl_p95_text} "
                f"overall_tps={run.overall_tps:.2f} decode_tps={decode_tps} "
                f"decode_after_first={decode_after_first} total/ttot={run.ttot_seconds:.3f}s "
                f"token_chunks={run.token_chunks} "
                f"chars={run.chars}"
            )

    result = {
        "summaries": [summarize_runs(label, runs) for label, runs in all_runs.items()],
        "runs": {label: [asdict(run) for run in runs] for label, runs in all_runs.items()},
    }
    print(json.dumps(result["summaries"], indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
