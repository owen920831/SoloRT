"""Small terminal client that prints only assistant text."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Iterable, Iterator
from typing import Any


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


def content_from_sse_payload(payload: str) -> str | None:
    if payload == "[DONE]":
        return None
    event = json.loads(payload)
    choices = event.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("delta", {}).get("content")
    return str(content) if content else None


def stream_chat_text(
    *,
    url: str,
    model: str,
    session_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float | None,
    max_repeated_token_run: int | None,
    timeout: float,
) -> Iterator[str]:
    payload: dict[str, Any] = {
        "model": model,
        "session_id": session_id,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if max_repeated_token_run is not None:
        payload["max_repeated_token_run"] = max_repeated_token_run

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event_payload in iter_sse_payloads(response):
            content = content_from_sse_payload(event_payload)
            if content:
                yield content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with SoloRT and print only assistant text.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. Omit for interactive mode.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--session-id", default="chat")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--max-repeated-token-run", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def print_response(args: argparse.Namespace, prompt: str) -> None:
    for chunk in stream_chat_text(
        url=args.url,
        model=args.model,
        session_id=args.session_id,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_repeated_token_run=args.max_repeated_token_run,
        timeout=args.timeout,
    ):
        print(chunk, end="", flush=True)
    print()


def repl(args: argparse.Namespace) -> None:
    print(f"SoloRT chat session={args.session_id}. Ctrl-D to exit.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            continue
        print_response(args, prompt)


def main() -> None:
    args = build_parser().parse_args()
    prompt = " ".join(args.prompt).strip()
    try:
        if prompt:
            print_response(args, prompt)
        else:
            repl(args)
    except KeyboardInterrupt:
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
