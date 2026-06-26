#!/usr/bin/env python3
"""Interactive multi-turn chat client for a SoloRT (OpenAI-compatible) server.

SoloRT keeps conversation history server-side per `session_id`: this client picks one session id at
startup and sends only your new message each turn, so the server prepends the prior turns and appends
each assistant reply automatically (no need to resend history).

Usage (server must be running, e.g. on :8000):
  python3 scripts/chat.py                                   # 127.0.0.1:8000, Qwen/Qwen3-4B
  python3 scripts/chat.py --model Qwen/Qwen3-0.6B
  python3 scripts/chat.py --url http://127.0.0.1:8000 --max-tokens 512 --temperature 0

At the `> ` prompt: type a message and press Enter (streams the reply live).
  /reset  start a fresh conversation (new session)     /exit  quit (or Ctrl-D)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import uuid


def stream_reply(url: str, model: str, session_id: str, user_text: str,
                 max_tokens: int, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "stream": True,
        "session_id": session_id,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    pieces: list[str] = []
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                pieces.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    session_id = f"chat-{uuid.uuid4().hex[:12]}"
    print(f"SoloRT chat — {args.model} @ {args.url}")
    print("輸入訊息後按 Enter(回覆會即時串流)。 /reset 開新對話, /exit 或 Ctrl-D 離開。")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            session_id = f"chat-{uuid.uuid4().hex[:12]}"
            print("(已開新對話)")
            continue
        try:
            stream_reply(args.url, args.model, session_id, user, args.max_tokens, args.temperature)
        except KeyboardInterrupt:
            print("\n(中斷)")
        except Exception as exc:  # noqa: BLE001
            print(f"[error: {exc}]")


if __name__ == "__main__":
    main()
