from __future__ import annotations

from solort.cli import build_parser, content_from_sse_payload, iter_sse_payloads


def test_cli_sse_parser_extracts_content_only() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"blue"},"finish_reason":null}]}\n',
        "\n",
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        "\n",
        "data: [DONE]\n",
        "\n",
    ]

    payloads = list(iter_sse_payloads(lines))

    assert content_from_sse_payload(payloads[0]) == "blue"
    assert content_from_sse_payload(payloads[1]) is None
    assert content_from_sse_payload(payloads[2]) is None


def test_cli_defaults_are_chat_friendly() -> None:
    args = build_parser().parse_args([])

    assert args.max_tokens == 512
    assert args.temperature == 0.7
    assert args.top_p == 0.8
    assert args.top_k == 20
    assert args.repetition_penalty == 1.08
    assert args.max_repeated_token_run == 16
