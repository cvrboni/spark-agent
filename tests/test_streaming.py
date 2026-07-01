from __future__ import annotations

from spark_agent.core.streaming import merge_tool_call_delta, parse_sse_data_line


def test_parse_sse_data_line_ignores_done() -> None:
    assert parse_sse_data_line("data: [DONE]") is None


def test_parse_sse_data_line_decodes_chunk() -> None:
    chunk = parse_sse_data_line(
        'data: {"choices":[{"delta":{"content":"hi"}}]}'
    )
    assert chunk is not None
    assert chunk["choices"][0]["delta"]["content"] == "hi"


def test_merge_tool_call_delta_accumulates_arguments() -> None:
    accumulated: dict[int, dict] = {}
    merge_tool_call_delta(
        accumulated,
        {"index": 0, "id": "call_1", "function": {"name": "read", "arguments": '{"path":'}},
    )
    merge_tool_call_delta(
        accumulated,
        {"index": 0, "function": {"arguments": ' "main.py"}'}},
    )
    assert accumulated[0]["id"] == "call_1"
    assert accumulated[0]["function"]["name"] == "read"
    assert accumulated[0]["function"]["arguments"] == '{"path": "main.py"}'
