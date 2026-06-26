#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.tools.codebase import tool_definitions

type JsonObject = dict[str, Any]


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def auth_headers(api_key: str | None) -> dict[str, str]:
    resolved = api_key or os.getenv("SPARK_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved:
        return {}
    return {"Authorization": f"Bearer {resolved}"}


@dataclass(frozen=True, slots=True)
class TTFTSample:
    turn: int
    stable_ttft_ms: float
    unstable_ttft_ms: float


def build_engine(*, unstable_prefix: bool = False, turn: int = 0) -> PromptEngine:
    volatile = f"\nvolatile_run_marker={time.time_ns()}" if unstable_prefix else ""
    return PromptEngine(
        static_blocks=[
            PromptBlock(
                name="system",
                content=(
                    "You are SparkAgent. Solve repository tasks by using tools before editing."
                    f"{volatile}"
                ),
            ),
            PromptBlock(
                name="codebase_context",
                content=(
                    "Large repository context is intentionally not inlined. "
                    "Use repo_grep and view_file_outline for targeted retrieval."
                ),
            ),
            PromptBlock.from_jsonable("tool_definitions", tool_definitions()),
        ],
        dynamic_events=[
            {
                "role": "user",
                "content": f"Turn {turn}: inspect the repository and propose the next precise action.",
            }
        ],
    )


async def measure_stream_ttft(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: Sequence[JsonObject],
    api_key: str | None,
    timeout_s: float,
) -> float:
    payload: JsonObject = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
        "tools": tool_definitions(),
        "tool_choice": "auto",
    }
    started = time.perf_counter()
    async with client.stream(
        "POST",
        chat_completions_url(base_url),
        json=payload,
        headers=auth_headers(api_key),
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            return (time.perf_counter() - started) * 1000
    return (time.perf_counter() - started) * 1000


async def run_benchmark(args: argparse.Namespace) -> list[TTFTSample]:
    samples: list[TTFTSample] = []
    async with httpx.AsyncClient() as client:
        stable_engine = build_engine()
        for turn in range(1, args.turns + 1):
            stable_engine.append_user_message(
                f"Turn {turn}: continue the development task with cache-stable context."
            )
            unstable_engine = build_engine(unstable_prefix=True, turn=turn)

            stable_ttft = await measure_stream_ttft(
                client,
                base_url=args.base_url,
                model=args.model,
                messages=stable_engine.render_messages(),
                api_key=args.api_key,
                timeout_s=args.timeout,
            )
            unstable_ttft = await measure_stream_ttft(
                client,
                base_url=args.base_url,
                model=args.model,
                messages=unstable_engine.render_messages(),
                api_key=args.api_key,
                timeout_s=args.timeout,
            )
            samples.append(
                TTFTSample(turn=turn, stable_ttft_ms=stable_ttft, unstable_ttft_ms=unstable_ttft)
            )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TTFT for cache-stable vs volatile-prefix prompt layouts."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def print_report(samples: Sequence[TTFTSample]) -> None:
    stable = [sample.stable_ttft_ms for sample in samples]
    unstable = [sample.unstable_ttft_ms for sample in samples]
    print("turn,stable_ttft_ms,unstable_ttft_ms,delta_ms")
    for sample in samples:
        print(
            f"{sample.turn},"
            f"{sample.stable_ttft_ms:.2f},"
            f"{sample.unstable_ttft_ms:.2f},"
            f"{sample.unstable_ttft_ms - sample.stable_ttft_ms:.2f}"
        )
    print()
    print(f"stable_total_ms={sum(stable):.2f}")
    print(f"unstable_total_ms={sum(unstable):.2f}")
    print(f"stable_median_ms={statistics.median(stable):.2f}")
    print(f"unstable_median_ms={statistics.median(unstable):.2f}")


def main() -> None:
    args = parse_args()
    try:
        samples = asyncio.run(run_benchmark(args))
    except httpx.HTTPError as exc:
        raise SystemExit(f"vLLM request failed: {exc}") from exc
    print_report(samples)


if __name__ == "__main__":
    main()
