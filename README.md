# SparkAgent

SparkAgent is a minimal async Python framework for AI agents running on local vLLM clusters.
It is designed around one constraint: preserve vLLM prefix caching by keeping the static prompt
prefix byte-stable and appending all dynamic agent state only at the tail.

Repository: https://github.com/cvrboni/spark-agent

## Why

Traditional agent frameworks often inject timestamps, session IDs, trace metadata, or dynamic
formatting near the beginning of prompts. On local long-context inference this invalidates the
prefix cache and increases Time-To-First-Token for every multi-turn step.

SparkAgent splits prompts into:

- static prefix: system prompt, codebase context, and tool definitions
- append-only tail: user turns, assistant reasoning blocks, tool calls, tool results, and observations

The static prefix is serialized deterministically once and can be checked with `static_hash`.

## Target Runtime

SparkAgent expects a vLLM server exposing OpenAI-compatible chat completions:

```bash
vllm serve deepseek-ai/DeepSeek-V4 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp"}' \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice
```

Exact vLLM flags can vary by version and deployment topology. The important requirements are
OpenAI-compatible `/v1/chat/completions`, prefix caching, and structured DeepSeek tool calls.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/cvrboni/spark-agent/main/scripts/install.sh | sh
```

or from source:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick Start

For local vLLM or another OpenAI-compatible server:

```bash
spark-agent init --base-url http://localhost:8000/v1 --model deepseek-v4-flash
spark-agent doctor
spark-agent run "Analizza questa repo e dimmi da dove iniziare"
```

SparkAgent stores configuration in `~/.config/spark-agent/config.toml`.

For persistent coding work, use sessions:

```bash
spark-agent chat "Implementa la prossima milestone e valida le modifiche"
spark-agent continue
```

Sessions are stored append-only under `.spark-agent/sessions` in the active repository.

## Python API

```python
import asyncio

from spark_agent.core.executor import AgentExecutor, VLLMClientConfig
from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.tools.codebase import codebase_tool_specs, tool_definitions


async def main() -> None:
    prompt = PromptEngine(
        static_blocks=[
            PromptBlock("system", "You are SparkAgent. Use tools before editing code."),
            PromptBlock("codebase_context", "Repository context must be retrieved selectively."),
            PromptBlock.from_jsonable("tool_definitions", tool_definitions()),
        ],
    )
    async with AgentExecutor(
        prompt_engine=prompt,
    config=VLLMClientConfig(
            base_url="http://localhost:8000",
            model="deepseek-ai/DeepSeek-V4",
            api_key=None,
        ),
        tools=codebase_tool_specs(),
    ) as agent:
        answer = await agent.run("Find where PromptEngine appends dynamic events.")
        print(answer)


asyncio.run(main())
```

## CLI Smoke Test

```bash
spark-agent prompt-preview
```

This prints the deterministic static prompt prefix and append-only tail marker.

## Benchmarks

Run a 5-turn TTFT comparison between a stable prefix and a volatile-prefix baseline:

```bash
python benchmarks/run_cache_test.py \
  --base-url http://localhost:8000/v1 \
  --model deepseek-ai/DeepSeek-V4
```

The script streams responses and records the first received token event for each turn.

## Built-in Tools

`repo_grep`

Searches a repository with `ripgrep` and returns bounded matching snippets. It avoids loading
entire files into the model context.

`view_file_outline`

Returns a compact outline of classes, functions, and methods. Python files use `ast`; other
languages use a conservative text outline.

`list_files`, `read_file`, `apply_patch`, `run_command`

Persistent sessions can inspect repository files, apply unified diffs, and run allowlisted
validation commands such as `git status`, `git diff`, `pytest`, `ruff check`, and `compileall`.

## Development

```bash
pytest
ruff check .
python -m compileall spark_agent benchmarks main.py
```

More docs:

- [Install](docs/INSTALL.md)
- [Local Models](docs/LOCAL_MODELS.md)

## Design Rules

- Static prompt blocks are immutable during a run.
- Tool definitions belong in the static prefix and are serialized canonically.
- Dynamic state is append-only and always follows `<spark_append_only_tail_v1>`.
- Model tool calls are consumed from structured payloads, not parsed with regex.
- Parallel tool execution uses `asyncio.TaskGroup`.
