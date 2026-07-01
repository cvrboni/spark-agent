# SparkAgent

SparkAgent is a minimal async Python framework for AI agents running on local vLLM clusters and
OpenAI-compatible local model gateways.
It is designed around one constraint: preserve vLLM prefix caching by keeping the static prompt
prefix byte-stable and appending all dynamic agent state only at the tail.

Repository: https://github.com/cvrboni/spark-agent

## Why

Traditional agent frameworks often inject timestamps, session IDs, trace metadata, or dynamic
formatting near the beginning of prompts. On local long-context inference this invalidates the
prefix cache and increases Time-To-First-Token for every multi-turn step.

SparkAgent splits prompts into:

- static prefix: system prompt, codebase context, provider tool-call contract, and tool definitions
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
OpenAI-compatible `/v1/chat/completions`, prefix caching, and structured tool calls. SparkAgent
also includes a narrow `<tool_call>...</tool_call>` fallback contract for local models that do not
reliably emit provider-native tool calls.

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

Relevant provider settings:

```toml
[provider]
base_url = "http://localhost:8000/v1"
model = "deepseek-v4-flash"
timeout_s = 120
max_tokens = 2048
max_retries = 1
retry_backoff_s = 0.25
```

Environment overrides include `SPARK_AGENT_BASE_URL`, `SPARK_AGENT_MODEL`,
`SPARK_AGENT_MAX_RETRIES`, and `SPARK_AGENT_RETRY_BACKOFF_S`.

For persistent coding work, use sessions:

```bash
spark-agent chat
spark-agent chat "Implementa la prossima milestone e valida le modifiche"
spark-agent continue
```

Running `spark-agent chat` without a prompt opens a terminal chat that stays in the same session
until `/exit`. Sessions are stored append-only under `.spark-agent/sessions` in the active
repository.

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

```bash
spark-agent doctor
```

This checks provider connectivity and prints runtime hints such as retry settings and the local
repository index cache path.

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

Tool arguments are treated as untrusted model output. Before execution, SparkAgent validates them
against each tool's JSON Schema, rejects unknown properties when schemas disallow them, and returns
structured errors to the agent loop instead of calling the handler.

## Local Context Layer

`RepoIndex`

Builds an incremental local repository index with file size, mtime, language, and SHA-256 metadata.
The cache is stored at `.spark-agent/index/files.json` and ignores internal or noisy directories
such as `.git`, `.spark-agent`, `.venv`, `__pycache__`, and `node_modules`.

`ContextPacker`

Creates budget-aware repository snapshots for `--retrieval-mode local`. It selects files relevant
to the task, includes compact outlines for Python and documentation/config files, and reduces the
selected set until the snapshot fits the configured character budget.

## Security Model

SparkAgent has a local sandbox policy layer, not a full OS isolation boundary.

- `read_file` is restricted to the active repository and refuses sensitive files such as `.env`,
  common private key names, and certificate/key suffixes.
- `apply_patch` validates unified diff paths before invoking `git apply`.
- `run_command` uses exec-style subprocess calls without a shell and only permits allowlisted
  validation commands.
- `apply_patch` and `run_command` are routed through `LocalSandbox`, so stronger backends such as
  Docker, bubblewrap, or firejail can be added without changing tool schemas.

Do not run SparkAgent against untrusted repositories with `--yes` unless the repository is already
isolated by your own container or VM.

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
- Provider-native tool calls are preferred; tagged fallback tool calls must use the stable
  `<tool_call>{"name":"tool","arguments":{...}}</tool_call>` contract.
- Tool arguments are validated against JSON Schema before approval or execution.
- Local command and patch execution goes through `LocalSandbox`.
- Parallel tool execution uses `asyncio.TaskGroup`.
