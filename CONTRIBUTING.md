# Contributing

SparkAgent is intentionally small. Contributions should preserve the core design:
the static prompt prefix must remain deterministic and dynamic state must be append-only.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Guidelines

- Keep prompt serialization byte-stable across turns.
- Do not add timestamps, random IDs, tracing metadata, or dynamic formatting to the static prefix.
- Prefer targeted repository retrieval over loading complete files into context.
- Use `asyncio.TaskGroup` for parallel async orchestration.
- Avoid regex parsing for model tool calls; consume structured OpenAI-compatible payloads.

