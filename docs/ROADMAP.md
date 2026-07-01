# SparkAgent Priority Roadmap

SparkAgent's goal is to become a local-first coding agent optimized for OpenAI-compatible local
inference servers, especially vLLM. The priority order is intentionally biased toward capabilities
that make the agent continuous, recoverable, and useful on real repositories.

## 1. Persistent Terminal Chat UI

Status: in progress (streaming + TTFT in chat/run).

- Make `spark-agent chat` open a persistent terminal session when no prompt is provided.
- Keep the user inside one chat until `/exit` or EOF.
- Preserve all turns in the same append-only session.
- Support slash commands such as `/help`, `/session`, and `/exit`.
- Done: streaming tokens and TTFT metrics.
- Next: richer rendering inside the same UI.

## 2. Persistent Local Agent Loop

Status: in progress.

- Persist sessions under `.spark-agent/sessions`.
- Resume the latest or a named session.
- Save every user, assistant, and tool event append-only.
- Keep the static prompt prefix stable across resumed turns.
- Provide `spark-agent chat` and `spark-agent continue`.

## 3. Safe Workspace Tools

Status: in progress.

- Inspect files with `list_files`, `read_file`, `repo_grep`, and `view_file_outline`.
- Edit with narrow unified diffs through `apply_patch`.
- Validate with allowlisted commands through `run_command`.
- Keep file access constrained to the active repository.

## 4. Local Performance Layer

Status: in progress (context budget enforcement done).

- Done: stream model responses and expose TTFT in normal runs.
- Done: enforce `repo_context_budget` on the append-only tail.
- Cache repository indexes by file hash.
- Build a context packer that chooses relevant files before every model step.
- Add provider profiles for vLLM, llama.cpp, Ollama-compatible gateways, and hosted fallbacks.

## 5. Competitive Coding UX

Status: in progress (approval policy done).

- Done: approval policy for risky commands (`auto`, `interactive`, `never`).
- Track todos and step progress in session state.
- Add a test-fix loop with explicit validation summaries.
- Add project memory that stores durable facts outside the prompt tail.

## 6. Evaluation

Status: planned.

- Benchmark TTFT, tokens/sec, and task success on local models.
- Record tool counts, patch success, validation pass rate, and resume reliability.
- Compare against other local coding agents using reproducible repository tasks.
