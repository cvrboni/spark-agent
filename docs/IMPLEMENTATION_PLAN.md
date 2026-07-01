# SparkAgent — Piano di Implementazione

Obiettivo: portare SparkAgent da framework alpha a prodotto locale completo, mantenendo il moat del
prefix caching vLLM.

Ogni fase è ordinata per **urgenza decrescente**. Implementare in sequenza: P0 prima di P1, ecc.

---

## P0 — Bloccanti usabilità (URGENTE)

Senza questi, l'agent non è usabile come coding agent quotidiano.

| # | Task | File principali | Stato |
|---|------|-----------------|-------|
| P0.1 | Default `model` mode; `--retrieval-mode auto` segue `prefer_local_tools` | `cli.py`, `config.py` | fatto |
| P0.2 | Approval policy (`auto` / `interactive` / `never`) per `apply_patch` e `run_command` | `core/approval.py`, `executor.py`, `config.py` | fatto |
| P0.3 | Streaming SSE in executor + output token in chat | `core/streaming.py`, `executor.py`, `cli.py` | fatto |
| P0.4 | Enforcement `repo_context_budget` sul dynamic tail | `prompt_engine.py`, `executor.py` | fatto |
| P0.5 | Default `max_tool_rounds` più alti per `run` (4 invece di 1) | `cli.py` | fatto |

**Criterio di done P0:** `spark-agent chat` con tool, streaming, approval interattivo, e `run` che
modifica codice senza sorprese di default.

---

## P1 — Agent loop robusto (ALTA)

| # | Task | File principali |
|---|------|-----------------|
| P1.1 | `ModelAdapter` — normalizza tool call da DeepSeek/Hermes/Llama | `core/model_adapter.py` |
| P1.2 | Provider profiles (vLLM DeepSeek, Ollama, llama.cpp) | `config.py`, `core/provider.py` |
| P1.3 | Todo tracking in session state (`/todos`, eventi strutturati) | `session.py`, `cli.py` |
| P1.4 | Test-fix loop esplicito con summary validazione | `cli.py`, system prompt |
| P1.5 | Retry con backoff su errori provider transitori | `executor.py` |
| P1.6 | `doctor` esteso: probe tool-call + prefix-cache hint | `cli.py` |

**Criterio di done P1:** task multi-file completati autonomamente con recovery da patch falliti.

---

## P2 — Performance layer locale (ALTA-MEDIA)

| # | Task | File principali |
|---|------|-----------------|
| P2.1 | Repo index cache per hash file (`.spark-agent/index/`) | `core/repo_index.py` |
| P2.2 | Context packer — selezione file rilevanti pre-turn | `core/context_packer.py` |
| P2.3 | Tail compaction senza invalidare static prefix | `prompt_engine.py` |
| P2.4 | Metriche TTFT/token esposte a ogni run | `executor.py`, `cli.py` |
| P2.5 | Espandere allowlist `run_command` con profili (`dev`, `ci`, `strict`) | `workspace.py`, `config.py` |

**Criterio di done P2:** dimostrare TTFT inferiore vs baseline volatile su benchmark + task reali.

---

## P3 — Estensibilità (MEDIA)

| # | Task | File principali |
|---|------|-----------------|
| P3.1 | MCP client minimale (stdio transport) | `mcp/client.py` |
| P3.2 | Plugin API — `ToolSpec` da entry points / config | `core/plugins.py` |
| P3.3 | Project rules (`.spark-agent/rules.md`) nel static prefix | `cli.py`, `config.py` |
| P3.4 | Tool aggiuntivi: `glob`, `search_replace`, `write_file` | `tools/workspace.py` |
| P3.5 | Audit log strutturato patch/comandi | `session.py` |

---

## P4 — Integrazione ambiente (MEDIA-BASSA)

| # | Task | File principali |
|---|------|-----------------|
| P4.1 | TUI ricca (Textual): pannelli diff/tool/log | `tui/` |
| P4.2 | Estensione VS Code minimale (o bridge LSP) | `vscode-extension/` |
| P4.3 | Docker compose vLLM + SparkAgent | `docker/` |
| P4.4 | Wizard `init` interattivo con health check completo | `cli.py` |

---

## P5 — Credibilità e distribuzione (BASSA)

| # | Task | File principali |
|---|------|-----------------|
| P5.1 | Benchmark suite task su repo reali | `benchmarks/tasks/` |
| P5.2 | CI evaluation con modelli locali (opzionale) | `.github/workflows/` |
| P5.3 | Confronto riproducibile vs Aider/Roo (documentato) | `docs/BENCHMARKS.md` |
| P5.4 | Esempi end-to-end (refactor, bugfix, test gen) | `examples/` |
| P5.5 | Project memory cross-session (SQLite o markdown) | `core/memory.py` |

---

## Ordine di esecuzione

```
P0 (settimana 1-2) → P1 (settimana 3-4) → P2 (settimana 5-7)
    → P3 (settimana 8-10) → P4 (settimana 11-14) → P5 (continuo)
```

## Principi durante l'implementazione

1. **Non invalidare il static prefix** — ogni feature dinamica va nel tail append-only.
2. **Test offline** — nessun test che richiede vLLM live in CI.
3. **Diff minimi** — una PR/issue per task P0.x, P1.x, ecc.
4. **Backward compat** — `auto` retrieval mode mantiene comportamento configurabile.
