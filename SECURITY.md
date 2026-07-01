# Security

SparkAgent treats model-selected tool names and arguments as untrusted input.

The built-in workspace tools enforce a local policy layer:

- file access is constrained to the active repository;
- `read_file` refuses common sensitive files such as `.env`, private key names, and certificate or
  key suffixes;
- internal or noisy directories such as `.git`, `.spark-agent`, `.venv`, `__pycache__`, and
  `node_modules` are ignored or blocked where relevant;
- tool arguments are validated against JSON Schema before execution;
- `run_command` uses exec-style subprocess calls without a shell and only allows known validation
  command prefixes;
- `apply_patch` validates patch paths before invoking `git apply`.

This is not a complete OS sandbox. `LocalSandbox` is a policy and subprocess boundary, not a
container, VM, seccomp profile, or namespace isolation layer. For untrusted repositories or
auto-approval mode, run SparkAgent inside your own container or VM.

When integrating custom tools, apply the same rule: model output is hostile until validated.

Please report vulnerabilities privately through the GitHub repository:

https://github.com/cvrboni/spark-agent/security

## Supported Versions

The project is pre-1.0. Security fixes are applied to the latest `main` branch.
