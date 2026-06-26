# Install

SparkAgent targets Python 3.13+ and local OpenAI-compatible inference servers.

## curl

```bash
curl -fsSL https://raw.githubusercontent.com/cvrboni/spark-agent/main/scripts/install.sh | sh
```

The installer uses the first available strategy:

1. `pipx`
2. `uv tool`
3. a dedicated virtual environment in:

```text
~/.local/share/spark-agent
```

4. `python3 -m pip --user`

It links:

```text
~/.local/bin/spark-agent
```

On Debian/Ubuntu, if virtual environment creation fails with `ensurepip is not available`, install:

```bash
sudo apt install python3.13-venv
```

or install `pipx`/`uv` and rerun the installer.

## From Source

```bash
git clone https://github.com/cvrboni/spark-agent
cd spark-agent
python -m pip install -e ".[dev]"
spark-agent init
spark-agent doctor
```

## Configuration

The default config path is:

```text
~/.config/spark-agent/config.toml
```

Create it with:

```bash
spark-agent init \
  --base-url http://localhost:8000/v1 \
  --model deepseek-v4-flash \
  --language auto
```

Environment variables override file config:

```bash
export SPARK_AGENT_BASE_URL=http://llm.local:8000/v1
export SPARK_AGENT_MODEL=deepseek-v4-flash
export SPARK_AGENT_LANGUAGE=auto
export SPARK_AGENT_API_KEY=...
```

## Python Selection

The installer requires Python 3.13+. It prefers `python3.13`, then `python3`.

If your system has multiple Python versions and `uv` tries to install with Python 3.12, point the
installer explicitly at Python 3.13:

```bash
SPARK_AGENT_PYTHON=/usr/bin/python3.13 \
  sh scripts/install.sh
```

or with curl:

```bash
curl -fsSL https://raw.githubusercontent.com/cvrboni/spark-agent/main/scripts/install.sh \
  | SPARK_AGENT_PYTHON=/usr/bin/python3.13 sh
```
