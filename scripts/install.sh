#!/bin/sh
set -eu

REPO_URL="${SPARK_AGENT_REPO:-https://github.com/cvrboni/spark-agent.git}"
REF="${SPARK_AGENT_REF:-main}"
INSTALL_DIR="${SPARK_AGENT_INSTALL_DIR:-$HOME/.local/share/spark-agent}"
BIN_DIR="${SPARK_AGENT_BIN_DIR:-$HOME/.local/bin}"
PYTHON_BIN="${SPARK_AGENT_PYTHON:-}"

say() {
  printf '%s\n' "$1"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

find_python() {
  if [ -n "$PYTHON_BIN" ]; then
    command -v "$PYTHON_BIN"
    return
  fi
  if need_cmd python3.13; then
    command -v python3.13
    return
  fi
  if need_cmd python3; then
    command -v python3
    return
  fi
  return 1
}

if ! PYTHON_BIN="$(find_python)"; then
  say "Python 3.13+ is required."
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VERSION" in
  3.13|3.14|3.15|3.16|3.17|3.18|3.19) ;;
  *)
    say "Python 3.13+ is required. Found Python $PY_VERSION at $PYTHON_BIN."
    say "Set SPARK_AGENT_PYTHON=/path/to/python3.13 and retry if Python 3.13 is installed elsewhere."
    exit 1
    ;;
esac

mkdir -p "$BIN_DIR"

install_with_pipx() {
  say "Installing SparkAgent with pipx..."
  pipx install --force "git+$REPO_URL@$REF" || return 1
}

install_with_uv() {
  say "Installing SparkAgent with uv tool using $PYTHON_BIN..."
  uv tool install --force --python "$PYTHON_BIN" "git+$REPO_URL@$REF" || return 1
}

install_with_venv() {
  say "Installing SparkAgent into $INSTALL_DIR..."
  if ! "$PYTHON_BIN" -m venv "$INSTALL_DIR"; then
    say "python3 -m venv failed. On Debian/Ubuntu this usually means python3.13-venv is missing."
    return 1
  fi
  "$INSTALL_DIR/bin/python" -m pip install --upgrade pip || return 1
  "$INSTALL_DIR/bin/python" -m pip install --force-reinstall "git+$REPO_URL@$REF" || return 1
  ln -sf "$INSTALL_DIR/bin/spark-agent" "$BIN_DIR/spark-agent" || return 1
}

install_with_user_pip() {
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    return 1
  fi
  say "Installing SparkAgent with python3 -m pip --user..."
  "$PYTHON_BIN" -m pip install --user --force-reinstall "git+$REPO_URL@$REF" || return 1
  USER_BASE="$("$PYTHON_BIN" -m site --user-base)"
  if [ -x "$USER_BASE/bin/spark-agent" ]; then
    ln -sf "$USER_BASE/bin/spark-agent" "$BIN_DIR/spark-agent" || return 1
  fi
}

if need_cmd pipx; then
  install_with_pipx
elif need_cmd uv; then
  install_with_uv
elif install_with_venv; then
  :
elif install_with_user_pip; then
  :
else
  say "Could not install SparkAgent automatically."
  say "Install one of these and retry:"
  say "  sudo apt install pipx"
  say "  sudo apt install python3.13-venv"
  say "  python3 -m pip install --user pipx"
  say "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v spark-agent >/dev/null 2>&1; then
  say "SparkAgent installed, but $BIN_DIR is not on PATH."
  say "Add this to your shell profile:"
  say "  export PATH=\"$BIN_DIR:\$PATH\""
fi

say "Creating default config if missing..."
if command -v spark-agent >/dev/null 2>&1; then
  spark-agent init
elif [ -x "$BIN_DIR/spark-agent" ]; then
  "$BIN_DIR/spark-agent" init
else
  say "Could not find spark-agent on PATH. Try opening a new shell or add $BIN_DIR to PATH."
  exit 1
fi

say "Done. Run:"
say "  spark-agent doctor"
say "  spark-agent run \"Inspect this repository and summarize the architecture\""
