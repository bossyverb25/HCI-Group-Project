#!/usr/bin/env bash
# ==============================================================================
# The Invisible Interpreter (Prototype v1) — Fedora Linux Setup & Runner
# ==============================================================================
# Run this script directly from inside the cloned GitHub repository:
#   bash setup_fedora.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/4] Checking Python (3.11 / 3.12 / 3.13) on Fedora ==="
# Fedora 41+ may ship Python 3.13/3.14; prefer python3.12 or python3.11 if available for faster-whisper wheels
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [ "$PY_VER" != "3.14" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Python 3.14 detected without Python 3.12 installed. Installing python3.12 via dnf..."
    sudo dnf install -y python3.12
    PYTHON_BIN="python3.12"
fi

echo "Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version))"

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== [2/4] Checking Ollama installation ==="
if ! command -v ollama &>/dev/null; then
    echo "Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed: $(command -v ollama)"
fi

echo ""
echo "=== [3/4] Starting Ollama service & pulling model ($OLLAMA_MODEL) ==="
if ! curl -fsS http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo "Starting ollama serve in the background..."
    nohup ollama serve >/tmp/ollama_server.log 2>&1 &
    for _ in {1..20}; do
        if curl -fsS http://127.0.0.1:11434/api/tags &>/dev/null; then
            break
        fi
        sleep 0.5
    done
fi

echo "Ensuring model '$OLLAMA_MODEL' is downloaded..."
ollama pull "$OLLAMA_MODEL"

echo ""
echo "=== [4/4] Launching The Invisible Interpreter on http://localhost:8000 ==="
python app.py
