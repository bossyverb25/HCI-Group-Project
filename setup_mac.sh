#!/usr/bin/env bash
# ==============================================================================
# The Invisible Interpreter (Prototype v1) — macOS Setup & Runner
# ==============================================================================
# Run this script directly from inside the cloned GitHub repository:
#   bash setup_mac.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/Applications/Ollama.app/Contents/Resources:$PATH"

echo "=== [1/4] Checking Python (3.11 / 3.12 / 3.13) on macOS ==="
# Prefer python3.12, python3.11, or python3.13 for pre-built faster-whisper / ctranslate2 wheels
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        if [[ "$PY_VER" == "3.10" || "$PY_VER" == "3.11" || "$PY_VER" == "3.12" || "$PY_VER" == "3.13" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    if command -v brew &>/dev/null; then
        echo "Compatible Python (3.10–3.13) not found. Installing python@3.12 via Homebrew..."
        brew install python@3.12
        export PATH="$(brew --prefix python@3.12)/bin:$PATH"
        PYTHON_BIN="python3.12"
    else
        echo "Error: Compatible Python 3 (3.10–3.13) is not installed and Homebrew was not found."
        echo "Please install Homebrew (https://brew.sh) or Python 3.12 (https://www.python.org/downloads/macos/) and re-run this script."
        exit 1
    fi
fi

echo "Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version))"

if [ -d ".venv" ]; then
    VENV_VER=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
    if [ "$VENV_VER" = "3.14" ]; then
        echo "Removing existing Python 3.14 .venv and rebuilding with $PYTHON_BIN..."
        rm -rf .venv
    fi
fi

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
    if command -v brew &>/dev/null; then
        echo "Installing Ollama via Homebrew..."
        brew install --cask ollama || brew install ollama
    else
        echo "Downloading Ollama for macOS from ollama.com..."
        curl -fL --progress-bar https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama-darwin.zip
        if [ -w "/Applications" ]; then
            unzip -q -o /tmp/Ollama-darwin.zip -d /Applications
            export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"
        else
            mkdir -p "$HOME/Applications"
            unzip -q -o /tmp/Ollama-darwin.zip -d "$HOME/Applications"
            export PATH="$HOME/Applications/Ollama.app/Contents/Resources:$PATH"
        fi
        rm -f /tmp/Ollama-darwin.zip
    fi
else
    echo "Ollama is already installed: $(command -v ollama)"
fi

# Also check user Applications if installed there
if ! command -v ollama &>/dev/null && [ -x "$HOME/Applications/Ollama.app/Contents/Resources/ollama" ]; then
    export PATH="$HOME/Applications/Ollama.app/Contents/Resources:$PATH"
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
