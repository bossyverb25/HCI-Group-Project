#!/usr/bin/env bash
# ==============================================================================
# The Invisible Interpreter (Prototype v1) — Ubuntu Lab Machine Setup & Runner
# ==============================================================================
# Run this script directly from inside the cloned GitHub repository:
#   bash setup_ubuntu.sh
#
# It works both on personal Ubuntu machines (with sudo) AND on restricted
# university Ubuntu lab computers (without sudo, installing Ollama to ~/.local).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/4] Checking Python 3 environment ==="
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    if ! python3 -m venv .venv 2>/dev/null; then
        echo "python3-venv missing; creating virtualenv without pip and bootstrapping..."
        python3 -m venv --without-pip .venv
        source .venv/bin/activate
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3
    fi
fi

source .venv/bin/activate
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== [2/4] Checking Ollama installation ==="
if ! command -v ollama &>/dev/null; then
    echo "Ollama not found. Installing Ollama..."
    # If user has sudo access, use standard installer; otherwise install binary into ~/.local (for lab PCs)
    if sudo -n true 2>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "No root/sudo privileges detected (University Lab Mode) — installing Ollama to ~/.local ..."
        mkdir -p "$HOME/.local/bin"
        curl -fL --progress-bar https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama-linux-amd64.tgz
        tar -xzf /tmp/ollama-linux-amd64.tgz -C "$HOME/.local"
        rm -f /tmp/ollama-linux-amd64.tgz
    fi
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
python3 app.py
