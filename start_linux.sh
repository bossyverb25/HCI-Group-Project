#!/usr/bin/env bash
# ==============================================================================
# The Invisible Interpreter — Linux (Ubuntu & Fedora) Quick Start Script
# ==============================================================================
# Run from inside the repository:
#   bash start_linux.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Please run setup_ubuntu.sh or setup_fedora.sh first."
    exit 1
fi

# Ensure Ollama server is running in the background
if command -v ollama &>/dev/null; then
    if ! curl -fsS http://127.0.0.1:11434/api/tags &>/dev/null; then
        echo "Starting Ollama background server..."
        nohup ollama serve >/tmp/ollama_server.log 2>&1 &
        for _ in {1..20}; do
            if curl -fsS http://127.0.0.1:11434/api/tags &>/dev/null; then
                break
            fi
            sleep 0.3
        done
    fi
fi

echo "Activating .venv and starting The Invisible Interpreter on http://localhost:8000 ..."
source .venv/bin/activate
python app.py
