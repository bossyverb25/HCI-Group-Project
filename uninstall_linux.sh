#!/usr/bin/env bash
# ==============================================================================
# The Invisible Interpreter — Linux (Ubuntu & Fedora) Uninstaller & Cleanup
# ==============================================================================
# Run from inside the repository:
#   bash uninstall_linux.sh
#
# Removes:
#   1. Downloaded Ollama models (llama3.2:3b, llama3.2:1b)
#   2. Cached Faster-Whisper model weights (~/.cache/huggingface/hub/models--Systran--faster-whisper*)
#   3. Local Python virtual environment (.venv)
#   4. (Optional prompt) Uninstalls Ollama binary/service and ~/.ollama directory
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.local/bin:$PATH"

echo "=== [1/4] Removing downloaded Ollama models (llama3.2:3b, llama3.2:1b) ==="
if command -v ollama &>/dev/null; then
    ollama rm llama3.2:3b 2>/dev/null || true
    ollama rm llama3.2:1b 2>/dev/null || true
fi

echo ""
echo "=== [2/4] Removing Faster-Whisper cached model weights ==="
HF_CACHE="$HOME/.cache/huggingface/hub"
if [ -d "$HF_CACHE" ]; then
    rm -rf "$HF_CACHE"/models--Systran--faster-whisper*
    echo "Removed Faster-Whisper cache from $HF_CACHE"
fi

echo ""
echo "=== [3/4] Removing Python virtual environment (.venv) ==="
if [ -d ".venv" ]; then
    rm -rf .venv
    echo "Removed .venv"
fi

echo ""
read -r -p "Do you also want to completely uninstall Ollama and delete ~/.ollama? [y/N] " response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "=== [4/4] Uninstalling Ollama ==="
    pkill -f "ollama" 2>/dev/null || true
    rm -rf "$HOME/.ollama"
    rm -f "$HOME/.local/bin/ollama"
    rm -rf "$HOME/.local/lib/ollama"
    if sudo -n true 2>/dev/null; then
        sudo systemctl stop ollama 2>/dev/null || true
        sudo systemctl disable ollama 2>/dev/null || true
        sudo rm -f /etc/systemd/system/ollama.service
        sudo rm -f /usr/local/bin/ollama
        sudo rm -rf /usr/local/lib/ollama
        sudo rm -rf /usr/share/ollama
    fi
    echo "Uninstalled Ollama and removed model storage."
else
    echo "Kept Ollama binary installed (only removed project models and .venv)."
fi

echo ""
echo "Cleanup complete!"
