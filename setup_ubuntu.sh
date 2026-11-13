#!/usr/bin/env bash
set -e

echo "=== Setting up The Invisible Interpreter (Prototype v1) ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! Starting server on http://localhost:8000 ..."
python3 app.py
