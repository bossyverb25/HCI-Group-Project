# Invisible Interpreter - Prototype v1

An adaptive, ambient conversational assistant built for **LZSCC.241 Human-Computer Interaction (Project 12: The Invisible Interpreter)**.

Prototype v1 implements the core 3-system architecture on a single machine via a FastAPI + WebSocket backend and a calm, device-agnostic Web App frontend:

1. **System 1 — Listening & Transcription (`backend/transcription.py`)**:
   - Captures 16 kHz mono PCM audio in the browser via the Web Audio API (`static/audio-processor.js`) and streams it over a WebSocket to the Python backend.
   - Uses a rolling 30-second audio buffer + Voice Activity Detection (energy & silence gating) with **Faster-Whisper** (`base.en` by default) to continuously transcribe speech in real time.

2. **System 2 — Judgment & Rolling Cache Manager (`backend/judgment.py`)**:
   - Continuously watches transcript updates in the background and pre-generates/caches phrase completions for the latest 30 seconds of context.
   - Exposes a manual **"I'm Stuck"** button (and `Spacebar` trigger) in Prototype v1 (standing in for the future automated confidence judgment / brain-sensing signal), delivering **0 ms perceived latency** from the rolling cache.
   - Includes a collapsible Settings menu (`⚙️`) with two sliders: **Suggestions** (1–10, default 6) and **Detail** (1–12 words per suggestion, default 6).
   - Automatically logs all session events, transcript updates, cache latencies, and button triggers to anonymised CSV and JSONL files in `logs/` for HCI evaluation.

3. **System 3 — Suggestion Engine (`backend/suggestion.py`)**:
   - Uses a local **Ollama** LLM (`llama3.2:3b` by default, at `http://127.0.0.1:11434`) with raw token continuation to generate context-aware English phrase completions with confidence ratings (`high` / `medium`) to satisfy the **"Honest about uncertainty"** requirement.

---

## One-Command Setup & Run Scripts

Clone the repository and run the setup script for your operating system. Each script automatically checks/installs a compatible Python version, creates `.venv`, installs `requirements.txt`, installs & starts **Ollama**, pulls **`llama3.2:3b`**, and starts the web server on **`http://localhost:8000`**.

### Windows (PowerShell or Double-Click)
- **Option A:** Double-click **`setup_windows.bat`** in File Explorer.
- **Option B:** Run in PowerShell:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
  ```

### Ubuntu / Linux (Personal or University Lab Computers)
```bash
bash setup_ubuntu.sh
```
*(Works with or without `sudo` privileges on university lab machines).*

### Fedora Linux
```bash
bash setup_fedora.sh
```

---

## AI-Use & Simulation Statement (Coursework Requirement)
- **Simulated Components (Prototype v1):** In accordance with the Project 12 brief, the "confusion detection" signal is simulated in Prototype v1 via a manual **"I'm Stuck"** button (`Spacebar`) (planned to be replaced in the full version).
- **AI Tools Used:** Implemented with assistance from Antigravity (Gemini) for FastAPI/WebSocket scaffolding, Web Audio PCM capture, and Faster-Whisper / Ollama integration. Design was still done on paper by @snowfalllll.
