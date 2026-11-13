"""
The Invisible Interpreter — Prototype v1 Server
-----------------------------------------------
Coordinates:
  - System 1: Listening & Transcription (`backend/transcription.py`)
  - System 2: Judgment & Rolling Cache Manager (`backend/judgment.py`)
  - System 3: Suggestion Engine (`backend/suggestion.py`)
and serves the calm-technology Web App UI at `http://localhost:8000`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.judgment import HCILogger, JudgmentCacheManager
from backend.suggestion import SuggestionEngine
from backend.transcription import FasterWhisperEngine, RollingAudioSession, TranscriptState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("interpreter.app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

whisper_engine = FasterWhisperEngine()
suggestion_engine = SuggestionEngine()
hci_logger = HCILogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm Faster-Whisper in background if available so the first speaker doesn't wait
    if whisper_engine.is_available:
        asyncio.create_task(whisper_engine.ensure_loaded())
    yield
    await suggestion_engine.close()


app = FastAPI(
    title="The Invisible Interpreter (Prototype v1)",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> JSONResponse:
    ollama_ok = await suggestion_engine._check_ollama()
    return JSONResponse(
        {
            "status": "ok",
            "whisper_available": whisper_engine.is_available,
            "whisper_model": whisper_engine.model_size if whisper_engine.is_available else "web-speech-hybrid",
            "ollama_available": ollama_ok,
            "ollama_model": suggestion_engine.ollama_model if ollama_ok else None,
            "groq_configured": bool(suggestion_engine.groq_api_key),
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    judgment = JudgmentCacheManager(suggestion_engine=suggestion_engine, hci_logger=hci_logger)

    async def handle_transcript_update(state: TranscriptState) -> None:
        """
        Whenever System 1 updates the rolling transcript:
        1. Immediately send the latest snippet to the Web UI.
        2. Trigger System 2 & 3 to refresh the rolling suggestion cache in the background
           and push the warm cache to the client so clicking 'I'm Stuck' takes 0ms.
        """
        try:
            await ws.send_json(
                {
                    "type": "transcript_update",
                    "last_snippet": state.last_snippet,
                    "full_text": state.full_text,
                    "confidence": state.word_confidence,
                    "engine": state.engine,
                }
            )
            bundle = await judgment.on_transcript_updated(state)
            await ws.send_json(
                {
                    "type": "cache_update",
                    "cache": bundle.to_dict(),
                    "rolling_cache_size": len(judgment._rolling_cache),
                }
            )
        except Exception as exc:
            logger.debug("WebSocket send skipped: %s", exc)

    audio_session = RollingAudioSession(
        whisper_engine=whisper_engine,
        on_transcript_update=handle_transcript_update,
    )
    audio_session.start()

    # Send initial warm cache on connect
    initial_bundle = await judgment.initialize_default_cache()
    await ws.send_json(
        {
            "type": "session_init",
            "session_id": judgment.session_id,
            "whisper_available": whisper_engine.is_available,
            "cache": initial_bundle.to_dict(),
        }
    )

    try:
        while True:
            message = await ws.receive()
            if "bytes" in message and message["bytes"] is not None:
                # Raw 16kHz 16-bit mono PCM audio streamed from browser microphone
                audio_session.append_pcm16_bytes(message["bytes"])

            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "browser_transcript":
                    # Browser Web Speech API or simulated role-play turn
                    text = data.get("text", "")
                    engine_name = data.get("engine", "web-speech-api")
                    await audio_session.ingest_external_transcript(text, engine=engine_name)

                elif msg_type == "im_stuck":
                    mode = data.get("mode", "words")  # "words" or "phrases"
                    context_override = data.get("context", "")
                    result = await judgment.trigger_im_stuck(mode=mode, fallback_context=context_override)
                    await ws.send_json({"type": "im_stuck_result", **result})

                elif msg_type == "suggestion_selected":
                    selected = data.get("text", "")
                    mode = data.get("mode", "")
                    hci_logger.log_event(
                        session_id=judgment.session_id,
                        event_type="suggestion_used",
                        mode=mode,
                        context_snippet=audio_session.state.last_snippet,
                        selected_suggestion=selected,
                        provider=judgment._latest_bundle.provider if judgment._latest_bundle else "",
                    )
                    # Append the selected completion to the rolling context so the next suggestion continues naturally
                    combined = f"{audio_session.state.full_text} {selected}".strip()
                    await audio_session.ingest_external_transcript(combined, engine=audio_session.state.engine)

                elif msg_type == "listening_toggled":
                    is_listening = bool(data.get("listening", False))
                    hci_logger.log_event(
                        session_id=judgment.session_id,
                        event_type="listening_on" if is_listening else "listening_off",
                        context_snippet=audio_session.state.last_snippet,
                    )

                elif msg_type == "clear_context":
                    audio_session.clear()
                    bundle = await judgment.initialize_default_cache()
                    await ws.send_json(
                        {
                            "type": "cache_update",
                            "cache": bundle.to_dict(),
                            "rolling_cache_size": len(judgment._rolling_cache),
                        }
                    )

    except WebSocketDisconnect:
        logger.info("Client disconnected (%s)", judgment.session_id)
    except Exception as exc:
        logger.error("WebSocket error (%s): %s", judgment.session_id, exc)
    finally:
        await audio_session.stop()


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
