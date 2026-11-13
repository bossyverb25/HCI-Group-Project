"""
The Invisible Interpreter — Prototype v1 Server
-----------------------------------------------
Coordinates:
  - System 1: Listening & Transcription via Faster-Whisper (`backend/transcription.py`)
  - System 2: Judgment & Rolling Cache Manager (`backend/judgment.py`)
  - System 3: Suggestion Engine via Local Ollama (`backend/suggestion.py`)
and serves the Web App UI at `http://localhost:8000`.
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
from backend.suggestion import SuggestionBundle, SuggestionEngine
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
    asyncio.create_task(whisper_engine.ensure_loaded())
    yield
    await suggestion_engine.close()


app = FastAPI(
    title="The Invisible Interpreter",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> JSONResponse:
    ollama_ok = await suggestion_engine.check_ollama()
    return JSONResponse(
        {
            "status": "ok",
            "whisper_model": whisper_engine.model_size,
            "ollama_available": ollama_ok,
            "ollama_model": suggestion_engine.ollama_model,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    async def push_cache_ready(bundle: SuggestionBundle, cache_size: int) -> None:
        try:
            await ws.send_json(
                {
                    "type": "cache_update",
                    "cache": bundle.to_dict(),
                    "rolling_cache_size": cache_size,
                }
            )
        except Exception as exc:
            logger.debug("WebSocket cache_update skipped: %s", exc)

    judgment = JudgmentCacheManager(
        suggestion_engine=suggestion_engine,
        hci_logger=hci_logger,
        on_cache_ready=push_cache_ready,
    )

    async def handle_transcript_update(state: TranscriptState) -> None:
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
            await judgment.on_transcript_updated(state)
        except Exception as exc:
            logger.debug("WebSocket send skipped: %s", exc)

    audio_session = RollingAudioSession(
        whisper_engine=whisper_engine,
        on_transcript_update=handle_transcript_update,
    )
    audio_session.start()

    initial_bundle = await judgment.initialize_default_cache()
    await ws.send_json(
        {
            "type": "session_init",
            "session_id": judgment.session_id,
            "whisper_model": whisper_engine.model_size,
            "cache": initial_bundle.to_dict(),
        }
    )

    try:
        while True:
            message = await ws.receive()
            if "bytes" in message and message["bytes"] is not None:
                audio_session.append_pcm16_bytes(message["bytes"])

            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "im_stuck":
                    count = data.get("count", judgment.suggestion_count)
                    detail_words = data.get("detail_words", judgment.detail_words)
                    context_override = data.get("context", "")
                    result = await judgment.trigger_im_stuck(
                        count=count, detail_words=detail_words, fallback_context=context_override
                    )
                    await ws.send_json({"type": "im_stuck_result", **result})

                elif msg_type == "update_settings":
                    count = data.get("count")
                    detail_words = data.get("detail_words")
                    await judgment.update_settings(
                        count=count,
                        detail_words=detail_words,
                        current_context=audio_session.state.full_text,
                    )

                elif msg_type == "listening_toggled":
                    is_listening = bool(data.get("listening", False))
                    hci_logger.log_event(
                        session_id=judgment.session_id,
                        event_type="listening_on" if is_listening else "listening_off",
                        num_suggestions=judgment.suggestion_count,
                        detail_words=judgment.detail_words,
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
