"""
System 1: Listening & Transcription System
------------------------------------------
Maintains a rolling 30-second 16kHz mono PCM audio buffer per client session
and uses Faster-Whisper (`base.en` by default) with energy-based
Voice Activity Detection (VAD) to transcribe speech continuously.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("interpreter.transcription")

SAMPLE_RATE = 16000
MAX_BUFFER_SECONDS = 30
TRANSCRIBE_WINDOW_SECONDS = 30  # Transcribe the full 30-second rolling buffer for full context
MIN_AUDIO_SECONDS = 0.6         # Minimum audio before attempting transcription
SILENCE_RMS_THRESHOLD = 0.008   # RMS energy threshold to skip transcribing pure silence


@dataclass
class TranscriptState:
    """Tracks rolling transcription for the last 30 seconds."""
    full_text: str = ""
    last_snippet: str = ""
    word_confidence: float = 0.90
    engine: str = "faster-whisper"
    updated_at: float = field(default_factory=time.time)

    @staticmethod
    def extract_snippet(text: str, max_words: int = 24) -> str:
        """Return up to the last 24 words so the user sees a larger context window."""
        words = text.strip().split()
        if not words:
            return ""
        if len(words) <= max_words:
            return " ".join(words)
        return "... " + " ".join(words[-max_words:])


class FasterWhisperEngine:
    """Singleton wrapper around Faster-Whisper so the model loads once."""

    def __init__(self) -> None:
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
        self.device = os.getenv("WHISPER_DEVICE", "auto")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self._model: Optional[WhisperModel] = None
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            logger.info(
                "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                ),
            )
            logger.info("Faster-Whisper model '%s' loaded successfully.", self.model_size)
            return self._model

    async def transcribe_window(self, audio_float32: np.ndarray) -> tuple[str, float]:
        """
        Transcribe a 16kHz float32 numpy array using Faster-Whisper.
        Returns (transcribed_text, avg_confidence).
        """
        model = await self.ensure_loaded()
        loop = asyncio.get_running_loop()

        def _run_whisper() -> tuple[str, float]:
            segments, _ = model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
                condition_on_previous_text=True,
            )
            texts: list[str] = []
            probs: list[float] = []
            for seg in segments:
                cleaned = seg.text.strip()
                if cleaned:
                    texts.append(cleaned)
                    avg_logprob = getattr(seg, "avg_logprob", -0.25)
                    probs.append(float(np.clip(np.exp(avg_logprob), 0.1, 0.99)))
            joined = " ".join(texts).strip()
            confidence = float(np.mean(probs)) if probs else 0.85
            return joined, round(confidence, 2)

        return await loop.run_in_executor(None, _run_whisper)


class RollingAudioSession:
    """
    Manages the rolling 30-second audio buffer for a single connected client
    and triggers periodic Faster-Whisper transcriptions when new speech arrives.
    """

    def __init__(
        self,
        whisper_engine: FasterWhisperEngine,
        on_transcript_update: Callable[[TranscriptState], Awaitable[None]],
    ) -> None:
        self.whisper_engine = whisper_engine
        self.on_transcript_update = on_transcript_update

        self.max_samples = SAMPLE_RATE * MAX_BUFFER_SECONDS
        self._buffer = np.zeros(0, dtype=np.float32)
        self._has_new_speech = False
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.state = TranscriptState(engine=f"faster-whisper ({whisper_engine.model_size})")
        self._history: deque[tuple[float, str]] = deque()

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._transcription_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def clear(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._history.clear()
        self.state.full_text = ""
        self.state.last_snippet = ""

    def append_pcm16_bytes(self, pcm_bytes: bytes) -> None:
        """
        Receive raw 16-bit signed integer PCM audio at 16kHz from the browser WebSocket,
        convert to float32 [-1.0, 1.0], and append to the rolling 30s buffer.
        """
        if not pcm_bytes:
            return
        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if int16_samples.size == 0:
            return
        float_samples = int16_samples.astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(np.square(float_samples))))
        if rms >= SILENCE_RMS_THRESHOLD:
            self._has_new_speech = True

        self._buffer = np.concatenate((self._buffer, float_samples))
        if self._buffer.size > self.max_samples:
            self._buffer = self._buffer[-self.max_samples :]

    async def append_selected_suggestion(self, text: str) -> None:
        """
        When the user clicks a suggested completion on screen, append it to the
        current context so the next Ollama suggestion continues from it.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        now = time.time()
        self.state.full_text = cleaned
        self.state.last_snippet = TranscriptState.extract_snippet(cleaned)
        self.state.updated_at = now
        await self.on_transcript_update(self.state)

    async def _transcription_loop(self) -> None:
        """
        Every ~0.75 seconds, if new voice activity was detected in the rolling audio
        buffer, transcribe the rolling window using Faster-Whisper.
        """
        window_samples = SAMPLE_RATE * TRANSCRIBE_WINDOW_SECONDS
        min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)

        while self._running:
            await asyncio.sleep(0.75)
            if not self._has_new_speech:
                continue
            if self._buffer.size < min_samples:
                continue

            self._has_new_speech = False
            audio_slice = self._buffer[-window_samples:].copy()

            try:
                text, confidence = await self.whisper_engine.transcribe_window(audio_slice)
                if text and text != self.state.full_text:
                    self.state.full_text = text
                    self.state.last_snippet = TranscriptState.extract_snippet(text)
                    self.state.word_confidence = confidence
                    self.state.engine = f"faster-whisper ({self.whisper_engine.model_size})"
                    self.state.updated_at = time.time()
                    await self.on_transcript_update(self.state)
            except Exception as exc:
                logger.error("Error in Faster-Whisper transcription loop: %s", exc)
