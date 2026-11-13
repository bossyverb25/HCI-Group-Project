"""
System 2: Judgment & Rolling Cache Manager + HCI Evaluation Logger
------------------------------------------------------------------
In Prototype v1:
- Continuously receives rolling transcript updates from System 1.
- Pre-generates and stores a rolling 30-second cache of phrase completion
  suggestions via System 3 so that when the user presses "I'm Stuck",
  the suggestions are already cached (0ms latency).
- Supports dynamic suggestion count (default 6, range 1–10) and detail level
  (default 6 words per suggestion, range 1–12).
- Automatically logs all events to anonymised CSV and JSONL files in `logs/`.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable, Optional

from backend.suggestion import SuggestionBundle, SuggestionEngine
from backend.transcription import TranscriptState

logger = logging.getLogger("interpreter.judgment")

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CSV_LOG_PATH = LOGS_DIR / "hci_events.csv"
JSONL_LOG_PATH = LOGS_DIR / "hci_events.jsonl"


class HCILogger:
    """Writes anonymised interaction logs to CSV and JSONL for coursework evaluation."""

    CSV_HEADERS = [
        "timestamp_iso",
        "session_id",
        "event_type",
        "num_suggestions",
        "detail_words",
        "context_snippet",
        "suggestions_shown",
        "provider",
        "cache_latency_ms",
    ]

    def __init__(self) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if not CSV_LOG_PATH.exists():
            with open(CSV_LOG_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def log_event(
        self,
        session_id: str,
        event_type: str,
        num_suggestions: int = 6,
        detail_words: int = 6,
        context_snippet: str = "",
        suggestions_shown: Optional[list[str]] = None,
        provider: str = "",
        cache_latency_ms: float = 0.0,
    ) -> dict:
        now_iso = datetime.now(timezone.utc).isoformat()
        suggestions_list = suggestions_shown or []
        record = {
            "timestamp_iso": now_iso,
            "session_id": session_id,
            "event_type": event_type,
            "num_suggestions": num_suggestions,
            "detail_words": detail_words,
            "context_snippet": context_snippet,
            "suggestions_shown": " | ".join(suggestions_list),
            "provider": provider,
            "cache_latency_ms": round(cache_latency_ms, 1),
        }

        try:
            with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([record[h] for h in self.CSV_HEADERS])

            with open(JSONL_LOG_PATH, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to write HCI evaluation log: %s", exc)

        return record


class JudgmentCacheManager:
    """
    Manages the rolling 30-second suggestion cache and handles 'I'm Stuck'
    triggers for a single participant session.
    """

    def __init__(
        self,
        suggestion_engine: SuggestionEngine,
        hci_logger: HCILogger,
        on_cache_ready: Optional[Callable[[SuggestionBundle, int], Awaitable[None]]] = None,
    ) -> None:
        self.session_id = f"anon-{uuid.uuid4().hex[:8]}"
        self.suggestion_engine = suggestion_engine
        self.hci_logger = hci_logger
        self.on_cache_ready = on_cache_ready
        self.suggestion_count: int = 6
        self.detail_words: int = 6

        self._rolling_cache: deque[SuggestionBundle] = deque(maxlen=30)
        self._latest_bundle: Optional[SuggestionBundle] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._pending_context: Optional[str] = None
        self._last_context_generated: str = ""
        self._last_count_generated: int = 0
        self._last_detail_generated: int = 0

    async def update_settings(
        self, count: Optional[int] = None, detail_words: Optional[int] = None, current_context: str = ""
    ) -> None:
        """Update the user's desired suggestion count (1..10) and detail words (1..12) and refresh cache."""
        changed = False
        if count is not None:
            new_count = max(1, min(10, int(count)))
            if new_count != self.suggestion_count:
                self.suggestion_count = new_count
                changed = True
        if detail_words is not None:
            new_detail = max(1, min(12, int(detail_words)))
            if new_detail != self.detail_words:
                self.detail_words = new_detail
                changed = True

        if changed and (
            self.suggestion_count > self._last_count_generated
            or self.detail_words != self._last_detail_generated
        ):
            ctx = current_context.strip() or self._last_context_generated or "Hello, I would like to find the"
            self._pending_context = ctx
            self._last_context_generated = ""
            if self._worker_task is None or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._process_pending_loop())

    async def initialize_default_cache(self) -> SuggestionBundle:
        """Seed an initial suggestion bundle so the cache is warm before speech even starts."""
        bundle = await self.suggestion_engine.generate(
            "Hello, I would like to find the",
            count=self.suggestion_count,
            detail_words=self.detail_words,
        )
        self._latest_bundle = bundle
        self._last_context_generated = "Hello, I would like to find the"
        self._last_count_generated = len(bundle.phrases)
        self._last_detail_generated = self.detail_words
        self._rolling_cache.append(bundle)
        return bundle

    async def on_transcript_updated(self, state: TranscriptState) -> None:
        """
        Called whenever System 1 produces a new transcript snippet.
        Queues the newest full transcript for background generation.
        """
        context = state.full_text.strip()
        if not context or (
            context == self._last_context_generated
            and self._last_count_generated >= self.suggestion_count
            and self._last_detail_generated == self.detail_words
        ):
            return

        self._pending_context = context
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_pending_loop())

    async def _process_pending_loop(self) -> None:
        """
        Processes only the MOST RECENT pending transcript context.
        """
        while self._pending_context is not None:
            context = self._pending_context
            self._pending_context = None
            target_count = self.suggestion_count
            target_detail = self.detail_words
            if (
                context == self._last_context_generated
                and self._last_count_generated >= target_count
                and self._last_detail_generated == target_detail
            ):
                continue

            bundle = await self.suggestion_engine.generate(
                context, count=target_count, detail_words=target_detail
            )
            self._latest_bundle = bundle
            self._last_context_generated = context
            self._last_count_generated = len(bundle.phrases)
            self._last_detail_generated = target_detail

            now = time.time()
            while self._rolling_cache and (now - self._rolling_cache[0].generated_at) > 30.0:
                self._rolling_cache.popleft()
            self._rolling_cache.append(bundle)

            if self.on_cache_ready is not None:
                await self.on_cache_ready(bundle, len(self._rolling_cache))

    async def trigger_im_stuck(
        self,
        count: Optional[int] = None,
        detail_words: Optional[int] = None,
        fallback_context: str = "",
    ) -> dict:
        """
        Triggered when the user clicks 'I'm Stuck' (or presses Spacebar).
        """
        t0 = time.perf_counter()
        if count is not None:
            self.suggestion_count = max(1, min(10, int(count)))
        if detail_words is not None:
            self.detail_words = max(1, min(12, int(detail_words)))

        if self._worker_task is not None and not self._worker_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=1.2)
            except asyncio.TimeoutError:
                pass

        if (
            self._latest_bundle is None
            or len(self._latest_bundle.phrases) < self.suggestion_count
            or self._last_detail_generated != self.detail_words
        ):
            ctx = fallback_context.strip() or self._last_context_generated or "Hello, I would like to find the"
            self._latest_bundle = await self.suggestion_engine.generate(
                ctx, count=self.suggestion_count, detail_words=self.detail_words
            )
            self._last_context_generated = ctx
            self._last_count_generated = len(self._latest_bundle.phrases)
            self._last_detail_generated = self.detail_words
            self._rolling_cache.append(self._latest_bundle)

        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        bundle = self._latest_bundle
        items = bundle.phrases[: self.suggestion_count]

        self.hci_logger.log_event(
            session_id=self.session_id,
            event_type="im_stuck_triggered",
            num_suggestions=self.suggestion_count,
            detail_words=self.detail_words,
            context_snippet=TranscriptState.extract_snippet(bundle.context_snippet),
            suggestions_shown=[item.text for item in items],
            provider=bundle.provider,
            cache_latency_ms=retrieval_ms,
        )

        return {
            "context_snippet": TranscriptState.extract_snippet(bundle.context_snippet),
            "suggestions": [
                {"text": item.text, "confidence": item.confidence, "score": item.score}
                for item in items
            ],
            "all_cached": bundle.to_dict(),
            "cache_retrieval_ms": round(retrieval_ms, 2),
            "rolling_cache_size": len(self._rolling_cache),
        }
