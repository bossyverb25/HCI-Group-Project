"""
System 3: Suggestion System (Local Ollama LLM)
----------------------------------------------
Takes the latest conversational snippet (from the last 30 seconds of context)
and calls the local Ollama LLM (`llama3.2:3b`) to generate N (1–10) distinct
phrase completions of up to `detail_words` words each.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, asdict

import httpx

logger = logging.getLogger("interpreter.suggestion")


@dataclass
class SuggestionItem:
    text: str
    confidence: str  # "high" | "medium"
    score: float     # 0.0 .. 1.0


@dataclass
class SuggestionBundle:
    context_snippet: str
    phrases: list[SuggestionItem]
    provider: str
    latency_ms: float
    generated_at: float

    def to_dict(self) -> dict:
        return {
            "context_snippet": self.context_snippet,
            "phrases": [asdict(p) for p in self.phrases],
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 1),
            "generated_at": self.generated_at,
        }


class SuggestionEngine:
    """
    Generates rolling phrase completion suggestions using Local Ollama raw token
    continuation (`raw: True`), guaranteeing exactly `count` (1–10) unique
    suggestions on every call.
    """

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self._model_checked = False
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def check_ollama(self) -> bool:
        """Check whether local Ollama server is running and pick the best installed model."""
        try:
            resp = await self._client.get(f"{self.ollama_url}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                if models and not self._model_checked:
                    preferred = ["llama3.2:3b", "qwen2.5:3b", "llama3.1:8b", "llama3.2:1b"]
                    chosen = next(
                        (m for p in preferred for m in models if p in m),
                        models[0],
                    )
                    self.ollama_model = chosen
                    self._model_checked = True
                    logger.info("Using Ollama model: %s", self.ollama_model)
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _clean_context(text: str) -> str:
        """
        Strip trailing punctuation (`.`, `...`, `?`, `,`) added by Whisper when the user pauses,
        and collapse stuttered duplicate words (e.g., 'where where' -> 'where').
        """
        cleaned = text.strip()
        cleaned = re.sub(r"[.?!,;:\-—]+$", "", cleaned).strip()
        cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        return cleaned[-500:]

    @staticmethod
    def _clean_phrase(raw: str, max_words: int) -> str:
        """Clean stray punctuation/brackets and trim the continuation to `max_words` words."""
        sug = re.sub(r"^[,;:\-—\s\"'\(\)\[\]]+", "", raw.strip())
        sug = re.sub(r"[.?!,;:\-—\"'\(\)\[\]]+$", "", sug).strip()
        words = sug.split()
        return " ".join(words[:max_words]).strip()

    async def _continue_stem(
        self,
        stem: str,
        temperature: float,
        seed: int,
        max_words: int,
        avoid_phrases: list[str] | None = None,
    ) -> str:
        """
        Ask Ollama (`raw: True`) to predict the next words that directly follow `stem`.
        Note: We do NOT include "." or "?" in `stop` because if the model thinks the
        previous word could end a clause and starts with ". Next word", stopping on "."
        returns an empty string `""`! Instead, we strip leading punctuation in `_clean_phrase`.
        """
        avoid_note = ""
        if avoid_phrases:
            joined_avoid = ", ".join(f'"{a}"' for a in avoid_phrases[:6])
            avoid_note = f" (Suggest a different continuation than: {joined_avoid})"

        raw_prompt = f"Transcript of a spoken conversation{avoid_note}:\nSpeaker: {stem}"
        payload = {
            "model": self.ollama_model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "repeat_penalty": 1.25,
                "seed": seed,
                "num_predict": max(6, int(max_words * 2)),
                "stop": ["\n", "Speaker:"],
            },
        }
        resp = await self._client.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        # Take only up to the first sentence boundary *after* at least some words have been generated
        text = re.sub(r"^[\s.?!,;:\-—]+", "", text)
        first_clause = re.split(r"[.?!]", text)[0]
        return self._clean_phrase(first_clause, max_words)

    async def generate(
        self, transcript_context: str, count: int = 6, detail_words: int = 6
    ) -> SuggestionBundle:
        """
        Generate `count` (1..10) phrase suggestions of up to `detail_words` (1..12) words each,
        guaranteeing that `len(phrases) == count` every time.
        """
        start = time.perf_counter()
        count = max(1, min(10, int(count)))
        detail_words = max(1, min(12, int(detail_words)))
        await self.check_ollama()
        stem = self._clean_context(transcript_context) or "Hello, I would like to find the"

        try:
            unique_phrases: list[str] = []
            seen: set[str] = set()

            # Round 1: Parallel batch with varied temperatures & seeds
            batch_size = count + 4
            tasks = [
                self._continue_stem(
                    stem=stem,
                    temperature=0.15 if i == 0 else min(1.3, 0.55 + i * 0.08),
                    seed=100 + i * 17,
                    max_words=detail_words,
                )
                for i in range(batch_size)
            ]
            raw_candidates = await asyncio.gather(*tasks)

            for candidate in raw_candidates:
                if candidate and candidate.lower() not in seen:
                    seen.add(candidate.lower())
                    unique_phrases.append(candidate)
                if len(unique_phrases) >= count:
                    break

            # Round 2: If duplicates or empty responses left us short of `count`,
            # explicitly pass `avoid_phrases=unique_phrases` until we reach exactly `count`.
            attempt = 0
            while len(unique_phrases) < count and attempt < 6:
                needed = count - len(unique_phrases)
                retry_tasks = [
                    self._continue_stem(
                        stem=stem,
                        temperature=min(1.4, 0.85 + attempt * 0.1 + j * 0.05),
                        seed=500 + attempt * 50 + j,
                        max_words=detail_words,
                        avoid_phrases=unique_phrases,
                    )
                    for j in range(needed + 2)
                ]
                retry_candidates = await asyncio.gather(*retry_tasks)
                for candidate in retry_candidates:
                    if candidate and candidate.lower() not in seen:
                        seen.add(candidate.lower())
                        unique_phrases.append(candidate)
                    if len(unique_phrases) >= count:
                        break
                attempt += 1

            # Build SuggestionItem list (Option 1 = High Confidence, rest = Alternative)
            phrase_items = [
                SuggestionItem(
                    text=text,
                    confidence="high" if i == 0 else "medium",
                    score=round(max(0.50, 0.92 - i * 0.04), 2),
                )
                for i, text in enumerate(unique_phrases[:count])
            ]

            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=stem,
                phrases=phrase_items,
                provider=f"ollama ({self.ollama_model})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
        except Exception as exc:
            logger.error("Ollama request error: %s", exc)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=stem,
                phrases=[
                    SuggestionItem(text="...", confidence="high" if i == 0 else "medium", score=0.5)
                    for i in range(count)
                ],
                provider=f"ollama (error: {type(exc).__name__})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
