"""
System 3: Suggestion System (Local Ollama LLM)
----------------------------------------------
Takes the latest conversational snippet (from the last 30 seconds of context)
and calls the local Ollama LLM (`llama3.2:3b`) to generate up to N (1–10)
distinct phrase completions of configurable word length (`detail_words`)
that naturally continue after the speaker's last word.
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
    Generates rolling phrase completion suggestions using Local Ollama.
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
                    chosen = None
                    for p in preferred:
                        if any(p in m for m in models):
                            chosen = next(m for m in models if p in m)
                            break
                    self.ollama_model = chosen or models[0]
                    self._model_checked = True
                    logger.info("Using Ollama model: %s", self.ollama_model)
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _clean_context(text: str) -> str:
        """
        1. Strip trailing punctuation (`.`, `...`, `?`, `,`) added by Whisper when the user pauses.
        2. Collapse immediate stuttered duplicate words at the end (e.g., 'where where' -> 'where').
        """
        cleaned = text.strip()
        cleaned = re.sub(r"[.?!,;:\-—]+$", "", cleaned).strip()
        cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        return cleaned[-500:]

    @staticmethod
    def _clean_phrase(raw: str, context: str, max_words: int = 6) -> str:
        """Clean punctuation, strip repeated words from the end of context, and trim to `max_words`."""
        sug = re.sub(r"^[\d\.\)\-\*\s,;:\"'—]+", "", raw.strip())
        sug = re.sub(r"[.?!,;:\-—\"']+$", "", sug).strip()
        if not sug:
            return ""

        ctx_words = re.findall(r"[a-zA-Z']+", context.lower())
        sug_words = sug.split()
        if ctx_words and sug_words:
            for k in range(min(4, len(ctx_words), len(sug_words)), 0, -1):
                ctx_tail = [w.lower().strip(".,?!") for w in ctx_words[-k:]]
                sug_head = [w.lower().strip(".,?!") for w in sug_words[:k]]
                if ctx_tail == sug_head:
                    sug_words = sug_words[k:]
                    break

        return " ".join(sug_words[:max_words]).strip()

    async def _raw_continue_single(
        self, stem: str, temperature: float, seed: int, max_words: int = 6
    ) -> str:
        """Run a raw continuation from the speaker's unfinished stem."""
        raw_prompt = (
            "Transcript of a spoken conversation:\n"
            f"Speaker: {stem}"
        )
        payload = {
            "model": self.ollama_model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.92,
                "repeat_penalty": 1.25,
                "seed": seed,
                "num_predict": max(4, int(max_words * 1.6)),
                "stop": ["\n", ".", "?", "!", "\"", "Speaker:"],
            },
        }
        resp = await self._client.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        return self._clean_phrase(text, stem, max_words=max_words)

    async def _generate_list_completions(self, stem: str, count: int, max_words: int = 6) -> list[str]:
        """
        Ask `llama3.2:3b` to generate `count` distinct continuations of ~`max_words` words each.
        """
        min_w = max(1, max_words - 2)
        prompt = (
            f"Give {count} different ways ({min_w} to {max_words} words each) to finish the end of this incomplete spoken sentence.\n"
            "Only write the NEW words that come immediately after the last word. Do not repeat the sentence.\n\n"
            f"Incomplete sentence: \"{stem} ...\"\n"
            "Continuations (one per line):\n1."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.92,
                "repeat_penalty": 1.2,
                "num_predict": max(30, count * (max_words + 4)),
                "stop": ["\n\n", "Incomplete sentence:"],
            },
        }
        resp = await self._client.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        raw_text = "1. " + resp.json().get("response", "")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        results: list[str] = []
        for line in lines:
            cleaned = self._clean_phrase(line, stem, max_words=max_words)
            if cleaned and cleaned.lower() not in {r.lower() for r in results}:
                results.append(cleaned)
        return results

    async def generate(
        self, transcript_context: str, count: int = 6, detail_words: int = 6
    ) -> SuggestionBundle:
        """
        Generate `count` (1..10) distinct phrase completion suggestions of up to
        `detail_words` (1..12) words each for the given transcript context.
        """
        start = time.perf_counter()
        count = max(1, min(10, int(count)))
        detail_words = max(1, min(12, int(detail_words)))
        await self.check_ollama()
        cleaned = self._clean_context(transcript_context) or "Hello, I would like to find the"

        try:
            phrases_text: list[str] = []

            if count <= 3:
                temps = [0.15, 0.75, 1.0][:count]
                seeds = [101, 202, 303][:count]
                raw_results = await asyncio.gather(
                    *[
                        self._raw_continue_single(cleaned, t, s, max_words=detail_words)
                        for t, s in zip(temps, seeds)
                    ]
                )
                for r in raw_results:
                    if r and r.lower() not in {p.lower() for p in phrases_text}:
                        phrases_text.append(r)

            if len(phrases_text) < count:
                top_raw, list_items = await asyncio.gather(
                    self._raw_continue_single(cleaned, temperature=0.15, seed=101, max_words=detail_words)
                    if not phrases_text
                    else asyncio.sleep(0, result=""),
                    self._generate_list_completions(cleaned, count + 2, max_words=detail_words),
                )
                if top_raw and top_raw.lower() not in {p.lower() for p in phrases_text}:
                    phrases_text.insert(0, top_raw)
                for item in list_items:
                    if item and item.lower() not in {p.lower() for p in phrases_text}:
                        phrases_text.append(item)
                    if len(phrases_text) >= count:
                        break

            idx = 1
            while len(phrases_text) < count:
                extra = await self._raw_continue_single(
                    cleaned, temperature=0.6 + 0.1 * idx, seed=400 + idx, max_words=detail_words
                )
                if extra and extra.lower() not in {p.lower() for p in phrases_text}:
                    phrases_text.append(extra)
                else:
                    phrases_text.append(f"option {len(phrases_text) + 1}")
                idx += 1
                if idx > 4:
                    break

            phrases_text = phrases_text[:count]
            phrase_items: list[SuggestionItem] = []
            for i, text in enumerate(phrases_text):
                score = round(max(0.55, 0.92 - i * 0.05), 2)
                confidence = "high" if i == 0 else "medium"
                phrase_items.append(SuggestionItem(text=text, confidence=confidence, score=score))

            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=cleaned,
                phrases=phrase_items,
                provider=f"ollama ({self.ollama_model})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
        except Exception as exc:
            logger.error("Ollama request error: %s", exc)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=cleaned,
                phrases=[
                    SuggestionItem(text="...", confidence="high" if i == 0 else "medium", score=0.5)
                    for i in range(count)
                ],
                provider=f"ollama (error: {type(exc).__name__})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
