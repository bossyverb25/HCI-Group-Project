"""
System 3: Suggestion System
---------------------------
Takes the latest conversational snippet (from the last 30 seconds of context)
and generates BOTH:
  - 2 short word suggestions (1–2 words)
  - 2 phrase completions (3–7 words)
along with confidence levels ("high" vs "medium") to support the HCI design
requirement "Honest about uncertainty".

Backends supported (auto-detected, $0 budget):
1. Local Ollama API (`http://localhost:11434`, default model `llama3.2:1b` or `qwen2.5:1.5b`)
2. Groq Free Tier Cloud API (`llama-3.1-8b-instant` if `GROQ_API_KEY` env var is set)
3. Fast Contextual Fallback Engine (instantaneous <2ms local conversational predictor
   tailored to everyday & pharmacy/shop role-play scenarios so Prototype v1 always
   works out-of-the-box even without Ollama or internet).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

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
    words: list[SuggestionItem]
    phrases: list[SuggestionItem]
    provider: str
    latency_ms: float
    generated_at: float

    def to_dict(self) -> dict:
        return {
            "context_snippet": self.context_snippet,
            "words": [asdict(w) for w in self.words],
            "phrases": [asdict(p) for p in self.phrases],
            "provider": self.provider,
            "latency_ms": round(self.latency_ms, 1),
            "generated_at": self.generated_at,
        }


SYSTEM_PROMPT = """You are an ambient English conversation completion assistant ("The Invisible Interpreter").
Given the user's incomplete spoken sentence or the last few words of a conversation, predict how the speaker can naturally continue or finish their thought.
Return ONLY valid JSON matching this exact structure:
{
  "words": [
    {"text": "<1-2 words>", "confidence": "high", "score": 0.92},
    {"text": "<1-2 words alternative>", "confidence": "medium", "score": 0.74}
  ],
  "phrases": [
    {"text": "<3-6 words natural completion>", "confidence": "high", "score": 0.89},
    {"text": "<3-6 words alternative completion>", "confidence": "medium", "score": 0.71}
  ]
}
Keep suggestions natural, concise, and directly continuable from the last word spoken."""


class SuggestionEngine:
    """Generates rolling word and phrase suggestions using Local Ollama, Groq, or contextual rules."""

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        self._client = httpx.AsyncClient(timeout=2.5)
        self._ollama_available: Optional[bool] = None
        self._last_ollama_check: float = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def _check_ollama(self) -> bool:
        """Check whether local Ollama server is running and has a model available."""
        now = time.time()
        if self._ollama_available is not None and (now - self._last_ollama_check) < 15.0:
            return self._ollama_available

        self._last_ollama_check = now
        try:
            resp = await self._client.get(f"{self.ollama_url}/api/tags", timeout=0.8)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if models:
                    # Prefer configured model if present, otherwise pick first installed model
                    if not any(self.ollama_model in m for m in models):
                        self.ollama_model = models[0]
                    self._ollama_available = True
                    return True
        except Exception:
            pass

        self._ollama_available = False
        return False

    async def generate(self, transcript_context: str) -> SuggestionBundle:
        """
        Generate 2 word suggestions and 2 phrase suggestions for the given transcript context.
        Tries Local Ollama first, then Groq Cloud API (if key configured), then instant contextual predictor.
        """
        start = time.perf_counter()
        cleaned = transcript_context.strip()
        if not cleaned:
            cleaned = "Hello, I would like"

        # 1. Try Local Ollama
        if await self._check_ollama():
            bundle = await self._generate_ollama(cleaned, start)
            if bundle is not None:
                return bundle

        # 2. Try Groq Cloud API (if GROQ_API_KEY is set)
        if self.groq_api_key:
            bundle = await self._generate_groq(cleaned, start)
            if bundle is not None:
                return bundle

        # 3. Instant Contextual Predictor (0ms fallback, always available)
        return self._generate_contextual_fallback(cleaned, start)

    async def _generate_ollama(self, context: str, start_time: float) -> Optional[SuggestionBundle]:
        try:
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Spoken context so far: \"{context[-240:]}\""},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 90,
                },
            }
            resp = await self._client.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "{}")
            return self._parse_llm_json(content, context, f"ollama ({self.ollama_model})", start_time)
        except Exception as exc:
            logger.debug("Ollama generation skipped/failed: %s", exc)
            self._ollama_available = False
            return None

    async def _generate_groq(self, context: str, start_time: float) -> Optional[SuggestionBundle]:
        try:
            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Spoken context so far: \"{context[-240:]}\""},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 90,
            }
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            resp = await self._client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse_llm_json(content, context, f"groq ({self.groq_model})", start_time)
        except Exception as exc:
            logger.warning("Groq generation failed: %s", exc)
            return None

    def _parse_llm_json(
        self, raw_json: str, context: str, provider: str, start_time: float
    ) -> Optional[SuggestionBundle]:
        try:
            data = json.loads(raw_json)
            raw_words = data.get("words", [])
            raw_phrases = data.get("phrases", [])
            if not raw_words or not raw_phrases:
                return None

            words = [
                SuggestionItem(
                    text=str(item.get("text", "")).strip(),
                    confidence=str(item.get("confidence", "high" if i == 0 else "medium")),
                    score=float(item.get("score", 0.9 if i == 0 else 0.72)),
                )
                for i, item in enumerate(raw_words[:2])
                if str(item.get("text", "")).strip()
            ]
            phrases = [
                SuggestionItem(
                    text=str(item.get("text", "")).strip(),
                    confidence=str(item.get("confidence", "high" if i == 0 else "medium")),
                    score=float(item.get("score", 0.88 if i == 0 else 0.70)),
                )
                for i, item in enumerate(raw_phrases[:2])
                if str(item.get("text", "")).strip()
            ]
            if len(words) < 2 or len(phrases) < 2:
                return None

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return SuggestionBundle(
                context_snippet=context,
                words=words,
                phrases=phrases,
                provider=provider,
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
        except Exception:
            return None

    def _generate_contextual_fallback(self, context: str, start_time: float) -> SuggestionBundle:
        """
        Zero-latency (<2ms) contextual predictor covering common conversational frames,
        pharmacy/bakery/medical scenarios from Project 12, and grammar continuations.
        """
        lower = context.lower().strip()
        tokens = re.findall(r"[a-z']+", lower)
        last_word = tokens[-1] if tokens else ""
        last_two = " ".join(tokens[-2:]) if len(tokens) >= 2 else last_word

        # Default general continuations
        words = [("please", 0.88), ("today", 0.74)]
        phrases = [("could you help me with this", 0.86), ("can you repeat that more slowly", 0.72)]

        # 1. Exact ending word/bigram grammar triggers
        if last_two in ("i have", "i've got", "having a", "got a"):
            words = [("a headache", 0.93), ("allergies", 0.78)]
            phrases = [("a bad headache since yesterday", 0.91), ("a sore throat and fever", 0.76)]
        elif last_two in ("i need", "looking for", "like some", "want some", "get some"):
            words = [("painkillers", 0.92), ("something", 0.79)]
            phrases = [("something for a sore throat", 0.90), ("medicine for a cold please", 0.75)]
        elif last_two in ("how often", "how many", "when should"):
            words = [("a day", 0.94), ("after meals", 0.81)]
            phrases = [("should I take these tablets", 0.92), ("times a day after meals", 0.79)]
        elif last_two in ("twice a", "once a", "three times"):
            words = [("day", 0.96), ("daily", 0.82)]
            phrases = [("day after meals with water", 0.94), ("day in the morning and evening", 0.80)]
        elif last_word in ("take", "taking"):
            words = [("two tablets", 0.91), ("this medicine", 0.77)]
            phrases = [("one tablet twice a day", 0.90), ("this after eating a meal", 0.75)]
        elif last_word in ("after", "before", "with"):
            words = [("meals", 0.94), ("water", 0.82)]
            phrases = [("meals with a full glass of water", 0.92), ("breakfast and dinner every day", 0.77)]
        elif last_word in ("allergic", "allergy"):
            words = [("to penicillin", 0.93), ("to ibuprofen", 0.76)]
            phrases = [("to penicillin and some antibiotics", 0.91), ("to dairy or gluten products", 0.73)]
        elif last_word in ("side", "any"):
            words = [("effects", 0.94), ("drowsiness", 0.78)]
            phrases = [("side effects like drowsiness", 0.92), ("other symptoms I should watch for", 0.76)]
        elif last_word in ("prescription", "doctor", "pharmacy", "medicine", "tablets", "pills"):
            words = [("dosage", 0.89), ("instructions", 0.75)]
            phrases = [("do I need a prescription for this", 0.90), ("should I take it after meals", 0.77)]
        elif last_word in ("how", "much", "cost", "price", "pay"):
            words = [("much is it", 0.92), ("by card", 0.79)]
            phrases = [("much does this cost in total", 0.90), ("can I pay by contactless card", 0.76)]
        elif last_word in ("sorry", "pardon", "understand", "mean", "what"):
            words = [("that word", 0.90), ("more slowly", 0.78)]
            phrases = [("could you say that more slowly", 0.92), ("what does that instruction mean", 0.78)]
        elif last_word in ("i", "i'm", "im"):
            words = [("looking for", 0.91), ("not sure", 0.77)]
            phrases = [("am looking for cold medicine", 0.89), ("didn't quite catch that last word", 0.75)]
        elif last_word in ("can", "could", "would"):
            words = [("you help", 0.92), ("I get", 0.80)]
            phrases = [("you explain how to take this", 0.90), ("I have something for a cough", 0.76)]
        elif last_word in ("the", "a", "an", "my"):
            words = [("prescription", 0.87), ("symptoms", 0.74)]
            phrases = [("right dosage for an adult", 0.86), ("instructions on the label", 0.72)]
        elif last_word in ("is", "are", "was"):
            words = [("available", 0.86), ("safe", 0.74)]
            phrases = [("safe to take on an empty stomach", 0.88), ("there a generic version available", 0.73)]
        elif last_word in ("to", "for"):
            words = [("relieve pain", 0.88), ("sleep better", 0.75)]
            phrases = [("help with a headache and fever", 0.89), ("take twice a day after meals", 0.76)]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return SuggestionBundle(
            context_snippet=context,
            words=[
                SuggestionItem(text=words[0][0], confidence="high", score=words[0][1]),
                SuggestionItem(text=words[1][0], confidence="medium", score=words[1][1]),
            ],
            phrases=[
                SuggestionItem(text=phrases[0][0], confidence="high", score=phrases[0][1]),
                SuggestionItem(text=phrases[1][0], confidence="medium", score=phrases[1][1]),
            ],
            provider="local-contextual-engine",
            latency_ms=latency_ms,
            generated_at=time.time(),
        )
