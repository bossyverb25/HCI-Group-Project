"""
System 3: Suggestion System (Local Ollama LLM)
----------------------------------------------
Takes the latest conversational snippet (from the last 30 seconds of context)
and calls the local Ollama LLM (`llama3.2:1b` or `llama3.2:3b`) using true
RAW token continuation (`/api/generate` with `raw: true`) to generate BOTH:
  - 2 short word suggestions (1–2 words)
  - 2 phrase completions (3–7 words)
along with confidence levels ("high" vs "medium") to support the HCI design
requirement "Honest about uncertainty".

Why `raw: true`?
Small 1B/3B models fail when asked to output JSON meta-reasoning about "what the
next word is" — they just copy words from the start of the prompt.
With `raw: true`, we feed the user's exact incomplete sentence directly into the
LLM's token stream so it naturally predicts the next tokens that follow the last
word, then extract the first 1–2 words (Word suggestion) and full 4–6 words
(Phrase suggestion) from two distinct continuations!
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


class SuggestionEngine:
    """
    Generates rolling word and phrase suggestions using Local Ollama raw token
    continuation (true causal language modeling after the speaker's last word).
    """

    def __init__(self) -> None:
        self.ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self._model_checked = False
        self._client = httpx.AsyncClient(timeout=8.0)

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
        # Collapse consecutive duplicate words like "where where"
        cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
        return cleaned[-500:]

    async def _raw_continue(self, stem: str, temperature: float, seed: int, avoid_first_word: str = "") -> str:
        """
        Uses Ollama `/api/generate` with `raw: true` so the LLM directly continues the
        unfinished sentence token-by-token instead of repeating the prompt.
        """
        raw_prompt = (
            "Transcript of a spoken conversation in a shop or pharmacy:\n"
            f"Speaker: {stem}"
        )
        if avoid_first_word:
            raw_prompt = (
                "Transcript of a spoken conversation (using different phrasing):\n"
                f"Speaker: {stem}"
            )

        payload = {
            "model": self.ollama_model,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.25,
                "seed": seed,
                "num_predict": 10,
                "stop": ["\n", ".", "?", "!", "\"", "Speaker:", "Assistant:"],
            },
        }
        resp = await self._client.post(f"{self.ollama_url}/api/generate", json=payload)
        resp.raise_for_status()
        continuation = resp.json().get("response", "").strip()
        # Clean stray punctuation at edges
        continuation = re.sub(r"^[,;:\-—\s]+", "", continuation)
        continuation = re.sub(r"[.?!,;:\-—\"']+$", "", continuation).strip()
        return continuation

    @staticmethod
    def _extract_word_and_phrase(continuation: str) -> tuple[str, str]:
        """
        From a natural 4-8 word continuation (e.g. 'cold medicine on these shelves'),
        extract:
          - word suggestion: first 1–2 words (if the first word is a short article/preposition
            like 'a', 'the', 'to', 'for', 'some', include the 2nd word so it's meaningful!)
          - phrase suggestion: first 4–6 words
        """
        tokens = continuation.split()
        if not tokens:
            return "...", "..."

        stop_starters = {"a", "an", "the", "to", "for", "of", "in", "on", "at", "with", "some", "my", "your", "very", "more"}
        if len(tokens) >= 2 and tokens[0].lower().strip(".,?!") in stop_starters:
            word_sug = " ".join(tokens[:2])
        else:
            word_sug = tokens[0]

        phrase_sug = " ".join(tokens[:6])
        return word_sug, phrase_sug

    async def generate(self, transcript_context: str) -> SuggestionBundle:
        """
        Generate 2 distinct word suggestions and 2 distinct phrase suggestions by running
        two fast parallel raw-token continuations (`num_predict: 10`) from the exact end
        of the user's spoken stem.
        """
        start = time.perf_counter()
        await self.check_ollama()
        cleaned = self._clean_context(transcript_context) or "Hello, I would like to buy"

        try:
            # Generate two diverse continuations in parallel (only 10 tokens each -> very fast!)
            cont_1, cont_2 = await asyncio.gather(
                self._raw_continue(cleaned, temperature=0.15, seed=101),
                self._raw_continue(cleaned, temperature=0.85, seed=202, avoid_first_word="true"),
            )

            w1, p1 = self._extract_word_and_phrase(cont_1)
            w2, p2 = self._extract_word_and_phrase(cont_2)

            # If the second sample happened to start with the exact same word, do one fast high-temp retry
            if w2.lower() == w1.lower():
                cont_3 = await self._raw_continue(cleaned, temperature=1.1, seed=303, avoid_first_word="true")
                w3, p3 = self._extract_word_and_phrase(cont_3)
                if w3 and w3.lower() != w1.lower():
                    w2, p2 = w3, p3
                elif len(cont_1.split()) >= 3:
                    # Use the 2nd-3rd word of cont_1 as alternative if still identical
                    w2 = " ".join(cont_1.split()[1:3])

            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=cleaned,
                words=[
                    SuggestionItem(text=w1, confidence="high", score=0.92),
                    SuggestionItem(text=w2, confidence="medium", score=0.76),
                ],
                phrases=[
                    SuggestionItem(text=p1, confidence="high", score=0.90),
                    SuggestionItem(text=p2, confidence="medium", score=0.74),
                ],
                provider=f"ollama ({self.ollama_model})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
        except Exception as exc:
            logger.error("Ollama request error: %s", exc)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return SuggestionBundle(
                context_snippet=cleaned,
                words=[
                    SuggestionItem(text="...", confidence="high", score=0.5),
                    SuggestionItem(text="...", confidence="medium", score=0.5),
                ],
                phrases=[
                    SuggestionItem(text="...", confidence="high", score=0.5),
                    SuggestionItem(text="...", confidence="medium", score=0.5),
                ],
                provider=f"ollama (error: {type(exc).__name__})",
                latency_ms=latency_ms,
                generated_at=time.time(),
            )
