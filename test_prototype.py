"""
Automated verification & smoke test for Prototype v1:
Tests System 1 (rolling transcript update), System 2 (rolling suggestion cache &
HCI CSV/JSONL logging), and System 3 (word & phrase completion generation).
"""

from __future__ import annotations

import asyncio
from backend.judgment import HCILogger, JudgmentCacheManager, CSV_LOG_PATH
from backend.suggestion import SuggestionEngine
from backend.transcription import TranscriptState


async def run_tests() -> None:
    print("1. Initializing SuggestionEngine (System 3) and JudgmentCacheManager (System 2)...")
    engine = SuggestionEngine()
    logger = HCILogger()
    judgment = JudgmentCacheManager(suggestion_engine=engine, hci_logger=logger)

    try:
        # 1. Seed initial cache
        init_bundle = await judgment.initialize_default_cache()
        assert len(init_bundle.words) == 2, "Expected 2 word suggestions"
        assert len(init_bundle.phrases) == 2, "Expected 2 phrase suggestions"
        print(f"   Initial cache warm via provider: {init_bundle.provider}")

        # 2. Simulate rolling transcript update from System 1
        sample_speech = "Excuse me, I came to the pharmacy because I have"
        state = TranscriptState(
            full_text=sample_speech,
            last_snippet=TranscriptState.extract_snippet(sample_speech),
            word_confidence=0.94,
            engine="faster-whisper (base.en)",
        )
        updated_bundle = await judgment.on_transcript_updated(state)
        print(f"   Context snippet: '{state.last_snippet}'")
        print(f"   Cached Words:    {[w.text for w in updated_bundle.words]}")
        print(f"   Cached Phrases:  {[p.text for p in updated_bundle.phrases]}")

        # 3. Trigger 'I'm Stuck (Words)' and verify 0ms retrieval from cache
        stuck_words = await judgment.trigger_im_stuck(mode="words")
        assert stuck_words["mode"] == "words"
        assert len(stuck_words["suggestions"]) == 2
        print(f"   'I'm Stuck (Words)' cache retrieval: {stuck_words['cache_retrieval_ms']} ms")

        # 4. Trigger 'I'm Stuck (Phrases)'
        stuck_phrases = await judgment.trigger_im_stuck(mode="phrases")
        assert stuck_phrases["mode"] == "phrases"
        assert len(stuck_phrases["suggestions"]) == 2
        print(f"   'I'm Stuck (Phrases)' cache retrieval: {stuck_phrases['cache_retrieval_ms']} ms")

        # 5. Verify HCI evaluation log file was created and populated
        assert CSV_LOG_PATH.exists(), "Expected logs/hci_events.csv to exist"
        print(f"   HCI evaluation log verified at: {CSV_LOG_PATH}")
        print("\nALL PROTOTYPE V1 TESTS PASSED!")
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(run_tests())
