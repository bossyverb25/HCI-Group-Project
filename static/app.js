/**
 * The Invisible Interpreter — Prototype v1 Frontend Client
 * --------------------------------------------------------
 * - Streams 16kHz mono PCM audio over WebSocket to Faster-Whisper (`backend/transcription.py`)
 *   AND runs browser Web Speech API in parallel for ultra-low-latency interim text & fallback.
 * - Maintains the latest pre-generated `words` and `phrases` suggestion bundle pushed from
 *   the backend's 30-second rolling cache (`backend/judgment.py`).
 * - Immediately reveals the 2 big suggestions (0ms perceived latency) when the user clicks
 *   "I'm Stuck — Words" or "I'm Stuck — Phrases", while logging all events to the backend.
 */

(() => {
  // UI Elements
  const listeningPill = document.getElementById("listening-pill");
  const listeningLabel = document.getElementById("listening-label");
  const cacheStateEl = document.getElementById("cache-state");
  const sttEngineBadge = document.getElementById("stt-engine-badge");
  const llmEngineBadge = document.getElementById("llm-engine-badge");
  const latencyBadge = document.getElementById("latency-badge");
  const contextSnippetEl = document.getElementById("context-snippet");

  const calmIdleBox = document.getElementById("calm-idle-box");
  const suggestionsGrid = document.getElementById("suggestions-grid");
  const dismissRow = document.getElementById("dismiss-row");
  const suggestionModeTitle = document.getElementById("suggestion-mode-title");

  const card0 = document.getElementById("suggestion-card-0");
  const card1 = document.getElementById("suggestion-card-1");
  const text0 = document.getElementById("suggestion-text-0");
  const text1 = document.getElementById("suggestion-text-1");
  const conf0 = document.getElementById("conf-badge-0");
  const conf1 = document.getElementById("conf-badge-1");

  const btnToggleListen = document.getElementById("btn-toggle-listen");
  const toggleText = document.getElementById("toggle-text");
  const btnStuckWords = document.getElementById("btn-stuck-words");
  const btnStuckPhrases = document.getElementById("btn-stuck-phrases");
  const btnDismiss = document.getElementById("btn-dismiss");

  // State
  let ws = null;
  let isListening = false;
  let audioCtx = null;
  let mediaStream = null;
  let workletNode = null;
  let speechRecognition = null;
  let activeMode = null; // "words" | "phrases" | null
  let latestTranscript = "";

  // Rolling pre-generated cache pushed continuously from the server
  let warmCache = {
    words: [
      { text: "a headache", confidence: "high", score: 0.93 },
      { text: "allergies", confidence: "medium", score: 0.78 },
    ],
    phrases: [
      { text: "a bad headache since yesterday", confidence: "high", score: 0.91 },
      { text: "something for a sore throat", confidence: "medium", score: 0.76 },
    ],
    provider: "local-cache",
    latency_ms: 0.0,
  };

  // Connect WebSocket
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      cacheStateEl.textContent = "Warm (Ready)";
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (err) {
        console.error("Failed to parse server message:", err);
      }
    };

    ws.onclose = () => {
      cacheStateEl.textContent = "Reconnecting…";
      setTimeout(connectWebSocket, 1500);
    };
  }

  function handleServerMessage(msg) {
    if (msg.type === "session_init") {
      if (msg.cache) {
        warmCache = msg.cache;
        llmEngineBadge.textContent = `Engine: ${msg.cache.provider}`;
      }
      sttEngineBadge.textContent = msg.whisper_available
        ? "STT: Faster-Whisper (Active)"
        : "STT: Web Speech / Hybrid";
    } else if (msg.type === "transcript_update") {
      if (msg.last_snippet) {
        latestTranscript = msg.full_text || msg.last_snippet;
        contextSnippetEl.textContent = msg.last_snippet;
        contextSnippetEl.classList.remove("placeholder");
      }
      if (msg.engine) {
        sttEngineBadge.textContent = `STT: ${msg.engine}`;
      }
    } else if (msg.type === "cache_update") {
      if (msg.cache) {
        warmCache = msg.cache;
        llmEngineBadge.textContent = `Engine: ${msg.cache.provider}`;
        cacheStateEl.textContent = `Warm (${msg.rolling_cache_size || 1} cached)`;
        // If suggestions are already open on screen, keep them fresh
        if (activeMode) {
          renderSuggestionsFromCache(activeMode, 0.0);
        }
      }
    } else if (msg.type === "im_stuck_result") {
      if (msg.all_cached) {
        warmCache = msg.all_cached;
      }
      renderSuggestionsList(
        msg.mode,
        msg.suggestions,
        msg.cache_retrieval_ms ?? 0.0,
        warmCache.provider
      );
    }
  }

  // Instant 0ms Reveal from Rolling Cache + Log to Backend
  function triggerImStuck(mode) {
    activeMode = mode;
    const t0 = performance.now();
    renderSuggestionsFromCache(mode, performance.now() - t0);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "im_stuck",
          mode: mode,
          context: latestTranscript,
        })
      );
    }
  }

  function renderSuggestionsFromCache(mode, retrievalMs) {
    const items = mode === "words" ? warmCache.words : warmCache.phrases;
    renderSuggestionsList(mode, items, retrievalMs, warmCache.provider);
  }

  function renderSuggestionsList(mode, items, retrievalMs, provider) {
    if (!items || items.length < 2) return;

    calmIdleBox.classList.add("hidden");
    suggestionsGrid.classList.remove("hidden");
    dismissRow.classList.remove("hidden");

    suggestionModeTitle.textContent =
      mode === "words"
        ? "NEXT WORD SUGGESTIONS (1–2 WORDS)"
        : "PHRASE COMPLETION SUGGESTIONS";

    if (provider) {
      llmEngineBadge.textContent = `Engine: ${provider}`;
    }
    latencyBadge.textContent = `${retrievalMs.toFixed(1)} ms (from rolling cache)`;

    // Card 0
    text0.textContent = items[0].text;
    const pct0 = Math.round((items[0].score || 0.9) * 100);
    conf0.textContent =
      items[0].confidence === "high"
        ? `High Confidence (${pct0}%)`
        : `Alternative (${pct0}%)`;

    // Card 1
    text1.textContent = items[1].text;
    const pct1 = Math.round((items[1].score || 0.75) * 100);
    conf1.textContent =
      items[1].confidence === "high"
        ? `High Confidence (${pct1}%)`
        : `Alternative (${pct1}%)`;
  }

  function hideSuggestions() {
    activeMode = null;
    suggestionsGrid.classList.add("hidden");
    dismissRow.classList.add("hidden");
    calmIdleBox.classList.remove("hidden");
    suggestionModeTitle.textContent =
      'SUGGESTIONS (PRESS "I\'M STUCK" BELOW TO REVEAL)';
  }

  function selectSuggestion(index) {
    if (!activeMode) return;
    const chosenText = index === 0 ? text0.textContent : text1.textContent;
    if (!chosenText || chosenText === "—") return;

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "suggestion_selected",
          mode: activeMode,
          text: chosenText,
        })
      );
    }
    hideSuggestions();
  }

  // Audio Capture: 16kHz PCM AudioWorklet -> Faster-Whisper + Browser Web Speech API
  async function startListening() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await audioCtx.audioWorklet.addModule("/static/audio-processor.js");

      const source = audioCtx.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioCtx, "pcm16-downsampler");

      workletNode.port.onmessage = (e) => {
        if (isListening && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      source.connect(workletNode);
    } catch (err) {
      console.warn("Microphone AudioWorklet warning:", err);
    }

    // Start Web Speech API in parallel if supported by browser for instant interim word updates
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      try {
        speechRecognition = new SpeechRec();
        speechRecognition.lang = "en-US";
        speechRecognition.continuous = true;
        speechRecognition.interimResults = true;

        speechRecognition.onresult = (event) => {
          let transcript = "";
          for (let i = Math.max(0, event.results.length - 3); i < event.results.length; i++) {
            transcript += event.results[i][0].transcript + " ";
          }
          const cleaned = transcript.trim();
          if (cleaned && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "browser_transcript",
                text: cleaned,
                engine: "faster-whisper + web-speech",
              })
            );
          }
        };

        speechRecognition.onend = () => {
          if (isListening && speechRecognition) {
            try {
              speechRecognition.start();
            } catch (_) {}
          }
        };

        speechRecognition.start();
      } catch (err) {
        console.debug("Web Speech API skipped:", err);
      }
    }

    isListening = true;
    updateListeningUI();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "listening_toggled", listening: true }));
    }
  }

  function stopListening() {
    isListening = false;
    if (speechRecognition) {
      try {
        speechRecognition.onend = null;
        speechRecognition.stop();
      } catch (_) {}
      speechRecognition = null;
    }
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    updateListeningUI();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "listening_toggled", listening: false }));
    }
  }

  function updateListeningUI() {
    if (isListening) {
      listeningPill.classList.remove("off");
      listeningPill.classList.add("on");
      listeningLabel.textContent = "Listening & Caching";
      btnToggleListen.classList.remove("off");
      btnToggleListen.classList.add("on");
      btnToggleListen.setAttribute("aria-pressed", "true");
      toggleText.textContent = "Listening: ON";
    } else {
      listeningPill.classList.remove("on");
      listeningPill.classList.add("off");
      listeningLabel.textContent = "Mic Off (Paused)";
      btnToggleListen.classList.remove("on");
      btnToggleListen.classList.add("off");
      btnToggleListen.setAttribute("aria-pressed", "false");
      toggleText.textContent = "Listening: OFF";
    }
  }

  // Event Listeners
  btnToggleListen.addEventListener("click", () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  });

  btnStuckWords.addEventListener("click", () => triggerImStuck("words"));
  btnStuckPhrases.addEventListener("click", () => triggerImStuck("phrases"));
  btnDismiss.addEventListener("click", hideSuggestions);

  card0.addEventListener("click", () => selectSuggestion(0));
  card1.addEventListener("click", () => selectSuggestion(1));

  // Role-play quick prompt buttons
  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const promptText = btn.getAttribute("data-prompt") || "";
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "browser_transcript",
            text: promptText,
            engine: "roleplay-simulator",
          })
        );
      }
    });
  });

  // Keyboard Shortcuts for rapid testing / Wizard-of-Oz / Hands-free triggers
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "1" || e.key === "ArrowLeft") {
      e.preventDefault();
      triggerImStuck("words");
    } else if (e.key === "2" || e.key === "ArrowRight") {
      e.preventDefault();
      triggerImStuck("phrases");
    } else if (e.key.toLowerCase() === "l") {
      e.preventDefault();
      btnToggleListen.click();
    } else if (e.key === "Escape") {
      hideSuggestions();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectSuggestion(0);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      selectSuggestion(1);
    }
  });

  connectWebSocket();
})();
