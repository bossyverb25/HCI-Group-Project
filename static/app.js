/**
 * The Invisible Interpreter — Prototype v1 Frontend Client
 * --------------------------------------------------------
 * - Streams 16kHz mono PCM audio over WebSocket exclusively to Faster-Whisper (`backend/transcription.py`).
 * - Maintains the latest pre-generated phrase completion bundle pushed from
 *   the backend's 30-second rolling Ollama cache (`backend/judgment.py`).
 * - Settings menu (hidden by default, toggled via ⚙️ button) has two sliders:
 *     1. Suggestions (1–10, default 6)
 *     2. Detail (1–12 words per suggestion, default 6)
 * - "I'm Stuck" button (and Spacebar) toggles between showing suggestions ("Hide Suggestions")
 *   and hiding them ("I'm Stuck").
 */

(() => {
  // UI Elements
  const listeningPill = document.getElementById("listening-pill");
  const listeningLabel = document.getElementById("listening-label");
  const connectionBadge = document.getElementById("connection-badge");
  const connectionStateEl = document.getElementById("connection-state");
  const sttEngineBadge = document.getElementById("stt-engine-badge");
  const llmEngineBadge = document.getElementById("llm-engine-badge");
  const contextSnippetEl = document.getElementById("context-snippet");

  const calmIdleBox = document.getElementById("calm-idle-box");
  const suggestionsGrid = document.getElementById("suggestions-grid");
  const suggestionModeTitle = document.getElementById("suggestion-mode-title");

  const settingsMenu = document.getElementById("settings-menu");
  const btnSettings = document.getElementById("btn-settings");

  const countSliderEl = document.getElementById("suggestion-count-slider");
  const countValEl = document.getElementById("suggestion-count-value");
  const detailSliderEl = document.getElementById("detail-slider");
  const detailValEl = document.getElementById("detail-value");

  const btnToggleListen = document.getElementById("btn-toggle-listen");
  const toggleText = document.getElementById("toggle-text");
  const btnStuck = document.getElementById("btn-stuck");
  const stuckText = document.getElementById("stuck-text");

  // State
  let ws = null;
  let isListening = false;
  let audioCtx = null;
  let mediaStream = null;
  let workletNode = null;
  let isShowingSuggestions = false;
  let suggestionCount = parseInt(countSliderEl.value, 10) || 6;
  let detailWords = parseInt(detailSliderEl.value, 10) || 6;
  let latestTranscript = "";

  // Rolling pre-generated cache pushed continuously from the server
  let warmCache = {
    phrases: Array.from({ length: 6 }, (_, i) => ({
      text: "...",
      confidence: i === 0 ? "high" : "medium",
      score: Math.max(0.55, 0.92 - i * 0.05),
    })),
    provider: "ollama (llama3.2:3b)",
    latency_ms: 0.0,
  };

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      connectionStateEl.textContent = "Connected";
      connectionBadge.classList.add("connected");
      ws.send(
        JSON.stringify({
          type: "update_settings",
          count: suggestionCount,
          detail_words: detailWords,
        })
      );
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
      connectionStateEl.textContent = "Disconnected";
      connectionBadge.classList.remove("connected");
      setTimeout(connectWebSocket, 1500);
    };
  }

  function handleServerMessage(msg) {
    if (msg.type === "session_init") {
      if (msg.cache) {
        warmCache = msg.cache;
        llmEngineBadge.textContent = `LLM: ${msg.cache.provider}`;
      }
      sttEngineBadge.textContent = `STT: faster-whisper (${msg.whisper_model || "base.en"})`;
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
        llmEngineBadge.textContent = `LLM: ${msg.cache.provider}`;
        if (isShowingSuggestions) {
          renderSuggestionsFromCache(0.0);
        }
      }
    } else if (msg.type === "im_stuck_result") {
      if (msg.all_cached) {
        warmCache = msg.all_cached;
      }
      if (isShowingSuggestions) {
        renderSuggestionsList(
          msg.suggestions,
          msg.cache_retrieval_ms ?? 0.0,
          warmCache.provider
        );
      }
    }
  }

  // Toggle between showing suggestions and hiding them
  function toggleSuggestions() {
    if (isShowingSuggestions) {
      hideSuggestions();
    } else {
      showSuggestions();
    }
  }

  function showSuggestions() {
    isShowingSuggestions = true;
    stuckText.textContent = "Hide Suggestions";
    btnStuck.classList.add("active");

    const t0 = performance.now();
    renderSuggestionsFromCache(performance.now() - t0);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "im_stuck",
          count: suggestionCount,
          detail_words: detailWords,
          context: latestTranscript,
        })
      );
    }
  }

  function hideSuggestions() {
    isShowingSuggestions = false;
    stuckText.textContent = "I'm Stuck";
    btnStuck.classList.remove("active");
    suggestionsGrid.classList.add("hidden");
    calmIdleBox.classList.remove("hidden");
    suggestionModeTitle.textContent = "SUGGESTIONS";
  }

  function renderSuggestionsFromCache(retrievalMs) {
    const items = (warmCache.phrases || []).slice(0, suggestionCount);
    renderSuggestionsList(items, retrievalMs, warmCache.provider);
  }

  function renderSuggestionsList(items, retrievalMs, provider) {
    if (!items || items.length === 0) return;

    calmIdleBox.classList.add("hidden");
    suggestionsGrid.classList.remove("hidden");

    suggestionModeTitle.textContent = "PHRASE COMPLETION SUGGESTIONS";

    if (provider) {
      llmEngineBadge.textContent = `LLM: ${provider}`;
    }

    suggestionsGrid.innerHTML = "";
    items.slice(0, suggestionCount).forEach((item, idx) => {
      const isHigh = item.confidence === "high" || idx === 0;
      const pct = Math.round((item.score || Math.max(0.55, 0.9 - idx * 0.05)) * 100);

      const card = document.createElement("div");
      card.className = `suggestion-card ${isHigh ? "high-confidence" : "medium-confidence"}`;
      card.innerHTML = `
        <div class="card-top">
          <span class="option-number">Option ${idx + 1}</span>
          <span class="confidence-badge ${isHigh ? "high" : "medium"}">
            ${isHigh ? `High Confidence (${pct}%)` : `Alternative (${pct}%)`}
          </span>
        </div>
        <div class="suggestion-text"></div>
      `;
      const trimmedWords = (item.text || "").split(/\s+/).slice(0, detailWords).join(" ");
      card.querySelector(".suggestion-text").textContent = trimmedWords;
      suggestionsGrid.appendChild(card);
    });
  }

  // Audio Capture: 16kHz PCM AudioWorklet -> Faster-Whisper
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
      isListening = true;
      updateListeningUI();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "listening_toggled", listening: true }));
      }
    } catch (err) {
      console.error("Failed to access microphone:", err);
      alert("Could not access microphone. Please allow microphone permissions in your browser.");
    }
  }

  function stopListening() {
    isListening = false;
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
      listeningLabel.textContent = "Listening";
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

  function pushSettingsUpdate() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "update_settings",
          count: suggestionCount,
          detail_words: detailWords,
        })
      );
    }
    if (isShowingSuggestions) {
      showSuggestions();
    }
  }

  // Sliders inside Settings Menu
  countSliderEl.addEventListener("input", () => {
    suggestionCount = parseInt(countSliderEl.value, 10) || 6;
    countValEl.textContent = String(suggestionCount);
    pushSettingsUpdate();
  });

  detailSliderEl.addEventListener("input", () => {
    detailWords = parseInt(detailSliderEl.value, 10) || 6;
    detailValEl.textContent = String(detailWords);
    pushSettingsUpdate();
  });

  // Settings menu toggle
  btnSettings.addEventListener("click", () => {
    settingsMenu.classList.toggle("hidden");
    btnSettings.classList.toggle("open");
  });

  btnToggleListen.addEventListener("click", () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  });

  btnStuck.addEventListener("click", () => toggleSuggestions());

  // Spacebar toggles suggestions on/off
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.code === "Space" || e.key === " ") {
      e.preventDefault();
      toggleSuggestions();
    }
  });

  connectWebSocket();
})();
