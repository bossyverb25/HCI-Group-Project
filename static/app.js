/**
 * The Invisible Interpreter — Prototype v1 Frontend Client
 * --------------------------------------------------------
 * - Streams 16kHz mono PCM audio over WebSocket exclusively to Faster-Whisper (`backend/transcription.py`).
 * - Maintains the latest pre-generated phrase completion bundle pushed from
 *   the backend's 30-second rolling Ollama cache (`backend/judgment.py`).
 * - Clicking "I'm Stuck" (or Spacebar) displays a snapshot of the latest set of suggestions
 *   (no hide-suggestions toggle; stays at "I'm Stuck").
 * - Clicking the top-right Mic Off / Listening pill toggles the microphone on/off.
 */

(() => {
  // UI Elements
  const listeningPill = document.getElementById("listening-pill");
  const listeningLabel = document.getElementById("listening-label");
  const connectionBadge = document.getElementById("connection-badge");
  const connectionStateEl = document.getElementById("connection-state");
  const contextSnippetEl = document.getElementById("context-snippet");

  const blurTarget = document.getElementById("blur-target");
  const micOverlay = document.getElementById("mic-overlay");
  const btnOverlayMic = document.getElementById("btn-overlay-mic");

  const calmIdleBox = document.getElementById("calm-idle-box");
  const suggestionsList = document.getElementById("suggestions-list");

  const settingsMenu = document.getElementById("settings-menu");
  const btnSettings = document.getElementById("btn-settings");
  const countSliderEl = document.getElementById("suggestion-count-slider");
  const countValEl = document.getElementById("suggestion-count-value");
  const detailSliderEl = document.getElementById("detail-slider");
  const detailValEl = document.getElementById("detail-value");
  const settingsLangSelect = document.getElementById("settings-language-select");
  const overlayLangSelect = document.getElementById("overlay-language-select");

  const btnPauseMic = document.getElementById("btn-pause-mic");
  const btnPrimaryAction = document.getElementById("btn-primary-action");
  const primaryActionText = document.getElementById("primary-action-text");

  // State
  let ws = null;
  let isListening = false;
  let hasStartedMic = false;
  let audioCtx = null;
  let mediaStream = null;
  let workletNode = null;
  let suggestionCount = parseInt(countSliderEl.value, 10) || 4;
  let detailWords = parseInt(detailSliderEl.value, 10) || 6;
  let latestTranscript = "";

  // Rolling pre-generated cache pushed continuously from the server
  let warmCache = {
    phrases: Array.from({ length: 4 }, () => ({ text: "..." })),
  };

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const basePath = window.location.pathname.endsWith("/")
      ? window.location.pathname
      : `${window.location.pathname}/`;
    ws = new WebSocket(`${protocol}//${window.location.host}${basePath}ws`);
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
      }
    } else if (msg.type === "transcript_update") {
      if (msg.last_snippet) {
        latestTranscript = msg.full_text || msg.last_snippet;
        contextSnippetEl.textContent = msg.last_snippet;
        contextSnippetEl.classList.remove("placeholder");
      }
    } else if (msg.type === "cache_update") {
      // Silently update background cache without overwriting currently displayed suggestions
      if (msg.cache) {
        warmCache = msg.cache;
      }
    } else if (msg.type === "im_stuck_result") {
      if (msg.all_cached) {
        warmCache = msg.all_cached;
      }
      renderSuggestionsList(msg.suggestions);
    }
  }

  // Sync all UI states (blur overlay, pause button, primary button label)
  function updateUIState() {
    if (isListening) {
      listeningPill.classList.remove("off");
      listeningPill.classList.add("on");
      listeningLabel.textContent = "Listening";
    } else {
      listeningPill.classList.remove("on");
      listeningPill.classList.add("off");
      listeningLabel.textContent = "Mic Off";
    }

    if (!isListening && !hasStartedMic) {
      blurTarget.classList.add("blurred");
      micOverlay.classList.remove("hidden");
    } else {
      blurTarget.classList.remove("blurred");
      micOverlay.classList.add("hidden");
    }

    if (isListening) {
      btnPauseMic.classList.remove("hidden");
      btnPrimaryAction.classList.remove("turn-mic-on");
      primaryActionText.textContent = "I'm Stuck";
    } else {
      btnPauseMic.classList.add("hidden");
      if (hasStartedMic) {
        btnPrimaryAction.classList.add("turn-mic-on");
        primaryActionText.textContent = "Turn Mic On";
      } else {
        btnPrimaryAction.classList.remove("turn-mic-on");
        primaryActionText.textContent = "I'm Stuck";
      }
    }
  }

  // Show the latest set of suggestions from cache and log to backend
  function showLatestSuggestions() {
    if (!isListening) return;
    renderSuggestionsList((warmCache.phrases || []).slice(0, suggestionCount));
    updateUIState();

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

  function renderSuggestionsList(items) {
    if (!items || items.length === 0) return;

    calmIdleBox.classList.add("hidden");
    suggestionsList.classList.remove("hidden");
    suggestionsList.innerHTML = "";

    const visibleItems = items.slice(0, suggestionCount);

    visibleItems.forEach((item, idx) => {
      const trimmedWords = (item.text || "").split(/\s+/).slice(0, detailWords).join(" ");
      const bubble = document.createElement("div");
      bubble.className = idx === 0 ? "suggestion-bubble-primary" : "suggestion-bubble-alt";
      bubble.style.animationDelay = `${idx * 25}ms`;
      bubble.textContent = trimmedWords;
      suggestionsList.appendChild(bubble);
    });
  }

  // Audio Capture: 16kHz PCM AudioWorklet -> Faster-Whisper
  async function startListening() {
    if (isListening) return;
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
      await audioCtx.audioWorklet.addModule("static/audio-processor.js");

      const source = audioCtx.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioCtx, "pcm16-downsampler");

      workletNode.port.onmessage = (e) => {
        if (isListening && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };

      source.connect(workletNode);
      isListening = true;
      hasStartedMic = true;
      updateUIState();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "listening_toggled", listening: true }));
      }
    } catch (err) {
      console.error("Failed to access microphone:", err);
      alert("Could not access microphone. Please allow microphone permissions in your browser.");
    }
  }

  function stopListening() {
    if (!isListening) return;
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
    updateUIState();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "listening_toggled", listening: false }));
    }
  }

  function toggleListening() {
    if (isListening) {
      stopListening();
    } else {
      startListening();
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
  }

  function closeSettingsMenu() {
    if (!settingsMenu.classList.contains("collapsed")) {
      settingsMenu.classList.add("collapsed");
      btnSettings.classList.remove("open");
      btnSettings.setAttribute("aria-expanded", "false");
    }
  }

  // Settings Sliders & Collapsible Menu
  btnSettings.addEventListener("click", (e) => {
    e.stopPropagation();
    const isCollapsed = settingsMenu.classList.toggle("collapsed");
    btnSettings.classList.toggle("open", !isCollapsed);
    btnSettings.setAttribute("aria-expanded", String(!isCollapsed));
  });

  // Auto-collapse settings menu whenever any other element on the site is clicked/pressed
  document.addEventListener("pointerdown", (e) => {
    if (
      !settingsMenu.classList.contains("collapsed") &&
      !settingsMenu.contains(e.target) &&
      !btnSettings.contains(e.target)
    ) {
      closeSettingsMenu();
    }
  });

  // Keep both language dropdowns in sync (placeholder for future translation support)
  if (overlayLangSelect && settingsLangSelect) {
    overlayLangSelect.addEventListener("change", () => {
      settingsLangSelect.value = overlayLangSelect.value;
    });
    settingsLangSelect.addEventListener("change", () => {
      overlayLangSelect.value = settingsLangSelect.value;
    });
  }

  countSliderEl.addEventListener("input", () => {
    suggestionCount = parseInt(countSliderEl.value, 10) || 4;
    countValEl.textContent = String(suggestionCount);
    pushSettingsUpdate();
  });

  detailSliderEl.addEventListener("input", () => {
    detailWords = parseInt(detailSliderEl.value, 10) || 6;
    detailValEl.textContent = String(detailWords);
    pushSettingsUpdate();
  });

  // Button Listeners
  listeningPill.addEventListener("click", () => toggleListening());
  btnOverlayMic.addEventListener("click", () => startListening());
  btnPauseMic.addEventListener("click", () => stopListening());

  btnPrimaryAction.addEventListener("click", () => {
    if (!isListening) {
      startListening();
    } else {
      showLatestSuggestions();
    }
  });

  // Spacebar shows latest suggestions when listening
  window.addEventListener("keydown", (e) => {
    if (
      e.target.tagName === "INPUT" ||
      e.target.tagName === "TEXTAREA" ||
      e.target.tagName === "SELECT"
    ) {
      return;
    }
    if (e.code === "Space" || e.key === " ") {
      e.preventDefault();
      if (isListening) {
        showLatestSuggestions();
      }
    }
  });

  updateUIState();
  connectWebSocket();
})();
