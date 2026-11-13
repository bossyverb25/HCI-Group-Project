/**
 * AudioWorkletProcessor that downsamples microphone audio to 16kHz mono
 * 16-bit signed PCM and emits ~250ms chunks (4000 samples) to the main thread
 * for streaming over WebSocket to Faster-Whisper.
 */
class PCM16DownsamplerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.chunkSize = 4000; // 250ms at 16kHz
    this.buffer = new Int16Array(this.chunkSize);
    this.bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }
    const channelData = input[0];
    const ratio = sampleRate / this.targetSampleRate;

    let i = 0;
    while (i < channelData.length) {
      const idx = Math.floor(i);
      const sample = Math.max(-1, Math.min(1, channelData[idx] || 0));
      const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      this.buffer[this.bufferIndex++] = int16;

      if (this.bufferIndex >= this.chunkSize) {
        const out = this.buffer.slice(0);
        this.port.postMessage(out.buffer, [out.buffer]);
        this.bufferIndex = 0;
      }
      i += ratio;
    }
    return true;
  }
}

registerProcessor("pcm16-downsampler", PCM16DownsamplerProcessor);
