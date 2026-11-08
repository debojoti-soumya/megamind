/**
 * Voice questions for the Memory Theater: hold to talk (release to send) or tap to start / tap again to stop.
 *
 * The mic is captured with Web Audio, downsampled to 16 kHz mono and encoded as a WAV, which is what the
 * server's STT providers expect (`megamind.agent.stt`, the same input the /ui push-to-talk sends). The clip
 * goes to `onAudio(wav)`, which posts it to `POST /v1/chat` (multipart `audio`): the existing STT
 * transcribes it and the agent answers in the same request, exactly like a typed question. There is no
 * separate transcription endpoint.
 *
 * The microphone is opened per question and released as soon as recording stops (the headset's mic
 * indicator goes off between questions).
 *
 * Ported from Recall's frontend/src/components/VoiceButton.tsx (React + MediaRecorder + /api/transcribe) to a
 * plain ES module: a `VoiceRecorder` state machine (no DOM) plus `mountVoiceButton` for the page's button.
 * The theater's in-headset "Ask by voice" button drives the same recorder.
 */

export const HOLD_MS = 350; // a press longer than this is hold-to-talk; shorter is a tap (toggle)
export const MAX_SECONDS = 30;
export const MIN_MS = 300; // shorter recordings are almost certainly accidental
export const TARGET_RATE = 16000;
const ERROR_FADE_MS = 6000;

/** Why the mic cannot be used on this page, or null when it can. */
export function micUnavailableReason(env = globalThis) {
  if (!env.isSecureContext) {
    const port = env.location?.port || "8000";
    return (
      `The mic needs a secure page. On the Quest, open http://localhost:${port}/xr ` +
      `(adb reverse tcp:${port} tcp:${port}), not the LAN IP.`
    );
  }
  if (!env.navigator?.mediaDevices?.getUserMedia || !(env.AudioContext || env.webkitAudioContext)) {
    return "This browser can't record audio.";
  }
  return null;
}

/** A user-facing message for a getUserMedia / recording failure. */
export function describeMicError(e) {
  const name = e && typeof e === "object" ? e.name : "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Microphone permission denied. Allow the mic for this site and try again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No microphone found.";
  if (name === "NotReadableError") return "The microphone is busy in another app.";
  return e instanceof Error ? e.message : String(e);
}

/** Box-filter downsample of mono float samples (`toRate <= fromRate`); returns the input when rates match. */
export function downsample(samples, fromRate, toRate = TARGET_RATE) {
  if (!fromRate || fromRate <= toRate) return samples;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = a; j < b; j++) sum += samples[j];
    out[i] = b > a ? sum / (b - a) : 0;
  }
  return out;
}

/** 16-bit PCM mono WAV (RIFF) from float samples in [-1, 1]. */
export function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const out = new ArrayBuffer(44 + n * 2);
  const v = new DataView(out);
  const str = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0, o = 44; i < n; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out;
}

function concat(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

const WORKLET =
  "class R extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];" +
  "if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('mm-xr-rec',R)";

/**
 * Mic -> float chunks. `open()` asks for the mic; `close()` returns the clip as a 16 kHz WAV ArrayBuffer and
 * releases the mic and the AudioContext.
 */
class MicCapture {
  async open() {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ac = new AC();
      if (this.ac.state === "suspended") await this.ac.resume();
      const src = this.ac.createMediaStreamSource(this.stream);
      if (this.ac.audioWorklet) {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
        try {
          await this.ac.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        this.node = new AudioWorkletNode(this.ac, "mm-xr-rec");
        this.node.port.onmessage = (e) => this.chunks.push(e.data);
        src.connect(this.node);
      } else {
        this.node = this.ac.createScriptProcessor(4096, 1, 1);
        this.node.onaudioprocess = (e) => this.chunks.push(e.inputBuffer.getChannelData(0).slice(0));
        src.connect(this.node);
        this.node.connect(this.ac.destination);
      }
    } catch (e) {
      this.release();
      throw e;
    }
  }

  /** Stop and return {wav, ms}: the 16 kHz WAV (null when nothing was captured) and its duration. */
  close() {
    const rate = this.ac?.sampleRate || TARGET_RATE;
    const samples = downsample(concat(this.chunks || []), rate, TARGET_RATE);
    this.release();
    const outRate = rate > TARGET_RATE ? TARGET_RATE : rate;
    return { wav: samples.length ? encodeWav(samples, outRate) : null, ms: (samples.length / outRate) * 1000 };
  }

  release() {
    try {
      this.node?.disconnect();
    } catch {
      /* ignore */
    }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ac?.close().catch(() => {});
    this.node = this.stream = this.ac = null;
  }
}

/**
 * Phases: idle -> starting -> listening -> transcribing -> idle, or -> error (fades back to idle).
 *
 * @param {{onAudio: (wav: Blob) => Promise<string>, onChange?: (r: VoiceRecorder) => void, capture?: object}} opts
 *   `onAudio` sends the clip and resolves to the transcript ("" = nothing heard); a rejection becomes an error.
 */
export class VoiceRecorder {
  constructor({ onAudio, onChange, capture } = {}) {
    this.onAudio = onAudio;
    this.onChange = onChange || (() => {});
    this.capture = capture || new MicCapture();
    this.phase = "idle";
    this.mode = "none"; // "pressing" (pointer is down) | "toggle" (tap started it; tap again to stop)
    this.error = null;
    this.elapsed = 0;
    this.startedAt = 0;
    this.pressAt = 0;
    this.stopWanted = false; // released before the mic had opened
    this.timer = null;
    this.fadeTimer = null;
  }

  get busy() {
    return this.phase === "starting" || this.phase === "listening";
  }

  get title() {
    if (this.phase === "listening")
      return this.mode === "toggle" ? "Listening. Tap again to send" : "Listening. Release to send";
    if (this.phase === "starting") return "Opening the mic…";
    if (this.phase === "transcribing") return "Transcribing…";
    return "Ask by voice: hold to talk, or tap to start and tap again to stop";
  }

  set(phase, error = null) {
    this.phase = phase;
    this.error = error;
    clearTimeout(this.fadeTimer);
    if (phase === "error") {
      this.fadeTimer = setTimeout(() => {
        if (this.phase === "error") this.set("idle");
      }, ERROR_FADE_MS);
    }
    this.onChange(this);
  }

  fail(msg) {
    clearInterval(this.timer);
    this.capture.release();
    this.mode = "none";
    this.set("error", msg);
  }

  /** Pointer down: start listening (or end a tap-started recording). */
  press() {
    if (this.phase === "transcribing") return;
    if (this.busy && this.mode === "toggle") {
      this.mode = "none";
      this.stop(); // second tap ends a toggled recording
      return;
    }
    if (this.busy) return;
    this.pressAt = performance.now();
    this.mode = "pressing";
    this.start();
  }

  /** Pointer up: a long press sends; a short one leaves it listening until the next tap. */
  release() {
    if (this.mode !== "pressing") return;
    if (performance.now() - this.pressAt >= HOLD_MS) {
      this.mode = "none";
      this.stop();
    } else {
      this.mode = "toggle";
      this.stopWanted = false;
      this.onChange(this); // the title changes to "tap again"
    }
  }

  /** Keyboard / in-headset button: tap to start, tap again to send. */
  toggle() {
    if (this.phase === "transcribing") return;
    if (this.busy) {
      this.mode = "none";
      this.stop();
    } else {
      this.mode = "toggle";
      this.start();
    }
  }

  async start() {
    const why = micUnavailableReason();
    if (why) return this.fail(why);
    this.elapsed = 0;
    this.stopWanted = false;
    this.set("starting");
    try {
      await this.capture.open();
    } catch (e) {
      return this.fail(describeMicError(e));
    }
    if (this.phase !== "starting") {
      this.capture.release(); // disposed (or failed) while the permission prompt was up
      return;
    }
    this.startedAt = performance.now();
    this.set("listening");
    this.timer = setInterval(() => {
      this.elapsed = (performance.now() - this.startedAt) / 1000;
      if (this.elapsed >= MAX_SECONDS) this.stop();
      else this.onChange(this);
    }, 100);
    if (this.stopWanted) this.stop();
  }

  stop() {
    if (this.phase === "starting") {
      this.stopWanted = true; // still waiting on getUserMedia
      return;
    }
    if (this.phase !== "listening") return;
    clearInterval(this.timer);
    const { wav, ms } = this.capture.close();
    this.mode = "none";
    if (!wav || ms < MIN_MS)
      return this.fail("Too short. Hold the mic button while you speak, or tap once to start and again to stop.");
    this.send(new Blob([wav], { type: "audio/wav" }));
  }

  async send(blob) {
    this.set("transcribing");
    try {
      const text = String((await this.onAudio(blob)) || "").trim();
      if (this.phase !== "transcribing") return;
      if (!text) return this.fail("Didn't catch that. Try again, a little closer to the mic.");
      this.set("idle");
    } catch (e) {
      if (this.phase === "transcribing") this.fail(describeMicError(e));
    }
  }

  dispose() {
    clearInterval(this.timer);
    clearTimeout(this.fadeTimer);
    this.capture.release();
    this.phase = "disposed";
  }
}

/** Wire a <button> (and an optional tip element) to a recorder; returns a render function for onChange. */
export function mountVoiceButton(button, tip, recorder) {
  const mic = button.querySelector(".voice-mic");
  const dots = button.querySelector(".voice-dots");
  const label = button.querySelector(".voice-label");

  button.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    try {
      button.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    recorder.press();
  });
  button.addEventListener("pointerup", () => recorder.release());
  button.addEventListener("pointercancel", () => recorder.release());
  button.addEventListener("contextmenu", (e) => e.preventDefault());
  button.addEventListener("keydown", (e) => {
    if (e.key !== " " && e.key !== "Enter") return;
    e.preventDefault();
    if (!e.repeat) recorder.toggle();
  });

  return function render(r = recorder) {
    button.className = `voice-btn voice-${r.phase}`;
    button.title = r.title;
    button.setAttribute("aria-label", r.title);
    button.setAttribute("aria-pressed", String(r.busy));
    if (mic) mic.classList.toggle("voice-hide", r.phase === "transcribing");
    if (dots) dots.hidden = r.phase !== "transcribing";
    if (label) {
      label.textContent = r.phase === "listening" ? `${r.elapsed.toFixed(1)}s` : "";
      label.hidden = r.phase !== "listening";
    }
    if (!tip) return;
    const msg = r.phase === "error" ? r.error : r.busy || r.phase === "transcribing" ? r.title : "";
    tip.textContent = msg || "";
    tip.hidden = !msg;
    tip.classList.toggle("voice-tip-error", r.phase === "error");
    tip.setAttribute("role", r.phase === "error" ? "alert" : "status");
  };
}
