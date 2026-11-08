// Node harness for server/megamind/xr/static/voice.js (run by tests/test_xr_voice.py).
// argv[2] = file URL of a copy of voice.js. No browser: the mic is a fake capture, time is a fake clock.
import assert from "node:assert/strict";

const voice = await import(process.argv[2]);
const { VoiceRecorder, encodeWav, downsample, micUnavailableReason, describeMicError, HOLD_MS } = voice;

// --- pure helpers
{
  const wav = new DataView(encodeWav(new Float32Array([0, 1, -1, 0.5]), 16000));
  const tag = (o) => String.fromCharCode(...[0, 1, 2, 3].map((i) => wav.getUint8(o + i)));
  assert.equal(tag(0), "RIFF");
  assert.equal(tag(8), "WAVE");
  assert.equal(wav.getUint16(22, true), 1); // mono
  assert.equal(wav.getUint32(24, true), 16000);
  assert.equal(wav.getUint16(34, true), 16); // 16-bit
  assert.equal(wav.getUint32(40, true), 8);
  assert.equal(wav.getInt16(46, true), 0x7fff);
  assert.equal(wav.getInt16(48, true), -0x8000);

  const d = downsample(new Float32Array([1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5]), 48000, 16000);
  assert.deepEqual(Array.from(d), [1, 0, 0.5]);
  const same = new Float32Array([0.1, 0.2]);
  assert.equal(downsample(same, 16000, 16000), same);

  const insecure = micUnavailableReason({ isSecureContext: false, location: { port: "8123" } });
  assert.match(insecure, /secure page/);
  assert.match(insecure, /localhost:8123\/xr/);
  assert.match(insecure, /adb reverse tcp:8123 tcp:8123/);
  assert.match(micUnavailableReason({ isSecureContext: true, navigator: {} }), /can't record audio/);
  assert.equal(
    micUnavailableReason({ isSecureContext: true, navigator: { mediaDevices: { getUserMedia() {} } }, AudioContext() {} }),
    null,
  );
  assert.match(describeMicError({ name: "NotAllowedError" }), /permission denied/);
  assert.match(describeMicError({ name: "NotFoundError" }), /No microphone/);
}

// --- recorder state machine
let now = 1000;
performance.now = () => now;
globalThis.isSecureContext = true;
Object.defineProperty(globalThis, "navigator", {
  value: { mediaDevices: { getUserMedia() {} } },
  configurable: true,
});
globalThis.AudioContext = function AudioContext() {};

const flush = () => new Promise((r) => setTimeout(r, 0));

class FakeCapture {
  constructor({ ms = 1200, openError = null, gate = null } = {}) {
    Object.assign(this, { ms, openError, gate, opened: 0, released: 0 });
  }
  async open() {
    this.opened++;
    if (this.gate) await this.gate;
    if (this.openError) throw this.openError;
  }
  close() {
    this.released++;
    return { wav: new ArrayBuffer(44 + 32), ms: this.ms };
  }
  release() {
    this.released++;
  }
}

function make(opts = {}, transcript = "where are my keys") {
  const sent = [];
  const phases = [];
  const capture = new FakeCapture(opts);
  const r = new VoiceRecorder({
    capture,
    onAudio: async (blob) => {
      sent.push(blob);
      return typeof transcript === "function" ? transcript() : transcript;
    },
    onChange: (rec) => {
      if (phases.at(-1) !== rec.phase) phases.push(rec.phase);
    },
  });
  return { r, sent, phases, capture };
}

// hold to talk: a long press sends on release
{
  const { r, sent, phases, capture } = make();
  r.press();
  await flush();
  assert.equal(r.phase, "listening");
  assert.match(r.title, /Release to send/);
  now += HOLD_MS + 500;
  r.release();
  await flush();
  assert.deepEqual(phases, ["starting", "listening", "transcribing", "idle"]);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].type, "audio/wav");
  assert.equal(capture.opened, 1);
  assert.ok(capture.released >= 1, "mic released after recording");
  r.dispose();
}

// tap to toggle: a short press keeps listening; the next tap sends
{
  const { r, sent, phases } = make();
  r.press();
  await flush();
  now += 100;
  r.release();
  assert.equal(r.phase, "listening");
  assert.equal(r.mode, "toggle");
  assert.match(r.title, /Tap again to send/);
  now += 2000;
  r.press();
  await flush();
  assert.deepEqual(phases, ["starting", "listening", "transcribing", "idle"]);
  assert.equal(sent.length, 1);
  r.dispose();
}

// keyboard / in-headset toggle()
{
  const { r, sent } = make();
  r.toggle();
  await flush();
  assert.equal(r.phase, "listening");
  r.toggle();
  await flush();
  assert.equal(r.phase, "idle");
  assert.equal(sent.length, 1);
  r.dispose();
}

// too short a clip is an error, nothing is sent
{
  const { r, sent } = make({ ms: 100 });
  r.toggle();
  await flush();
  r.toggle();
  await flush();
  assert.equal(r.phase, "error");
  assert.match(r.error, /Too short/);
  assert.equal(sent.length, 0);
  r.dispose();
}

// nothing heard (empty transcript)
{
  const { r, sent } = make({}, "  ");
  r.toggle();
  await flush();
  r.toggle();
  await flush();
  assert.equal(sent.length, 1);
  assert.equal(r.phase, "error");
  assert.match(r.error, /Didn't catch that/);
  r.dispose();
}

// superseded by a newer question (onAudio resolves null): back to idle, no "didn't catch that"
{
  const { r, sent } = make({}, null);
  r.toggle();
  await flush();
  r.toggle();
  await flush();
  assert.equal(sent.length, 1);
  assert.equal(r.phase, "idle");
  assert.equal(r.error, null);
  r.dispose();
}

// server failure surfaces its message
{
  const { r } = make({}, () => {
    throw new Error("500 stt down");
  });
  r.toggle();
  await flush();
  r.toggle();
  await flush();
  assert.equal(r.phase, "error");
  assert.match(r.error, /500 stt down/);
  r.dispose();
}

// permission denied
{
  const { r, sent } = make({ openError: Object.assign(new Error("denied"), { name: "NotAllowedError" }) });
  r.press();
  await flush();
  assert.equal(r.phase, "error");
  assert.match(r.error, /permission denied/);
  assert.equal(sent.length, 0);
  r.dispose();
}

// insecure page (LAN IP over http): explained, the mic is never requested
{
  globalThis.isSecureContext = false;
  const { r, capture } = make();
  r.press();
  await flush();
  assert.equal(r.phase, "error");
  assert.match(r.error, /secure page/);
  assert.equal(capture.opened, 0);
  r.dispose();
  globalThis.isSecureContext = true;
}

// released (after a hold) while the permission prompt is still up: stops as soon as the mic opens
{
  let open;
  const gate = new Promise((res) => (open = res));
  const { r, sent } = make({ gate });
  r.press();
  await flush();
  assert.equal(r.phase, "starting");
  now += HOLD_MS + 50;
  r.release();
  open();
  await flush();
  await flush();
  assert.equal(r.phase, "idle");
  assert.equal(sent.length, 1);
  r.dispose();
}

// transcribing blocks new presses
{
  let finish;
  const pending = new Promise((res) => (finish = res));
  const { r, capture } = make({}, () => pending);
  r.toggle();
  await flush();
  r.toggle();
  await flush();
  assert.equal(r.phase, "transcribing");
  r.press();
  r.toggle();
  await flush();
  assert.equal(capture.opened, 1);
  finish("hello");
  await flush();
  assert.equal(r.phase, "idle");
  r.dispose();
}

console.log("ok");
