/**
 * /xr page controller: wires the Memory Theater to the Megamind v1 API.
 *
 *   GET  /v1/memories/recent   newest memories (cards) and their thumbnails (/v1/thumbs/<id>.jpg)
 *   POST /v1/chat/stream       ask (SSE): answer text as it is generated, then the POST /v1/chat body
 *   POST /v1/chat              ask -> answer + cited[] (thumb_url, replay_url) + optional TTS audio_url
 *                              (fallback when the server has no /v1/chat/stream)
 *   GET  /v1/replay.mp4        the cited moment (replay_url), or a +-5 s window around a picked card
 *
 * URL params: ?device=<device_id> scopes memories/chat/replay to one headset; ?session=<id> keeps a chat
 * session across reloads (default: a fresh one per page load).
 *
 * Ported from Recall's frontend/src/xr/XRMode.tsx (React) to plain DOM code.
 */
import { MemoryTheater } from "./theater.js";

const params = new URLSearchParams(window.location.search);
const DEVICE = params.get("device") || null;
const SESSION = params.get("session") || `xr-${Math.random().toString(36).slice(2, 10)}`;
const RECENT_K = 24;
const REFRESH_MS = 30_000;
const PAD_S = 5; // matches the server's REPLAY_PAD_BEFORE_S / REPLAY_PAD_AFTER_S defaults

const $ = (id) => document.getElementById(id);
const els = {
  host: $("xr-host"),
  sub: $("xr-sub"),
  form: $("xr-ask"),
  input: $("xr-q"),
  enter: $("xr-enter"),
  hint: $("xr-hint"),
  error: $("xr-error"),
  status: $("xr-status"),
};

let recent = [];
let cited = [];
let askToken = 0;

function showError(msg) {
  els.error.textContent = msg || "";
  els.error.hidden = !msg;
}

function setStatus(msg) {
  els.status.textContent = msg;
}

async function httpError(r) {
  let detail = "";
  try {
    detail = (await r.json()).detail;
  } catch {
    /* not JSON */
  }
  return new Error(`${r.status} ${typeof detail === "string" ? detail : r.statusText}`);
}

async function getJSON(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) throw await httpError(r);
  return r.json();
}

/**
 * POST /v1/chat/stream, calling onText(textSoFar) as answer text arrives; resolves to the final event (the
 * POST /v1/chat body). Falls back to plain POST /v1/chat on servers without the stream endpoint.
 */
async function chat(init, onText) {
  const r = await fetch("/v1/chat/stream", init);
  if (r.status === 404 || r.status === 405 || !r.body) return getJSON("/v1/chat", init);
  if (!r.ok) throw await httpError(r);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let text = "";
  let final = null;
  let error = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      let event = "message";
      let data = "";
      for (const line of buf.slice(0, i).split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      buf = buf.slice(i + 2);
      if (!data) continue;
      const ev = JSON.parse(data);
      if (event === "delta") onText((text += ev.text));
      else if (event === "reset") onText((text = ""));
      else if (event === "final") final = ev;
      else if (event === "error") error = ev.detail;
    }
  }
  if (final) return final;
  throw new Error(error || "stream ended without an answer");
}

function withDevice(u) {
  if (DEVICE) u.searchParams.set("device_id", DEVICE);
  return u;
}

/** The newest memories (keyframes with thumbnails), oldest first so the arc reads left to right. */
async function fetchRecent() {
  const u = withDevice(new URL("/v1/memories/recent", window.location.href));
  u.searchParams.set("k", String(RECENT_K));
  const { results } = await getJSON(u);
  return results.sort((a, b) => a.ts - b.ts);
}

function renderCards() {
  // Cited moments that are not among the recent ones still get a card, so they can be re-played.
  const byId = new Map(recent.map((h) => [h.id, h]));
  for (const c of cited) if (!byId.has(c.id)) byId.set(c.id, c);
  const hits = [...byId.values()].sort((a, b) => a.ts - b.ts);
  theater.setMemories(
    hits,
    cited.map((c) => c.id),
  );
  setStatus(hits.length ? `${hits.length} moments` : "No memories yet: feed frames with tools/feeder.py");
}

async function refreshRecent() {
  try {
    recent = await fetchRecent();
    renderCards();
  } catch (e) {
    setStatus(`Could not load memories (${e.message})`);
  }
}

function replayFor(m) {
  if (m.replay_url) return m.replay_url;
  if (!m.thumb_url) return null; // speech-only tokens have no frames to replay
  const u = withDevice(new URL("/v1/replay.mp4", window.location.href));
  u.searchParams.set("from", (m.ts - PAD_S).toFixed(3));
  u.searchParams.set("to", (m.ts + PAD_S).toFixed(3));
  return u.pathname + u.search;
}

async function ask(question) {
  const q = question.trim();
  if (!q) return;
  const token = ++askToken;
  showError("");
  theater.setThinking(q);
  try {
    const body = { session_id: SESSION, text: q };
    if (DEVICE) body.device_id = DEVICE;
    const init = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const resp = await chat(init, (text) => {
      if (token === askToken) theater.setAnswer(q, text || "Searching your day…");
    });
    if (token !== askToken) return; // a newer question superseded this one
    cited = resp.cited || [];
    theater.setAnswer(q, resp.answer || "(no answer)", cited.length > 0);
    renderCards();
    const first = cited.find((c) => c.replay_url) || cited.find((c) => c.thumb_url);
    if (first) theater.play(first);
    if (resp.audio_url) new Audio(resp.audio_url).play().catch(() => {});
  } catch (e) {
    if (token !== askToken) return;
    theater.setAnswer(q, `Something went wrong: ${e.message}`, false);
  }
}

const theater = new MemoryTheater(els.host, {
  onSelectMemory: (m) => theater.play({ ...m, replay_url: replayFor(m) }),
  onAsk: (q) => ask(q),
  onExit: () => {
    els.enter.disabled = false;
    els.enter.textContent = els.enter.dataset.label;
  },
});
window.memoryTheater = theater; // handy for debugging from the console

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value;
  els.input.value = "";
  ask(q);
});

els.enter.addEventListener("click", async () => {
  showError("");
  try {
    const mode = await theater.enterXR();
    if (mode === "ar" || mode === "vr") {
      els.enter.disabled = true;
      els.enter.textContent = "In headset";
    }
  } catch (e) {
    showError(e instanceof Error ? e.message : String(e));
  }
});

{
  // The hint names this server's port, which is what `adb reverse` must forward.
  const port = window.location.port || (window.location.protocol === "https:" ? "443" : "80");
  const codes = els.hint.querySelectorAll("code");
  if (codes.length === 2) {
    codes[0].textContent = `adb reverse tcp:${port} tcp:${port}`;
    codes[1].textContent = `localhost:${port}/xr`;
  }
}

MemoryTheater.xrSupport().then((support) => {
  if (support) {
    els.enter.dataset.label = support === "ar" ? "Enter passthrough" : "Enter VR";
    els.enter.textContent = els.enter.dataset.label;
    els.enter.hidden = false;
    els.hint.hidden = true;
    els.sub.textContent = support === "ar" ? "Quest passthrough ready" : "VR ready";
  } else {
    els.sub.textContent = "3D preview: drag to look, click to select";
  }
});

refreshRecent();
setInterval(() => {
  if (!document.hidden) refreshRecent();
}, REFRESH_MS);
