/**
 * Memory Theater: a WebXR (Meta Quest 3) view of your day, served by the Megamind server at /xr.
 *
 * A large floating screen replays the cited moment (`replay_url`, an mp4 from GET /v1/replay.mp4) in
 * passthrough (immersive-ar, falling back to immersive-vr). When no replay is available the screen shows the
 * cited thumbnail instead. Recent memories float as cards in an arc below it, the latest answer is on a
 * panel to the left, and suggested questions are buttons on the right. Controllers (or hand pinch, which
 * WebXR reports as a "select") point-and-select. Without WebXR it renders the same scene inline
 * ("preview") with mouse drag-to-look and click picking.
 *
 * Ported from Recall's frontend/src/xr/theater.ts (TypeScript + bundler) to a plain ES module: three.js
 * comes from the importmap in index.html, so the server needs no frontend build.
 *
 * Main-thread budget: the render loop only raycasts and draws. Thumbnails are fetched and decoded off the
 * main thread (fetch + createImageBitmap); canvas panels are redrawn only when their content changes.
 */
import * as THREE from "three";

export const SUGGESTIONS = [
  "Where did I leave my keys?",
  "Who did I meet today?",
  "What did I do this afternoon?",
  "What did I see most recently?",
];
const MAX_CARDS = 24;

export function formatClock(ts) {
  if (typeof ts !== "number" || !Number.isFinite(ts)) return "";
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Decode an image without blocking the main thread. Resolves null on any failure. */
export async function loadBitmap(url) {
  if (!url) return null;
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    const blob = await r.blob();
    if ("createImageBitmap" in window) return await createImageBitmap(blob);
    const img = new Image();
    img.src = URL.createObjectURL(blob);
    await img.decode();
    return img;
  } catch {
    return null;
  }
}

function canvasPanel(w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return { canvas, ctx, tex };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function wrap(ctx, text, maxW) {
  const out = [];
  for (const para of String(text).split("\n")) {
    let line = "";
    for (const word of para.split(/\s+/)) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxW && line) {
        out.push(line);
        line = word;
      } else line = test;
    }
    out.push(line);
  }
  return out;
}

/** Draw `img` into the (x, y, w, h) box, cropping to cover it. */
function drawCover(ctx, img, x, y, w, h) {
  const r = Math.max(w / img.width, h / img.height);
  ctx.drawImage(img, x + (w - img.width * r) / 2, y + (h - img.height * r) / 2, img.width * r, img.height * r);
}

export class MemoryTheater {
  /**
   * @param {HTMLElement} container
   * @param {{onSelectMemory: (m: object) => void, onAsk: (q: string) => void, onExit: () => void}} cb
   */
  constructor(container, cb) {
    this.container = container;
    this.cb = cb;
    this.scene = new THREE.Scene();
    this.screenLabel = canvasPanel(1024, 96);
    this.answer = canvasPanel(900, 1000);
    this.still = canvasPanel(960, 540); // thumbnail fallback shown on the main screen
    this.cardsGroup = new THREE.Group();
    this.selectables = [];
    this.controllers = [];
    this.raycaster = new THREE.Raycaster();
    this.hovered = null;
    this.pointer = new THREE.Vector2(-10, -10);
    this.yaw = 0;
    this.pitch = -0.06;
    this.drag = null;
    this.disposed = false;
    this.cardMeshes = new Map();
    this.playToken = 0;
    this.mode = "preview";

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(80, container.clientWidth / container.clientHeight, 0.05, 50);
    // Preview camera sits a step back so the side panels are visible; in XR the headset pose takes over.
    this.camera.position.set(0, 1.5, 0.35);
    this.setBackground(true);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.2));

    // --- main screen: video texture for replays, canvas texture for the thumbnail fallback
    this.video = document.createElement("video");
    this.video.crossOrigin = "anonymous";
    this.video.playsInline = true;
    this.video.preload = "auto";
    this.video.muted = true; // autoplay-safe until a user gesture (enterXR) unmutes
    this.videoTex = new THREE.VideoTexture(this.video);
    this.videoTex.colorSpace = THREE.SRGBColorSpace;
    const frame = new THREE.Mesh(new THREE.PlaneGeometry(2.04, 1.2), new THREE.MeshBasicMaterial({ color: 0x0b0f17 }));
    frame.position.set(0, 1.62, -2.21);
    this.scene.add(frame);
    this.screen = new THREE.Mesh(
      new THREE.PlaneGeometry(1.92, 1.08),
      new THREE.MeshBasicMaterial({ map: this.still.tex, color: 0xffffff }),
    );
    this.screen.position.set(0, 1.62, -2.2);
    this.scene.add(this.screen);
    this.drawStill(null, "Ask a question or pick a moment below");
    const label = new THREE.Mesh(
      new THREE.PlaneGeometry(1.92, 0.18),
      new THREE.MeshBasicMaterial({ map: this.screenLabel.tex, transparent: true }),
    );
    label.position.set(0, 2.3, -2.2);
    this.scene.add(label);
    this.drawLabel("Megamind · your day", "Ask a question or pick a moment below");

    // tap the screen to play / pause / replay
    this.screen.userData.onSelect = () => {
      if (this.screen.material.map !== this.videoTex) return;
      if (this.video.ended) this.video.currentTime = 0;
      if (this.video.paused) this.video.play().catch(() => {});
      else this.video.pause();
    };
    this.selectables.push(this.screen);

    // --- answer panel (left)
    const ans = new THREE.Mesh(
      new THREE.PlaneGeometry(0.9, 1.0),
      new THREE.MeshBasicMaterial({ map: this.answer.tex, transparent: true }),
    );
    ans.position.set(-1.65, 1.6, -1.55);
    ans.rotation.y = 0.75;
    this.scene.add(ans);
    this.setAnswer(null, "Your latest answer will appear here.");

    // --- suggestion buttons (right) + exit
    SUGGESTIONS.forEach((q, i) =>
      this.addButton(q, new THREE.Vector3(1.62, 2.0 - i * 0.2, -1.5), -0.75, () => this.cb.onAsk(q)),
    );
    this.addButton(
      "Exit",
      new THREE.Vector3(1.62, 2.0 - SUGGESTIONS.length * 0.2 - 0.06, -1.5),
      -0.75,
      () => this.exit(),
      "#3a1d24",
    );

    this.scene.add(this.cardsGroup);
    this.setupControllers();
    this.setupPointer();
    this.onResize = this.onResize.bind(this);
    this.tick = this.tick.bind(this);
    window.addEventListener("resize", this.onResize);
    this.renderer.setAnimationLoop(this.tick);
  }

  // ---------- public API

  /** Recent memories (MemoryHit[] from GET /v1/memories) as cards; `highlighted` ids glow (cited). */
  setMemories(hits, highlighted = []) {
    let pool = hits.filter((h) => h.thumb_url); // speech tokens have no image; they live in the answer text
    if (pool.length > MAX_CARDS) {
      const keep = new Set(highlighted);
      const step = (pool.length - 1) / (MAX_CARDS - 1);
      pool = pool.filter((h, i) => keep.has(h.id) || Math.abs(i / step - Math.round(i / step)) < 1e-6);
    }
    const ids = pool.map((h) => h.id).join(",");
    if (this.cardsGroup.userData.ids !== ids) {
      this.cardsGroup.userData.ids = ids;
      this.buildCards(pool);
    }
    this.highlight(highlighted);
  }

  highlight(ids) {
    const hl = new Set(ids);
    for (const [id, mesh] of this.cardMeshes) mesh.userData.glow.visible = hl.has(id);
  }

  setAnswer(question, text, found = true) {
    const { ctx, canvas, tex } = this.answer;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(15,20,32,0.92)";
    roundRect(ctx, 0, 0, canvas.width, canvas.height, 36);
    ctx.fill();
    let y = 70;
    if (question) {
      ctx.fillStyle = "#8fb3ff";
      ctx.font = "600 38px system-ui, sans-serif";
      for (const l of wrap(ctx, `“${question}”`, canvas.width - 80).slice(0, 3)) {
        ctx.fillText(l, 40, y);
        y += 48;
      }
      y += 20;
    }
    ctx.fillStyle = found ? "#f2f5fa" : "#a9b1c2";
    ctx.font = "400 40px system-ui, sans-serif";
    for (const l of wrap(ctx, text, canvas.width - 80).slice(0, 16)) {
      ctx.fillText(l, 40, y);
      y += 52;
    }
    tex.needsUpdate = true;
  }

  setThinking(question) {
    this.setAnswer(question, "Searching your day…");
  }

  /**
   * Show a moment on the main screen. `m` is a Citation or MemoryHit ({id, ts, caption, thumb_url,
   * replay_url?}). Plays `replay_url` when present; on a missing or failing replay it shows the thumbnail.
   */
  play(m) {
    if (!m) return;
    const token = ++this.playToken;
    const title = formatClock(m.ts) || "Memory";
    const sub = (m.caption || "").slice(0, 120);
    const showStill = (note) => {
      if (token !== this.playToken) return;
      this.video.pause();
      this.screen.material.map = this.still.tex;
      this.screen.material.needsUpdate = true;
      this.drawLabel(title, note ? `${note} · ${sub}` : sub);
      this.drawStill(null, m.caption || "");
      loadBitmap(m.thumb_url).then((img) => {
        if (token === this.playToken && img) this.drawStill(img, "");
      });
    };
    if (!m.replay_url) {
      showStill("Still frame (no replay buffered)");
      return;
    }
    this.drawLabel(title, `Loading replay… · ${sub}`);
    const v = this.video;
    const onReady = () => {
      if (token !== this.playToken) return;
      this.screen.material.map = this.videoTex;
      this.screen.material.needsUpdate = true;
      this.drawLabel(title, sub);
      v.currentTime = 0;
      v.play().catch(() => {});
    };
    const onError = () => showStill("Replay unavailable");
    v.onloadeddata = onReady;
    v.onerror = onError;
    v.src = new URL(m.replay_url, window.location.href).href;
    v.load();
  }

  /** Must be called from a user gesture (button click). */
  async enterXR() {
    this.video.muted = false; // unlock audio playback with the same gesture
    const xr = navigator.xr;
    if (!xr) return "preview";
    const mode = (await xr.isSessionSupported("immersive-ar").catch(() => false))
      ? "immersive-ar"
      : (await xr.isSessionSupported("immersive-vr").catch(() => false))
        ? "immersive-vr"
        : null;
    if (!mode) return "preview";
    const session = await xr.requestSession(mode, { optionalFeatures: ["local-floor", "hand-tracking"] });
    this.renderer.xr.setReferenceSpaceType("local-floor");
    await this.renderer.xr.setSession(session);
    this.mode = mode === "immersive-ar" ? "ar" : "vr";
    this.setBackground(this.mode !== "ar");
    session.addEventListener("end", () => {
      this.mode = "preview";
      this.setBackground(true);
      this.onResize();
      this.cb.onExit();
    });
    return this.mode;
  }

  exit() {
    const s = this.renderer.xr.getSession();
    if (s) s.end().catch(() => {});
    else this.cb.onExit();
  }

  static async xrSupport() {
    const xr = navigator.xr;
    if (!xr || !window.isSecureContext) return null;
    if (await xr.isSessionSupported("immersive-ar").catch(() => false)) return "ar";
    if (await xr.isSessionSupported("immersive-vr").catch(() => false)) return "vr";
    return null;
  }

  dispose() {
    this.disposed = true;
    this.renderer.xr.getSession()?.end().catch(() => {});
    this.renderer.setAnimationLoop(null);
    this.video.pause();
    this.video.removeAttribute("src");
    window.removeEventListener("resize", this.onResize);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---------- internals

  setBackground(opaque) {
    this.scene.background = opaque ? new THREE.Color(0x0a0e16) : null;
    if (opaque && !this.scene.getObjectByName("floor")) {
      const grid = new THREE.GridHelper(20, 40, 0x24304a, 0x161d2c);
      grid.name = "floor";
      this.scene.add(grid);
    }
    const floor = this.scene.getObjectByName("floor");
    if (floor) floor.visible = opaque;
  }

  drawLabel(title, sub) {
    const { ctx, canvas, tex } = this.screenLabel;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(15,20,32,0.85)";
    roundRect(ctx, 0, 0, canvas.width, canvas.height, 24);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "700 40px system-ui, sans-serif";
    ctx.fillText(title, 28, 44);
    ctx.fillStyle = "#b8c2d6";
    ctx.font = "400 26px system-ui, sans-serif";
    ctx.fillText(wrap(ctx, sub, canvas.width - 56)[0] ?? "", 28, 80);
    tex.needsUpdate = true;
  }

  drawStill(img, text) {
    const { ctx, canvas, tex } = this.still;
    ctx.fillStyle = "#0b0f17";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (img) drawCover(ctx, img, 0, 0, canvas.width, canvas.height);
    if (text) {
      ctx.fillStyle = "#6f7c96";
      ctx.font = "400 34px system-ui, sans-serif";
      const lines = wrap(ctx, text, canvas.width - 120).slice(0, 5);
      lines.forEach((l, i) => ctx.fillText(l, 60, canvas.height / 2 - (lines.length - 1) * 22 + i * 44));
    }
    tex.needsUpdate = true;
  }

  addButton(text, pos, rotY, onSelect, bg = "#1c2a4a") {
    const p = canvasPanel(720, 128);
    p.ctx.fillStyle = bg;
    roundRect(p.ctx, 4, 4, 712, 120, 40);
    p.ctx.fill();
    p.ctx.strokeStyle = "#6f8fd8";
    p.ctx.lineWidth = 4;
    p.ctx.stroke();
    p.ctx.fillStyle = "#f2f5fa";
    p.ctx.font = "600 44px system-ui, sans-serif";
    p.ctx.textBaseline = "middle";
    p.ctx.fillText(text, 36, 66);
    p.tex.needsUpdate = true;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(0.9, 0.16),
      new THREE.MeshBasicMaterial({ map: p.tex, transparent: true }),
    );
    m.position.copy(pos);
    m.rotation.y = rotY;
    m.userData.onSelect = onSelect;
    m.userData.hoverScale = 1.06;
    this.scene.add(m);
    this.selectables.push(m);
  }

  buildCards(hits) {
    const old = new Set(this.cardMeshes.values());
    this.selectables = this.selectables.filter((s) => !old.has(s));
    for (const m of old) {
      m.material.map?.dispose();
      m.material.dispose();
      m.geometry.dispose();
    }
    this.cardsGroup.clear();
    this.cardMeshes.clear();
    const n = hits.length;
    const radius = 1.75;
    const spread = Math.min(Math.PI * 0.6, n * 0.13);
    hits.forEach((h, i) => {
      const a = n === 1 ? 0 : -spread / 2 + (spread * i) / (n - 1);
      const p = canvasPanel(320, 220);
      const draw = (img) => {
        p.ctx.fillStyle = "#121826";
        roundRect(p.ctx, 0, 0, 320, 220, 18);
        p.ctx.fill();
        if (img) {
          p.ctx.save();
          roundRect(p.ctx, 8, 8, 304, 164, 12);
          p.ctx.clip();
          drawCover(p.ctx, img, 8, 8, 304, 164);
          p.ctx.restore();
        }
        p.ctx.fillStyle = "#e8edf6";
        p.ctx.font = "600 30px system-ui, sans-serif";
        p.ctx.fillText(formatClock(h.ts), 14, 206);
        p.tex.needsUpdate = true;
      };
      draw();
      loadBitmap(h.thumb_url).then((img) => {
        if (img && !this.disposed) draw(img);
      });
      const card = new THREE.Mesh(
        new THREE.PlaneGeometry(0.3, 0.206),
        new THREE.MeshBasicMaterial({ map: p.tex, transparent: true }),
      );
      const row = i % 2;
      card.position.set(Math.sin(a) * radius, 0.78 + row * 0.24, -Math.cos(a) * radius);
      card.lookAt(0, 0.95 + row * 0.24, 0);
      const glow = new THREE.Mesh(
        new THREE.PlaneGeometry(0.33, 0.236),
        new THREE.MeshBasicMaterial({ color: 0x7aa2ff, transparent: true, opacity: 0.9 }),
      );
      glow.position.z = -0.002;
      glow.visible = false;
      card.add(glow);
      card.userData.glow = glow;
      card.userData.onSelect = () => this.cb.onSelectMemory(h);
      card.userData.hoverScale = 1.12;
      this.cardsGroup.add(card);
      this.cardMeshes.set(h.id, card);
      this.selectables.push(card);
    });
  }

  setupControllers() {
    for (let i = 0; i < 2; i++) {
      const c = this.renderer.xr.getController(i);
      const ray = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
        new THREE.LineBasicMaterial({ color: 0x9fbaff }),
      );
      ray.scale.z = 4;
      c.add(ray);
      // "select" fires for controller triggers and for hand-tracking pinches alike.
      c.addEventListener("select", () => this.pick(c)?.userData.onSelect?.());
      this.scene.add(c);
      this.controllers.push(c);
    }
  }

  setupPointer() {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", (e) => (this.drag = { x: e.clientX, y: e.clientY, moved: false }));
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      if (this.drag && (Math.abs(e.clientX - this.drag.x) > 3 || Math.abs(e.clientY - this.drag.y) > 3)) {
        this.yaw -= (e.clientX - this.drag.x) * 0.004;
        this.pitch = Math.max(-0.9, Math.min(0.9, this.pitch - (e.clientY - this.drag.y) * 0.004));
        this.drag = { x: e.clientX, y: e.clientY, moved: true };
      }
    });
    el.addEventListener("pointerup", (e) => {
      if (this.drag && !this.drag.moved) {
        const r = el.getBoundingClientRect();
        this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const hit = this.raycaster.intersectObjects(this.selectables, false)[0];
        hit?.object.userData.onSelect?.();
      }
      this.drag = null;
    });
    el.addEventListener("pointerleave", () => {
      this.pointer.set(-10, -10);
      this.drag = null;
    });
  }

  pick(c) {
    const m = new THREE.Matrix4().extractRotation(c.matrixWorld);
    this.raycaster.ray.origin.setFromMatrixPosition(c.matrixWorld);
    this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(m);
    const hit = this.raycaster.intersectObjects(this.selectables, false)[0];
    return hit?.object ?? null;
  }

  onResize() {
    if (this.renderer.xr.isPresenting) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  tick() {
    if (this.disposed) return;
    let hover = null;
    if (this.renderer.xr.isPresenting) {
      for (const c of this.controllers) hover = hover ?? this.pick(c);
    } else {
      this.camera.rotation.set(this.pitch, this.yaw, 0, "YXZ");
      this.raycaster.setFromCamera(this.pointer, this.camera);
      hover = this.raycaster.intersectObjects(this.selectables, false)[0]?.object ?? null;
      this.renderer.domElement.style.cursor = hover ? "pointer" : this.drag ? "grabbing" : "grab";
    }
    if (hover !== this.hovered) {
      this.hovered?.scale.setScalar(1);
      if (hover?.userData.hoverScale) hover.scale.setScalar(hover.userData.hoverScale);
      this.hovered = hover;
    }
    this.renderer.render(this.scene, this.camera);
  }
}
