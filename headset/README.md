# Megamind headset app (Unity, Quest 3)

Single Unity app that (1) captures passthrough keyframes and uploads them to the
server, and (2) hosts the voice-first AR chat panel. Both streams share this project
(design doc §2, decision D1). The project in `headset/Megamind/` is a complete Unity
project: open it in Unity Hub, let the packages resolve, run one menu item, build.

> Status: **playtested on a Quest 3 (Horizon OS v74 / Android 14), 2026-11-07.** Built headless
> (Unity 6000.0.66f2 + Meta SDK 85 / MRUK 85) into `Builds/megamind.apk`, sideloaded over USB and run
> against a laptop server with mock providers via `adb reverse tcp:8000 tcp:8000` +
> `megamind.json` (`{"serverBaseUrl":"http://127.0.0.1:8000"}`). Verified on device: camera + mic
> permission prompts, passthrough capture at 1280x960 / 1 fps uploading to `/v1/frames`, the
> world-locked panel in passthrough, B/Y re-anchor, right-trigger push-to-talk → `/v1/chat` →
> transcript bubble + reply with 3 cited thumbnails. Not yet exercised on device: hand-pinch PTT,
> the settings sub-panel buttons (controller-ray UI), TTS playback (mock TTS returns no audio),
> and real STT/TTS (needs API keys on the server). Ambient audio memory (`AmbientAudioRecorder`, below) was added
> after this playtest and has **not** run on a headset yet (verify on device).

New to Quest? Start with the **[Quest 3 guide](../docs/quest3-guide.md)**: headset basics, every physical
connection (USB/adb, wireless adb, Wi-Fi/Tailscale to the server, casting, power), a test runbook, and a
demo-day quick card.

## Exact versions
| Thing | Version | Where it is pinned |
|---|---|---|
| Unity Editor | **6000.0.66f2** (Unity 6, any 6000.0.38f1+ works) | `ProjectSettings/ProjectVersion.txt` |
| Meta XR Core SDK (`com.meta.xr.sdk.core`) | **85.0.0** | `Packages/manifest.json` |
| Meta MR Utility Kit (`com.meta.xr.mrutilitykit`) — provides `PassthroughCameraAccess` | **85.0.0** | `Packages/manifest.json` |
| OpenXR plugin / XR Management | 1.15.1 / 4.5.4 | `Packages/manifest.json` |
| Headset | Quest 3 / 3S, Horizon OS **v74+**, developer mode on | — |
| Android target | **ARM64, IL2CPP, min SDK 32, ASTC**, OpenXR (Meta Quest feature group) | `ProjectSettings/ProjectSettings.asset`, `Assets/XR/` |
| Package id | `com.megamind.quest` | `ProjectSettings/ProjectSettings.asset` |

The project settings, XR settings (`Assets/XR/`), `Assets/Oculus/OculusProjectConfig.asset`
(passthrough + passthrough camera access enabled, target Horizon OS SDK 74) and the
Android manifest were copied from Meta's official
[Unity-PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)
and renamed. The camera is read through MRUK's `Meta.XR.PassthroughCameraAccess`
component (the current official path; the older `WebCamTextureManager` sample is deprecated).

## Build + sideload, step by step
1. **Install Unity 6000.0.66f2** via Unity Hub with the *Android Build Support* module
   **including OpenJDK, Android SDK & NDK Tools**. (Hub → Installs → Install Editor → 6000.0.66f2.)
2. **Open the project**: Hub → Add → `headset/Megamind`. First open takes a few minutes
   while it downloads the Meta packages from the Unity registry. If a dialog asks to
   enable the new Input System / restart, accept.
3. **Switch platform to Android**: File → Build Profiles (or Build Settings) → Android → *Switch Platform*.
4. **Meta project setup**: Edit → Project Settings → Meta XR. Click **Fix All** / **Apply All**
   for *Required* items (it will not change the manifest permissions we need). Ignore
   *Recommended* items if short on time.
5. **Create the capture scene**: menu **Megamind → 1. Create Capture Scene**. This
   generates `Assets/Megamind/Scenes/Megamind.unity` containing
   `OVRCameraRig` (+ `OVRManager` with passthrough enabled, `OVRPassthroughLayer`),
   a `PassthroughCameraAccess` component (left camera, 1280×960), and a `Megamind`
   object with `MegamindPermissions`, `FrameUploader`, `CaptureHud`, `AmbientAudioRecorder`, and sets it as the
   only scene in Build Settings. It also creates `Assets/Megamind/MegamindConfig.asset`
   (server URL / device id) and `Assets/Megamind/CaptureSettings.asset` (fps, resolution, JPEG).
   Run **Megamind → Check Project** — every line should be `OK`.
6. **Set the server URL**: select `Assets/Megamind/MegamindConfig.asset` (menu
   *Megamind → Select Config*) and set `Server Base Url` to the laptop's Tailscale IP,
   e.g. `http://100.101.102.103:8000` (see *Networking*). `Device Id` can stay `quest3-1`.
7. **Apply Android player settings**: **Megamind → 2. Apply Android Player Settings**
   (ARM64, IL2CPP, ASTC, min SDK 32, Internet permission required, insecure HTTP allowed,
   package `com.megamind.quest`). Idempotent; safe to re-run.
8. **Connect the headset**: USB-C, put it on, accept *Allow USB debugging* (tick *Always
   allow*). Verify from a terminal:
   ```sh
   adb devices            # must list the headset as "device", not "unauthorized"
   ```
9. **Build & install**: **Megamind → 4. Build and Run on headset** (builds
   `Builds/megamind.apk` and installs + launches it), or **Megamind → 3. Build APK** then
   ```sh
   adb install -r Builds/megamind.apk
   adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity
   ```
   First IL2CPP build is ~10 min; incremental builds ~2 min.
10. **On the headset**, the app appears under *Library → Unknown Sources → Megamind*.
    On first launch it shows **two permission prompts**, one after the other: **Camera** (the `HEADSET_CAMERA`
    permission, for frames — allow) and **Microphone** (`RECORD_AUDIO`, for push-to-talk *and* ambient audio
    memory — allow) (verify on device: order/wording). The capture HUD then shows `camera starting…` →
    `capturing 1280x960 → 960px JPEG` and `sent N / accepted M (reason)` updating once per second, plus
    `mic: chunks sent N  accepted M  (on -NN dB)` once per 15 s audio chunk (verify on device).
11. **Watch logs** from the laptop:
    ```sh
    adb logcat -s Unity:I | grep -i megamind
    ```
    Expected lines: `[Megamind] permissions: camera=True`, `[Megamind] capture: server=…`,
    `[Megamind] camera playing: 1280x960 maxFps=30`, `[Megamind] ambient audio: mic started 16000 Hz x1, 15 s chunks`,
    and on the server `POST /v1/frames … 200` / `POST /v1/audio … 200`.

### Headless build (CI / a box without a display)
Each `Megamind` menu item is a static method, so the whole flow runs from a shell (`U=~/Unity/Hub/Editor/6000.0.66f2/Editor/Unity`):
```sh
sh headset/tools/patch-metaxr-linux.sh   # Linux only
for m in MegamindSceneBuilder.CreateScene MegamindChatPanelBuilder.Build \
         MegamindSceneBuilder.ApplyAndroidSettings MegamindSceneBuilder.CheckProject MegamindSceneBuilder.BuildApk; do
  $U -batchmode -nographics -quit -buildTarget Android -projectPath headset/Megamind \
     -executeMethod Megamind.EditorTools.$m -logFile /tmp/$m.log || { grep -h "error\|\[Megamind\]" /tmp/$m.log; break; }
done
```
`Builds/megamind.apk` is the output. The first run also downloads/imports the Meta packages (several minutes).

### Pointing the app at a different server without rebuilding
The URL is resolved in this order: PlayerPrefs (`megamind.serverBaseUrl`, written by the
settings panel) → `megamind.json` pushed over adb → `MegamindConfig.asset`. From the editor use
**Megamind → Push config to headset (adb)**, or by hand:
```sh
adb shell mkdir -p /sdcard/Android/data/com.megamind.quest/files
adb shell 'echo "{\"serverBaseUrl\":\"http://100.101.102.103:8000\",\"deviceId\":\"quest3-1\"}" > /sdcard/Android/data/com.megamind.quest/files/megamind.json'
adb shell am force-stop com.megamind.quest && adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity
```
Optional keys: `captureFps` (default 1).

## Networking (Tailscale)
1. Install Tailscale on the laptop running the server and on the Quest (Tailscale is
   on the Horizon store; or sideload the Android APK from tailscale.com). Log both into
   the same tailnet.
2. On the laptop: `tailscale ip -4` → e.g. `100.101.102.103`. Start the server bound
   to all interfaces: `cd server && uvicorn megamind.main:app --host 0.0.0.0 --port 8000`.
3. From the headset's browser (or `adb shell curl http://100.101.102.103:8000/v1/health`)
   confirm the health JSON loads. Use that URL as `serverBaseUrl` (plain HTTP — the app
   manifest allows cleartext traffic).
4. Fallback without Tailscale: same Wi-Fi + laptop LAN IP, or `ngrok http 8000` and use
   the HTTPS URL.

## Capture pipeline (stream 1)
`FrameUploader` waits for the camera permission (`MegamindPermissions`), then for
`PassthroughCameraAccess.IsPlaying`, then every `1/captureFps` s:
1. (settings from `CaptureSettings.asset`) skips the frame if `maxPendingUploads` uploads are still in flight (slow network),
   while backing off after upload failures (network error / 5xx / 429: `backoffBaseSeconds` 1 s doubling to
   `backoffMaxSeconds` 30 s, ±20% jitter, reset by the next success; no offline queue — stale frames are
   worthless to a live memory) or, if `maxHeadDegPerSec` > 0, when the head is turning faster than that (off by
   default — the server's blur filter handles motion blur);
2. blits `cameraAccess.GetTexture()` into a ≤ `jpegMaxWidth` (960 px) texture (GPU resize +
   `ReadPixels`), then JPEG-encodes it on the *next* frame (`jpegQuality` 75 ⇒ ~60–120 KB) so
   readback and encode never hitch the same render frame;
3. POSTs multipart `image` + `meta` JSON to `/v1/frames` per `docs/api.md`: `ts` is
   `PassthroughCameraAccess.Timestamp` (image time, falls back to wall clock) and `pose`
   x/y/z/yaw is `GetCameraPose()` (RGB camera pose at that time, falls back to
   `CenterEyeAnchor`). The camera stream itself is capped at `cameraMaxFramerate` = 30 fps
   (we sample 1 fps; saves GPU/battery vs the API default 60);
4. updates counters shown by `CaptureHud` (state, resolution, sent/accepted, last server
   reason, latency, JPEG size, skipped-motion/backlog) — a visible recording indicator is
   always on screen while capturing. **A / X button pauses and resumes capture.** While paused, and while
   the app is backgrounded or unfocused (headset off, system menu open), nothing is captured and the camera
   stream is stopped (`stopCameraWhenPaused`); it restarts on resume.

## Ambient audio memory (stream 1)
So *"what did I hear?"* / *"who did I talk to?"* work from the Unity app (until now only the native
`headset-android/` app uploaded audio). `AmbientAudioRecorder` (on the `Megamind` object) records the default
microphone with Unity's `Microphone` API and uploads it to `POST /v1/audio` (`docs/api.md`) with the **same server
URL and device id as frames** (`CaptureSettings.ServerBaseUrl/DeviceId`: settings panel → `megamind.json` →
`MegamindConfig.asset`):
- **Format** = the native app's, so the server path is identical: 16 kHz mono PCM16 little-endian in a 44-byte
  RIFF/WAVE header, **15 s chunks** (`audioChunkSeconds`; docs/api.md recommends ~15–30 s — fewer STT calls, more context per transcript; the native app defaults to 5 s), `meta` = `{"device_id":…,"ts_start":…}` (+ head
  `pose` when available), file name `yyyyMMdd_HHmmss_SSS.wav` (`AudioChunkCodec.cs`). `ts_start` is the Unix wall
  clock of the chunk's first sample, the clock `FrameUploader` stamps frames with, re-derived from
  `Microphone.GetPosition` every 100 ms so it does not drift.
- **Silence gate:** a chunk whose loudest 100 ms window is below `audioSilenceDb` (−50 dBFS) is dropped on the
  headset (no Whisper cost; Horizon also feeds digital silence while the headset is off-head). HUD `silent N`.
- **Uploads:** one at a time, oldest first (consecutive chunks merge into one speech token on the server); up to
  `audioMaxQueuedChunks` (8 × 15 s = 2 min) kept in memory while the server is unreachable; network error / 429 / 5xx
  back off like frames (`backoffBaseSeconds` doubling to `backoffMaxSeconds`, ±20% jitter); a chunk is dropped
  after `audioMaxAttempts` 429/5xx answers or any other 4xx.
- **Pauses** (partial chunk ≥ 1 s is flushed, the mic is released): A/X capture pause (one button stops all
  recording), app backgrounded/unfocused (same `OnApplicationPause`/`OnApplicationFocus` hooks as
  `FrameUploader`), and **while push-to-talk records**: `ChatClient` calls `AmbientAudioRecorder.PushToTalkStarting()`
  just before its `Microphone.Start` and `PushToTalkEnded()` after `Microphone.End`, so the two never fight over
  the device. Push-to-talk questions therefore are not also stored as ambient speech.
- **Toggle:** `CaptureSettings.ambientAudio` (default **on**) in `Assets/Megamind/CaptureSettings.asset`, or without
  rebuilding: add `"ambientAudio":"off"` (or `"on"`) to `megamind.json` and restart the app. Off = the mic is
  only used for push-to-talk.
- **HUD:** `mic: chunks sent N  accepted M  (state)` where state is `on -NN dB` (last chunk level), `paused`,
  `push-to-talk`, `waiting for mic permission`, `no microphone` or `off`; plus `silent N` / `queued N` / `ERR …`.

Test without a headset: `server/tests/test_quest_unity_audio.py` replays the exact bytes (mock STT). On the
headset (verify on device): start the server with mock providers, launch, talk near the headset for ~30 s (a chunk is stored ~15 s after it starts), watch
the HUD `mic:` counters and `POST /v1/audio 200` in the server log, then hold the trigger and ask *"what did I
hear?"* — the answer cites `Speech heard nearby: "[mock transcript]"` (real words with `AUDIO_STT_PROVIDER=openai`
or `faster-whisper` on the server, see `server/.env.example`). `curl 'http://<server>:8000/v1/memories/recent?kind=transcript&k=5'`
lists the stored speech tokens.

Permissions: `Assets/Plugins/Android/AndroidManifest.xml` declares
`horizonos.permission.HEADSET_CAMERA`, `android.permission.CAMERA`,
`android.permission.RECORD_AUDIO`, `android.permission.INTERNET`, and
`android:usesCleartextTraffic="true"` (research digest #0). There is exactly **one** runtime
permission flow: `MegamindPermissions` calls `OVRPermissionsRequester.Request` once (camera +
mic), the same mechanism as Meta's sample `RequestPermissionsOnce`; Android shows that as **two prompts**, camera
then microphone. `PassthroughCameraAccess` only checks `HEADSET_CAMERA`, so `android.permission.CAMERA` is never
requested at runtime and only one *camera* dialog appears. The scene builder clears OVRManager's *Permission Requests On
Startup* so nothing else asks, and *Meta > Tools > Update AndroidManifest.xml* must not be
allowed to remove `HEADSET_CAMERA` (it regenerates the file without it).

## Scripts (`Assets/Megamind/Scripts`)
| File | Stream | Purpose |
|---|---|---|
| `MegamindConfig.cs` | 3 | ScriptableObject with server URL / device id (stream 3 owns it) |
| `CaptureSettings.cs` | 1 | ScriptableObject: fps, camera resolution, JPEG width/quality, backpressure, `ambientAudio` toggle + chunk/gate/queue; `megamind.json` adb override for the server URL (and `ambientAudio`) |
| `Dto.cs` | shared | C# mirrors of `docs/api.md` payloads |
| `MegamindPermissions.cs` | 1 | requests `HEADSET_CAMERA` + mic at start, exposes `CameraGranted` / `MicrophoneGranted` |
| `FrameUploader.cs` | 1 | 1 fps sample of `PassthroughCameraAccess.GetTexture()` → JPEG → `POST /v1/frames` with pose meta |
| `AmbientAudioRecorder.cs` | 1 | ambient mic → 15 s 16 kHz mono PCM16 WAV chunks → `POST /v1/audio`; silence gate, backoff, pauses for push-to-talk / A/X / background |
| `AudioChunkCodec.cs` | 1 | pure C# WAV + meta encoding (native-app layout) and the RMS level for the silence gate |
| `CaptureHud.cs` | 1 | world-locked capture status HUD (+ `mic:` line) + recording indicator, A/X pause |
| `ChatClient.cs` | 3 | push-to-talk (trigger or index pinch) → 16 kHz WAV → `POST /v1/chat` → play audio; barge-in, latency; tells `AmbientAudioRecorder` to release the mic while recording |
| `ChatPanel.cs` | 3 | transcript bubbles, cited thumbnails, confidence/latency line, status HUD (`/v1/health` poll) |
| `PanelAnchor.cs` | 3 | places the canvas `panelDistance` m ahead on first use; hold B/Y 0.5 s to re-anchor |
| `SettingsPanel.cs` | 3 | runtime server URL / device id, new session, purge memory |
| `Editor/MegamindSceneBuilder.cs` | 1 | the `Megamind` menu: create scene, apply player settings, build, build & run, push config, check project |
| `Editor/MegamindChatPanelBuilder.cs` | 3 | `Megamind > 1b. Add Chat Panel To Current Scene`: bubble/thumb prefabs + the world-locked chat panel, all references wired |

## Stream 3 chat panel

After **Megamind > 1. Create Capture Scene**, run **Megamind > 1b. Add Chat Panel To Current Scene** before
building. It creates `Assets/Megamind/Prefabs/{UserBubble,AssistantBubble,ThumbRow,Thumb}.prefab` and this
hierarchy (one World Space Canvas, 600x720 px, scaled by `PanelAnchor` to `panelWidthMeters`), wires every
field on `ChatPanel`/`ChatClient`/`PanelAnchor`/`SettingsPanel`, finds `CenterEyeAnchor`, `OVRHand`s and the
`FrameUploader`, and adds an `EventSystem` with Meta's `OVRInputModule` + `OVRRaycaster` (right-controller
ray) when the SDK provides them:
```
MegamindPanel (Canvas)  + PanelAnchor + ChatPanel + ChatClient + SettingsPanel + AudioSource
  Background, Header (StatusLine, ServerDot, Settings button)
  Scroll -> Viewport -> Content (VerticalLayoutGroup)        = ChatPanel.content
  EmptyHint, Indicators (ListeningIndicator, ThinkingIndicator)
  Settings (inactive): ServerUrl, DeviceId, Save / New session / Purge, Reset / Close, Feedback
```
Re-running the menu item replaces a previous `MegamindPanel`; tweak the result in the Inspector and save.
Push-to-talk: hold the controller trigger (or index pinch); hold B/Y 0.5 s to drag the panel to your gaze.

A browser prototype of the same panel (same API calls, same states) is served by the server at
`http://<server>:8000/ui` — use it to agree on layout/behaviour and as a hardware-free demo fallback.

## Quest test checklist (first human run — please report back)

Nothing in this project has run on a headset yet. Hour-0, in this order:

1. Build Meta's `CameraViewer` sample **unchanged** ([Unity-PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples), Unity 6000.0.66f2) on each headset. If it shows camera frames, the OS/dev-mode side is fine. Note the Horizon OS version (Quick controls → Settings → General → Software update).
2. Build Megamind (steps above), launch, grab logcat: `adb logcat -s Unity | grep Megamind`. Paste the lines
   `[Megamind] camera: os=... supported=[...]` (resolutions the OS offers) and
   `[Megamind] camera playing: 1280x960 maxFps=30`.
3. Verify and report:
   - **Permission:** two prompts on first launch — one camera dialog, then one microphone dialog; HUD leaves `waiting for camera permission` / `mic: … (waiting for mic permission)` after Allow. (If two *camera* dialogs appear, `android.permission.CAMERA` in the manifest is the suspect — delete that line and tell us.) (verify on device)
   - **Camera:** HUD shows `1280x960`, frames ~1/s; if `supported=[...]` lacks 1280x960, pick one from the list in `CaptureSettings.cameraResolution`.
   - **Upload:** HUD `sent/accepted` climbs and `server:ok` / `server:duplicate` reasons appear; server log shows `POST /v1/frames 200`.
   - **Pose/ts:** server-side `meta.pose.yaw` changes when you turn your head; `ts` is within ~1 s of the server clock.
   - **Perf:** no visible hitch each second (readback and JPEG encode run on consecutive frames); note `enc` ms on the HUD.
   - **Image:** open a stored thumbnail: colours correct (skin not blue — reboot if so), upright, not mirrored.
   - **Ambient audio (verify on device):** HUD `mic: chunks sent/accepted` climbs every ~15 s while people talk and `silent N` climbs in a quiet room; server log `POST /v1/audio 200`; the `mic:` state shows `push-to-talk` while you hold the trigger and goes back to `on` after the reply; A/X shows `paused`; push-to-talk replies still work. Then ask *"what did I hear?"*.
4. Demo day: reboot the headset in the morning (RGB-flip bug), disable auto-sleep (Settings → General → Power), keep the HUD visible as a self-test.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Permission dialog appears twice / never granted | Two mechanisms are requesting it. Only `MegamindPermissions` should request; make sure OVRManager's *Permission Requests On Startup* list is empty (re-run *Megamind > 1. Create Capture Scene*) and the Meta sample's `RequestPermissionsOnce`/`PassthroughCameraPermissions` is not in the project. |
| Colours wrong (skin looks blue) | Known v74 issue after first install of a camera app: reboot the headset. |
| App won't launch, `SensorLockActivity` in logcat | Headset was idle/doffed: put it on, press the power button once, relaunch. |
| Build regenerated the manifest and camera stopped working | *Meta > Tools > Update AndroidManifest.xml* drops `HEADSET_CAMERA`; re-add it (or `git checkout` the manifest). |
| Hub says the editor version is missing | Install exactly 6000.0.66f2, or open with another 6000.0.x (≥ 38f1) and accept the upgrade; do not use 2022.3 (MRUK 85 needs Unity 6). |
| Package Manager errors resolving `com.meta.xr.*` | Check internet / proxy; Window → Package Manager → refresh. Meta packages come from the Unity registry, no scoped registry needed. |
| Compile errors mentioning `PassthroughCameraAccess` | MRUK must be 81+ (we pin 85). Project Settings → Meta XR → make sure "Passthrough Camera Access" is enabled (it is in `OculusProjectConfig.asset`). |
| HUD `mic: … (waiting for mic permission)` | The microphone prompt was denied. Settings → Apps → Megamind → Permissions → Microphone → Allow, or `adb shell pm grant com.megamind.quest android.permission.RECORD_AUDIO`; push-to-talk needs it too. |
| HUD `mic:` only climbs `silent N`, never `sent` | Every chunk is below `CaptureSettings.audioSilenceDb` (−50 dBFS): mic muted (headset off-head, system mute) or a very quiet room. Lower it (e.g. −60) or set −120 to upload everything. |
| HUD `mic: … ERR 0 …` while frames upload fine | `/v1/audio` transcribes synchronously; a slow STT provider can exceed `audioUploadTimeoutSeconds` (60 s). Check the server log for the `/v1/audio` request. |
| HUD stuck at `waiting for camera permission` | The permission dialog was dismissed. Settings → Apps → Megamind → Permissions → Camera → Allow, or `adb shell pm grant com.megamind.quest horizonos.permission.HEADSET_CAMERA` (and `android.permission.CAMERA`). |
| HUD `camera starting…` forever / `camera not supported` | Headset must be Quest 3/3S on Horizon OS v74+ (Quick controls → Settings → General → Software update). Passthrough Camera API does **not** work in the XR Simulator or on Quest 2/Pro; only on device (or Horizon Link 2.1+). |
| Black image / 0×0 resolution in HUD | Passthrough disabled: `OVRManager.isInsightPassthroughEnabled` must be on (scene builder sets it); re-run *Megamind → Check Project*. |
| `sent N` grows but `accepted 0`, reason `net:…` | Server unreachable: check `adb shell curl <url>/v1/health`, Tailscale connected on both ends, server bound to `0.0.0.0`, URL has no trailing path. |
| Reason `http:422` | `meta` JSON rejected — `Dto.cs` and `server/megamind/models.py` out of sync; check the server log. |
| Most frames `too_soon` / `duplicate` | Expected when the scene is static — the server filter keeps ~10–30 % at 1 fps. Walk around. |
| Many `blurry` / `dark` | Raise room light; the camera auto-exposes slowly after passthrough starts. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` on `adb install` | A build with a different signing key is installed: `adb uninstall com.megamind.quest` then reinstall. |
| Linux editor: `MetaXRSimulator/Installer.cs(88): error CS0103 'downloadedInstallerPath'` | Meta XR Core 85 bug (variable only declared for Windows/macOS). Run `sh headset/tools/patch-metaxr-linux.sh` (patches `Library/PackageCache`; re-run after a package re-resolve). |
| `adb devices` shows `unauthorized` / nothing | Put the headset on and accept the USB debugging dialog; enable developer mode in the Meta Horizon phone app; try another cable/port. |
| Build fails with Gradle/JDK error | Use the Hub-installed OpenJDK/SDK/NDK (Edit → Preferences → External Tools: all "Installed with Unity" ticked). |
| App restarts when the headset is taken off | Normal: Horizon OS suspends background apps. Keep the headset on or set proximity sensor off in developer settings while testing. |
