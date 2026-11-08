# Megamind headset app (Unity, Quest 3)

Single Unity app that (1) captures passthrough keyframes and uploads them to the
server, and (2) hosts the voice-first AR chat panel. Both streams share this project
(design doc §2, decision D1). The project in `headset/Megamind/` is a complete Unity
project: open it in Unity Hub, let the packages resolve, run one menu item, build.

> Status: the project has been assembled and compile-checked against the Meta SDK
> 85 / MRUK 85 API surface, but **not yet built or run on a headset**. Expect to
> iterate on device; the troubleshooting list below covers the likely failures.

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
   object with `MegamindPermissions`, `FrameUploader`, `CaptureHud`, and sets it as the
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
    On first launch it asks for **Camera** (this is the `HEADSET_CAMERA` permission —
    allow) and **Microphone**. The capture HUD then shows `camera starting…` →
    `capturing 1280x960 → 960px JPEG` and `sent N / accepted M (reason)` updating once per second.
11. **Watch logs** from the laptop:
    ```sh
    adb logcat -s Unity:I | grep -i megamind
    ```
    Expected lines: `[Megamind] permissions: camera=True`, `[Megamind] capture: server=…`,
    `[Megamind] camera playing: 1280x960 maxFps=30`, and on the server `POST /v1/frames … 200`.

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
1. (settings from `CaptureSettings.asset`) skips the frame if `maxPendingUploads` uploads are still in flight (slow network) or,
   if `maxHeadDegPerSec` > 0, when the head is turning faster than that (off by default —
   the server's blur filter handles motion blur);
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
   always on screen while capturing. **A / X button pauses and resumes capture.**

Permissions: `Assets/Plugins/Android/AndroidManifest.xml` declares
`horizonos.permission.HEADSET_CAMERA`, `android.permission.CAMERA`,
`android.permission.RECORD_AUDIO`, `android.permission.INTERNET`, and
`android:usesCleartextTraffic="true"` (research digest #0). There is exactly **one** runtime
permission flow: `MegamindPermissions` calls `OVRPermissionsRequester.Request` once (camera +
mic), the same mechanism as Meta's sample `RequestPermissionsOnce`. `PassthroughCameraAccess`
only checks `HEADSET_CAMERA`, so `android.permission.CAMERA` is never requested at runtime and
only one dialog appears. The scene builder clears OVRManager's *Permission Requests On
Startup* so nothing else asks, and *Meta > Tools > Update AndroidManifest.xml* must not be
allowed to remove `HEADSET_CAMERA` (it regenerates the file without it).

## Scripts (`Assets/Megamind/Scripts`)
| File | Stream | Purpose |
|---|---|---|
| `MegamindConfig.cs` | 3 | ScriptableObject with server URL / device id (stream 3 owns it) |
| `CaptureSettings.cs` | 1 | ScriptableObject: fps, camera resolution, JPEG width/quality, backpressure; `megamind.json` adb override for the server URL |
| `Dto.cs` | shared | C# mirrors of `docs/api.md` payloads |
| `MegamindPermissions.cs` | 1 | requests `HEADSET_CAMERA` + mic at start, exposes `CameraGranted` |
| `FrameUploader.cs` | 1 | 1 fps sample of `PassthroughCameraAccess.GetTexture()` → JPEG → `POST /v1/frames` with pose meta |
| `CaptureHud.cs` | 1 | world-locked capture status HUD + recording indicator, A/X pause |
| `ChatClient.cs` | 3 | mic record on push-to-talk → `POST /v1/chat` → play audio |
| `ChatPanel.cs` | 3 | transcript bubbles + cited thumbnails on the world-locked canvas |
| `Editor/MegamindSceneBuilder.cs` | 1 | the `Megamind` menu: create scene, apply player settings, build, build & run, push config, check project |

## Quest test checklist (first human run — please report back)

Nothing in this project has run on a headset yet. Hour-0, in this order:

1. Build Meta's `CameraViewer` sample **unchanged** ([Unity-PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples), Unity 6000.0.66f2) on each headset. If it shows camera frames, the OS/dev-mode side is fine. Note the Horizon OS version (Quick controls → Settings → General → Software update).
2. Build Megamind (steps above), launch, grab logcat: `adb logcat -s Unity | grep Megamind`. Paste the lines
   `[Megamind] camera: os=... supported=[...]` (resolutions the OS offers) and
   `[Megamind] camera playing: 1280x960 maxFps=30`.
3. Verify and report:
   - **Permission:** exactly one camera dialog on first launch; HUD leaves `waiting for camera permission` after Allow. (If two dialogs appear, `android.permission.CAMERA` in the manifest is the suspect — delete that line and tell us.)
   - **Camera:** HUD shows `1280x960`, frames ~1/s; if `supported=[...]` lacks 1280x960, pick one from the list in `CaptureSettings.cameraResolution`.
   - **Upload:** HUD `sent/accepted` climbs and `server:ok` / `server:duplicate` reasons appear; server log shows `POST /v1/frames 200`.
   - **Pose/ts:** server-side `meta.pose.yaw` changes when you turn your head; `ts` is within ~1 s of the server clock.
   - **Perf:** no visible hitch each second (readback and JPEG encode run on consecutive frames); note `enc` ms on the HUD.
   - **Image:** open a stored thumbnail: colours correct (skin not blue — reboot if so), upright, not mirrored.
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
| HUD stuck at `waiting for camera permission` | The permission dialog was dismissed. Settings → Apps → Megamind → Permissions → Camera → Allow, or `adb shell pm grant com.megamind.quest horizonos.permission.HEADSET_CAMERA` (and `android.permission.CAMERA`). |
| HUD `camera starting…` forever / `camera not supported` | Headset must be Quest 3/3S on Horizon OS v74+ (Quick controls → Settings → General → Software update). Passthrough Camera API does **not** work in the XR Simulator or on Quest 2/Pro; only on device (or Horizon Link 2.1+). |
| Black image / 0×0 resolution in HUD | Passthrough disabled: `OVRManager.isInsightPassthroughEnabled` must be on (scene builder sets it); re-run *Megamind → Check Project*. |
| `sent N` grows but `accepted 0`, reason `net:…` | Server unreachable: check `adb shell curl <url>/v1/health`, Tailscale connected on both ends, server bound to `0.0.0.0`, URL has no trailing path. |
| Reason `http:422` | `meta` JSON rejected — `Dto.cs` and `server/megamind/models.py` out of sync; check the server log. |
| Most frames `too_soon` / `duplicate` | Expected when the scene is static — the server filter keeps ~10–30 % at 1 fps. Walk around. |
| Many `blurry` / `dark` | Raise room light; the camera auto-exposes slowly after passthrough starts. |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` on `adb install` | A build with a different signing key is installed: `adb uninstall com.megamind.quest` then reinstall. |
| `adb devices` shows `unauthorized` / nothing | Put the headset on and accept the USB debugging dialog; enable developer mode in the Meta Horizon phone app; try another cable/port. |
| Build fails with Gradle/JDK error | Use the Hub-installed OpenJDK/SDK/NDK (Edit → Preferences → External Tools: all "Installed with Unity" ticked). |
| App restarts when the headset is taken off | Normal: Horizon OS suspends background apps. Keep the headset on or set proximity sensor off in developer settings while testing. |
