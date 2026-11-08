# Megamind headset app (Unity, Quest 3)

Single Unity app that (1) captures passthrough keyframes and uploads them to the
server, and (2) hosts the voice-first AR chat panel. Both streams share this project
(design doc §2, decision D1). The project in `headset/Megamind/` is a complete Unity
project: open it in Unity Hub, let the packages resolve, run one menu item, build.

> Status: **playtested on a Quest 3 (Horizon OS v74 / Android 14), 2026-11-07.** Built headless
> (Unity 6000.0.66f2 + Meta SDK 85 / MRUK 85) into `Builds/megamind.apk`, sideloaded over USB and run
> against a laptop server with mock providers via `adb reverse tcp:8000 tcp:8000` +
> `megamind.json` (`{"serverBaseUrl":"http://127.0.0.1:8000"}`; on a later Quest 3 / Android 14 test the app could no
> longer read an adb-written `megamind.json`, so use the launch-intent extras below). Verified on device: camera + mic
> permission prompts, passthrough capture at 1280x960 / 1 fps uploading to `/v1/frames`, the
> world-locked panel in passthrough, B/Y re-anchor, right-trigger push-to-talk → `/v1/chat` →
> transcript bubble + reply with 3 cited thumbnails. Not yet exercised on device: hand-pinch PTT,
> the settings sub-panel buttons (controller-ray UI; scenes built before the hand/ray fix had no `OVRHand` and no
> ray, see [Hands and the UI ray](#hands-and-the-ui-ray)), TTS playback (mock TTS returns no audio),
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

## Get the APK (no Unity needed)
[`.github/workflows/headset-apk.yml`](../.github/workflows/headset-apk.yml) builds two APKs on every push to `main`
that touches `headset/**` or `headset-android/**`, and on demand (Actions → **headset-apk** → *Run workflow*):

| Artifact | App | Package / launch activity | Needs |
|---|---|---|---|
| `megamind.apk` | this Unity app (capture + AR chat panel) | `com.megamind.quest` / `com.unity3d.player.UnityPlayerGameActivity` | Unity licence secrets (below) |
| `megamind-native.apk` | [`headset-android/`](../headset-android/README.md) capture-only app | `ai.megamind.quest` / `ai.megamind.quest.MainActivity` | nothing |

**Download:** GitHub → *Actions* → *headset-apk* → newest green run on `main` → *Artifacts* (zipped; unzip to get the
`.apk`), or with the CLI: `gh run download -R debojoti-soumya/megamind -n megamind.apk` (artifacts expire after 90 days).
**Releases:** pushing a tag `headset-vX.Y` (e.g. `git tag headset-v0.1 && git push origin headset-v0.1`) also attaches
both APKs to a [GitHub Release](https://github.com/debojoti-soumya/megamind/releases); if the Unity job fails for that
tag the release gets only `megamind-native.apk` and says so in its notes.
Install with `adb install -r megamind.apk` ([Quest 3 guide §4](../docs/quest3-guide.md#4-getting-megamind-on-the-headset)).

Notes on the prebuilt Unity APK:
- It is built from the **committed** scene/assets/player settings (`Assets/Megamind/Scenes/Megamind.unity`,
  `ProjectSettings/ProjectSettings.asset`: IL2CPP, ARM64, min/target SDK 32, ASTC), via the same entry point as
  *Megamind → 3. Build APK*. If you change the scene via the menu items, commit the result or CI will not see it.
- Its baked-in server URL is whatever `MegamindConfig.asset` holds (`http://100.64.0.1:8000`); point it at your server
  without rebuilding with launch-intent extras (`adb shell am start ... --es serverBaseUrl ...`) or the settings panel (see *Pointing the app at a different server* below).
- CI signs with a throwaway debug key, so installing it over a build from another machine/run fails with
  `INSTALL_FAILED_UPDATE_INCOMPATIBLE`: `adb uninstall com.megamind.quest` first (this clears the app's PlayerPrefs).
- Nobody has installed a CI-built APK on a headset yet **(verify on device)**.

### One-time setup: Unity licence secrets (repo admin)
The Unity job uses [game-ci/unity-builder](https://game.ci/docs/github/builder) (Docker image
`unityci/editor:ubuntu-6000.0.66f2-android-3`), which must activate a Unity licence. Without these secrets the Unity
job's first step reports `Repository secret … is not set`: on a push to `main` as a **warning** (the build is skipped
and the job stays green, so `main` is not red), on *Run workflow* / `headset-v*` tags as an **error** (job fails).
Until the secrets exist, **the native job (`megamind-native.apk`, also run on PRs touching `headset-android/`) is the
only path guaranteed to go green and produce an APK**; a green Unity job on `main` does not mean `megamind.apk` was
built, so check the run's *Artifacts* (or its warning). Add them under
GitHub → repo → *Settings* → *Secrets and variables* → *Actions* → *New repository secret*
([game-ci activation docs](https://game.ci/docs/github/activation)):

| Licence | Secrets |
|---|---|
| **Personal (free)** | `UNITY_LICENSE` = full contents of the `.ulf` file, `UNITY_EMAIL`, `UNITY_PASSWORD` (the Unity account that owns the licence) |
| **Plus / Pro** | `UNITY_SERIAL` (`XX-XXXX-XXXX-XXXX-XXXX-XXXX`, from [id.unity.com → Subscriptions](https://id.unity.com/en/subscriptions)), `UNITY_EMAIL`, `UNITY_PASSWORD` |

Getting the Personal `.ulf` (needs **Unity Hub only**, not the Editor; any OS works, the licence is not tied to a
platform or Unity version):
1. Install [Unity Hub](https://unity.com/download) and sign in with the account you will put in `UNITY_EMAIL`.
2. Hub → *Preferences* (Settings) → *Licenses* → **Add** → **Get a free personal license**. Click *Add* even if a
   licence is already listed, otherwise the `.ulf` file may not be written.
3. Copy the file into the `UNITY_LICENSE` secret:
   - Windows `C:\ProgramData\Unity\Unity_lic.ulf`
   - macOS `/Library/Application Support/Unity/Unity_lic.ulf`
   - Linux `~/.local/share/unity3d/Unity/Unity_lic.ulf`
   e.g. `gh secret set UNITY_LICENSE -R debojoti-soumya/megamind < "/Library/Application Support/Unity/Unity_lic.ulf"`.
   If Hub offers no way to produce a `.ulf`, see the game-ci activation docs (they track Hub UI changes).

Then re-run the workflow (Actions → headset-apk → *Run workflow*). The first run downloads the ~7 GB editor image and
imports the Meta packages (rough estimate 30–60 min, not measured); the `Library/` cache makes later runs faster.
How CI gets around the Linux editor's Meta XR Core 85 compile bug: it embeds the registry copy of
`com.meta.xr.sdk.core` (version from `Packages/manifest.json`, checksum-verified) in `Packages/` and runs
`headset/tools/patch-metaxr-linux.sh` on it before Unity starts; nothing is committed.

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
CI does the same with only `BuildApk` (the scene and player settings are committed); see *Get the APK* above.

### Pointing the app at a different server without rebuilding
Each value (`serverBaseUrl`, `deviceId`, `apiToken`, `ambientAudio`) is resolved in this order (`RuntimeConfig.cs`):
1. **Launch-intent extras** on `adb shell am start`. The app saves them to PlayerPrefs, so later launches from the
   Library keep them until you pass new ones.
2. **PlayerPrefs saved earlier**: by a previous intent launch, or by the settings panel (URL and device id only).
3. **`megamind.json`** in `/sdcard/Android/data/com.megamind.quest/files/`, *if the app can read it* (see below).
4. **`MegamindConfig.asset`** (URL, device id, `Api Token`) / `CaptureSettings.asset` (`ambientAudio`), baked into the APK.

```sh
adb shell am force-stop com.megamind.quest
adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity \
  --es serverBaseUrl http://100.101.102.103:8000 --es deviceId quest3-1 \
  --es apiToken <API_TOKEN>      # only for a server with API_TOKEN set (Modal); omit otherwise
```
Add `--es ambientAudio off` (or `on`) to toggle ambient audio. `--es apiToken "''"` (an empty value) removes a
saved key; `--ez resetConfig true` removes every saved value first. The editor menu **Megamind → Push config to
headset (adb)** runs the same two commands with the values from `MegamindConfig.asset`. **Force-stop first**: the
activity is `singleTask`, so `am start` on a running app only brings it to the front and the extras are not read
(verify on device). `adb uninstall` or `adb shell pm clear com.megamind.quest` also wipes the saved values.

Check it worked: `adb logcat -s Unity | grep "\[Megamind\] config"` prints each value with its source, e.g.
`serverBaseUrl=http://100.101.102.103:8000 (intent) deviceId=quest3-1 (intent) apiToken=(set, 32 chars) (intent)
ambientAudio=on (asset)` (the token is never logged), and the HUD's last line ends `url:intent` (`saved` on later
launches, `panel`, `megamind.json` or `asset` otherwise) (verify on device).

**Why not `megamind.json`?** On a Quest 3 on Horizon OS / Android 14 (2026-11-08), a `megamind.json` written by
`adb shell 'echo ... > .../files/megamind.json'` was not readable by the app: logcat showed
`could not read megamind.json: Access ... is denied`, even after deleting the folder and letting the app recreate it.
`adb push` writes the file as the same shell user, so expect the same result (not tested). No other adb route is documented for
this build: `adb shell run-as com.megamind.quest` only works for debuggable apps, and `BuildApk` makes a
non-debuggable release build (`BuildOptions.None`). Shared folders such as `/sdcard/Download` don't help either:
under scoped storage (this APK targets API 32) an app can't open another app's non-media files there without the
Storage Access Framework or all-files access. The app
still reads `megamind.json` when it can (e.g. in the Editor, or a file the app wrote itself) and accepts
`serverBaseUrl`, `deviceId`, `apiToken`, `ambientAudio` and `captureFps` keys.

**Fallback (no adb at all):** set `Server Base Url`, `Device Id` and `Api Token` in `MegamindConfig.asset`
(*Megamind → Select Config*) and rebuild. This is what worked on the Android 14 headset before intent extras were
added. Don't commit a real token.

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
   the app is backgrounded (`OnApplicationPause(true)`: headset off, app switched away), nothing is captured and
   the camera stream is stopped (`stopCameraWhenPaused`); it restarts on resume. Losing focus only
   (`OnApplicationFocus(false)` / `OVRManager.InputFocusLost`: system menu, boundary warning, Meta-button dialog)
   does **not** pause capture unless `CaptureSettings.pauseOnFocusLoss` is on (default off). See
   [Boundary (Guardian) and walking around](#boundary-guardian-and-walking-around).

## Boundary (Guardian) and walking around
The demo walks between rooms with capture running. Field report: crossing the boundary pulled the app out of the
foreground and the log showed `app backgrounded: pausing capture`. Two changes:

- **Boundaryless app.** `Assets/Plugins/Android/AndroidManifest.xml` declares
  `<uses-feature android:name="com.oculus.feature.BOUNDARYLESS_APP" android:required="true"/>`, Meta's switch that
  disables the boundary for the whole app ([Boundaryless Mode for Mixed Reality in Unity](https://developers.meta.com/horizon/documentation/unity/unity-boundaryless/)).
  Per that doc: it only works from a standalone APK on the headset (not over Link), the app must not have
  immersive (non-passthrough) parts, it is OS-version gated, and it is mutually exclusive with the contextual
  Boundary API (`OVRManager.shouldBoundaryVisibilityBeSuppressed`). So Megamind leaves that API off
  (`OculusProjectConfig` *Boundary Visibility Support* = None; the scene builder sets the OVRManager flag to false).
  *Megamind → Check Project* checks both. *Meta → Tools → Update AndroidManifest.xml* does not generate this
  line: re-add it after regenerating.
- **Focus loss is not background.** Capture and ambient audio pause only on `OnApplicationPause(true)`. The app is
  focus aware ([Focus awareness](https://developers.meta.com/horizon/documentation/unity/unity-focus-awareness/)),
  so system UI (menu, boundary dialog) overlays it while it keeps running; capture continues and resumes on its
  own when either event ends. Set `CaptureSettings.pauseOnFocusLoss` to restore the old pause-on-focus-loss.
  Logcat shows which one happened: `app backgrounded (OnApplicationPause): pausing capture`,
  `app focus lost (OnApplicationFocus: …): capture keeps running`, `OVR input focus lost (system UI overlay)`.
  HUD: `capture PAUSED (app backgrounded)` vs `(press A/X or middle-pinch)`, `(unfocused)` after `REC` while an overlay has focus,
  and `bg N unfocus M` counters on line 2 once either happened.

**Verify on device** (not run on a Quest yet): walk out of the boundary and into another room; the boundary
should not appear and the HUD should stay `REC` with `bg 0`. If `bg` still rises, the OS really paused the app.
**Fallback:** MVDH can disable the boundary system for testing ([MVDH](https://developers.meta.com/vr/documentation/unity/ts-mqdh/));
the headset's developer settings may also have a boundary toggle (verify on device). Or draw one large roomscale
boundary covering the walk. Meta's [boundaryless safety best practices](https://developers.meta.com/horizon/design/boundaryless-best-practices/)
apply: keep head-locked content out of the lower and central field of view, and don't stage world-locked content on
the Stage tracking space (it is ill-defined without a boundary).

## Ambient audio memory (stream 1)
So *"what did I hear?"* / *"who did I talk to?"* work from the Unity app (until now only the native
`headset-android/` app uploaded audio). `AmbientAudioRecorder` (on the `Megamind` object) records the default
microphone with Unity's `Microphone` API and uploads it to `POST /v1/audio` (`docs/api.md`) with the **same server
URL and device id as frames** (`CaptureSettings.ServerBaseUrl/DeviceId`: intent extras → saved PlayerPrefs / settings
panel → `megamind.json` → `MegamindConfig.asset`, see *Pointing the app at a different server* above):
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
  recording), app backgrounded (same `OnApplicationPause` hook as `FrameUploader`; focus loss pauses only
  with `pauseOnFocusLoss`), and **while push-to-talk records**: `ChatClient` calls `AmbientAudioRecorder.PushToTalkStarting()`
  just before its `Microphone.Start` and `PushToTalkEnded()` after `Microphone.End`, so the two never fight over
  the device. Push-to-talk questions therefore are not also stored as ambient speech.
- **Toggle:** `CaptureSettings.ambientAudio` (default **on**) in `Assets/Megamind/CaptureSettings.asset`, or without
  rebuilding: relaunch with `--es ambientAudio off` (or `on`), see *Pointing the app at a different server* above. Off = the mic is
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
| `CaptureSettings.cs` | 1 | ScriptableObject: fps, camera resolution, JPEG width/quality, backpressure, `ambientAudio` toggle + chunk/gate/queue; resolves the server URL / device id / `apiToken` ([api.md#auth](../docs/api.md#auth)) / `ambientAudio` overrides via `RuntimeConfig` |
| `RuntimeConfig.cs` | 1 | runtime overrides without a rebuild: launch-intent extras (`am start --es ...`, saved to PlayerPrefs) > saved PlayerPrefs > `megamind.json` > assets; logs each value's source |
| `Dto.cs` | shared | C# mirrors of `docs/api.md` payloads |
| `MegamindPermissions.cs` | 1 | requests `HEADSET_CAMERA` + mic at start, exposes `CameraGranted` / `MicrophoneGranted` |
| `FrameUploader.cs` | 1 | 1 fps sample of `PassthroughCameraAccess.GetTexture()` → JPEG → `POST /v1/frames` with pose meta |
| `AmbientAudioRecorder.cs` | 1 | ambient mic → 15 s 16 kHz mono PCM16 WAV chunks → `POST /v1/audio`; silence gate, backoff, pauses for push-to-talk / A/X / background |
| `AudioChunkCodec.cs` | 1 | pure C# WAV + meta encoding (native-app layout) and the RMS level for the silence gate |
| `CaptureHud.cs` | 1 | world-locked capture status HUD (+ `mic:` line) + recording indicator, A/X or middle-pinch pause |
| `ChatClient.cs` | 3 | push-to-talk (trigger or index pinch) → 16 kHz WAV → `POST /v1/chat` → play audio; barge-in, latency; tells `AmbientAudioRecorder` to release the mic while recording |
| `ChatPanel.cs` | 3 | transcript bubbles, cited thumbnails, confidence/latency line, status HUD (`/v1/health` poll) |
| `PanelAnchor.cs` | 3 | tag-along panel: eye height, `panelDistance` m ahead, glides back when the head leaves a 25° / 0.5 m dead zone; Pin/Unpin (world-lock), hold B/Y 0.5 s to drag; A-/A+ UI scale (PlayerPrefs `megamind.uiScale`) |
| `SettingsPanel.cs` | 3 | runtime server URL / device id, new session, purge memory |
| `Editor/MegamindSceneBuilder.cs` | 1 | the `Megamind` menu: create scene, apply player settings, build, build & run, push config, check project |
| `Editor/MegamindChatPanelBuilder.cs` | 3 | `Megamind > 1b. Add Chat Panel To Current Scene`: bubble/thumb prefabs + the chat panel, all references wired |

## Stream 3 chat panel

After **Megamind > 1. Create Capture Scene**, run **Megamind > 1b. Add Chat Panel To Current Scene** before
building. It creates `Assets/Megamind/Prefabs/{UserBubble,AssistantBubble,ThumbRow,Thumb}.prefab` and this
hierarchy (one World Space Canvas, 600x720 px, scaled by `PanelAnchor` to `panelWidthMeters` x the UI scale), wires every
field on `ChatPanel`/`ChatClient`/`PanelAnchor`/`SettingsPanel`, finds `CenterEyeAnchor`, `OVRHand`s and the
`FrameUploader`, and adds an `EventSystem` with Meta's `OVRInputModule` + `OVRRaycaster` (right-controller
ray) when the SDK provides them:
```
MegamindPanel (Canvas + CanvasScaler)  + PanelAnchor + ChatPanel + ChatClient + SettingsPanel + AudioSource
  Background, Header (ServerDot, StatusLine, A- / A+ / Pin / Settings buttons)
  Scroll -> Viewport -> Content (VerticalLayoutGroup)        = ChatPanel.content
  EmptyHint, Indicators (ListeningIndicator, ThinkingIndicator)
  Settings (inactive): ServerUrl, DeviceId, Save / New session / Purge, Reset / Close, Feedback
```
Re-running the menu item replaces a previous `MegamindPanel`; tweak the result in the Inspector and save.
Push-to-talk: hold the controller trigger (or index pinch); hold B/Y 0.5 s to drag the panel to your gaze.

### Panel placement and readability (verify on device)

`PanelAnchor` keeps the panel at **eye height** (`CenterEyeAnchor.position` + `eyeHeightOffset`, -8 cm), `panelDistance`
(1.0 m) ahead, upright. It is a lazy **tag-along** by default: it stays put while your head turns less than
`deadZoneDegrees` (25°) or moves less than `deadZoneMeters` (0.5 m) / `deadZoneHeightMeters` (0.25 m, e.g. standing up),
then glides back in front of you (`recenterLerp`, no snapping) and is pushed out again if you come closer than
`minDistance` (0.55 m). **Pin** in the header (or the `pinned` field) world-locks it where it is; **Unpin** resumes the
tag-along. Holding **B/Y** for 0.5 s drags it with your gaze in either mode. The first placement waits for a real head
pose: with the FloorLevel tracking origin `CenterEyeAnchor` reports y = 0 until tracking starts, which is what used to
leave the panel at floor level ("near my legs") on launch.

Text: the canvas has a `CanvasScaler` with `dynamicPixelsPerUnit` = 3 so uGUI glyphs are rasterised at 3x (a 1x atlas is
upscaled ~1.4x on Quest 3 and looks soft); transcript bubbles are 26 px (≈1.5° per line at 1 m), status/buttons 17–20
px, thumbnails 180x135 (3 per row). **A-** / **A+** step the whole panel through 1x / 1.25x / 1.5x / 2x
(`PanelAnchor.uiScale`, saved in PlayerPrefs `megamind.uiScale`, so it survives relaunches). The server has no thumbnail
size parameter (`GET /v1/thumbs/{id}.jpg` returns the stored keyframe), so thumbnails are just drawn larger.

### Hands and the UI ray

**Megamind > 1** adds Meta's `OVRHandPrefab` (`OVRHand` + `OVRSkeleton` + hand mesh) as `LeftOVRHand` /
`RightOVRHand` under `TrackingSpace/LeftHandAnchor` / `RightHandAnchor`. It also adds `OVRControllerPrefab` + an
`OVRRayHelper` under `RightHandAnchor/RightControllerAnchor`, puts an `OVRRayHelper` on the right hand, sets
OVRProjectConfig *Hand Tracking Support* = **Controllers And Hands**, and wires `CaptureHud.leftHand/rightHand`.
**Megamind > 1b** then wires `ChatClient.leftHand/rightHand` by `OVRHand.GetHand()` and points
`OVRInputModule.rayTransform` at `RightControllerAnchor`. Menu 1 makes a new scene, so re-run 1b after every 1.
All of this is **verify on device**:

- **Enable hand tracking on the headset:** *Settings → Movement tracking → Hand tracking* (older Horizon OS:
  *Settings → Hands & body*). Put the controllers down to switch to hands.
- **Index pinch** (thumb + index, either hand) = push-to-talk: hold to record, release to send.
- **Middle pinch** (thumb + middle, either hand) = capture pause/resume, same as A/X.
- **Panel UI:** the right controller draws a ray with a cursor on the panel; the trigger clicks. With hands, the
  right hand draws the ray and an index pinch clicks. A trigger or pinch on a button also starts push-to-talk,
  but taps shorter than `ChatClient.minRecordSeconds` (0.3 s) are dropped.

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
   - **Hands + ray (verify on device):** with hand tracking on, an index pinch shows `push-to-talk` and sends a question, a middle pinch toggles `capture PAUSED`; the right controller (or right hand) draws a ray on the panel and the Settings button opens the sub-panel.
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
| Log `app backgrounded (OnApplicationPause): pausing capture` / HUD `bg N` rising when crossing the boundary | The OS paused the app. Check *Megamind → Check Project* reports `BOUNDARYLESS_APP` and that you installed a standalone APK (not Link). Fallback: disable the boundary in MVDH or draw a large roomscale boundary (see [Boundary (Guardian)](#boundary-guardian-and-walking-around)). |
| App restarts when the headset is taken off | Normal: Horizon OS suspends background apps. Keep the headset on or set proximity sensor off in developer settings while testing. |
