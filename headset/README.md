# Megamind headset app (Unity, Quest 3)

Single Unity app that (1) captures passthrough keyframes and uploads them, and
(2) hosts the voice-first AR chat panel. Both streams share this project (see
design doc §2, decision D1).

## Requirements
- Quest 3 / 3S on Horizon OS **v74+**, developer mode on, adb working (`adb devices`).
- Unity **6000.0.38f1+** (pinned in `ProjectSettings/ProjectVersion.txt`; 2022.3 LTS also works with the
  older `WebCamTextureManager` sample) with Android Build Support (SDK/NDK/OpenJDK).
- Meta XR Core SDK + MR Utility Kit **v81+** (`Packages/manifest.json` pins 81.0.0 from Meta's npm registry;
  bump to what the samples repo requires). Passthrough Camera API ships from v74; `PassthroughCameraAccess`
  from v81 (see `research/reports/05-quest3-passthrough-camera-api.md`).
- Meta's sample: https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples — copy its
  `PassthroughCamera/` folder into `Assets/` (git-ignored). `FrameUploader` currently targets the sample's
  `WebCamTextureManager`; porting it to `PassthroughCameraAccess` is a stream-1 task (report 05 §2).

## What is committed
```
headset/Megamind/
  Packages/manifest.json              Meta XR Core + MRUK (scoped registry), uGUI, OpenXR
  ProjectSettings/ProjectVersion.txt  editor version only — Unity regenerates the rest; see Megamind > 1
  Assets/Plugins/Android/AndroidManifest.xml   HEADSET_CAMERA, RECORD_AUDIO, INTERNET, cleartext, Quest 3/3S
  Assets/Megamind/Scripts/*.cs        runtime scripts (table below)
  Assets/Megamind/Editor/             menu items that build the project (no hand-written scene/prefab YAML):
    MegamindProjectSetup.cs           Megamind > 1. Apply Player Settings — Android, IL2CPP/ARM64, minSdk 32, Vulkan, ASTC, http allowed
    MegamindSceneBuilder.cs           Megamind > 2. Build Chat Panel — creates MegamindConfig.asset, bubble/thumb prefabs, the panel + settings UI, wires every reference
    MegamindBuild.cs                  Megamind > 3. Build Android APK — Build/megamind.apk (also -executeMethod for CI)
```
`Library/`, `Temp/`, `Build/`, `UserSettings/`, `Assets/PassthroughCamera/` and `packages-lock.json` are git-ignored.
Whoever opens the project first should commit the generated `.meta` files.

## Project bring-up (Engineer 1, hour 0)
1. Open `headset/Megamind/` in Unity Hub (installs the pinned editor if missing). Package Manager resolves
   Meta XR Core/MRUK from the scoped registry. Run **Megamind > 1. Apply Player Settings**.
2. Run *Meta > Tools > Project Setup Tool* and fix all. If it regenerates `AndroidManifest.xml`, re-add
   `horizonos.permission.HEADSET_CAMERA`, `RECORD_AUDIO` and `usesCleartextTraffic="true"`.
3. Import the Passthrough Camera API sample into `Assets/PassthroughCamera/`. Build & run the **unmodified**
   `CameraViewer` scene first. Confirm live camera frames on device. Do not proceed until this works.
4. With the `CameraViewer` scene open and saved, run **Megamind > 2. Build Chat Panel In Current Scene**. It
   adds `MegamindPanel` (Canvas + `PanelAnchor`/`ChatPanel`/`ChatClient`/`SettingsPanel`/`AudioSource`),
   the `EventSystem` (Meta `OVRInputModule` + `OVRRaycaster` when present, right-controller ray), finds
   `CenterEyeAnchor`, `OVRHand`s and the sample's `FrameUploader`, and sets the scene as the only one in
   Build Settings. Add `FrameUploader` to the `WebCamTextureManager` GameObject yourself (stream 1) and
   assign its `config` (`Assets/Megamind/MegamindConfig.asset`) and `headPose`.
5. Set `serverBaseUrl` on `MegamindConfig.asset` to the laptop's Tailscale IP (plain HTTP, backup: ngrok
   HTTPS URL), or change it at runtime in the panel's Settings.
6. **Megamind > 3. Build Android APK**, then
   `adb install -r Build/megamind.apk && adb shell monkey -p com.megamind.quest 1`.
   Pre-grant permissions for dev loops: `adb shell pm grant com.megamind.quest horizonos.permission.HEADSET_CAMERA`
   and `… android.permission.RECORD_AUDIO`.
   Headless: `Unity -batchmode -quit -projectPath headset/Megamind -executeMethod Megamind.Editor.MegamindBuild.Android`.

## Scripts (`Assets/Megamind/Scripts`)
| File | Stream | Purpose |
|---|---|---|
| `MegamindConfig.cs` | shared | ScriptableObject with server URL / device id (+ PlayerPrefs overrides) |
| `Dto.cs` | shared | C# mirrors of `docs/api.md` payloads |
| `FrameUploader.cs` | 1 | 1 fps sample of `WebCamTexture` → JPEG → `POST /v1/frames` |
| `ChatClient.cs` | 3 | push-to-talk (trigger or index pinch) → 16 kHz WAV → `POST /v1/chat` → play audio |
| `ChatPanel.cs` | 3 | transcript bubbles, cited thumbnails, confidence/latency line, status HUD (`/v1/health` poll) |
| `PanelAnchor.cs` | 3 | places the canvas `panelDistance` m ahead on first use; hold B/Y 0.5 s to re-anchor |
| `SettingsPanel.cs` | 3 | runtime server URL / device id, new session, purge memory |

## Stream 3 panel (Engineer 2)
`Megamind > 2` builds this hierarchy; tweak it in the Inspector and save the scene (all under one World Space Canvas):
```
MegamindPanel (Canvas: World Space, 600x720 px, CanvasScaler off)   + PanelAnchor + ChatPanel + ChatClient + AudioSource
  Header            Text statusLine, Image serverDot, Button(⚙ -> SettingsPanel.Toggle)
  Scroll            ScrollRect -> Viewport (Mask) -> Content (VerticalLayoutGroup, ContentSizeFitter)   = ChatPanel.content
  EmptyHint         Text "Hold the trigger and ask…"                                                   = ChatPanel.emptyHint
  Indicators        ListeningIndicator (red dot / bars), ThinkingIndicator (spinner)
  Settings          (inactive) InputField serverUrl, InputField deviceId, Buttons Save / New session / Purge, Text feedback
Prefabs (Assets/Megamind/Prefabs): UserBubble (Text, right-aligned), AssistantBubble (Text), ThumbRow (HorizontalLayoutGroup),
  Thumb (RawImage 132x99 with child Text for "n min ago")
```
`PanelAnchor.Start` scales the canvas so it is `panelWidthMeters` (0.6 m) wide. Set `head` to
`OVRCameraRig/TrackingSpace/CenterEyeAnchor`. For pinch-to-talk assign `OVRHand` components from
the `OVRHandPrefab`s to `ChatClient.leftHand/rightHand`; the controller trigger works without them.

A browser prototype of the same panel (same API calls, same states) is served by the server at
`http://<server>:8000/ui` — use it to agree on layout/behaviour before building the prefab, and as a
hardware-free demo fallback.

Scripts and the editor builders are written against Unity 6 / 2022.3 APIs and the Meta sample's
`PassthroughCameraSamples.WebCamTextureManager`. Nothing here has been opened in Unity or run on
hardware yet (written without a Unity install); expect compile nits and to iterate on device.
