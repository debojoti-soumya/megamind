# Megamind headset app (Unity, Quest 3)

Single Unity app that (1) captures passthrough keyframes and uploads them, and
(2) hosts the voice-first AR chat panel. Both streams share this project (see
design doc §2, decision D1).

## Requirements
- Quest 3 / 3S on Horizon OS **v74+**, developer mode on, adb working (`adb devices`).
- Unity **2022.3 LTS** (or Unity 6) with Android Build Support (SDK/NDK/OpenJDK).
- Meta XR Core SDK **v74+** (`com.meta.xr.sdk.core`) — the Passthrough Camera API
  ships from v74. Install via Package Manager or the "Meta XR All-in-One SDK".
- Meta's sample: https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples
  (copy `PassthroughCamera/` into `Assets/`; we use its `WebCamTextureManager`).

## Project bring-up (Engineer 1, hour 0)
1. Create the Unity project at `headset/Megamind/` (this folder; `Library/`, `Temp/`,
   etc. are git-ignored). Switch platform to Android, Texture compression ASTC.
2. Install Meta XR Core SDK. Run *Meta > Tools > Project Setup Tool* and fix all.
3. Import the Passthrough Camera API sample. Build & run the **unmodified** sample
   scene first. Confirm live camera frames on device. Do not proceed until this works.
4. Add `Assets/Megamind/Scripts/*.cs` (this repo) to the sample scene:
   - `MegamindConfig` ScriptableObject: server base URL, device id.
   - `FrameUploader` on the same GameObject as `WebCamTextureManager`.
   - `ChatClient` + `ChatPanel` on a world-locked Canvas (0.6 m wide, 1 m ahead).
5. `AndroidManifest.xml` must include:
   ```xml
   <uses-permission android:name="horizonos.permission.HEADSET_CAMERA" />
   <uses-permission android:name="android.permission.CAMERA" />
   <uses-permission android:name="android.permission.RECORD_AUDIO" />
   <uses-permission android:name="android.permission.INTERNET" />
   <application android:usesCleartextTraffic="true" ...>
   ```
6. Networking: put the laptop and headset on the same Tailscale tailnet and use the
   laptop's tailnet IP as `serverBaseUrl` (plain HTTP). Backup: ngrok HTTPS URL.

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

## Stream 3 panel prefab (Engineer 2)
Hierarchy to build in the sample scene (all under one World Space Canvas):
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

Scripts are written against Unity 2022.3 APIs and the Meta sample's
`PassthroughCameraSamples.WebCamTextureManager`. They compile-check only; they have not
run on hardware yet. Expect to iterate on device.
