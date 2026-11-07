# 05 — Quest 3 Passthrough Camera API: requirements, pitfalls, and what the Meta samples already solve (answers issue #12)

*Research bot (stream-5), 2026-11-07. Applies to: **capture** (stream-1, `headset/Megamind/Assets/Megamind/Scripts/FrameUploader.cs`). Sources: Meta Horizon OS docs (updated Oct 2026), `oculus-samples/Unity-PassthroughCameraApiSamples`, `xrdevrob/QuestCameraKit`. Uncertain claims marked (?).*

## Question
What does it take to reliably get 1 fps JPEGs + pose out of the Quest 3 cameras, and which failure modes should the hour-0 spike expect?

## Findings

### Requirements (Meta docs)
- **Quest 3 / 3S, Horizon OS v74+**; the Unity path needs **MRUK** (`com.meta.xr.mrutilitykit`) — **v81+ for the `PassthroughCameraAccess` component**, which replaces the old `WebCamTextureManager`+`WebCamTexture` sample. Meta publishes a migration guide; the current samples repo uses only `PassthroughCameraAccess`.
- Permission: `horizonos.permission.HEADSET_CAMERA` **(preferred; passthrough cams only)** or `android.permission.CAMERA` (also avatar cam ⇒ an extra dialog). The migration guide says to *remove* `android.permission.CAMERA` from the manifest and keep only `HEADSET_CAMERA`. Permission is granted at runtime by the wearer.
- **Passthrough must be enabled** in the app (OVRPassthroughLayer / Passthrough building block), and **OVRManager → "Enabled Passthrough Camera Access"** must be ticked. Add the Camera Rig + Passthrough building blocks, run **Meta > Tools > Project Setup Tool** until clean.
- **No editor/simulator path:** XR Simulator does not support PCA; you need a physical headset or Meta Horizon Link v2.1+. Plan for build-and-run only.

### API shape (`PassthroughCameraAccess`, MRUK v81+)
- `CameraPosition` (Left/Right), `RequestedResolution`, `MaxFramerate` (default 60), `GetSupportedResolutions()` (static, callable before init), `CurrentResolution`, `IsPlaying`, `GetTexture()` (GPU texture; **"can be black for a couple of frames, but it's already safe to use"**), **`GetColors()` → `Color32[]` on CPU** (what the BrightnessEstimation sample uses), `Timestamp` (DateTime of the latest image), `GetCameraPose()` (**world pose of the RGB camera at the image timestamp**), `Intrinsics` (focal length, principal point, sensor res, lens offset), `ViewportPointToRay()`.
- Supported resolutions: **1280×960 (4:3) on every OS version**; 1280×1280 and 640×360/1280×720 etc. from v83; **1728×1728 from v209**. Max frame rate 60 Hz; internal format YUV420; capture latency 20–40 ms; ~1–2 % GPU per streamed camera; ~45 MB RAM.
- Permission request pattern in the samples: a single `RequestPermissionsOnce` that calls `OVRPermissionsRequester.Request(new[]{ Permission.Scene, Permission.PassthroughCameraAccess })` once after scene load; check with `OVRPermissionsRequester.IsPermissionGranted(Permission.PassthroughCameraAccess)`.

### Documented pitfalls (Meta "Known issues" / "Common integration pitfalls")
1. **Resolution auto-select breaks apps.** `new WebCamTexture()` with no size, or requesting 0×0 / "highest", picks the largest supported resolution, which changed in v83 (1280×1280) and v209 (1728×1728). **Request a concrete tested resolution (1280×960)** and read `CurrentResolution` after start; never hard-code buffer sizes.
2. **RGB channels flipped (skin looks blue)** after the first install of a camera app on a freshly flashed v74 — fix: reboot the headset.
3. **The camera image covers less than the wearer sees** (1280×960 is a rectangular crop of the passthrough FoV; 1280×1280 widens vertical FoV, still not full). Objects at the edge of the wearer's view may be outside the frame — relevant for "I saw it out of the corner of my eye".
4. **Duplicate permission prompts / never-granted permission** if more than one mechanism requests it (OVRManager "Permission Requests On Startup", `OVRPermissionsRequester`, the sample's `PassthroughCameraPermissions`). Use exactly one.
5. **Unity 6 manifest regeneration drops the permission.** After *Meta > Tools > Update/Create store-compatible AndroidManifest.xml*, re-add `horizonos.permission.HEADSET_CAMERA` by hand.
6. **Teen/Youth parental restrictions are not applied to MQDH-installed builds** — fine for us, but means a store-installed build may behave differently (?).
7. Lifecycle: the component stops the camera when disabled or when the app is paused; **Horizon OS suspends backgrounded apps**, so capture only runs while Megamind is the foreground app (already a design decision: one app does capture + chat).

### Community pitfalls (QuestCameraKit troubleshooting, Sept 2026)
- **Camera feed does not start** → headset must be awake *and unlocked*, permissions accepted, and it must be a headset build (not Link/editor).
- **`SensorLockActivity` blocks launch** (system sensor lock after doff/idle) → put the headset on and press the physical power button once, relaunch. Expect this during a science-fair demo every time the headset sits on the table.
- Git LFS / package pinning issues are the usual reasons a sample "doesn't build"; QuestCameraKit pins Unity 6000.3.12f1 + Meta XR Core/MRUK 205.0.0; Meta samples: Unity 6000.0.38f1+ and MRUK v81+.

## Concrete recommendations for Megamind (capture)

1. **Hour-0 spike = build Meta's `CameraViewer` sample unchanged** (Unity 6000.0.38f1+, MRUK ≥ v81) on *each* of the four headsets; record OS version (`adb shell getprop ro.build.version.incremental` / Settings → Software) and `GetSupportedResolutions()` output in `research/digest/stream-1.md`. If any headset is < v74, update it first (this alone can take 20 min).
2. **Switch `FrameUploader.cs` from `WebCamTextureManager` to `PassthroughCameraAccess`** (the current script and its `width <= 16` workaround target the deprecated path). Pattern: `while (!cam.IsPlaying) yield return null;` then every 1 s: `var px = cam.GetColors();` → `Texture2D.SetPixels32` → downscale → `EncodeToJPG(75)`. Set `RequestedResolution = 1280×960`, `MaxFramerate = 30` (lower = less GPU/battery; the API default is 60).
3. **Use `cam.GetCameraPose()` and `cam.Timestamp` for the frame's `pose`/`ts`** instead of `OVRCameraRig` head pose + `Time.time`: it is the RGB camera's pose at the image timestamp (eye-to-camera offset included), and the timestamp lets the server align with other events. Convert `Timestamp` (DateTime) to Unix seconds UTC for `meta.ts`.
4. **Keep encode cost off the frame budget:** `GetColors()` + `EncodeToJPG` at 1280×960 on the Quest 3 CPU is ~tens of ms (?) — downscale to 960 px (or request 640×480 for the demo if OCR isn't needed… it is, keep 960) and run the capture in a coroutine with `yield return null` between readback and encode so the 72/90 Hz render loop doesn't hitch. If hitching persists, use `AsyncGPUReadback.Request(cam.GetTexture())` (GPU path) and encode on the callback.
5. **One permission flow:** adopt the samples' `RequestPermissionsOnce` (request `PassthroughCameraAccess` + `Scene` once); disable OVRManager's "Permission Requests On Startup"; show a panel message "Camera permission needed — tap Allow" until `IsPermissionGranted` is true. For dev iterations, pre-grant via `adb shell pm grant <package> horizonos.permission.HEADSET_CAMERA` (standard Android; Meta docs' "Managing Permissions" page describes manual grants).
6. **Demo-day runbook (ui/harness, from pitfalls above):** reboot headset in the morning (RGB-flip), keep it awake (Settings → Power → longer auto-sleep; disable proximity-sensor sleep in developer settings if available (?)), after any doff press power once to clear `SensorLockActivity`, confirm the panel shows "capturing ● N frames" before each judge.
7. **Expose a capture self-test on the panel:** `IsPlaying`, `CurrentResolution`, frames/min accepted by the server, last server error. The #1 risk in the design doc deserves a visible status, not logcat.
8. **Frame content caveat for retrieval (stream-3):** the camera FoV is narrower than the wearer's; weight "not_found" wording accordingly ("I didn't get a clear look at…").

## Sources
- PCA Overview (prerequisites, vendor tags, known issues, common pitfalls, performance): https://developers.meta.com/horizon/documentation/unity/unity-pca-overview/
- PCA Getting Started / `PassthroughCameraAccess` reference, resolutions table, troubleshooting: https://developers.meta.com/horizon/documentation/unity/unity-pca-documentation/
- Migration from WebCamTexture to PassthroughCameraAccess (v81+): https://developers.meta.com/horizon/documentation/unity/unity-pca-migration-from-webcamtexture/
- Meta samples (CameraViewer, BrightnessEstimation `GetColors()`, `RequestPermissionsOnce`): https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples
- QuestCameraKit README (troubleshooting, pinned versions): https://github.com/xrdevrob/QuestCameraKit
