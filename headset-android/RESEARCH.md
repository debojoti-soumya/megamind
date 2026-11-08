# Quest 3 Passthrough Camera API (PCA): research notes

Collected 2026-11-07 for the native capture app (first built for our "Recall" prototype, ported here as `headset-android/`). Sources are Meta's current docs, Meta's official sample, and `adb shell dumpsys media.camera` on our own Quest 3 (Horizon OS build 209, Android 14).

## Sources

- Android apps, Passthrough Camera: https://developers.meta.com/horizon/documentation/android-apps/passthrough-camera
- Native PCA overview (resolutions, metadata, known issues): https://developers.meta.com/horizon/documentation/native/android/pca-native-overview/
- Spatial SDK PCA overview (same content): https://developers.meta.com/horizon/documentation/spatial-sdk/spatial-sdk-pca-overview/
- Samples page: https://developers.meta.com/horizon/documentation/android-apps/passthrough-camera-samples
- Official Kotlin/Camera2 sample: https://github.com/meta-quest/Meta-Passthrough-Camera-API-Samples (`app/src/main/java/com/oculus/camerademo/`)

## Key facts

| Topic | Finding |
|---|---|
| Devices / OS | Quest 3 and 3S only. Horizon OS **v74+** (our headset runs v209, which adds 1728x1728 and dynamic metadata). |
| API | Plain Android **Camera2** (also NDK, Camera1, CameraX). No Meta SDK or library is needed. |
| Permissions | Either `android.permission.CAMERA` (passthrough and avatar cameras) or `horizonos.permission.HEADSET_CAMERA` (passthrough only). Passthrough must be enabled in the system. Meta's sample declares and requests **both**, and refuses to open the camera unless both are granted. We do the same. Both are runtime ("dangerous") permissions. |
| Manifest (from sample) | `<uses-permission android:name="android.permission.CAMERA"/>`, `<uses-permission android:name="horizonos.permission.HEADSET_CAMERA"/>`, `<uses-feature android:name="android.hardware.camera2.full" android:required="true"/>`, `<uses-feature android:name="android.hardware.vr.headtracking" android:required="true"/>`, `<uses-feature android:name="com.oculus.feature.PASSTHROUGH"/>`, `<meta-data android:name="com.oculus.supportedDevices" android:value="quest3\|quest3s"/>`. |
| 2D panel vs immersive | **A plain 2D panel app works.** Meta's official sample is an ordinary Compose `ComponentActivity` with no OpenXR or immersive activity. The docs don't say this outright; the sample shows it. |
| Camera identification | Vendor `CameraCharacteristics` keys: `com.meta.extra_metadata.camera_source` (tag 0x80004d00): **0 = passthrough RGB**. `com.meta.extra_metadata.position` (0x80004d01): **0 = left, 1 = right** (wearer's view). The sample reads both as `CameraCharacteristics.Key(name, Int::class.java)`. The HAL stores them as `byte[1]`, so our app tries Int first, then Byte. |
| Camera ids on our headset (dumpsys) | `50`: source 0, position 0 = **left passthrough (we use this)**. `51`: source 0, position 1 = right passthrough. `60` and `70`: source 3 (spatial/depth-style streams, 3456x972 and others). `1`: source 1, LENS_FACING_FRONT = avatar/selfie camera (the one the browser exposes). |
| Resolutions | Documented: 1280x960 (4:3), 1280x1280 (v83+, taller FOV), 1728x1728 (v209+, max). Meta advises picking a concrete tested size, not the largest. Our headset's stream configs list 320x240 through 1728x1728 for PRIVATE(34), YUV_420_888(35) and JPEG/BLOB(33). |
| Format / fps / latency | Internal format YUV420, 60 Hz. AE target fps ranges are [15,15], [30,30], [60,60]. Capture latency is 20-40 ms. Cost is about 1-2% GPU and about 45 MB RAM per streamed camera. |
| Known issues | v74 only: RGB channels can be swapped after the first camera app install following an OS flash (a reboot fixes it). The camera image is a smaller rectangle than the passthrough view the user sees. |
| Background / unfocused / doffed | **Not documented.** Meta's sample stops the camera in `onPause` and resumes it in `onResume`. Android 14 rules also apply: a camera foreground service has to be started while the app is in the foreground, and it needs `FOREGROUND_SERVICE_CAMERA` and `FOREGROUND_SERVICE_MICROPHONE`. We run capture in a `camera\|microphone` foreground service and record what happens on-device (see README "Observed behaviour"). On Quest the cameras are expected to idle while the headset is off-head (proximity sensor triggers sleep). |
| Privacy | Camera images count as "Device User Data" under Meta's Developer Data Use Policy. |

## Design consequences for the app

1. Use Camera2 directly. Enumerate `cameraIdList` and pick `camera_source == 0 && position == 0` (left). Fall back to any source-0 camera, then to any LENS_FACING_BACK camera.
2. Use ImageReader at **1280x960** YUV_420_888 with the AE target fps at 15. Every `interval` seconds, convert one frame to NV21 and compress it to JPEG (quality 80) with `YuvImage`. JPEG(33) is also listed, but encoding one frame per second from YUV is cheap and stays under our control.
3. Request CAMERA, HEADSET_CAMERA, RECORD_AUDIO and POST_NOTIFICATIONS at runtime. They can also be granted via `adb shell pm grant`.
4. Run the foreground service with `foregroundServiceType="camera|microphone"`.
