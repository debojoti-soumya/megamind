# Megamind Capture for Quest 3 (native Android, optional)

A small native Android app for Meta Quest 3/3S that captures the real world through the **Passthrough Camera
API** (plain Camera2) plus the headset mic, and streams it to the Megamind server:

- frames: `POST /v1/frames`, one JPEG per request, `meta = {"device_id", "ts"}`
- ambient audio: `POST /v1/audio`, one 16 kHz mono PCM16 WAV per chunk, `meta = {"device_id", "ts_start"}`
- real video: `POST /v1/video`, one 30 fps H.264 mp4 per chunk, `meta = {"device_id", "ts_start"}` (on by default;
  replays and montages then use real video instead of 1 fps frames)

exactly as [`docs/api.md`](../docs/api.md) specifies. It is a capture-only client: asking questions still
happens elsewhere (the Unity app's chat panel, `tools/ask.py`, or the Quest Browser).

## Which headset app should I use?

| | `headset/` (Unity) | `headset-android/` (this, native Kotlin) |
|---|---|---|
| What | Capture **and** AR chat panel (push-to-talk, TTS, cited thumbnails) | Capture only (frames + ambient audio + real video) |
| Device status | Playtested on a Quest 3 (Horizon OS v74), see [`headset/README.md`](../headset/README.md) | Capture/service code **verified on a real Quest 3, Horizon OS v209** (2026-11-07, as Recall's app); the Megamind upload path is verified against a local server only, see below |
| Build | Unity 6 + Meta XR SDK 85 (package `com.megamind.quest`) | `./gradlew assembleDebug` (JDK 17 + Android SDK, no Unity), ~2.6 MB APK, package `ai.megamind.quest` |
| Background | Horizon suspends it when it loses focus | Foreground service (`camera\|microphone`): keeps capturing behind the Quest Browser and after Home |
| Offline | - | Persistent on-disk queue, resumes uploads after outages (3 GB cap) |
| Pose | Sends head pose | No pose (`pose` is optional in the contract; a 2D panel app has no head pose) |

The two are independent: this directory does not change the Unity app or the server. Use the native app to stream
memories into the server from a background service with an offline queue; use the Unity app for the full AR experience
(capture plus the chat panel).
Running both at once would capture twice (two `device_id`s); pick one for capture.

## What it does

- Package `ai.megamind.quest`, Kotlin, plain Android Views, no AndroidX/Compose. The only runtime dependency is
  the Kotlin stdlib (JUnit for host tests).
- Camera: left passthrough camera, picked by Meta metadata (`camera_source==0`, `position==0`, camera id `50` on
  our headset), YUV_420_888 at 1280x960 / 15 fps. One frame per interval (default 1 s) goes NV21 -> JPEG and is
  uploaded at the full 1280x960 (`MAX_UPLOAD_WIDTH`).
- Video (ported from Recall's app, `VideoRecorder.kt`): a hardware H.264 encoder (`MediaCodec`, `video/avc`) records
  **real 30 fps video** at 1280x960, ~3 Mbps VBR, IDR every 1 s. Its input `Surface` is a second Camera2 output in
  the same session as the YUV `ImageReader` (zero-copy, no CPU-side YUV conversion); with it on, the camera runs at
  30 fps (`TEMPLATE_RECORD`) instead of 15. One `MediaMuxer` per chunk writes `video.mp4` (video only); at each chunk
  boundary the app requests a sync frame, so every clip starts on a keyframe, and the previous chunk's mp4 is
  finalized before the chunk is queued. `meta.video` = `{file, offset, duration, frames, fps, width, height, codec,
  bitrate}`, `offset` = first frame's time after the chunk start (codec PTS = sensor timestamp, mapped to wall clock
  like the JPEG times). **Fallback:** if the encoder fails to start or errors, if the 2-output session fails to
  configure, or if it delivers no camera frames within 5 s, the camera reopens JPEG-only at 15 fps (exactly the
  previous behaviour) and keeps capturing. Turn video off with the **Record real video** setting or
  `--ez video false` (applies from the next Start).
- Audio: `AudioRecord` 16 kHz mono PCM16 (VOICE_RECOGNITION source).
- Chunks: every chunk length (default 5 s) the current `files/chunks/<id>.partial/` is closed into `<id>/`
  (`meta.json` + JPEGs + `audio.pcm16` + `video.mp4`). The upload thread sends a chunk's frames to `/v1/frames`
  (`ts` = chunk start + frame offset), then its audio to `/v1/audio` as a WAV (`ts_start` = chunk start; late
  mic audio is padded with leading silence so offsets line up), then its `video.mp4` to `/v1/video` (`ts_start` =
  chunk start + `meta.video.offset`). Each accepted part is recorded in `sent.txt`, so a retry never re-sends; the
  chunk is deleted once every part got a 2xx. The 1 fps JPEGs still drive memory; the video only feeds replay and
  montage on the server.
- Retries: unreachable backend or 5xx -> exponential backoff (2 s doubling to 60 s; "Retry uploads" skips the
  wait). `/v1/frames` 400/413/415/422 -> the chunk moves to `files/chunks_failed/` ("Retry uploads" re-queues
  it). A rejected audio part (400/413/415/422), a server without `/v1/audio` (404), or 3 consecutive 5xx on audio
  (e.g. STT provider down, 502) skip that chunk's audio so frames keep flowing. Video works the same way: a
  rejected clip (400/409/413/415/422) or 3 consecutive 5xx skip that chunk's video; a server without `/v1/video`
  (404/405, an older server or `VIDEO_ENABLED=false`) turns video uploads off until "Retry uploads" / Save. A
  `"reason":"privacy"` answer (server face blur on) means the server discarded the clip. Filtered frames
  (`accepted: false`) are normal and count as sent.
- Settings (in-app or by intent): backend URL (default `http://localhost:8000`), frame interval (1 s), chunk
  length (5 s; also the length of each audio and video clip), device ID (default `quest3-native`), record real video
  (on). Cleartext HTTP is allowed.

Audio clip length: `docs/api.md` suggests 15-30 s audio chunks; this app sends one clip per chunk, 5 s by default
so frames reach the server within ~5 s ("what am I looking at right now?"). The server accepts any length up to
600 s. If transcripts get split mid-sentence, raise **Chunk length** to 15 (frames then arrive up to 15 s late).

## Toolchain (macOS, Homebrew)

```bash
export HOMEBREW_NO_AUTO_UPDATE=1
brew install openjdk@17                       # ~305 MB
brew install --cask android-commandlinetools  # ~173 MB (sdkmanager)
brew install --cask android-platform-tools    # adb

export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=$HOME/Library/Android/sdk
SDKM=/opt/homebrew/share/android-commandlinetools/cmdline-tools/latest/bin/sdkmanager
yes | $SDKM --sdk_root=$ANDROID_HOME --licenses
$SDKM --sdk_root=$ANDROID_HOME "platforms;android-36" "build-tools;35.0.0"   # ~320 MB
```

Versions: Gradle 8.14.1 (wrapper), AGP 8.11.1, Kotlin 2.2.21, compileSdk 36, targetSdk 34 (Horizon OS v209 is
Android 14), minSdk 32. Either export `ANDROID_HOME` or put `sdk.dir=...` in `local.properties` (gitignored).

## Build, test, install (USB)

```bash
cd headset-android
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=$HOME/Library/Android/sdk
./gradlew assembleDebug testDebugUnitTest     # -> app/build/outputs/apk/debug/app-debug.apk

adb install -r app/build/outputs/apk/debug/app-debug.apk
for p in android.permission.CAMERA horizonos.permission.HEADSET_CAMERA \
         android.permission.RECORD_AUDIO android.permission.POST_NOTIFICATIONS; do
  adb shell pm grant ai.megamind.quest $p
done
adb reverse tcp:8000 tcp:8000                # headset localhost:8000 -> Mac server (re-run after every replug)

adb shell am start -n ai.megamind.quest/.MainActivity --ez autostart true   # start capture
adb shell am start -n ai.megamind.quest/.MainActivity --es action pause
adb shell am start -n ai.megamind.quest/.MainActivity --es action stop
adb shell am start -n ai.megamind.quest/.MainActivity --es device_id quest3-asher
adb shell am start -n ai.megamind.quest/.MainActivity --es api_token <token>   # server API_TOKEN ("" clears it)
adb logcat -v time | grep MegamindQuest      # camera open, "Uploaded <id>: frames accepted=..", "Mic silenced by system"
```

Start the server as in the top-level README (`uvicorn megamind.main:app --port 8000` from `server/`). Without adb,
open **Megamind Capture** from the Library (Unknown Sources), allow the camera, headset-camera, microphone and
notification prompts, then press **Start**. Intent extras are one-shot (not replayed when the activity is recreated).

## Using it over Wi-Fi (no cable)

1. Mac and Quest on the same Wi-Fi. Find the Mac's LAN IP: `ipconfig getifaddr en0` (e.g. `192.168.1.20`).
2. Start the server on all interfaces: `uvicorn megamind.main:app --host 0.0.0.0 --port 8000`, and allow incoming
   connections in the macOS firewall. Check from the Mac: `curl http://192.168.1.20:8000/v1/health`.
3. In the app's Settings set **Backend URL** to `http://192.168.1.20:8000`, press **Save**, then **Test
   connection** ("Connected to Megamind ... HTTP 200", it calls `GET /v1/health`). Or, while still on USB:
   `adb shell am start -n ai.megamind.quest/.MainActivity --es backend_url http://192.168.1.20:8000 --ez autostart true`
4. Unplug. Chunks queue on the headset while the server is unreachable and upload in order once it is back.
   Plain HTTP is allowed (`network_security_config.xml`), so only do this on a trusted network.

## Token-protected servers (Modal deploy)

A server with `API_TOKEN` set (always the case for the Modal deploy, [`deploy/README.md`](../deploy/README.md),
[`docs/api.md#auth`](../docs/api.md#auth)) needs the token on every request. Set it in Settings → **API token**
(then **Save**) or over adb; it is saved in the app's preferences and sent as `Authorization: Bearer <token>` on
`/v1/frames`, `/v1/audio`, `/v1/video` and the **Test connection** `GET /v1/health`:

```bash
adb shell am start -n ai.megamind.quest/.MainActivity \
  --es backend_url https://<workspace>--megamind-web.modal.run --es api_token <token> --ez autostart true
```

A 401/403 is not retried: uploads **pause** with "PAUSED: server rejected the API token" on the status line and
`HTTP 401: API token missing or wrong` as the last error (logcat `MegamindQuest:Upload`). Chunks stay queued (up
to the on-device cap) and upload once a new token is saved, an `api_token` intent arrives, or **Retry uploads** is
pressed. HTTPS needs no extra config (verify on device).

## Verifying the request format without a headset

The request bodies are built by `MegamindApi.kt` (pure JVM, no Android classes), and the uploader sends exactly
those bytes. Two checks:

- **Host JVM tests** (`app/src/test/.../MegamindApiTest.kt`): exact `meta` JSON, multipart layout, WAV header.
  With a server running they also POST a real frame and a 5 s clip through the same code path:
  ```bash
  (cd ../server && uvicorn megamind.main:app --port 8300) &
  ./gradlew testDebugUnitTest -Pmegamind.url=http://127.0.0.1:8300
  # POST /v1/frames -> 202 {"frame_id":"...","accepted":true,"reason":null}
  # POST /v1/audio  -> 200 {"audio_id":"...","duration_s":5.0,"transcript":"[mock transcript]","token_ids":[...]}
  # POST /v1/video  -> 200 {"video_id":"...","accepted":true,"reason":null,"ts_start":...,"duration_s":1.0,...}
  ```
  Against a server with `API_TOKEN` set, add `-Pmegamind.token=<token>` (or `MEGAMIND_API_TOKEN`): the live tests
  send it, and `liveRejectsMissingToken` checks the 401 without it. The always-on `postSendsBearerTokenOnEveryPath`
  / `missingOrWrongTokenIs401AuthFailure` tests check the header and the 401 against an in-process HTTP server.
  Without `-Pmegamind.url` (or `MEGAMIND_URL`) the live tests are skipped. The video test posts
  `app/src/test/resources/clip.mp4` (1 s, 160x120, 30 fps H.264, made with ffmpeg).
- **Server pytest** (`server/tests/test_quest_native_payloads.py`, runs in CI): replays the same bytes through the
  FastAPI app with mocks, so a server-side contract change that would break this app fails CI.

## Verified on device (Quest 3, Horizon OS v209, 2026-11-07)

These results come from the same capture code running in our "Recall" prototype on a real Quest 3. Only the upload
path (Recall's single chunk endpoint -> Megamind's `/v1/frames` + `/v1/audio`) and the 960 px downscale are new;
those are verified against a local Megamind server as above, not yet on the headset.

- Permissions granted (CAMERA, HEADSET_CAMERA, RECORD_AUDIO, POST_NOTIFICATIONS); `startForeground` type
  `camera|microphone` accepted.
- Camera list: `1` (avatar/virtual, no Meta metadata), `50` (source 0, left), `51` (source 0, right). The app opens
  **50**, streams 1280x960 at 15 fps and keeps one frame per second. Frames show the real room, not the avatar.
- Each 5 s chunk had 5 frames plus ~5 s of PCM (153,600-163,200 bytes), uploaded within ~50 ms of closing.
- **Home / Quest Browser:** capture keeps running (frames and audio) while the browser has focus and after Home.
- **Headset off-head/asleep:** the camera keeps streaming (dim frames of wherever it points). Horizon turns the
  system mic privacy toggle on while asleep, so chunks go up frame-only; the mic comes back on its own within ~0.5 s
  of waking. The UI shows "MIC MUTED BY SYSTEM".
- Pause/Start/Stop via intents work; Stop clears the "capture wanted" flag so a sticky restart never resumes.
- To keep the headset awake without wearing it: `adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`
  (undo: `adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable`).

See [RESEARCH.md](RESEARCH.md) for the Passthrough Camera API facts (camera ids, metadata keys, permissions) and links.

## Known limitations

- Audio is only captured while the headset is worn/awake (Horizon mutes the mic system-wide otherwise).
- Mic level is low (VOICE_RECOGNITION source, no AGC); distant voices may be lost.
- Starting capture requires the app in the foreground (Android 14 forbids starting camera/mic foreground services
  from the background). If the process is killed while recording, the sticky restart may be refused; reopen and
  press Start.
- One camera (left), no head pose, no on-device privacy filtering (the server's filters still apply).
- Battery: a partial wake lock plus 15 fps camera streaming (30 fps + the H.264 encoder with video on) run until
  Stop, including while the headset sleeps.
- Video is new and not yet verified on a headset: the Camera2 2-output session + `MediaCodec` surface encoder is
  Recall's code, built here, and the upload/server path is verified with a synthetic clip against a local server.
  Check logcat for `MegamindQuest:Video` (`Encoder ... started`, `Video <chunk>: N frames, X fps, KB, offset`) and
  the UI's `video:` line. If the headset can't feed both outputs, the app falls back to JPEG-only by itself.
- Bandwidth/disk with video on: ~1.9 MB of H.264 per 5 s chunk (3 Mbps) on top of the JPEGs and PCM, ~1.8 GB/hour;
  the 3 GB offline backlog then holds ~1.7 h. Lower `VideoRecorder.BITRATE` or turn video off to trade quality for
  size. The server keeps clips for `HOT_RETENTION_H` (72 h) by default (`VIDEO_*` in `server/.env.example`).
- `adb reverse` must be re-run after every USB reconnect; otherwise uploads retry with "Backend unreachable".
