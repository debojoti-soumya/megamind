// Stream 1: capture tuning that lives next to (not inside) the shared MegamindConfig, which stream 3 owns.
// Server URL / device id still come from MegamindConfig (+ its PlayerPrefs overrides written by SettingsPanel);
// this asset adds an adb-pushable file override so a tester can repoint the headset without rebuilding:
//   adb shell 'echo "{\"serverBaseUrl\":\"http://100.101.102.103:8000\"}" > /sdcard/Android/data/com.megamind.quest/files/megamind.json'
// Priority for the URL: PlayerPrefs (Settings panel) > megamind.json > MegamindConfig asset.
using System;
using System.IO;
using UnityEngine;

namespace Megamind
{
    [CreateAssetMenu(fileName = "CaptureSettings", menuName = "Megamind/Capture Settings")]
    public class CaptureSettings : ScriptableObject
    {
        public const string OverrideFileName = "megamind.json";
        private const string UrlPrefKey = "megamind.serverBaseUrl"; // same key SettingsPanel/MegamindConfig use
        private const string DevicePrefKey = "megamind.deviceId";

        [Tooltip("Keyframe sampling rate. The server filters further; 1 fps is the design target.")]
        public float captureFps = 1f;
        [Tooltip("Requested passthrough camera resolution (left camera). 1280x960 is the Quest 3 maximum.")]
        public Vector2Int cameraResolution = new Vector2Int(1280, 960);
        [Tooltip("Camera stream frame rate cap. We sample 1 fps, so 30 saves GPU/battery vs the API default of 60.")]
        public int cameraMaxFramerate = 30;
        [Tooltip("Frames are downscaled to this width before JPEG encoding (docs/api.md recommends <= 960).")]
        public int jpegMaxWidth = 960;
        [Range(30, 95)] public int jpegQuality = 75;
        [Tooltip("Uploads in flight before new frames are skipped (slow-network backpressure).")]
        public int maxPendingUploads = 4;
        public int uploadTimeoutSeconds = 10;
        [Tooltip("After a network error / 5xx / 429, skip captures for this long, doubling per consecutive failure (with +-20% jitter) up to backoffMaxSeconds. Saves encode + radio while the server is unreachable; there is no offline queue (frames are only useful live).")]
        public float backoffBaseSeconds = 1f;
        public float backoffMaxSeconds = 30f;
        [Tooltip("Stop the passthrough camera stream while capture is paused (A/X or app backgrounded); restarted on resume (battery + thermals).")]
        public bool stopCameraWhenPaused = true;
        [Tooltip("Also pause capture + ambient audio when the app only loses focus (system menu, boundary warning, Meta-button dialog). Off: only a real background (OnApplicationPause) pauses, so recording continues behind system overlays.")]
        public bool pauseOnFocusLoss = false;
        [Tooltip("Optional client-side motion gate: skip the frame when the head rotates faster than this (deg/s). 0 = off (the server's blur filter handles motion blur).")]
        public float maxHeadDegPerSec = 0f;
        [Tooltip("Start capturing automatically once the camera permission is granted.")]
        public bool captureOnStart = true;

        [Header("Ambient audio (AmbientAudioRecorder -> POST /v1/audio)")]
        [Tooltip("Record the headset mic in fixed chunks and upload them so \"what did I hear?\" works. Paused with A/X, while backgrounded and during push-to-talk.")]
        public bool ambientAudio = true;
        [Tooltip("Chunk length in seconds. 15 = docs/api.md's recommended ~15-30 s: fewer STT calls and more context per transcript; a chunk is stored ~15 s after it starts.")]
        public int audioChunkSeconds = 15;
        [Tooltip("Requested microphone rate (16 kHz mono PCM16 is what Whisper wants). The actual clip rate is written in the WAV header.")]
        public int audioSampleRate = 16000;
        [Tooltip("Drop a chunk on the device when its loudest 100 ms window is below this level (dBFS): saves Whisper cost on silence. -120 = upload everything.")]
        public float audioSilenceDb = -50f;
        [Tooltip("Chunks kept in memory while the server is unreachable (oldest dropped first). 8 x 15 s = 2 min.")]
        public int audioMaxQueuedChunks = 8;
        [Tooltip("Tries per chunk on 429/5xx (e.g. STT provider error) before it is dropped. Network errors retry until the queue overflows.")]
        public int audioMaxAttempts = 3;
        [Tooltip("POST /v1/audio transcribes synchronously, so allow longer than a frame upload.")]
        public int audioUploadTimeoutSeconds = 60;

        /// <summary>Where the active server URL came from after Resolve(): "asset", "megamind.json" or "PlayerPrefs".</summary>
        [NonSerialized] public string overrideSource = "asset";
        [NonSerialized] public string fileServerBaseUrl;
        [NonSerialized] public string fileDeviceId;
        [NonSerialized] public string fileApiToken;

        /// <summary>Parsed megamind.json, or null when absent/unreadable.</summary>
        public static FileOverrides ReadOverrideFile()
        {
            try
            {
                var path = OverrideFilePath;
                if (!File.Exists(path)) return null;
                return JsonUtility.FromJson<FileOverrides>(File.ReadAllText(path));
            }
            catch (Exception) { return null; }
        }

        [Serializable]
        public class FileOverrides
        {
            public string serverBaseUrl;
            public string deviceId;
            public float captureFps;
            public string apiToken; // server API_TOKEN (public deploys); empty = no auth
            public string ambientAudio; // "on" / "off"; absent = keep the asset value
        }

        public static string OverrideFilePath => Path.Combine(Application.persistentDataPath, OverrideFileName);

        /// <summary>Returns <paramref name="asset"/> (or defaults) with megamind.json applied. Safe to call repeatedly.</summary>
        public static CaptureSettings Resolve(CaptureSettings asset)
        {
            if (asset == null)
            {
                asset = CreateInstance<CaptureSettings>();
                Debug.LogWarning("[Megamind] no CaptureSettings assigned; using defaults.");
            }
            asset.LoadFileOverrides();
            return asset;
        }

        public void LoadFileOverrides()
        {
            fileServerBaseUrl = null;
            fileDeviceId = null;
            fileApiToken = null;
            overrideSource = PlayerPrefs.HasKey(UrlPrefKey) ? "PlayerPrefs" : "asset";
            try
            {
                var path = OverrideFilePath;
                if (!File.Exists(path)) return;
                var o = JsonUtility.FromJson<FileOverrides>(File.ReadAllText(path));
                if (o == null) return;
                if (!string.IsNullOrEmpty(o.serverBaseUrl)) fileServerBaseUrl = o.serverBaseUrl.Trim();
                if (!string.IsNullOrEmpty(o.deviceId)) fileDeviceId = o.deviceId.Trim();
                if (!string.IsNullOrEmpty(o.apiToken)) fileApiToken = o.apiToken.Trim();
                if (o.captureFps > 0f) captureFps = o.captureFps;
                var aa = (o.ambientAudio ?? "").Trim().ToLowerInvariant();
                if (aa == "on" || aa == "true") ambientAudio = true;
                else if (aa == "off" || aa == "false") ambientAudio = false;
                if (fileServerBaseUrl != null && !PlayerPrefs.HasKey(UrlPrefKey)) overrideSource = OverrideFileName;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Megamind] could not read {OverrideFileName}: {e.Message}");
            }
        }

        /// <summary>Server base URL honouring PlayerPrefs > megamind.json > MegamindConfig asset.</summary>
        public string ServerBaseUrl(MegamindConfig config)
        {
            if (PlayerPrefs.HasKey(UrlPrefKey)) return PlayerPrefs.GetString(UrlPrefKey).Trim();
            if (!string.IsNullOrEmpty(fileServerBaseUrl)) return fileServerBaseUrl;
            return config != null ? (config.serverBaseUrl ?? "").Trim() : "";
        }

        public string DeviceId(MegamindConfig config)
        {
            if (PlayerPrefs.HasKey(DevicePrefKey)) return PlayerPrefs.GetString(DevicePrefKey).Trim();
            if (!string.IsNullOrEmpty(fileDeviceId)) return fileDeviceId;
            if (config != null && !string.IsNullOrEmpty(config.deviceId)) return config.deviceId.Trim();
            return "quest3-" + SystemInfo.deviceUniqueIdentifier.Substring(0, 6);
        }

        /// <summary>API token: megamind.json > MegamindConfig asset; empty = none.</summary>
        public string ApiToken(MegamindConfig config) =>
            !string.IsNullOrEmpty(fileApiToken) ? fileApiToken : (config != null ? config.apiToken ?? "" : "");

        public string Url(MegamindConfig config, string path) => ServerBaseUrl(config).TrimEnd('/') + path;
    }
}
