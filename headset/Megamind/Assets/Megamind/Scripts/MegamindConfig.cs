using UnityEngine;

namespace Megamind
{
    [CreateAssetMenu(fileName = "MegamindConfig", menuName = "Megamind/Config")]
    public class MegamindConfig : ScriptableObject
    {
        private const string UrlPrefKey = "megamind.serverBaseUrl";
        private const string DevicePrefKey = "megamind.deviceId";

        [Tooltip("e.g. http://100.101.102.103:8000 (Tailscale IP of the laptop running the server). Overridden at runtime by SettingsPanel via PlayerPrefs.")]
        public string serverBaseUrl = "http://100.64.0.1:8000";
        public string deviceId = "quest3-1";

        [Header("Capture")]
        public float captureFps = 1f;
        public int jpegMaxWidth = 960;
        [Range(30, 95)] public int jpegQuality = 75;
        public int maxPendingUploads = 10;

        [Header("Voice")]
        public int micSampleRate = 16000;
        public int maxRecordSeconds = 15;
        public bool speakAnswers = true;
        public int chatTimeoutSeconds = 60;

        [Header("Panel")]
        [Tooltip("Distance in metres in front of the head where the panel is placed on (re-)anchor.")]
        public float panelDistance = 1.0f;
        public float panelWidthMeters = 0.6f;
        public int maxTranscriptMessages = 40;
        public float healthPollSeconds = 4f;

        public string Url(string path) => serverBaseUrl.TrimEnd('/') + path;

        /// <summary>Apply runtime overrides: PlayerPrefs (SettingsPanel) > megamind.json (adb push) > asset.
        /// Same precedence as CaptureSettings, so chat and capture always talk to the same server. Call once at startup.</summary>
        public void LoadOverrides()
        {
            var file = CaptureSettings.ReadOverrideFile();
            if (file != null)
            {
                if (!string.IsNullOrEmpty(file.serverBaseUrl)) serverBaseUrl = file.serverBaseUrl.Trim();
                if (!string.IsNullOrEmpty(file.deviceId)) deviceId = file.deviceId.Trim();
            }
            if (PlayerPrefs.HasKey(UrlPrefKey)) serverBaseUrl = PlayerPrefs.GetString(UrlPrefKey);
            if (PlayerPrefs.HasKey(DevicePrefKey)) deviceId = PlayerPrefs.GetString(DevicePrefKey);
        }

        public void SaveOverrides(string url, string device)
        {
            serverBaseUrl = url.Trim();
            deviceId = device.Trim();
            PlayerPrefs.SetString(UrlPrefKey, serverBaseUrl);
            PlayerPrefs.SetString(DevicePrefKey, deviceId);
            PlayerPrefs.Save();
        }

        public void ClearOverrides()
        {
            PlayerPrefs.DeleteKey(UrlPrefKey);
            PlayerPrefs.DeleteKey(DevicePrefKey);
            PlayerPrefs.Save();
        }
    }
}
