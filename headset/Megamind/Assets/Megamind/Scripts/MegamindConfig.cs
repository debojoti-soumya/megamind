using UnityEngine;

namespace Megamind
{
    [CreateAssetMenu(fileName = "MegamindConfig", menuName = "Megamind/Config")]
    public class MegamindConfig : ScriptableObject
    {
        [Tooltip("e.g. http://100.101.102.103:8000 (Tailscale IP of the laptop running the server)")]
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

        public string Url(string path) => serverBaseUrl.TrimEnd('/') + path;
    }
}
