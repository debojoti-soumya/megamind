// Stream 3: world-locked transcript panel with cited memory thumbnails + status HUD.
//
// Prefab layout (see headset/README.md):
//   Canvas (World Space, 0.6 m wide x uiScale)  <- PanelAnchor (tag-along / pinned, A-/A+ scale)
//     Header: statusLine (Text), A- / A+ / Pin / Settings buttons
//     ScrollRect -> Viewport -> Content (VerticalLayoutGroup + ContentSizeFitter)   <- `content`
//     Indicators: listeningIndicator, thinkingIndicator
// Message prefabs:
//   userBubblePrefab      : Text inside a right-aligned bubble
//   assistantBubblePrefab : Text inside a left-aligned bubble
//   thumbnailRowPrefab    : HorizontalLayoutGroup; thumbnails are instantiated under it
//   thumbnailPrefab       : RawImage (180x135 px canvas units) with optional child Text for "n min ago"
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace Megamind
{
    public class ChatPanel : MonoBehaviour
    {
        public MegamindConfig config;
        public Transform content;          // VerticalLayoutGroup
        public ScrollRect scroll;          // optional, auto-scrolls to the newest message
        public Text userBubblePrefab;
        public Text assistantBubblePrefab;
        public RectTransform thumbnailRowPrefab;
        public RawImage thumbnailPrefab;
        public GameObject listeningIndicator;
        public GameObject thinkingIndicator;
        public GameObject emptyHint;        // "hold the trigger and ask…" shown until the first message

        [Header("Status HUD")]
        public FrameUploader uploader;      // optional, for the capture line
        public ChatClient client;           // optional, for latency
        public Text statusLine;
        public Image serverDot;
        public Color okColor = new Color(0.22f, 0.84f, 0.48f);
        public Color errColor = new Color(1f, 0.36f, 0.42f);
        public Color errorTextColor = new Color(1f, 0.70f, 0.73f);

        private readonly Queue<GameObject> _messages = new Queue<GameObject>();
        private bool _serverOk;
        private int _storeCount = -1;
        private string _serverHost = "";

        private IEnumerator Start()
        {
            SetListening(false);
            SetThinking(false);
            while (true)
            {
                yield return PollHealth();
                yield return new WaitForSeconds(config != null ? config.healthPollSeconds : 4f);
            }
        }

        public void SetListening(bool on) { if (listeningIndicator) listeningIndicator.SetActive(on); }
        public void SetThinking(bool on) { if (thinkingIndicator) thinkingIndicator.SetActive(on); }

        public void AddUserMessage(string text)
        {
            var t = Instantiate(userBubblePrefab, content);
            t.text = text;
            Track(t.gameObject);
        }

        public void AddAssistantMessage(string text, List<Citation> cited) => AddAssistantMessage(text, cited, false, null, 0f);

        public void AddAssistantMessage(string text, List<Citation> cited, bool isError) => AddAssistantMessage(text, cited, isError, null, 0f);

        public void AddAssistantMessage(string text, List<Citation> cited, bool isError, string confidence, float latencySeconds)
        {
            var t = Instantiate(assistantBubblePrefab, content);
            t.text = text;
            if (isError) t.color = errorTextColor;
            Track(t.gameObject);

            if (cited != null && cited.Count > 0 && thumbnailPrefab != null)
            {
                Transform row = content;
                if (thumbnailRowPrefab != null)
                {
                    var r = Instantiate(thumbnailRowPrefab, content);
                    Track(r.gameObject);
                    row = r;
                }
                foreach (var c in cited)
                {
                    var img = Instantiate(thumbnailPrefab, row);
                    var label = img.GetComponentInChildren<Text>();
                    if (label) label.text = Ago(c.ts);
                    if (thumbnailRowPrefab == null) Track(img.gameObject);
                    StartCoroutine(LoadThumb(img, config.Url(c.thumb_url)));
                }
            }

            if (!string.IsNullOrEmpty(confidence) || latencySeconds > 0f)
            {
                var meta = Instantiate(assistantBubblePrefab, content);
                meta.fontSize = Mathf.Max(8, assistantBubblePrefab.fontSize * 3 / 4);
                meta.color = new Color(0.6f, 0.65f, 0.72f);
                meta.text = (latencySeconds > 0f ? latencySeconds.ToString("0.0") + "s" : "")
                          + (!string.IsNullOrEmpty(confidence) ? "  " + confidence.Replace('_', ' ') : "");
                Track(meta.gameObject);
            }
        }

        public void Clear()
        {
            while (_messages.Count > 0) Destroy(_messages.Dequeue());
            if (emptyHint) emptyHint.SetActive(true);
        }

        private void Track(GameObject go)
        {
            if (emptyHint && emptyHint.activeSelf) emptyHint.SetActive(false);
            _messages.Enqueue(go);
            int cap = config != null ? config.maxTranscriptMessages : 40;
            while (_messages.Count > cap) Destroy(_messages.Dequeue());
            StartCoroutine(ScrollToEnd());
        }

        private IEnumerator ScrollToEnd()
        {
            yield return null; // wait for layout
            if (scroll) scroll.verticalNormalizedPosition = 0f;
        }

        private IEnumerator LoadThumb(RawImage target, string url)
        {
            using (var req = UnityWebRequestTexture.GetTexture(url))
            {
                config.Authorize(req);
                req.timeout = 10;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success && target != null)
                    target.texture = DownloadHandlerTexture.GetContent(req);
            }
        }

        private IEnumerator PollHealth()
        {
            if (config == null) yield break;
            using (var req = UnityWebRequest.Get(config.Url("/v1/health")))
            {
                config.Authorize(req);
                req.timeout = 3;
                yield return req.SendWebRequest();
                _serverOk = req.result == UnityWebRequest.Result.Success;
                if (_serverOk)
                {
                    try { _storeCount = JsonUtility.FromJson<HealthResponse>(req.downloadHandler.text).store_count; }
                    catch { _storeCount = -1; }
                }
                _serverHost = HostOf(config.serverBaseUrl);
            }
        }

        private static string HostOf(string url)
        {
            try { return new System.Uri(url).Authority; } catch { return url; }
        }

        public static string Ago(double ts)
        {
            double s = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0 - ts;
            if (s < 60) return Mathf.RoundToInt((float)s) + "s ago";
            if (s < 3600) return Mathf.RoundToInt((float)(s / 60)) + " min ago";
            return (s / 3600).ToString("0.0") + " h ago";
        }

        private void Update()
        {
            if (serverDot) serverDot.color = _serverOk ? okColor : errColor;
            if (!statusLine) return;
            string srv = _serverOk ? _serverHost : "offline";
            string mem = _storeCount >= 0 ? _storeCount + " memories" : "";
            string cap = uploader ? $"capture {uploader.Accepted}/{uploader.Sent} ({uploader.LastReason})" : "capture off";
            string lat = client && client.LastLatencySeconds > 0f ? $"{client.LastLatencySeconds:0.0}s" : "";
            statusLine.text = string.Join("  ·  ", System.Array.FindAll(new[] { srv, mem, cap, lat }, x => !string.IsNullOrEmpty(x)));
        }
    }
}
