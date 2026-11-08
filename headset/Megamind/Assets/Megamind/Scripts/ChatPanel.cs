// Stream 3: world-locked transcript panel with cited memory thumbnails + status HUD.
//
// Prefab layout (see headset/README.md):
//   Canvas (World Space, 0.6 m wide x uiScale)  <- PanelAnchor (tag-along / pinned, A-/A+ scale)
//     Header: recDot (always visible), statusLine (CaptureHud.ShortStatus), Menu button -> PanelMenu dropdown
//     ScrollRect -> Viewport -> Content (VerticalLayoutGroup + ContentSizeFitter)   <- `content`
//     Indicators: listeningIndicator, thinkingIndicator
// Message prefabs:
//   userBubblePrefab      : ChatBubble row, bubble right-aligned in the accent colour
//   assistantBubblePrefab : ChatBubble row, avatar glyph + neutral bubble, left-aligned
//   systemChipPrefab      : ChatBubble row, small centred chip (voice command feedback)
//   metaLinePrefab        : caption under an answer (latency, confidence)
//   thumbnailRowPrefab    : HorizontalLayoutGroup; thumbnails are instantiated under it
//   thumbnailPrefab       : rounded frame with a RawImage (180x135 px canvas units) and a child Text for "n min ago"
// Colours come from MegamindTheme (applied by the builder; the state colours below default to it).
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
        public ChatBubble userBubblePrefab;
        public ChatBubble assistantBubblePrefab;
        public ChatBubble systemChipPrefab;
        public RectTransform metaLinePrefab;
        public RectTransform thumbnailRowPrefab;
        public RectTransform thumbnailPrefab;
        public GameObject listeningIndicator;
        public GameObject thinkingIndicator;
        public GameObject emptyHint;        // "hold the trigger and ask…" shown until the first message
        public SessionStore sessions;       // optional: every message is appended to the active chat; Render() replays one

        [Header("Status strip")]
        public FrameUploader uploader;      // optional, for the capture line (fallback when `hud` is empty)
        public CaptureHud hud;              // optional: header strip = hud.ShortStatus()
        public ChatClient client;           // optional, for latency
        public Text statusLine;
        public Image serverDot;             // server health (green/red); with `recDot` set it doubles as the REC dot
        public Image recDot;                // always-visible recording indicator (privacy); blinks while capturing
        public Color okColor = MegamindTheme.Success;
        public Color errColor = MegamindTheme.Error;
        public Color recColor = MegamindTheme.Rec;
        public Color recOffColor = MegamindTheme.Paused;
        public Color errorTextColor = MegamindTheme.ErrorText;
        public Color errorBubbleColor = new Color(MegamindTheme.Error.r, MegamindTheme.Error.g, MegamindTheme.Error.b, 0.18f);
        public Color systemTextColor = MegamindTheme.TextDim;
        public Color thumbFailedColor = new Color(0.30f, 0.16f, 0.18f); // thumbnail that failed to download

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
            var b = Instantiate(userBubblePrefab, content);
            b.Text = text;
            Track(b.gameObject);
            Record("user", text, null);
        }

        /// <summary>Adds an empty assistant bubble the caller fills in later (streaming).</summary>
        public ChatBubble AddAssistantBubble(string text)
        {
            var b = Instantiate(assistantBubblePrefab, content);
            b.Text = text;
            Track(b.gameObject);
            return b;
        }

        public void AddAssistantMessage(string text, List<Citation> cited) => AddAssistantMessage(text, cited, false, null, 0f);

        public void AddAssistantMessage(string text, List<Citation> cited, bool isError) => AddAssistantMessage(text, cited, isError, null, 0f);

        public void AddAssistantMessage(string text, List<Citation> cited, bool isError, string confidence, float latencySeconds)
        {
            var b = AddAssistantBubble(text);
            if (isError) b.SetError(errorTextColor, errorBubbleColor);
            AddCitations(cited);
            AddMetaLine(confidence, latencySeconds);
            if (!isError) Record("assistant", text, cited);
        }

        public void AddCitations(List<Citation> cited)
        {
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
                    if (string.IsNullOrEmpty(c.thumb_url)) continue; // text-only memory: nothing to show
                    var frame = Instantiate(thumbnailPrefab, row);
                    var img = frame.GetComponentInChildren<RawImage>();
                    var label = frame.GetComponentInChildren<Text>();
                    if (label) label.text = Ago(c.ts);
                    if (thumbnailRowPrefab == null) Track(frame.gameObject);
                    if (img) StartCoroutine(LoadThumb(img, config.Url(c.thumb_url)));
                }
            }
        }

        public void AddMetaLine(string confidence, float latencySeconds)
        {
            if (metaLinePrefab == null || (string.IsNullOrEmpty(confidence) && latencySeconds <= 0f)) return;
            var meta = Instantiate(metaLinePrefab, content);
            var t = meta.GetComponentInChildren<Text>();
            if (t) t.text = (latencySeconds > 0f ? latencySeconds.ToString("0.0") + " s" : "")
                          + (!string.IsNullOrEmpty(confidence) ? "  ·  " + confidence.Replace('_', ' ') : "");
            Track(meta.gameObject);
        }

        /// <summary>Small dim centred line, e.g. a recognised voice command ("open menu" -> menu opened).</summary>
        public void AddSystemMessage(string text)
        {
            var b = Instantiate(systemChipPrefab != null ? systemChipPrefab : assistantBubblePrefab, content);
            b.Text = text;
            Track(b.gameObject);
            Record("system", text, null);
        }

        /// <summary>Replaces the transcript with a stored chat (launch, session switch). Nothing is re-recorded.</summary>
        public void Render(ChatSession session)
        {
            Clear();
            if (session == null) return;
            _replaying = true;
            try
            {
                foreach (var m in session.messages)
                {
                    switch (m.role)
                    {
                        case "user": AddUserMessage(m.text); break;
                        case "system": AddSystemMessage(m.text); break;
                        default: AddAssistantBubble(m.text); AddCitations(m.cited); break;
                    }
                }
            }
            finally { _replaying = false; }
        }

        private bool _replaying;
        private void Record(string role, string text, List<Citation> cited)
        {
            if (_replaying || sessions == null) return;
            sessions.Append(role, text, cited);
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
                if (target == null) yield break;
                if (req.result == UnityWebRequest.Result.Success)
                {
                    target.texture = DownloadHandlerTexture.GetContent(req);
                    target.color = Color.white; // the prefab's grey is only the loading placeholder: RawImage.color multiplies the texture
                }
                else target.color = thumbFailedColor;
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
            if (recDot)
            {
                bool capturing = hud ? hud.IsCapturing : uploader && uploader.CurrentState == FrameUploader.State.Capturing;
                recDot.color = capturing ? recColor : recOffColor;
                recDot.enabled = !capturing || (Time.unscaledTime % 1f) < 0.7f; // blink while recording, solid grey when paused
            }
            if (!statusLine) return;
            if (hud)
            {
                // one line: "REC 1 fps · 143 sent · mic 9 · url:intent" (+ "OFFLINE" when /v1/health fails)
                statusLine.text = (_serverOk ? "" : "OFFLINE · ") + hud.ShortStatus();
                statusLine.color = _serverOk ? systemTextColor : errorTextColor;
                return;
            }
            string srv = _serverOk ? _serverHost : "offline";
            string mem = _storeCount >= 0 ? _storeCount + " memories" : "";
            string cap = uploader ? $"capture {uploader.Accepted}/{uploader.Sent} ({uploader.LastReason})" : "capture off";
            string lat = client && client.LastLatencySeconds > 0f ? $"{client.LastLatencySeconds:0.0}s" : "";
            statusLine.text = string.Join("  ·  ", System.Array.FindAll(new[] { srv, mem, cap, lat }, x => !string.IsNullOrEmpty(x)));
        }
    }
}
