// Stream 3: world-locked transcript panel with cited memory thumbnails + status HUD.
//
// Prefab layout (see headset/README.md):
//   Canvas (World Space, 0.6 m wide x uiScale)  <- PanelAnchor (tag-along / pinned, A-/A+ scale)
//     Header: recDot (always visible), statusLine (CaptureHud.ShortStatus), Menu button -> PanelMenu dropdown
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

        [Header("Status strip")]
        public FrameUploader uploader;      // optional, for the capture line (fallback when `hud` is empty)
        public CaptureHud hud;              // optional: header strip = hud.ShortStatus()
        public ChatClient client;           // optional, for latency
        public Text statusLine;
        public Image serverDot;             // server health (green/red); with `recDot` set it doubles as the REC dot
        public Image recDot;                // always-visible recording indicator (privacy); blinks while capturing
        public Color okColor = new Color(0.22f, 0.84f, 0.48f);
        public Color errColor = new Color(1f, 0.36f, 0.42f);
        public Color recColor = new Color(1f, 0.25f, 0.25f);
        public Color recOffColor = new Color(0.45f, 0.48f, 0.52f);
        public Color errorTextColor = new Color(1f, 0.70f, 0.73f);
        public Color systemTextColor = new Color(0.62f, 0.67f, 0.74f);
        public Color thumbFailedColor = new Color(0.30f, 0.16f, 0.18f); // thumbnail that failed to download

        private readonly Queue<GameObject> _messages = new Queue<GameObject>();
        private Text _draft;                // streaming answer bubble (POST /v1/chat/stream deltas), null when none
        private bool _scrollPending;
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
            => AddAssistantMessage(text, cited, isError, confidence, latencySeconds, 0f);

        /// <summary>firstTextSeconds &gt; 0 (streamed answers): the meta line shows time to the first words too.</summary>
        public void AddAssistantMessage(string text, List<Citation> cited, bool isError, string confidence, float latencySeconds,
                                        float firstTextSeconds)
        {
            var t = Instantiate(assistantBubblePrefab, content);
            t.text = text;
            if (isError) t.color = errorTextColor;
            Track(t.gameObject);
            AddAnswerDetails(cited, confidence, latencySeconds, firstTextSeconds);
        }

        /// <summary>Streaming answer text so far (POST /v1/chat/stream `delta`s), shown in one assistant bubble that
        /// grows in place. "" hides the bubble (`reset`); FinishDraft / DiscardDraft end it.</summary>
        public void SetDraft(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                if (_draft != null) _draft.gameObject.SetActive(false);
                return;
            }
            if (_draft == null)
            {
                _draft = Instantiate(assistantBubblePrefab, content);
                _draft.text = text;
                Track(_draft.gameObject);
                return;
            }
            if (!_draft.gameObject.activeSelf) _draft.gameObject.SetActive(true);
            _draft.text = text;
            RequestScroll();
        }

        public bool HasDraft => _draft != null;

        /// <summary>The final answer replaces the draft text (it is authoritative) and gets the usual thumbnails and
        /// meta line; without a draft this is AddAssistantMessage.</summary>
        public void FinishDraft(string text, List<Citation> cited, string confidence, float latencySeconds, float firstTextSeconds)
        {
            var t = _draft;
            _draft = null;
            if (t == null)
            {
                AddAssistantMessage(text, cited, false, confidence, latencySeconds, firstTextSeconds);
                return;
            }
            t.gameObject.SetActive(true);
            t.transform.SetAsLastSibling(); // thumbnails/meta go right under it even if a system chip came in between
            t.text = text;
            AddAnswerDetails(cited, confidence, latencySeconds, firstTextSeconds);
            RequestScroll();
        }

        /// <summary>Drops the streaming bubble (the turn failed, or it was a voice command).</summary>
        public void DiscardDraft()
        {
            if (_draft != null) Destroy(_draft.gameObject);
            _draft = null;
        }

        private void AddAnswerDetails(List<Citation> cited, string confidence, float latencySeconds, float firstTextSeconds)
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
                meta.text = (latencySeconds > 0f && firstTextSeconds > 0f ? firstTextSeconds.ToString("0.0") + "s first \u00b7 " : "")
                          + (latencySeconds > 0f ? latencySeconds.ToString("0.0") + "s" : "")
                          + (!string.IsNullOrEmpty(confidence) ? "  " + confidence.Replace('_', ' ') : "");
                Track(meta.gameObject);
            }
        }

        /// <summary>Small dim centred line, e.g. a recognised voice command ("open menu" -> menu opened).</summary>
        public void AddSystemMessage(string text)
        {
            var t = Instantiate(assistantBubblePrefab, content);
            t.text = text;
            t.fontSize = Mathf.Max(8, assistantBubblePrefab.fontSize * 3 / 4);
            t.color = systemTextColor;
            t.alignment = TextAnchor.MiddleCenter;
            Track(t.gameObject);
        }

        public void Clear()
        {
            while (_messages.Count > 0) DestroyTracked(_messages.Dequeue());
            _draft = null;
            if (emptyHint) emptyHint.SetActive(true);
        }

        private void Track(GameObject go)
        {
            if (emptyHint && emptyHint.activeSelf) emptyHint.SetActive(false);
            _messages.Enqueue(go);
            int cap = config != null ? config.maxTranscriptMessages : 40;
            while (_messages.Count > cap) DestroyTracked(_messages.Dequeue());
            RequestScroll();
        }

        private static void DestroyTracked(GameObject go)
        {
            if (go != null) Destroy(go); // a discarded draft is already gone
        }

        private void RequestScroll()
        {
            if (_scrollPending || !isActiveAndEnabled) return;
            _scrollPending = true;
            StartCoroutine(ScrollToEnd());
        }

        private IEnumerator ScrollToEnd()
        {
            yield return null; // wait for layout
            _scrollPending = false;
            if (scroll) scroll.verticalNormalizedPosition = 0f;
        }

        private void OnDisable() => _scrollPending = false; // a pending ScrollToEnd stopped with the object

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
