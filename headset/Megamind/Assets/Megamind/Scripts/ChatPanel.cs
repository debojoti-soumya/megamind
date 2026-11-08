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
//   collagePrefab         : optional frame with a RawImage for photo montages (default: a thumbnailPrefab sized collageWidth)
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
        public RectTransform thumbnailRowPrefab;  // legacy thumbnails (used when cardPrefab is empty)
        public RectTransform thumbnailPrefab;
        public RectTransform collagePrefab;
        public float collageWidth = 540f;  // canvas units; height follows the image's aspect ratio
        public RectTransform cardStripPrefab;     // horizontal ScrollRect of CitationCards (candidates / cited)
        public CitationCard cardPrefab;
        public RectTransform turnActionsPrefab;   // meta caption + Copy + Retry under an answer
        public Lightbox lightbox;                 // opened by card taps
        public GameObject typingIndicator;        // "● ● ●" bubble in the transcript while thinking (optional)
        public Button jumpToLatest;               // "↓ Latest" pill, shown when the user scrolled up
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
        private ChatBubble _draft;          // streaming answer bubble (POST /v1/chat/stream deltas), null when none
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

        public void SetThinking(bool on)
        {
            if (thinkingIndicator) thinkingIndicator.SetActive(on);
            if (typingIndicator)
            {
                if (on) { typingIndicator.transform.SetAsLastSibling(); if (emptyHint && emptyHint.activeSelf) emptyHint.SetActive(false); }
                typingIndicator.SetActive(on);
                if (on) RequestScroll();
            }
        }

        /// <summary>Last question asked (typed text or voice transcript), for Retry.</summary>
        public string LastQuestion { get; private set; }

        public void AddUserMessage(string text)
        {
            var b = Instantiate(userBubblePrefab, content);
            b.Text = text;
            Track(b.gameObject);
            if (!_replaying) LastQuestion = text;
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
            => AddAssistantMessage(text, cited, isError, confidence, latencySeconds, 0f, null);

        /// <summary>`collage` (photo montage) replaces the thumbnail row: it already shows every cited photo.</summary>
        public void AddAssistantMessage(string text, List<Citation> cited, bool isError, string confidence, float latencySeconds,
                                        CollageResponse collage)
            => AddAssistantMessage(text, cited, isError, confidence, latencySeconds, 0f, collage);

        /// <summary>firstTextSeconds &gt; 0 (streamed answers): the meta line shows time to the first words too.</summary>
        public void AddAssistantMessage(string text, List<Citation> cited, bool isError, string confidence, float latencySeconds,
                                        float firstTextSeconds, CollageResponse collage = null)
        {
            var b = AddAssistantBubble(text);
            if (isError) b.SetError(errorTextColor, errorBubbleColor);
            if (isError) ClearCandidates();
            else AddAnswerDetails(text, cited, confidence, latencySeconds, firstTextSeconds, collage);
            if (!isError) Record("assistant", text, cited);
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
                _draft.Text = text;
                Track(_draft.gameObject);
                return;
            }
            if (!_draft.gameObject.activeSelf) _draft.gameObject.SetActive(true);
            _draft.Text = text;
            RequestScroll();
        }

        public bool HasDraft => _draft != null;

        /// <summary>The final answer replaces the draft text (it is authoritative) and gets the usual thumbnails and
        /// meta line; without a draft this is AddAssistantMessage.</summary>
        public void FinishDraft(string text, List<Citation> cited, string confidence, float latencySeconds, float firstTextSeconds,
                                CollageResponse collage = null)
        {
            var b = _draft;
            _draft = null;
            if (b == null)
            {
                AddAssistantMessage(text, cited, false, confidence, latencySeconds, firstTextSeconds, collage);
                return;
            }
            b.gameObject.SetActive(true);
            b.transform.SetAsLastSibling(); // thumbnails/meta go right under it even if a system chip came in between
            b.Text = text;
            AddAnswerDetails(text, cited, confidence, latencySeconds, firstTextSeconds, collage);
            Record("assistant", text, cited);
            RequestScroll();
        }

        /// <summary>Drops the streaming bubble (the turn failed, or it was a voice command).</summary>
        public void DiscardDraft()
        {
            if (_draft != null) Destroy(_draft.gameObject);
            _draft = null;
        }

        private void AddAnswerDetails(string answer, List<Citation> cited, string confidence, float latencySeconds, float firstTextSeconds,
                                      CollageResponse collage)
        {
            ClearCandidates(); // the answer's own citations replace the retrieval preview
            if (!AddCollage(collage)) AddCitations(cited);
            AddTurnActions(answer, confidence, latencySeconds, firstTextSeconds);
        }

        // ---- retrieval candidates (stream `retrieval` event): a strip shown right away, replaced by the cited cards ----

        private GameObject _candidates;

        public void ShowCandidates(List<Citation> candidates)
        {
            ClearCandidates();
            if (candidates == null || candidates.Count == 0 || cardPrefab == null || cardStripPrefab == null) return;
            var strip = MakeCardStrip(candidates);
            if (strip == null) return;
            _candidates = strip.gameObject;
            Track(_candidates);
            if (typingIndicator && typingIndicator.activeSelf) typingIndicator.transform.SetAsLastSibling();
        }

        public void ClearCandidates()
        {
            if (_candidates) Destroy(_candidates);
            _candidates = null;
        }

        /// <summary>Strip of CitationCards for every citation with an image; null when none has one.</summary>
        private RectTransform MakeCardStrip(List<Citation> cited)
        {
            var withImage = cited.FindAll(c => c != null && !string.IsNullOrEmpty(c.thumb_url));
            if (withImage.Count == 0) return null;
            var strip = Instantiate(cardStripPrefab, content);
            var scroll = strip.GetComponent<ScrollRect>();
            Transform row = scroll != null && scroll.content != null ? scroll.content : strip.transform;
            for (int i = 0; i < withImage.Count; i++)
            {
                var card = Instantiate(cardPrefab, row);
                card.Bind(withImage[i], withImage, i, lightbox);
                if (card.image) StartCoroutine(LoadThumb(card.image, config.Url(withImage[i].thumb_url)));
            }
            return strip;
        }

        private bool AddCollage(CollageResponse collage)
        {
            var pre = collagePrefab != null ? collagePrefab : thumbnailPrefab;
            if (collage == null || string.IsNullOrEmpty(collage.url) || pre == null) return false;
            var frame = Instantiate(pre, content);
            var img = frame.GetComponentInChildren<RawImage>();
            var label = frame.GetComponentInChildren<Text>();
            if (label) label.text = "";
            float aspect = collage.width > 0 && collage.height > 0 ? (float)collage.height / collage.width : 0.75f;
            var size = new Vector2(collageWidth, collageWidth * aspect);
            frame.sizeDelta = size;
            var le = frame.GetComponent<LayoutElement>();
            if (le == null) le = frame.gameObject.AddComponent<LayoutElement>();
            le.preferredWidth = size.x;
            le.preferredHeight = size.y;
            Track(frame.gameObject);
            if (img) StartCoroutine(LoadThumb(img, config.Url(collage.url)));
            return true;
        }

        public void AddCitations(List<Citation> cited)
        {
            if (cited == null || cited.Count == 0) return;
            if (cardPrefab != null && cardStripPrefab != null)
            {
                var strip = MakeCardStrip(cited);
                if (strip != null) Track(strip.gameObject);
                return;
            }
            if (thumbnailPrefab != null)
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

        public void AddMetaLine(string confidence, float latencySeconds) => AddMetaLine(confidence, latencySeconds, 0f);

        /// <summary>Row under an answer: meta caption + Copy (system clipboard) + Retry (asks LastQuestion again).</summary>
        public void AddTurnActions(string answer, string confidence, float latencySeconds, float firstTextSeconds)
        {
            if (turnActionsPrefab == null) { AddMetaLine(confidence, latencySeconds, firstTextSeconds); return; }
            var row = Instantiate(turnActionsPrefab, content);
            var label = row.Find("Label")?.GetComponent<Text>();
            if (label) label.text = MetaText(confidence, latencySeconds, firstTextSeconds);
            var copy = row.Find("Copy")?.GetComponent<Button>();
            var retry = row.Find("Retry")?.GetComponent<Button>();
            if (copy) copy.onClick.AddListener(() =>
            {
                GUIUtility.systemCopyBuffer = answer ?? "";
                var t = copy.GetComponentInChildren<Text>(); if (t) t.text = "Copied";
            });
            if (retry) retry.onClick.AddListener(() => { if (client != null) client.Retry(); });
            Track(row.gameObject);
        }

        private static string MetaText(string confidence, float latencySeconds, float firstTextSeconds)
            => (latencySeconds > 0f && firstTextSeconds > 0f ? firstTextSeconds.ToString("0.0") + " s first  \u00b7  " : "")
             + (latencySeconds > 0f ? latencySeconds.ToString("0.0") + " s" : "")
             + (!string.IsNullOrEmpty(confidence) ? (latencySeconds > 0f ? "  \u00b7  " : "") + confidence.Replace('_', ' ') : "");

        /// <summary>"0.9 s first · 2.4 s · high" under an answer; firstTextSeconds &gt; 0 only for streamed turns.</summary>
        public void AddMetaLine(string confidence, float latencySeconds, float firstTextSeconds)
        {
            if (metaLinePrefab == null || (string.IsNullOrEmpty(confidence) && latencySeconds <= 0f)) return;
            var meta = Instantiate(metaLinePrefab, content);
            var t = meta.GetComponentInChildren<Text>();
            if (t) t.text = MetaText(confidence, latencySeconds, firstTextSeconds);
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
            while (_messages.Count > 0) DestroyTracked(_messages.Dequeue());
            _draft = null; _candidates = null;
            if (lightbox) lightbox.Close();
            if (emptyHint) emptyHint.SetActive(true);
            _stickToBottom = true;
            if (jumpToLatest) jumpToLatest.gameObject.SetActive(false);
        }

        // ---- auto-scroll: follow the newest message unless the user scrolled up; then offer "↓ Latest" ----

        private bool _stickToBottom = true;
        private bool _layoutSettling;

        private void OnEnable()
        {
            if (scroll) scroll.onValueChanged.AddListener(OnScrolled);
            if (jumpToLatest) { jumpToLatest.onClick.RemoveListener(JumpToLatest); jumpToLatest.onClick.AddListener(JumpToLatest); jumpToLatest.gameObject.SetActive(false); }
        }

        private void OnScrolled(Vector2 _)
        {
            if (_layoutSettling || _scrollPending || scroll == null) return;
            // content shorter than the viewport reports 0 or NaN-ish values: always "at bottom" then
            bool atBottom = scroll.verticalNormalizedPosition <= 0.02f || scroll.content.rect.height <= scroll.viewport.rect.height + 1f;
            if (atBottom != _stickToBottom)
            {
                _stickToBottom = atBottom;
                if (jumpToLatest) jumpToLatest.gameObject.SetActive(!atBottom);
            }
        }

        public void JumpToLatest()
        {
            _stickToBottom = true;
            if (jumpToLatest) jumpToLatest.gameObject.SetActive(false);
            RequestScroll();
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
            if (!isActiveAndEnabled) return;
            if (!_stickToBottom)
            {
                if (jumpToLatest && !jumpToLatest.gameObject.activeSelf) jumpToLatest.gameObject.SetActive(true);
                return;
            }
            if (_scrollPending) return;
            _scrollPending = true;
            StartCoroutine(ScrollToEnd());
        }

        private IEnumerator ScrollToEnd()
        {
            yield return null; // wait for layout
            _scrollPending = false;
            if (scroll && _stickToBottom)
            {
                _layoutSettling = true;
                scroll.verticalNormalizedPosition = 0f;
                _layoutSettling = false;
            }
        }

        private void OnDisable()
        {
            _scrollPending = false; // a pending ScrollToEnd stopped with the object
            if (scroll) scroll.onValueChanged.RemoveListener(OnScrolled);
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

        public static string LocalTime(double ts)
            => System.DateTimeOffset.FromUnixTimeMilliseconds((long)(ts * 1000)).ToLocalTime().ToString("ddd HH:mm");

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
