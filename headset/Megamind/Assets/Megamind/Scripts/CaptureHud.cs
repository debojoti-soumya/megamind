// Capture status data source + the A/X / middle-pinch capture toggle.
//
// `ShortStatus()` is the one-line strip in the chat panel header ("REC 1 fps · 143 sent · mic 9 · url:intent") and
// `FullReport()` the multi-line debug text (state, accepted/sent, last server reason, latency, skipped counters,
// bg/unfocus, ambient-audio line, server URL + source). ChatPanel / PanelMenu read these; the always-visible
// recording dot lives in the chat panel header (privacy: capture is never silent).
//
// The old separate head-locked HUD canvas is still available for debugging: set `showOwnCanvas` (or assign `target`).
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class CaptureHud : MonoBehaviour
    {
        public FrameUploader uploader;
        [Tooltip("Ambient audio recorder for the mic line. Found automatically when empty.")]
        public AmbientAudioRecorder audioRecorder;
        [Tooltip("CenterEyeAnchor; the HUD is parented here. Found automatically when empty.")]
        public Transform head;
        [Tooltip("Legacy head-locked HUD canvas (off: the chat panel header shows the status instead).")]
        public bool showOwnCanvas = false;
        [Tooltip("Optional existing UI Text that receives FullReport(); set automatically by showOwnCanvas.")]
        public Text target;
        public Vector3 localOffset = new Vector3(0f, -0.22f, 1.0f);
        public float widthMeters = 0.55f;
        public int fontSize = 22;
        public float refreshSeconds = 0.25f;
        [Tooltip("A/X button or either hand's middle-finger pinch toggles capture on/off.")]
        public bool toggleWithButton = true;
        [Tooltip("Hand-tracking hands for the middle-finger pinch toggle. Found automatically when empty.")]
        public OVRHand leftHand, rightHand;

        private float _next;
        private Image _dot;
        private bool _middlePinchHeld;

        private void Start()
        {
            if (uploader == null) uploader = FindAnyObjectByType<FrameUploader>();
            if (audioRecorder == null) audioRecorder = FindAnyObjectByType<AmbientAudioRecorder>();
            if (head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) head = rig.centerEyeAnchor != null ? rig.centerEyeAnchor : rig.transform;
            }
            if (leftHand == null || rightHand == null)
            {
                foreach (var h in FindObjectsByType<OVRHand>(FindObjectsSortMode.None))
                {
                    if (leftHand == null && h.GetHand() == OVRPlugin.Hand.HandLeft) leftHand = h;
                    else if (rightHand == null && h.GetHand() == OVRPlugin.Hand.HandRight) rightHand = h;
                }
            }
            if (target == null && showOwnCanvas) BuildCanvas();
        }

        private static bool IsMiddlePinching(OVRHand hand)
            => hand != null && hand.IsTracked && !hand.IsSystemGestureInProgress
               && hand.GetFingerIsPinching(OVRHand.HandFinger.Middle)
               && hand.GetFingerConfidence(OVRHand.HandFinger.Middle) == OVRHand.TrackingConfidence.High;

        private void BuildCanvas()
        {
            var go = new GameObject("MegamindHud", typeof(Canvas), typeof(CanvasScaler));
            go.layer = gameObject.layer;
            var canvas = go.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.sortingOrder = 100;
            var rt = go.GetComponent<RectTransform>();
            const float pxPerMeter = 1000f;
            rt.sizeDelta = new Vector2(widthMeters * pxPerMeter, 0.17f * pxPerMeter);
            rt.localScale = Vector3.one / pxPerMeter;
            if (head != null)
            {
                rt.SetParent(head, false);
                rt.localPosition = localOffset;
                rt.localRotation = Quaternion.identity;
            }

            var bg = new GameObject("bg", typeof(Image)).GetComponent<Image>();
            bg.transform.SetParent(go.transform, false);
            bg.color = new Color(0f, 0f, 0f, 0.55f);
            Stretch(bg.rectTransform);

            _dot = new GameObject("rec", typeof(Image)).GetComponent<Image>();
            _dot.transform.SetParent(go.transform, false);
            _dot.color = Color.red;
            var drt = _dot.rectTransform;
            drt.anchorMin = drt.anchorMax = new Vector2(0f, 1f);
            drt.pivot = new Vector2(0f, 1f);
            drt.anchoredPosition = new Vector2(12f, -12f);
            drt.sizeDelta = new Vector2(16f, 16f);

            target = new GameObject("text", typeof(Text)).GetComponent<Text>();
            target.transform.SetParent(go.transform, false);
            target.font = LoadFont();
            target.fontSize = fontSize;
            target.color = Color.white;
            target.alignment = TextAnchor.UpperLeft;
            target.horizontalOverflow = HorizontalWrapMode.Wrap;
            target.verticalOverflow = VerticalWrapMode.Overflow;
            Stretch(target.rectTransform, new Vector2(36f, 8f), new Vector2(-8f, -8f));
        }

        private static void Stretch(RectTransform r, Vector2 min = default, Vector2 max = default)
        {
            r.anchorMin = Vector2.zero;
            r.anchorMax = Vector2.one;
            r.offsetMin = min;
            r.offsetMax = max;
        }

        private static Font LoadFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }

        /// <summary>True while passthrough frames are being captured and uploaded (drives the REC dot).</summary>
        public bool IsPaused => uploader != null && uploader.CurrentState == FrameUploader.State.Paused;
        /// <summary>Short state word for the footer: "camera starting…", "camera permission…", "PAUSED (background)"…</summary>
        public string StateLabel => uploader == null ? "no FrameUploader" : StateText(uploader);

        public bool IsCapturing => uploader != null && uploader.CurrentState == FrameUploader.State.Capturing;

        /// <summary>Blink phase for the REC dot (on 70 % of each second); false when not capturing.</summary>
        public bool RecDotOn => IsCapturing && (Time.unscaledTime % 1f) < 0.7f;

        /// <summary>Toggle capture (same as A/X / middle pinch). Bound to the panel's Record/Pause control and voice.</summary>
        public void ToggleCapture() { if (uploader != null) uploader.ToggleCapturing(); }

        public void SetCapturing(bool on) { if (uploader != null) uploader.SetCapturing(on); }

        private void Update()
        {
            bool middlePinch = IsMiddlePinching(leftHand) || IsMiddlePinching(rightHand);
            bool middlePinchDown = middlePinch && !_middlePinchHeld;
            _middlePinchHeld = middlePinch;
            if (toggleWithButton && uploader != null &&
                (OVRInput.GetDown(OVRInput.Button.One) || OVRInput.GetDown(OVRInput.Button.Three) || middlePinchDown))
                uploader.ToggleCapturing();

            if (target == null || Time.unscaledTime < _next) return;
            _next = Time.unscaledTime + refreshSeconds;
            target.text = FullReport();
            if (_dot != null) _dot.enabled = RecDotOn;
        }

        /// <summary>One line for the chat panel header, e.g. "REC 1 fps · 143 sent · mic 9 · url:intent".</summary>
        static string StateText(FrameUploader u)
        {
            var cfg = u.settings;
            return u.CurrentState switch
            {
                FrameUploader.State.Capturing => $"REC {cfg.captureFps:0.#} fps" + (u.AppFocused ? "" : " (unfocused)"),
                FrameUploader.State.WaitingForPermission => "camera permission…",
                FrameUploader.State.WaitingForCamera => "camera starting…",
                FrameUploader.State.Paused => u.PauseReason == "background" ? "PAUSED (background)" : "PAUSED",
                FrameUploader.State.Unsupported => "camera unsupported",
                FrameUploader.State.NoCameraComponent => "no camera component",
                _ => u.CurrentState.ToString(),
            };
        }

        public string ShortStatus()
        {
            var u = uploader;
            if (u == null) return "no FrameUploader";
            var cfg = u.settings;
            string state = StateText(u);
            string sent = u.Accepted == u.Sent ? $"{u.Sent} sent" : $"{u.Accepted}/{u.Sent} sent";
            string mic = audioRecorder != null ? $" · mic {audioRecorder.Sent}" : "";
            string net = u.Failed > 0 && !u.ServerReachable ? " · NET ERR" : "";
            return $"{state} · {sent}{mic} · url:{cfg.overrideSource}{net}";
        }

        /// <summary>Everything the old HUD showed, as multi-line text for the Debug view.</summary>
        public string FullReport()
        {
            var u = uploader;
            if (u == null) return "Megamind: no FrameUploader";
            var res = u.CameraResolution;
            var cfg = u.settings;
            string line1 = u.CurrentState switch
            {
                FrameUploader.State.Capturing => $"REC {cfg.captureFps:0.#} fps  {res.x}x{res.y}  head {u.HeadDegPerSec:0}°/s" + (u.AppFocused ? "" : "  (unfocused)"),
                FrameUploader.State.WaitingForPermission => "waiting for camera permission…",
                FrameUploader.State.WaitingForCamera => "starting passthrough camera…",
                FrameUploader.State.Paused => u.PauseReason switch
                {
                    "background" => "capture PAUSED (app backgrounded)",
                    "focus lost" => "capture PAUSED (focus lost: system menu / boundary)",
                    _ => "capture PAUSED (press A/X or middle-pinch)",
                },
                FrameUploader.State.Unsupported => "Passthrough Camera API unsupported (Quest 3 + v74 required)",
                FrameUploader.State.NoCameraComponent => "no PassthroughCameraAccess in scene",
                _ => u.CurrentState.ToString(),
            };
            string net = u.Failed > 0 && !u.ServerReachable ? $"  NET ERR: {u.LastError}" : "";
            return
                $"{line1}\n" +
                $"sent {u.Sent}  accepted {u.Accepted}  last {u.LastReason}  {u.LastLatencyMs:0} ms  {u.LastJpegBytes / 1024} KB" +
                (u.SkippedMotion + u.SkippedBacklog + u.SkippedBackoff > 0 ? $"  skipped m{u.SkippedMotion}/b{u.SkippedBacklog}/r{u.SkippedBackoff}" : "") +
                (u.BackgroundCount + u.FocusLossCount > 0 ? $"  bg {u.BackgroundCount} unfocus {u.FocusLossCount}" : "") + "\n" +
                MicLine() +
                $"{u.ServerBaseUrl}  [{u.DeviceId}, url:{cfg.overrideSource}]{net}";
        }

        private string MicLine()
        {
            var a = audioRecorder;
            if (a == null) return "";
            string state = a.CurrentState switch
            {
                AmbientAudioRecorder.State.Off => "off",
                AmbientAudioRecorder.State.WaitingForPermission => "waiting for mic permission",
                AmbientAudioRecorder.State.NoMicrophone => "no microphone",
                AmbientAudioRecorder.State.Paused => "paused",
                AmbientAudioRecorder.State.PushToTalk => "push-to-talk",
                _ => $"on {a.LastLevelDb:0} dB",
            };
            string extra = (a.DroppedSilent > 0 ? $"  silent {a.DroppedSilent}" : "") +
                           (a.Queued > 1 ? $"  queued {a.Queued}" : "") +
                           (a.Failed > 0 && !a.ServerReachable ? $"  ERR {a.LastError}" : "");
            return $"mic: chunks sent {a.Sent}  accepted {a.Accepted}  ({state}){extra}\n";
        }
    }
}
