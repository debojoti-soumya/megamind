// Small head-locked status line: capture state, accepted/sent, last server reason, latency, server URL.
// Also the visible "recording" indicator (privacy: capture is never silent).
// Builds its own world-space Canvas at runtime so no prefab is needed; assign `target` to use your own Text instead.
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class CaptureHud : MonoBehaviour
    {
        public FrameUploader uploader;
        [Tooltip("CenterEyeAnchor; the HUD is parented here. Found automatically when empty.")]
        public Transform head;
        [Tooltip("Optional existing UI Text. When empty a canvas is created under `head`.")]
        public Text target;
        public Vector3 localOffset = new Vector3(0f, -0.22f, 1.0f);
        public float widthMeters = 0.55f;
        public int fontSize = 22;
        public float refreshSeconds = 0.25f;
        [Tooltip("A/X button or either hand's middle-finger pinch toggles capture on/off.")]
        public bool toggleWithButton = true;

        private float _next;
        private Image _dot;

        private void Start()
        {
            if (uploader == null) uploader = FindAnyObjectByType<FrameUploader>();
            if (head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) head = rig.centerEyeAnchor != null ? rig.centerEyeAnchor : rig.transform;
            }
            if (target == null) BuildCanvas();
        }

        private void BuildCanvas()
        {
            var go = new GameObject("MegamindHud", typeof(Canvas), typeof(CanvasScaler));
            go.layer = gameObject.layer;
            var canvas = go.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.sortingOrder = 100;
            var rt = go.GetComponent<RectTransform>();
            const float pxPerMeter = 1000f;
            rt.sizeDelta = new Vector2(widthMeters * pxPerMeter, 0.14f * pxPerMeter);
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

        private void Update()
        {
            if (toggleWithButton && uploader != null && (OVRInput.GetDown(OVRInput.Button.One) || OVRInput.GetDown(OVRInput.Button.Three)))
                uploader.ToggleCapturing();

            if (Time.unscaledTime < _next || target == null) return;
            _next = Time.unscaledTime + refreshSeconds;
            if (uploader == null)
            {
                target.text = "Megamind: no FrameUploader";
                return;
            }
            var u = uploader;
            var res = u.CameraResolution;
            var cfg = u.settings;
            string line1 = u.CurrentState switch
            {
                FrameUploader.State.Capturing => $"REC {cfg.captureFps:0.#} fps  {res.x}x{res.y}  head {u.HeadDegPerSec:0}°/s",
                FrameUploader.State.WaitingForPermission => "waiting for camera permission…",
                FrameUploader.State.WaitingForCamera => "starting passthrough camera…",
                FrameUploader.State.Paused => "capture PAUSED (press A/X)",
                FrameUploader.State.Unsupported => "Passthrough Camera API unsupported (Quest 3 + v74 required)",
                FrameUploader.State.NoCameraComponent => "no PassthroughCameraAccess in scene",
                _ => u.CurrentState.ToString(),
            };
            string net = u.Failed > 0 && !u.ServerReachable ? $"  NET ERR: {u.LastError}" : "";
            target.text =
                $"{line1}\n" +
                $"sent {u.Sent}  accepted {u.Accepted}  last {u.LastReason}  {u.LastLatencyMs:0} ms  {u.LastJpegBytes / 1024} KB" +
                (u.SkippedMotion + u.SkippedBacklog > 0 ? $"  skipped m{u.SkippedMotion}/b{u.SkippedBacklog}" : "") + "\n" +
                $"{u.ServerBaseUrl}  [{u.DeviceId}, url:{cfg.overrideSource}]{net}";
            if (_dot != null)
            {
                bool rec = u.CurrentState == FrameUploader.State.Capturing;
                _dot.enabled = rec && (Time.unscaledTime % 1f) < 0.7f;
            }
        }
    }
}
