// "Where is my pen?" -> a world-locked arrow pointing down at the pen. For "where is X" questions the server returns
// ChatResponse.markers: X's box in a frame this headset uploaded (normalized, origin top-left) plus that frame's ts.
// FrameUploader remembers the camera pose of every uploaded frame, so the box centre becomes a world ray from where
// the camera was (PassthroughCameraAccess.ViewportPointToRay), hit-tested against the real room
// (EnvironmentRaycastManager = Depth API; needs com.oculus.permission.USE_SCENE, see MegamindPermissions).
// No hit yet (depth not ready, or that spot is out of view now): the pin waits on the ray at fallbackDistance and
// re-casts until it lands. The ring under the arrow is sized to the box, so a loose box reads as "around here".
// When the marker is outside the view (behind you, off to a side), a head-locked arrow at the edge of the view points
// toward it.
// Self-installs at scene load (next to FrameUploader) so scenes built before this component get it too.
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class ObjectMarkers : MonoBehaviour
    {
        public ChatClient chat;
        public FrameUploader uploader;
        public PassthroughCameraAccess cameraAccess;
        public EnvironmentRaycastManager raycast;
        public Transform head;
        public float fallbackDistance = 1.0f;
        public float maxDistance = 6f;
        public float lifetimeSeconds = 45f;
        public float retrySeconds = 0.25f;
        public float minRingDiameter = 0.06f; // metres
        public Color color = new Color(1f, 0.8f, 0.1f, 1f);
        public float bobAmplitude = 12f; // mm
        public float guideAngle = 30f; // degrees off view centre before the edge arrow appears
        public float guideRadius = 0.18f; // metres from view centre, at guideDistance
        public float guideDistance = 0.8f;

        public bool GuideVisible => _guide != null && _guide.activeSelf;

        public int ActivePins => _pins.Count;
        public string LastStatus { get; private set; } = "-";

        private class Pin
        {
            public GameObject root;
            public RectTransform ring, stem, arrow, label;
            public float arrowY;
            public Ray ray;
            public float halfAngleRad, until, nextTry;
            public bool landed;
        }

        private readonly List<Pin> _pins = new List<Pin>();
        private static Sprite s_ring, s_arrow;
        private GameObject _guide;
        private static Font s_font;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (FindAnyObjectByType<ObjectMarkers>(FindObjectsInactive.Include) != null) return;
            var up = FindAnyObjectByType<FrameUploader>(FindObjectsInactive.Include);
            if (up == null) return;
            up.gameObject.AddComponent<ObjectMarkers>();
        }

        private void Start()
        {
            if (uploader == null) uploader = FindAnyObjectByType<FrameUploader>(FindObjectsInactive.Include);
            if (cameraAccess == null) cameraAccess = uploader != null && uploader.cameraAccess != null
                ? uploader.cameraAccess : FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
            if (head == null) head = uploader != null && uploader.headPose != null ? uploader.headPose : Camera.main?.transform;
            if (raycast == null) raycast = FindAnyObjectByType<EnvironmentRaycastManager>(FindObjectsInactive.Include);
            if (raycast == null && EnvironmentRaycastManager.IsSupported)
                raycast = new GameObject(nameof(EnvironmentRaycastManager)).AddComponent<EnvironmentRaycastManager>();
            if (chat == null) chat = FindAnyObjectByType<ChatClient>(FindObjectsInactive.Include);
            if (chat != null) chat.OnResponse += Show;
            else Debug.LogWarning("[Megamind] ObjectMarkers: no ChatClient in the scene; markers disabled");
        }

        private void OnDestroy()
        {
            if (chat != null) chat.OnResponse -= Show;
            Clear();
            if (_guide != null) Destroy(_guide);
        }

        public void Show(ChatResponse resp)
        {
            Clear();
            if (resp == null || resp.markers == null) return;
            foreach (var m in resp.markers) Place(m);
        }

        public void Clear()
        {
            foreach (var p in _pins) if (p.root != null) Destroy(p.root);
            _pins.Clear();
        }

        private void Place(Marker m)
        {
            if (uploader == null || cameraAccess == null || !cameraAccess.IsPlaying) { Status($"'{m.label}': camera not running"); return; }
            if (!string.IsNullOrEmpty(m.device_id) && m.device_id != uploader.DeviceId) { Status($"'{m.label}': frame from device {m.device_id}"); return; }
            if (!uploader.TryGetFramePose(m.ts, out var pose)) { Status($"'{m.label}': no camera pose for frame ts={m.ts:F3}"); return; }

            var ray = cameraAccess.ViewportPointToRay(Viewport(m.x, m.y), pose);
            float half = 0.02f;
            if (m.box != null && m.box.Count == 4)
            {
                float w = Vector3.Angle(cameraAccess.ViewportPointToRay(Viewport(m.box[0], m.y), pose).direction,
                                        cameraAccess.ViewportPointToRay(Viewport(m.box[2], m.y), pose).direction);
                float h = Vector3.Angle(cameraAccess.ViewportPointToRay(Viewport(m.x, m.box[1]), pose).direction,
                                        cameraAccess.ViewportPointToRay(Viewport(m.x, m.box[3]), pose).direction);
                half = Mathf.Clamp(0.5f * Mathf.Max(w, h) * Mathf.Deg2Rad, 0.005f, 0.35f);
            }
            var pin = new Pin { ray = ray, halfAngleRad = half, until = Time.time + lifetimeSeconds };
            Build(pin, string.IsNullOrEmpty(m.label) ? "here" : m.label);
            _pins.Add(pin);
            Position(pin, fallbackDistance);
            TryLand(pin);
            Status($"'{m.label}': {(pin.landed ? "on surface" : "on ray, waiting for depth")}");
        }

        // Server boxes are image coordinates (origin top-left); camera viewport is origin bottom-left.
        private static Vector2 Viewport(float x, float y) => new Vector2(Mathf.Clamp01(x), 1f - Mathf.Clamp01(y));

        private void Update()
        {
            for (int i = _pins.Count - 1; i >= 0; i--)
            {
                var p = _pins[i];
                if (p.root == null || Time.time > p.until)
                {
                    if (p.root != null) Destroy(p.root);
                    _pins.RemoveAt(i);
                    continue;
                }
                if (!p.landed && Time.time >= p.nextTry) TryLand(p);
                p.arrow.anchoredPosition = new Vector2(0f, p.arrowY + bobAmplitude * (0.5f + 0.5f * Mathf.Sin(Time.time * 4f)));
                if (head != null)
                {
                    var away = p.root.transform.position - head.position;
                    if (away.sqrMagnitude > 1e-6f) p.root.transform.rotation = Quaternion.LookRotation(away, Vector3.up);
                }
            }
            UpdateGuide();
        }

        private void UpdateGuide()
        {
            var target = _pins.Count > 0 && head != null ? _pins[0].root : null;
            var local = target != null ? head.InverseTransformPoint(target.transform.position) : Vector3.forward;
            bool show = target != null && Vector3.Angle(Vector3.forward, local) > guideAngle;
            if (!show) { if (_guide != null) _guide.SetActive(false); return; }
            if (_guide == null) _guide = BuildGuide();
            var dir = new Vector2(local.x, local.y);
            dir = dir.sqrMagnitude > 1e-6f ? dir.normalized : Vector2.right;
            _guide.SetActive(true);
            _guide.transform.SetPositionAndRotation(
                head.TransformPoint(new Vector3(dir.x * guideRadius, dir.y * guideRadius, guideDistance)),
                head.rotation * Quaternion.Euler(0f, 0f, Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg + 90f));
        }

        private GameObject BuildGuide()
        {
            var root = new GameObject("ObjectMarkers: guide");
            var canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.sortingOrder = 60;
            var rt = (RectTransform)root.transform;
            rt.sizeDelta = new Vector2(10f, 10f);
            rt.localScale = Vector3.one * 0.001f;
            var img = NewImage("Arrow", rt, color);
            img.sprite = ArrowSprite(); // points down (-Y); the root's roll aims it at the marker
            img.rectTransform.sizeDelta = new Vector2(40f, 40f);
            return root;
        }

        private void TryLand(Pin p)
        {
            p.nextTry = Time.time + retrySeconds;
            if (raycast == null || !raycast.isActiveAndEnabled || !EnvironmentRaycastManager.IsSupported) return;
            if (!raycast.Raycast(p.ray, out var hit, maxDistance)) return;
            Position(p, Vector3.Distance(p.ray.origin, hit.point));
            p.landed = true;
            Status("landed on surface");
        }

        private void Position(Pin p, float distance)
        {
            p.root.transform.position = p.ray.GetPoint(distance);
            float d = Mathf.Max(minRingDiameter, 2f * distance * Mathf.Tan(p.halfAngleRad)) * 1000f; // canvas units = mm
            p.ring.sizeDelta = new Vector2(d, d);
            p.stem.anchoredPosition = new Vector2(0f, d * 0.5f);
            p.arrowY = d * 0.5f + 6f;
            p.label.anchoredPosition = new Vector2(0f, p.arrowY + p.stem.sizeDelta.y + bobAmplitude);
        }

        private void Build(Pin p, string text)
        {
            var root = new GameObject("ObjectMarker: " + text);
            var canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.sortingOrder = 50;
            var rt = (RectTransform)root.transform;
            rt.sizeDelta = new Vector2(10f, 10f);
            rt.localScale = Vector3.one * 0.001f;
            p.root = root;

            var ring = NewImage("Ring", rt, color);
            ring.sprite = RingSprite();
            p.ring = ring.rectTransform;

            var stem = NewImage("Stem", rt, color);
            p.stem = stem.rectTransform;
            p.stem.pivot = new Vector2(0.5f, 0f);
            p.stem.sizeDelta = new Vector2(10f, 110f);
            p.stem.gameObject.SetActive(false); // spacer only: the arrow below is the visible stem

            var arrow = NewImage("Arrow", rt, color);
            arrow.sprite = ArrowSprite();
            p.arrow = arrow.rectTransform;
            p.arrow.pivot = new Vector2(0.5f, 0f);
            p.arrow.sizeDelta = new Vector2(60f, 110f);

            var bg = NewImage("Label", rt, new Color(0f, 0f, 0f, 0.65f));
            p.label = bg.rectTransform;
            p.label.pivot = new Vector2(0.5f, 0f);
            var fitter = bg.gameObject.AddComponent<HorizontalLayoutGroup>();
            fitter.padding = new RectOffset(14, 14, 6, 6);
            fitter.childForceExpandWidth = fitter.childForceExpandHeight = false;
            var size = bg.gameObject.AddComponent<ContentSizeFitter>();
            size.horizontalFit = size.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            var t = new GameObject("Text").AddComponent<Text>();
            t.transform.SetParent(p.label, false);
            t.font = LoadFont();
            t.fontSize = 32;
            t.color = color;
            t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.text = text;
        }

        private static Image NewImage(string name, RectTransform parent, Color c)
        {
            var img = new GameObject(name).AddComponent<Image>();
            img.transform.SetParent(parent, false);
            img.color = c;
            img.raycastTarget = false;
            return img;
        }

        private static Sprite RingSprite()
        {
            if (s_ring != null) return s_ring;
            const int n = 128;
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[n * n];
            float r0 = n * 0.5f - 1f, r1 = r0 - n * 0.07f, c = (n - 1) * 0.5f;
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    float d = Mathf.Sqrt((x - c) * (x - c) + (y - c) * (y - c));
                    float a = Mathf.Clamp01(r0 - d + 0.5f) * Mathf.Clamp01(d - r1 + 0.5f);
                    px[y * n + x] = new Color32(255, 255, 255, (byte)(a * 255));
                }
            tex.SetPixels32(px);
            tex.Apply(false);
            s_ring = Sprite.Create(tex, new Rect(0, 0, n, n), new Vector2(0.5f, 0.5f));
            return s_ring;
        }

        // Down-pointing arrow: shaft on top, head at the bottom tip (the tip is the sprite's bottom edge).
        private static Sprite ArrowSprite()
        {
            if (s_arrow != null) return s_arrow;
            const int w = 64, h = 128, headH = 56;
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[w * h];
            float c = (w - 1) * 0.5f;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float half = y < headH ? (w * 0.5f) * y / headH : w * 0.14f;
                    float a = Mathf.Clamp01(half - Mathf.Abs(x - c) + 0.5f);
                    px[y * w + x] = new Color32(255, 255, 255, (byte)(a * 255));
                }
            tex.SetPixels32(px);
            tex.Apply(false);
            s_arrow = Sprite.Create(tex, new Rect(0, 0, w, h), new Vector2(0.5f, 0f));
            return s_arrow;
        }

        private static Font LoadFont()
        {
            if (s_font != null) return s_font;
            s_font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (s_font == null) s_font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return s_font;
        }

        private void Status(string s)
        {
            LastStatus = s;
            Debug.Log("[Megamind] marker " + s);
        }
    }
}
