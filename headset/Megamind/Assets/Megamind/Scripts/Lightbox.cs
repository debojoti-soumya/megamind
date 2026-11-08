// Full-size viewer for a cited / candidate memory frame, inside the panel (≥ 80 % of its width). Opens from a
// CitationCard tap (controller ray or hand pinch), shows the server image (thumb_url: the stored keyframe, i.e. the
// full 1280x960 upload — the server has no larger still; replay_url is a video clip), caption, time and "n / N",
// Prev / Next across the set, pan / zoom with the thumbstick (right stick: zoom, left stick: pan) or by dragging the
// image with the ray / pinch, and +/- buttons. Close: X, click outside the image, "close" by voice (PanelMenu.CloseAll).
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace Megamind
{
    public class Lightbox : MonoBehaviour, IBeginDragHandler, IDragHandler, IScrollHandler
    {
        public MegamindConfig config;
        public GameObject root;             // whole overlay (inactive when closed)
        public RectTransform viewport;      // clipping rect the image pans inside
        public RawImage image;              // centred, resized to fit the viewport x zoom
        public Text caption;
        public Text timeLabel;
        public Text counter;                // "2 / 3"
        public Button prev, next;
        public Color placeholder = MegamindTheme.ThumbPlaceholder;
        public float minZoom = 1f, maxZoom = 5f;
        [Tooltip("Thumbstick zoom speed (x per second at full deflection).")]
        public float stickZoomPerSecond = 1.6f;
        [Tooltip("Thumbstick pan speed (canvas px per second at full deflection, at 1x).")]
        public float stickPanPerSecond = 700f;
        public OVRInput.Controller controllers = OVRInput.Controller.Touch;

        public bool IsOpen => root != null && root.activeSelf;

        private List<Citation> _items = new List<Citation>();
        private int _index;
        private float _zoom = 1f;
        private Vector2 _pan;
        private Vector2 _fitSize;
        private Coroutine _load;

        private void Awake() { if (root) root.SetActive(false); }

        public void Show(List<Citation> items, int index)
        {
            if (items == null || items.Count == 0) return;
            _items = items;
            _index = Mathf.Clamp(index, 0, items.Count - 1);
            if (root) root.SetActive(true);
            Load();
        }

        public void Close()
        {
            if (_load != null) { StopCoroutine(_load); _load = null; }
            if (root) root.SetActive(false);
        }

        public void Prev() { if (_items.Count == 0) return; _index = (_index - 1 + _items.Count) % _items.Count; Load(); }
        public void Next() { if (_items.Count == 0) return; _index = (_index + 1) % _items.Count; Load(); }
        public void ZoomIn() => SetZoom(_zoom * 1.5f);
        public void ZoomOut() => SetZoom(_zoom / 1.5f);

        private void Load()
        {
            var c = _items[_index];
            _zoom = 1f; _pan = Vector2.zero;
            if (counter) counter.text = (_index + 1) + " / " + _items.Count;
            if (timeLabel) timeLabel.text = ChatPanel.Ago(c.ts) + "  \u00b7  " + ChatPanel.LocalTime(c.ts);
            if (caption) caption.text = string.IsNullOrEmpty(c.caption) ? "" : c.caption;
            if (prev) prev.gameObject.SetActive(_items.Count > 1);
            if (next) next.gameObject.SetActive(_items.Count > 1);
            if (image)
            {
                image.texture = null; image.color = placeholder;
                _fitSize = viewport ? viewport.rect.size : new Vector2(512f, 384f);
                Apply();
            }
            if (_load != null) StopCoroutine(_load);
            if (!string.IsNullOrEmpty(c.thumb_url) && config != null) _load = StartCoroutine(LoadImage(config.Url(c.thumb_url)));
        }

        private IEnumerator LoadImage(string url)
        {
            using (var req = UnityWebRequestTexture.GetTexture(url))
            {
                config.Authorize(req);
                req.timeout = 15;
                yield return req.SendWebRequest();
                _load = null;
                if (image == null) yield break;
                if (req.result != UnityWebRequest.Result.Success) { image.color = new Color(0.30f, 0.16f, 0.18f); yield break; }
                var tex = DownloadHandlerTexture.GetContent(req);
                image.texture = tex; image.color = Color.white;
                // fit inside the viewport, keeping the frame's aspect
                var vp = viewport ? viewport.rect.size : new Vector2(512f, 384f);
                float s = Mathf.Min(vp.x / tex.width, vp.y / tex.height);
                _fitSize = new Vector2(tex.width * s, tex.height * s);
                Apply();
            }
        }

        private void SetZoom(float z)
        {
            _zoom = Mathf.Clamp(z, minZoom, maxZoom);
            Apply();
        }

        private void Apply()
        {
            if (image == null) return;
            var rt = image.rectTransform;
            rt.sizeDelta = _fitSize * _zoom;
            // keep the image covering the viewport: clamp the pan to the overflow
            var vp = viewport ? viewport.rect.size : _fitSize;
            var max = Vector2.Max((rt.sizeDelta - vp) * 0.5f, Vector2.zero);
            _pan = new Vector2(Mathf.Clamp(_pan.x, -max.x, max.x), Mathf.Clamp(_pan.y, -max.y, max.y));
            rt.anchoredPosition = _pan;
        }

        private void Update()
        {
            if (!IsOpen) return;
            var zoomStick = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick, controllers); // right controller
            var panStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, controllers);    // left controller
            if (Mathf.Abs(zoomStick.y) > 0.2f) SetZoom(_zoom * Mathf.Pow(stickZoomPerSecond, zoomStick.y * Time.unscaledDeltaTime));
            if (panStick.sqrMagnitude > 0.04f)
            {
                _pan -= panStick * (stickPanPerSecond * Time.unscaledDeltaTime); // push right = look right (image moves left)
                Apply();
            }
            if (OVRInput.GetDown(OVRInput.Button.Two, controllers)) Close(); // B / Y = back
        }

        // Drag the picture with the ray (trigger held) or a pinch: pan.
        public void OnBeginDrag(PointerEventData e) {}
        public void OnDrag(PointerEventData e)
        {
            if (viewport == null) return;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(viewport, e.position, e.pressEventCamera, out var now);
            RectTransformUtility.ScreenPointToLocalPointInRectangle(viewport, e.position - e.delta, e.pressEventCamera, out var before);
            _pan += now - before;
            Apply();
        }

        public void OnScroll(PointerEventData e) => SetZoom(_zoom * (1f + 0.1f * Mathf.Sign(e.scrollDelta.y)));
    }
}
