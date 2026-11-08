// Right-side drawer: slides DOWN the right edge of the panel (~35 % of its width) with a short ease instead of popping,
// and rotates the corner tab's chevron. PanelMenu owns what is inside (sessions list + menu actions); this only animates.
// The drawer sits under a RectMask2D so the part still above the panel is clipped while it moves.
using System.Collections;
using UnityEngine;

namespace Megamind
{
    public class PanelDrawer : MonoBehaviour
    {
        public RectTransform drawer;        // anchored to the panel's top-right, slides along y
        public CanvasGroup group;           // fades + blocks rays only when open
        public GameObject backdrop;         // transparent full-panel raycast target behind the drawer; click = close
        public RectTransform chevron;       // on the corner tab, rotates 180° while open
        [Tooltip("Slide time in seconds (ease-out).")]
        public float duration = 0.15f;
        [Tooltip("How far above its resting place the drawer starts (canvas px). Default = panel height.")]
        public float hiddenOffset = 720f;
        [Tooltip("Deactivate the drawer object while closed (off for the whole window: its children keep running).")]
        public bool deactivateWhenClosed = true;
        [Tooltip("PlayerPrefs key: when set, open/closed is remembered across relaunch (default open).")]
        public string prefsKey = "";

        public bool IsOpen { get; private set; }
        public bool IsAnimating => _anim != null;

        private float _t; // 0 = closed, 1 = open
        private Coroutine _anim;

        private void Awake()
        {
            IsOpen = !string.IsNullOrEmpty(prefsKey) && PlayerPrefs.GetInt(prefsKey, 1) == 1;
            Apply(IsOpen ? 1f : 0f);
            if (drawer) drawer.gameObject.SetActive(IsOpen || !deactivateWhenClosed);
            if (backdrop) backdrop.SetActive(IsOpen);
        }

        private void Save()
        {
            if (string.IsNullOrEmpty(prefsKey)) return;
            PlayerPrefs.SetInt(prefsKey, IsOpen ? 1 : 0);
            PlayerPrefs.Save();
        }

        public void Toggle() { if (IsOpen) Close(); else Open(); }

        public void Open()
        {
            if (IsOpen) return;
            IsOpen = true;
            if (drawer) drawer.gameObject.SetActive(true);
            if (backdrop) backdrop.SetActive(true);
            Save();
            Animate(1f);
        }

        public void Close()
        {
            if (!IsOpen) return;
            IsOpen = false;
            if (backdrop) backdrop.SetActive(false);
            Save();
            Animate(0f);
        }

        private void Animate(float target)
        {
            if (_anim != null) StopCoroutine(_anim);
            if (!isActiveAndEnabled) { Apply(target); if (target <= 0f && drawer && deactivateWhenClosed) drawer.gameObject.SetActive(false); return; }
            _anim = StartCoroutine(Run(target));
        }

        private IEnumerator Run(float target)
        {
            float from = _t, t0 = Time.unscaledTime;
            float dur = Mathf.Max(0.01f, duration) * Mathf.Abs(target - from);
            while (Time.unscaledTime - t0 < dur)
            {
                float k = Mathf.Clamp01((Time.unscaledTime - t0) / dur);
                k = 1f - (1f - k) * (1f - k) * (1f - k); // ease-out cubic
                Apply(Mathf.Lerp(from, target, k));
                yield return null;
            }
            Apply(target);
            if (target <= 0f && drawer && deactivateWhenClosed) drawer.gameObject.SetActive(false);
            _anim = null;
        }

        private void Apply(float t)
        {
            _t = t;
            if (drawer) drawer.anchoredPosition = new Vector2(drawer.anchoredPosition.x, (1f - t) * hiddenOffset);
            if (group)
            {
                group.alpha = t;
                group.interactable = group.blocksRaycasts = t > 0.5f;
            }
            if (chevron) chevron.localRotation = Quaternion.Euler(0f, 0f, Mathf.Lerp(0f, 180f, t));
        }

        private void OnDisable()
        {
            // Panel hidden / destroyed mid-slide: land in the final state without a coroutine.
            _anim = null;
            Apply(IsOpen ? 1f : 0f);
            if (!IsOpen && drawer && deactivateWhenClosed) drawer.gameObject.SetActive(false);
        }
    }
}
