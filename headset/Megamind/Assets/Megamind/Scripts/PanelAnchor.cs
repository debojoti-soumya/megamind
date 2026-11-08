// Stream 3: keeps the chat Canvas in front of the wearer.
//
// Modes (field `pinned`, button `TogglePinned`):
//   follow (default) - lazy "tag-along": the panel rests where it is while the head stays inside a dead zone
//                      (`deadZoneDegrees` of yaw / `deadZoneMeters` of travel); once the wearer looks or walks past
//                      it the panel glides (lerp, never snaps) back to eye height, `panelDistance` m ahead.
//   pinned           - world-locked where it was last placed.
// In both modes holding the re-anchor button (left Y / right B) for `holdSeconds` drags the panel with the gaze;
// release leaves it there. `Snap()` places it instantly (UI button).
//
// Eye height: the target is CenterEyeAnchor.position (+ `eyeHeightOffset`), never the rig/floor origin. With the
// FloorLevel tracking origin the very first frames report the head at the rig origin (y = 0) until tracking
// arrives, which put the old one-shot Snap() at floor level ("near my legs"); the first placement now waits for a
// plausible head pose (`minHeadHeight`, with a timeout for EyeLevel origins).
//
// UI scale: `uiScale` (1x / 1.25x / 1.5x / 2x, PlayerPrefs "megamind.uiScale") multiplies the canvas scale so the
// panel, its text and its thumbnails all grow together; `ScaleUp` / `ScaleDown` are bound to the A- / A+ buttons.
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class PanelAnchor : MonoBehaviour
    {
        public const string UiScalePrefKey = "megamind.uiScale";
        public static readonly float[] UiScaleSteps = { 1f, 1.25f, 1.5f, 2f };

        public MegamindConfig config;
        public Transform head;              // OVRCameraRig CenterEyeAnchor
        [Tooltip("World-locked (true) or lazy follow (false). Toggled by the Pin button / voice.")]
        public bool pinned = false;
        public OVRInput.Button reanchorButton = OVRInput.Button.Two; // B (right) / Y (left)
        public OVRInput.Controller controllers = OVRInput.Controller.Touch;
        public float holdSeconds = 0.5f;
        [Tooltip("Lerp speed while the button is held (direct drag).")]
        public float followLerp = 12f;
        [Tooltip("Lerp speed while tag-along recentres the panel.")]
        public float recenterLerp = 3.5f;
        [Header("Tag-along dead zone")]
        [Tooltip("Head yaw away from the panel centre before it recentres.")]
        public float deadZoneDegrees = 25f;
        [Tooltip("Head travel (horizontal) away from the panel's resting spot before it recentres.")]
        public float deadZoneMeters = 0.5f;
        [Tooltip("Eye-height change before the panel recentres vertically.")]
        public float deadZoneHeightMeters = 0.25f;
        [Tooltip("The panel is pushed back out when the wearer gets closer than this.")]
        public float minDistance = 0.55f;
        [Tooltip("Panel centre relative to eye height (negative = a little below the eye line).")]
        public float eyeHeightOffset = -0.08f;
        [Tooltip("Keep the panel upright (ignore head roll/pitch) so text stays readable.")]
        public bool keepUpright = true;
        [Header("First placement")]
        [Tooltip("Ignore head poses lower than this above the rig origin (FloorLevel origin reports y=0 before tracking starts).")]
        public float minHeadHeight = 0.3f;
        [Tooltip("Place anyway after this many seconds (EyeLevel tracking origin keeps the head near y=0).")]
        public float placeTimeoutSeconds = 2f;
        [Tooltip("Optional caption of the Pin button; shows Pin / Unpin.")]
        public Text pinLabel;
        [Header("UI scale")]
        public float uiScale = 1f;

        public bool IsFollowing { get; private set; }     // button held: dragging with the gaze
        public bool IsRecentering { get; private set; }   // tag-along glide in progress

        private float _heldSince = -1f;
        private bool _placed;
        private float _startTime;
        private RectTransform _rt;
        private float _baseScale = 1f;

        private void Awake()
        {
            _rt = GetComponent<RectTransform>();
            uiScale = PlayerPrefs.GetFloat(UiScalePrefKey, uiScale);
        }

        private void Start()
        {
            if (head == null && Camera.main != null) head = Camera.main.transform;
            _startTime = Time.unscaledTime;
            float width = config != null ? config.panelWidthMeters : 0.6f;
            // scale canvas units (px) to metres so the panel is `panelWidthMeters` wide (times uiScale)
            _baseScale = _rt != null && _rt.rect.width > 0f ? width / _rt.rect.width : transform.localScale.x;
            ApplyScale();
            RefreshPinLabel();
        }

        private void LateUpdate()
        {
            if (head == null) return;
            if (!_placed)
            {
                if (!HeadPoseReady() && Time.unscaledTime - _startTime < placeTimeoutSeconds) return;
                Snap();
                return;
            }

            bool held = OVRInput.Get(reanchorButton, controllers);
            if (held)
            {
                if (_heldSince < 0f) _heldSince = Time.unscaledTime;
                if (Time.unscaledTime - _heldSince >= holdSeconds) IsFollowing = true;
            }
            else
            {
                _heldSince = -1f;
                IsFollowing = false;
            }

            if (IsFollowing)
            {
                IsRecentering = false;
                MoveTowards(TargetPosition(), followLerp);
                return;
            }

            if (pinned) { IsRecentering = false; return; }

            Vector3 target = TargetPosition();
            if (!IsRecentering && OutsideDeadZone(target)) IsRecentering = true;
            if (IsRecentering)
            {
                MoveTowards(target, recenterLerp);
                if ((transform.position - target).sqrMagnitude < 0.02f * 0.02f) IsRecentering = false;
            }
            KeepOutOfFace();
        }

        private bool HeadPoseReady()
        {
            // CenterEyeAnchor is a child of TrackingSpace; before tracking arrives it sits at the local origin.
            if (head.localPosition.sqrMagnitude < 1e-6f) return false;
            return head.localPosition.y >= minHeadHeight;
        }

        private bool OutsideDeadZone(Vector3 target)
        {
            Vector3 toPanel = transform.position - head.position;
            Vector3 fwd = FlatForward();
            Vector3 flat = new Vector3(toPanel.x, 0f, toPanel.z);
            if (flat.sqrMagnitude > 1e-4f && Vector3.Angle(fwd, flat) > deadZoneDegrees) return true;
            Vector3 d = transform.position - target;
            if (new Vector2(d.x, d.z).magnitude > deadZoneMeters) return true;
            if (Mathf.Abs(d.y) > deadZoneHeightMeters) return true;
            return flat.magnitude < minDistance;
        }

        private void MoveTowards(Vector3 target, float lerp)
        {
            float k = 1f - Mathf.Exp(-lerp * Time.unscaledDeltaTime);
            transform.position = Vector3.Lerp(transform.position, target, k);
            transform.rotation = Quaternion.Slerp(transform.rotation, FacingRotation(transform.position), k);
        }

        private void KeepOutOfFace()
        {
            Vector3 toPanel = transform.position - head.position;
            Vector3 flat = new Vector3(toPanel.x, 0f, toPanel.z);
            if (flat.sqrMagnitude < 1e-6f || flat.magnitude >= minDistance) return;
            Vector3 pushed = head.position + flat.normalized * minDistance;
            pushed.y = transform.position.y;
            transform.position = Vector3.Lerp(transform.position, pushed, 1f - Mathf.Exp(-recenterLerp * Time.unscaledDeltaTime));
        }

        /// <summary>Immediately place the panel in front of the head (also bound to a UI button).</summary>
        public void Snap()
        {
            if (head == null) return;
            Vector3 pos = TargetPosition();
            transform.SetPositionAndRotation(pos, FacingRotation(pos));
            _placed = true;
            IsRecentering = false;
        }

        public void TogglePinned() => SetPinned(!pinned);

        public void SetPinned(bool value)
        {
            pinned = value;
            if (!pinned) IsRecentering = false; // the dead-zone check decides on the next frame
            RefreshPinLabel();
        }

        private void RefreshPinLabel() { if (pinLabel) pinLabel.text = pinned ? "Unpin" : "Pin"; }

        // ---------- UI scale ----------

        public void ScaleUp() => SetUiScale(NextScale(+1));
        public void ScaleDown() => SetUiScale(NextScale(-1));
        /// <summary>1x -> 1.25x -> 1.5x -> 2x -> 1x (menu "Text size" entry).</summary>
        public void CycleScale() => SetUiScale(NearestStep() == UiScaleSteps.Length - 1 ? UiScaleSteps[0] : UiScaleSteps[NearestStep() + 1]);
        public string UiScaleLabel => uiScale.ToString("0.##") + "x";

        public void SetUiScale(float scale)
        {
            uiScale = Mathf.Clamp(scale, UiScaleSteps[0], UiScaleSteps[UiScaleSteps.Length - 1]);
            PlayerPrefs.SetFloat(UiScalePrefKey, uiScale);
            PlayerPrefs.Save();
            ApplyScale();
        }

        private float NextScale(int dir) => UiScaleSteps[Mathf.Clamp(NearestStep() + dir, 0, UiScaleSteps.Length - 1)];

        private int NearestStep()
        {
            int best = 0;
            for (int i = 0; i < UiScaleSteps.Length; i++)
                if (Mathf.Abs(UiScaleSteps[i] - uiScale) < Mathf.Abs(UiScaleSteps[best] - uiScale)) best = i;
            return best;
        }

        private void ApplyScale() => transform.localScale = Vector3.one * (_baseScale * uiScale);

        // ---------- geometry ----------

        private Vector3 FlatForward()
        {
            Vector3 fwd = head.forward;
            if (keepUpright) fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
            return fwd.normalized;
        }

        private Vector3 TargetPosition()
        {
            float d = config != null ? config.panelDistance : 1f;
            if (!keepUpright) return head.position + head.forward * d;
            return head.position + FlatForward() * d + Vector3.up * eyeHeightOffset;
        }

        private Quaternion FacingRotation(Vector3 panelPos)
        {
            // Unity UI canvases face -Z by default, so look away from the head
            Vector3 away = panelPos - head.position;
            if (keepUpright) away.y = 0f;
            if (away.sqrMagnitude < 1e-6f) away = FlatForward();
            return Quaternion.LookRotation(away.normalized, Vector3.up);
        }
    }
}
