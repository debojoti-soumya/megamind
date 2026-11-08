// Stream 3: world-locks the chat Canvas 1 m ahead of the head on first use, and re-anchors it
// when the wearer holds the re-anchor button (default: left-hand Y / right-hand B) for 0.5 s.
// While the button is held the panel follows the gaze; on release it stays put.
using UnityEngine;

namespace Megamind
{
    public class PanelAnchor : MonoBehaviour
    {
        public MegamindConfig config;
        public Transform head;              // OVRCameraRig CenterEyeAnchor
        public OVRInput.Button reanchorButton = OVRInput.Button.Two; // B (right) / Y (left)
        public OVRInput.Controller controllers = OVRInput.Controller.Touch;
        public float holdSeconds = 0.5f;
        public float followLerp = 12f;
        [Tooltip("Keep the panel upright (ignore head roll/pitch) so text stays readable.")]
        public bool keepUpright = true;

        public bool IsFollowing { get; private set; }

        private float _heldSince = -1f;
        private bool _placed;

        private void Start()
        {
            if (head == null && Camera.main != null) head = Camera.main.transform;
            var canvas = GetComponent<Canvas>();
            if (canvas != null && config != null)
            {
                var rt = canvas.GetComponent<RectTransform>();
                // scale canvas units (px) to metres so the panel is `panelWidthMeters` wide
                if (rt.rect.width > 0f) rt.localScale = Vector3.one * (config.panelWidthMeters / rt.rect.width);
            }
        }

        private void LateUpdate()
        {
            if (head == null) return;
            if (!_placed)
            {
                Snap();
                _placed = true;
                return;
            }

            if (OVRInput.Get(reanchorButton, controllers))
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
                TargetPose(out var pos, out var rot);
                float k = 1f - Mathf.Exp(-followLerp * Time.unscaledDeltaTime);
                transform.position = Vector3.Lerp(transform.position, pos, k);
                transform.rotation = Quaternion.Slerp(transform.rotation, rot, k);
            }
        }

        /// <summary>Immediately place the panel in front of the head (also bound to a UI button).</summary>
        public void Snap()
        {
            TargetPose(out var pos, out var rot);
            transform.SetPositionAndRotation(pos, rot);
        }

        private void TargetPose(out Vector3 pos, out Quaternion rot)
        {
            float d = config != null ? config.panelDistance : 1f;
            Vector3 fwd = head.forward;
            if (keepUpright) { fwd.y = 0f; if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward; fwd.Normalize(); }
            pos = head.position + fwd * d;
            if (!keepUpright) pos = head.position + head.forward * d;
            // Unity UI canvases face -Z by default, so look away from the head
            rot = Quaternion.LookRotation(keepUpright ? fwd : (pos - head.position).normalized, Vector3.up);
        }
    }
}
