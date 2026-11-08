// Tracks whether any UI ray (controller or hand, OVRInputModule) is over this panel. uGUI sends pointer enter/exit
// up the hierarchy to the common root, so one handler on the panel root sees every child button/field/background.
// ChatClient uses it to ignore push-to-talk while the user is clicking the panel (trigger = click AND talk otherwise).
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

namespace Megamind
{
    public class PanelHover : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
    {
        [Tooltip("Keep reporting 'hovered' this long after the ray leaves, so a click on the panel edge never starts a recording.")]
        public float exitGraceSeconds = 0.15f;

        private readonly HashSet<int> _pointersInside = new HashSet<int>();
        private float _lastExit = -10f;

        public bool IsHovered => _pointersInside.Count > 0 || Time.unscaledTime - _lastExit < exitGraceSeconds;

        public void OnPointerEnter(PointerEventData e) => _pointersInside.Add(e.pointerId);

        public void OnPointerExit(PointerEventData e)
        {
            _pointersInside.Remove(e.pointerId);
            _lastExit = Time.unscaledTime;
        }

        private void OnDisable() => _pointersInside.Clear();
    }
}
