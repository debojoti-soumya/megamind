// Footer "Hold to talk": pointer down (trigger / pinch while the ray is on the button) starts the same push-to-talk
// path as the controller trigger, pointer up sends (uGUI delivers pointer-up to the pressed object even if the ray
// drifted off it). ChatClient also ends the recording on trigger/pinch release, so whichever event comes first wins
// (EndRecordingAndSendNow is a no-op when nothing is recording).
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace Megamind
{
    public class HoldToTalkButton : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
    {
        public ChatClient client;
        public Image background;
        public Color idleColor = new Color(0.16f, 0.19f, 0.24f);
        public Color activeColor = new Color(0.85f, 0.25f, 0.25f);

        private bool _held;

        public void OnPointerDown(PointerEventData e)
        {
            if (client == null) return;
            _held = true;
            client.BeginRecording();
        }

        public void OnPointerUp(PointerEventData e) => Release();

        private void Release()
        {
            _held = false;
            if (client != null) client.EndRecordingAndSendNow();
        }

        private void Update()
        {
            if (background) background.color = client != null && client.IsRecording ? activeColor : idleColor;
        }

        private void OnDisable() { if (_held) Release(); }
    }
}
