// Footer control bar + minimise: Record/Pause toggle (continuous capture via CaptureHud/FrameUploader), state text,
// Type (TouchScreenKeyboard -> ChatClient.SendText; typed short commands go through PanelMenu like spoken ones),
// Hide/Show (collapses the panel to the header strip so the REC dot stays visible).
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class PanelControls : MonoBehaviour
    {
        public CaptureHud hud;
        public ChatClient client;
        public PanelMenu menu;
        public ChatPanel chatPanel;

        [Header("Footer")]
        public Text recordLabel;        // "Pause" while capturing, "Record" while paused
        public Image recordKnob;        // red while capturing, grey while paused
        public Text stateLabel;         // "Recording continuously" / "Paused"
        public Color recColor = new Color(1f, 0.25f, 0.25f);
        public Color pausedColor = new Color(0.45f, 0.48f, 0.52f);

        [Header("Minimise")]
        public Text minimizeLabel;      // "Hide" / "Show"
        public GameObject[] hiddenWhenMinimized;

        public bool IsMinimized { get; private set; }
        public bool KeyboardOpen => _keyboard != null && _keyboard.active;

        private TouchScreenKeyboard _keyboard;

        private void Update()
        {
            RefreshRecord();
            PollKeyboard();
        }

        public void ToggleRecord()
        {
            if (hud) hud.ToggleCapture();
            RefreshRecord();
        }

        public void OpenKeyboard()
        {
            if (!TouchScreenKeyboard.isSupported)
            {
                if (chatPanel) chatPanel.AddSystemMessage("(no on-screen keyboard on this platform)");
                return;
            }
            if (KeyboardOpen) return;
            if (menu) menu.CloseAll();
            _keyboard = TouchScreenKeyboard.Open("", TouchScreenKeyboardType.Default, false, false, false, false, "Ask Megamind…");
        }

        public void ToggleMinimized() => SetMinimized(!IsMinimized);

        public void SetMinimized(bool minimized)
        {
            IsMinimized = minimized;
            if (minimized && menu) menu.CloseAll();
            if (hiddenWhenMinimized != null)
                foreach (var go in hiddenWhenMinimized)
                    if (go) go.SetActive(!minimized);
            if (minimizeLabel) minimizeLabel.text = minimized ? "Show" : "Hide";
        }

        private void RefreshRecord()
        {
            bool capturing = hud != null && hud.IsCapturing;
            if (recordLabel) recordLabel.text = capturing ? "Pause" : "Record";
            if (recordKnob) recordKnob.color = capturing ? recColor : pausedColor;
            if (stateLabel)
            {
                if (hud == null) stateLabel.text = "no capture component";
                else if (capturing) stateLabel.text = "Recording continuously";
                else stateLabel.text = hud.IsPaused ? "Paused" : "Paused · " + hud.StateLabel;
            }
        }

        private void PollKeyboard()
        {
            if (_keyboard == null) return;
            switch (_keyboard.status)
            {
                case TouchScreenKeyboard.Status.Done:
                    string text = _keyboard.text;
                    _keyboard = null;
                    if (string.IsNullOrWhiteSpace(text)) return;
                    if (menu != null && menu.TryVoiceCommand(text)) return;
                    if (client) client.SendText(text.Trim());
                    break;
                case TouchScreenKeyboard.Status.Canceled:
                case TouchScreenKeyboard.Status.LostFocus:
                    _keyboard = null;
                    break;
            }
        }
    }
}
