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

        [Header("Input bar")]
        public InputField inputField;   // "Ask Megamind…" (opens the Quest keyboard on select)
        public Button sendButton;
        public Text phaseLabel;         // Listening / Transcribing / Thinking / Answering / Speaking
        public Image phaseDot;
        public Image micMeter;          // filled Image inside Hold to talk: fillAmount = ChatClient.MicLevel

        [Header("Minimise")]
        public Text minimizeLabel;      // "Hide" / "Show"
        public GameObject[] hiddenWhenMinimized;
        public PanelDrawer window;      // when set, Hide/Show (and the top-right arrow) slide the whole window up/down instead

        public bool IsMinimized { get => window ? !window.IsOpen : _minimized; private set => _minimized = value; }
        private bool _minimized;
        public bool KeyboardOpen => _keyboard != null && _keyboard.active;

        private TouchScreenKeyboard _keyboard;

        private void Update()
        {
            RefreshRecord();
            RefreshPhase();
            PollKeyboard();
        }

        /// <summary>Send button / keyboard Done: sends the input bar text (typed commands run like spoken ones).</summary>
        public void Send()
        {
            if (inputField == null) { OpenKeyboard(); return; }
            string text = inputField.text;
            if (string.IsNullOrWhiteSpace(text)) { inputField.ActivateInputField(); return; }
            inputField.text = "";
            Submit(text);
        }

        private void Submit(string text)
        {
            text = text.Trim();
            if (menu != null && menu.TryVoiceCommand(text)) return;
            if (client) client.SendText(text);
        }

        private void RefreshPhase()
        {
            if (phaseLabel == null && phaseDot == null && micMeter == null) return;
            var phase = client != null ? client.CurrentPhase : ChatClient.Phase.Idle;
            if (micMeter) micMeter.fillAmount = phase == ChatClient.Phase.Listening ? client.MicLevel : 0f;
            string text; Color color;
            switch (phase)
            {
                case ChatClient.Phase.Listening: text = "Listening…"; color = MegamindTheme.Rec; break;
                case ChatClient.Phase.Transcribing: text = "Transcribing…"; color = MegamindTheme.Warning; break;
                case ChatClient.Phase.Thinking: text = "Thinking…"; color = MegamindTheme.Secondary; break;
                case ChatClient.Phase.Answering: text = "Answering…"; color = MegamindTheme.Secondary; break;
                case ChatClient.Phase.Speaking: text = "Speaking…"; color = MegamindTheme.Success; break;
                default: text = ""; color = MegamindTheme.TextMuted; break;
            }
            if (phaseLabel) { phaseLabel.text = text; phaseLabel.color = color; }
            if (phaseDot) { phaseDot.color = color; phaseDot.enabled = phase != ChatClient.Phase.Idle && (phase != ChatClient.Phase.Listening || (Time.unscaledTime % 0.8f) < 0.55f); }
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
            if (inputField != null) { inputField.ActivateInputField(); return; } // the field drives the keyboard; Send submits
            _keyboard = TouchScreenKeyboard.Open("", TouchScreenKeyboardType.Default, false, false, false, false, "Ask Megamind…");
        }

        public void ToggleMinimized() => SetMinimized(!IsMinimized);

        public void SetMinimized(bool minimized)
        {
            IsMinimized = minimized;
            if (minimized && menu) menu.CloseAll();
            if (window) { if (minimized) window.Close(); else window.Open(); }
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
                    Submit(text);
                    break;
                case TouchScreenKeyboard.Status.Canceled:
                case TouchScreenKeyboard.Status.LostFocus:
                    _keyboard = null;
                    break;
            }
        }
    }
}
