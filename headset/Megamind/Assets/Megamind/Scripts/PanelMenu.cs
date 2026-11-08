// Top-right dropdown of the chat panel: Settings, Debug (CaptureHud.FullReport), Pin/Unpin, Text size, New session,
// Purge. Only one overlay (dropdown / settings / debug) is open at a time; the dropdown closes when the wearer
// clicks the button again or anywhere else on the panel (`backdrop`). Also executes voice commands recognised by
// VoiceCommands (ChatClient calls TryVoiceCommand on the STT transcript).
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class PanelMenu : MonoBehaviour
    {
        public GameObject root;         // the dropdown list (inactive until opened)
        public GameObject backdrop;     // transparent full-panel raycast target behind the list; click = close
        public GameObject debugRoot;    // debug sub-panel
        public Text debugText;
        public Text pinLabel;           // "Pin panel" / "Unpin panel"
        public Text textSizeLabel;      // "Text size 1.25x"
        public float debugRefreshSeconds = 0.25f;

        public CaptureHud hud;
        public SettingsPanel settings;
        public PanelAnchor anchor;
        public ChatPanel chatPanel;

        public bool IsOpen => root != null && root.activeSelf;
        public bool IsDebugOpen => debugRoot != null && debugRoot.activeSelf;

        private float _nextDebug;

        private void Awake()
        {
            if (root) root.SetActive(false);
            if (backdrop) backdrop.SetActive(false);
            if (debugRoot) debugRoot.SetActive(false);
        }

        private void Update()
        {
            if (IsDebugOpen && debugText && Time.unscaledTime >= _nextDebug)
            {
                _nextDebug = Time.unscaledTime + debugRefreshSeconds;
                debugText.text = hud ? hud.FullReport() : "no CaptureHud in scene";
            }
        }

        public void Toggle() { if (IsOpen) Close(); else Open(); }

        public void Open()
        {
            if (settings) settings.Hide();
            CloseDebug();
            RefreshLabels();
            if (backdrop) backdrop.SetActive(true);
            if (root) root.SetActive(true);
        }

        public void Close()
        {
            if (root) root.SetActive(false);
            if (backdrop) backdrop.SetActive(false);
        }

        /// <summary>Closes every overlay (dropdown, settings, debug).</summary>
        public void CloseAll()
        {
            Close();
            CloseDebug();
            if (settings) settings.Hide();
        }

        public void OpenSettings()
        {
            Close();
            CloseDebug();
            if (settings) settings.Show();
        }

        public void OpenDebug()
        {
            Close();
            if (settings) settings.Hide();
            if (debugRoot) debugRoot.SetActive(true);
            _nextDebug = 0f;
        }

        public void CloseDebug() { if (debugRoot) debugRoot.SetActive(false); }

        public void TogglePin()
        {
            if (anchor) anchor.TogglePinned();
            RefreshLabels();
            Close();
        }

        public void CycleTextSize()
        {
            if (anchor) anchor.CycleScale();
            RefreshLabels();
        }

        public void NewSession()
        {
            Close();
            if (settings) settings.NewSession();
        }

        public void Purge()
        {
            Close();
            if (settings) settings.Purge();
        }

        private void RefreshLabels()
        {
            if (pinLabel && anchor) pinLabel.text = anchor.pinned ? "Unpin panel" : "Pin panel";
            if (textSizeLabel && anchor) textSizeLabel.text = "Text size " + anchor.UiScaleLabel;
        }

        /// <summary>Executes a recognised voice command and posts a system chip; false = not a command.</summary>
        public bool TryVoiceCommand(string transcript)
        {
            var cmd = VoiceCommands.Match(transcript);
            if (cmd == VoiceCommand.None) return false;
            Execute(cmd);
            if (chatPanel) chatPanel.AddSystemMessage("\u201c" + transcript.Trim() + "\u201d \u2192 " + VoiceCommands.Describe(cmd));
            return true;
        }

        public void Execute(VoiceCommand cmd)
        {
            switch (cmd)
            {
                case VoiceCommand.OpenMenu: Open(); break;
                case VoiceCommand.CloseMenu: CloseAll(); break;
                case VoiceCommand.OpenSettings: OpenSettings(); break;
                case VoiceCommand.OpenDebug: OpenDebug(); break;
                case VoiceCommand.StartRecording: if (hud) hud.SetCapturing(true); break;
                case VoiceCommand.StopRecording: if (hud) hud.SetCapturing(false); break;
                case VoiceCommand.BiggerText: if (anchor) anchor.ScaleUp(); RefreshLabels(); break;
                case VoiceCommand.SmallerText: if (anchor) anchor.ScaleDown(); RefreshLabels(); break;
                case VoiceCommand.PinPanel: if (anchor) anchor.SetPinned(true); RefreshLabels(); break;
                case VoiceCommand.UnpinPanel: if (anchor) anchor.SetPinned(false); RefreshLabels(); break;
            }
        }
    }
}
