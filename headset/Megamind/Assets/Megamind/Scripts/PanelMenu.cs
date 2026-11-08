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
        public GameObject root;         // the drawer (inactive until opened)
        public GameObject backdrop;     // transparent full-panel raycast target behind the drawer; click = close
        public PanelDrawer drawer;      // optional: animates root/backdrop (slide down the right edge); without it they just toggle
        public Lightbox lightbox;       // optional: closed by CloseAll ("close" by voice)
        public GameObject debugRoot;    // debug sub-panel
        public GameObject controlsRoot; // controls cheat-sheet sub-panel
        public Text debugText;
        public Text pinLabel;           // "Pin panel" / "Unpin panel"
        public Text textSizeLabel;      // "Text size 1.25x"
        public float debugRefreshSeconds = 0.25f;

        public CaptureHud hud;
        public SettingsPanel settings;
        public PanelAnchor anchor;
        public ChatPanel chatPanel;

        public bool IsOpen => drawer != null ? drawer.IsOpen : root != null && root.activeSelf;
        public bool IsDebugOpen => debugRoot != null && debugRoot.activeSelf;
        public bool IsControlsOpen => controlsRoot != null && controlsRoot.activeSelf;

        private float _nextDebug;

        private void Awake()
        {
            if (drawer == null)
            {
                if (root) root.SetActive(false);
                if (backdrop) backdrop.SetActive(false);
            }
            if (debugRoot) debugRoot.SetActive(false);
            if (controlsRoot) controlsRoot.SetActive(false);
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
            CloseControls();
            RefreshLabels();
            if (drawer) { drawer.Open(); return; }
            if (backdrop) backdrop.SetActive(true);
            if (root) root.SetActive(true);
        }

        public void Close()
        {
            if (drawer) { drawer.Close(); return; }
            if (root) root.SetActive(false);
            if (backdrop) backdrop.SetActive(false);
        }

        /// <summary>Closes every overlay (dropdown, settings, debug).</summary>
        public void CloseAll()
        {
            Close();
            if (lightbox) lightbox.Close();
            CloseDebug();
            CloseControls();
            if (settings) settings.Hide();
        }

        public void OpenSettings()
        {
            Close();
            CloseDebug();
            CloseControls();
            if (settings) settings.Show();
        }

        public void OpenDebug()
        {
            Close();
            CloseControls();
            if (settings) settings.Hide();
            if (debugRoot) debugRoot.SetActive(true);
            _nextDebug = 0f;
        }

        public void CloseDebug() { if (debugRoot) debugRoot.SetActive(false); }

        public void OpenControls()
        {
            Close();
            CloseDebug();
            if (settings) settings.Hide();
            if (controlsRoot) controlsRoot.SetActive(true);
        }

        public void CloseControls() { if (controlsRoot) controlsRoot.SetActive(false); }

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
        public bool TryVoiceCommand(string transcript) => TryVoiceCommand(transcript, null);

        /// <summary>serverPhrase: POST /v1/stt `command.phrase` (its own fuzzy match), used when the local matcher has none.</summary>
        public bool TryVoiceCommand(string transcript, string serverPhrase)
        {
            var cmd = VoiceCommands.Match(transcript, out string phrase, out bool exact);
            if (cmd == VoiceCommand.None && !string.IsNullOrEmpty(serverPhrase))
            {
                cmd = VoiceCommands.FromPhrase(serverPhrase);
                phrase = serverPhrase; exact = false;
            }
            if (cmd == VoiceCommand.None) return false;
            Execute(cmd);
            // confirmation chip: what was heard, and (for fuzzy hits) what it was taken as
            string heard = "\u201c" + (transcript ?? "").Trim() + "\u201d";
            if (chatPanel) chatPanel.AddSystemMessage(exact ? heard + "  \u2192  " + VoiceCommands.Describe(cmd)
                                                            : heard + "  \u2192  \u201c" + phrase + "\u201d  \u2192  " + VoiceCommands.Describe(cmd));
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
                case VoiceCommand.OpenControls: OpenControls(); break;
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
