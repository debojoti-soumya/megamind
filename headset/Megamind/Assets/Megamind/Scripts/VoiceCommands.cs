// Tiny normalized-string matcher for short spoken commands ("open menu", "pause recording", "bigger text", ...).
// Runs on the transcript the server returns from POST /v1/chat (STT only happens there, see docs/api.md), so a
// recognised command replaces the answer instead of skipping the request. No wake word, no dependencies.
using System.Text;

namespace Megamind
{
    public enum VoiceCommand
    {
        None, OpenMenu, CloseMenu, OpenSettings, OpenDebug, StartRecording, StopRecording,
        BiggerText, SmallerText, PinPanel, UnpinPanel,
    }

    public static class VoiceCommands
    {
        public const int MaxWords = 6;

        // Phrases are matched after Normalize(): lower-case, letters/digits/spaces only, filler words removed.
        static readonly (string phrase, VoiceCommand cmd)[] Phrases =
        {
            ("open menu", VoiceCommand.OpenMenu), ("show menu", VoiceCommand.OpenMenu), ("menu", VoiceCommand.OpenMenu),
            ("open dropdown", VoiceCommand.OpenMenu), ("drop down", VoiceCommand.OpenMenu), ("dropdown", VoiceCommand.OpenMenu),
            ("close menu", VoiceCommand.CloseMenu), ("hide menu", VoiceCommand.CloseMenu), ("close", VoiceCommand.CloseMenu),
            ("close settings", VoiceCommand.CloseMenu), ("close debug", VoiceCommand.CloseMenu),
            ("open settings", VoiceCommand.OpenSettings), ("show settings", VoiceCommand.OpenSettings), ("settings", VoiceCommand.OpenSettings),
            ("open debug", VoiceCommand.OpenDebug), ("show debug", VoiceCommand.OpenDebug), ("debug", VoiceCommand.OpenDebug),
            ("show stats", VoiceCommand.OpenDebug), ("show statistics", VoiceCommand.OpenDebug), ("debug menu", VoiceCommand.OpenDebug),
            ("start recording", VoiceCommand.StartRecording), ("resume recording", VoiceCommand.StartRecording),
            ("start capture", VoiceCommand.StartRecording), ("resume capture", VoiceCommand.StartRecording),
            ("start capturing", VoiceCommand.StartRecording), // no single-word "record": a misheard word must not resume capture
            ("stop recording", VoiceCommand.StopRecording), ("pause recording", VoiceCommand.StopRecording),
            ("stop capture", VoiceCommand.StopRecording), ("pause capture", VoiceCommand.StopRecording),
            ("stop capturing", VoiceCommand.StopRecording), ("pause", VoiceCommand.StopRecording),
            ("bigger text", VoiceCommand.BiggerText), ("larger text", VoiceCommand.BiggerText), ("increase text size", VoiceCommand.BiggerText),
            ("make text bigger", VoiceCommand.BiggerText), ("make it bigger", VoiceCommand.BiggerText), ("zoom in", VoiceCommand.BiggerText),
            ("smaller text", VoiceCommand.SmallerText), ("decrease text size", VoiceCommand.SmallerText),
            ("make text smaller", VoiceCommand.SmallerText), ("make it smaller", VoiceCommand.SmallerText), ("zoom out", VoiceCommand.SmallerText),
            ("pin panel", VoiceCommand.PinPanel), ("pin", VoiceCommand.PinPanel), ("lock panel", VoiceCommand.PinPanel),
            ("unpin panel", VoiceCommand.UnpinPanel), ("unpin", VoiceCommand.UnpinPanel), ("unlock panel", VoiceCommand.UnpinPanel),
            ("follow me", VoiceCommand.UnpinPanel),
        };

        static readonly string[] Fillers = { "please", "hey", "ok", "okay", "megamind", "can", "you", "could", "would", "the", "a", "my", "now", "up" };

        /// <summary>Human-readable list for the panel's empty-state hint and the docs.</summary>
        public const string Hint = "\"open menu\", \"open settings\", \"show debug\", \"pause recording\", \"start recording\", " +
                                   "\"bigger text\", \"smaller text\", \"pin panel\", \"unpin panel\"";

        public static VoiceCommand Match(string transcript)
        {
            string n = Normalize(transcript);
            if (n.Length == 0) return VoiceCommand.None;
            int words = 1;
            foreach (char c in n) if (c == ' ') words++;
            if (words > MaxWords) return VoiceCommand.None;
            foreach (var (phrase, cmd) in Phrases)
                if (n == phrase) return cmd;
            return VoiceCommand.None;
        }

        public static string Normalize(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            var sb = new StringBuilder(s.Length);
            foreach (char ch in s.ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(ch)) sb.Append(ch);
                else if (ch == ' ' || ch == '-' || ch == '\'' || char.IsPunctuation(ch)) sb.Append(' ');
            }
            var kept = new StringBuilder();
            foreach (var w in sb.ToString().Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries))
            {
                if (System.Array.IndexOf(Fillers, w) >= 0) continue;
                if (kept.Length > 0) kept.Append(' ');
                kept.Append(w);
            }
            return kept.ToString();
        }

        public static string Describe(VoiceCommand c) => c switch
        {
            VoiceCommand.OpenMenu => "menu opened",
            VoiceCommand.CloseMenu => "menu closed",
            VoiceCommand.OpenSettings => "settings opened",
            VoiceCommand.OpenDebug => "debug opened",
            VoiceCommand.StartRecording => "recording resumed",
            VoiceCommand.StopRecording => "recording paused",
            VoiceCommand.BiggerText => "text bigger",
            VoiceCommand.SmallerText => "text smaller",
            VoiceCommand.PinPanel => "panel pinned",
            VoiceCommand.UnpinPanel => "panel unpinned (follows you)",
            _ => "",
        };
    }
}
