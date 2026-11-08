// Tiny normalized-string matcher for short spoken commands ("open menu", "pause recording", "bigger text", ...).
// Runs on the POST /v1/stt transcript before any chat request (or, on older servers, on the chat stream's transcript).
// Exact phrase first, then fuzzy: every word of a multi-word phrase present in the utterance with only courtesy words
// around it ("please open the menu for me"; "did I start recording" stays a question), or a small edit distance ("paws recording" -> pause recording). Single-word phrases match exactly only, so
// a stray "pause" in a question never fires. No wake word, no dependencies.
using System.Text;

namespace Megamind
{
    public enum VoiceCommand
    {
        None, OpenMenu, CloseMenu, OpenSettings, OpenDebug, StartRecording, StopRecording,
        BiggerText, SmallerText, PinPanel, UnpinPanel, OpenControls,
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
            ("show controls", VoiceCommand.OpenControls), ("open controls", VoiceCommand.OpenControls), ("controls", VoiceCommand.OpenControls),
            ("help", VoiceCommand.OpenControls), ("show help", VoiceCommand.OpenControls),
        };

        // Extra words allowed around a multi-word phrase in the fuzzy containment match (not dropped by Normalize).
        static readonly string[] Courtesy = { "for", "me", "just", "go", "ahead", "and", "to", "i", "want", "lets", "let", "s", "this", "panel", "it" };

        static readonly string[] Fillers = { "please", "hey", "ok", "okay", "megamind", "can", "you", "could", "would", "the", "a", "my", "now", "up" };

        /// <summary>Human-readable list for the panel's empty-state hint and the docs.</summary>
        public const string Hint = "\"open menu\", \"open settings\", \"show debug\", \"pause recording\", \"start recording\", " +
                                   "\"bigger text\", \"smaller text\", \"pin panel\", \"unpin panel\", \"show controls\"";

        /// <summary>All phrases, for POST /v1/stt `command_hints`.</summary>
        public static string[] AllPhrases()
        {
            var a = new string[Phrases.Length];
            for (int i = 0; i < a.Length; i++) a[i] = Phrases[i].phrase;
            return a;
        }

        public static VoiceCommand Match(string transcript) => Match(transcript, out _, out _);

        /// <summary>exact = false for fuzzy hits (the panel shows what it understood); phrase = the matched phrase.</summary>
        public static VoiceCommand Match(string transcript, out string phrase, out bool exact)
        {
            phrase = null; exact = false;
            string n = Normalize(transcript);
            if (n.Length == 0) return VoiceCommand.None;
            var words = n.Split(' ');
            if (words.Length > MaxWords) return VoiceCommand.None;
            foreach (var (p, cmd) in Phrases)
                if (n == p) { phrase = p; exact = true; return cmd; }
            // fuzzy: longest multi-word phrase whose words are all in the utterance, else the closest by edit distance
            VoiceCommand best = VoiceCommand.None; int bestLen = 0;
            foreach (var (p, cmd) in Phrases)
            {
                if (p.IndexOf(' ') < 0 || p.Length <= bestLen) continue;
                var pw = p.Split(' ');
                bool all = true;
                foreach (var w in pw) if (System.Array.IndexOf(words, w) < 0) { all = false; break; }
                // every other word must be courtesy ("for me", "just"): "did I start recording" is a question
                foreach (var w in words) if (System.Array.IndexOf(pw, w) < 0 && System.Array.IndexOf(Courtesy, w) < 0) { all = false; break; }
                if (all) { best = cmd; bestLen = p.Length; phrase = p; }
            }
            if (best != VoiceCommand.None) return best;
            int bestDist = int.MaxValue;
            foreach (var (p, cmd) in Phrases)
            {
                if (p.IndexOf(' ') < 0) continue;
                int allowed = p.Length >= 12 ? 2 : 1;
                int d = Levenshtein(n, p);
                if (d <= allowed && d < bestDist) { bestDist = d; best = cmd; phrase = p; }
            }
            return best;
        }

        public static VoiceCommand FromPhrase(string phrase)
        {
            string n = Normalize(phrase);
            foreach (var (p, cmd) in Phrases) if (p == n) return cmd;
            return VoiceCommand.None;
        }

        static int Levenshtein(string a, string b)
        {
            var prev = new int[b.Length + 1]; var cur = new int[b.Length + 1];
            for (int j = 0; j <= b.Length; j++) prev[j] = j;
            for (int i = 1; i <= a.Length; i++)
            {
                cur[0] = i;
                for (int j = 1; j <= b.Length; j++)
                    cur[j] = System.Math.Min(System.Math.Min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
                var t = prev; prev = cur; cur = t;
            }
            return prev[b.Length];
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
            VoiceCommand.OpenControls => "controls opened",
            _ => "",
        };
    }
}
