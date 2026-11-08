// One memory in a card strip: rounded thumbnail + "n min ago" + a caption snippet. Tap (ray / pinch) opens the Lightbox
// on the same set (retrieval candidates or the answer's citations) at this card.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class CitationCard : MonoBehaviour
    {
        public RawImage image;
        public Text age;
        public Text caption;
        public Button button;
        public const int SnippetChars = 56;

        public Citation Data { get; private set; }
        private List<Citation> _set;
        private int _index;
        private Lightbox _lightbox;

        public void Bind(Citation c, List<Citation> set, int index, Lightbox lightbox)
        {
            Data = c; _set = set; _index = index; _lightbox = lightbox;
            if (age) age.text = ChatPanel.Ago(c.ts);
            if (caption) caption.text = Snippet(c.caption);
            if (button)
            {
                button.onClick.RemoveAllListeners();
                button.onClick.AddListener(Open);
            }
        }

        public void Open()
        {
            if (_lightbox != null && _set != null) _lightbox.Show(_set, _index);
        }

        public static string Snippet(string text)
        {
            text = (text ?? "").Trim().Replace('\n', ' ');
            if (text.StartsWith("LIVE NOW")) { int i = text.IndexOf(':'); if (i > 0 && i + 1 < text.Length) text = text.Substring(i + 1).Trim(); }
            return text.Length <= SnippetChars ? text : text.Substring(0, SnippetChars - 1).TrimEnd() + "\u2026";
        }
    }

    /// <summary>"● ● ●" that cycles while Megamind is thinking (typing indicator bubble).</summary>
    public class TypingIndicator : MonoBehaviour
    {
        public Text label;
        public float stepSeconds = 0.35f;
        private void OnEnable() { if (label) label.text = "\u25cf"; }
        private void Update()
        {
            if (!label) return;
            int n = 1 + (int)(Time.unscaledTime / stepSeconds) % 3;
            label.text = n == 1 ? "\u25cf \u25cb \u25cb" : n == 2 ? "\u25cf \u25cf \u25cb" : "\u25cf \u25cf \u25cf";
        }
    }
}
