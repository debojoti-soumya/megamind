// One transcript entry. The prefab root (this component) is a transparent full-width row (HorizontalLayoutGroup) that
// aligns its bubble left (assistant, with an avatar glyph) or right (user). The bubble child is an Image + padding
// LayoutGroup around the Text, shrink-wrapped to the text by ChatBubbleFit so short messages read as chat bubbles
// instead of full-width rows. ChatPanel instantiates the prefab and talks to this component.
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public class ChatBubble : MonoBehaviour
    {
        public Text label;
        public Image background;   // the bubble Image, tinted for error states

        public string Text
        {
            get => label ? label.text : "";
            set { if (label) label.text = value; }
        }

        public void Append(string delta) { if (label) label.text += delta; }

        public void SetError(Color text, Color bubble)
        {
            if (label) label.color = text;
            if (background) background.color = bubble;
        }
    }

    /// <summary>Caps a bubble's width at min(text width + padding, maxWidth); height comes from the LayoutGroup.</summary>
    [RequireComponent(typeof(RectTransform))]
    public class ChatBubbleFit : MonoBehaviour, ILayoutElement
    {
        public Text label;
        [Tooltip("Widest the bubble may grow (canvas px) before the text wraps.")]
        public float maxWidth = 440f;
        [Tooltip("Left + right padding of the LayoutGroup on this object.")]
        public float horizontalPadding = 32f;

        private float _preferredWidth = -1f;

        public void CalculateLayoutInputHorizontal()
        {
            _preferredWidth = label ? Mathf.Min(maxWidth, Mathf.Ceil(label.preferredWidth) + horizontalPadding) : -1f;
        }
        public void CalculateLayoutInputVertical() {}
        public float minWidth => -1f;
        public float preferredWidth => _preferredWidth;
        public float flexibleWidth => 0f;
        public float minHeight => -1f;
        public float preferredHeight => -1f;
        public float flexibleHeight => -1f;
        public int layoutPriority => 2; // wins over the LayoutGroup (0) and LayoutElement (1) for the width properties
    }
}
