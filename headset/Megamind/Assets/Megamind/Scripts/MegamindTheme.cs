// Megamind design system: the single source of colours, type sizes, spacing and sprites for the chat panel.
// Editor/MegamindChatPanelBuilder applies it when generating the panel (Megamind > 1b); runtime scripts read it for
// state colours (REC, paused, error, success) so nothing hard-codes a colour twice.
//
// Palette: dark glass panel, one warm accent (coral) for the user's own voice, a cool secondary (teal) for Megamind,
// semantic colours for status. Spacing is an 8-pt grid, type is a 4-step scale (title / body / caption / mono).
using UnityEngine;
using UnityEngine.UI;

namespace Megamind
{
    public static class MegamindTheme
    {
        // ---- palette ----
        public static readonly Color PanelBg = new Color(0.06f, 0.07f, 0.10f, 0.90f);   // dark glass
        public static readonly Color Surface = new Color(1f, 1f, 1f, 0.06f);            // header / footer strip
        public static readonly Color SurfaceRaised = new Color(0.09f, 0.10f, 0.14f, 0.985f); // drawers, sheets
        public static readonly Color Divider = new Color(1f, 1f, 1f, 0.10f);

        public static readonly Color Accent = new Color(1.00f, 0.47f, 0.38f);            // coral: user, primary actions
        public static readonly Color AccentDark = new Color(0.82f, 0.34f, 0.27f);
        public static readonly Color AccentSoft = new Color(1.00f, 0.47f, 0.38f, 0.22f);
        public static readonly Color Secondary = new Color(0.38f, 0.80f, 0.88f);         // teal: Megamind avatar, links
        public static readonly Color SecondarySoft = new Color(0.38f, 0.80f, 0.88f, 0.18f);

        public static readonly Color TextMain = new Color(0.95f, 0.96f, 0.98f);
        public static readonly Color TextOnAccent = new Color(0.12f, 0.06f, 0.05f);
        public static readonly Color TextDim = new Color(0.66f, 0.70f, 0.78f);
        public static readonly Color TextMuted = new Color(0.46f, 0.50f, 0.58f);

        public static readonly Color Rec = new Color(1f, 0.30f, 0.30f);                  // capturing / listening
        public static readonly Color Paused = new Color(0.50f, 0.53f, 0.60f);
        public static readonly Color Error = new Color(1f, 0.42f, 0.46f);
        public static readonly Color ErrorText = new Color(1f, 0.72f, 0.74f);
        public static readonly Color Success = new Color(0.30f, 0.85f, 0.52f);
        public static readonly Color Warning = new Color(1f, 0.74f, 0.32f);

        public static readonly Color AssistantBubble = new Color(1f, 1f, 1f, 0.08f);
        public static readonly Color UserBubble = Accent;
        public static readonly Color SystemChip = new Color(1f, 1f, 1f, 0.06f);
        public static readonly Color FieldBg = new Color(0f, 0f, 0f, 0.35f);
        public static readonly Color ThumbPlaceholder = new Color(0.16f, 0.18f, 0.23f);

        // ---- type scale (canvas px; the canvas renders its font atlas at 3x so these stay sharp) ----
        public const int FontTitle = 28;
        public const int FontBody = 24;
        public const int FontButton = 20;
        public const int FontCaption = 17;
        public const int FontMono = 16;

        // ---- spacing (8-pt grid) and radii ----
        public const float Unit = 8f;
        public const float Space1 = 8f, Space2 = 16f, Space3 = 24f, Space4 = 32f;
        public const float RadiusS = 10f, RadiusM = 16f, RadiusL = 24f;
        public const float ControlH = 56f;            // one-row control height (buttons, fields)
        public const float BubbleMaxWidthFraction = 0.80f;

        /// <summary>Ghost button: translucent white that brightens on hover, tints with the accent while pressed.</summary>
        public static ColorBlock GhostButton => new ColorBlock
        {
            normalColor = new Color(1f, 1f, 1f, 0.10f),
            highlightedColor = new Color(1f, 1f, 1f, 0.20f),
            pressedColor = new Color(Accent.r, Accent.g, Accent.b, 0.70f),
            selectedColor = new Color(1f, 1f, 1f, 0.14f),
            disabledColor = new Color(1f, 1f, 1f, 0.04f),
            colorMultiplier = 1f,
            fadeDuration = 0.08f,
        };

        /// <summary>Primary button: solid accent.</summary>
        public static ColorBlock AccentButton => new ColorBlock
        {
            normalColor = Accent,
            highlightedColor = Color.Lerp(Accent, Color.white, 0.15f),
            pressedColor = AccentDark,
            selectedColor = Accent,
            disabledColor = new Color(Accent.r, Accent.g, Accent.b, 0.35f),
            colorMultiplier = 1f,
            fadeDuration = 0.08f,
        };

        /// <summary>Danger button (Purge): ghost that turns red on hover/press.</summary>
        public static ColorBlock DangerButton
        {
            get
            {
                var c = GhostButton;
                c.highlightedColor = new Color(Error.r, Error.g, Error.b, 0.35f);
                c.pressedColor = new Color(Error.r, Error.g, Error.b, 0.85f);
                return c;
            }
        }

        // ---- sprites (generated, so the panel needs no hand-made art) ----

        /// <summary>Anti-aliased white rounded rectangle, sized for 9-slicing: use border = radius on all sides.</summary>
        public static Texture2D RoundedRectTexture(int radius)
        {
            int size = radius * 2 + 4; // 4 px of flat centre to slice
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false) { name = "Rounded" + radius, wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    px[y * size + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(255f * RoundedCoverage(x, y, size, size, radius)));
            tex.SetPixels32(px); tex.Apply(false, false);
            return tex;
        }

        /// <summary>Anti-aliased white disc.</summary>
        public static Texture2D CircleTexture(int diameter)
        {
            var tex = new Texture2D(diameter, diameter, TextureFormat.RGBA32, false) { name = "Circle" + diameter, wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[diameter * diameter];
            for (int y = 0; y < diameter; y++)
                for (int x = 0; x < diameter; x++)
                    px[y * diameter + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(255f * RoundedCoverage(x, y, diameter, diameter, diameter / 2f)));
            tex.SetPixels32(px); tex.Apply(false, false);
            return tex;
        }

        public static Sprite RoundedRectSprite(int radius)
        {
            var tex = RoundedRectTexture(radius);
            return Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect,
                                 new Vector4(radius, radius, radius, radius));
        }

        public static Sprite CircleSprite(int diameter)
        {
            var tex = CircleTexture(diameter);
            return Sprite.Create(tex, new Rect(0, 0, diameter, diameter), new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect);
        }

        // Fraction of pixel (x, y) inside a w x h rounded rectangle with corner radius r (4x4 supersampling).
        static float RoundedCoverage(int x, int y, int w, int h, float r)
        {
            const int S = 4;
            int inside = 0;
            for (int sy = 0; sy < S; sy++)
                for (int sx = 0; sx < S; sx++)
                {
                    float px = x + (sx + 0.5f) / S, py = y + (sy + 0.5f) / S;
                    float dx = Mathf.Max(r - px, px - (w - r), 0f);
                    float dy = Mathf.Max(r - py, py - (h - r), 0f);
                    if (dx * dx + dy * dy <= r * r) inside++;
                }
            return inside / (float)(S * S);
        }
    }
}
