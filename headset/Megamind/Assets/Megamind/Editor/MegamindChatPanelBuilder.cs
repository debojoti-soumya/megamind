// Editor-only. Stream 3: adds the voice-first chat panel to the capture scene made by MegamindSceneBuilder
// (Megamind > 1), creates the bubble/thumbnail prefabs and the generated sprites (Assets/Megamind/Art), and wires every
// public reference on ChatPanel / ChatClient / PanelAnchor / SettingsPanel / PanelMenu (header status strip + REC dot fed
// by CaptureHud, corner tab -> right-side drawer: New chat, sessions, Settings / Debug / Pin / Text size / Controls / Purge; footer
// Record|Pause / Hold to talk / Type / Hide controls, PanelControls + PanelHover). Run Megamind > 1, then 1b, then build.
//
// Look & feel comes from Scripts/MegamindTheme.cs (palette, type scale, 8-pt grid, radii, button states): change it
// there, re-run 1b. Sprites (rounded rectangles, circle) are rendered in code and imported as 9-sliced sprites, so the
// project needs no hand-made art.
//
// Why a script instead of committed .unity/.prefab YAML: hand-written scene YAML with cross-file GUIDs is
// brittle across Unity versions and nobody can open it here to verify. This is re-runnable (it deletes a
// previous MegamindPanel first) and the result can be tweaked in the Inspector and saved normally.
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Events;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using static Megamind.MegamindTheme;

namespace Megamind.EditorTools
{
    public static class MegamindChatPanelBuilder
    {
        const string Root = "Assets/Megamind";
        const string PrefabDir = Root + "/Prefabs";
        const string ArtDir = Root + "/Art";
        const float PanelW = 600f, PanelH = 720f, HeaderH = 64f, FooterH = 120f, FooterButtonsH = 72f;
        // uGUI Text rasterises glyphs at fontSize * dynamicPixelsPerUnit: at 1000 canvas px/m and ~25 px/deg on Quest 3 a
        // 1x glyph is upscaled ~1.4x and looks soft, so render the font atlas at 3x.
        const float DynamicPixelsPerUnit = 3f;
        // 3 thumbnails per row: 3 * 176 + 2 * 8 spacing = 544 < 600 - 2 * 16 padding. 4:3 like the 1280x960 capture.
        const float ThumbW = 176f, ThumbH = 132f;
        const float ContentPad = Space2;
        const float DrawerW = 216f, TabW = 72f, TabH = 40f;   // drawer ≈ 35 % of the panel; tab hugs the top-right corner
        const float AvatarSize = 32f;
        static readonly float BubbleMaxW = Mathf.Round((PanelW - 2f * ContentPad - AvatarSize - Space1) * BubbleMaxWidthFraction);

        static Font _font;
        static Font UIFont => _font ??= Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        // Generated sprites (see LoadArt): panel / sheets, buttons, chips, bubbles, dots.
        static Sprite RoundedL, RoundedM, RoundedS, Bubble, Circle;

        [MenuItem("Megamind/1b. Add Chat Panel To Current Scene", false, 1)]
        public static void Build()
        {
            var scene = EditorSceneManager.GetActiveScene();
            if (scene.path != MegamindSceneBuilder.ScenePath)
            {
                // Batch mode (or a fresh editor) starts with an untitled scene: use the capture scene if it exists.
                if (!File.Exists(MegamindSceneBuilder.ScenePath))
                    throw new InvalidOperationException("Run Megamind > 1. Create Capture Scene first (saves " + MegamindSceneBuilder.ScenePath + ").");
                if (!string.IsNullOrEmpty(scene.path) && scene.isDirty) EditorSceneManager.SaveScene(scene);
                scene = EditorSceneManager.OpenScene(MegamindSceneBuilder.ScenePath, OpenSceneMode.Single);
            }

            EnsureFolder(Root); EnsureFolder(PrefabDir); EnsureFolder(ArtDir);
            LoadArt();
            var config = MegamindSceneBuilder.GetOrCreateConfig();
            var userBubble = BubblePrefab("UserBubble", user: true);
            var assistantBubble = BubblePrefab("AssistantBubble", user: false);
            var systemChip = SystemChipPrefab();
            var metaLine = MetaLinePrefab();
            var thumbRow = ThumbRowPrefab();
            var thumb = ThumbPrefab();
            var sessionEntry = SessionEntryPrefab();
            var sessionGroup = SessionGroupPrefab();
            var card = CitationCardPrefab();
            var cardStrip = CardStripPrefab();
            var turnActions = TurnActionsPrefab();

            var old = GameObject.Find("MegamindPanel");
            if (old) Undo.DestroyObjectImmediate(old);

            var head = FindHead();
            var panel = BuildPanel(config, head, userBubble, assistantBubble, systemChip, metaLine, thumbRow, thumb, sessionEntry, sessionGroup,
                                   card, cardStrip, turnActions);
            EnsureEventSystem(head);

            Undo.RegisterCreatedObjectUndo(panel, "Build Megamind panel");
            Selection.activeGameObject = panel;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            AssetDatabase.SaveAssets();
            Debug.Log("[Megamind] Chat panel built. " + (head ? "Head = " + head.name : "No OVRCameraRig/CenterEyeAnchor found — assign PanelAnchor.head and FrameUploader.headPose manually."));
        }

        // ---------- scene ----------

        static GameObject BuildPanel(MegamindConfig config, Transform head, ChatBubble userBubble, ChatBubble assistantBubble,
                                     ChatBubble systemChip, RectTransform metaLine, RectTransform thumbRow, RectTransform thumb,
                                     RectTransform sessionEntry, Text sessionGroup, CitationCard card, RectTransform cardStrip,
                                     RectTransform turnActions)
        {
            var go = new GameObject("MegamindPanel", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var rt = go.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(PanelW, PanelH);
            var canvas = go.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = head ? head.GetComponent<Camera>() : Camera.main;
            var scaler = go.GetComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = DynamicPixelsPerUnit; scaler.referencePixelsPerUnit = 100f;
            // PanelAnchor.Start rescales so the panel is config.panelWidthMeters wide; this is just a sane default.
            rt.localScale = Vector3.one * (config.panelWidthMeters / PanelW);
            rt.position = new Vector3(0f, 1.3f, config.panelDistance);

            var background = MakeImage(go.transform, "Background", PanelBg, RoundedL);
            background.rectTransform.Stretch();
            background.raycastTarget = true; // the whole panel counts as "ray over UI" (push-to-talk suppression)

            // Header: REC dot (always visible, privacy), server dot, one-line status strip (CaptureHud.ShortStatus), Hide, Menu.
            // Transparent over the glass background, separated from the transcript by a hairline.
            var header = new GameObject("Header", typeof(RectTransform));
            header.transform.SetParent(go.transform, false);
            header.GetComponent<RectTransform>().AnchorTop(HeaderH);
            Hairline(header.transform, bottom: true);
            var recDot = MakeImage(header.transform, "RecDot", Rec, Circle);
            recDot.type = Image.Type.Simple; recDot.AnchorLeftCentre(Space2, 16f, 16f);
            var dot = MakeImage(header.transform, "ServerDot", Error, Circle);
            dot.type = Image.Type.Simple; dot.AnchorLeftCentre(Space2 + 24f, 10f, 10f);
            var statusLine = MakeLabel(header.transform, "StatusLine", "connecting…", FontCaption, TextDim, TextAnchor.MiddleLeft);
            statusLine.horizontalOverflow = HorizontalWrapMode.Overflow; statusLine.verticalOverflow = VerticalWrapMode.Truncate;
            statusLine.rectTransform.Fill(new Vector2(Space2 + 44f, 0f), new Vector2(-(Space2 + TabW + Space1 + 80f + Space1), 0f));
            var minimizeButton = MakeButton(header.transform, "MinimizeButton", "Hide", FontCaption, sprite: RoundedS);
            minimizeButton.GetComponent<RectTransform>().AnchorRightCentre(Space2 + TabW + Space1, 80f, TabH);

            // Scroll view with the transcript
            var scrollGo = new GameObject("Scroll", typeof(RectTransform), typeof(ScrollRect));
            scrollGo.transform.SetParent(go.transform, false);
            scrollGo.GetComponent<RectTransform>().Fill(new Vector2(0f, FooterH), new Vector2(0f, -HeaderH));
            var scroll = scrollGo.GetComponent<ScrollRect>();
            scroll.horizontal = false; scroll.movementType = ScrollRect.MovementType.Clamped; scroll.scrollSensitivity = 30f;

            var viewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D));
            viewport.transform.SetParent(scrollGo.transform, false);
            viewport.GetComponent<RectTransform>().Stretch();

            var content = new GameObject("Content", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(ContentSizeFitter));
            content.transform.SetParent(viewport.transform, false);
            var crt = content.GetComponent<RectTransform>();
            crt.anchorMin = new Vector2(0f, 1f); crt.anchorMax = Vector2.one; crt.pivot = new Vector2(0.5f, 1f);
            crt.offsetMin = Vector2.zero; crt.offsetMax = Vector2.zero;
            var vlg = content.GetComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset((int)ContentPad, (int)ContentPad, (int)Space2, (int)Space2); vlg.spacing = Space2;
            vlg.childControlWidth = true; vlg.childControlHeight = true;
            vlg.childForceExpandWidth = true; vlg.childForceExpandHeight = false;
            content.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.viewport = viewport.GetComponent<RectTransform>();
            scroll.content = crt;

            var emptyHint = MakeLabel(go.transform, "EmptyHint",
                "Ask Megamind about something you saw.\n\nHold the trigger (or pinch) and talk, or tap Type.\n\nShort commands work too: " + VoiceCommands.Hint + ".",
                FontButton, TextDim, TextAnchor.MiddleCenter);
            emptyHint.lineSpacing = 1.15f;
            emptyHint.rectTransform.Fill(new Vector2(Space4 + Space1, FooterH), new Vector2(-(Space4 + Space1), -HeaderH));

            // Typing indicator: an assistant bubble ("● ○ ○" cycling) kept at the end of the transcript while thinking.
            var typing = (ChatBubble)PrefabUtility.InstantiatePrefab(assistantBubble, crt);
            typing.gameObject.name = "Typing";
            typing.Text = "\u25cf \u25cb \u25cb";
            typing.label.color = Secondary;
            typing.gameObject.AddComponent<TypingIndicator>().label = typing.label;
            typing.gameObject.SetActive(false);

            // "↓ Latest" pill above the footer, shown by ChatPanel when the user scrolled up and new messages arrive.
            var jump = MakeButton(go.transform, "JumpToLatest", "\u2193  Latest", FontCaption, sprite: RoundedM);
            var jrt = jump.GetComponent<RectTransform>();
            jrt.anchorMin = jrt.anchorMax = new Vector2(0.5f, 0f); jrt.pivot = new Vector2(0.5f, 0f);
            jrt.anchoredPosition = new Vector2(0f, FooterH + Space1); jrt.sizeDelta = new Vector2(128f, 36f);
            jump.GetComponent<Image>().color = SurfaceRaised; // solid pill over the transcript
            jump.GetComponentInChildren<Text>().color = Accent;
            jump.gameObject.SetActive(false);

            // Footer: state line ("Recording continuously" / "Paused") + listening/thinking indicators on top,
            // big Record|Pause / Hold to talk / Type buttons below (3 controls: no crowding).
            var footer = new GameObject("Footer", typeof(RectTransform));
            footer.transform.SetParent(go.transform, false);
            footer.GetComponent<RectTransform>().AnchorBottom(FooterH);
            Hairline(footer.transform, bottom: false);
            // Top row: capture pill (Record | Pause + knob) and state text on the left, turn phase on the right.
            float topH = FooterH - FooterButtonsH;
            var recordButton = MakeButton(footer.transform, "RecordButton", "Record", FontCaption, sprite: RoundedS);
            var rbRt = recordButton.GetComponent<RectTransform>();
            rbRt.anchorMin = new Vector2(0f, 1f); rbRt.anchorMax = new Vector2(0f, 1f); rbRt.pivot = new Vector2(0f, 1f);
            rbRt.anchoredPosition = new Vector2(Space2, -Space1); rbRt.sizeDelta = new Vector2(104f, topH - Space2);
            var recordLabel = recordButton.GetComponentInChildren<Text>();
            recordLabel.alignment = TextAnchor.MiddleLeft; recordLabel.rectTransform.offsetMin = new Vector2(Space3 + 4f, 0f);
            var recordKnob = MakeImage(recordButton.transform, "Knob", Paused, Circle);
            recordKnob.type = Image.Type.Simple; recordKnob.AnchorLeftCentre(12f, 12f, 12f);
            var stateLabel = MakeLabel(footer.transform, "StateLabel", "Paused", FontCaption, TextDim, TextAnchor.MiddleLeft);
            stateLabel.horizontalOverflow = HorizontalWrapMode.Overflow; stateLabel.verticalOverflow = VerticalWrapMode.Truncate;
            stateLabel.rectTransform.anchorMin = new Vector2(0f, 1f); stateLabel.rectTransform.anchorMax = Vector2.one; stateLabel.rectTransform.pivot = new Vector2(0.5f, 1f);
            stateLabel.rectTransform.offsetMin = new Vector2(Space2 + 104f + Space1 + 4f, -topH); stateLabel.rectTransform.offsetMax = new Vector2(-220f, 0f);
            var indicators = new GameObject("Indicators", typeof(RectTransform));
            indicators.transform.SetParent(footer.transform, false);
            var irt = indicators.GetComponent<RectTransform>();
            irt.anchorMin = new Vector2(1f, 1f); irt.anchorMax = Vector2.one; irt.pivot = new Vector2(1f, 1f);
            irt.anchoredPosition = new Vector2(-Space3, 0f); irt.sizeDelta = new Vector2(200f, topH);
            var phaseDot = MakeImage(indicators.transform, "PhaseDot", Rec, Circle);
            phaseDot.type = Image.Type.Simple;
            phaseDot.rectTransform.anchorMin = phaseDot.rectTransform.anchorMax = new Vector2(1f, 0.5f); phaseDot.rectTransform.pivot = new Vector2(1f, 0.5f);
            phaseDot.rectTransform.anchoredPosition = Vector2.zero; phaseDot.rectTransform.sizeDelta = new Vector2(10f, 10f);
            var phaseLabel = MakeLabel(indicators.transform, "PhaseLabel", "", FontCaption, TextDim, TextAnchor.MiddleRight);
            phaseLabel.horizontalOverflow = HorizontalWrapMode.Overflow;
            phaseLabel.rectTransform.Fill(Vector2.zero, new Vector2(-(10f + Space1), 0f));
            // ChatPanel's listening/thinking indicators are driven by PanelControls' phase line now; keep hidden stand-ins so
            // the fields stay valid for scenes/scripts that read them.
            var listening = MakeImage(indicators.transform, "ListeningIndicator", Rec, null); listening.rectTransform.sizeDelta = Vector2.zero;
            var thinking = MakeLabel(indicators.transform, "ThinkingIndicator", "", FontCaption, Secondary, TextAnchor.MiddleRight); thinking.rectTransform.sizeDelta = Vector2.zero;

            // Input bar (3 controls): Hold to talk with a level meter, the question field (Quest keyboard), Send.
            var buttons = new GameObject("Buttons", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            buttons.transform.SetParent(footer.transform, false);
            var brt = buttons.GetComponent<RectTransform>();
            brt.AnchorBottom(FooterButtonsH); brt.offsetMin = new Vector2(Space2, Space2); brt.offsetMax = new Vector2(-Space2, FooterButtonsH);
            var bhlg = buttons.GetComponent<HorizontalLayoutGroup>();
            bhlg.spacing = Space1; bhlg.childControlWidth = true; bhlg.childControlHeight = true; bhlg.childForceExpandWidth = false;
            var talkButton = MakeButton(buttons.transform, "HoldToTalkButton", "Hold to talk", FontButton);
            var tle = talkButton.gameObject.AddComponent<LayoutElement>(); tle.preferredWidth = 168f; tle.minWidth = 168f;
            var talkLabel = talkButton.GetComponentInChildren<Text>();
            talkLabel.rectTransform.offsetMin = new Vector2(Space3 + 8f, 0f);
            var micIcon = MakeImage(talkButton.transform, "Mic", TextMain, Circle);
            micIcon.type = Image.Type.Simple; micIcon.AnchorLeftCentre(Space2, 12f, 12f);
            var meterTrack = MakeImage(talkButton.transform, "Meter", new Color(1f, 1f, 1f, 0.12f), RoundedS);
            meterTrack.rectTransform.anchorMin = new Vector2(0f, 0f); meterTrack.rectTransform.anchorMax = new Vector2(1f, 0f); meterTrack.rectTransform.pivot = new Vector2(0.5f, 0f);
            meterTrack.rectTransform.offsetMin = new Vector2(Space2, 6f); meterTrack.rectTransform.offsetMax = new Vector2(-Space2, 10f);
            var meterFill = MakeImage(meterTrack.transform, "Fill", Rec, RoundedS);
            meterFill.type = Image.Type.Filled; meterFill.fillMethod = Image.FillMethod.Horizontal; meterFill.fillOrigin = 0; meterFill.fillAmount = 0f;
            meterFill.rectTransform.Stretch();
            var inputField = MakeInputField(buttons.transform, "Question", "Ask Megamind\u2026");
            inputField.GetComponent<LayoutElement>().flexibleWidth = 1f;
            inputField.GetComponent<LayoutElement>().preferredHeight = 0f;
            var sendButton = MakeButton(buttons.transform, "SendButton", "Send", FontButton, AccentButton);
            var sle = sendButton.gameObject.AddComponent<LayoutElement>(); sle.preferredWidth = 96f; sle.minWidth = 96f;

            // Settings sheet (inactive by default; SettingsPanel.Awake hides it)
            var settingsRoot = Sheet(go.transform, "Settings");
            var svlg = settingsRoot.gameObject.AddComponent<VerticalLayoutGroup>();
            svlg.padding = new RectOffset((int)Space3, (int)Space3, (int)Space3, (int)Space3); svlg.spacing = Space1;
            svlg.childControlWidth = true; svlg.childControlHeight = true; svlg.childForceExpandHeight = false;
            MakeLabel(settingsRoot.transform, "Title", "Settings", FontTitle, TextMain, TextAnchor.MiddleLeft).PreferredHeight(40f);
            Spacer(settingsRoot.transform, Space1);
            MakeLabel(settingsRoot.transform, "ServerLabel", "Server base URL (laptop Tailscale IP)", FontCaption, TextDim, TextAnchor.MiddleLeft).PreferredHeight(24f);
            var serverField = MakeInputField(settingsRoot.transform, "ServerUrl", "http://100.x.y.z:8000");
            Spacer(settingsRoot.transform, Space1);
            MakeLabel(settingsRoot.transform, "DeviceLabel", "Device id", FontCaption, TextDim, TextAnchor.MiddleLeft).PreferredHeight(24f);
            var deviceField = MakeInputField(settingsRoot.transform, "DeviceId", "quest3-1");
            Spacer(settingsRoot.transform, Space2);
            var row = ButtonRow(settingsRoot.transform, "Buttons");
            var save = MakeButton(row, "Save", "Save", FontButton, AccentButton);
            var newSession = MakeButton(row, "NewSession", "New session", FontButton);
            var purge = MakeButton(row, "Purge", "Purge memory", FontButton, DangerButton);
            var row2 = ButtonRow(settingsRoot.transform, "Buttons2");
            var reset = MakeButton(row2, "Reset", "Reset overrides", FontButton);
            var close = MakeButton(row2, "Close", "Close", FontButton);
            Spacer(settingsRoot.transform, Space1);
            var feedback = MakeLabel(settingsRoot.transform, "Feedback", "", FontCaption, TextDim, TextAnchor.UpperLeft);
            feedback.PreferredHeight(72f);

            // Debug sheet (CaptureHud.FullReport in a scroll view), same footprint as Settings
            var debugRoot = Sheet(go.transform, "Debug");
            SheetTitle(debugRoot.transform, "Debug");
            var debugClose = SheetClose(debugRoot.transform);
            var debugScrollGo = new GameObject("Scroll", typeof(RectTransform), typeof(ScrollRect));
            debugScrollGo.transform.SetParent(debugRoot.transform, false);
            debugScrollGo.GetComponent<RectTransform>().Fill(new Vector2(Space3, Space2), new Vector2(-Space3, -(Space3 + 40f + Space2)));
            var debugScroll = debugScrollGo.GetComponent<ScrollRect>();
            debugScroll.horizontal = false; debugScroll.movementType = ScrollRect.MovementType.Clamped; debugScroll.scrollSensitivity = 30f;
            var debugViewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D));
            debugViewport.transform.SetParent(debugScrollGo.transform, false);
            debugViewport.GetComponent<RectTransform>().Stretch();
            var debugText = MakeLabel(debugViewport.transform, "Text", "", FontMono, TextMain, TextAnchor.UpperLeft);
            debugText.lineSpacing = 1.2f;
            var dtRt = debugText.rectTransform;
            dtRt.anchorMin = new Vector2(0f, 1f); dtRt.anchorMax = Vector2.one; dtRt.pivot = new Vector2(0.5f, 1f);
            dtRt.offsetMin = new Vector2(0f, 0f); dtRt.offsetMax = new Vector2(0f, 0f);
            debugText.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            debugScroll.viewport = debugViewport.GetComponent<RectTransform>(); debugScroll.content = dtRt;

            // Controls cheat-sheet
            var controlsRoot = Sheet(go.transform, "Controls");
            SheetTitle(controlsRoot.transform, "Controls");
            var controlsClose = SheetClose(controlsRoot.transform);
            var controlsText = MakeLabel(controlsRoot.transform, "Text", ControlsText, FontButton, TextMain, TextAnchor.UpperLeft);
            controlsText.lineSpacing = 1.1f;
            controlsText.rectTransform.Fill(new Vector2(Space3, Space2), new Vector2(-Space3, -(Space3 + 40f + Space2)));
            controlsText.verticalOverflow = VerticalWrapMode.Truncate;

            // Lightbox: full-size viewer over the panel (scrim click = close). Viewport clips the pannable/zoomable image.
            var lightRoot = Sheet(go.transform, "Lightbox");
            lightRoot.color = new Color(PanelBg.r, PanelBg.g, PanelBg.b, 0.985f);
            var scrim = lightRoot.gameObject.AddComponent<Button>(); scrim.transition = Selectable.Transition.None;
            var lbCounter = MakeLabel(lightRoot.transform, "Counter", "1 / 1", FontTitle, TextMain, TextAnchor.MiddleLeft);
            lbCounter.rectTransform.AnchorTop(40f); lbCounter.rectTransform.offsetMin = new Vector2(Space3, -(Space3 + 40f)); lbCounter.rectTransform.offsetMax = new Vector2(-(Space3 + 96f + Space1), -Space3);
            var lbTime = MakeLabel(lightRoot.transform, "Time", "", FontCaption, TextDim, TextAnchor.MiddleLeft);
            lbTime.rectTransform.AnchorTop(40f); lbTime.rectTransform.offsetMin = new Vector2(Space3 + 96f, -(Space3 + 40f)); lbTime.rectTransform.offsetMax = new Vector2(-(Space3 + 96f + Space1), -Space3);
            var lbClose = SheetClose(lightRoot.transform);
            var lbViewport = MakeImage(lightRoot.transform, "Viewport", new Color(0f, 0f, 0f, 0.45f), RoundedM);
            lbViewport.raycastTarget = true; // drag target
            lbViewport.gameObject.AddComponent<RectMask2D>();
            lbViewport.rectTransform.Fill(new Vector2(Space3, Space3 + 88f), new Vector2(-Space3, -(Space3 + 40f + Space2)));
            var lbImage = new GameObject("Image", typeof(RectTransform), typeof(RawImage));
            lbImage.transform.SetParent(lbViewport.transform, false);
            var lbiRt = lbImage.GetComponent<RectTransform>();
            lbiRt.anchorMin = lbiRt.anchorMax = new Vector2(0.5f, 0.5f); lbiRt.pivot = new Vector2(0.5f, 0.5f); lbiRt.sizeDelta = new Vector2(480f, 360f);
            lbImage.GetComponent<RawImage>().color = ThumbPlaceholder; lbImage.GetComponent<RawImage>().raycastTarget = false;
            var lbPrev = RoundButton(lightRoot.transform, "Prev", "\u2039");
            lbPrev.GetComponent<RectTransform>().anchorMin = lbPrev.GetComponent<RectTransform>().anchorMax = new Vector2(0f, 0.5f);
            lbPrev.GetComponent<RectTransform>().anchoredPosition = new Vector2(Space3 + Space1, 20f);
            var lbNext = RoundButton(lightRoot.transform, "Next", "\u203a");
            lbNext.GetComponent<RectTransform>().anchorMin = lbNext.GetComponent<RectTransform>().anchorMax = new Vector2(1f, 0.5f);
            lbNext.GetComponent<RectTransform>().anchoredPosition = new Vector2(-(Space3 + Space1), 20f);
            var lbCaption = MakeLabel(lightRoot.transform, "Caption", "", FontCaption, TextMain, TextAnchor.UpperLeft);
            lbCaption.verticalOverflow = VerticalWrapMode.Truncate;
            lbCaption.rectTransform.anchorMin = Vector2.zero; lbCaption.rectTransform.anchorMax = new Vector2(1f, 0f); lbCaption.rectTransform.pivot = new Vector2(0.5f, 0f);
            lbCaption.rectTransform.offsetMin = new Vector2(Space3, Space3); lbCaption.rectTransform.offsetMax = new Vector2(-(Space3 + 2f * 44f + Space1 + Space2), Space3 + 64f);
            var lbZoomOut = RoundButton(lightRoot.transform, "ZoomOut", "\u2212");
            lbZoomOut.GetComponent<RectTransform>().anchorMin = lbZoomOut.GetComponent<RectTransform>().anchorMax = new Vector2(1f, 0f);
            lbZoomOut.GetComponent<RectTransform>().anchoredPosition = new Vector2(-(Space3 + 44f + Space1 + 22f), Space3 + 32f);
            var lbZoomIn = RoundButton(lightRoot.transform, "ZoomIn", "+");
            lbZoomIn.GetComponent<RectTransform>().anchorMin = lbZoomIn.GetComponent<RectTransform>().anchorMax = new Vector2(1f, 0f);
            lbZoomIn.GetComponent<RectTransform>().anchoredPosition = new Vector2(-(Space3 + 22f), Space3 + 32f);
            var lightbox = lbViewport.gameObject.AddComponent<Lightbox>();
            lightbox.config = config; lightbox.root = lightRoot.gameObject; lightbox.viewport = lbViewport.rectTransform;
            lightbox.image = lbImage.GetComponent<RawImage>(); lightbox.caption = lbCaption; lightbox.timeLabel = lbTime; lightbox.counter = lbCounter;
            lightbox.prev = lbPrev; lightbox.next = lbNext;

            // Drawer: transparent backdrop (click anywhere = close) + a column down the right edge that PanelDrawer slides
            // in from above (clipped by DrawerClip). Built after the sheets so it draws on top of them.
            var backdrop = MakeImage(go.transform, "MenuBackdrop", new Color(0f, 0f, 0f, 0f), null);
            backdrop.raycastTarget = true;
            backdrop.rectTransform.Stretch();
            var backdropButton = backdrop.gameObject.AddComponent<Button>();
            backdropButton.transition = Selectable.Transition.None;
            var drawerClip = new GameObject("DrawerClip", typeof(RectTransform), typeof(RectMask2D));
            drawerClip.transform.SetParent(go.transform, false);
            drawerClip.GetComponent<RectTransform>().Stretch();
            var menuRoot = MakeImage(drawerClip.transform, "Drawer", SurfaceRaised, RoundedL);
            menuRoot.raycastTarget = true; // swallow rays so the transcript underneath is not clickable through gaps
            var mrt = menuRoot.rectTransform;
            mrt.anchorMin = new Vector2(1f, 0f); mrt.anchorMax = new Vector2(1f, 1f); mrt.pivot = new Vector2(1f, 1f);
            mrt.anchoredPosition = Vector2.zero; mrt.sizeDelta = new Vector2(DrawerW, 0f);
            var drawerGroup = menuRoot.gameObject.AddComponent<CanvasGroup>();
            var mvlg = menuRoot.gameObject.AddComponent<VerticalLayoutGroup>();
            mvlg.padding = new RectOffset((int)Space1 + 4, (int)Space1 + 4, (int)HeaderH, (int)Space2); mvlg.spacing = 4f;
            mvlg.childControlWidth = true; mvlg.childControlHeight = true; mvlg.childForceExpandWidth = true; mvlg.childForceExpandHeight = false;
            MakeLabel(menuRoot.transform, "SessionsTitle", "Chats", FontCaption, TextMuted, TextAnchor.MiddleLeft).PreferredHeight(24f);
            var miNewSession = DrawerEntry(menuRoot.transform, "NewSession", "+", Accent, "New chat", AccentButton);
            var sessionsList = DrawerList(menuRoot.transform, "Sessions");
            var sessionsEmpty = MakeLabel(sessionsList, "Empty", "Earlier chats will show up here.", FontCaption, TextMuted, TextAnchor.UpperLeft);
            HairlineRow(menuRoot.transform);
            var miSettings = DrawerEntry(menuRoot.transform, "Settings", "S", Secondary, "Settings");
            var miDebug = DrawerEntry(menuRoot.transform, "Debug", "D", Warning, "Debug stats");
            var miPin = DrawerEntry(menuRoot.transform, "Pin", "P", Success, "Pin panel");
            var miTextSize = DrawerEntry(menuRoot.transform, "TextSize", "A", Accent, "Text size 1x");
            var miControls = DrawerEntry(menuRoot.transform, "Controls", "?", TextDim, "Controls");
            var miPurge = DrawerEntry(menuRoot.transform, "Purge", "!", Error, "Purge memory", DangerButton);

            // Corner tab: hamburger glyph + chevron, glued to the top-right corner of the header. Last so it draws on top of
            // the backdrop (a click on it toggles instead of just closing) and stays visible when the panel is Hidden.
            var tab = MakeButton(header.transform, "MenuTab", "", FontCaption, sprite: RoundedM);
            tab.GetComponent<RectTransform>().AnchorRightCentre(Space2, TabW, TabH);
            var bars = new GameObject("Bars", typeof(RectTransform));
            bars.transform.SetParent(tab.transform, false);
            var barsRt = bars.GetComponent<RectTransform>();
            barsRt.anchorMin = barsRt.anchorMax = new Vector2(0f, 0.5f); barsRt.pivot = new Vector2(0f, 0.5f);
            barsRt.anchoredPosition = new Vector2(14f, 0f); barsRt.sizeDelta = new Vector2(18f, 14f);
            for (int i = 0; i < 3; i++)
            {
                var bar = MakeImage(bars.transform, "Bar" + i, TextMain, null);
                var brt2 = bar.rectTransform;
                brt2.anchorMin = new Vector2(0f, 1f); brt2.anchorMax = new Vector2(1f, 1f); brt2.pivot = new Vector2(0.5f, 1f);
                brt2.anchoredPosition = new Vector2(0f, -i * 6f); brt2.sizeDelta = new Vector2(0f, 2f);
            }
            var chevron = Chevron(tab.transform, TextMain);
            chevron.anchorMin = chevron.anchorMax = new Vector2(1f, 0.5f); chevron.pivot = new Vector2(0.5f, 0.5f);
            chevron.anchoredPosition = new Vector2(-20f, 0f); chevron.sizeDelta = new Vector2(16f, 16f);

            // Components
            var speaker = go.AddComponent<AudioSource>();
            speaker.playOnAwake = false; speaker.spatialBlend = 0f;

            var anchor = go.AddComponent<PanelAnchor>();
            anchor.config = config; anchor.head = head;

            var store = go.AddComponent<SessionStore>();

            var chatPanel = go.AddComponent<ChatPanel>();
            chatPanel.config = config; chatPanel.sessions = store;
            chatPanel.content = crt; chatPanel.scroll = scroll;
            chatPanel.userBubblePrefab = userBubble; chatPanel.assistantBubblePrefab = assistantBubble;
            chatPanel.systemChipPrefab = systemChip; chatPanel.metaLinePrefab = metaLine;
            chatPanel.thumbnailRowPrefab = thumbRow; chatPanel.thumbnailPrefab = thumb;
            chatPanel.cardPrefab = card; chatPanel.cardStripPrefab = cardStrip; chatPanel.turnActionsPrefab = turnActions;
            chatPanel.lightbox = lightbox; chatPanel.typingIndicator = typing.gameObject; chatPanel.jumpToLatest = jump;
            chatPanel.listeningIndicator = listening.gameObject; chatPanel.thinkingIndicator = thinking.gameObject;
            chatPanel.emptyHint = emptyHint.gameObject;
            chatPanel.statusLine = statusLine; chatPanel.serverDot = dot; chatPanel.recDot = recDot;
            chatPanel.uploader = UnityEngine.Object.FindFirstObjectByType<FrameUploader>();
            var hud = UnityEngine.Object.FindFirstObjectByType<CaptureHud>();
            chatPanel.hud = hud;

            var client = go.AddComponent<ChatClient>();
            client.config = config; client.panel = chatPanel; client.speaker = speaker; client.sessions = store;
            chatPanel.client = client;
            client.leftHand = MegamindSceneBuilder.FindHand(OVRPlugin.Hand.HandLeft);
            client.rightHand = MegamindSceneBuilder.FindHand(OVRPlugin.Hand.HandRight);
            if (hud != null)
            {
                if (hud.leftHand == null) hud.leftHand = client.leftHand;
                if (hud.rightHand == null) hud.rightHand = client.rightHand;
                hud.showOwnCanvas = false; // the panel header shows the status now
                EditorUtility.SetDirty(hud);
            }
            else Debug.LogWarning("[Megamind] no CaptureHud in the scene: the header status strip and Debug view will be empty. Re-run Megamind > 1.");
            if (client.leftHand == null || client.rightHand == null)
                Debug.LogWarning("[Megamind] no OVRHand for " + (client.leftHand == null ? "left " : "") + (client.rightHand == null ? "right " : "") +
                                 "in the scene: pinch push-to-talk is off. Re-run Megamind > 1. Create Capture Scene (Overwrite), then 1b.");

            var settings = go.AddComponent<SettingsPanel>();
            settings.config = config; settings.client = client; settings.chatPanel = chatPanel;
            settings.root = settingsRoot.gameObject; settings.serverUrlField = serverField; settings.deviceIdField = deviceField;
            settings.feedback = feedback;

            var drawer = go.AddComponent<PanelDrawer>();
            drawer.drawer = mrt; drawer.group = drawerGroup; drawer.backdrop = backdrop.gameObject; drawer.chevron = chevron;
            drawer.hiddenOffset = PanelH;
            var menu = go.AddComponent<PanelMenu>();
            menu.root = menuRoot.gameObject; menu.backdrop = backdrop.gameObject; menu.drawer = drawer; menu.lightbox = lightbox;
            menu.debugRoot = debugRoot.gameObject; menu.debugText = debugText; menu.controlsRoot = controlsRoot.gameObject;
            menu.pinLabel = miPin.GetComponentInChildren<Text>(); menu.textSizeLabel = miTextSize.GetComponentInChildren<Text>();
            menu.hud = hud; menu.settings = settings; menu.anchor = anchor; menu.chatPanel = chatPanel;
            client.menu = menu;
            client.panelHover = go.AddComponent<PanelHover>();

            var sessionList = go.AddComponent<SessionListView>();
            sessionList.store = store; sessionList.chatPanel = chatPanel; sessionList.menu = menu;
            sessionList.content = sessionsList; sessionList.entryPrefab = sessionEntry; sessionList.groupPrefab = sessionGroup;
            sessionList.emptyHint = sessionsEmpty.gameObject;

            var controls = go.AddComponent<PanelControls>();
            controls.hud = hud; controls.client = client; controls.menu = menu; controls.chatPanel = chatPanel;
            controls.recordLabel = recordLabel; controls.recordKnob = recordKnob; controls.stateLabel = stateLabel;
            controls.recColor = Rec; controls.pausedColor = Paused;
            controls.minimizeLabel = minimizeButton.GetComponentInChildren<Text>();
            controls.inputField = inputField; controls.sendButton = sendButton; controls.phaseLabel = phaseLabel; controls.phaseDot = phaseDot;
            controls.micMeter = meterFill;
            controls.hiddenWhenMinimized = new GameObject[0];
            var holdToTalk = talkButton.gameObject.AddComponent<HoldToTalkButton>();
            holdToTalk.client = client; holdToTalk.background = talkButton.GetComponent<Image>();
            holdToTalk.idleColor = GhostButton.normalColor; holdToTalk.activeColor = Rec;
            holdToTalk.background.color = holdToTalk.idleColor;
            talkButton.transition = Selectable.Transition.None; // HoldToTalkButton colours it (red while recording)

            UnityEventTools.AddPersistentListener(tab.onClick, new UnityAction(menu.Toggle));
            UnityEventTools.AddPersistentListener(backdropButton.onClick, new UnityAction(menu.Close));
            UnityEventTools.AddPersistentListener(miSettings.onClick, new UnityAction(menu.OpenSettings));
            UnityEventTools.AddPersistentListener(miDebug.onClick, new UnityAction(menu.OpenDebug));
            UnityEventTools.AddPersistentListener(miControls.onClick, new UnityAction(menu.OpenControls));
            UnityEventTools.AddPersistentListener(controlsClose.onClick, new UnityAction(menu.CloseControls));
            UnityEventTools.AddPersistentListener(recordButton.onClick, new UnityAction(controls.ToggleRecord));
            UnityEventTools.AddPersistentListener(sendButton.onClick, new UnityAction(controls.Send));
            UnityEventTools.AddPersistentListener(scrim.onClick, new UnityAction(lightbox.Close));
            UnityEventTools.AddPersistentListener(lbClose.onClick, new UnityAction(lightbox.Close));
            UnityEventTools.AddPersistentListener(lbPrev.onClick, new UnityAction(lightbox.Prev));
            UnityEventTools.AddPersistentListener(lbNext.onClick, new UnityAction(lightbox.Next));
            UnityEventTools.AddPersistentListener(lbZoomIn.onClick, new UnityAction(lightbox.ZoomIn));
            UnityEventTools.AddPersistentListener(lbZoomOut.onClick, new UnityAction(lightbox.ZoomOut));
            UnityEventTools.AddPersistentListener(minimizeButton.onClick, new UnityAction(controls.ToggleMinimized));
            UnityEventTools.AddPersistentListener(miPin.onClick, new UnityAction(menu.TogglePin));
            UnityEventTools.AddPersistentListener(miTextSize.onClick, new UnityAction(menu.CycleTextSize));
            UnityEventTools.AddPersistentListener(miNewSession.onClick, new UnityAction(menu.NewSession));
            UnityEventTools.AddPersistentListener(miPurge.onClick, new UnityAction(menu.Purge));
            UnityEventTools.AddPersistentListener(debugClose.onClick, new UnityAction(menu.CloseDebug));
            UnityEventTools.AddPersistentListener(save.onClick, new UnityAction(settings.Save));
            UnityEventTools.AddPersistentListener(newSession.onClick, new UnityAction(settings.NewSession));
            UnityEventTools.AddPersistentListener(purge.onClick, new UnityAction(settings.Purge));
            UnityEventTools.AddPersistentListener(reset.onClick, new UnityAction(settings.ResetToDefaults));
            UnityEventTools.AddPersistentListener(close.onClick, new UnityAction(settings.Hide));

            settingsRoot.gameObject.SetActive(false);
            debugRoot.gameObject.SetActive(false);
            controlsRoot.gameObject.SetActive(false);
            lightRoot.gameObject.SetActive(false);
            menuRoot.gameObject.SetActive(false);
            drawerGroup.alpha = 0f; drawerGroup.interactable = drawerGroup.blocksRaycasts = false;
            mrt.anchoredPosition = new Vector2(0f, PanelH); // PanelDrawer slides it down from here
            backdrop.gameObject.SetActive(false);
            listening.gameObject.SetActive(false);
            thinking.gameObject.SetActive(false);

            // Whole window drops down from a small arrow tab above the panel's top-right corner (REC dot rides on the tab,
            // so it stays visible while collapsed). Window = everything built above; WindowClip hides it once slid up.
            var windowClip = new GameObject("WindowClip", typeof(RectTransform), typeof(RectMask2D));
            windowClip.transform.SetParent(go.transform, false);
            windowClip.GetComponent<RectTransform>().Stretch();
            var window = new GameObject("Window", typeof(RectTransform), typeof(CanvasGroup));
            window.transform.SetParent(windowClip.transform, false);
            var wrt = window.GetComponent<RectTransform>();
            wrt.Stretch();
            var built = new System.Collections.Generic.List<Transform>();
            foreach (Transform child in go.transform) if (child != windowClip.transform) built.Add(child);
            foreach (var child in built) child.SetParent(window.transform, false);
            var windowTab = MakeButton(go.transform, "WindowTab", "", FontCaption, sprite: RoundedM);
            var wtrt = windowTab.GetComponent<RectTransform>();
            wtrt.anchorMin = wtrt.anchorMax = new Vector2(1f, 1f); wtrt.pivot = new Vector2(1f, 0f);
            wtrt.anchoredPosition = new Vector2(0f, Space1); wtrt.sizeDelta = new Vector2(TabW, TabH);
            recDot.transform.SetParent(windowTab.transform, false);
            recDot.AnchorLeftCentre(14f, 12f, 12f);
            var windowChevron = Chevron(windowTab.transform, TextMain);
            windowChevron.anchorMin = windowChevron.anchorMax = new Vector2(1f, 0.5f); windowChevron.pivot = new Vector2(0.5f, 0.5f);
            windowChevron.anchoredPosition = new Vector2(-22f, 0f); windowChevron.sizeDelta = new Vector2(16f, 16f);
            var windowDrawer = go.AddComponent<PanelDrawer>();
            windowDrawer.drawer = wrt; windowDrawer.group = window.GetComponent<CanvasGroup>(); windowDrawer.chevron = windowChevron;
            windowDrawer.hiddenOffset = PanelH; windowDrawer.deactivateWhenClosed = false; windowDrawer.prefsKey = "megamind.windowOpen";
            windowDrawer.duration = 0.2f;
            controls.window = windowDrawer;
            UnityEventTools.AddPersistentListener(windowTab.onClick, new UnityAction(controls.ToggleMinimized));
            return go;
        }

        static Transform FindHead()
        {
            var eye = GameObject.Find("CenterEyeAnchor");
            if (eye) return eye.transform;
            return Camera.main ? Camera.main.transform : null;
        }

        // Controller-ray UI input. Uses Meta's OVRInputModule/OVRRaycaster when the SDK provides them
        // (looked up by name so this file compiles even if they move), else Unity's StandaloneInputModule.
        static void EnsureEventSystem(Transform head)
        {
            var es = UnityEngine.Object.FindFirstObjectByType<EventSystem>();
            if (es == null)
            {
                var esGo = new GameObject("EventSystem", typeof(EventSystem));
                es = esGo.GetComponent<EventSystem>();
                Undo.RegisterCreatedObjectUndo(esGo, "Create EventSystem");
            }
            var ovrModule = FindType("OVRInputModule");
            var ovrRaycaster = FindType("OVRRaycaster");
            if (ovrModule != null)
            {
                // Re-configured on every run (the EventSystem survives re-runs). The visible rays + cursor are the
                // OVRRayHelpers that Megamind > 1 puts on the right controller and right hand (OVRInputModule input
                // sources: trigger / index pinch clicks). rayTransform is the module's fallback when neither is active.
                var module = es.GetComponent(ovrModule);
                if (module == null) module = es.gameObject.AddComponent(ovrModule);
                var rightRay = GameObject.Find("RightControllerAnchor");
                if (rightRay == null) rightRay = GameObject.Find("RightHandAnchor");
                var rayField = ovrModule.GetField("rayTransform");
                if (rayField != null && rightRay) rayField.SetValue(module, rightRay.transform);
                // Default fallback click is A, which CaptureHud uses for the capture toggle: click with the trigger.
                var clickField = ovrModule.GetField("joyPadClickButton");
                if (clickField != null && clickField.FieldType.IsEnum && Enum.IsDefined(clickField.FieldType, "PrimaryIndexTrigger"))
                    clickField.SetValue(module, Enum.Parse(clickField.FieldType, "PrimaryIndexTrigger"));
                EditorUtility.SetDirty(module);
                var std = es.GetComponent<StandaloneInputModule>();
                if (std) UnityEngine.Object.DestroyImmediate(std);
                if (!rightRay)
                    Debug.LogWarning("[Megamind] no RightControllerAnchor/RightHandAnchor: OVRInputModule.rayTransform not set.");
            }
            else if (ovrModule == null && es.GetComponent<BaseInputModule>() == null)
            {
                es.gameObject.AddComponent<StandaloneInputModule>();
            }
            var panel = GameObject.Find("MegamindPanel");
            if (ovrRaycaster != null && panel && panel.GetComponent(ovrRaycaster) == null)
            {
                panel.AddComponent(ovrRaycaster);
                var gr = panel.GetComponent<GraphicRaycaster>();
                if (gr) UnityEngine.Object.DestroyImmediate(gr);
            }
            if (ovrModule == null)
                Debug.LogWarning("[Megamind] OVRInputModule not found; the panel has no controller ray input. Add Meta's 'UI Set' building block or Interaction SDK ray + canvas pointer.");
        }

        static Type FindType(string name)
        {
            // OVRInputModule lives in UnityEngine.EventSystems, OVRRaycaster in the global namespace: match by simple name.
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(name);
                if (t != null) return t;
            }
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type[] types;
                try { types = asm.GetTypes(); } catch (System.Reflection.ReflectionTypeLoadException e) { types = e.Types; }
                foreach (var t in types)
                    if (t != null && t.Name == name && typeof(UnityEngine.Object).IsAssignableFrom(t)) return t;
            }
            return null;
        }

        // ---------- art (generated sprites) ----------

        static void LoadArt()
        {
            RoundedL = Art("rounded-24", RoundedRectTexture(24), 24);
            RoundedM = Art("rounded-16", RoundedRectTexture(16), 16);
            RoundedS = Art("rounded-10", RoundedRectTexture(10), 10);
            Bubble = Art("bubble-20", RoundedRectTexture(20), 20);
            Circle = Art("circle-64", CircleTexture(64), 0);
        }

        // Writes the texture as a PNG under Assets/Megamind/Art and imports it as a (9-sliced) UI sprite. Re-run safe.
        static Sprite Art(string name, Texture2D tex, int border)
        {
            string path = ArtDir + "/" + name + ".png";
            File.WriteAllBytes(path, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
            var ti = (TextureImporter)AssetImporter.GetAtPath(path);
            ti.textureType = TextureImporterType.Sprite; ti.spriteImportMode = SpriteImportMode.Single;
            ti.spriteBorder = new Vector4(border, border, border, border); ti.spritePixelsPerUnit = 100f;
            ti.mipmapEnabled = false; ti.alphaIsTransparency = true; ti.sRGBTexture = true;
            ti.filterMode = FilterMode.Bilinear; ti.textureCompression = TextureImporterCompression.Uncompressed;
            ti.SaveAndReimport();
            var sprite = AssetDatabase.LoadAssetAtPath<Sprite>(path);
            if (sprite == null) throw new InvalidOperationException("[Megamind] sprite import failed: " + path);
            return sprite;
        }

        // ---------- prefabs ----------

        // Row (full width) -> [Avatar] + Bubble (Image + padding + ChatBubbleFit) -> Label. See ChatBubble.cs.
        static ChatBubble BubblePrefab(string name, bool user)
        {
            var row = new GameObject(name, typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(ChatBubble));
            var h = row.GetComponent<HorizontalLayoutGroup>();
            h.childAlignment = user ? TextAnchor.UpperRight : TextAnchor.UpperLeft; h.spacing = Space1;
            h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = false; h.childForceExpandHeight = false;
            if (!user)
            {
                var avatar = MakeImage(row.transform, "Avatar", Secondary, Circle);
                avatar.type = Image.Type.Simple;
                var ale = avatar.gameObject.AddComponent<LayoutElement>();
                ale.minWidth = ale.preferredWidth = AvatarSize; ale.minHeight = ale.preferredHeight = AvatarSize;
                var glyph = MakeLabel(avatar.transform, "Glyph", "M", 18, PanelBgSolid, TextAnchor.MiddleCenter);
                glyph.fontStyle = FontStyle.Bold; glyph.rectTransform.Stretch();
            }
            var bubble = MakeImage(row.transform, "Bubble", user ? UserBubble : AssistantBubble, Bubble);
            var pad = bubble.gameObject.AddComponent<VerticalLayoutGroup>();
            pad.padding = new RectOffset((int)Space2, (int)Space2, 12, 12);
            pad.childControlWidth = true; pad.childControlHeight = true; pad.childForceExpandWidth = true; pad.childForceExpandHeight = false;
            var label = MakeLabel(bubble.transform, "Label", "", FontBody, user ? TextOnAccent : TextMain, TextAnchor.UpperLeft);
            label.supportRichText = false; label.lineSpacing = 1.1f;
            var fit = bubble.gameObject.AddComponent<ChatBubbleFit>();
            fit.label = label; fit.maxWidth = BubbleMaxW; fit.horizontalPadding = 2f * Space2;
            var cb = row.GetComponent<ChatBubble>();
            cb.label = label; cb.background = bubble;
            return SavePrefab(row).GetComponent<ChatBubble>();
        }

        // Small centred chip: "open menu -> menu opened", session switches, etc.
        static ChatBubble SystemChipPrefab()
        {
            var row = new GameObject("SystemChip", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(ChatBubble));
            var h = row.GetComponent<HorizontalLayoutGroup>();
            h.childAlignment = TextAnchor.UpperCenter;
            h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = false; h.childForceExpandHeight = false;
            var chip = MakeImage(row.transform, "Chip", SystemChip, RoundedS);
            var pad = chip.gameObject.AddComponent<VerticalLayoutGroup>();
            pad.padding = new RectOffset(12, 12, 6, 6);
            pad.childControlWidth = true; pad.childControlHeight = true; pad.childForceExpandWidth = true; pad.childForceExpandHeight = false;
            var label = MakeLabel(chip.transform, "Label", "", FontCaption, TextDim, TextAnchor.MiddleCenter);
            label.supportRichText = false;
            var fit = chip.gameObject.AddComponent<ChatBubbleFit>();
            fit.label = label; fit.maxWidth = PanelW - 2f * ContentPad - Space4; fit.horizontalPadding = 24f;
            var cb = row.GetComponent<ChatBubble>();
            cb.label = label; cb.background = chip;
            return SavePrefab(row).GetComponent<ChatBubble>();
        }

        // Caption under an answer ("1.8 s · high confidence"), indented to line up with the bubble text.
        static RectTransform MetaLinePrefab()
        {
            var go = new GameObject("MetaLine", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.padding = new RectOffset((int)(AvatarSize + Space1 + Space2), 0, 0, 0);
            h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = true; h.childForceExpandHeight = false;
            var label = MakeLabel(go.transform, "Label", "", FontMono, TextMuted, TextAnchor.MiddleLeft);
            label.gameObject.AddComponent<LayoutElement>().minHeight = 20f;
            return SavePrefab(go).GetComponent<RectTransform>();
        }

        // Drawer chat row: [Title ............][…] and a hidden second line [Rename][Delete] (SessionListView toggles it).
        static RectTransform SessionEntryPrefab()
        {
            var row = new GameObject("SessionEntry", typeof(RectTransform), typeof(VerticalLayoutGroup));
            var v = row.GetComponent<VerticalLayoutGroup>();
            v.spacing = 4f; v.childControlWidth = true; v.childControlHeight = true; v.childForceExpandWidth = true; v.childForceExpandHeight = false;
            var main = new GameObject("Main", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            main.transform.SetParent(row.transform, false);
            var h = main.GetComponent<HorizontalLayoutGroup>();
            h.spacing = 4f; h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = false; h.childForceExpandHeight = true;
            main.GetComponent<LayoutElement>().preferredHeight = 44f;
            var title = MakeButton(main.transform, "Title", "New chat", FontCaption, sprite: RoundedS);
            title.gameObject.AddComponent<LayoutElement>().flexibleWidth = 1f;
            var tl = title.GetComponentInChildren<Text>();
            tl.alignment = TextAnchor.MiddleLeft; tl.horizontalOverflow = HorizontalWrapMode.Overflow; tl.verticalOverflow = VerticalWrapMode.Truncate;
            tl.rectTransform.offsetMin = new Vector2(12f, 0f); tl.rectTransform.offsetMax = new Vector2(-6f, 0f);
            title.gameObject.AddComponent<RectMask2D>(); // clip long titles
            var more = MakeButton(main.transform, "More", "…", FontButton, sprite: RoundedS);
            var mle = more.gameObject.AddComponent<LayoutElement>(); mle.preferredWidth = mle.minWidth = 36f;
            var actions = new GameObject("Actions", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            actions.transform.SetParent(row.transform, false);
            var ah = actions.GetComponent<HorizontalLayoutGroup>();
            ah.spacing = 4f; ah.childControlWidth = true; ah.childControlHeight = true; ah.childForceExpandWidth = true;
            actions.GetComponent<LayoutElement>().preferredHeight = 40f;
            MakeButton(actions.transform, "Rename", "Rename", FontCaption, sprite: RoundedS);
            MakeButton(actions.transform, "Delete", "Delete", FontCaption, DangerButton, RoundedS);
            actions.SetActive(false);
            return SavePrefab(row).GetComponent<RectTransform>();
        }

        static Text SessionGroupPrefab()
        {
            var t = MakeLabel(null, "SessionGroup", "Today", 14, TextMuted, TextAnchor.LowerLeft);
            t.fontStyle = FontStyle.Bold;
            t.PreferredHeight(28f);
            t.rectTransform.offsetMin = new Vector2(12f, 0f);
            return SavePrefab(t.gameObject).GetComponent<Text>();
        }

        // Card: rounded thumbnail (Mask) + age + caption snippet; the whole card is a Button (opens the Lightbox).
        static CitationCard CitationCardPrefab()
        {
            var root = MakeImage(null, "CitationCard", Color.white, RoundedM);
            root.raycastTarget = true;
            var button = root.gameObject.AddComponent<Button>(); button.targetGraphic = root; button.colors = GhostButton;
            var le = root.gameObject.AddComponent<LayoutElement>(); le.preferredWidth = le.minWidth = ThumbW + 2f * Space1; le.preferredHeight = ThumbH + 76f;
            var frame = MakeImage(root.transform, "Frame", Color.white, RoundedS);
            frame.gameObject.AddComponent<Mask>().showMaskGraphic = false;
            frame.rectTransform.anchorMin = new Vector2(0f, 1f); frame.rectTransform.anchorMax = new Vector2(1f, 1f); frame.rectTransform.pivot = new Vector2(0.5f, 1f);
            frame.rectTransform.offsetMin = new Vector2(Space1, -(Space1 + ThumbH)); frame.rectTransform.offsetMax = new Vector2(-Space1, -Space1);
            var img = new GameObject("Image", typeof(RectTransform), typeof(RawImage));
            img.transform.SetParent(frame.transform, false);
            img.GetComponent<RectTransform>().Stretch();
            img.GetComponent<RawImage>().color = ThumbPlaceholder; img.GetComponent<RawImage>().raycastTarget = false;
            var age = MakeLabel(frame.transform, "Age", "n min ago", 15, TextMain, TextAnchor.LowerRight);
            age.rectTransform.Stretch(); age.rectTransform.offsetMin = new Vector2(Space1, 6f); age.rectTransform.offsetMax = new Vector2(-Space1, -6f);
            age.gameObject.AddComponent<Shadow>().effectColor = Color.black;
            var cap = MakeLabel(root.transform, "Caption", "", 15, TextDim, TextAnchor.UpperLeft);
            cap.verticalOverflow = VerticalWrapMode.Truncate; cap.lineSpacing = 1.05f;
            cap.rectTransform.anchorMin = Vector2.zero; cap.rectTransform.anchorMax = new Vector2(1f, 0f); cap.rectTransform.pivot = new Vector2(0.5f, 0f);
            cap.rectTransform.offsetMin = new Vector2(Space1 + 2f, Space1); cap.rectTransform.offsetMax = new Vector2(-(Space1 + 2f), Space1 + 56f);
            var card = root.gameObject.AddComponent<CitationCard>();
            card.image = img.GetComponent<RawImage>(); card.age = age; card.caption = cap; card.button = button;
            return SavePrefab(root.gameObject).GetComponent<CitationCard>();
        }

        // Horizontal strip of cards (scrolls sideways when more than fit).
        static RectTransform CardStripPrefab()
        {
            var go = new GameObject("CardStrip", typeof(RectTransform), typeof(ScrollRect), typeof(RectMask2D), typeof(LayoutElement));
            go.GetComponent<LayoutElement>().preferredHeight = ThumbH + 76f;
            var scroll = go.GetComponent<ScrollRect>();
            scroll.vertical = false; scroll.horizontal = true; scroll.movementType = ScrollRect.MovementType.Clamped; scroll.scrollSensitivity = 30f;
            var content = new GameObject("Content", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(ContentSizeFitter));
            content.transform.SetParent(go.transform, false);
            var crt = content.GetComponent<RectTransform>();
            crt.anchorMin = new Vector2(0f, 0f); crt.anchorMax = new Vector2(0f, 1f); crt.pivot = new Vector2(0f, 0.5f);
            crt.offsetMin = Vector2.zero; crt.offsetMax = Vector2.zero;
            var h = content.GetComponent<HorizontalLayoutGroup>();
            h.spacing = Space1; h.childAlignment = TextAnchor.MiddleLeft;
            h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = false; h.childForceExpandHeight = true;
            content.GetComponent<ContentSizeFitter>().horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.viewport = go.GetComponent<RectTransform>(); scroll.content = crt;
            return SavePrefab(go).GetComponent<RectTransform>();
        }

        // Under an answer: meta caption ("0.9 s first · 2.4 s · high") + Copy + Retry ghost buttons, indented like MetaLine.
        static RectTransform TurnActionsPrefab()
        {
            var go = new GameObject("TurnActions", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            go.GetComponent<LayoutElement>().preferredHeight = 32f;
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.padding = new RectOffset((int)(AvatarSize + Space1 + Space2), 0, 0, 0); h.spacing = Space1; h.childAlignment = TextAnchor.MiddleLeft;
            h.childControlWidth = true; h.childControlHeight = true; h.childForceExpandWidth = false; h.childForceExpandHeight = true;
            var label = MakeLabel(go.transform, "Label", "", FontMono, TextMuted, TextAnchor.MiddleLeft);
            label.gameObject.AddComponent<LayoutElement>().flexibleWidth = 1f;
            var copy = MakeButton(go.transform, "Copy", "Copy", 14, sprite: RoundedS);
            var cle = copy.gameObject.AddComponent<LayoutElement>(); cle.preferredWidth = 72f; cle.minWidth = 72f;
            var retry = MakeButton(go.transform, "Retry", "Retry", 14, sprite: RoundedS);
            var rle = retry.gameObject.AddComponent<LayoutElement>(); rle.preferredWidth = 72f; rle.minWidth = 72f;
            return SavePrefab(go).GetComponent<RectTransform>();
        }

        static RectTransform ThumbRowPrefab()
        {
            var go = new GameObject("ThumbRow", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.spacing = Space1; h.childAlignment = TextAnchor.MiddleLeft;
            h.childControlWidth = false; h.childControlHeight = false; h.childForceExpandWidth = false; h.childForceExpandHeight = false;
            go.GetComponent<LayoutElement>().preferredHeight = ThumbH;
            return SavePrefab(go).GetComponent<RectTransform>();
        }

        // Rounded frame (Image + Mask clips the RawImage to the corners) with the photo and an age caption.
        static RectTransform ThumbPrefab()
        {
            var frame = MakeImage(null, "Thumb", Color.white, RoundedS);
            frame.gameObject.AddComponent<Mask>().showMaskGraphic = false;
            frame.rectTransform.sizeDelta = new Vector2(ThumbW, ThumbH);
            var le = frame.gameObject.AddComponent<LayoutElement>(); le.preferredWidth = ThumbW; le.preferredHeight = ThumbH;
            var img = new GameObject("Image", typeof(RectTransform), typeof(RawImage));
            img.transform.SetParent(frame.transform, false);
            img.GetComponent<RectTransform>().Stretch();
            // Placeholder tint while the JPEG downloads; ChatPanel.LoadThumb resets it to white (RawImage.color multiplies the texture).
            img.GetComponent<RawImage>().color = ThumbPlaceholder;
            img.GetComponent<RawImage>().raycastTarget = false;
            var label = MakeLabel(frame.transform, "Age", "n min ago", 15, TextMain, TextAnchor.LowerRight);
            label.rectTransform.Stretch(); label.rectTransform.offsetMin = new Vector2(Space1, 6f); label.rectTransform.offsetMax = new Vector2(-Space1, -6f);
            label.gameObject.AddComponent<Shadow>().effectColor = Color.black;
            return SavePrefab(frame.gameObject).GetComponent<RectTransform>();
        }

        static GameObject SavePrefab(GameObject go)
        {
            string path = PrefabDir + "/" + go.name + ".prefab";
            var prefab = PrefabUtility.SaveAsPrefabAsset(go, path);
            UnityEngine.Object.DestroyImmediate(go);
            return prefab;
        }

        static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            AssetDatabase.CreateFolder(Path.GetDirectoryName(path).Replace('\\', '/'), Path.GetFileName(path));
        }

        // ---------- uGUI helpers ----------

        static readonly Color PanelBgSolid = new Color(PanelBg.r, PanelBg.g, PanelBg.b, 1f);

        static Image MakeImage(Transform parent, string name, Color color, Sprite sprite)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            if (parent) go.transform.SetParent(parent, false);
            var img = go.GetComponent<Image>();
            img.color = color; img.sprite = sprite; img.type = Image.Type.Sliced; img.raycastTarget = false;
            return img;
        }

        static Text MakeLabel(Transform parent, string name, string text, int size, Color color, TextAnchor align)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Text));
            if (parent) go.transform.SetParent(parent, false);
            var t = go.GetComponent<Text>();
            t.font = UIFont; t.text = text; t.fontSize = size; t.color = color; t.alignment = align;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.verticalOverflow = VerticalWrapMode.Overflow; t.raycastTarget = false;
            return t;
        }

        // Button whose Image is white and whose colour (incl. hover / pressed / disabled) comes from the ColorBlock,
        // so the states are real tints rather than alpha-on-alpha. Default: MegamindTheme.GhostButton on RoundedM.
        static Button MakeButton(Transform parent, string name, string caption, int size, ColorBlock? colors = null, Sprite sprite = null)
        {
            var img = MakeImage(parent, name, Color.white, sprite != null ? sprite : RoundedM);
            img.raycastTarget = true;
            var b = img.gameObject.AddComponent<Button>();
            b.targetGraphic = img;
            var block = colors ?? GhostButton;
            b.colors = block;
            img.color = Color.white;
            bool accent = block.normalColor.a > 0.9f;
            var label = MakeLabel(img.transform, "Label", caption, size, accent ? TextOnAccent : TextMain, TextAnchor.MiddleCenter);
            label.rectTransform.Stretch();
            return b;
        }

        // 1-px separator line hugging the top or bottom edge of `parent`.
        static void Hairline(Transform parent, bool bottom)
        {
            var line = MakeImage(parent, "Hairline", Divider, null);
            var lrt = line.rectTransform;
            lrt.anchorMin = new Vector2(0f, bottom ? 0f : 1f); lrt.anchorMax = new Vector2(1f, bottom ? 0f : 1f);
            lrt.pivot = new Vector2(0.5f, bottom ? 0f : 1f);
            lrt.offsetMin = new Vector2(Space2, bottom ? 0f : -1f); lrt.offsetMax = new Vector2(-Space2, bottom ? 1f : 0f);
        }

        // Full-height sheet below the header (Settings / Debug / Controls): raised glass, same footprint for all.
        static Image Sheet(Transform parent, string name)
        {
            var sheet = MakeImage(parent, name, SurfaceRaised, RoundedL);
            sheet.raycastTarget = true; // swallow rays so the transcript underneath is not clickable
            sheet.rectTransform.Fill(new Vector2(0f, 0f), new Vector2(0f, -HeaderH));
            return sheet;
        }

        static Text SheetTitle(Transform sheet, string title)
        {
            var t = MakeLabel(sheet, "Title", title, FontTitle, TextMain, TextAnchor.MiddleLeft);
            t.rectTransform.AnchorTop(40f);
            t.rectTransform.offsetMin = new Vector2(Space3, -(Space3 + 40f)); t.rectTransform.offsetMax = new Vector2(-(Space3 + 96f + Space1), -Space3);
            return t;
        }

        static Button SheetClose(Transform sheet)
        {
            var b = MakeButton(sheet, "Close", "Close", FontCaption, sprite: RoundedS);
            var rt = b.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(1f, 1f); rt.pivot = new Vector2(1f, 1f);
            rt.anchoredPosition = new Vector2(-Space3, -Space3); rt.sizeDelta = new Vector2(96f, 40f);
            return b;
        }

        // 44 px round ghost button with a single glyph (lightbox prev / next / zoom).
        static Button RoundButton(Transform parent, string name, string glyph)
        {
            var b = MakeButton(parent, name, glyph, FontTitle, sprite: Circle);
            b.GetComponent<Image>().type = Image.Type.Simple;
            var rt = b.GetComponent<RectTransform>();
            rt.pivot = new Vector2(0.5f, 0.5f); rt.sizeDelta = new Vector2(44f, 44f);
            var c = b.colors; c.normalColor = new Color(1f, 1f, 1f, 0.18f); b.colors = c;
            return b;
        }

        static Transform ButtonRow(Transform parent, string name)
        {
            var row = new GameObject(name, typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            row.transform.SetParent(parent, false);
            var hlg = row.GetComponent<HorizontalLayoutGroup>(); hlg.spacing = Space1; hlg.childControlWidth = true; hlg.childControlHeight = true;
            row.GetComponent<LayoutElement>().preferredHeight = ControlH;
            return row.transform;
        }

        static void Spacer(Transform parent, float height)
        {
            var go = new GameObject("Spacer", typeof(RectTransform), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            go.GetComponent<LayoutElement>().preferredHeight = height;
        }

        const string ControlsText =
            "Controller\n" +
            "  Trigger (hold)  talk to Megamind; release = send\n" +
            "  Trigger on a button  click (no recording)\n" +
            "  A / X  record toggle (continuous capture on/off)\n" +
            "  B / Y (hold 0.5 s)  move the panel with your gaze\n\n" +
            "Hands\n" +
            "  Index pinch (hold)  talk; release = send\n" +
            "  Middle pinch  record toggle\n" +
            "  Index pinch on a button  click\n\n" +
            "Panel\n" +
            "  Record / Pause  continuous capture on/off\n" +
            "  Hold to talk  point, hold trigger or pinch, speak, release\n" +
            "  Ask Megamind…  field opens the keyboard; Send asks\n" +
            "  Thumbnail  tap to open it large; ‹ › prev/next, sticks pan/zoom, B/Y back\n" +
            "  Copy / Retry  under each answer\n" +
            "  Hide / arrow  fold the window up behind the top-right arrow (REC dot stays)\n" +
            "  Tab (top-right)  drawer: New chat, chats, Settings, Debug, Pin, Text size, Controls, Purge\n\n" +
            "Voice (hold trigger, say):\n  " + VoiceCommands.Hint;

        // Drawer row: small coloured disc with a one-letter glyph, then the caption. 48 px tall, left aligned.
        static Button DrawerEntry(Transform parent, string name, string glyph, Color glyphColor, string caption, ColorBlock? colors = null)
        {
            var b = MakeButton(parent, name, caption, FontCaption, colors, RoundedS);
            var label = b.GetComponentInChildren<Text>();
            label.alignment = TextAnchor.MiddleLeft;
            label.horizontalOverflow = HorizontalWrapMode.Overflow; label.verticalOverflow = VerticalWrapMode.Truncate;
            label.rectTransform.offsetMin = new Vector2(Space1 + 28f + Space1 + 2f, 0f);
            b.gameObject.AddComponent<LayoutElement>().preferredHeight = 48f;
            bool accent = (colors ?? GhostButton).normalColor.a > 0.9f;
            var icon = MakeImage(b.transform, "Icon", accent ? TextOnAccent : new Color(glyphColor.r, glyphColor.g, glyphColor.b, 0.22f), Circle);
            icon.type = Image.Type.Simple; icon.AnchorLeftCentre(Space1 + 2f, 28f, 28f);
            var g = MakeLabel(icon.transform, "Glyph", glyph, 15, accent ? Accent : glyphColor, TextAnchor.MiddleCenter);
            g.fontStyle = FontStyle.Bold; g.rectTransform.Stretch();
            return b;
        }

        // Scrollable list that takes whatever height the drawer has left (sessions, PR 3). Returns the content transform.
        static Transform DrawerList(Transform parent, string name)
        {
            var scrollGo = new GameObject(name, typeof(RectTransform), typeof(ScrollRect), typeof(RectMask2D), typeof(LayoutElement));
            scrollGo.transform.SetParent(parent, false);
            var le = scrollGo.GetComponent<LayoutElement>(); le.flexibleHeight = 1f; le.minHeight = 48f;
            var scroll = scrollGo.GetComponent<ScrollRect>();
            scroll.horizontal = false; scroll.movementType = ScrollRect.MovementType.Clamped; scroll.scrollSensitivity = 30f;
            var content = new GameObject("Content", typeof(RectTransform), typeof(VerticalLayoutGroup), typeof(ContentSizeFitter));
            content.transform.SetParent(scrollGo.transform, false);
            var crt = content.GetComponent<RectTransform>();
            crt.anchorMin = new Vector2(0f, 1f); crt.anchorMax = Vector2.one; crt.pivot = new Vector2(0.5f, 1f);
            crt.offsetMin = Vector2.zero; crt.offsetMax = Vector2.zero;
            var vlg = content.GetComponent<VerticalLayoutGroup>();
            vlg.padding = new RectOffset(0, 0, 4, 4); vlg.spacing = 4f;
            vlg.childControlWidth = true; vlg.childControlHeight = true; vlg.childForceExpandWidth = true; vlg.childForceExpandHeight = false;
            content.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.viewport = scrollGo.GetComponent<RectTransform>(); scroll.content = crt;
            return content.transform;
        }

        // Hairline as a layout row (1 px line with 8 px of air around it).
        static void HairlineRow(Transform parent)
        {
            var go = new GameObject("Divider", typeof(RectTransform), typeof(LayoutElement));
            go.transform.SetParent(parent, false);
            go.GetComponent<LayoutElement>().preferredHeight = Space2;
            var line = MakeImage(go.transform, "Hairline", Divider, null);
            line.rectTransform.anchorMin = new Vector2(0f, 0.5f); line.rectTransform.anchorMax = new Vector2(1f, 0.5f);
            line.rectTransform.offsetMin = new Vector2(Space1, -0.5f); line.rectTransform.offsetMax = new Vector2(-Space1, 0.5f);
        }

        // "v" chevron from two 2 px bars; PanelDrawer rotates it 180° while the drawer is open.
        static RectTransform Chevron(Transform parent, Color color)
        {
            var go = new GameObject("Chevron", typeof(RectTransform));
            go.transform.SetParent(parent, false);
            for (int i = 0; i < 2; i++)
            {
                var bar = MakeImage(go.transform, i == 0 ? "Left" : "Right", color, null);
                var rt = bar.rectTransform;
                rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f); rt.pivot = new Vector2(0.5f, 0.5f);
                rt.sizeDelta = new Vector2(10f, 2f);
                rt.anchoredPosition = new Vector2(i == 0 ? -3f : 3f, 0f);
                rt.localRotation = Quaternion.Euler(0f, 0f, i == 0 ? -40f : 40f);
            }
            return go.GetComponent<RectTransform>();
        }

        static InputField MakeInputField(Transform parent, string name, string placeholder)
        {
            var img = MakeImage(parent, name, FieldBg, RoundedS);
            img.raycastTarget = true;
            img.gameObject.AddComponent<LayoutElement>().preferredHeight = ControlH;
            var f = img.gameObject.AddComponent<InputField>();
            var text = MakeLabel(img.transform, "Text", "", FontButton, TextMain, TextAnchor.MiddleLeft);
            text.supportRichText = false;
            text.rectTransform.Stretch(); text.rectTransform.offsetMin = new Vector2(Space2, 4f); text.rectTransform.offsetMax = new Vector2(-Space2, -4f);
            var ph = MakeLabel(img.transform, "Placeholder", placeholder, FontButton, TextMuted, TextAnchor.MiddleLeft);
            ph.fontStyle = FontStyle.Italic;
            ph.rectTransform.Stretch(); ph.rectTransform.offsetMin = new Vector2(Space2, 4f); ph.rectTransform.offsetMax = new Vector2(-Space2, -4f);
            f.targetGraphic = img; f.textComponent = text; f.placeholder = ph; f.lineType = InputField.LineType.SingleLine;
            return f;
        }

        static void Stretch(this RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        static void Fill(this RectTransform rt, Vector2 offsetMin, Vector2 offsetMax)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;
        }

        static void AnchorTop(this Graphic g, float height) => AnchorTop(g.rectTransform, height);
        static void AnchorTop(this RectTransform rt, float height)
        {
            rt.anchorMin = new Vector2(0f, 1f); rt.anchorMax = Vector2.one; rt.pivot = new Vector2(0.5f, 1f);
            rt.offsetMin = new Vector2(0f, -height); rt.offsetMax = Vector2.zero;
        }

        static void AnchorBottom(this RectTransform rt, float height)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = new Vector2(1f, 0f); rt.pivot = new Vector2(0.5f, 0f);
            rt.offsetMin = Vector2.zero; rt.offsetMax = new Vector2(0f, height);
        }

        static void AnchorLeftCentre(this Graphic g, float x, float w, float h)
        {
            var rt = g.rectTransform;
            rt.anchorMin = new Vector2(0f, 0.5f); rt.anchorMax = new Vector2(0f, 0.5f); rt.pivot = new Vector2(0f, 0.5f);
            rt.anchoredPosition = new Vector2(x, 0f); rt.sizeDelta = new Vector2(w, h);
        }

        static void AnchorRightCentre(this RectTransform rt, float x, float w, float h)
        {
            rt.anchorMin = new Vector2(1f, 0.5f); rt.anchorMax = new Vector2(1f, 0.5f); rt.pivot = new Vector2(1f, 0.5f);
            rt.anchoredPosition = new Vector2(-x, 0f); rt.sizeDelta = new Vector2(w, h);
        }

        static void PreferredHeight(this Text t, float h)
        {
            var le = t.gameObject.GetComponent<LayoutElement>() ?? t.gameObject.AddComponent<LayoutElement>();
            le.preferredHeight = h;
        }
    }
}
