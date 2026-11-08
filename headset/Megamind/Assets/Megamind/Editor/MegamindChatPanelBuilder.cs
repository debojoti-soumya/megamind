// Editor-only. Stream 3: adds the voice-first chat panel to the capture scene made by MegamindSceneBuilder
// (Megamind > 1), creates the bubble/thumbnail prefabs and the generated sprites (Assets/Megamind/Art), and wires every
// public reference on ChatPanel / ChatClient / PanelAnchor / SettingsPanel / PanelMenu (header status strip + REC dot fed
// by CaptureHud, top-right Menu dropdown: Settings / Debug / Pin / Text size / Controls / New session / Purge; footer
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

            var old = GameObject.Find("MegamindPanel");
            if (old) Undo.DestroyObjectImmediate(old);

            var head = FindHead();
            var panel = BuildPanel(config, head, userBubble, assistantBubble, systemChip, metaLine, thumbRow, thumb);
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
                                     ChatBubble systemChip, RectTransform metaLine, RectTransform thumbRow, RectTransform thumb)
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
            statusLine.rectTransform.Fill(new Vector2(Space2 + 44f, 0f), new Vector2(-(Space2 + 88f + Space1 + 80f + Space1), 0f));
            var menuButton = MakeButton(header.transform, "MenuButton", "Menu", FontCaption, sprite: RoundedS);
            menuButton.GetComponent<RectTransform>().AnchorRightCentre(Space2, 88f, 40f);
            var minimizeButton = MakeButton(header.transform, "MinimizeButton", "Hide", FontCaption, sprite: RoundedS);
            minimizeButton.GetComponent<RectTransform>().AnchorRightCentre(Space2 + 88f + Space1, 80f, 40f);

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

            // Footer: state line ("Recording continuously" / "Paused") + listening/thinking indicators on top,
            // big Record|Pause / Hold to talk / Type buttons below (3 controls: no crowding).
            var footer = new GameObject("Footer", typeof(RectTransform));
            footer.transform.SetParent(go.transform, false);
            footer.GetComponent<RectTransform>().AnchorBottom(FooterH);
            Hairline(footer.transform, bottom: false);
            var stateLabel = MakeLabel(footer.transform, "StateLabel", "Paused", FontCaption, TextDim, TextAnchor.MiddleLeft);
            stateLabel.rectTransform.anchorMin = new Vector2(0f, 1f); stateLabel.rectTransform.anchorMax = Vector2.one; stateLabel.rectTransform.pivot = new Vector2(0.5f, 1f);
            stateLabel.rectTransform.offsetMin = new Vector2(Space3, -(FooterH - FooterButtonsH)); stateLabel.rectTransform.offsetMax = new Vector2(-280f, 0f);
            var indicators = new GameObject("Indicators", typeof(RectTransform));
            indicators.transform.SetParent(footer.transform, false);
            var irt = indicators.GetComponent<RectTransform>();
            irt.anchorMin = new Vector2(1f, 1f); irt.anchorMax = Vector2.one; irt.pivot = new Vector2(1f, 1f);
            irt.anchoredPosition = new Vector2(-Space3, 0f); irt.sizeDelta = new Vector2(260f, FooterH - FooterButtonsH);
            var listening = MakeImage(indicators.transform, "ListeningIndicator", Rec, Circle);
            listening.type = Image.Type.Simple; listening.AnchorLeftCentre(0f, 14f, 14f);
            var listeningLabel = MakeLabel(listening.transform, "Label", "listening…", FontCaption, TextMain, TextAnchor.MiddleLeft);
            listeningLabel.rectTransform.anchorMin = new Vector2(1f, 0.5f); listeningLabel.rectTransform.anchorMax = new Vector2(1f, 0.5f);
            listeningLabel.rectTransform.pivot = new Vector2(0f, 0.5f); listeningLabel.rectTransform.anchoredPosition = new Vector2(Space1, 0f);
            listeningLabel.rectTransform.sizeDelta = new Vector2(236f, 28f);
            var thinking = MakeLabel(indicators.transform, "ThinkingIndicator", "thinking…", FontCaption, Secondary, TextAnchor.MiddleRight);
            thinking.rectTransform.Stretch();
            var buttons = new GameObject("Buttons", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            buttons.transform.SetParent(footer.transform, false);
            var brt = buttons.GetComponent<RectTransform>();
            brt.AnchorBottom(FooterButtonsH); brt.offsetMin = new Vector2(Space2, Space2); brt.offsetMax = new Vector2(-Space2, FooterButtonsH);
            var bhlg = buttons.GetComponent<HorizontalLayoutGroup>();
            bhlg.spacing = Space1; bhlg.childControlWidth = true; bhlg.childControlHeight = true; bhlg.childForceExpandWidth = true;
            var recordButton = MakeButton(buttons.transform, "RecordButton", "Record", FontButton);
            var recordLabel = recordButton.GetComponentInChildren<Text>();
            recordLabel.rectTransform.offsetMin = new Vector2(Space3, 0f);
            var recordKnob = MakeImage(recordButton.transform, "Knob", Paused, Circle);
            recordKnob.type = Image.Type.Simple; recordKnob.AnchorLeftCentre(Space2, 14f, 14f);
            var talkButton = MakeButton(buttons.transform, "HoldToTalkButton", "Hold to talk", FontButton);
            var typeButton = MakeButton(buttons.transform, "TypeButton", "Type", FontButton);
            typeButton.gameObject.AddComponent<LayoutElement>().flexibleWidth = 0.6f;

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

            // Dropdown: transparent backdrop (click anywhere = close) + list under the Menu button. Built last so it draws on top.
            var backdrop = MakeImage(go.transform, "MenuBackdrop", new Color(0f, 0f, 0f, 0f), null);
            backdrop.raycastTarget = true;
            backdrop.rectTransform.Stretch();
            var backdropButton = backdrop.gameObject.AddComponent<Button>();
            backdropButton.transition = Selectable.Transition.None;
            var menuRoot = MakeImage(go.transform, "MenuDropdown", SurfaceRaised, RoundedM);
            var mrt = menuRoot.rectTransform;
            mrt.anchorMin = mrt.anchorMax = new Vector2(1f, 1f); mrt.pivot = new Vector2(1f, 1f);
            mrt.anchoredPosition = new Vector2(-Space1, -HeaderH); mrt.sizeDelta = new Vector2(296f, 0f);
            var mvlg = menuRoot.gameObject.AddComponent<VerticalLayoutGroup>();
            mvlg.padding = new RectOffset((int)Space1, (int)Space1, (int)Space1, (int)Space1); mvlg.spacing = 4f;
            mvlg.childControlWidth = true; mvlg.childControlHeight = true; mvlg.childForceExpandHeight = false;
            menuRoot.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            var miSettings = MenuEntry(menuRoot.transform, "Settings", "Settings");
            var miDebug = MenuEntry(menuRoot.transform, "Debug", "Debug");
            var miPin = MenuEntry(menuRoot.transform, "Pin", "Pin panel");
            var miTextSize = MenuEntry(menuRoot.transform, "TextSize", "Text size 1x");
            var miControls = MenuEntry(menuRoot.transform, "Controls", "Controls");
            var miNewSession = MenuEntry(menuRoot.transform, "NewSession", "New session");
            var miPurge = MenuEntry(menuRoot.transform, "Purge", "Purge memory", DangerButton);

            // Components
            var speaker = go.AddComponent<AudioSource>();
            speaker.playOnAwake = false; speaker.spatialBlend = 0f;

            var anchor = go.AddComponent<PanelAnchor>();
            anchor.config = config; anchor.head = head;

            var chatPanel = go.AddComponent<ChatPanel>();
            chatPanel.config = config;
            chatPanel.content = crt; chatPanel.scroll = scroll;
            chatPanel.userBubblePrefab = userBubble; chatPanel.assistantBubblePrefab = assistantBubble;
            chatPanel.systemChipPrefab = systemChip; chatPanel.metaLinePrefab = metaLine;
            chatPanel.thumbnailRowPrefab = thumbRow; chatPanel.thumbnailPrefab = thumb;
            chatPanel.listeningIndicator = listening.gameObject; chatPanel.thinkingIndicator = thinking.gameObject;
            chatPanel.emptyHint = emptyHint.gameObject;
            chatPanel.statusLine = statusLine; chatPanel.serverDot = dot; chatPanel.recDot = recDot;
            chatPanel.uploader = UnityEngine.Object.FindFirstObjectByType<FrameUploader>();
            var hud = UnityEngine.Object.FindFirstObjectByType<CaptureHud>();
            chatPanel.hud = hud;

            var client = go.AddComponent<ChatClient>();
            client.config = config; client.panel = chatPanel; client.speaker = speaker;
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

            var menu = go.AddComponent<PanelMenu>();
            menu.root = menuRoot.gameObject; menu.backdrop = backdrop.gameObject;
            menu.debugRoot = debugRoot.gameObject; menu.debugText = debugText; menu.controlsRoot = controlsRoot.gameObject;
            menu.pinLabel = miPin.GetComponentInChildren<Text>(); menu.textSizeLabel = miTextSize.GetComponentInChildren<Text>();
            menu.hud = hud; menu.settings = settings; menu.anchor = anchor; menu.chatPanel = chatPanel;
            client.menu = menu;
            client.panelHover = go.AddComponent<PanelHover>();

            var controls = go.AddComponent<PanelControls>();
            controls.hud = hud; controls.client = client; controls.menu = menu; controls.chatPanel = chatPanel;
            controls.recordLabel = recordLabel; controls.recordKnob = recordKnob; controls.stateLabel = stateLabel;
            controls.recColor = Rec; controls.pausedColor = Paused;
            controls.minimizeLabel = minimizeButton.GetComponentInChildren<Text>();
            controls.hiddenWhenMinimized = new[] { background.gameObject, scrollGo, emptyHint.gameObject, footer.gameObject };
            var holdToTalk = talkButton.gameObject.AddComponent<HoldToTalkButton>();
            holdToTalk.client = client; holdToTalk.background = talkButton.GetComponent<Image>();
            holdToTalk.idleColor = GhostButton.normalColor; holdToTalk.activeColor = Rec;
            holdToTalk.background.color = holdToTalk.idleColor;
            talkButton.transition = Selectable.Transition.None; // HoldToTalkButton colours it (red while recording)

            UnityEventTools.AddPersistentListener(menuButton.onClick, new UnityAction(menu.Toggle));
            UnityEventTools.AddPersistentListener(backdropButton.onClick, new UnityAction(menu.Close));
            UnityEventTools.AddPersistentListener(miSettings.onClick, new UnityAction(menu.OpenSettings));
            UnityEventTools.AddPersistentListener(miDebug.onClick, new UnityAction(menu.OpenDebug));
            UnityEventTools.AddPersistentListener(miControls.onClick, new UnityAction(menu.OpenControls));
            UnityEventTools.AddPersistentListener(controlsClose.onClick, new UnityAction(menu.CloseControls));
            UnityEventTools.AddPersistentListener(recordButton.onClick, new UnityAction(controls.ToggleRecord));
            UnityEventTools.AddPersistentListener(typeButton.onClick, new UnityAction(controls.OpenKeyboard));
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
            menuRoot.gameObject.SetActive(false);
            backdrop.gameObject.SetActive(false);
            listening.gameObject.SetActive(false);
            thinking.gameObject.SetActive(false);
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
            go.transform.SetParent(parent, false);
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
            "  Type  on-screen keyboard for a typed question\n" +
            "  Hide / Show  collapse to the header (REC dot stays)\n" +
            "  Menu  Settings, Debug, Pin, Text size, Controls, New session, Purge\n\n" +
            "Voice (hold trigger, say):\n  " + VoiceCommands.Hint;

        static Button MenuEntry(Transform parent, string name, string caption, ColorBlock? colors = null)
        {
            var b = MakeButton(parent, name, caption, FontButton, colors, RoundedS);
            b.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
            var lrt = b.GetComponentInChildren<Text>().rectTransform; lrt.offsetMin = new Vector2(Space2, 0f);
            b.gameObject.AddComponent<LayoutElement>().preferredHeight = ControlH;
            return b;
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
