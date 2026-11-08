// Editor-only. Stream 3: adds the voice-first chat panel to the capture scene made by MegamindSceneBuilder
// (Megamind > 1), creates the bubble/thumbnail prefabs, and wires every public reference on
// ChatPanel / ChatClient / PanelAnchor / SettingsPanel / PanelMenu (header status strip + REC dot fed by CaptureHud,
// top-right Menu dropdown: Settings / Debug / Pin / Text size / Controls / New session / Purge; footer Record|Pause /
// Hold to talk / Type / Hide controls, PanelControls + PanelHover). Run Megamind > 1, then 1b, then build.
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

namespace Megamind.EditorTools
{
    public static class MegamindChatPanelBuilder
    {
        const string Root = "Assets/Megamind";
        const string PrefabDir = Root + "/Prefabs";
        const float PanelW = 600f, PanelH = 720f, HeaderH = 56f, FooterH = 100f, FooterButtonsH = 60f;
        // uGUI Text rasterises glyphs at fontSize * dynamicPixelsPerUnit: at 1000 canvas px/m and ~25 px/deg on Quest 3 a
        // 1x glyph is upscaled ~1.4x and looks soft, so render the font atlas at 3x.
        const float DynamicPixelsPerUnit = 3f;
        // 3 thumbnails per row: 3 * 180 + 2 * 8 spacing = 556 < 600 - 28 padding. 4:3 like the 1280x960 capture.
        const float ThumbW = 180f, ThumbH = 135f;

        static readonly Color PanelBg = new Color(0.07f, 0.08f, 0.11f, 0.86f);
        static readonly Color HeaderBg = new Color(1f, 1f, 1f, 0.06f);
        static readonly Color TextMain = new Color(0.93f, 0.95f, 0.98f);
        static readonly Color TextDim = new Color(0.62f, 0.67f, 0.74f);
        static readonly Color UserBubbleColor = new Color(0.55f, 0.75f, 1f);
        static readonly Color ButtonBg = new Color(1f, 1f, 1f, 0.12f);
        static readonly Color FieldBg = new Color(0f, 0f, 0f, 0.35f);

        static Font _font;
        static Font UIFont => _font ??= Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        static Sprite Knob => AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/Knob.psd");
        static Sprite Rounded => AssetDatabase.GetBuiltinExtraResource<Sprite>("UI/Skin/UISprite.psd");

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

            EnsureFolder(Root); EnsureFolder(PrefabDir);
            var config = MegamindSceneBuilder.GetOrCreateConfig();
            var userBubble = BubblePrefab("UserBubble", TextAnchor.UpperRight, UserBubbleColor);
            var assistantBubble = BubblePrefab("AssistantBubble", TextAnchor.UpperLeft, TextMain);
            var thumbRow = ThumbRowPrefab();
            var thumb = ThumbPrefab();

            var old = GameObject.Find("MegamindPanel");
            if (old) Undo.DestroyObjectImmediate(old);

            var head = FindHead();
            var panel = BuildPanel(config, head, userBubble, assistantBubble, thumbRow, thumb);
            EnsureEventSystem(head);

            Undo.RegisterCreatedObjectUndo(panel, "Build Megamind panel");
            Selection.activeGameObject = panel;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
            AssetDatabase.SaveAssets();
            Debug.Log("[Megamind] Chat panel built. " + (head ? "Head = " + head.name : "No OVRCameraRig/CenterEyeAnchor found — assign PanelAnchor.head and FrameUploader.headPose manually."));
        }

        // ---------- scene ----------

        static GameObject BuildPanel(MegamindConfig config, Transform head, Text userBubble, Text assistantBubble, RectTransform thumbRow, RawImage thumb)
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

            var background = MakeImage(go.transform, "Background", PanelBg, Rounded);
            background.rectTransform.Stretch();
            background.raycastTarget = true; // the whole panel counts as "ray over UI" (push-to-talk suppression)

            // Header: REC dot (always visible, privacy), server dot, one-line status strip (CaptureHud.ShortStatus), Menu
            var header = MakeImage(go.transform, "Header", HeaderBg, Rounded);
            header.AnchorTop(HeaderH);
            var recDot = MakeImage(header.transform, "RecDot", new Color(1f, 0.25f, 0.25f), Knob);
            recDot.AnchorLeftCentre(12f, 18f, 18f);
            var dot = MakeImage(header.transform, "ServerDot", new Color(1f, 0.36f, 0.42f), Knob);
            dot.AnchorLeftCentre(36f, 10f, 10f);
            var statusLine = MakeLabel(header.transform, "StatusLine", "connecting…", 18, TextDim, TextAnchor.MiddleLeft);
            statusLine.horizontalOverflow = HorizontalWrapMode.Overflow; statusLine.verticalOverflow = VerticalWrapMode.Truncate;
            statusLine.rectTransform.Fill(new Vector2(54f, 0f), new Vector2(-172f, 0f));
            var menuButton = MakeButton(header.transform, "MenuButton", "Menu", 17);
            menuButton.GetComponent<RectTransform>().AnchorRightCentre(10f, 80f, 40f);
            var minimizeButton = MakeButton(header.transform, "MinimizeButton", "Hide", 17);
            minimizeButton.GetComponent<RectTransform>().AnchorRightCentre(96f, 70f, 40f);

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
            vlg.padding = new RectOffset(14, 14, 12, 12); vlg.spacing = 10f;
            vlg.childControlWidth = true; vlg.childControlHeight = true;
            vlg.childForceExpandWidth = true; vlg.childForceExpandHeight = false;
            content.GetComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.viewport = viewport.GetComponent<RectTransform>();
            scroll.content = crt;

            var emptyHint = MakeLabel(go.transform, "EmptyHint",
                "Hold the trigger (or pinch) and ask about something you saw.\n\nSay a short command instead of a question: " + VoiceCommands.Hint + ".",
                22, TextDim, TextAnchor.MiddleCenter);
            emptyHint.rectTransform.Fill(new Vector2(40f, FooterH), new Vector2(-40f, -HeaderH));

            // Footer: state line ("Recording continuously" / "Paused") + listening/thinking indicators on top,
            // big Record|Pause / Hold to talk / Type buttons below.
            var footer = MakeImage(go.transform, "Footer", HeaderBg, Rounded);
            footer.rectTransform.AnchorBottom(FooterH);
            var stateLabel = MakeLabel(footer.transform, "StateLabel", "Paused", 18, TextDim, TextAnchor.MiddleLeft);
            stateLabel.rectTransform.anchorMin = new Vector2(0f, 1f); stateLabel.rectTransform.anchorMax = Vector2.one; stateLabel.rectTransform.pivot = new Vector2(0.5f, 1f);
            stateLabel.rectTransform.offsetMin = new Vector2(18f, -(FooterH - FooterButtonsH)); stateLabel.rectTransform.offsetMax = new Vector2(-280f, 0f);
            var indicators = new GameObject("Indicators", typeof(RectTransform));
            indicators.transform.SetParent(footer.transform, false);
            var irt = indicators.GetComponent<RectTransform>();
            irt.anchorMin = new Vector2(1f, 1f); irt.anchorMax = Vector2.one; irt.pivot = new Vector2(1f, 1f);
            irt.anchoredPosition = new Vector2(-14f, 0f); irt.sizeDelta = new Vector2(260f, FooterH - FooterButtonsH);
            var listening = MakeImage(indicators.transform, "ListeningIndicator", new Color(1f, 0.3f, 0.3f), Knob);
            listening.AnchorLeftCentre(0f, 16f, 16f);
            var listeningLabel = MakeLabel(listening.transform, "Label", "listening…", 18, TextMain, TextAnchor.MiddleLeft);
            listeningLabel.rectTransform.anchorMin = new Vector2(1f, 0.5f); listeningLabel.rectTransform.anchorMax = new Vector2(1f, 0.5f);
            listeningLabel.rectTransform.pivot = new Vector2(0f, 0.5f); listeningLabel.rectTransform.anchoredPosition = new Vector2(8f, 0f);
            listeningLabel.rectTransform.sizeDelta = new Vector2(236f, 28f);
            var thinking = MakeLabel(indicators.transform, "ThinkingIndicator", "thinking…", 18, TextDim, TextAnchor.MiddleRight);
            thinking.rectTransform.Stretch();
            var buttons = new GameObject("Buttons", typeof(RectTransform), typeof(HorizontalLayoutGroup));
            buttons.transform.SetParent(footer.transform, false);
            var brt = buttons.GetComponent<RectTransform>();
            brt.AnchorBottom(FooterButtonsH); brt.offsetMin = new Vector2(10f, 6f); brt.offsetMax = new Vector2(-10f, FooterButtonsH - 2f);
            var bhlg = buttons.GetComponent<HorizontalLayoutGroup>();
            bhlg.spacing = 8f; bhlg.childControlWidth = true; bhlg.childControlHeight = true; bhlg.childForceExpandWidth = true;
            var recordButton = MakeButton(buttons.transform, "RecordButton", "Record", 21);
            var recordLabel = recordButton.GetComponentInChildren<Text>();
            recordLabel.rectTransform.offsetMin = new Vector2(30f, 0f);
            var recordKnob = MakeImage(recordButton.transform, "Knob", new Color(0.45f, 0.48f, 0.52f), Knob);
            recordKnob.AnchorLeftCentre(14f, 16f, 16f);
            var talkButton = MakeButton(buttons.transform, "HoldToTalkButton", "Hold to talk", 21);
            var typeButton = MakeButton(buttons.transform, "TypeButton", "Type", 21);
            typeButton.gameObject.AddComponent<LayoutElement>().flexibleWidth = 0.6f;

            // Settings sub-panel (inactive by default; SettingsPanel.Awake hides it)
            var settingsRoot = MakeImage(go.transform, "Settings", new Color(0.05f, 0.06f, 0.08f, 0.97f), Rounded);
            settingsRoot.GetComponent<RectTransform>().Fill(new Vector2(0f, 0f), new Vector2(0f, -HeaderH));
            var svlg = settingsRoot.gameObject.AddComponent<VerticalLayoutGroup>();
            svlg.padding = new RectOffset(18, 18, 16, 16); svlg.spacing = 10f;
            svlg.childControlWidth = true; svlg.childControlHeight = true; svlg.childForceExpandHeight = false;
            MakeLabel(settingsRoot.transform, "Title", "Settings", 26, TextMain, TextAnchor.MiddleLeft).PreferredHeight(36f);
            MakeLabel(settingsRoot.transform, "ServerLabel", "Server base URL (laptop Tailscale IP)", 17, TextDim, TextAnchor.MiddleLeft).PreferredHeight(24f);
            var serverField = MakeInputField(settingsRoot.transform, "ServerUrl", "http://100.x.y.z:8000");
            MakeLabel(settingsRoot.transform, "DeviceLabel", "Device id", 17, TextDim, TextAnchor.MiddleLeft).PreferredHeight(24f);
            var deviceField = MakeInputField(settingsRoot.transform, "DeviceId", "quest3-1");
            var row = new GameObject("Buttons", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            row.transform.SetParent(settingsRoot.transform, false);
            var hlg = row.GetComponent<HorizontalLayoutGroup>(); hlg.spacing = 8f; hlg.childControlWidth = true; hlg.childControlHeight = true;
            row.GetComponent<LayoutElement>().preferredHeight = 52f;
            var save = MakeButton(row.transform, "Save", "Save", 19);
            var newSession = MakeButton(row.transform, "NewSession", "New session", 19);
            var purge = MakeButton(row.transform, "Purge", "Purge memory", 19);
            var row2 = new GameObject("Buttons2", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            row2.transform.SetParent(settingsRoot.transform, false);
            var hlg2 = row2.GetComponent<HorizontalLayoutGroup>(); hlg2.spacing = 8f; hlg2.childControlWidth = true; hlg2.childControlHeight = true;
            row2.GetComponent<LayoutElement>().preferredHeight = 52f;
            var reset = MakeButton(row2.transform, "Reset", "Reset overrides", 19);
            var close = MakeButton(row2.transform, "Close", "Close", 19);
            var feedback = MakeLabel(settingsRoot.transform, "Feedback", "", 17, TextDim, TextAnchor.UpperLeft);
            feedback.PreferredHeight(72f);

            // Debug sub-panel (CaptureHud.FullReport in a scroll view), same footprint as Settings
            var debugRoot = MakeImage(go.transform, "Debug", new Color(0.05f, 0.06f, 0.08f, 0.97f), Rounded);
            debugRoot.GetComponent<RectTransform>().Fill(new Vector2(0f, 0f), new Vector2(0f, -HeaderH));
            var debugTitle = MakeLabel(debugRoot.transform, "Title", "Debug", 26, TextMain, TextAnchor.MiddleLeft);
            debugTitle.rectTransform.AnchorTop(44f); debugTitle.rectTransform.offsetMin = new Vector2(18f, -44f); debugTitle.rectTransform.offsetMax = new Vector2(-120f, -4f);
            var debugClose = MakeButton(debugRoot.transform, "Close", "Close", 19);
            var dcRt = debugClose.GetComponent<RectTransform>();
            dcRt.anchorMin = dcRt.anchorMax = new Vector2(1f, 1f); dcRt.pivot = new Vector2(1f, 1f);
            dcRt.anchoredPosition = new Vector2(-14f, -8f); dcRt.sizeDelta = new Vector2(96f, 40f);
            var debugScrollGo = new GameObject("Scroll", typeof(RectTransform), typeof(ScrollRect));
            debugScrollGo.transform.SetParent(debugRoot.transform, false);
            debugScrollGo.GetComponent<RectTransform>().Fill(new Vector2(14f, 12f), new Vector2(-14f, -56f));
            var debugScroll = debugScrollGo.GetComponent<ScrollRect>();
            debugScroll.horizontal = false; debugScroll.movementType = ScrollRect.MovementType.Clamped; debugScroll.scrollSensitivity = 30f;
            var debugViewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D));
            debugViewport.transform.SetParent(debugScrollGo.transform, false);
            debugViewport.GetComponent<RectTransform>().Stretch();
            var debugText = MakeLabel(debugViewport.transform, "Text", "", 18, TextMain, TextAnchor.UpperLeft);
            var dtRt = debugText.rectTransform;
            dtRt.anchorMin = new Vector2(0f, 1f); dtRt.anchorMax = Vector2.one; dtRt.pivot = new Vector2(0.5f, 1f);
            dtRt.offsetMin = new Vector2(4f, 0f); dtRt.offsetMax = new Vector2(-4f, 0f);
            debugText.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            debugScroll.viewport = debugViewport.GetComponent<RectTransform>(); debugScroll.content = dtRt;

            // Controls cheat-sheet sub-panel
            var controlsRoot = MakeImage(go.transform, "Controls", new Color(0.05f, 0.06f, 0.08f, 0.97f), Rounded);
            controlsRoot.GetComponent<RectTransform>().Fill(new Vector2(0f, 0f), new Vector2(0f, -HeaderH));
            var controlsTitle = MakeLabel(controlsRoot.transform, "Title", "Controls", 26, TextMain, TextAnchor.MiddleLeft);
            controlsTitle.rectTransform.AnchorTop(44f); controlsTitle.rectTransform.offsetMin = new Vector2(18f, -44f); controlsTitle.rectTransform.offsetMax = new Vector2(-120f, -4f);
            var controlsClose = MakeButton(controlsRoot.transform, "Close", "Close", 19);
            var ccRt = controlsClose.GetComponent<RectTransform>();
            ccRt.anchorMin = ccRt.anchorMax = new Vector2(1f, 1f); ccRt.pivot = new Vector2(1f, 1f);
            ccRt.anchoredPosition = new Vector2(-14f, -8f); ccRt.sizeDelta = new Vector2(96f, 40f);
            var controlsText = MakeLabel(controlsRoot.transform, "Text", ControlsText, 20, TextMain, TextAnchor.UpperLeft);
            controlsText.rectTransform.Fill(new Vector2(18f, 12f), new Vector2(-18f, -56f));
            controlsText.verticalOverflow = VerticalWrapMode.Truncate;

            // Dropdown: transparent backdrop (click anywhere = close) + list under the Menu button. Built last so it draws on top.
            var backdrop = MakeImage(go.transform, "MenuBackdrop", new Color(0f, 0f, 0f, 0f), null);
            backdrop.raycastTarget = true;
            backdrop.rectTransform.Stretch();
            var backdropButton = backdrop.gameObject.AddComponent<Button>();
            backdropButton.transition = Selectable.Transition.None;
            var menuRoot = MakeImage(go.transform, "MenuDropdown", new Color(0.05f, 0.06f, 0.08f, 0.98f), Rounded);
            var mrt = menuRoot.rectTransform;
            mrt.anchorMin = mrt.anchorMax = new Vector2(1f, 1f); mrt.pivot = new Vector2(1f, 1f);
            mrt.anchoredPosition = new Vector2(-8f, -HeaderH - 4f); mrt.sizeDelta = new Vector2(280f, 0f);
            var mvlg = menuRoot.gameObject.AddComponent<VerticalLayoutGroup>();
            mvlg.padding = new RectOffset(8, 8, 8, 8); mvlg.spacing = 6f;
            mvlg.childControlWidth = true; mvlg.childControlHeight = true; mvlg.childForceExpandHeight = false;
            menuRoot.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            var miSettings = MenuEntry(menuRoot.transform, "Settings", "Settings");
            var miDebug = MenuEntry(menuRoot.transform, "Debug", "Debug");
            var miPin = MenuEntry(menuRoot.transform, "Pin", "Pin panel");
            var miTextSize = MenuEntry(menuRoot.transform, "TextSize", "Text size 1x");
            var miControls = MenuEntry(menuRoot.transform, "Controls", "Controls");
            var miNewSession = MenuEntry(menuRoot.transform, "NewSession", "New session");
            var miPurge = MenuEntry(menuRoot.transform, "Purge", "Purge memory");

            // Components
            var speaker = go.AddComponent<AudioSource>();
            speaker.playOnAwake = false; speaker.spatialBlend = 0f;

            var anchor = go.AddComponent<PanelAnchor>();
            anchor.config = config; anchor.head = head;

            var chatPanel = go.AddComponent<ChatPanel>();
            chatPanel.config = config;
            chatPanel.content = crt; chatPanel.scroll = scroll;
            chatPanel.userBubblePrefab = userBubble; chatPanel.assistantBubblePrefab = assistantBubble;
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
            controls.minimizeLabel = minimizeButton.GetComponentInChildren<Text>();
            controls.hiddenWhenMinimized = new[] { background.gameObject, scrollGo, emptyHint.gameObject, footer.gameObject };
            var holdToTalk = talkButton.gameObject.AddComponent<HoldToTalkButton>();
            holdToTalk.client = client; holdToTalk.background = talkButton.GetComponent<Image>();
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

        // ---------- assets ----------


        static Text BubblePrefab(string name, TextAnchor align, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Text), typeof(LayoutElement), typeof(Shadow));
            var t = go.GetComponent<Text>();
            t.font = UIFont; t.fontSize = 26; t.color = color; t.alignment = align;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.verticalOverflow = VerticalWrapMode.Overflow;
            t.supportRichText = false;
            go.GetComponent<Shadow>().effectColor = new Color(0f, 0f, 0f, 0.6f);
            go.GetComponent<LayoutElement>().minHeight = 32f;
            return SavePrefab(go).GetComponent<Text>();
        }

        static RectTransform ThumbRowPrefab()
        {
            var go = new GameObject("ThumbRow", typeof(RectTransform), typeof(HorizontalLayoutGroup), typeof(LayoutElement));
            var h = go.GetComponent<HorizontalLayoutGroup>();
            h.spacing = 8f; h.childAlignment = TextAnchor.MiddleLeft;
            h.childControlWidth = false; h.childControlHeight = false; h.childForceExpandWidth = false; h.childForceExpandHeight = false;
            go.GetComponent<LayoutElement>().preferredHeight = ThumbH;
            return SavePrefab(go).GetComponent<RectTransform>();
        }

        static RawImage ThumbPrefab()
        {
            var go = new GameObject("Thumb", typeof(RectTransform), typeof(RawImage), typeof(LayoutElement));
            go.GetComponent<RectTransform>().sizeDelta = new Vector2(ThumbW, ThumbH);
            var le = go.GetComponent<LayoutElement>(); le.preferredWidth = ThumbW; le.preferredHeight = ThumbH;
            // Placeholder tint while the JPEG downloads; ChatPanel.LoadThumb resets it to white (RawImage.color multiplies the texture).
            go.GetComponent<RawImage>().color = new Color(0.2f, 0.22f, 0.26f);
            var label = MakeLabel(go.transform, "Age", "n min ago", 15, TextMain, TextAnchor.LowerRight);
            label.rectTransform.Stretch(); label.rectTransform.offsetMin = new Vector2(4f, 3f); label.rectTransform.offsetMax = new Vector2(-4f, -3f);
            label.gameObject.AddComponent<Shadow>().effectColor = Color.black;
            return SavePrefab(go).GetComponent<RawImage>();
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

        static Image MakeImage(Transform parent, string name, Color color, Sprite sprite)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(parent, false);
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

        static Button MakeButton(Transform parent, string name, string caption, int size)
        {
            var img = MakeImage(parent, name, ButtonBg, Rounded);
            img.raycastTarget = true;
            var b = img.gameObject.AddComponent<Button>();
            b.targetGraphic = img;
            var colors = b.colors; colors.highlightedColor = new Color(1f, 1f, 1f, 0.25f); colors.pressedColor = new Color(0.55f, 0.75f, 1f, 0.6f); b.colors = colors;
            var label = MakeLabel(img.transform, "Label", caption, size, TextMain, TextAnchor.MiddleCenter);
            label.rectTransform.Stretch();
            return b;
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

        static Button MenuEntry(Transform parent, string name, string caption)
        {
            var b = MakeButton(parent, name, caption, 20);
            b.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
            var lrt = b.GetComponentInChildren<Text>().rectTransform; lrt.offsetMin = new Vector2(14f, 0f);
            b.gameObject.AddComponent<LayoutElement>().preferredHeight = 48f;
            return b;
        }

        static InputField MakeInputField(Transform parent, string name, string placeholder)
        {
            var img = MakeImage(parent, name, FieldBg, Rounded);
            img.raycastTarget = true;
            img.gameObject.AddComponent<LayoutElement>().preferredHeight = 48f;
            var f = img.gameObject.AddComponent<InputField>();
            var text = MakeLabel(img.transform, "Text", "", 20, TextMain, TextAnchor.MiddleLeft);
            text.supportRichText = false;
            text.rectTransform.Stretch(); text.rectTransform.offsetMin = new Vector2(10f, 4f); text.rectTransform.offsetMax = new Vector2(-10f, -4f);
            var ph = MakeLabel(img.transform, "Placeholder", placeholder, 20, new Color(1f, 1f, 1f, 0.3f), TextAnchor.MiddleLeft);
            ph.fontStyle = FontStyle.Italic;
            ph.rectTransform.Stretch(); ph.rectTransform.offsetMin = new Vector2(10f, 4f); ph.rectTransform.offsetMax = new Vector2(-10f, -4f);
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
