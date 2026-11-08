// Editor-only. Stream 3: adds the voice-first chat panel to the capture scene made by MegamindSceneBuilder
// (Megamind > 1), creates the bubble/thumbnail prefabs, and wires every public reference on
// ChatPanel / ChatClient / PanelAnchor / SettingsPanel. Run Megamind > 1, then Megamind > 1b, then build.
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
        const float PanelW = 600f, PanelH = 720f, HeaderH = 56f, FooterH = 56f;
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

            MakeImage(go.transform, "Background", PanelBg, Rounded).rectTransform.Stretch();

            // Header: status line, server dot, A- / A+ (UI scale), Pin, Settings
            var header = MakeImage(go.transform, "Header", HeaderBg, Rounded);
            header.AnchorTop(HeaderH);
            var dot = MakeImage(header.transform, "ServerDot", new Color(1f, 0.36f, 0.42f), Knob);
            dot.AnchorLeftCentre(14f, 14f, 14f);
            var statusLine = MakeLabel(header.transform, "StatusLine", "connecting…", 18, TextDim, TextAnchor.MiddleLeft);
            statusLine.rectTransform.Fill(new Vector2(36f, 0f), new Vector2(-262f, 0f));
            var gear = MakeButton(header.transform, "SettingsButton", "Settings", 17);
            gear.GetComponent<RectTransform>().AnchorRightCentre(10f, 86f, 40f);
            var pin = MakeButton(header.transform, "PinButton", "Pin", 17);
            pin.GetComponent<RectTransform>().AnchorRightCentre(102f, 64f, 40f);
            var scaleUp = MakeButton(header.transform, "ScaleUpButton", "A+", 19);
            scaleUp.GetComponent<RectTransform>().AnchorRightCentre(172f, 40f, 40f);
            var scaleDown = MakeButton(header.transform, "ScaleDownButton", "A-", 19);
            scaleDown.GetComponent<RectTransform>().AnchorRightCentre(216f, 40f, 40f);

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

            var emptyHint = MakeLabel(go.transform, "EmptyHint", "Hold the trigger (or pinch) and ask about something you saw.", 24, TextDim, TextAnchor.MiddleCenter);
            emptyHint.rectTransform.Fill(new Vector2(40f, FooterH), new Vector2(-40f, -HeaderH));

            // Footer indicators
            var footer = new GameObject("Indicators", typeof(RectTransform));
            footer.transform.SetParent(go.transform, false);
            footer.GetComponent<RectTransform>().AnchorBottom(FooterH);
            var listening = MakeImage(footer.transform, "ListeningIndicator", new Color(1f, 0.3f, 0.3f), Knob);
            listening.AnchorLeftCentre(18f, 16f, 16f);
            var listeningLabel = MakeLabel(listening.transform, "Label", "listening…", 20, TextMain, TextAnchor.MiddleLeft);
            listeningLabel.rectTransform.anchorMin = new Vector2(1f, 0.5f); listeningLabel.rectTransform.anchorMax = new Vector2(1f, 0.5f);
            listeningLabel.rectTransform.pivot = new Vector2(0f, 0.5f); listeningLabel.rectTransform.anchoredPosition = new Vector2(8f, 0f);
            listeningLabel.rectTransform.sizeDelta = new Vector2(240f, 28f);
            var thinking = MakeLabel(footer.transform, "ThinkingIndicator", "thinking…", 20, TextDim, TextAnchor.MiddleCenter);
            thinking.rectTransform.Stretch();

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

            // Components
            var speaker = go.AddComponent<AudioSource>();
            speaker.playOnAwake = false; speaker.spatialBlend = 0f;

            var anchor = go.AddComponent<PanelAnchor>();
            anchor.config = config; anchor.head = head;
            anchor.pinLabel = pin.GetComponentInChildren<Text>();

            var chatPanel = go.AddComponent<ChatPanel>();
            chatPanel.config = config;
            chatPanel.content = crt; chatPanel.scroll = scroll;
            chatPanel.userBubblePrefab = userBubble; chatPanel.assistantBubblePrefab = assistantBubble;
            chatPanel.thumbnailRowPrefab = thumbRow; chatPanel.thumbnailPrefab = thumb;
            chatPanel.listeningIndicator = listening.gameObject; chatPanel.thinkingIndicator = thinking.gameObject;
            chatPanel.emptyHint = emptyHint.gameObject;
            chatPanel.statusLine = statusLine; chatPanel.serverDot = dot;
            chatPanel.uploader = UnityEngine.Object.FindFirstObjectByType<FrameUploader>();

            var client = go.AddComponent<ChatClient>();
            client.config = config; client.panel = chatPanel; client.speaker = speaker;
            chatPanel.client = client;
            client.leftHand = MegamindSceneBuilder.FindHand(OVRPlugin.Hand.HandLeft);
            client.rightHand = MegamindSceneBuilder.FindHand(OVRPlugin.Hand.HandRight);
            var hud = UnityEngine.Object.FindFirstObjectByType<CaptureHud>();
            if (hud != null)
            {
                if (hud.leftHand == null) hud.leftHand = client.leftHand;
                if (hud.rightHand == null) hud.rightHand = client.rightHand;
                EditorUtility.SetDirty(hud);
            }
            if (client.leftHand == null || client.rightHand == null)
                Debug.LogWarning("[Megamind] no OVRHand for " + (client.leftHand == null ? "left " : "") + (client.rightHand == null ? "right " : "") +
                                 "in the scene: pinch push-to-talk is off. Re-run Megamind > 1. Create Capture Scene (Overwrite), then 1b.");

            var settings = go.AddComponent<SettingsPanel>();
            settings.config = config; settings.client = client; settings.chatPanel = chatPanel;
            settings.root = settingsRoot.gameObject; settings.serverUrlField = serverField; settings.deviceIdField = deviceField;
            settings.feedback = feedback;

            UnityEventTools.AddPersistentListener(gear.onClick, new UnityAction(settings.Toggle));
            UnityEventTools.AddPersistentListener(pin.onClick, new UnityAction(anchor.TogglePinned));
            UnityEventTools.AddPersistentListener(scaleUp.onClick, new UnityAction(anchor.ScaleUp));
            UnityEventTools.AddPersistentListener(scaleDown.onClick, new UnityAction(anchor.ScaleDown));
            UnityEventTools.AddPersistentListener(save.onClick, new UnityAction(settings.Save));
            UnityEventTools.AddPersistentListener(newSession.onClick, new UnityAction(settings.NewSession));
            UnityEventTools.AddPersistentListener(purge.onClick, new UnityAction(settings.Purge));
            UnityEventTools.AddPersistentListener(reset.onClick, new UnityAction(settings.ResetToDefaults));
            UnityEventTools.AddPersistentListener(close.onClick, new UnityAction(settings.Hide));

            settingsRoot.gameObject.SetActive(false);
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
