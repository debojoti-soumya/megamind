// Editor-only helpers so the project can be built with zero manual scene wiring:
//   Megamind > 1. Create Capture Scene      builds Assets/Megamind/Scenes/Megamind.unity (OVRCameraRig + passthrough +
//                                           PassthroughCameraAccess + FrameUploader + HUD) and adds it to Build Settings
//   Megamind > 2. Apply Android Player Settings
//   Megamind > 3. Build APK                 -> Builds/megamind.apk
//   Megamind > 4. Build and Run             -> installs + launches on the connected headset
//   Megamind > Push config to headset (adb) -> writes megamind.json (server URL etc.) without rebuilding
//   Megamind > Check Project                -> prints what is still missing
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Meta.XR;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.SceneManagement;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Megamind.EditorTools
{
    public static class MegamindSceneBuilder
    {
        public const string ScenePath = "Assets/Megamind/Scenes/Megamind.unity";
        public const string ConfigPath = "Assets/Megamind/MegamindConfig.asset";
        public const string CaptureSettingsPath = "Assets/Megamind/CaptureSettings.asset";
        public const string ApkPath = "Builds/megamind.apk";
        public const string PackageName = "com.megamind.quest";
        private const string RigPrefabPath = "Packages/com.meta.xr.sdk.core/Prefabs/OVRCameraRig.prefab";
        private const string ManifestPath = "Assets/Plugins/Android/AndroidManifest.xml";

        [MenuItem("Megamind/1. Create Capture Scene", false, 1)]
        public static void CreateScene()
        {
            if (File.Exists(ScenePath) &&
                !EditorUtility.DisplayDialog("Megamind", $"{ScenePath} exists. Overwrite it?", "Overwrite", "Cancel"))
                return;
            var config = GetOrCreateConfig();
            var capture = GetOrCreateCaptureSettings();
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var light = new GameObject("Directional Light").AddComponent<Light>();
            light.type = LightType.Directional;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            // OVRCameraRig + OVRManager (passthrough on) + OVRPassthroughLayer
            GameObject rigGo;
            var rigPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(RigPrefabPath);
            if (rigPrefab != null)
            {
                rigGo = (GameObject)PrefabUtility.InstantiatePrefab(rigPrefab);
            }
            else
            {
                Debug.LogWarning($"[Megamind] {RigPrefabPath} not found; creating OVRCameraRig from components.");
                rigGo = new GameObject("OVRCameraRig");
                rigGo.AddComponent<OVRCameraRig>();
            }
            var manager = rigGo.GetComponent<OVRManager>();
            if (manager == null) manager = rigGo.AddComponent<OVRManager>();
            manager.isInsightPassthroughEnabled = true;
            ClearStartupPermissionRequests(manager);
            manager.trackingOriginType = OVRManager.TrackingOrigin.FloorLevel;
            if (rigGo.GetComponent<OVRPassthroughLayer>() == null)
            {
                var layer = rigGo.AddComponent<OVRPassthroughLayer>();
                layer.overlayType = OVROverlay.OverlayType.Underlay;
            }
            foreach (var cam in rigGo.GetComponentsInChildren<Camera>(true))
            {
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0f, 0f, 0f, 0f); // transparent so passthrough shows through
            }
            var centerEye = rigGo.transform.Find("TrackingSpace/CenterEyeAnchor");
            if (centerEye == null) centerEye = rigGo.transform;

            // Passthrough Camera API
            var pcaGo = new GameObject("PassthroughCameraAccess");
            var pca = pcaGo.AddComponent<PassthroughCameraAccess>();
            pca.CameraPosition = PassthroughCameraAccess.CameraPositionType.Left;
            pca.RequestedResolution = capture.cameraResolution;
            pca.MaxFramerate = capture.cameraMaxFramerate;

            // Megamind components
            var mm = new GameObject("Megamind");
            mm.AddComponent<MegamindPermissions>();
            var uploader = mm.AddComponent<FrameUploader>();
            uploader.config = config;
            uploader.settings = capture;
            uploader.cameraAccess = pca;
            uploader.headPose = centerEye;
            var hud = mm.AddComponent<CaptureHud>();
            hud.uploader = uploader;
            hud.head = centerEye;

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log($"[Megamind] created {ScenePath} and set it as the only scene in Build Settings. " +
                      "Now run Meta > Tools > Project Setup Tool and fix all, then Megamind > 3/4.");
            Selection.activeObject = config;
        }

        /// <summary>
        /// OVRManager "Permission Requests On Startup" must stay empty: MegamindPermissions is the single permission flow
        /// (two requesters = two camera dialogs / races). The fields are internal, so go through SerializedObject.
        /// </summary>
        private static void ClearStartupPermissionRequests(OVRManager manager)
        {
            var so = new SerializedObject(manager);
            foreach (var name in new[]
                     {
                         "requestBodyTrackingPermissionOnStartup", "requestFaceTrackingPermissionOnStartup",
                         "requestEyeTrackingPermissionOnStartup", "requestScenePermissionOnStartup",
                         "requestRecordAudioPermissionOnStartup", "requestPassthroughCameraAccessPermissionOnStartup",
                     })
            {
                var prop = so.FindProperty(name);
                if (prop != null) prop.boolValue = false;
            }
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        [MenuItem("Megamind/2. Apply Android Player Settings", false, 2)]
        public static void ApplyAndroidSettings()
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, PackageName);
            PlayerSettings.productName = "Megamind";
            PlayerSettings.companyName = "Megamind";
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = (AndroidSdkVersions)32;
            PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)32;
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed; // plain-HTTP server on the tailnet
            PlayerSettings.colorSpace = ColorSpace.Linear;
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;
            AssetDatabase.SaveAssets();
            Debug.Log("[Megamind] Android player settings applied. Remaining Meta-specific settings: Meta > Tools > Project Setup Tool > Fix All.");
        }

        [MenuItem("Megamind/3. Build APK", false, 3)]
        public static void BuildApk() => Build(BuildOptions.None);

        [MenuItem("Megamind/4. Build and Run on headset", false, 4)]
        public static void BuildAndRun() => Build(BuildOptions.AutoRunPlayer);

        private static void Build(BuildOptions options)
        {
            if (!File.Exists(ScenePath))
            {
                Debug.LogError($"[Megamind] {ScenePath} missing: run Megamind > 1. Create Capture Scene first.");
                return;
            }
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android) ApplyAndroidSettings();
            Directory.CreateDirectory(Path.GetDirectoryName(ApkPath));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = ApkPath,
                target = BuildTarget.Android,
                options = options,
            });
            Debug.Log($"[Megamind] build {report.summary.result}: {report.summary.totalSize / (1024 * 1024)} MB, " +
                      $"{report.summary.totalErrors} errors -> {ApkPath}");
        }

        [MenuItem("Megamind/Push config to headset (adb)", false, 20)]
        public static void PushConfig()
        {
            var config = GetOrCreateConfig();
            var capture = GetOrCreateCaptureSettings();
            var json = JsonUtility.ToJson(new OverrideJson
            {
                serverBaseUrl = config.serverBaseUrl,
                deviceId = config.deviceId,
                captureFps = capture.captureFps,
            });
            var dir = $"/sdcard/Android/data/{PackageName}/files";
            var tmp = Path.Combine(Path.GetTempPath(), CaptureSettings.OverrideFileName);
            File.WriteAllText(tmp, json);
            var output = Adb($"shell mkdir -p {dir}") + Adb($"push \"{tmp}\" {dir}/{CaptureSettings.OverrideFileName}");
            Debug.Log($"[Megamind] pushed {json} -> {dir}/{CaptureSettings.OverrideFileName}\n{output}\nRestart the app to apply.");
        }

        [MenuItem("Megamind/Check Project", false, 21)]
        public static void CheckProject()
        {
            var problems = 0;
            void Check(bool ok, string msg)
            {
                if (ok) Debug.Log("[Megamind] OK   " + msg);
                else { Debug.LogWarning("[Megamind] FIX  " + msg); problems++; }
            }
            var manifest = File.Exists(ManifestPath) ? File.ReadAllText(ManifestPath) : "";
            Check(manifest.Contains(OVRPermissionsRequester.PassthroughCameraAccessPermission), $"{ManifestPath} has horizonos.permission.HEADSET_CAMERA");
            Check(manifest.Contains("android.permission.CAMERA\""), "manifest declares android.permission.CAMERA (digest #0; runtime flow only asks for HEADSET_CAMERA)");
            Check(manifest.Contains("android.permission.INTERNET"), "manifest has INTERNET");
            Check(manifest.Contains("usesCleartextTraffic=\"true\""), "manifest allows cleartext HTTP (needed for http:// server URLs)");
            Check(File.Exists(ScenePath), $"{ScenePath} exists (Megamind > 1)");
            Check(EditorBuildSettings.scenes.Any(s => s.enabled && s.path == ScenePath), "scene is in Build Settings");
            Check(EditorUserBuildSettings.activeBuildTarget == BuildTarget.Android, "active build target is Android (Megamind > 2)");
            Check(PlayerSettings.GetScriptingBackend(NamedBuildTarget.Android) == ScriptingImplementation.IL2CPP, "IL2CPP");
            Check(PlayerSettings.Android.targetArchitectures == AndroidArchitecture.ARM64, "ARM64 only");
            Check(PlayerSettings.insecureHttpOption == InsecureHttpOption.AlwaysAllowed, "Allow downloads over HTTP = Always allowed");
            var config = AssetDatabase.LoadAssetAtPath<MegamindConfig>(ConfigPath);
            Check(config != null, $"{ConfigPath} exists");
            Check(AssetDatabase.LoadAssetAtPath<CaptureSettings>(CaptureSettingsPath) != null, $"{CaptureSettingsPath} exists");
            if (config != null)
                Check(config.serverBaseUrl.StartsWith("http"), $"server URL set ({config.serverBaseUrl})");
            var projectConfig = OVRProjectConfig.CachedProjectConfig;
            Check(projectConfig != null && projectConfig.isPassthroughCameraAccessEnabled, "OVRProjectConfig: Passthrough Camera Access enabled");
            Check(projectConfig != null && projectConfig.insightPassthroughSupport != OVRProjectConfig.FeatureSupport.None, "OVRProjectConfig: Passthrough Support = Supported/Required");
            Debug.Log(problems == 0
                ? "[Megamind] project check passed. Also run Meta > Tools > Project Setup Tool and fix any remaining items."
                : $"[Megamind] {problems} item(s) to fix (see warnings above).");
        }

        [MenuItem("Megamind/Select Config", false, 22)]
        public static void SelectConfig() => Selection.activeObject = GetOrCreateConfig();

        public static MegamindConfig GetOrCreateConfig()
        {
            var config = AssetDatabase.LoadAssetAtPath<MegamindConfig>(ConfigPath);
            if (config != null) return config;
            config = ScriptableObject.CreateInstance<MegamindConfig>();
            Directory.CreateDirectory(Path.GetDirectoryName(ConfigPath));
            AssetDatabase.CreateAsset(config, ConfigPath);
            AssetDatabase.SaveAssets();
            Debug.Log($"[Megamind] created {ConfigPath}: set serverBaseUrl to your laptop's Tailscale IP.");
            return config;
        }

        public static CaptureSettings GetOrCreateCaptureSettings()
        {
            var cs = AssetDatabase.LoadAssetAtPath<CaptureSettings>(CaptureSettingsPath);
            if (cs != null) return cs;
            cs = ScriptableObject.CreateInstance<CaptureSettings>();
            Directory.CreateDirectory(Path.GetDirectoryName(CaptureSettingsPath));
            AssetDatabase.CreateAsset(cs, CaptureSettingsPath);
            AssetDatabase.SaveAssets();
            return cs;
        }

        [Serializable]
        private class OverrideJson
        {
            public string serverBaseUrl;
            public string deviceId;
            public float captureFps;
        }

        private static string Adb(string args)
        {
            var sdk = EditorPrefs.GetString("AndroidSdkRoot", "");
            var candidate = string.IsNullOrEmpty(sdk) ? "" : Path.Combine(sdk, "platform-tools", "adb");
            var adb = !string.IsNullOrEmpty(candidate) && (File.Exists(candidate) || File.Exists(candidate + ".exe")) ? candidate : "adb";
            try
            {
                var psi = new ProcessStartInfo(adb, args)
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var p = Process.Start(psi);
                var output = p.StandardOutput.ReadToEnd() + p.StandardError.ReadToEnd();
                p.WaitForExit(15000);
                return output + "\n";
            }
            catch (Exception e)
            {
                Debug.LogError($"[Megamind] adb failed ({adb}): {e.Message}");
                return null;
            }
        }
    }
}
