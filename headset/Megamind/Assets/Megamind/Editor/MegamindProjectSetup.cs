// Editor-only. Applies the Android/Quest 3 player settings the design doc asks for, so a fresh clone
// does not depend on a committed ProjectSettings.asset (which Unity regenerates per version anyway).
// Run "Meta > Tools > Project Setup Tool" afterwards and fix whatever it still flags.
using UnityEditor;
using UnityEditor.Build;
using UnityEngine;
using UnityEngine.Rendering;

namespace Megamind.Editor
{
    public static class MegamindProjectSetup
    {
        public const string PackageName = "com.megamind.quest";

        [MenuItem("Megamind/1. Apply Player Settings (Android, Quest 3)")]
        public static void Apply()
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);

            PlayerSettings.companyName = "Megamind";
            PlayerSettings.productName = "Megamind";
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, PackageName);
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.SetApiCompatibilityLevel(NamedBuildTarget.Android, ApiCompatibilityLevel.NET_Standard);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevelAuto;
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.Android.forceSDCardPermission = false;
            PlayerSettings.Android.androidIsGame = false;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
            // plain-http to the laptop's Tailscale IP (design §Stream 3 networking)
            PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;

            AssetDatabase.SaveAssets();
            Debug.Log("[Megamind] Player settings applied. Now run Meta > Tools > Project Setup Tool and fix all.");
        }
    }
}
