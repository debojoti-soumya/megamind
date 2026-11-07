// Editor-only. Builds the APK from the scenes enabled in Build Settings. Also usable headless:
//   Unity -batchmode -quit -projectPath headset/Megamind -executeMethod Megamind.Editor.MegamindBuild.Android
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Megamind.Editor
{
    public static class MegamindBuild
    {
        public const string ApkPath = "Build/megamind.apk";

        [MenuItem("Megamind/3. Build Android APK")]
        public static void Android()
        {
            var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            if (scenes.Length == 0)
                throw new BuildFailedException("No scenes enabled in Build Settings. Run 'Megamind > 2. Build Chat Panel' in the sample scene first.");

            Directory.CreateDirectory(Path.GetDirectoryName(ApkPath));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = ApkPath,
                target = BuildTarget.Android,
                options = BuildOptions.None,
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new BuildFailedException("Megamind Android build failed: " + report.summary.result);
            Debug.Log($"[Megamind] Built {ApkPath} ({report.summary.totalSize / (1024 * 1024)} MB). " +
                      $"Install: adb install -r {ApkPath} && adb shell monkey -p {MegamindProjectSetup.PackageName} 1");
        }
    }
}
