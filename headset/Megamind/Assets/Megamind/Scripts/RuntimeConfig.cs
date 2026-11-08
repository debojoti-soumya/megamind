// Runtime overrides for serverBaseUrl / deviceId / apiToken / ambientAudio, so a tester can repoint the headset over adb
// without a rebuild. Sources, highest priority first:
//   1. launch-intent extras (this launch), saved to PlayerPrefs so later launches from the Library keep them:
//        adb shell am force-stop com.megamind.quest
//        adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity \
//          --es serverBaseUrl http://100.101.102.103:8000 --es deviceId quest3-1 --es apiToken <token> --es ambientAudio off
//      An empty value (--es apiToken "''") removes that saved key; --ez resetConfig true removes all of them first.
//   2. PlayerPrefs saved earlier: by a previous intent launch ("saved") or the settings panel ("panel").
//   3. megamind.json in Application.persistentDataPath, if the app can read it. On Android 11+ a file written there by
//      `adb shell` / `adb push` is owned by the shell user and the app gets "Access ... is denied", so prefer 1.
//   4. MegamindConfig.asset / CaptureSettings.asset (baked into the APK).
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Megamind
{
    public static class RuntimeConfig
    {
        public const string ServerBaseUrlKey = "serverBaseUrl";
        public const string DeviceIdKey = "deviceId";
        public const string ApiTokenKey = "apiToken";
        public const string AmbientAudioKey = "ambientAudio";
        public const string ResetKey = "resetConfig";
        public static readonly string[] Keys = { ServerBaseUrlKey, DeviceIdKey, ApiTokenKey, AmbientAudioKey };

        public const string SourceIntent = "intent";
        public const string SourceSaved = "saved";
        public const string SourcePanel = "panel";
        public const string SourceAsset = "asset";

        private static bool s_intentRead;
        private static readonly HashSet<string> s_fromIntent = new HashSet<string>();

        /// <summary>PlayerPrefs key for a config key; "megamind.serverBaseUrl"/"megamind.deviceId" are the keys the settings panel always used.</summary>
        public static string PrefKey(string key) => "megamind." + key;
        private static string SourcePrefKey(string key) => "megamind.source." + key;

        /// <summary>Reads the launch intent's extras once per process and saves them to PlayerPrefs. Safe to call repeatedly.</summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        public static void ApplyLaunchIntent()
        {
            if (s_intentRead) return;
            s_intentRead = true;
            var extras = ReadIntentExtras();
            if (extras != null && extras.Count > 0) ApplyExtras(extras);
        }

        /// <summary>Saves <paramref name="extras"/> (config key -> value) as intent-sourced PlayerPrefs. Returns the keys applied.</summary>
        public static List<string> ApplyExtras(IDictionary<string, string> extras)
        {
            var applied = new List<string>();
            if (extras.TryGetValue(ResetKey, out var reset) && IsTrue(reset))
            {
                ClearSaved();
                Debug.Log("[Megamind] config: resetConfig -> cleared saved overrides");
            }
            foreach (var key in Keys)
            {
                if (!extras.TryGetValue(key, out var raw)) continue;
                var value = (raw ?? "").Trim();
                if (value.Length == 0)
                {
                    PlayerPrefs.DeleteKey(PrefKey(key));
                    PlayerPrefs.DeleteKey(SourcePrefKey(key));
                    s_fromIntent.Remove(key);
                    Debug.Log($"[Megamind] config: intent extra {key}=\"\" -> removed saved value");
                }
                else
                {
                    PlayerPrefs.SetString(PrefKey(key), value);
                    PlayerPrefs.SetString(SourcePrefKey(key), SourceIntent);
                    s_fromIntent.Add(key);
                    Debug.Log($"[Megamind] config: intent extra {key}={Display(key, value)} (saved)");
                }
                applied.Add(key);
            }
            PlayerPrefs.Save();
            return applied;
        }

        /// <summary>Value saved in PlayerPrefs for <paramref name="key"/>, with its source: "intent" (this launch),
        /// "saved" (an earlier intent launch) or "panel" (settings panel).</summary>
        public static bool TryGetSaved(string key, out string value, out string source)
        {
            ApplyLaunchIntent();
            value = null;
            source = null;
            if (!PlayerPrefs.HasKey(PrefKey(key))) return false;
            value = PlayerPrefs.GetString(PrefKey(key)).Trim();
            if (value.Length == 0) return false;
            if (s_fromIntent.Contains(key)) source = SourceIntent;
            else source = PlayerPrefs.GetString(SourcePrefKey(key), SourcePanel) == SourceIntent ? SourceSaved : SourcePanel;
            return true;
        }

        /// <summary>Precedence for one key: saved PlayerPrefs (intent/panel) > megamind.json > asset.</summary>
        public static string Choose(string key, string fileValue, string assetValue, out string source)
        {
            if (TryGetSaved(key, out var saved, out source)) return saved;
            if (!string.IsNullOrEmpty(fileValue))
            {
                source = CaptureSettings.OverrideFileName;
                return fileValue;
            }
            source = SourceAsset;
            return assetValue;
        }

        /// <summary>Called by MegamindConfig.SaveOverrides (settings panel).</summary>
        public static void SaveFromPanel(string key, string value)
        {
            PlayerPrefs.SetString(PrefKey(key), value);
            PlayerPrefs.SetString(SourcePrefKey(key), SourcePanel);
            s_fromIntent.Remove(key);
        }

        public static void ClearSaved()
        {
            foreach (var key in Keys)
            {
                PlayerPrefs.DeleteKey(PrefKey(key));
                PlayerPrefs.DeleteKey(SourcePrefKey(key));
            }
            s_fromIntent.Clear();
            PlayerPrefs.Save();
        }

        public static bool IsTrue(string v)
        {
            v = (v ?? "").Trim().ToLowerInvariant();
            return v == "true" || v == "1" || v == "on" || v == "yes";
        }

        /// <summary>"on"/"true" -> true, "off"/"false" -> false, anything else -> null (keep the lower-priority value).</summary>
        public static bool? ParseOnOff(string v)
        {
            v = (v ?? "").Trim().ToLowerInvariant();
            if (v == "on" || v == "true" || v == "1") return true;
            if (v == "off" || v == "false" || v == "0") return false;
            return null;
        }

        /// <summary>Log-safe value: the API token is never printed.</summary>
        public static string Display(string key, string value) =>
            key == ApiTokenKey ? (string.IsNullOrEmpty(value) ? "(none)" : $"(set, {value.Length} chars)") : value;

        private static Dictionary<string, string> ReadIntentExtras()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                var activity = UnityEngine.Android.AndroidApplication.currentActivity; // shared by Unity: don't dispose
                using (var intent = activity?.Call<AndroidJavaObject>("getIntent"))
                using (var bundle = intent?.Call<AndroidJavaObject>("getExtras"))
                {
                    if (bundle == null) return null;
                    var extras = new Dictionary<string, string>();
                    foreach (var key in new[] { ServerBaseUrlKey, DeviceIdKey, ApiTokenKey, AmbientAudioKey, ResetKey })
                    {
                        if (!bundle.Call<bool>("containsKey", key)) continue;
                        // Bundle.get + toString so --ez/--ei work too, not only --es.
                        using (var obj = bundle.Call<AndroidJavaObject>("get", key))
                            extras[key] = obj == null ? "" : obj.Call<string>("toString");
                    }
                    return extras;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Megamind] config: could not read launch intent extras: {e.Message}");
                return null;
            }
#else
            return null;
#endif
        }
    }
}
