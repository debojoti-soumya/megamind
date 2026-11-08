// Stream 3: runtime settings (server base URL / device id) so the headset can be re-pointed at a
// new Tailscale IP or ngrok URL without a rebuild. Values persist in PlayerPrefs via MegamindConfig.
// Also hosts the privacy "purge memory" button and a "new session" button.
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace Megamind
{
    public class SettingsPanel : MonoBehaviour
    {
        public MegamindConfig config;
        public ChatClient client;
        public ChatPanel chatPanel;
        public GameObject root;             // the settings sub-panel to show/hide
        public InputField serverUrlField;   // TouchScreenKeyboard opens on focus on Quest
        public InputField deviceIdField;
        public Text feedback;

        private void Awake()
        {
            if (config != null) config.LoadOverrides();
            Hide();
        }

        public void Toggle() { if (root && root.activeSelf) Hide(); else Show(); }

        public void Show()
        {
            if (serverUrlField) serverUrlField.text = config.serverBaseUrl;
            if (deviceIdField) deviceIdField.text = config.deviceId;
            if (feedback) feedback.text = "";
            if (root) root.SetActive(true);
        }

        public void Hide() { if (root) root.SetActive(false); }

        public void Save()
        {
            string url = serverUrlField ? serverUrlField.text : config.serverBaseUrl;
            string dev = deviceIdField ? deviceIdField.text : config.deviceId;
            if (!url.StartsWith("http://") && !url.StartsWith("https://")) url = "http://" + url;
            config.SaveOverrides(url, dev);
            StartCoroutine(CheckHealth());
        }

        public void ResetToDefaults()
        {
            config.ClearOverrides();
            if (feedback) feedback.text = "Overrides cleared; restart the app to reload asset defaults.";
        }

        public void NewSession()
        {
            if (client) client.NewSession();
            if (chatPanel) chatPanel.Clear();
            if (feedback) feedback.text = "New session started.";
        }

        public void Purge() => StartCoroutine(PurgeCo());

        private IEnumerator CheckHealth()
        {
            using (var req = UnityWebRequest.Get(config.Url("/v1/health")))
            {
                config.Authorize(req);
                req.timeout = 5;
                yield return req.SendWebRequest();
                if (feedback)
                    feedback.text = req.result == UnityWebRequest.Result.Success
                        ? "Saved. Server reachable."
                        : "Saved, but server not reachable: " + req.error;
            }
        }

        private IEnumerator PurgeCo()
        {
            using (var req = UnityWebRequest.Delete(config.Url("/v1/memories?device_id=" + UnityWebRequest.EscapeURL(config.deviceId))))
            {
                req.downloadHandler = new DownloadHandlerBuffer();
                config.Authorize(req);
                req.timeout = 10;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    if (feedback) feedback.text = "Purge failed: " + req.error;
                    yield break;
                }
                int n = 0;
                try { n = JsonUtility.FromJson<PurgeResponse>(req.downloadHandler.text).deleted; } catch { }
                if (feedback) feedback.text = $"Purged {n} memory tokens for {config.deviceId}.";
                if (chatPanel) chatPanel.AddAssistantMessage($"Purged {n} memories for this device.", null);
            }
        }
    }
}
