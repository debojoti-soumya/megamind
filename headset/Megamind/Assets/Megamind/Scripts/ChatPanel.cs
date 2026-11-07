// Stream 3: world-locked transcript panel with cited memory thumbnails.
// TODO(stream-3): build the prefab (Canvas in World Space, 0.6 m wide, 1 m ahead; VerticalLayoutGroup).
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace Megamind
{
    public class ChatPanel : MonoBehaviour
    {
        public MegamindConfig config;
        public Transform content;          // VerticalLayoutGroup
        public Text userBubblePrefab;
        public Text assistantBubblePrefab;
        public RawImage thumbnailPrefab;
        public GameObject listeningIndicator;
        public GameObject thinkingIndicator;
        public FrameUploader uploader;      // optional, for the status line
        public Text statusLine;

        public void SetListening(bool on) { if (listeningIndicator) listeningIndicator.SetActive(on); }
        public void SetThinking(bool on) { if (thinkingIndicator) thinkingIndicator.SetActive(on); }

        public void AddUserMessage(string text)
        {
            var t = Instantiate(userBubblePrefab, content);
            t.text = text;
        }

        public void AddAssistantMessage(string text, List<Citation> cited)
        {
            var t = Instantiate(assistantBubblePrefab, content);
            t.text = text;
            if (cited == null) return;
            foreach (var c in cited)
            {
                var img = Instantiate(thumbnailPrefab, content);
                StartCoroutine(LoadThumb(img, config.Url(c.thumb_url)));
            }
        }

        private IEnumerator LoadThumb(RawImage target, string url)
        {
            using (var req = UnityWebRequestTexture.GetTexture(url))
            {
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success && target != null)
                    target.texture = DownloadHandlerTexture.GetContent(req);
            }
        }

        private void Update()
        {
            if (statusLine && uploader)
                statusLine.text = $"capture: {uploader.Accepted}/{uploader.Sent} ({uploader.LastReason})";
        }
    }
}
