// Stream 1: sample the passthrough camera at ~1 fps, JPEG-encode, POST /v1/frames.
// Attach next to PassthroughCameraSamples.WebCamTextureManager (Meta sample).
// TODO(stream-1): verify on device; WebCamTexture may need a few frames before width > 16.
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using PassthroughCameraSamples;

namespace Megamind
{
    public class FrameUploader : MonoBehaviour
    {
        public MegamindConfig config;
        public WebCamTextureManager cameraManager;
        public Transform headPose; // OVRCameraRig CenterEyeAnchor

        public int Sent { get; private set; }
        public int Accepted { get; private set; }
        public string LastReason { get; private set; } = "-";

        private Texture2D _scratch;
        private int _pending;

        private IEnumerator Start()
        {
            while (cameraManager == null || cameraManager.WebCamTexture == null || cameraManager.WebCamTexture.width <= 16)
                yield return null;
            var wait = new WaitForSeconds(1f / Mathf.Max(0.1f, config.captureFps));
            while (true)
            {
                if (_pending < config.maxPendingUploads)
                    StartCoroutine(CaptureAndUpload());
                yield return wait;
            }
        }

        private IEnumerator CaptureAndUpload()
        {
            var src = cameraManager.WebCamTexture;
            int w = src.width, h = src.height;
            float scale = Mathf.Min(1f, (float)config.jpegMaxWidth / w);
            int tw = Mathf.RoundToInt(w * scale), th = Mathf.RoundToInt(h * scale);

            if (_scratch == null || _scratch.width != w || _scratch.height != h)
                _scratch = new Texture2D(w, h, TextureFormat.RGB24, false);
            _scratch.SetPixels32(src.GetPixels32());
            _scratch.Apply(false);

            byte[] jpeg;
            if (scale < 1f)
            {
                var rt = RenderTexture.GetTemporary(tw, th, 0, RenderTextureFormat.ARGB32);
                Graphics.Blit(_scratch, rt);
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                var small = new Texture2D(tw, th, TextureFormat.RGB24, false);
                small.ReadPixels(new Rect(0, 0, tw, th), 0, 0);
                small.Apply(false);
                RenderTexture.active = prev;
                RenderTexture.ReleaseTemporary(rt);
                jpeg = small.EncodeToJPG(config.jpegQuality);
                Destroy(small);
            }
            else
            {
                jpeg = _scratch.EncodeToJPG(config.jpegQuality);
            }

            var meta = new FrameMeta
            {
                device_id = config.deviceId,
                ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                pose = headPose != null
                    ? new Pose { x = headPose.position.x, y = headPose.position.y, z = headPose.position.z, yaw = headPose.eulerAngles.y }
                    : null,
            };

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("image", jpeg, "frame.jpg", "image/jpeg"),
                new MultipartFormDataSection("meta", JsonUtility.ToJson(meta)),
            };

            _pending++;
            using (var req = UnityWebRequest.Post(config.Url("/v1/frames"), form))
            {
                req.timeout = 10;
                yield return req.SendWebRequest();
                _pending--;
                Sent++;
                if (req.result == UnityWebRequest.Result.Success)
                {
                    var resp = JsonUtility.FromJson<FrameAccepted>(req.downloadHandler.text);
                    if (resp.accepted) Accepted++;
                    LastReason = resp.reason ?? "ok";
                }
                else
                {
                    LastReason = "net:" + req.error;
                    Debug.LogWarning($"[Megamind] frame upload failed: {req.error}");
                }
            }
        }
    }
}
