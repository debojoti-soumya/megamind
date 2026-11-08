// Stream 1: sample the passthrough camera at ~1 fps, JPEG-encode a <= jpegMaxWidth (1280 px: full frame) copy and
// POST multipart (image + meta JSON) to /v1/frames. See docs/api.md.
//
// Camera source: MRUK's PassthroughCameraAccess (com.meta.xr.mrutilitykit >= 81, Horizon OS v74+),
// the component used by Meta's Unity-PassthroughCameraApiSamples. Its texture is a RenderTexture,
// so we Blit -> ReadPixels -> EncodeToJPG (the same path QuestCameraKit uses on device).
//
// Pipeline per tick:  backlog gate (pending uploads)  ->  backoff gate (server unreachable)  ->  motion gate (head deg/s)
//                     ->  readback  ->  encode (next frame)  ->  upload.
// Paused (A/X) or app backgrounded (OnApplicationPause): no capture and the camera stream is stopped (CaptureSettings.stopCameraWhenPaused).
// Focus loss alone (system menu, boundary warning, Meta-button dialog: OnApplicationFocus(false) / OVRManager.InputFocusLost)
// keeps capturing unless CaptureSettings.pauseOnFocusLoss; the app is focus aware and keeps running behind the overlay.
// Everything else (dedupe, blur, scene change, budget) is the server's job (server/megamind/ingest/filter.py).
using System;
using System.Collections;
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;
using UnityEngine.Networking;

namespace Megamind
{
    public class FrameUploader : MonoBehaviour
    {
        public enum State { Idle, WaitingForPermission, WaitingForCamera, Capturing, Paused, Unsupported, NoCameraComponent }

        [Tooltip("Shared config (server URL, device id). Owned by stream 3; only read here.")]
        public MegamindConfig config;
        [Tooltip("Capture tuning (fps, resolution, JPEG, backpressure). Created by Megamind > 1. Create Capture Scene.")]
        public CaptureSettings settings;
        [Tooltip("MRUK PassthroughCameraAccess (left camera). Found automatically when empty.")]
        public PassthroughCameraAccess cameraAccess;
        [Tooltip("OVRCameraRig/TrackingSpace/CenterEyeAnchor. Found automatically when empty.")]
        public Transform headPose;

        // --- status (read by CaptureHud / ChatPanel) ---
        public State CurrentState { get; private set; } = State.Idle;
        public bool Capturing { get; private set; }
        public int Sent { get; private set; }          // responses received from the server
        public int Accepted { get; private set; }      // accepted=true
        public int Failed { get; private set; }        // network / HTTP errors
        public int SkippedMotion { get; private set; } // dropped by the client-side motion gate
        public int SkippedBacklog { get; private set; }
        public int SkippedBackoff { get; private set; } // dropped while backing off after upload failures
        public int Pending => _pending;
        public string LastReason { get; private set; } = "-";
        public string LastError { get; private set; } = "";
        public float LastLatencyMs { get; private set; }
        public float LastEncodeMs { get; private set; }
        public int LastJpegBytes { get; private set; }
        public float HeadDegPerSec { get; private set; }
        public double LastUploadTs { get; private set; }
        public bool ServerReachable => Sent > 0 && _consecutiveFailures == 0;
        public Vector2Int CameraResolution => cameraAccess != null ? cameraAccess.CurrentResolution : Vector2Int.zero;
        public int poseHistory = 1800; // ~30 min at 1 fps, matching the server LOCATE_MAX_AGE_S; camera poses for ObjectMarkers
        public string ServerBaseUrl => settings.ServerBaseUrl(config);
        public string DeviceId => settings.DeviceId(config);
        /// <summary>Why capture is paused: "A/X", "background", "focus lost"; null while capture is allowed.</summary>
        public string PauseReason => !Capturing ? "A/X" : _appPaused ? "background" : PausedByFocus ? "focus lost" : null;
        public bool AppFocused => !_appUnfocused;
        public int BackgroundCount { get; private set; } // OnApplicationPause(true) events since launch
        public int FocusLossCount { get; private set; }  // OnApplicationFocus(false) events since launch

        private Texture2D _scratch;
        private int _pending;
        private int _consecutiveFailures;
        private Quaternion _lastHeadRot;
        private bool _hasLastHeadRot;
        private float _nextCaptureTime;
        private float _nextCameraRetry;
        private bool _loggedPlaying;
        private bool _appActive = true;  // false while backgrounded, or unfocused with pauseOnFocusLoss
        private bool _appPaused;         // OnApplicationPause(true): really backgrounded (headset off, app switched away)
        private bool _appUnfocused;      // OnApplicationFocus(false): system overlay / boundary warning / Meta-button dialog
        private bool PausedByFocus => _appUnfocused && settings != null && settings.pauseOnFocusLoss;
        private float _backoffUntil;

        private void Awake()
        {
            if (config == null)
            {
                config = ScriptableObject.CreateInstance<MegamindConfig>();
                Debug.LogWarning("[Megamind] FrameUploader: no MegamindConfig assigned; using defaults.");
            }
            settings = CaptureSettings.Resolve(settings);
            Capturing = settings.captureOnStart;
            Debug.Log($"[Megamind] capture: server={ServerBaseUrl} device={DeviceId} fps={settings.captureFps} url-source={settings.overrideSource}");
            Debug.Log($"[Megamind] config: {settings.DescribeConfig(config)}");
        }

        private IEnumerator Start()
        {
            if (cameraAccess == null) cameraAccess = FindAnyObjectByType<PassthroughCameraAccess>(FindObjectsInactive.Include);
            if (headPose == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) headPose = rig.centerEyeAnchor != null ? rig.centerEyeAnchor : rig.transform;
            }
            if (cameraAccess == null)
            {
                CurrentState = State.NoCameraComponent;
                Debug.LogError("[Megamind] FrameUploader: no PassthroughCameraAccess in the scene (run Megamind > 1. Create Capture Scene).");
                yield break;
            }
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!PassthroughCameraAccess.IsSupported)
            {
                CurrentState = State.Unsupported;
                Debug.LogError("[Megamind] Passthrough Camera API unsupported: needs Quest 3 / 3S on Horizon OS v74+.");
                yield break;
            }
#endif
            try
            {
                var supported = PassthroughCameraAccess.GetSupportedResolutions(cameraAccess.CameraPosition);
                Debug.Log($"[Megamind] camera: os={SystemInfo.operatingSystem} device={SystemInfo.deviceModel} " +
                          $"position={cameraAccess.CameraPosition} requested={settings.cameraResolution.x}x{settings.cameraResolution.y} " +
                          $"supported=[{string.Join(", ", System.Array.ConvertAll(supported, r => r.x + "x" + r.y))}]");
            }
            catch (Exception e) { Debug.LogWarning("[Megamind] GetSupportedResolutions failed: " + e.Message); }

            // RequestedResolution / MaxFramerate can only be changed while the component is disabled.
            if (!cameraAccess.IsPlaying &&
                (cameraAccess.RequestedResolution != settings.cameraResolution || cameraAccess.MaxFramerate != settings.cameraMaxFramerate))
            {
                bool wasEnabled = cameraAccess.enabled;
                cameraAccess.enabled = false;
                cameraAccess.RequestedResolution = settings.cameraResolution;
                cameraAccess.MaxFramerate = settings.cameraMaxFramerate;
                cameraAccess.enabled = wasEnabled;
            }

            var poll = new WaitForSeconds(0.25f);
            while (true)
            {
                if (!Capturing || !_appActive)
                {
                    CurrentState = State.Paused;
                    StopCameraWhilePaused();
                    yield return poll;
                    continue;
                }
                if (!MegamindPermissions.CameraGranted)
                {
                    CurrentState = State.WaitingForPermission;
                    yield return poll;
                    continue;
                }
                if (!cameraAccess.IsPlaying)
                {
                    _loggedPlaying = false;
                    CurrentState = State.WaitingForCamera;
                    // PassthroughCameraAccess disables itself when Play() fails (e.g. camera busy); retry every 3 s.
                    if (!cameraAccess.enabled && Time.unscaledTime >= _nextCameraRetry)
                    {
                        _nextCameraRetry = Time.unscaledTime + 3f;
                        cameraAccess.enabled = true;
                    }
                    yield return poll;
                    continue;
                }

                if (!_loggedPlaying)
                {
                    _loggedPlaying = true;
                    Debug.Log($"[Megamind] camera playing: {cameraAccess.CurrentResolution.x}x{cameraAccess.CurrentResolution.y} maxFps={cameraAccess.MaxFramerate}");
                }
                CurrentState = State.Capturing;
                if (Time.unscaledTime >= _nextCaptureTime)
                {
                    _nextCaptureTime = Time.unscaledTime + 1f / Mathf.Max(0.1f, settings.captureFps);
                    TryCapture();
                }
                yield return null;
            }
        }

        private void Update()
        {
            if (headPose == null) return;
            var rot = headPose.rotation;
            if (_hasLastHeadRot && Time.unscaledDeltaTime > 1e-4f)
            {
                var inst = Quaternion.Angle(_lastHeadRot, rot) / Time.unscaledDeltaTime;
                HeadDegPerSec = Mathf.Lerp(HeadDegPerSec, inst, 0.3f); // light smoothing against tracking jitter
            }
            _lastHeadRot = rot;
            _hasLastHeadRot = true;
        }

        public void SetCapturing(bool on)
        {
            Capturing = on;
            if (on) _nextCaptureTime = 0f;
            else StopCameraWhilePaused();
        }

        private void OnEnable()
        {
            OVRManager.InputFocusLost += OnInputFocusLost;
            OVRManager.InputFocusAcquired += OnInputFocusAcquired;
        }

        private void OnDisable()
        {
            OVRManager.InputFocusLost -= OnInputFocusLost;
            OVRManager.InputFocusAcquired -= OnInputFocusAcquired;
        }

        // Diagnostics only: input focus moves to system UI (menu, boundary dialog) while the app keeps running.
        private void OnInputFocusLost() => Debug.Log("[Megamind] OVR input focus lost (system UI overlay): capture keeps running");
        private void OnInputFocusAcquired() => Debug.Log("[Megamind] OVR input focus acquired");

        private void OnApplicationPause(bool paused)
        {
            if (paused == _appPaused) return;
            _appPaused = paused;
            if (paused) BackgroundCount++;
            Debug.Log(paused
                ? "[Megamind] app backgrounded (OnApplicationPause): pausing capture"
                : "[Megamind] app foregrounded (OnApplicationPause): resuming capture");
            UpdateAppActive();
        }

        private void OnApplicationFocus(bool focused)
        {
            if (Application.isEditor) return; // in the Editor, clicking another window must not count as focus loss
            if (focused != _appUnfocused) return;
            _appUnfocused = !focused;
            if (!focused) FocusLossCount++;
            bool pause = settings != null && settings.pauseOnFocusLoss;
            Debug.Log(focused
                ? "[Megamind] app focus regained (OnApplicationFocus)" + (pause ? ": resuming capture" : "")
                : "[Megamind] app focus lost (OnApplicationFocus: system menu / boundary / dialog): " +
                  (pause ? "pausing capture (pauseOnFocusLoss)" : "capture keeps running"));
            UpdateAppActive();
        }

        private void UpdateAppActive()
        {
            bool active = !_appPaused && !PausedByFocus;
            if (active == _appActive) return;
            _appActive = active;
            if (active) _nextCaptureTime = 0f;
            else StopCameraWhilePaused();
        }

        /// <summary>The camera keeps streaming at cameraMaxFramerate even when we don't sample it; stop it while paused.
        /// The capture loop's WaitingForCamera branch re-enables it on resume.</summary>
        private void StopCameraWhilePaused()
        {
            if (settings == null || !settings.stopCameraWhenPaused || cameraAccess == null || !cameraAccess.enabled) return;
            cameraAccess.enabled = false;
            _nextCameraRetry = 0f;
            _loggedPlaying = false;
        }

        public void ToggleCapturing() => SetCapturing(!Capturing);

        /// <summary>Capture one frame now, bypassing the schedule (still subject to the backlog gate).</summary>
        public void CaptureNow() => TryCapture(force: true);

        private void TryCapture(bool force = false)
        {
            if (cameraAccess == null || !cameraAccess.IsPlaying) return;
            if (_pending >= settings.maxPendingUploads)
            {
                SkippedBacklog++;
                LastReason = "client:backlog";
                return;
            }
            if (!force && Time.unscaledTime < _backoffUntil)
            {
                SkippedBackoff++;
                LastReason = "client:backoff";
                return;
            }
            if (!force && settings.maxHeadDegPerSec > 0f && HeadDegPerSec > settings.maxHeadDegPerSec)
            {
                SkippedMotion++;
                LastReason = "client:motion";
                return;
            }
            StartCoroutine(CaptureAndUpload());
        }

        private IEnumerator CaptureAndUpload()
        {
            var meta = BuildMeta();
            var t0 = Time.realtimeSinceStartup;
            if (!ReadbackCurrentFrame())
            {
                LastError = "camera texture not ready";
                yield break;
            }
            _pending++; // count the frame from readback on so the backlog gate sees it
            yield return null; // encode on the next frame: ReadPixels + EncodeToJPG together would hitch the render loop
            byte[] jpeg = _scratch.EncodeToJPG(settings.jpegQuality);
            LastEncodeMs = (Time.realtimeSinceStartup - t0) * 1000f;
            if (jpeg == null || jpeg.Length == 0)
            {
                _pending--;
                LastError = "encode failed";
                yield break;
            }
            LastJpegBytes = jpeg.Length;
            yield return Upload(jpeg, meta);
        }

        /// <summary>ts = camera image timestamp (falls back to now), pose = RGB camera pose at that time (falls back to CenterEyeAnchor).</summary>
        private FrameMeta BuildMeta()
        {
            double now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
            double ts = now;
            try
            {
                var stamp = cameraAccess.Timestamp;
                if (stamp.Kind == DateTimeKind.Unspecified) stamp = DateTime.SpecifyKind(stamp, DateTimeKind.Local);
                double camTs = new DateTimeOffset(stamp.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0;
                if (Math.Abs(camTs - now) < 5.0) ts = camTs; // ignore if not wall-clock based
            }
            catch (Exception) { /* keep wall clock */ }

            var meta = new FrameMeta { device_id = DeviceId, ts = ts };
            var camPose = cameraAccess.GetCameraPose();
            if (camPose.rotation.w != 0f) // default Pose (camera not ready) has an all-zero quaternion
            {
                var p = camPose.position;
                meta.pose = new Pose { x = p.x, y = p.y, z = p.z, yaw = camPose.rotation.eulerAngles.y };
                RememberPose(ts, camPose);
            }
            else if (headPose != null)
            {
                var p = headPose.position;
                meta.pose = new Pose { x = p.x, y = p.y, z = p.z, yaw = headPose.eulerAngles.y };
            }
            return meta;
        }

        private struct FramePose { public double ts; public UnityEngine.Pose pose; }
        private readonly Queue<FramePose> _poses = new Queue<FramePose>();

        private void RememberPose(double ts, UnityEngine.Pose pose)
        {
            _poses.Enqueue(new FramePose { ts = ts, pose = pose });
            while (_poses.Count > Mathf.Max(1, poseHistory)) _poses.Dequeue();
        }

        /// <summary>Full RGB camera pose of the uploaded frame with FrameMeta.ts == ts (nearest within toleranceS).</summary>
        public bool TryGetFramePose(double ts, out UnityEngine.Pose pose, double toleranceS = 0.05)
        {
            pose = default;
            double best = toleranceS;
            bool found = false;
            foreach (var f in _poses)
            {
                double d = Math.Abs(f.ts - ts);
                if (d <= best) { best = d; pose = f.pose; found = true; }
            }
            return found;
        }

        /// <summary>Downscale the camera texture to <= jpegMaxWidth into _scratch (Blit + ReadPixels, a few ms on device).</summary>
        private bool ReadbackCurrentFrame()
        {
            var src = cameraAccess.GetTexture();
            if (src == null || src.width <= 16) return false;
            int w = src.width, h = src.height;
            float scale = Mathf.Min(1f, (float)settings.jpegMaxWidth / w);
            int tw = Mathf.Max(1, Mathf.RoundToInt(w * scale));
            int th = Mathf.Max(1, Mathf.RoundToInt(h * scale));

            var rt = RenderTexture.GetTemporary(tw, th, 0, RenderTextureFormat.ARGB32);
            var prev = RenderTexture.active;
            try
            {
                Graphics.Blit(src, rt);
                RenderTexture.active = rt;
                if (_scratch == null || _scratch.width != tw || _scratch.height != th)
                {
                    if (_scratch != null) Destroy(_scratch);
                    _scratch = new Texture2D(tw, th, TextureFormat.RGB24, false);
                }
                _scratch.ReadPixels(new Rect(0, 0, tw, th), 0, 0);
                _scratch.Apply(false);
            }
            finally
            {
                RenderTexture.active = prev;
                RenderTexture.ReleaseTemporary(rt);
            }
            return true;
        }

        private IEnumerator Upload(byte[] jpeg, FrameMeta meta)
        {
            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("image", jpeg, "frame.jpg", "image/jpeg"),
                new MultipartFormDataSection("meta", JsonUtility.ToJson(meta)),
            };
            var t0 = Time.realtimeSinceStartup;
            using (var req = UnityWebRequest.Post(settings.Url(config, "/v1/frames"), form))
            {
                MegamindConfig.Authorize(req, settings.ApiToken(config));
                req.timeout = Mathf.Max(1, settings.uploadTimeoutSeconds);
                yield return req.SendWebRequest();
                _pending--;
                LastLatencyMs = (Time.realtimeSinceStartup - t0) * 1000f;
                LastUploadTs = meta.ts;
                if (req.result == UnityWebRequest.Result.Success)
                {
                    Sent++;
                    _consecutiveFailures = 0;
                    _backoffUntil = 0f;
                    LastError = "";
                    FrameAccepted resp = null;
                    try { resp = JsonUtility.FromJson<FrameAccepted>(req.downloadHandler.text); }
                    catch (Exception e) { LastError = "bad json: " + e.Message; }
                    if (resp != null)
                    {
                        if (resp.accepted) Accepted++;
                        LastReason = resp.accepted ? "ok" : (string.IsNullOrEmpty(resp.reason) ? "rejected" : resp.reason);
                    }
                }
                else
                {
                    Sent++;
                    Failed++;
                    _consecutiveFailures++;
                    var body = req.downloadHandler != null ? req.downloadHandler.text : "";
                    LastError = $"{req.responseCode} {req.error} {Truncate(body, 120)}".Trim();
                    LastReason = "net";
                    long code = req.responseCode;
                    if (code == 0 || code == 429 || code >= 500) // unreachable / overloaded; a 4xx is a client bug, retrying faster won't help either way
                    {
                        float delay = settings.backoffBaseSeconds * Mathf.Pow(2f, Mathf.Min(_consecutiveFailures - 1, 10));
                        delay = Mathf.Min(settings.backoffMaxSeconds, delay) * UnityEngine.Random.Range(0.8f, 1.2f);
                        _backoffUntil = Mathf.Max(_backoffUntil, Time.unscaledTime + delay);
                    }
                    if (_consecutiveFailures == 1 || _consecutiveFailures % 20 == 0)
                        Debug.LogWarning($"[Megamind] frame upload failed ({_consecutiveFailures}x): {LastError} url={settings.Url(config, "/v1/frames")}");
                }
            }
        }

        private static string Truncate(string s, int n) => string.IsNullOrEmpty(s) || s.Length <= n ? s : s.Substring(0, n) + "…";

        private void OnDestroy()
        {
            if (_scratch != null) Destroy(_scratch);
        }
    }
}
