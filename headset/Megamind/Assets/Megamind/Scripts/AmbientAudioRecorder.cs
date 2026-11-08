// Stream 1: ambient audio memory, so "what did I hear?" works from the Unity app.
// Records the headset mic through Unity's Microphone API (16 kHz mono), cuts it into fixed chunks
// (CaptureSettings.audioChunkSeconds = 5 s, the native headset-android app's default chunk_sec) and POSTs each
// chunk to /v1/audio as a PCM16 mono WAV + meta {"device_id", "ts_start"} (docs/api.md, AudioChunkCodec).
//
//   ts_start   wall clock (Unix s) of the chunk's first sample, the same clock FrameUploader stamps frames with;
//              re-anchored per 100 ms block from Microphone.GetPosition, so it does not drift over a session.
//   silence    a chunk whose loudest 100 ms window is below audioSilenceDb is dropped on the device (saves Whisper cost).
//   uploads    one at a time, oldest first (the server merges consecutive chunks into one transcript token, so order
//              matters); bounded in-memory queue; network error / 429 / 5xx -> exponential backoff (+-20% jitter).
//   pauses     app backgrounded / unfocused (same hooks as FrameUploader), capture paused with A/X (one button stops
//              all recording), and while ChatClient's push-to-talk owns the microphone (PushToTalkStarting/Ended).
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

namespace Megamind
{
    public class AmbientAudioRecorder : MonoBehaviour
    {
        public enum State { Off, WaitingForPermission, NoMicrophone, Recording, Paused, PushToTalk }

        [Tooltip("Shared config (server URL, device id). Only read here.")]
        public MegamindConfig config;
        [Tooltip("CaptureSettings.ambientAudio toggles this recorder; audio* fields tune it.")]
        public CaptureSettings settings;
        [Tooltip("Audio follows the frame-capture pause (A/X). Found automatically when empty.")]
        public FrameUploader uploader;
        [Tooltip("CenterEyeAnchor, for the optional pose in meta. Found automatically when empty.")]
        public Transform headPose;

        // --- status (read by CaptureHud) ---
        public State CurrentState { get; private set; } = State.Off;
        public int ChunksRecorded { get; private set; }
        public int Sent { get; private set; }            // upload attempts answered (or failed)
        public int Accepted { get; private set; }        // 200 OK
        public int Failed { get; private set; }
        public int DroppedSilent { get; private set; }   // below the RMS gate, never uploaded
        public int DroppedQueue { get; private set; }    // oldest chunk dropped because the queue was full
        public int DroppedRejected { get; private set; } // 4xx, or retries exhausted
        public int SpeechTokens { get; private set; }    // token_ids returned by the server
        public int Queued => _queue.Count;
        public float LastLevelDb { get; private set; } = -120f;
        public string LastTranscript { get; private set; } = "";
        public string LastError { get; private set; } = "";
        public int SampleRate => _rate;
        public bool ServerReachable => Sent > 0 && _consecutiveFailures == 0;

        private static AmbientAudioRecorder s_instance;
        private static bool s_pushToTalk;
        private static float s_pushToTalkSince;
        private const float PushToTalkStaleSeconds = 120f; // a ChatClient destroyed mid-recording must not mute us forever
        private const float MinChunkSeconds = 1f;          // like the native ChunkStore: < ~1 s of audio is not uploaded

        private class Chunk { public byte[] wav; public string meta; public string fileName; public int attempts; }

        private readonly Queue<Chunk> _queue = new Queue<Chunk>();
        private AudioClip _clip;
        private bool _micOn;
        private int _rate = 16000, _channels = 1, _readPos, _blockFrames;
        private float[] _block, _chunk, _wrapA, _wrapB;
        private int _chunkFill;
        private double _chunkTs;
        private Pose _chunkPose;
        private float _nextMicRetry;
        private bool _appActive = true;
        private int _consecutiveFailures;
        private float _backoffUntil;

        /// <summary>Called by ChatClient right before it starts push-to-talk on the (shared, default) microphone.
        /// Flushes the partial chunk and releases the device synchronously so Microphone.Start(null, ...) gets it.</summary>
        public static void PushToTalkStarting()
        {
            s_pushToTalk = true;
            s_pushToTalkSince = Time.unscaledTime;
            if (s_instance != null) s_instance.StopMic(flush: true);
        }

        /// <summary>Called by ChatClient after Microphone.End(null); ambient recording resumes on the next Update.</summary>
        public static void PushToTalkEnded() => s_pushToTalk = false;

        private void Awake()
        {
            s_instance = this;
            settings = CaptureSettings.Resolve(settings);
        }

        private void Start()
        {
            if (uploader == null) uploader = FindAnyObjectByType<FrameUploader>();
            if (headPose == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) headPose = rig.centerEyeAnchor != null ? rig.centerEyeAnchor : rig.transform;
            }
            Debug.Log($"[Megamind] ambient audio: {(settings.ambientAudio ? "on" : "off")} chunk={settings.audioChunkSeconds}s " +
                      $"gate={settings.audioSilenceDb:0} dBFS url={settings.Url(config, "/v1/audio")}");
            StartCoroutine(UploadLoop());
        }

        private void Update()
        {
            if (s_pushToTalk && Time.unscaledTime - s_pushToTalkSince > PushToTalkStaleSeconds) s_pushToTalk = false;

            if (!settings.ambientAudio) { StopMic(flush: true); CurrentState = State.Off; return; }
            if (s_pushToTalk) { CurrentState = State.PushToTalk; return; } // mic already released in PushToTalkStarting
            if (!_appActive || (uploader != null && !uploader.Capturing)) { StopMic(flush: true); CurrentState = State.Paused; return; }
            if (!MegamindPermissions.MicrophoneGranted) { CurrentState = State.WaitingForPermission; return; }
            if (Microphone.devices.Length == 0) { CurrentState = State.NoMicrophone; return; }

            if (!_micOn)
            {
                if (Time.unscaledTime < _nextMicRetry) return;
                if (!StartMic()) { _nextMicRetry = Time.unscaledTime + 3f; return; }
            }
            if (!Microphone.IsRecording(null))
            {
                // Someone else ended the default device (e.g. an older push-to-talk path without the hook): restart.
                LastError = "microphone stopped externally";
                StopMic(flush: true);
                _nextMicRetry = Time.unscaledTime + 1f;
                return;
            }
            CurrentState = State.Recording;
            Pump(final: false);
        }

        private bool StartMic()
        {
            int rate = Mathf.Max(8000, settings.audioSampleRate);
            Microphone.GetDeviceCaps(null, out int minFreq, out int maxFreq);
            if (minFreq > 0 || maxFreq > 0) rate = Mathf.Clamp(rate, minFreq, maxFreq); // 0/0 = any rate
            int chunkSeconds = Mathf.Clamp(settings.audioChunkSeconds, 1, 60);
            int ringSeconds = Mathf.Max(10, chunkSeconds * 2);
            _clip = Microphone.Start(null, true, ringSeconds, rate);
            if (_clip == null)
            {
                LastError = "Microphone.Start failed";
                Debug.LogWarning("[Megamind] ambient audio: Microphone.Start returned null; retrying");
                return false;
            }
            _rate = _clip.frequency;
            _channels = Mathf.Max(1, _clip.channels);
            _blockFrames = Mathf.Max(1, _rate / 10);
            _block = new float[_blockFrames * _channels];
            int chunkFrames = _rate * chunkSeconds;
            if (_chunk == null || _chunk.Length != chunkFrames) _chunk = new float[chunkFrames];
            _chunkFill = 0;
            _readPos = 0;
            _micOn = true;
            Debug.Log($"[Megamind] ambient audio: mic started {_rate} Hz x{_channels}, {chunkSeconds} s chunks");
            return true;
        }

        /// <summary>Releases the microphone; with flush, the partial chunk (>= 1 s) is queued first.</summary>
        private void StopMic(bool flush)
        {
            if (!_micOn) return;
            if (flush)
            {
                try { Pump(final: true); }
                catch (Exception e) { LastError = "flush: " + e.Message; }
                FinishChunk();
            }
            Microphone.End(null);
            _micOn = false;
            _chunkFill = 0;
            if (_clip != null) Destroy(_clip);
            _clip = null;
        }

        /// <summary>Moves recorded samples from the looping mic clip into the current chunk, 100 ms at a time
        /// (fixed-size reads, no per-frame allocation). With final, the sub-block remainder is read too.</summary>
        private void Pump(bool final)
        {
            if (_clip == null) return;
            int pos = Microphone.GetPosition(null);
            if (pos < 0) return;
            int total = _clip.samples;
            int avail = pos - _readPos;
            if (avail < 0) avail += total;
            double now = WallClock();
            while (avail >= _blockFrames)
            {
                Read(_block, _readPos, _blockFrames, total);
                Append(_block, _blockFrames, now - (double)avail / _rate);
                _readPos = (_readPos + _blockFrames) % total;
                avail -= _blockFrames;
            }
            if (final && avail > 0)
            {
                var rest = new float[avail * _channels];
                Read(rest, _readPos, avail, total);
                Append(rest, avail, now - (double)avail / _rate);
                _readPos = (_readPos + avail) % total;
            }
        }

        /// <summary>frames starting at pos of the looping mic clip; split explicitly at the wrap point.</summary>
        private void Read(float[] dst, int pos, int frames, int total)
        {
            int first = Mathf.Min(frames, total - pos);
            if (first == frames) { _clip.GetData(dst, pos); return; }
            if (_wrapA == null || _wrapA.Length != first * _channels) _wrapA = new float[first * _channels];
            int second = frames - first;
            if (_wrapB == null || _wrapB.Length != second * _channels) _wrapB = new float[second * _channels];
            _clip.GetData(_wrapA, pos);
            _clip.GetData(_wrapB, 0);
            Array.Copy(_wrapA, 0, dst, 0, _wrapA.Length);
            Array.Copy(_wrapB, 0, dst, _wrapA.Length, _wrapB.Length);
        }

        private void Append(float[] interleaved, int frames, double firstFrameTs)
        {
            for (int i = 0; i < frames; i++)
            {
                if (_chunkFill == 0)
                {
                    _chunkTs = firstFrameTs + (double)i / _rate;
                    _chunkPose = CurrentPose();
                }
                float v;
                if (_channels == 1) v = interleaved[i];
                else
                {
                    v = 0f;
                    for (int c = 0; c < _channels; c++) v += interleaved[i * _channels + c];
                    v /= _channels;
                }
                _chunk[_chunkFill++] = v;
                if (_chunkFill >= _chunk.Length) FinishChunk();
            }
        }

        private void FinishChunk()
        {
            int n = _chunkFill;
            _chunkFill = 0;
            if (n < _rate * MinChunkSeconds) return;
            ChunksRecorded++;
            LastLevelDb = AudioChunkCodec.LoudestWindowDb(_chunk, n, _rate / 10);
            if (settings.audioSilenceDb > -120f && LastLevelDb < settings.audioSilenceDb)
            {
                DroppedSilent++;
                return;
            }
            var chunk = new Chunk
            {
                wav = AudioChunkCodec.EncodeWav(_chunk, n, _rate),
                meta = AudioChunkCodec.MetaJson(DeviceId, _chunkTs, _chunkPose),
                fileName = AudioChunkCodec.ChunkFileName(_chunkTs),
            };
            _queue.Enqueue(chunk);
            while (_queue.Count > Mathf.Max(1, settings.audioMaxQueuedChunks))
            {
                _queue.Dequeue();
                DroppedQueue++;
            }
        }

        private IEnumerator UploadLoop()
        {
            while (true)
            {
                if (_queue.Count == 0 || !_appActive || Time.unscaledTime < _backoffUntil)
                {
                    yield return null;
                    continue;
                }
                var chunk = _queue.Peek();
                var form = new List<IMultipartFormSection>
                {
                    new MultipartFormFileSection("audio", chunk.wav, chunk.fileName, "audio/wav"),
                    new MultipartFormDataSection("meta", chunk.meta),
                };
                string url = settings.Url(config, "/v1/audio");
                using (var req = UnityWebRequest.Post(url, form))
                {
                    req.timeout = Mathf.Max(1, settings.audioUploadTimeoutSeconds); // the server transcribes synchronously
                    yield return req.SendWebRequest();
                    Sent++;
                    long code = req.responseCode;
                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        Accepted++;
                        _consecutiveFailures = 0;
                        _backoffUntil = 0f;
                        LastError = "";
                        try
                        {
                            var resp = JsonUtility.FromJson<AudioAccepted>(req.downloadHandler.text);
                            if (resp != null)
                            {
                                SpeechTokens += resp.token_ids != null ? resp.token_ids.Count : 0;
                                if (!string.IsNullOrEmpty(resp.transcript)) LastTranscript = resp.transcript;
                            }
                        }
                        catch (Exception e) { LastError = "bad json: " + e.Message; }
                        DequeueIfHead(chunk);
                        continue;
                    }

                    Failed++;
                    _consecutiveFailures++;
                    var body = req.downloadHandler != null ? req.downloadHandler.text : "";
                    LastError = $"{code} {req.error} {Truncate(body, 120)}".Trim();
                    bool retryable = code == 0 || code == 429 || code >= 500;
                    chunk.attempts++;
                    // Unreachable server (code 0): keep the chunk (bounded queue). 429/5xx (e.g. STT provider error): a few
                    // tries, like the native app. Other 4xx: client/contract bug, retrying will not help.
                    if (!retryable || (code != 0 && chunk.attempts >= Mathf.Max(1, settings.audioMaxAttempts)))
                    {
                        DroppedRejected++;
                        DequeueIfHead(chunk);
                    }
                    if (retryable)
                    {
                        float delay = settings.backoffBaseSeconds * Mathf.Pow(2f, Mathf.Min(_consecutiveFailures - 1, 10));
                        delay = Mathf.Min(settings.backoffMaxSeconds, delay) * UnityEngine.Random.Range(0.8f, 1.2f);
                        _backoffUntil = Time.unscaledTime + delay;
                    }
                    if (_consecutiveFailures == 1 || _consecutiveFailures % 20 == 0)
                        Debug.LogWarning($"[Megamind] audio upload failed ({_consecutiveFailures}x): {LastError} url={url}");
                }
            }
        }

        private void DequeueIfHead(Chunk chunk)
        {
            if (_queue.Count > 0 && ReferenceEquals(_queue.Peek(), chunk)) _queue.Dequeue();
        }

        private string DeviceId => settings.DeviceId(config);

        private Pose CurrentPose()
        {
            if (headPose == null) return null;
            var p = headPose.position;
            return new Pose { x = p.x, y = p.y, z = p.z, yaw = headPose.eulerAngles.y };
        }

        /// <summary>Same clock as FrameUploader's fallback frame timestamp.</summary>
        private static double WallClock() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;

        private void OnApplicationPause(bool paused) => SetAppActive(!paused);

        private void OnApplicationFocus(bool focused)
        {
#if !UNITY_EDITOR
            SetAppActive(focused); // in the Editor, clicking another window must not pause recording
#endif
        }

        private void SetAppActive(bool active)
        {
            if (active == _appActive) return;
            _appActive = active;
            Debug.Log($"[Megamind] app {(active ? "active: resuming ambient audio" : "backgrounded: pausing ambient audio")}");
            if (!active) StopMic(flush: true); // the app may be suspended before the next Update
        }

        private void OnDisable() => StopMic(flush: false);

        private void OnDestroy()
        {
            if (s_instance == this) s_instance = null;
        }

        private static string Truncate(string s, int n) => string.IsNullOrEmpty(s) || s.Length <= n ? s : s.Substring(0, n) + "…";
    }
}
