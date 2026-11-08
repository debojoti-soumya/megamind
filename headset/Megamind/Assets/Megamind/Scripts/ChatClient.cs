// Stream 3: push-to-talk -> record mic -> POST /v1/chat (multipart WAV) -> play reply audio.
// Also supports text questions for debugging (SendText).
//
// Input: controller index trigger (either hand) OR hand-tracking index pinch (OVRHand). Hold to
// record, release to send. Taps shorter than `minRecordSeconds` are ignored. The panel shows a
// listening indicator while recording and a thinking indicator until the server replies.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Networking;

namespace Megamind
{
    public class ChatClient : MonoBehaviour
    {
        public MegamindConfig config;
        public ChatPanel panel;
        public AudioSource speaker;
        [Tooltip("Optional: short spoken commands (\"open menu\", \"pause recording\", ...) are executed instead of shown as an answer.")]
        public PanelMenu menu;

        [Header("Push-to-talk")]
        public OVRInput.Button pushToTalk = OVRInput.Button.PrimaryIndexTrigger;
        public OVRInput.Controller controllers = OVRInput.Controller.Touch;
        [Tooltip("Optional: OVRHand components for pinch-to-talk (stretch).")]
        public OVRHand leftHand, rightHand;
        public float minRecordSeconds = 0.3f;
        [Tooltip("Trigger/pinch while the UI ray is over the panel is a click, not push-to-talk (#146 quirk).")]
        public bool ignorePushToTalkOverUi = true;
        public PanelHover panelHover;

        public event Action<ChatResponse> OnResponse;
        public float LastLatencySeconds { get; private set; }
        public bool IsRecording => _recordingActive;
        public bool IsThinking { get; private set; }

        private string _sessionId;
        private AudioClip _recording;
        private bool _recordingActive;
        private float _recordStart;
        private bool _pinchHeld;

        private void Awake() => _sessionId = Guid.NewGuid().ToString("N");

        private void Update()
        {
            bool pinch = IsPinching(leftHand) || IsPinching(rightHand);
            bool pinchDown = pinch && !_pinchHeld, pinchUp = !pinch && _pinchHeld;
            _pinchHeld = pinch;

            if ((OVRInput.GetDown(pushToTalk, controllers) || pinchDown) && !RayOverUi()) BeginRecording();
            if ((OVRInput.GetUp(pushToTalk, controllers) || pinchUp) && _recordingActive) StartCoroutine(EndRecordingAndSend());
        }

        private bool RayOverUi()
        {
            if (!ignorePushToTalkOverUi) return false;
            if (panelHover != null && panelHover.IsHovered) return true;
            var es = EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }

        /// <summary>Footer "Hold to talk" release: same send path as the trigger release; no-op when not recording.</summary>
        public void EndRecordingAndSendNow()
        {
            if (_recordingActive) StartCoroutine(EndRecordingAndSend());
        }

        private static bool IsPinching(OVRHand hand)
            => hand != null && hand.IsTracked && hand.GetFingerIsPinching(OVRHand.HandFinger.Index)
               && hand.GetFingerConfidence(OVRHand.HandFinger.Index) == OVRHand.TrackingConfidence.High;

        public void BeginRecording()
        {
            if (_recordingActive || IsThinking) return;
            if (Microphone.devices.Length == 0)
            {
                panel?.AddAssistantMessage("(no microphone available)", null, true);
                return;
            }
            if (speaker != null && speaker.isPlaying) speaker.Stop(); // barge-in
            AmbientAudioRecorder.PushToTalkStarting(); // ambient audio memory releases the shared mic
            _recording = Microphone.Start(null, false, config.maxRecordSeconds, config.micSampleRate);
            _recordingActive = true;
            _recordStart = Time.unscaledTime;
            panel?.SetListening(true);
        }

        public void CancelRecording()
        {
            if (!_recordingActive) return;
            Microphone.End(null);
            _recordingActive = false;
            AmbientAudioRecorder.PushToTalkEnded();
            panel?.SetListening(false);
        }

        private IEnumerator EndRecordingAndSend()
        {
            int pos = Microphone.GetPosition(null);
            Microphone.End(null);
            _recordingActive = false;
            AmbientAudioRecorder.PushToTalkEnded();
            panel?.SetListening(false);
            if (_recording == null || pos <= 0 || Time.unscaledTime - _recordStart < minRecordSeconds) yield break;

            var samples = new float[pos * _recording.channels];
            _recording.GetData(samples, 0);
            byte[] wav = WavEncoder.Encode(samples, _recording.channels, config.micSampleRate);

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormDataSection("session_id", _sessionId),
                new MultipartFormDataSection("device_id", config.deviceId),
                new MultipartFormFileSection("audio", wav, "q.wav", "audio/wav"),
            };
            using (var req = UnityWebRequest.Post(config.Url("/v1/chat"), form))
            {
                config.Authorize(req);
                req.timeout = config.chatTimeoutSeconds;
                yield return Send(req, echoTranscript: true);
            }
        }

        public void SendText(string text) => StartCoroutine(SendTextCo(text));

        private IEnumerator SendTextCo(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) yield break;
            var body = JsonUtility.ToJson(new ChatRequest { session_id = _sessionId, device_id = config.deviceId, text = text });
            using (var req = new UnityWebRequest(config.Url("/v1/chat"), "POST"))
            {
                req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                config.Authorize(req);
                req.timeout = config.chatTimeoutSeconds;
                panel?.AddUserMessage(text);
                yield return Send(req, echoTranscript: false);
            }
        }

        private IEnumerator Send(UnityWebRequest req, bool echoTranscript)
        {
            IsThinking = true;
            panel?.SetThinking(true);
            float t0 = Time.unscaledTime;
            yield return req.SendWebRequest();
            LastLatencySeconds = Time.unscaledTime - t0;
            IsThinking = false;
            panel?.SetThinking(false);
            HandleChatResponse(req, echoTranscript);
        }

        private void HandleChatResponse(UnityWebRequest req, bool echoTranscript)
        {
            if (req.result != UnityWebRequest.Result.Success)
            {
                panel?.AddAssistantMessage("(network error: " + req.error + ")", null, true);
                return;
            }
            ChatResponse resp;
            try { resp = JsonUtility.FromJson<ChatResponse>(req.downloadHandler.text); }
            catch (Exception e)
            {
                panel?.AddAssistantMessage("(bad response: " + e.Message + ")", null, true);
                return;
            }
            // STT happens inside /v1/chat, so voice commands can only be recognised from the returned transcript.
            // The server has already run the agent (LLM + TTS) on the utterance; a matched command discards that answer.
            // A future STT-only endpoint (e.g. /v1/stt) would let commands skip the agent turn.
            if (echoTranscript && menu != null && menu.TryVoiceCommand(resp.transcript)) return;
            if (echoTranscript && !string.IsNullOrEmpty(resp.transcript)) panel?.AddUserMessage(resp.transcript);
            panel?.AddAssistantMessage(resp.answer, resp.cited, false, resp.Confidence(), LastLatencySeconds,
                resp.HasCollage() ? resp.collage : null);
            OnResponse?.Invoke(resp);
            if (!string.IsNullOrEmpty(resp.audio_url) && config.speakAnswers) StartCoroutine(PlayAudio(config.Url(resp.audio_url)));
        }

        public void NewSession() => _sessionId = Guid.NewGuid().ToString("N");

        private IEnumerator PlayAudio(string url)
        {
            using (var req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV))
            {
                config.Authorize(req);
                req.timeout = 20;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success || speaker == null) yield break;
                speaker.clip = DownloadHandlerAudioClip.GetContent(req);
                speaker.Play();
            }
        }
    }

    public static class WavEncoder
    {
        public static byte[] Encode(float[] samples, int channels, int sampleRate)
        {
            using (var ms = new MemoryStream())
            using (var w = new BinaryWriter(ms))
            {
                int dataLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + dataLen);
                w.Write(Encoding.ASCII.GetBytes("WAVE")); w.Write(Encoding.ASCII.GetBytes("fmt "));
                w.Write(16); w.Write((short)1); w.Write((short)channels); w.Write(sampleRate);
                w.Write(sampleRate * channels * 2); w.Write((short)(channels * 2)); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(dataLen);
                foreach (var s in samples) w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
                return ms.ToArray();
            }
        }
    }
}
