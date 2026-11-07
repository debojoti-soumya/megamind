// Stream 3: push-to-talk -> record mic -> POST /v1/chat (multipart WAV) -> play reply audio.
// Also supports text questions for debugging (SendText).
// TODO(stream-3): wire the push-to-talk input (OVRInput trigger / hand pinch) and test on device.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Megamind
{
    public class ChatClient : MonoBehaviour
    {
        public MegamindConfig config;
        public ChatPanel panel;
        public AudioSource speaker;
        public OVRInput.Button pushToTalk = OVRInput.Button.PrimaryIndexTrigger;

        private string _sessionId;
        private AudioClip _recording;
        private bool _recording_active;

        private void Awake() => _sessionId = Guid.NewGuid().ToString("N");

        private void Update()
        {
            if (OVRInput.GetDown(pushToTalk)) BeginRecording();
            if (OVRInput.GetUp(pushToTalk) && _recording_active) StartCoroutine(EndRecordingAndSend());
        }

        public void BeginRecording()
        {
            if (_recording_active) return;
            _recording = Microphone.Start(null, false, config.maxRecordSeconds, config.micSampleRate);
            _recording_active = true;
            panel?.SetListening(true);
        }

        private IEnumerator EndRecordingAndSend()
        {
            int pos = Microphone.GetPosition(null);
            Microphone.End(null);
            _recording_active = false;
            panel?.SetListening(false);
            if (_recording == null || pos <= 0) yield break;

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
                req.timeout = 60;
                panel?.SetThinking(true);
                yield return req.SendWebRequest();
                panel?.SetThinking(false);
                HandleChatResponse(req);
            }
        }

        public void SendText(string text) => StartCoroutine(SendTextCo(text));

        private IEnumerator SendTextCo(string text)
        {
            var body = JsonUtility.ToJson(new ChatRequest { session_id = _sessionId, device_id = config.deviceId, text = text });
            using (var req = new UnityWebRequest(config.Url("/v1/chat"), "POST"))
            {
                req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Content-Type", "application/json");
                req.timeout = 60;
                panel?.AddUserMessage(text);
                panel?.SetThinking(true);
                yield return req.SendWebRequest();
                panel?.SetThinking(false);
                HandleChatResponse(req);
            }
        }

        private void HandleChatResponse(UnityWebRequest req)
        {
            if (req.result != UnityWebRequest.Result.Success)
            {
                panel?.AddAssistantMessage("(network error: " + req.error + ")", null);
                return;
            }
            var resp = JsonUtility.FromJson<ChatResponse>(req.downloadHandler.text);
            if (!string.IsNullOrEmpty(resp.transcript)) panel?.AddUserMessage(resp.transcript);
            panel?.AddAssistantMessage(resp.answer, resp.cited);
            if (!string.IsNullOrEmpty(resp.audio_url)) StartCoroutine(PlayAudio(config.Url(resp.audio_url)));
        }

        private IEnumerator PlayAudio(string url)
        {
            using (var req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV))
            {
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success) yield break;
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
