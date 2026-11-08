// Stream 3: push-to-talk -> record mic -> POST /v1/chat/stream (multipart WAV) -> show the answer as it streams and
// play its sentence clips as they arrive. Also supports typed questions (SendText). Servers without the stream
// endpoint (404/405), or a stream that fails before its first event, get the same request as POST /v1/chat.
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

        [Header("Streaming")]
        [Tooltip("POST /v1/chat/stream: show the answer as it is written and speak it sentence by sentence. Falls back to " +
                 "POST /v1/chat on servers without the stream endpoint.")]
        public bool streamAnswers = true;

        public event Action<ChatResponse> OnResponse;
        /// <summary>Seconds from sending the question to the complete answer.</summary>
        public float LastLatencySeconds { get; private set; }
        /// <summary>Streamed turns: seconds to the first answer words / the first spoken sentence (0 = none).</summary>
        public float LastFirstTextSeconds { get; private set; }
        public float LastFirstAudioSeconds { get; private set; }
        public bool IsRecording => _recordingActive;
        public bool IsThinking { get; private set; }
        /// <summary>Mic input level 0..1 while recording (smoothed RMS), for the hold-to-talk meter.</summary>
        public float MicLevel { get; private set; }

        public enum Phase { Idle, Listening, Transcribing, Thinking, Answering, Speaking }
        private Phase _turnPhase = Phase.Idle;
        /// <summary>What the footer status line shows: Listening / Transcribing / Thinking / Answering / Speaking.</summary>
        public Phase CurrentPhase
        {
            get
            {
                if (_recordingActive) return Phase.Listening;
                if (IsThinking) return _turnPhase;
                if (speaker != null && speaker.isPlaying) return Phase.Speaking;
                return Phase.Idle;
            }
        }

        [Tooltip("Optional: persistent chats; the active chat's id is the server session_id. Without it, a random id per launch.")]
        public SessionStore sessions;

        private string _sessionId;
        /// <summary>Server session_id for the current turn (active chat when a SessionStore is wired).</summary>
        public string SessionId => sessions != null && sessions.Active != null ? sessions.Active.id : _sessionId;
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
            UpdateMicLevel();
        }

        private readonly float[] _levelBuf = new float[512];
        private void UpdateMicLevel()
        {
            if (!_recordingActive || _recording == null) { MicLevel = Mathf.MoveTowards(MicLevel, 0f, Time.unscaledDeltaTime * 4f); return; }
            int pos = Microphone.GetPosition(null);
            int n = _levelBuf.Length;
            if (pos < n) return;
            _recording.GetData(_levelBuf, pos - n);
            float sum = 0f;
            for (int i = 0; i < n; i++) sum += _levelBuf[i] * _levelBuf[i];
            float rms = Mathf.Sqrt(sum / n);
            float level = Mathf.Clamp01(rms * 6f); // speech peaks ~0.15 rms -> ~0.9
            MicLevel = level > MicLevel ? level : Mathf.Lerp(MicLevel, level, Time.unscaledDeltaTime * 8f);
        }

        /// <summary>Asks the last question again (typed text or voice transcript), as text.</summary>
        public void Retry()
        {
            string q = panel != null ? panel.LastQuestion : null;
            if (string.IsNullOrWhiteSpace(q) || IsThinking) return;
            SendText(q);
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
            StopSpeech(); // barge-in: also drops queued sentence clips
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
            if (IsThinking) yield break; // a typed question is still being answered

            var samples = new float[pos * _recording.channels];
            _recording.GetData(samples, 0);
            byte[] wav = WavEncoder.Encode(samples, _recording.channels, config.micSampleRate);

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormDataSection("session_id", SessionId),
                new MultipartFormDataSection("device_id", config.deviceId),
                new MultipartFormFileSection("audio", wav, "q.wav", "audio/wav"),
            };
            // Same bytes and Content-Type as UnityWebRequest.Post(url, form), serialised once for stream + fallback.
            byte[] boundary = UnityWebRequest.GenerateBoundary();
            byte[] body = UnityWebRequest.SerializeFormSections(form, boundary);
            string contentType = "multipart/form-data; boundary=" + Encoding.UTF8.GetString(boundary, 0, boundary.Length);
            yield return Ask(body, contentType, voice: true);
        }

        public void SendText(string text) => StartCoroutine(SendTextCo(text));

        private IEnumerator SendTextCo(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) yield break;
            if (IsThinking)
            {
                panel?.AddSystemMessage("(still answering the last question)");
                yield break;
            }
            var body = JsonUtility.ToJson(new ChatRequest { session_id = SessionId, device_id = config.deviceId, text = text });
            panel?.AddUserMessage(text);
            yield return Ask(Encoding.UTF8.GetBytes(body), "application/json", voice: false);
        }

        private UnityWebRequest MakeChatRequest(string path, byte[] body, string contentType, DownloadHandler download)
        {
            var upload = new UploadHandlerRaw(body) { contentType = contentType };
            var req = new UnityWebRequest(config.Url(path), "POST") { uploadHandler = upload, downloadHandler = download };
            req.SetRequestHeader("Content-Type", contentType);
            config.Authorize(req);
            req.timeout = config.chatTimeoutSeconds;
            return req;
        }

        /// <summary>One question: POST /v1/chat/stream (text and sentence audio as they are generated), or POST /v1/chat
        /// when streaming is off, the server has no stream endpoint (404/405), or the stream failed before any event.</summary>
        private IEnumerator Ask(byte[] body, string contentType, bool voice)
        {
            StopSpeech(); // the previous answer's audio, if still playing
            IsThinking = true;
            _turnPhase = voice ? Phase.Transcribing : Phase.Thinking;
            panel?.SetThinking(true);
            float t0 = Time.unscaledTime;
            LastFirstTextSeconds = 0f;
            LastFirstAudioSeconds = 0f;

            bool fallBack = true;
            if (streamAnswers && !StreamUnsupported)
            {
                var turn = new StreamTurn { Voice = voice, T0 = t0, Released = !voice };
                yield return StreamTurnCo(body, contentType, turn);
                fallBack = turn.FallBack;
            }
            if (fallBack)
            {
                using (var req = MakeChatRequest("/v1/chat", body, contentType, new DownloadHandlerBuffer()))
                    yield return Send(req, voice, t0);
                yield break;
            }
            IsThinking = false;
            panel?.SetThinking(false);
        }

        private IEnumerator Send(UnityWebRequest req, bool echoTranscript, float t0)
        {
            IsThinking = true;
            panel?.SetThinking(true);
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
            if (resp == null)
            {
                panel?.AddAssistantMessage("(bad response: empty body)", null, true);
                return;
            }
            // POST /v1/chat runs STT + the whole turn (LLM + TTS) before the headset sees the transcript, so a matched
            // command discards that answer. The stream path matches on the early `transcript` event and cancels instead.
            if (echoTranscript && menu != null && menu.TryVoiceCommand(resp.transcript)) return;
            if (echoTranscript && !string.IsNullOrEmpty(resp.transcript)) panel?.AddUserMessage(resp.transcript);
            panel?.AddAssistantMessage(resp.answer, resp.cited, false, resp.Confidence(), LastLatencySeconds);
            OnResponse?.Invoke(resp);
            if (!string.IsNullOrEmpty(resp.audio_url) && config.speakAnswers) StartCoroutine(PlayAudio(config.Url(resp.audio_url)));
        }

        // ---- POST /v1/chat/stream (docs/api.md): transcript, retrieval, delta, reset, audio, final | error ----

        private class StreamTurn
        {
            public bool Voice;
            public float T0;
            public bool Released;       // voice: false until the transcript is known not to be a command
            public bool TranscriptSeen;
            public bool Over;           // final / error handled, or cancelled (voice command)
            public bool FallBack;       // retry as POST /v1/chat
            public int Events;
            public int AudioEvents;
            public float FirstText;
            public List<Citation> Candidates;
            public readonly StringBuilder Text = new StringBuilder();
            public SpeechQueue Speech;
        }

        private string _noStreamServer; // serverBaseUrl that answered 404/405 to /v1/chat/stream (older server)
        private bool StreamUnsupported => _noStreamServer != null && _noStreamServer == config.serverBaseUrl;

        private IEnumerator StreamTurnCo(byte[] body, string contentType, StreamTurn turn)
        {
            var sse = new SseDownloadHandler();
            var events = new List<SseEvent>();
            using (var req = MakeChatRequest("/v1/chat/stream", body, contentType, sse))
            {
                req.SetRequestHeader("Accept", "text/event-stream");
                var op = req.SendWebRequest();
                while (true)
                {
                    bool finished = op.isDone; // read first: events that completed with the response are drained below
                    long code = req.responseCode;
                    if (code >= 200 && code < 300 && sse.Drain(events) > 0)
                    {
                        foreach (var e in events)
                        {
                            if (turn.Over) break;
                            HandleStreamEvent(e, turn);
                        }
                        events.Clear();
                    }
                    if (turn.Over)
                    {
                        if (!finished) req.Abort(); // a voice command: the server stops the turn when the client goes away
                        break;
                    }
                    if (finished) break;
                    yield return null;
                }
                if (!turn.Over) EndWithoutFinal(req, turn);
            }
        }

        private void EndWithoutFinal(UnityWebRequest req, StreamTurn turn)
        {
            long code = req.responseCode;
            if (turn.Events == 0)
            {
                if (code == 404 || code == 405) _noStreamServer = config.serverBaseUrl; // older server: POST /v1/chat from now on
                // Nothing was shown yet, so POST /v1/chat can safely answer instead (it reports the same HTTP errors).
                // Not after a timeout-length wait, which would only double it.
                if (code == 404 || code == 405 || Time.unscaledTime - turn.T0 < config.chatTimeoutSeconds * 0.5f)
                {
                    turn.FallBack = true;
                    return;
                }
            }
            Debug.LogWarning($"[Megamind] chat stream ended without an answer ({code}, {req.error}, {turn.Events} events)");
            FailTurn(turn, "(network error: " + (string.IsNullOrEmpty(req.error) ? "the answer stream ended early" : req.error) + ")");
        }

        private static T ParseEvent<T>(string json) where T : class
        {
            if (string.IsNullOrEmpty(json)) return null;
            try { return JsonUtility.FromJson<T>(json); }
            catch (Exception) { return null; }
        }

        private void HandleStreamEvent(SseEvent e, StreamTurn turn)
        {
            turn.Events++;
            switch (e.Event)
            {
                case "transcript":
                    OnTranscript(ParseEvent<ChatStreamTranscript>(e.Data)?.text, turn);
                    break;
                case "delta":
                    var delta = ParseEvent<ChatStreamDelta>(e.Data);
                    if (delta == null || string.IsNullOrEmpty(delta.text)) break;
                    turn.Text.Append(delta.text);
                    ShowDraft(turn);
                    break;
                case "reset": // that text will not be the answer: drop it, and any of its speech not yet played
                    turn.Text.Length = 0;
                    if (turn.Released) panel?.SetDraft("");
                    if (turn.Speech != null)
                    {
                        turn.Speech.DropPending();
                        if (turn.Speech == _speech && speaker != null && speaker.isPlaying) speaker.Stop();
                    }
                    break;
                case "audio":
                    OnAudio(ParseEvent<ChatStreamAudio>(e.Data), turn);
                    break;
                case "final":
                    var resp = ParseEvent<ChatResponse>(e.Data);
                    if (resp == null) FailTurn(turn, "(bad response: unreadable final event)");
                    else OnFinal(resp, turn);
                    break;
                case "error":
                    var err = ParseEvent<ChatStreamError>(e.Data);
                    FailTurn(turn, "(error: " + (string.IsNullOrEmpty(err?.detail) ? "the answer failed" : err.detail) + ")");
                    break;
                case "retrieval": // candidate moments: preview them now; `final` replaces them with the cited cards
                    var ret = ParseEvent<ChatStreamRetrieval>(e.Data);
                    if (ret?.candidates != null && ret.candidates.Count > 0)
                    {
                        turn.Candidates = ret.candidates;
                        if (turn.Released) panel?.ShowCandidates(turn.Candidates);
                    }
                    break;
                // unknown events: ignored
            }
        }

        private void OnTranscript(string text, StreamTurn turn)
        {
            if (turn.TranscriptSeen) return;
            turn.TranscriptSeen = true;
            if (!turn.Voice) return; // typed: the question bubble is already shown
            if (menu != null && menu.TryVoiceCommand(text))
            {
                CancelTurn(turn);
                return;
            }
            if (!string.IsNullOrEmpty(text)) panel?.AddUserMessage(text);
            _turnPhase = Phase.Thinking;
            Release(turn);
        }

        /// <summary>Voice turns hold back answer text and audio until the transcript is known not to be a command.</summary>
        private void Release(StreamTurn turn)
        {
            if (turn.Released) return;
            turn.Released = true;
            if (turn.Speech != null) turn.Speech.Released = true;
            if (turn.Candidates != null) panel?.ShowCandidates(turn.Candidates);
            if (turn.Text.Length > 0) ShowDraft(turn);
        }

        private void ShowDraft(StreamTurn turn)
        {
            if (!turn.Released || turn.Text.Length == 0) return;
            panel?.SetDraft(turn.Text.ToString());
            _turnPhase = Phase.Answering;
            if (turn.FirstText > 0f) return;
            turn.FirstText = LastFirstTextSeconds = Mathf.Max(0.001f, Time.unscaledTime - turn.T0);
            panel?.SetThinking(false); // the growing bubble shows progress now
        }

        private void OnFinal(ChatResponse resp, StreamTurn turn)
        {
            turn.Over = true;
            LastLatencySeconds = Time.unscaledTime - turn.T0;
            if (turn.Voice && !turn.TranscriptSeen) // server without the `transcript` event: same as POST /v1/chat
            {
                turn.TranscriptSeen = true;
                if (menu != null && menu.TryVoiceCommand(resp.transcript))
                {
                    CancelTurn(turn);
                    return;
                }
                if (!string.IsNullOrEmpty(resp.transcript)) panel?.AddUserMessage(resp.transcript);
            }
            turn.Released = true;
            panel?.FinishDraft(resp.answer, resp.cited, resp.Confidence(), LastLatencySeconds, turn.FirstText);
            OnResponse?.Invoke(resp);
            string audioUrl = string.IsNullOrEmpty(resp.audio_url) ? null : config.Url(resp.audio_url);
            if (turn.Speech != null)
            {
                // sentence clips play; final.audio_url only if none of them could be played
                turn.Speech.FallbackUrl = audioUrl;
                turn.Speech.Released = true;
                turn.Speech.Ended = true;
            }
            else if (turn.AudioEvents == 0 && audioUrl != null && config.speakAnswers) StartCoroutine(PlayAudio(audioUrl));
            Debug.Log($"[Megamind] chat stream: first text {turn.FirstText:0.00}s, answer {LastLatencySeconds:0.00}s, " +
                      $"{turn.AudioEvents} audio clips, {turn.Events} events");
        }

        private void FailTurn(StreamTurn turn, string message)
        {
            turn.Over = true;
            panel?.DiscardDraft();
            panel?.ClearCandidates();
            if (turn.Speech != null && turn.Speech == _speech) StopSpeech();
            else turn.Speech?.Cancel();
            panel?.AddAssistantMessage(message, null, true);
        }

        private void CancelTurn(StreamTurn turn)
        {
            turn.Over = true;
            panel?.DiscardDraft();
            panel?.ClearCandidates();
            if (turn.Speech != null && turn.Speech == _speech) StopSpeech();
            else turn.Speech?.Cancel();
        }

        // ---- answer speech: sentence clips (`audio` events) in seq order, or final.audio_url ----

        private class SpeechClip
        {
            public int Seq;
            public AudioClip Clip;
            public bool Done;    // decoded / downloaded, or failed (Clip null)
            public bool Dropped; // a reset or cancel came first: destroy the clip when it arrives
        }

        private class SpeechQueue
        {
            public readonly List<SpeechClip> Pending = new List<SpeechClip>();
            public bool Released, Ended, Cancelled;
            public string FallbackUrl;
            public int Played; // since the last reset
            public float T0;

            public void Insert(SpeechClip c)
            {
                int i = Pending.Count;
                while (i > 0 && Pending[i - 1].Seq > c.Seq) i--;
                Pending.Insert(i, c);
            }

            public void DropPending()
            {
                foreach (var c in Pending)
                {
                    c.Dropped = true;
                    if (c.Clip != null) UnityEngine.Object.Destroy(c.Clip);
                    c.Clip = null;
                }
                Pending.Clear();
                Played = 0;
            }

            public void Cancel()
            {
                Cancelled = true;
                DropPending();
            }
        }

        private SpeechQueue _speech;   // the queue currently allowed to use the speaker
        private int _speechGen;        // bumped by StopSpeech; a final.audio_url download from an older turn is dropped
        private AudioClip _ownedClip;  // last clip this client put on the speaker (destroyed when replaced)

        private void StopSpeech()
        {
            _speechGen++;
            if (_speech != null) _speech.Cancel();
            _speech = null;
            if (speaker != null && speaker.isPlaying) speaker.Stop();
        }

        private void OnAudio(ChatStreamAudio a, StreamTurn turn)
        {
            turn.AudioEvents++;
            if (a == null || !config.speakAnswers || speaker == null) return;
            if (turn.Speech == null)
            {
                turn.Speech = new SpeechQueue { Released = turn.Released, T0 = turn.T0 };
                _speech = turn.Speech;
                StartCoroutine(PlaySpeech(turn.Speech));
            }
            var clip = new SpeechClip { Seq = a.seq };
            turn.Speech.Insert(clip);
            if (!string.IsNullOrEmpty(a.b64)) DecodeInlineClip(clip, a);
            else if (!string.IsNullOrEmpty(a.url)) StartCoroutine(FetchClip(clip, a, turn.Speech));
            else clip.Done = true;
        }

        private static void DecodeInlineClip(SpeechClip c, ChatStreamAudio a)
        {
            c.Done = true;
            byte[] bytes;
            try { bytes = Convert.FromBase64String(a.b64); }
            catch (FormatException) { Debug.LogWarning("[Megamind] audio event: bad base64"); return; }
            if (!SpeechClipCodec.TryDecode(bytes, a.format, a.sample_rate, 1, out var samples, out int channels, out int rate)
                || samples.Length < channels)
            {
                Debug.LogWarning($"[Megamind] audio event {a.seq}: cannot decode format '{a.format}'");
                return;
            }
            var clip = AudioClip.Create("answer-" + a.seq, samples.Length / channels, channels, rate, false);
            clip.SetData(samples, 0);
            c.Clip = clip;
        }

        private static AudioType AudioTypeOf(string format, string url)
        {
            string f = (format ?? "").Trim().ToLowerInvariant();
            if (f.Length == 0)
            {
                string u = (url ?? "").ToLowerInvariant();
                int q = u.IndexOf('?');
                if (q >= 0) u = u.Substring(0, q);
                f = u.EndsWith(".mp3") ? "mp3" : u.EndsWith(".ogg") ? "ogg" : "wav";
            }
            switch (f)
            {
                case "mp3": case "mpeg": return AudioType.MPEG;
                case "ogg": case "vorbis": return AudioType.OGGVORBIS;
                default: return AudioType.WAV;
            }
        }

        private IEnumerator FetchClip(SpeechClip c, ChatStreamAudio a, SpeechQueue sp)
        {
            using (var req = UnityWebRequestMultimedia.GetAudioClip(config.Url(a.url), AudioTypeOf(a.format, a.url)))
            {
                config.Authorize(req);
                req.timeout = 20;
                yield return req.SendWebRequest();
                AudioClip clip = null;
                if (req.result == UnityWebRequest.Result.Success)
                {
                    try { clip = DownloadHandlerAudioClip.GetContent(req); }
                    catch (Exception e) { Debug.LogWarning("[Megamind] audio clip " + a.seq + ": " + e.Message); }
                }
                else Debug.LogWarning("[Megamind] audio clip " + a.seq + ": " + req.error);
                if (clip != null && (c.Dropped || sp.Cancelled))
                {
                    Destroy(clip);
                    clip = null;
                }
                c.Clip = clip;
                c.Done = true;
            }
        }

        private IEnumerator PlaySpeech(SpeechQueue sp)
        {
            float nextAt = 0f; // earliest time for the next clip (isPlaying can lag a Play() call)
            while (!sp.Cancelled && speaker != null)
            {
                if (sp.Released && !speaker.isPlaying && Time.unscaledTime >= nextAt)
                {
                    if (sp.Pending.Count > 0)
                    {
                        var c = sp.Pending[0];
                        if (c.Done)
                        {
                            sp.Pending.RemoveAt(0);
                            if (c.Clip == null) continue; // failed clip: skip it
                            var clip = c.Clip;
                            c.Clip = null;
                            PlayClip(clip);
                            nextAt = Time.unscaledTime + 0.9f * clip.length;
                            if (sp.Played++ == 0 && LastFirstAudioSeconds <= 0f)
                            {
                                LastFirstAudioSeconds = Time.unscaledTime - sp.T0;
                                Debug.Log($"[Megamind] chat stream: first audio {LastFirstAudioSeconds:0.00}s");
                            }
                        }
                    }
                    else if (sp.Ended)
                    {
                        if (sp.Played == 0 && !string.IsNullOrEmpty(sp.FallbackUrl)) StartCoroutine(PlayAudio(sp.FallbackUrl));
                        break;
                    }
                }
                yield return null;
            }
            if (_speech == sp) _speech = null;
        }

        private void PlayClip(AudioClip clip)
        {
            if (speaker == null) { Destroy(clip); return; }
            var prev = _ownedClip;
            speaker.Stop();
            speaker.clip = clip;
            _ownedClip = clip;
            speaker.Play();
            if (prev != null && prev != clip) Destroy(prev);
        }

        public void NewSession()
        {
            if (sessions != null) sessions.NewSession();
            else _sessionId = Guid.NewGuid().ToString("N");
        }

        private IEnumerator PlayAudio(string url)
        {
            int gen = _speechGen;
            using (var req = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.WAV))
            {
                config.Authorize(req);
                req.timeout = 20;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success || speaker == null) yield break;
                var clip = DownloadHandlerAudioClip.GetContent(req);
                if (gen != _speechGen) // barge-in or a newer question while downloading
                {
                    if (clip != null) Destroy(clip);
                    yield break;
                }
                PlayClip(clip);
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
