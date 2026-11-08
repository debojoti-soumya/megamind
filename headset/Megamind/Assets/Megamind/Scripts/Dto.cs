// C# mirrors of docs/api.md. Keep in sync with server/megamind/models.py.
// Note: JsonUtility cannot deserialize `tool_calls[].args` (a free-form object); only name/confidence are read.
using System;
using System.Collections.Generic;

namespace Megamind
{
    [Serializable] public class Pose { public float x, y, z, yaw; }

    // transcript: optional text heard around this frame; the server stores it as a kind="transcript" token and it
    // never affects the keep decision. Leave null/empty (JsonUtility sends ""), the server treats "" as none.
    [Serializable] public class FrameMeta { public string device_id; public double ts; public Pose pose; public string transcript; }

    [Serializable] public class FrameAccepted { public string frame_id; public bool accepted; public string reason; }

    // POST /v1/audio (ambient speech). ts_start = unix seconds of the chunk's first sample.
    [Serializable] public class AudioMeta { public string device_id; public double ts_start; public Pose pose; }

    [Serializable] public class AudioAccepted { public string audio_id; public double duration_s; public string transcript; public List<string> token_ids = new List<string>(); }

    // POST /v1/video (optional real H.264 clip for replay/montage; the native Quest app sends it, Unity does not).
    // ts_start = unix seconds of the clip's first frame. reason: null | "duplicate" (already stored) | "privacy" (not kept).
    [Serializable] public class VideoMeta { public string device_id; public double ts_start; }

    [Serializable] public class VideoAccepted { public string video_id; public bool accepted; public string reason; public double ts_start; public double duration_s; public int width; public int height; }

    // thumb_url is "" for tokens without an image (kind="transcript", heard speech, or a frame whose pixels aged
    // out, tier="text" on the server): skip the thumbnail. replay_url may be keyframe-rate for older memories.
    // thumb_url: /v1/thumbs/{id}.jpg, or /v1/live/{id}.jpg for live "NOW" frames (id "now_..."), or "" (speech)
    [Serializable] public class Citation { public string id; public double ts; public string caption; public string thumb_url; public string replay_url; }

    [Serializable] public class CollageImage { public string token_id; public double ts; public string title; public string thumb_url; }

    /// <summary>Photo montage (POST /v1/collage, or ChatResponse.collage): one JPEG grid at `url`.</summary>
    [Serializable]
    public class CollageResponse
    {
        public string id;
        public string url;   // /v1/collages/{id}.jpg; JsonUtility leaves this "" when the chat turn made no collage
        public string title;
        public string narration;
        public string subject;
        public bool verified;
        public int candidates;
        public int matches;
        public int width;
        public int height;
        public List<CollageImage> images = new List<CollageImage>();
    }

    /// <summary>3D model of an object from one photo (POST /v1/model3d, or ChatResponse.model3d).</summary>
    [Serializable]
    public class Model3DResponse
    {
        public string id;
        public string url;         // /v1/models3d/{id}.glb|.ply; "" when the chat turn made no model
        public string viewer_url;  // /ui/3d?m={id}&f=...: WebXR viewer for Quest Browser
        public string format;      // glb | ply
        public string kind;        // mesh | splat | points
        public bool full_3d;       // false: a relief of the side the camera saw
        public string status;      // building (multi-view, a few minutes; the viewer waits) | ready | failed
        public int views;          // photos a multi-view model is made from
        public string title;
        public string narration;
        public string subject;
        public string token_id;
        public double ts;
        public string thumb_url;
    }

    [Serializable] public class ToolCallRecord { public string name; public string confidence; }

    [Serializable] public class ChatRequest { public string session_id; public string device_id; public string text; }

    [Serializable]
    public class ChatResponse
    {
        public string transcript;
        public string answer;
        public string audio_url;
        public List<Citation> cited = new List<Citation>();
        public List<ToolCallRecord> tool_calls = new List<ToolCallRecord>();
        public string device_id_used;  // device whose memories were searched; null/"" = all devices
        public string warning;         // e.g. "device 'x' has no memories; active devices: quest3-1"; null/"" = none
        public List<Marker> markers = new List<Marker>(); // "where is X": X's box in a cited frame (ObjectMarkers)
        public CollageResponse collage; // photo montage the turn made (make_collage); null or url "" = none

        public Model3DResponse model3d; // 3D model the turn made (make_3d_model); null or url "" = none

        public bool HasCollage() => collage != null && !string.IsNullOrEmpty(collage.url);
        public bool HasModel3D() => model3d != null && !string.IsNullOrEmpty(model3d.viewer_url);

        /// <summary>Confidence of the recall_memory call, or null if the agent did not recall.</summary>
        public string Confidence()
        {
            if (tool_calls == null) return null;
            foreach (var t in tool_calls)
                if (t.name == "recall_memory" && !string.IsNullOrEmpty(t.confidence)) return t.confidence;
            return null;
        }
    }

    // ChatResponse.markers: where the asked-about object is in cited frame token_id, captured by device_id at ts
    // (FrameMeta.ts). x/y = box centre and box = [x_min, y_min, x_max, y_max], all normalized, origin top-left.
    [Serializable] public class Marker { public string label; public string token_id; public string device_id; public double ts; public float x; public float y; public List<float> box = new List<float>(); public float confidence; }

    // POST /v1/chat/stream (Server-Sent Events): `event: <name>` + one `data:` JSON line per event. The `final`
    // event's data is a ChatResponse (same as POST /v1/chat); the others deserialize into these.
    [Serializable] public class ChatStreamRetrieval { public string confidence; public List<Citation> candidates = new List<Citation>(); }

    [Serializable] public class ChatStreamDelta { public string text; }

    [Serializable] public class ChatStreamError { public string detail; }

    // `transcript`: what STT heard (voice) or the typed text, before retrieval/Claude; ChatClient matches voice
    // commands on it and cancels the turn. Servers without it: the transcript arrives only in `final`.
    [Serializable] public class ChatStreamTranscript { public string text; }

    // `audio`: one spoken sentence of the answer, in `seq` order (0, 1, ...). Either `url` (a short clip on the server,
    // e.g. /v1/audio/{id}.wav; `format` "wav" | "mp3") or `b64` (inline bytes; `format` "wav" or "pcm16" = raw
    // little-endian PCM16 at `sample_rate`, default 24000, mono). `text` = the sentence. When a turn sent any `audio`
    // event the client does not also play `final.audio_url`. A `reset` drops clips that have not played yet.
    [Serializable]
    public class ChatStreamAudio
    {
        public int seq;
        public string url;
        public string b64;
        public string text;
        public string format;
        public int sample_rate;
    }

    // POST /v1/stt: transcript only (voice commands never reach Claude/TTS). command: null/empty when no hint matched.
    [Serializable] public class SttCommand { public string phrase; public float score; }
    [Serializable] public class SttResponse { public string transcript; public float duration_s; public string model; public SttCommand command; }

    [Serializable] public class HealthModels { public string claude, reader, embed; }

    // warmup: "off" | "running" | "done" | "done (failed: <steps>)" (server startup warmup; informational, missing on older servers)
    [Serializable] public class HealthResponse { public string status; public int store_count; public HealthModels models; public string warmup; }

    [Serializable] public class PurgeResponse { public int deleted; }

    // POST /v1/montage. time_from/time_to: unix seconds or ISO-8601 as strings; JsonUtility sends "" for null,
    // which the server reads as unbounded (and query "" as no query). max_clips 0 = server default.
    // mode: "" = auto (server decides), "smart" = content-filtered (every clip shows the query's subject), "classic".
    [Serializable] public class MontageRequest { public string device_id; public string query; public string time_from; public string time_to; public int max_clips; public string mode; }

    // source: "video" (real headset video) | "replay" (buffered frames) | "timelapse" (stored keyframes)
    [Serializable] public class MontageClip { public string token_id; public double ts; public string title; public double duration_s; public string source; public string thumb_url; }

    [Serializable]
    public class MontageResponse
    {
        public string id;
        public string url;
        public string title;
        public string narration;
        public string planner;
        public List<MontageClip> clips = new List<MontageClip>();
        // content-filtered ("smart") montages; classic: "classic", "", 0, 0
        public string mode;
        public string subject;
        public int candidates;
        public int matches;
    }
}
