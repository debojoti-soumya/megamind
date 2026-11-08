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

    // thumb_url is "" for tokens without an image (kind="transcript", heard speech, or a frame whose pixels aged
    // out, tier="text" on the server): skip the thumbnail. replay_url may be keyframe-rate for older memories.
    // thumb_url: /v1/thumbs/{id}.jpg, or /v1/live/{id}.jpg for live "NOW" frames (id "now_..."), or "" (speech)
    [Serializable] public class Citation { public string id; public double ts; public string caption; public string thumb_url; public string replay_url; }

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

        /// <summary>Confidence of the recall_memory call, or null if the agent did not recall.</summary>
        public string Confidence()
        {
            if (tool_calls == null) return null;
            foreach (var t in tool_calls)
                if (t.name == "recall_memory" && !string.IsNullOrEmpty(t.confidence)) return t.confidence;
            return null;
        }
    }

    // POST /v1/chat/stream (Server-Sent Events): `event: <name>` + one `data:` JSON line per event. The `final`
    // event's data is a ChatResponse (same as POST /v1/chat); the others deserialize into these.
    [Serializable] public class ChatStreamRetrieval { public string confidence; public List<Citation> candidates = new List<Citation>(); }

    [Serializable] public class ChatStreamDelta { public string text; }

    [Serializable] public class ChatStreamError { public string detail; }

    [Serializable] public class HealthModels { public string claude, reader, embed; }

    [Serializable] public class HealthResponse { public string status; public int store_count; public HealthModels models; }

    [Serializable] public class PurgeResponse { public int deleted; }

    // POST /v1/montage. time_from/time_to: unix seconds or ISO-8601 as strings; JsonUtility sends "" for null,
    // which the server reads as unbounded (and query "" as no query). max_clips 0 = server default.
    [Serializable] public class MontageRequest { public string device_id; public string query; public string time_from; public string time_to; public int max_clips; }

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
    }
}
