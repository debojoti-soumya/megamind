// C# mirrors of docs/api.md. Keep in sync with server/megamind/models.py.
// Note: JsonUtility cannot deserialize `tool_calls[].args` (a free-form object); only name/confidence are read.
using System;
using System.Collections.Generic;

namespace Megamind
{
    [Serializable] public class Pose { public float x, y, z, yaw; }

    [Serializable] public class FrameMeta { public string device_id; public double ts; public Pose pose; }

    [Serializable] public class FrameAccepted { public string frame_id; public bool accepted; public string reason; }

    // POST /v1/audio (ambient speech). ts_start = unix seconds of the chunk's first sample.
    [Serializable] public class AudioMeta { public string device_id; public double ts_start; public Pose pose; }

    [Serializable] public class AudioAccepted { public string audio_id; public double duration_s; public string transcript; public List<string> token_ids = new List<string>(); }

    // thumb_url is "" for tokens without an image (kind="transcript", heard speech): skip the thumbnail.
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

    [Serializable] public class HealthModels { public string claude, reader, embed; }

    [Serializable] public class HealthResponse { public string status; public int store_count; public HealthModels models; }

    [Serializable] public class PurgeResponse { public int deleted; }
}
