// C# mirrors of docs/api.md. Keep in sync with server/megamind/models.py.
using System;
using System.Collections.Generic;

namespace Megamind
{
    [Serializable] public class Pose { public float x, y, z, yaw; }

    [Serializable] public class FrameMeta { public string device_id; public double ts; public Pose pose; public bool force; public string source; public string transcript; }

    [Serializable] public class FrameAccepted { public string frame_id; public bool accepted; public string reason; }

    [Serializable] public class Citation { public string id; public double ts; public string caption; public string thumb_url; }

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
    }
}
