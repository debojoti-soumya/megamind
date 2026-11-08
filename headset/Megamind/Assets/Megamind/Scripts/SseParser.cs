// Stream 3: incremental Server-Sent Events parser for POST /v1/chat/stream (docs/api.md). Pure C# (no UnityEngine
// dependency), so it can be compiled and tested outside Unity. Feed it bytes as they arrive (any chunking, UTF-8
// sequences and CR/LF pairs may be split across chunks); it returns every event completed by a blank line.
// Rules (WHATWG SSE): lines end in LF, CRLF or CR; ":" lines are comments (keep-alives); "event" sets the name
// (default "message"); "data" lines are joined with "\n"; one leading space after the colon is dropped; "id",
// "retry" and unknown fields are ignored; an event with no data line is not dispatched.
using System.Collections.Generic;
using System.Text;

namespace Megamind
{
    public struct SseEvent
    {
        public string Event;
        public string Data;

        public SseEvent(string evt, string data) { Event = evt; Data = data; }

        public override string ToString() => Event + ": " + Data;
    }

    public sealed class SseParser
    {
        private readonly Decoder _utf8 = new UTF8Encoding(false).GetDecoder();
        private readonly StringBuilder _line = new StringBuilder(256);
        private readonly StringBuilder _data = new StringBuilder(1024);
        private string _event;
        private bool _hasData;
        private bool _lastWasCr;
        private char[] _chars = new char[4096];

        /// <summary>Total bytes fed so far (for diagnostics).</summary>
        public long BytesFed { get; private set; }

        /// <summary>Parse data[0..length) and append completed events to `output`. Returns how many were added.</summary>
        public int Feed(byte[] data, int length, List<SseEvent> output)
        {
            if (data == null || length <= 0) return 0;
            BytesFed += length;
            int max = Encoding.UTF8.GetMaxCharCount(length);
            if (_chars.Length < max) _chars = new char[max];
            int n = _utf8.GetChars(data, 0, length, _chars, 0, false);
            return FeedChars(_chars, n, output);
        }

        /// <summary>Same as Feed(byte[]) for already-decoded text.</summary>
        public int Feed(string text, List<SseEvent> output)
        {
            if (string.IsNullOrEmpty(text)) return 0;
            return FeedChars(text.ToCharArray(), text.Length, output);
        }

        /// <summary>End of stream: a last event that was not followed by a blank line is still dispatched (lenient).</summary>
        public int Flush(List<SseEvent> output)
        {
            int added = 0;
            int n = _utf8.GetChars(new byte[0], 0, 0, _chars, 0, true);
            if (n > 0) added += FeedChars(_chars, n, output);
            if (_line.Length > 0) { ProcessLine(_line.ToString()); _line.Length = 0; }
            if (Dispatch(output)) added++;
            return added;
        }

        private int FeedChars(char[] chars, int count, List<SseEvent> output)
        {
            int added = 0;
            for (int i = 0; i < count; i++)
            {
                char c = chars[i];
                if (c == '\n' && _lastWasCr) { _lastWasCr = false; continue; } // second half of CRLF
                _lastWasCr = c == '\r';
                if (c != '\n' && c != '\r') { _line.Append(c); continue; }
                if (_line.Length == 0)
                {
                    if (Dispatch(output)) added++;
                }
                else
                {
                    ProcessLine(_line.ToString());
                    _line.Length = 0;
                }
            }
            return added;
        }

        private void ProcessLine(string line)
        {
            if (line.Length == 0 || line[0] == ':') return; // comment / keep-alive
            int colon = line.IndexOf(':');
            string field = colon < 0 ? line : line.Substring(0, colon);
            string value = "";
            if (colon >= 0)
            {
                int start = colon + 1;
                if (start < line.Length && line[start] == ' ') start++;
                value = line.Substring(start);
            }
            if (field == "event") _event = value;
            else if (field == "data")
            {
                if (_hasData) _data.Append('\n');
                _data.Append(value);
                _hasData = true;
            }
        }

        private bool Dispatch(List<SseEvent> output)
        {
            bool dispatched = false;
            if (_hasData)
            {
                output?.Add(new SseEvent(string.IsNullOrEmpty(_event) ? "message" : _event, _data.ToString()));
                dispatched = true;
            }
            _event = null;
            _data.Length = 0;
            _hasData = false;
            return dispatched;
        }
    }
}
