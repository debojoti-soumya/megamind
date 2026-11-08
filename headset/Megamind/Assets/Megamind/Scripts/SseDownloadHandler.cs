// Stream 3: UnityWebRequest download handler for POST /v1/chat/stream. Bytes are parsed by SseParser as they arrive
// (ReceiveData), and completed events wait in a queue until ChatClient drains it from its coroutine, so all UI work
// stays on the main thread whichever thread Unity calls the handler on. Nothing is buffered beyond the current
// partial event (no `.data`/`.text`; an error status's body, e.g. {"detail": ...}, is parsed and yields no events).
using System.Collections.Generic;
using UnityEngine.Networking;

namespace Megamind
{
    public class SseDownloadHandler : DownloadHandlerScript
    {
        private readonly SseParser _parser = new SseParser();
        private readonly List<SseEvent> _ready = new List<SseEvent>();
        private readonly object _lock = new object();
        private bool _complete;

        public SseDownloadHandler() : base(new byte[16 * 1024]) { } // preallocated: no garbage per chunk

        /// <summary>True once the response body has fully arrived (CompleteContent), after the last Drain-able event.</summary>
        public bool IsComplete { get { lock (_lock) return _complete; } }

        public long BytesReceived { get { lock (_lock) return _parser.BytesFed; } }

        /// <summary>Moves the events completed so far into `into` (main thread). Returns how many.</summary>
        public int Drain(List<SseEvent> into)
        {
            lock (_lock)
            {
                int n = _ready.Count;
                if (n > 0) { into.AddRange(_ready); _ready.Clear(); }
                return n;
            }
        }

        protected override bool ReceiveData(byte[] data, int dataLength)
        {
            if (data == null || dataLength <= 0) return true;
            lock (_lock) _parser.Feed(data, dataLength, _ready);
            return true; // keep receiving
        }

        protected override void CompleteContent()
        {
            lock (_lock)
            {
                _parser.Flush(_ready);
                _complete = true;
            }
        }
    }
}
