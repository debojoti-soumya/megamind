// Offline frame spool: frames captured while the server is unreachable (hotspot dead zone, Wi-Fi drop) are written to
// disk and uploaded oldest-first once it answers again, with their original FrameMeta.ts. Order matters: the server's
// keyframe filter drops a frame older than the device's newest kept one ("too_soon"), so FrameUploader queues new
// frames behind anything spooled instead of uploading them directly. Plain System.IO (no UnityEngine), so it survives
// an app restart and can be tested outside the editor.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Megamind
{
    public sealed class FrameSpool
    {
        private readonly string _dir;
        private readonly int _maxFrames;
        private readonly long _maxBytes;
        private readonly SortedDictionary<string, long> _files = new SortedDictionary<string, long>(StringComparer.Ordinal); // stem -> bytes on disk
        private long _bytes;
        private int _seq;

        public int Count => _files.Count;
        public long Bytes => _bytes;
        /// <summary>Frames dropped (oldest first) because the spool hit maxFrames or maxBytes.</summary>
        public int Dropped { get; private set; }

        public FrameSpool(string dir, int maxFrames, long maxBytes)
        {
            _dir = dir;
            _maxFrames = Math.Max(1, maxFrames);
            _maxBytes = Math.Max(1, maxBytes);
            Directory.CreateDirectory(dir);
            foreach (var jpg in Directory.GetFiles(dir, "*.jpg"))
            {
                var stem = Path.GetFileNameWithoutExtension(jpg);
                var json = JsonPath(stem);
                if (File.Exists(json)) Add(stem, new FileInfo(jpg).Length + new FileInfo(json).Length);
                else TryDelete(jpg); // torn write: the JSON is written last
            }
            foreach (var json in Directory.GetFiles(dir, "*.json"))
                if (!_files.ContainsKey(Path.GetFileNameWithoutExtension(json))) TryDelete(json);
            Trim();
        }

        /// <summary>File name that sorts by capture time: zero-padded epoch ms, then a sequence number for equal ts.</summary>
        public static string Stem(double ts, int seq) =>
            ((long)Math.Round(ts * 1000.0)).ToString("D15") + "-" + (seq % 10000).ToString("D4");

        public void Enqueue(double ts, byte[] jpeg, string metaJson)
        {
            var stem = Stem(ts, _seq++);
            File.WriteAllBytes(JpgPath(stem), jpeg);
            File.WriteAllText(JsonPath(stem), metaJson);
            Add(stem, jpeg.Length + new FileInfo(JsonPath(stem)).Length);
            Trim();
        }

        /// <summary>The oldest spooled frame; unreadable entries are dropped on the way.</summary>
        public bool TryPeek(out string stem, out byte[] jpeg, out string metaJson)
        {
            while (_files.Count > 0)
            {
                stem = _files.Keys.First();
                try
                {
                    jpeg = File.ReadAllBytes(JpgPath(stem));
                    metaJson = File.ReadAllText(JsonPath(stem));
                    return true;
                }
                catch (Exception)
                {
                    Remove(stem);
                }
            }
            stem = null;
            jpeg = null;
            metaJson = null;
            return false;
        }

        /// <summary>Drops frames captured before cutoffTs (epoch s), e.g. a spool left from yesterday's run that would
        /// otherwise land after a DELETE /v1/memories wipe. Returns how many were dropped.</summary>
        public int DropOlderThan(double cutoffTs)
        {
            var cutoff = Stem(cutoffTs, 0);
            int n = 0;
            while (_files.Count > 0 && string.CompareOrdinal(_files.Keys.First(), cutoff) < 0)
            {
                Remove(_files.Keys.First());
                n++;
            }
            Dropped += n;
            return n;
        }

        public void Remove(string stem)
        {
            if (stem == null || !_files.TryGetValue(stem, out var n)) return;
            _files.Remove(stem);
            _bytes -= n;
            TryDelete(JpgPath(stem));
            TryDelete(JsonPath(stem));
        }

        private void Add(string stem, long n)
        {
            _files[stem] = n;
            _bytes += n;
        }

        private void Trim()
        {
            while (_files.Count > _maxFrames || (_bytes > _maxBytes && _files.Count > 1))
            {
                Remove(_files.Keys.First());
                Dropped++;
            }
        }

        private string JpgPath(string stem) => Path.Combine(_dir, stem + ".jpg");
        private string JsonPath(string stem) => Path.Combine(_dir, stem + ".json");

        private static void TryDelete(string path)
        {
            try { File.Delete(path); } catch (Exception) { /* best effort */ }
        }
    }
}
