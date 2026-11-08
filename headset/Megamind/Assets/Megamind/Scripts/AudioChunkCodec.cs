// Stream 1: pure-C# helpers for AmbientAudioRecorder (no UnityEngine dependency, so the exact bytes are
// compile- and byte-checked outside Unity). The encoding matches the native app's MegamindApi.kt
// (headset-android/): 44-byte RIFF/WAVE header + little-endian PCM16 mono, meta = {"device_id":...,"ts_start":...}.
using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace Megamind
{
    public static class AudioChunkCodec
    {
        /// <summary>PCM16 mono WAV of samples[0..count) (floats in [-1, 1]); same layout as MegamindApi.wavHeader.</summary>
        public static byte[] EncodeWav(float[] samples, int count, int sampleRate)
        {
            int dataLen = count * 2;
            using (var ms = new MemoryStream(44 + dataLen))
            using (var w = new BinaryWriter(ms))
            {
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + dataLen);
                w.Write(Encoding.ASCII.GetBytes("WAVE")); w.Write(Encoding.ASCII.GetBytes("fmt "));
                w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(sampleRate);
                w.Write(sampleRate * 2); w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(dataLen);
                for (int i = 0; i < count; i++)
                    w.Write((short)Math.Max(short.MinValue, Math.Min(short.MaxValue, (int)Math.Round(samples[i] * 32767f))));
                w.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>`meta` part of POST /v1/audio (docs/api.md). pose is omitted when null.</summary>
        public static string MetaJson(string deviceId, double tsStart, Pose pose)
        {
            var sb = new StringBuilder(128);
            sb.Append("{\"device_id\":").Append(JsonString(deviceId))
              .Append(",\"ts_start\":").Append(tsStart.ToString("0.000", CultureInfo.InvariantCulture));
            if (pose != null)
            {
                sb.Append(",\"pose\":{\"x\":").Append(Num(pose.x)).Append(",\"y\":").Append(Num(pose.y))
                  .Append(",\"z\":").Append(Num(pose.z)).Append(",\"yaw\":").Append(Num(pose.yaw)).Append('}');
            }
            return sb.Append('}').ToString();
        }

        /// <summary>RMS of the loudest `window`-sample window, in dBFS (-120 for digital silence). A short utterance in an
        /// otherwise quiet chunk still passes a gate on this, unlike whole-chunk RMS.</summary>
        public static float LoudestWindowDb(float[] samples, int count, int window)
        {
            window = Math.Max(1, window);
            double best = 0;
            for (int start = 0; start < count; start += window)
            {
                int end = Math.Min(count, start + window);
                double sum = 0;
                for (int i = start; i < end; i++) sum += (double)samples[i] * samples[i];
                double ms = sum / Math.Max(1, end - start);
                if (ms > best) best = ms;
            }
            return best <= 1e-12 ? -120f : (float)(10.0 * Math.Log10(best));
        }

        /// <summary>yyyyMMdd_HHmmss_SSS.wav of the chunk start in local time, like the native app's chunk ids.</summary>
        public static string ChunkFileName(double tsStart) =>
            DateTimeOffset.FromUnixTimeMilliseconds((long)Math.Round(tsStart * 1000.0)).ToLocalTime()
                .ToString("yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture) + ".wav";

        private static string Num(float v) =>
            float.IsNaN(v) || float.IsInfinity(v) ? "0" : v.ToString("0.###", CultureInfo.InvariantCulture);

        private static string JsonString(string s)
        {
            var sb = new StringBuilder("\"");
            foreach (char c in s ?? "")
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4"));
                        else sb.Append(c);
                        break;
                }
            }
            return sb.Append('"').ToString();
        }
    }
}
