// Stream 3: pure-C# decoding of the inline sentence clips of POST /v1/chat/stream `audio` events (`b64`, docs/api.md)
// into float samples for AudioClip.Create/SetData. No UnityEngine dependency, so it can be checked outside Unity.
// Supported: format "wav" (RIFF/WAVE, PCM16; a streamed header with an unknown 0xFFFFFFFF/0 data size reads to the
// end) and "pcm16" / "pcm" / "pcm_s16le" (raw little-endian PCM16, mono unless `channels` says otherwise).
using System;

namespace Megamind
{
    public static class SpeechClipCodec
    {
        public const int DefaultSampleRate = 24000; // OpenAI TTS PCM / the server's WAVs

        /// <summary>true when `format` (case-insensitive, "" = guess from the bytes) names raw PCM16.</summary>
        public static bool IsRawPcm(string format)
        {
            string f = (format ?? "").Trim().ToLowerInvariant();
            return f == "pcm" || f == "pcm16" || f == "pcm_s16le" || f == "s16le";
        }

        /// <summary>Decode `bytes` (a WAV file or raw PCM16) to interleaved floats in [-1, 1]. Returns false when the
        /// bytes are not decodable (unknown format, compressed WAV, no samples).</summary>
        public static bool TryDecode(byte[] bytes, string format, int sampleRate, int channels,
                                     out float[] samples, out int outChannels, out int outRate)
        {
            samples = null;
            outChannels = channels > 0 ? channels : 1;
            outRate = sampleRate > 0 ? sampleRate : DefaultSampleRate;
            if (bytes == null || bytes.Length < 2) return false;
            bool looksWav = bytes.Length >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F'
                            && bytes[8] == 'W' && bytes[9] == 'A' && bytes[10] == 'V' && bytes[11] == 'E';
            if (looksWav) return TryDecodeWav(bytes, out samples, out outChannels, out outRate);
            string f = (format ?? "").Trim().ToLowerInvariant();
            if (!IsRawPcm(f) && f.Length > 0) return false; // e.g. "mp3" inline: not supported here
            samples = Pcm16ToFloats(bytes, 0, bytes.Length);
            return samples.Length > 0;
        }

        public static bool TryDecodeWav(byte[] b, out float[] samples, out int channels, out int rate)
        {
            samples = null; channels = 1; rate = DefaultSampleRate;
            if (b == null || b.Length < 12) return false;
            int pos = 12, fmtTag = 0, bits = 0;
            bool haveFmt = false;
            while (pos + 8 <= b.Length)
            {
                string id = "" + (char)b[pos] + (char)b[pos + 1] + (char)b[pos + 2] + (char)b[pos + 3];
                uint size = BitConverter.ToUInt32(LittleEndian(b, pos + 4, 4), 0);
                int body = pos + 8;
                if (id == "fmt " && body + 16 <= b.Length)
                {
                    fmtTag = BitConverter.ToUInt16(LittleEndian(b, body, 2), 0);
                    channels = BitConverter.ToUInt16(LittleEndian(b, body + 2, 2), 0);
                    rate = (int)BitConverter.ToUInt32(LittleEndian(b, body + 4, 4), 0);
                    bits = BitConverter.ToUInt16(LittleEndian(b, body + 14, 2), 0);
                    haveFmt = true;
                }
                else if (id == "data")
                {
                    // fmtTag 0xFFFE = WAVE_FORMAT_EXTENSIBLE (PCM subformat assumed when 16-bit)
                    if (!haveFmt || bits != 16 || (fmtTag != 1 && fmtTag != 0xFFFE) || channels <= 0 || rate <= 0) return false;
                    long remaining = b.Length - body;
                    long len = size == 0 || size > remaining ? remaining : size; // streamed WAVs: unknown size
                    samples = Pcm16ToFloats(b, body, (int)len);
                    return samples.Length > 0;
                }
                if (size > int.MaxValue) return false;
                long next = (long)body + size + (size & 1); // chunks are word-aligned
                if (next > b.Length) return false;
                pos = (int)next;
            }
            return false;
        }

        public static float[] Pcm16ToFloats(byte[] b, int offset, int length)
        {
            int n = Math.Max(0, length) / 2;
            var s = new float[n];
            for (int i = 0; i < n; i++)
            {
                int j = offset + 2 * i;
                s[i] = (short)(b[j] | (b[j + 1] << 8)) / 32768f;
            }
            return s;
        }

        private static byte[] LittleEndian(byte[] b, int offset, int count)
        {
            var r = new byte[count];
            Array.Copy(b, offset, r, 0, count);
            if (!BitConverter.IsLittleEndian) Array.Reverse(r);
            return r;
        }
    }
}
