package ai.megamind.quest

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.UUID

/**
 * Request builders for Megamind's ingest contract (docs/api.md, `POST /v1/frames` and
 * `POST /v1/audio`). Pure JVM (no android.* or org.json), so the exact bytes the app sends are unit
 * tested on the host and can be replayed against a local server (see `MegamindApiTest`).
 *
 * - `/v1/frames`: multipart `image` (JPEG) + `meta` = `{"device_id": ..., "ts": <unix s>}`.
 * - `/v1/audio`: multipart `audio` (16-bit PCM mono WAV) + `meta` = `{"device_id": ..., "ts_start": <unix s>}`.
 * - `/v1/video`: multipart `video` (H.264 mp4) + `meta` = `{"device_id": ..., "ts_start": <unix s of the first frame>}`.
 *
 * `pose` is optional in both and omitted: a 2D panel app has no head pose without OpenXR.
 */
object MegamindApi {
  const val FRAMES_PATH = "/v1/frames"
  const val AUDIO_PATH = "/v1/audio"
  const val VIDEO_PATH = "/v1/video"
  const val HEALTH_PATH = "/v1/health"

  /** `detail` of the server's 400 for a corrupt JPEG (server/megamind/ingest/router.py): skip that frame only. */
  const val UNDECODABLE_JPEG = "not a decodable JPEG"

  fun frameMetaJson(deviceId: String, tsSec: Double): String =
      "{\"device_id\":${jsonString(deviceId)},\"ts\":${jsonNumber(tsSec)}}"

  fun audioMetaJson(deviceId: String, tsStartSec: Double): String =
      "{\"device_id\":${jsonString(deviceId)},\"ts_start\":${jsonNumber(tsStartSec)}}"

  fun videoMetaJson(deviceId: String, tsStartSec: Double): String =
      "{\"device_id\":${jsonString(deviceId)},\"ts_start\":${jsonNumber(tsStartSec)}}"

  /** One `/v1/video` request body: [mp4] is a chunk's video.mp4. */
  fun videoRequest(deviceId: String, tsStartSec: Double, mp4: Body, filename: String = "video.mp4"): Multipart =
      Multipart(
          listOf(
              Part("meta", null, null, Body.Bytes(videoMetaJson(deviceId, tsStartSec).toByteArray(Charsets.UTF_8))),
              Part("video", filename, "video/mp4", mp4),
          ))

  /** One `/v1/frames` request body. */
  fun frameRequest(deviceId: String, tsSec: Double, jpeg: Body, filename: String = "frame.jpg"): Multipart =
      Multipart(
          listOf(
              Part("meta", null, null, Body.Bytes(frameMetaJson(deviceId, tsSec).toByteArray(Charsets.UTF_8))),
              Part("image", filename, "image/jpeg", jpeg),
          ))

  /** One `/v1/audio` request body: [pcm] is raw little-endian PCM16 mono at [sampleRate]. */
  fun audioRequest(deviceId: String, tsStartSec: Double, pcm: Body, sampleRate: Int, filename: String = "audio.wav"): Multipart =
      Multipart(
          listOf(
              Part("meta", null, null, Body.Bytes(audioMetaJson(deviceId, tsStartSec).toByteArray(Charsets.UTF_8))),
              Part("audio", filename, "audio/wav", Body.Wav(pcm, sampleRate)),
          ))

  /** 44-byte RIFF/WAVE header for PCM16 mono. */
  fun wavHeader(dataBytes: Long, sampleRate: Int, channels: Int = 1): ByteArray {
    val byteRate = sampleRate * channels * 2
    val out = ByteArrayOutputStream(44)
    fun le32(v: Long) { for (i in 0 until 4) out.write(((v shr (8 * i)) and 0xFF).toInt()) }
    fun le16(v: Int) { out.write(v and 0xFF); out.write((v shr 8) and 0xFF) }
    out.write("RIFF".toByteArray(Charsets.US_ASCII))
    le32(36 + dataBytes)
    out.write("WAVE".toByteArray(Charsets.US_ASCII))
    out.write("fmt ".toByteArray(Charsets.US_ASCII))
    le32(16)
    le16(1) // PCM
    le16(channels)
    le32(sampleRate.toLong())
    le32(byteRate.toLong())
    le16(channels * 2) // block align
    le16(16) // bits per sample
    out.write("data".toByteArray(Charsets.US_ASCII))
    le32(dataBytes)
    return out.toByteArray()
  }

  sealed interface Body {
    val length: Long
    fun writeTo(out: OutputStream)

    class Bytes(val data: ByteArray) : Body {
      override val length: Long get() = data.size.toLong()
      override fun writeTo(out: OutputStream) = out.write(data)
    }

    class FileBody(val file: File) : Body {
      override val length: Long get() = file.length()
      override fun writeTo(out: OutputStream) { file.inputStream().use { it.copyTo(out) } }
    }

    /** Wraps raw PCM16 mono in a WAV header (streamed, never held in memory twice). */
    class Wav(val pcm: Body, val sampleRate: Int) : Body {
      override val length: Long get() = 44 + pcm.length
      override fun writeTo(out: OutputStream) {
        out.write(wavHeader(pcm.length, sampleRate))
        pcm.writeTo(out)
      }
    }
  }

  data class Part(val name: String, val filename: String?, val contentType: String?, val body: Body)

  class Multipart(val parts: List<Part>, val boundary: String = "----megamind" + UUID.randomUUID().toString().replace("-", "")) {
    val contentType: String get() = "multipart/form-data; boundary=$boundary"

    fun writeTo(out: OutputStream) {
      fun ascii(s: String) = out.write(s.toByteArray(Charsets.UTF_8))
      for (p in parts) {
        ascii("--$boundary\r\n")
        ascii("Content-Disposition: form-data; name=\"${p.name}\"" + (p.filename?.let { "; filename=\"$it\"" } ?: "") + "\r\n")
        p.contentType?.let { ascii("Content-Type: $it\r\n") }
        ascii("\r\n")
        p.body.writeTo(out)
        ascii("\r\n")
      }
      ascii("--$boundary--\r\n")
    }

    fun toByteArray(): ByteArray = ByteArrayOutputStream().also { writeTo(it) }.toByteArray()
  }

  data class Response(val code: Int, val body: String)

  /**
   * POSTs [body] to [base] + [path]. Throws IOException when the backend is unreachable.
   * `Connection: close`: pooled keep-alive sockets went stale through `adb reverse` in Recall.
   */
  @Throws(IOException::class)
  fun post(base: String, path: String, body: Multipart, readTimeoutMs: Int = 60_000): Response {
    val conn = URL(base + path).openConnection() as HttpURLConnection
    try {
      conn.connectTimeout = 10_000
      conn.readTimeout = readTimeoutMs
      conn.doOutput = true
      conn.requestMethod = "POST"
      conn.setChunkedStreamingMode(64 * 1024)
      conn.setRequestProperty("Content-Type", body.contentType)
      conn.setRequestProperty("Connection", "close")
      conn.outputStream.buffered(64 * 1024).use { body.writeTo(it) }
      val code = conn.responseCode
      val text =
          runCatching {
                (if (code in 200..299) conn.inputStream else conn.errorStream)?.bufferedReader()?.use { it.readText() }.orEmpty()
              }
              .getOrDefault("")
      return Response(code, text)
    } finally {
      conn.disconnect()
    }
  }

  /**
   * "192.168.1.20:8000/" -> "http://192.168.1.20:8000". A pasted API base ("…:8000/v1") is accepted too,
   * since the paths above already start with /v1. Null if unparseable.
   */
  fun normalizeBaseUrl(raw: String): String? {
    var url = raw.trim().trimEnd('/')
    if (url.endsWith("/v1")) url = url.removeSuffix("/v1").trimEnd('/')
    if (url.isEmpty()) return null
    if (!url.contains("://")) url = "http://$url"
    return try {
      val u = URL(url)
      if (u.host.isNullOrEmpty()) null else url
    } catch (e: Exception) {
      null
    }
  }

  private fun jsonNumber(d: Double): String {
    require(d.isFinite()) { "non-finite timestamp" }
    return String.format(Locale.US, "%.3f", d)
  }

  fun jsonString(s: String): String {
    val sb = StringBuilder("\"")
    for (c in s) {
      when {
        c == '"' -> sb.append("\\\"")
        c == '\\' -> sb.append("\\\\")
        c == '\n' -> sb.append("\\n")
        c == '\r' -> sb.append("\\r")
        c == '\t' -> sb.append("\\t")
        c < ' ' -> sb.append(String.format(Locale.US, "\\u%04x", c.code))
        else -> sb.append(c)
      }
    }
    return sb.append('"').toString()
  }
}
