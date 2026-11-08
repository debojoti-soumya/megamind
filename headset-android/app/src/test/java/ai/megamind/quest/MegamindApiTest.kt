package ai.megamind.quest

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import kotlin.concurrent.thread
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Host-JVM tests for the exact request bodies the app sends. With `-Pmegamind.url=http://127.0.0.1:8300`
 * (or MEGAMIND_URL) the `live*` tests also POST them to a running Megamind server.
 */
class MegamindApiTest {
  private val t0 = 1_762_000_000.25
  private val liveUrl: String = System.getProperty("megamind.url").orEmpty()
  private val liveToken: String = System.getProperty("megamind.token").orEmpty()

  @Test
  fun frameMetaMatchesContract() {
    assertEquals("{\"device_id\":\"quest3-native\",\"ts\":1762000000.250}", MegamindApi.frameMetaJson("quest3-native", t0))
  }

  @Test
  fun audioMetaMatchesContract() {
    assertEquals("{\"device_id\":\"quest3-native\",\"ts_start\":1762000000.250}", MegamindApi.audioMetaJson("quest3-native", t0))
  }

  @Test
  fun videoMetaMatchesContract() {
    assertEquals("{\"device_id\":\"quest3-native\",\"ts_start\":1762000000.250}", MegamindApi.videoMetaJson("quest3-native", t0))
  }

  @Test
  fun videoMultipartLayout() {
    val mp4 = testClip()
    val mp = MegamindApi.videoRequest("dev", t0, MegamindApi.Body.Bytes(mp4), "20261107_120000_000.mp4")
    val text = String(mp.toByteArray(), Charsets.ISO_8859_1)
    val b = mp.boundary
    assertTrue(text.startsWith("--$b\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n{\"device_id\":\"dev\",\"ts_start\":1762000000.250}\r\n"))
    assertTrue(text.contains("--$b\r\nContent-Disposition: form-data; name=\"video\"; filename=\"20261107_120000_000.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"))
    assertTrue(text.endsWith(String(mp4, Charsets.ISO_8859_1) + "\r\n--$b--\r\n"))
    assertEquals(MegamindApi.VIDEO_PATH, "/v1/video")
  }

  @Test
  fun jsonStringEscapes() {
    assertEquals("\"a\\\"b\\\\c\\n\"", MegamindApi.jsonString("a\"b\\c\n"))
  }

  @Test
  fun wavHeaderIsPcm16Mono() {
    val h = ByteBuffer.wrap(MegamindApi.wavHeader(32000, 16000)).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals(44, h.capacity())
    assertEquals("RIFF", String(h.array(), 0, 4, Charsets.US_ASCII))
    assertEquals(36 + 32000, h.getInt(4))
    assertEquals("WAVE", String(h.array(), 8, 4, Charsets.US_ASCII))
    assertEquals(1, h.getShort(20).toInt()) // PCM
    assertEquals(1, h.getShort(22).toInt()) // mono
    assertEquals(16000, h.getInt(24))
    assertEquals(32000, h.getInt(28)) // byte rate
    assertEquals(16, h.getShort(34).toInt())
    assertEquals(32000, h.getInt(40))
  }

  @Test
  fun frameMultipartLayout() {
    val jpeg = byteArrayOf(-1, -40, 1, 2, 3, -1, -39)
    val mp = MegamindApi.frameRequest("dev", t0, MegamindApi.Body.Bytes(jpeg), "000001.jpg")
    val text = String(mp.toByteArray(), Charsets.ISO_8859_1)
    val b = mp.boundary
    assertTrue(mp.contentType == "multipart/form-data; boundary=$b")
    assertTrue(text.startsWith("--$b\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n{\"device_id\":\"dev\",\"ts\":1762000000.250}\r\n"))
    assertTrue(text.contains("--$b\r\nContent-Disposition: form-data; name=\"image\"; filename=\"000001.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"))
    assertTrue(text.contains(String(jpeg, Charsets.ISO_8859_1) + "\r\n--$b--\r\n"))
    assertTrue(text.endsWith("--$b--\r\n"))
  }

  @Test
  fun audioMultipartWrapsPcmInWav() {
    val pcm = ByteArray(3200) { (it % 7).toByte() }
    val mp = MegamindApi.audioRequest("dev", t0, MegamindApi.Body.Bytes(pcm), 16000, "c.wav")
    val audio = mp.parts.single { it.name == "audio" }
    assertEquals("audio/wav", audio.contentType)
    assertEquals("c.wav", audio.filename)
    assertEquals(44L + pcm.size, audio.body.length)
    val bytes = ByteArrayOutputStream().also { audio.body.writeTo(it) }.toByteArray()
    assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
    assertTrue(bytes.copyOfRange(44, bytes.size).contentEquals(pcm))
    val meta = mp.parts.single { it.name == "meta" }
    assertEquals(null, meta.filename)
  }

  @Test
  fun fileBodyStreamsFile() {
    val f = File.createTempFile("megamind", ".pcm16").apply { writeBytes(ByteArray(10) { it.toByte() }); deleteOnExit() }
    val body = MegamindApi.Body.Wav(MegamindApi.Body.FileBody(f), 16000)
    assertEquals(54L, body.length)
  }

  @Test
  fun normalizeBaseUrl() {
    assertEquals("http://192.168.1.20:8000", MegamindApi.normalizeBaseUrl(" 192.168.1.20:8000/ "))
    assertEquals("https://x.example", MegamindApi.normalizeBaseUrl("https://x.example/"))
    assertEquals(null, MegamindApi.normalizeBaseUrl(""))
    assertEquals("http://192.168.1.20:8000", MegamindApi.normalizeBaseUrl("192.168.1.20:8000/v1/"))
    assertEquals(null, MegamindApi.normalizeBaseUrl("/v1"))
  }

  @Test
  fun authorizationHeader() {
    assertNull(MegamindApi.authorization(""))
    assertNull(MegamindApi.authorization("  "))
    assertNull(MegamindApi.authorization(null))
    assertEquals("Bearer s3cret", MegamindApi.authorization(" s3cret "))
  }

  @Test
  fun authFailuresAreNotRetryable() {
    assertTrue(MegamindApi.isAuthFailure(401))
    assertTrue(MegamindApi.isAuthFailure(403))
    for (code in listOf(200, 202, 400, 404, 413, 429, 500, 502, 503)) assertFalse("$code", MegamindApi.isAuthFailure(code))
    assertTrue(MegamindApi.authFailureMessage(MegamindApi.FRAMES_PATH, 401).contains("API token"))
  }

  /**
   * In-process stand-in for a server with API_TOKEN=tok (plain ServerSocket: android.jar has no
   * com.sun.net.httpserver): records each request's Authorization header, 202 for "Bearer tok", else 401.
   */
  private fun <T> withTokenServer(block: (base: String, seen: MutableList<String?>) -> T): T {
    val seen = java.util.Collections.synchronizedList(mutableListOf<String?>())
    val server = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
    val acceptor = thread(isDaemon = true) {
      while (!server.isClosed) {
        val sock = runCatching { server.accept() }.getOrNull() ?: break
        sock.use {
          val input = it.getInputStream()
          val headers = mutableMapOf<String, String>()
          while (true) {
            val line = readLine(input)
            if (line.isEmpty()) break
            val i = line.indexOf(':')
            if (i > 0) headers[line.substring(0, i).trim().lowercase()] = line.substring(i + 1).trim()
          }
          if (headers["transfer-encoding"].equals("chunked", ignoreCase = true)) {
            while (true) {
              val size = readLine(input).substringBefore(';').trim().toInt(16)
              if (size > 0) input.readNBytes(size)
              readLine(input)
              if (size == 0) break
            }
          } else {
            headers["content-length"]?.toIntOrNull()?.let { n -> input.readNBytes(n) }
          }
          val auth = headers["authorization"]
          seen.add(auth)
          val (code, body) =
              if (auth == "Bearer tok") 202 to "{\"accepted\":true}" else 401 to "{\"detail\":\"missing or invalid API token\"}"
          val bytes = body.toByteArray()
          it.getOutputStream().apply {
            write(("HTTP/1.1 $code X\r\nContent-Type: application/json\r\nContent-Length: ${bytes.size}\r\n" +
                "Connection: close\r\n\r\n").toByteArray())
            write(bytes)
            flush()
          }
        }
      }
    }
    try {
      return block("http://127.0.0.1:${server.localPort}", seen)
    } finally {
      server.close()
      acceptor.join(2_000)
    }
  }

  private fun readLine(input: InputStream): String {
    val sb = StringBuilder()
    while (true) {
      val c = input.read()
      if (c < 0 || c == '\n'.code) break
      if (c != '\r'.code) sb.append(c.toChar())
    }
    return sb.toString()
  }

  @Test
  fun postSendsBearerTokenOnEveryPath() = withTokenServer { base, seen ->
    val jpeg = MegamindApi.Body.Bytes(byteArrayOf(-1, -40, -1, -39))
    val pcm = MegamindApi.Body.Bytes(ByteArray(320))
    val requests =
        listOf(
            MegamindApi.FRAMES_PATH to MegamindApi.frameRequest("dev", t0, jpeg),
            MegamindApi.AUDIO_PATH to MegamindApi.audioRequest("dev", t0, pcm, 16000),
            MegamindApi.VIDEO_PATH to MegamindApi.videoRequest("dev", t0, MegamindApi.Body.Bytes(testClip())),
        )
    for ((path, body) in requests) assertEquals(path, 202, MegamindApi.post(base, path, body, token = "tok").code)
    assertEquals(202, MegamindApi.get(base, MegamindApi.HEALTH_PATH, token = "tok").code)
    assertEquals(listOf("Bearer tok", "Bearer tok", "Bearer tok", "Bearer tok"), seen.toList())
  }

  @Test
  fun missingOrWrongTokenIs401AuthFailure() = withTokenServer { base, seen ->
    val body = MegamindApi.frameRequest("dev", t0, MegamindApi.Body.Bytes(byteArrayOf(-1, -40, -1, -39)))
    val none = MegamindApi.post(base, MegamindApi.FRAMES_PATH, body)
    val wrong = MegamindApi.post(base, MegamindApi.FRAMES_PATH, body, token = "nope")
    assertEquals(401, none.code)
    assertTrue(MegamindApi.isAuthFailure(none.code) && MegamindApi.isAuthFailure(wrong.code))
    assertEquals(listOf(null, "Bearer nope"), seen.toList()) // no header at all when the token is empty
    assertEquals(401, MegamindApi.get(base, MegamindApi.HEALTH_PATH).code)
  }

  // ---- live: replay the exact bytes against a running server ----

  /** 640x480 checkerboard JPEG (test resource): bright and sharp, so the keyframe filter keeps it. */
  private fun testJpeg(): ByteArray = javaClass.classLoader!!.getResourceAsStream("frame.jpg")!!.use { it.readBytes() }

  /** 1 s 160x120 30 fps H.264 mp4 (test resource), shaped like a chunk's video.mp4. */
  private fun testClip(): ByteArray = javaClass.classLoader!!.getResourceAsStream("clip.mp4")!!.use { it.readBytes() }

  private fun tonePcm(seconds: Double, rate: Int = 16000): ByteArray {
    val n = (seconds * rate).toInt()
    val buf = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN)
    for (i in 0 until n) buf.putShort((3000 * Math.sin(2 * Math.PI * 440 * i / rate)).toInt().toShort())
    return buf.array()
  }

  @Test
  fun liveFramesAccepted() {
    assumeTrue("set -Pmegamind.url to run", liveUrl.isNotEmpty())
    val base = MegamindApi.normalizeBaseUrl(liveUrl)!!
    val ts = System.currentTimeMillis() / 1000.0
    val r = MegamindApi.post(base, MegamindApi.FRAMES_PATH, MegamindApi.frameRequest("quest3-native-test", ts, MegamindApi.Body.Bytes(testJpeg()), "000000.jpg"), token = liveToken)
    println("POST /v1/frames -> ${r.code} ${r.body}")
    assertEquals(202, r.code)
    assertTrue(r.body.contains("\"frame_id\""))
    assertTrue(r.body.contains("\"accepted\":true"))
  }

  @Test
  fun liveAudioAccepted() {
    assumeTrue("set -Pmegamind.url to run", liveUrl.isNotEmpty())
    val base = MegamindApi.normalizeBaseUrl(liveUrl)!!
    val ts = System.currentTimeMillis() / 1000.0
    val pcm = File.createTempFile("megamind", ".pcm16").apply { writeBytes(tonePcm(5.0)); deleteOnExit() }
    val r = MegamindApi.post(base, MegamindApi.AUDIO_PATH, MegamindApi.audioRequest("quest3-native-test", ts, MegamindApi.Body.FileBody(pcm), 16000, "chunk.wav"), token = liveToken)
    println("POST /v1/audio -> ${r.code} ${r.body}")
    assertEquals(200, r.code)
    assertTrue(r.body.contains("\"audio_id\""))
    assertTrue(r.body.contains("\"duration_s\":5.0"))
  }

  @Test
  fun liveVideoAccepted() {
    assumeTrue("set -Pmegamind.url to run", liveUrl.isNotEmpty())
    val base = MegamindApi.normalizeBaseUrl(liveUrl)!!
    val ts = System.currentTimeMillis() / 1000.0
    val r = MegamindApi.post(base, MegamindApi.VIDEO_PATH, MegamindApi.videoRequest("quest3-native-test", ts, MegamindApi.Body.Bytes(testClip()), "chunk.mp4"), token = liveToken)
    println("POST /v1/video -> ${r.code} ${r.body}")
    assertEquals(200, r.code)
    assertTrue(r.body.contains("\"video_id\""))
    assertTrue(r.body.contains("\"accepted\":true"))
  }

  @Test
  fun liveRejectsMissingToken() {
    assumeTrue("set -Pmegamind.url and -Pmegamind.token to run", liveUrl.isNotEmpty() && liveToken.isNotEmpty())
    val base = MegamindApi.normalizeBaseUrl(liveUrl)!!
    val ts = System.currentTimeMillis() / 1000.0
    val r = MegamindApi.post(base, MegamindApi.FRAMES_PATH, MegamindApi.frameRequest("quest3-native-test", ts, MegamindApi.Body.Bytes(testJpeg()), "000000.jpg"))
    println("POST /v1/frames without token -> ${r.code} ${r.body}")
    assertTrue(MegamindApi.isAuthFailure(r.code))
    assertEquals(200, MegamindApi.get(base, MegamindApi.HEALTH_PATH, token = liveToken).code)
  }
}
