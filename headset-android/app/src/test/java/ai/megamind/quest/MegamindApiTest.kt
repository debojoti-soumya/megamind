package ai.megamind.quest

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
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

  @Test
  fun frameMetaMatchesContract() {
    assertEquals("{\"device_id\":\"quest3-native\",\"ts\":1762000000.250}", MegamindApi.frameMetaJson("quest3-native", t0))
  }

  @Test
  fun audioMetaMatchesContract() {
    assertEquals("{\"device_id\":\"quest3-native\",\"ts_start\":1762000000.250}", MegamindApi.audioMetaJson("quest3-native", t0))
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

  // ---- live: replay the exact bytes against a running server ----

  /** 640x480 checkerboard JPEG (test resource): bright and sharp, so the keyframe filter keeps it. */
  private fun testJpeg(): ByteArray = javaClass.classLoader!!.getResourceAsStream("frame.jpg")!!.use { it.readBytes() }

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
    val r = MegamindApi.post(base, MegamindApi.FRAMES_PATH, MegamindApi.frameRequest("quest3-native-test", ts, MegamindApi.Body.Bytes(testJpeg()), "000000.jpg"))
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
    val r = MegamindApi.post(base, MegamindApi.AUDIO_PATH, MegamindApi.audioRequest("quest3-native-test", ts, MegamindApi.Body.FileBody(pcm), 16000, "chunk.wav"))
    println("POST /v1/audio -> ${r.code} ${r.body}")
    assertEquals(200, r.code)
    assertTrue(r.body.contains("\"audio_id\""))
    assertTrue(r.body.contains("\"duration_s\":5.0"))
  }
}
