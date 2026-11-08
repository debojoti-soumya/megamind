package ai.megamind.quest

import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import org.json.JSONObject

/**
 * Persistent upload queue (ported from Recall's Quest app): uploads closed chunks oldest-first to
 * Megamind (`POST /v1/frames` per frame, `POST /v1/audio` per chunk; see [MegamindApi]), deletes a
 * chunk once every part got a 2xx, retries with exponential backoff while the backend is
 * unreachable. The queue *is* the directory, so it survives restarts.
 */
object Uploader {
  private const val TAG = "MegamindQuest:Upload"
  private const val BACKOFF_INITIAL_MS = 2_000L
  private const val BACKOFF_MAX_MS = 60_000L
  private const val IDLE_POLL_MS = 30_000L
  private const val SENT_NAME = "sent.txt"
  private const val AUDIO_PART = "audio"
  private const val MAX_AUDIO_RETRIES = 3

  private val wake = LinkedBlockingQueue<Unit>()
  /** Only [retryNow] cuts a backoff short; new chunks ([kick]) don't, so backoff really backs off. */
  private val retryWake = LinkedBlockingQueue<Unit>()
  @Volatile private var backoffMs = BACKOFF_INITIAL_MS
  private var thread: Thread? = null
  private lateinit var app: Context

  @Synchronized
  fun start(context: Context) {
    if (thread != null) return
    app = context.applicationContext
    thread = Thread({ loop() }, "megamind-upload").apply { isDaemon = true; start() }
  }

  fun kick() {
    wake.offer(Unit)
  }

  fun retryNow() {
    backoffMs = BACKOFF_INITIAL_MS
    retryWake.offer(Unit)
    kick()
  }

  fun requeueFailed() {
    if (thread == null) return
    ChunkStore.requeueFailed(app)
    retryNow()
  }

  private sealed interface Outcome {
    data class Ok(val body: String) : Outcome
    data class Retry(val message: String) : Outcome
    data class Permanent(val message: String) : Outcome
  }

  private fun refreshCounts() {
    Status.chunksPending = ChunkStore.readyChunks(app).size
    Status.chunksFailed = ChunkStore.failedChunks(app).size
  }

  private fun waitFor(ms: Long) {
    wake.poll(ms, TimeUnit.MILLISECONDS)
    wake.clear()
  }

  private fun loop() {
    while (true) {
      try {
        refreshCounts()
        val next = ChunkStore.readyChunks(app).firstOrNull()
        if (next == null) {
          waitFor(IDLE_POLL_MS)
          continue
        }
        Status.uploadInfo = "uploading ${next.name}"
        when (val outcome = upload(next)) {
          is Outcome.Ok -> {
            next.deleteRecursively()
            backoffMs = BACKOFF_INITIAL_MS
            Status.chunksSent++
            Status.uploadInfo = "sent ${next.name}"
            Log.i(TAG, "Uploaded ${next.name}: ${outcome.body.take(200)}")
          }
          is Outcome.Permanent -> {
            Log.e(TAG, "Backend rejected ${next.name}: ${outcome.message}")
            ChunkStore.moveToFailed(app, next)
            Status.error("Upload rejected: ${outcome.message}")
          }
          is Outcome.Retry -> {
            val wait = backoffMs
            backoffMs = (backoffMs * 2).coerceAtMost(BACKOFF_MAX_MS)
            Log.w(TAG, "Upload of ${next.name} failed (${outcome.message}); retry in ${wait}ms")
            Status.error("${outcome.message} - retrying in ${wait / 1000}s")
            Status.uploadInfo = "waiting to retry"
            ChunkStore.enforceCap(app)
            refreshCounts()
            retryWake.poll(wait, TimeUnit.MILLISECONDS)
            retryWake.clear()
            wake.clear()
          }
        }
      } catch (t: Throwable) {
        Log.e(TAG, "Upload loop error", t)
        waitFor(5_000)
      }
    }
  }

  /** Parts of a chunk already accepted by the backend (survives restarts, so retries never re-send). */
  private fun sentParts(dir: File): MutableSet<String> =
      File(dir, SENT_NAME).takeIf { it.exists() }?.readLines()?.map { it.trim() }?.filter { it.isNotEmpty() }?.toMutableSet()
          ?: mutableSetOf()

  private fun markSent(dir: File, part: String) {
    File(dir, SENT_NAME).appendText(part + "\n")
  }

  /** Consecutive 5xx failures of a chunk's audio part (in memory; reset on restart). */
  private val audioFailures = HashMap<String, Int>()

  /**
   * Uploads one closed chunk to Megamind: each frame as its own `POST /v1/frames` (ts = chunk start +
   * frame offset), then the chunk's PCM as one WAV `POST /v1/audio` (ts_start = chunk start). Progress
   * is recorded per part in `sent.txt`, so a retry resumes where it stopped.
   */
  private fun upload(dir: File): Outcome {
    val base =
        normalizeBaseUrl(SettingsStore.get(app).backendUrl)
            ?: return Outcome.Retry("Invalid backend URL (check settings)")
    val deviceId = SettingsStore.get(app).deviceId
    return try {
      val meta = JSONObject(File(dir, "meta.json").readText())
      val startMs =
          meta.optLong("start_ms", 0L).takeIf { it > 0 }
              ?: return Outcome.Permanent("Chunk ${dir.name} has no start_ms")
      val sampleRate = meta.optInt("audio_sample_rate", 16000)
      val frames = meta.optJSONArray("frames")
      val audio = File(dir, ChunkStore.AUDIO_NAME).takeIf { it.exists() && it.length() > 0 }
      if ((frames == null || frames.length() == 0) && audio == null) {
        return Outcome.Permanent("Chunk ${dir.name} has neither frames nor audio")
      }
      val sent = sentParts(dir)
      var accepted = 0
      var dropped = 0
      var rejected = 0
      if (frames != null) {
        for (i in 0 until frames.length()) {
          val fr = frames.getJSONObject(i)
          val name = fr.getString("file")
          if (name in sent) continue
          val f = File(dir, name)
          if (!f.exists()) continue
          val ts = startMs / 1000.0 + fr.optDouble("t", 0.0)
          val r = MegamindApi.post(base, MegamindApi.FRAMES_PATH, MegamindApi.frameRequest(deviceId, ts, MegamindApi.Body.FileBody(f), name))
          when {
            r.code in 200..299 -> {
              if (r.body.contains("\"accepted\":true")) accepted++ else dropped++
              markSent(dir, name)
            }
            r.code == 400 && r.body.contains(MegamindApi.UNDECODABLE_JPEG) -> {
              // One corrupt JPEG (e.g. torn write before a crash) must not park the whole chunk,
              // and its audio, in chunks_failed: skip just this frame.
              Log.w(TAG, "Skipping undecodable frame ${dir.name}/$name")
              rejected++
              markSent(dir, name)
            }
            r.code == 400 || r.code == 413 || r.code == 415 || r.code == 422 ->
                return Outcome.Permanent("/v1/frames HTTP ${r.code}: ${r.body.take(200)}")
            else -> return Outcome.Retry("/v1/frames HTTP ${r.code} from backend")
          }
        }
      }
      var audioNote = ""
      if (audio != null && AUDIO_PART !in sent) {
        val r =
            MegamindApi.post(
                base, MegamindApi.AUDIO_PATH,
                MegamindApi.audioRequest(deviceId, startMs / 1000.0, MegamindApi.Body.FileBody(audio), sampleRate, dir.name + ".wav"),
                readTimeoutMs = 120_000) // transcription is synchronous
        when {
          r.code in 200..299 -> {
            markSent(dir, AUDIO_PART)
            audioFailures.remove(dir.name)
            audioNote = " audio: ${r.body.take(160)}"
          }
          r.code == 404 || r.code == 405 -> {
            // Server predates POST /v1/audio: keep frames flowing, skip the audio.
            markSent(dir, AUDIO_PART)
            audioNote = " audio skipped (server has no /v1/audio)"
            Status.error("Backend has no /v1/audio (HTTP ${r.code}); uploading frames only")
          }
          r.code == 400 || r.code == 413 || r.code == 415 || r.code == 422 -> {
            // A bad audio part must not block the frame queue behind it.
            markSent(dir, AUDIO_PART)
            audioNote = " audio rejected (HTTP ${r.code})"
            Status.error("Audio rejected: HTTP ${r.code}: ${r.body.take(160)}")
          }
          else -> {
            val n = (audioFailures[dir.name] ?: 0) + 1
            audioFailures[dir.name] = n
            if (n < MAX_AUDIO_RETRIES) return Outcome.Retry("/v1/audio HTTP ${r.code} from backend")
            // e.g. a persistent STT provider error (502): give up on this chunk's audio only.
            markSent(dir, AUDIO_PART)
            audioFailures.remove(dir.name)
            audioNote = " audio dropped after $n x HTTP ${r.code}"
            Status.error("Audio dropped after $n failed attempts (HTTP ${r.code})")
          }
        }
      }
      Outcome.Ok("frames accepted=$accepted filtered=$dropped" + (if (rejected > 0) " undecodable=$rejected" else "") + audioNote)
    } catch (e: IOException) {
      Outcome.Retry("Backend unreachable ($base): ${e.message ?: e.javaClass.simpleName}")
    } catch (e: Exception) {
      Log.e(TAG, "Upload error for ${dir.name}", e)
      Outcome.Retry("Upload error: ${e.message ?: e.javaClass.simpleName}")
    }
  }

  /** GET {backend}/v1/health. Blocking; call off the main thread. */
  fun testConnection(rawUrl: String): String {
    val base = normalizeBaseUrl(rawUrl) ?: return "Invalid URL: $rawUrl"
    return try {
      val c = URL(base + MegamindApi.HEALTH_PATH).openConnection() as HttpURLConnection
      c.connectTimeout = 5_000
      c.readTimeout = 5_000
      val code = c.responseCode
      c.disconnect()
      if (code in 200..299) "Connected to Megamind at $base (HTTP $code)" else "Reached $base but got HTTP $code (is this a Megamind server?)"
    } catch (e: Exception) {
      "Can't reach $base: ${e.message ?: e.javaClass.simpleName}"
    }
  }

  fun normalizeBaseUrl(raw: String): String? = MegamindApi.normalizeBaseUrl(raw)
}
