package ai.megamind.quest

import android.content.Context
import android.util.Log
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/**
 * On-disk layout (app-private storage):
 * ```
 * files/chunks/<id>.partial/   chunk being recorded: start.txt, frames.tsv, 000000.jpg..., audio.pcm16
 * files/chunks/<id>/           closed chunk waiting for upload: meta.json, *.jpg, audio.pcm16, sent.txt (parts already uploaded)
 * files/chunks_failed/<id>/    rejected by the backend (HTTP 400/413/415/422); can be re-queued
 * ```
 * A `.partial` chunk is recoverable after a crash: start.txt + frames.tsv are enough to rebuild
 * meta.json.
 */
object ChunkStore {
  private const val TAG = "MegamindQuest:Chunks"
  const val PARTIAL_SUFFIX = ".partial"
  const val AUDIO_NAME = "audio.pcm16"
  const val DEVICE_NAME = "quest3-native"

  fun root(context: Context): File = File(context.filesDir, "chunks").apply { mkdirs() }

  fun failedRoot(context: Context): File =
      File(context.filesDir, "chunks_failed").apply { mkdirs() }

  fun newId(wallMs: Long): String =
      SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date(wallMs))

  /** Closed chunks waiting for upload, oldest first. */
  fun readyChunks(context: Context): List<File> =
      root(context)
          .listFiles { f ->
            f.isDirectory && !f.name.endsWith(PARTIAL_SUFFIX) && File(f, "meta.json").exists()
          }
          ?.sortedBy { it.name }
          .orEmpty()

  fun failedChunks(context: Context): List<File> =
      failedRoot(context).listFiles { f -> f.isDirectory }?.sortedBy { it.name }.orEmpty()

  /** Finalize `.partial` chunks left behind by a crash. Only call while no capture is running. */
  @Synchronized
  fun recoverStale(context: Context) {
    root(context)
        .listFiles { f -> f.isDirectory && f.name.endsWith(PARTIAL_SUFFIX) }
        ?.forEach { dir ->
          Log.i(TAG, "Recovering stale chunk ${dir.name}")
          finalize(dir)
        }
  }

  /** Upper bound for queued + failed chunks on disk (~8 h of 1 fps + audio backlog). */
  const val MAX_QUEUE_BYTES = 3L * 1024 * 1024 * 1024

  private fun dirBytes(dir: File): Long = dir.listFiles()?.sumOf { it.length() } ?: 0L

  /**
   * Keeps the offline backlog bounded: while queued + failed chunks exceed [maxBytes], deletes the
   * oldest (failed chunks first). Returns how many chunks were dropped.
   */
  fun enforceCap(context: Context, maxBytes: Long = MAX_QUEUE_BYTES): Int {
    val failed = failedChunks(context).map { it to dirBytes(it) }
    val ready = readyChunks(context).map { it to dirBytes(it) }
    var total = failed.sumOf { it.second } + ready.sumOf { it.second }
    var dropped = 0
    for ((dir, bytes) in failed + ready) {
      if (total <= maxBytes) break
      if (dir.deleteRecursively()) {
        total -= bytes
        dropped++
      }
    }
    if (dropped > 0) Log.w(TAG, "Upload backlog over ${maxBytes shr 20} MB: dropped $dropped oldest chunks")
    return dropped
  }

  fun moveToFailed(context: Context, dir: File) {
    val target = File(failedRoot(context), dir.name)
    if (!dir.renameTo(target)) dir.deleteRecursively()
  }

  fun requeueFailed(context: Context): Int {
    var n = 0
    failedChunks(context).forEach { dir ->
      if (dir.renameTo(File(root(context), dir.name))) n++
    }
    return n
  }

  /**
   * Turns a `.partial` directory into an upload-ready chunk (writes meta.json, drops the suffix).
   * Returns false (and deletes the directory) if it holds neither frames nor at least ~1 s of audio.
   * Audio-only chunks (camera unavailable or paused) are kept: the backend indexes their transcript.
   */
  fun finalize(partial: File): Boolean {
    try {
      val startLines = File(partial, "start.txt").takeIf { it.exists() }?.readLines().orEmpty()
      val startMs = startLines.getOrNull(0)?.trim()?.toLongOrNull()
      val sampleRate = startLines.getOrNull(1)?.trim()?.toIntOrNull() ?: 16000
      val frames = JSONArray()
      File(partial, "frames.tsv").takeIf { it.exists() }?.forEachLine { line ->
        val parts = line.split('\t')
        if (parts.size == 2 && File(partial, parts[0]).exists()) {
          val t = parts[1].toDoubleOrNull() ?: return@forEachLine
          frames.put(JSONObject().put("file", parts[0]).put("t", t))
        }
      }
      val audio = File(partial, AUDIO_NAME)
      val hasAudio = audio.exists() && audio.length() >= sampleRate * 2L // >= ~1 s of PCM16 mono
      if (startMs == null || (frames.length() == 0 && !hasAudio)) {
        partial.deleteRecursively()
        return false
      }
      if (audio.exists() && !hasAudio && frames.length() > 0) audio.delete()
      val meta =
          JSONObject()
              .put("start_time", isoWithOffset(startMs))
              .put("start_ms", startMs) // unix ms; the uploader derives /v1/frames ts and /v1/audio ts_start from it
              .put("frames", frames)
              .put("audio_sample_rate", sampleRate)
              .put("device", DEVICE_NAME)
      File(partial, "meta.json").writeText(meta.toString())
      File(partial, "start.txt").delete()
      File(partial, "frames.tsv").delete()
      val ready = File(partial.parentFile, partial.name.removeSuffix(PARTIAL_SUFFIX))
      if (!partial.renameTo(ready)) {
        Log.e(TAG, "Failed to rename ${partial.name}")
        return false
      }
      return true
    } catch (e: Exception) {
      Log.e(TAG, "Failed to finalize ${partial.name}", e)
      return false
    }
  }

  /** e.g. 2026-11-07T14:03:22.415-08:00 (Python's datetime.fromisoformat accepts this). */
  fun isoWithOffset(wallMs: Long): String =
      OffsetDateTime.ofInstant(Instant.ofEpochMilli(wallMs), ZoneId.systemDefault())
          .format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)
}

/**
 * Writes the chunk currently being recorded. Thread-safe: frames arrive on the frame thread, PCM on
 * the audio thread, open/close on the main thread.
 */
class ChunkWriter(private val root: File) {
  private val lock = Any()
  private var dir: File? = null
  private var startWallMs = 0L
  private var frameCount = 0
  private var framesIndex: BufferedWriter? = null
  private var audioOut: BufferedOutputStream? = null
  private var audioBytesInChunk = 0L

  /** Sample rate of the PCM written into chunks opened from now on. */
  @Volatile var audioSampleRate: Int = 16000

  val isOpen: Boolean
    get() = synchronized(lock) { dir != null }

  val framesInChunk: Int
    get() = synchronized(lock) { frameCount }

  fun ageMs(): Long =
      synchronized(lock) { if (dir == null) 0L else System.currentTimeMillis() - startWallMs }

  fun open() {
    synchronized(lock) {
      if (dir != null) return
      val now = System.currentTimeMillis()
      val d = File(root, ChunkStore.newId(now) + ChunkStore.PARTIAL_SUFFIX)
      d.mkdirs()
      File(d, "start.txt").writeText("$now\n$audioSampleRate\n")
      framesIndex = File(d, "frames.tsv").bufferedWriter()
      audioOut = BufferedOutputStream(FileOutputStream(File(d, ChunkStore.AUDIO_NAME)), 64 * 1024)
      startWallMs = now
      frameCount = 0
      audioBytesInChunk = 0L
      dir = d
    }
  }

  /** Adds a JPEG captured at [wallMs]. Returns false if no chunk is open. */
  fun addFrame(jpeg: ByteArray, wallMs: Long): Boolean {
    synchronized(lock) {
      val d = dir ?: return false
      val name = String.format(Locale.US, "%06d.jpg", frameCount)
      File(d, name).writeBytes(jpeg)
      val t = (wallMs - startWallMs).coerceAtLeast(0L) / 1000.0
      framesIndex?.let {
        it.write(name + "\t" + String.format(Locale.US, "%.3f", t) + "\n")
        it.flush()
      }
      frameCount++
      return true
    }
  }

  /** Appends raw little-endian PCM16 mono. Dropped while no chunk is open. */
  fun writeAudio(buffer: ByteArray, length: Int) {
    synchronized(lock) {
      val out = audioOut ?: return
      try {
        if (audioBytesInChunk == 0L) {
          // The backend assumes audio.pcm16 starts at start_time. If the first PCM of this chunk
          // arrives late (mic was muted by the system and just came back), pad with silence so
          // the transcript lines up with the frames.
          val rate = audioSampleRate
          val lateSamples =
              (System.currentTimeMillis() - startWallMs) * rate / 1000L - length / 2
          if (lateSamples > rate * 3L / 10L) { // > 300 ms
            val zeros = ByteArray(8192)
            var remaining = lateSamples * 2
            while (remaining > 0) {
              val n = minOf(remaining, zeros.size.toLong()).toInt()
              out.write(zeros, 0, n)
              remaining -= n
            }
            audioBytesInChunk += lateSamples * 2
          }
        }
        out.write(buffer, 0, length)
        audioBytesInChunk += length
      } catch (e: Exception) {
        Log.w("MegamindQuest:Chunks", "Audio write failed", e)
      }
    }
  }

  /** Closes the current chunk. Returns true if it produced an upload-ready chunk. */
  fun close(): Boolean {
    synchronized(lock) {
      val d = dir ?: return false
      runCatching { framesIndex?.close() }
      runCatching { audioOut?.close() }
      framesIndex = null
      audioOut = null
      dir = null
      frameCount = 0
      return ChunkStore.finalize(d)
    }
  }
}
