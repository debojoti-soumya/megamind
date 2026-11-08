package ai.megamind.quest

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.io.File
import org.json.JSONObject

/**
 * Real video for each chunk: a hardware H.264 encoder whose input [Surface] is a second Camera2
 * output target (zero-copy: the camera HAL writes straight into the encoder's buffers, no YUV
 * conversion on the CPU), and one MediaMuxer per chunk writing `<chunk>/video.mp4`.
 *
 * The encoder runs continuously across chunks. At each chunk boundary [beginChunk] requests a sync
 * frame; the new chunk's muxer starts on the first keyframe after that, so every video.mp4 starts
 * on an IDR. The previous chunk's muxer is finalized synchronously in [endChunk] (before the chunk
 * directory is renamed for upload). The 1-3 frames between the boundary and the keyframe are
 * dropped; `video.json` records where the first frame sits relative to the chunk's start_time
 * (`offset`, seconds). [ChunkStore.finalize] copies it into `meta.video`, and the [Uploader] sends the
 * mp4 to `POST /v1/video` with `ts_start` = chunk start + offset (the server holds the last frame over
 * the small boundary gaps when it cuts replays/montages).
 *
 * Audio stays in audio.pcm16 (uploaded to `/v1/audio`), so the mp4 is video-only.
 *
 * Ported from Recall's Quest app (capture-app/quest/VideoRecorder.kt) unchanged apart from names.
 *
 * Codec PTS = camera sensor timestamp; [PassthroughCamera] publishes the sensor->wall-clock offset
 * ([clockOffsetMs], measured the same way as the JPEG timestamps) so offsets line up with frames.
 */
class VideoRecorder {
  companion object {
    private const val TAG = "MegamindQuest:Video"
    const val FILE_NAME = "video.mp4"
    const val INFO_NAME = "video.json"
    private const val MIME = MediaFormat.MIMETYPE_VIDEO_AVC
    const val BITRATE = 3_000_000
    private const val I_FRAME_INTERVAL_S = 1
    /** A chunk muxer that somehow outlives rotation is cut here (bounds disk per chunk). */
    private const val MAX_SEGMENT_US = 30_000_000L
  }

  private class Segment(val dir: File, val startWallMs: Long) {
    var muxer: MediaMuxer? = null
    var track = -1
    var firstPtsUs = -1L
    var lastPtsUs = -1L
    var frames = 0
    var bytes = 0L
    var firstWallMs = 0L
    var done = false
  }

  private val lock = Any()
  private val thread = HandlerThread("megamind-video").apply { start() }
  private val handler = Handler(thread.looper)
  private var codec: MediaCodec? = null
  private var surface: Surface? = null
  private var outputFormat: MediaFormat? = null
  private var fps = 30
  private var width = 0
  private var height = 0
  /** Chunk waiting for its first keyframe. */
  private var pending: Segment? = null
  /** Chunk whose muxer is receiving samples. */
  private var active: Segment? = null

  /** wallMs - sensorTimestampMs, updated by the camera for every frame. */
  @Volatile var clockOffsetMs: Long = Long.MIN_VALUE
  /** Set when the encoder hit an error; the camera then runs without it (JPEG-only fallback). */
  @Volatile var failed = false
    private set
  @Volatile var onFailure: (() -> Unit)? = null

  /** Creates and starts the encoder; returns its input surface, or null if unavailable. */
  fun startCodec(w: Int, h: Int, frameRate: Int): Surface? {
    synchronized(lock) { detachCodecLocked() }?.invoke()
    synchronized(lock) {
      if (failed) return null
      width = w
      height = h
      fps = frameRate
      return try {
        val fmt =
            MediaFormat.createVideoFormat(MIME, w, h).apply {
              setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
              setInteger(MediaFormat.KEY_BIT_RATE, BITRATE)
              setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
              setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
              setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_S)
            }
        val c = MediaCodec.createEncoderByType(MIME)
        c.setCallback(callback, handler)
        c.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val s = c.createInputSurface()
        c.start()
        codec = c
        surface = s
        Log.i(TAG, "Encoder ${c.name} started ${w}x$h @ $frameRate fps, ${BITRATE / 1000} kbps, IDR every ${I_FRAME_INTERVAL_S}s")
        s
      } catch (e: Exception) {
        Log.e(TAG, "Encoder start failed; continuing with JPEG frames only", e)
        markFailedLocked("encoder start failed: ${e.message}")
        null
      }
    }
  }

  /** Stops the encoder (camera closing). Finalizes the current chunk's partial video. */
  fun stopCodec() {
    val release =
        synchronized(lock) {
          finishActiveLocked("codec stopped")
          detachCodecLocked()
        }
    release?.invoke() // outside the lock: stop() must not wait on a callback blocked on it
  }

  fun release() {
    stopCodec()
    thread.quitSafely()
  }

  /** A new chunk opened at [startWallMs]: its video starts on the next keyframe. */
  fun beginChunk(dir: File, startWallMs: Long) {
    synchronized(lock) {
      pending = Segment(dir, startWallMs)
      requestSyncLocked()
    }
  }

  /**
   * The chunk in [dir] is closing: finalize its video.mp4 + video.json now (called before the
   * directory is renamed for upload).
   */
  fun endChunk(dir: File) {
    synchronized(lock) {
      if (pending?.dir == dir) pending = null
      if (active?.dir == dir) finishActiveLocked("chunk closed")
    }
  }

  private fun requestSyncLocked() {
    val c = codec ?: return
    try {
      c.setParameters(Bundle().apply { putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0) })
    } catch (e: Exception) {
      Log.w(TAG, "request sync frame failed: ${e.message}")
    }
  }

  private val callback =
      object : MediaCodec.Callback() {
        override fun onInputBufferAvailable(codec: MediaCodec, index: Int) {} // surface input

        override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {
          synchronized(lock) { outputFormat = format }
          Log.i(TAG, "Encoder output format: $format")
        }

        override fun onOutputBufferAvailable(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
          synchronized(lock) {
            try {
              if (codec === this@VideoRecorder.codec) onSampleLocked(codec, index, info)
            } catch (e: Exception) {
              Log.e(TAG, "Muxing failed", e)
              active?.let { abandonLocked(it) }
              active = null
            } finally {
              runCatching { codec.releaseOutputBuffer(index, false) }
            }
          }
        }

        override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
          Log.e(TAG, "Encoder error (transient=${e.isTransient}, recoverable=${e.isRecoverable})", e)
          if (e.isTransient) return
          val cb: (() -> Unit)?
          synchronized(lock) {
            if (codec !== this@VideoRecorder.codec) return
            finishActiveLocked("encoder error")
            markFailedLocked("encoder error: ${e.diagnosticInfo}")
            cb = onFailure
          }
          // The camera rebuilds its session without the encoder and calls stopCodec() from its own
          // thread (never stop/release a codec from inside its callback).
          cb?.invoke()
        }
      }

  private fun onSampleLocked(c: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) return // csd lives in outputFormat
    if (info.size <= 0) return
    val key = info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0
    Status.videoFramesEncoded++
    val p = pending
    if (key && p != null) {
      finishActiveLocked("next chunk")
      if (startSegmentLocked(p, info.presentationTimeUs)) active = p
      pending = null
    }
    val seg = active ?: return
    if (info.presentationTimeUs <= seg.lastPtsUs) return // muxer needs increasing timestamps
    if (info.presentationTimeUs - seg.firstPtsUs > MAX_SEGMENT_US) {
      finishActiveLocked("max length")
      return
    }
    val buf = c.getOutputBuffer(index) ?: return
    buf.position(info.offset)
    buf.limit(info.offset + info.size)
    val out = MediaCodec.BufferInfo()
    out.set(0, info.size, info.presentationTimeUs - seg.firstPtsUs, if (key) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0)
    seg.muxer!!.writeSampleData(seg.track, buf, out)
    seg.lastPtsUs = info.presentationTimeUs
    seg.frames++
    seg.bytes += info.size
  }

  private fun startSegmentLocked(seg: Segment, ptsUs: Long): Boolean {
    val fmt = outputFormat ?: return false
    if (!seg.dir.exists()) return false
    return try {
      val m = MediaMuxer(File(seg.dir, FILE_NAME).path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
      seg.track = m.addTrack(fmt)
      m.start()
      seg.muxer = m
      seg.firstPtsUs = ptsUs
      val off = clockOffsetMs
      seg.firstWallMs = if (off == Long.MIN_VALUE) System.currentTimeMillis() else ptsUs / 1000 + off
      true
    } catch (e: Exception) {
      Log.e(TAG, "MediaMuxer start failed for ${seg.dir.name}", e)
      abandonLocked(seg)
      false
    }
  }

  private fun finishActiveLocked(why: String) {
    val seg = active ?: return
    active = null
    if (seg.done) return
    seg.done = true
    val m = seg.muxer ?: return
    if (seg.frames == 0) {
      abandonLocked(seg)
      return
    }
    try {
      m.stop()
      m.release()
      seg.muxer = null
      val durS = (seg.lastPtsUs - seg.firstPtsUs) / 1e6 + 1.0 / fps
      val offsetS = (seg.firstWallMs - seg.startWallMs) / 1000.0
      val measuredFps = if (seg.frames > 1) (seg.frames - 1) * 1e6 / (seg.lastPtsUs - seg.firstPtsUs) else 0.0
      val info =
          JSONObject()
              .put("file", FILE_NAME)
              .put("offset", Math.round(offsetS * 1000) / 1000.0)
              .put("duration", Math.round(durS * 1000) / 1000.0)
              .put("frames", seg.frames)
              .put("fps", Math.round(measuredFps * 100) / 100.0)
              .put("width", width)
              .put("height", height)
              .put("codec", "h264")
              .put("bitrate", BITRATE)
      File(seg.dir, INFO_NAME).writeText(info.toString())
      Status.videoChunks++
      Status.videoInfo =
          String.format(java.util.Locale.US, "video: %dx%d %.1f fps, %d KB/chunk", width, height, measuredFps, seg.bytes / 1024)
      Log.i(
          TAG,
          String.format(
              java.util.Locale.US,
              "Video %s: %d frames, %.2f s, %.1f fps, %d KB (%.2f Mbps), offset %.3f s (%s)",
              seg.dir.name, seg.frames, durS, measuredFps, seg.bytes / 1024,
              seg.bytes * 8 / durS / 1e6, offsetS, why))
    } catch (e: Exception) {
      Log.e(TAG, "MediaMuxer stop failed for ${seg.dir.name}", e)
      abandonLocked(seg)
    }
  }

  private fun abandonLocked(seg: Segment) {
    seg.done = true
    runCatching { seg.muxer?.release() }
    seg.muxer = null
    File(seg.dir, FILE_NAME).delete()
    File(seg.dir, INFO_NAME).delete()
  }

  /** Detaches the codec under the lock; the returned function stops/releases it (call unlocked). */
  private fun detachCodecLocked(): (() -> Unit)? {
    val c = codec ?: return null
    val s = surface
    codec = null
    surface = null
    outputFormat = null
    return {
      runCatching { c.stop() }
      runCatching { c.release() }
      runCatching { s?.release() }
    }
  }

  private fun markFailedLocked(msg: String) {
    failed = true
    Status.videoInfo = "video: off ($msg)"
  }
}
