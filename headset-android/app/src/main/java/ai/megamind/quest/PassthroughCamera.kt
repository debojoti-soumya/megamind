package ai.megamind.quest

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executor

/**
 * Opens the Quest 3 world-facing passthrough camera via Camera2 (Passthrough Camera API) and
 * delivers one JPEG every [intervalMs] to [onFrame] (with the wall-clock capture time).
 *
 * Camera selection uses Meta's vendor metadata (see RESEARCH.md):
 *  com.meta.extra_metadata.camera_source == 0  -> passthrough RGB camera
 *  com.meta.extra_metadata.position      == 0  -> left (preferred), 1 -> right
 *
 * Real video (ported from Recall): when [video] is set and [videoWanted] returns true at open time, the
 * session gets a second output, the [VideoRecorder]'s H.264 encoder surface, and runs at [TARGET_FPS].
 * If the encoder or the 2-output session fails, the camera reopens JPEG-only (15 fps, as before).
 */
class PassthroughCamera(
    context: Context,
    private val intervalMs: () -> Long,
    private val video: VideoRecorder? = null,
    private val videoWanted: () -> Boolean = { video != null },
    private val onFrame: (jpeg: ByteArray, wallMs: Long, luma: Int) -> Unit,
) {
  companion object {
    private const val TAG = "MegamindQuest:Camera"
    private const val KEY_SOURCE = "com.meta.extra_metadata.camera_source"
    private const val KEY_POSITION = "com.meta.extra_metadata.position"
    private const val TARGET_W = 1280
    private const val TARGET_H = 960
    private const val JPEG_QUALITY = 80
    /** Full camera width: the server stores the upload as-is (small text on signs/receipts stays legible). */
    const val MAX_UPLOAD_WIDTH = 1280
    private const val REOPEN_DELAY_MS = 3_000L
    /** Camera (and video) frame rate with the encoder on. AE target ranges on Quest 3 are [15,15], [30,30], [60,60]. */
    const val TARGET_FPS = 30
    /** JPEG-only frame rate (unchanged from before video). */
    private const val JPEG_ONLY_FPS = 15
  }

  private val manager = context.getSystemService(CameraManager::class.java)
  private val thread = HandlerThread("megamind-camera").apply { start() }
  private val handler = Handler(thread.looper)
  private val executor = Executor { handler.post(it) }

  @Volatile private var wanted = false
  private var device: CameraDevice? = null
  private var session: CameraCaptureSession? = null
  private var reader: ImageReader? = null
  private var lastKeptMs = 0L
  private var openAttempts = 0
  /** Cleared after a session with the encoder surface fails, so we retry with JPEG frames only. */
  private var videoInSession = true

  init {
    video?.onFailure = {
      handler.post {
        // Encoder died: rebuild the session without its surface (JPEG-only behaviour).
        Log.w(TAG, "Encoder failed; restarting camera without video")
        if (wanted) {
          closeAll()
          open()
        }
      }
    }
  }

  fun start() {
    handler.post {
      wanted = true
      openAttempts = 0
      open()
    }
  }

  fun stop() {
    handler.post {
      wanted = false
      closeAll()
      Status.cameraInfo = "camera: closed"
    }
  }

  fun release() {
    stop()
    thread.quitSafely()
  }

  private fun readMetaInt(c: CameraCharacteristics, name: String): Int? {
    // Meta's sample reads these as Int; the HAL stores them as byte[1]. Try both.
    try {
      c.get(CameraCharacteristics.Key(name, Int::class.javaObjectType))?.let { return it }
    } catch (_: Exception) {}
    try {
      c.get(CameraCharacteristics.Key(name, Int::class.java))?.let { return it }
    } catch (_: Exception) {}
    try {
      c.get(CameraCharacteristics.Key(name, Byte::class.javaObjectType))?.let { return it.toInt() }
    } catch (_: Exception) {}
    try {
      c.get(CameraCharacteristics.Key(name, Byte::class.java))?.let { return it.toInt() }
    } catch (_: Exception) {}
    try {
      c.get(CameraCharacteristics.Key(name, ByteArray::class.java))?.firstOrNull()?.let { return it.toInt() }
    } catch (_: Exception) {}
    return null
  }

  private data class Candidate(val id: String, val source: Int?, val position: Int?, val facing: Int?)

  private fun pickCamera(): Candidate? {
    val all =
        manager.cameraIdList.map { id ->
          val c = manager.getCameraCharacteristics(id)
          Candidate(id, readMetaInt(c, KEY_SOURCE), readMetaInt(c, KEY_POSITION), c.get(CameraCharacteristics.LENS_FACING))
        }
    all.forEach { Log.i(TAG, "camera id=${it.id} source=${it.source} position=${it.position} facing=${it.facing}") }
    return all.firstOrNull { it.source == 0 && it.position == 0 }
        ?: all.firstOrNull { it.source == 0 }
        ?: all.firstOrNull { it.facing == CameraCharacteristics.LENS_FACING_BACK && it.source == null }
  }

  private fun chooseSize(c: CameraCharacteristics): Size {
    val sizes =
        c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)?.getOutputSizes(ImageFormat.YUV_420_888).orEmpty()
    sizes.firstOrNull { it.width == TARGET_W && it.height == TARGET_H }?.let { return it }
    return sizes.filter { it.width <= 1728 }.minByOrNull { kotlin.math.abs(it.width - TARGET_W) * 10 + kotlin.math.abs(it.height - TARGET_H) }
        ?: Size(TARGET_W, TARGET_H)
  }

  @SuppressLint("MissingPermission")
  private fun open() {
    if (!wanted || device != null) return
    openAttempts++
    val cand =
        try {
          pickCamera()
        } catch (e: Exception) {
          fail("Camera enumeration failed: ${e.message}")
          return
        }
    if (cand == null) {
      fail("No passthrough camera found (need Quest 3/3S, HzOS v74+, CAMERA + HEADSET_CAMERA permission)")
      return
    }
    val chars = manager.getCameraCharacteristics(cand.id)
    val size = chooseSize(chars)
    val useVideo = video != null && videoInSession && videoWanted()
    val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
    val fps =
        (if (useVideo) ranges.firstOrNull { it.lower == TARGET_FPS && it.upper == TARGET_FPS } else null)
            ?: ranges.firstOrNull { it.lower == JPEG_ONLY_FPS && it.upper == JPEG_ONLY_FPS }
            ?: ranges.minByOrNull { it.upper }
    val encoderSurface = if (useVideo) video?.startCodec(size.width, size.height, fps?.upper ?: JPEG_ONLY_FPS) else null
    if (video != null && !useVideo && !video.failed) Status.videoInfo = "video: off (setting)"
    val r = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 3)
    r.setOnImageAvailableListener({ onImage(it) }, handler)
    reader = r
    val side = when (cand.position) { 0 -> "left"; 1 -> "right"; else -> "?" }
    Status.cameraInfo = "camera: opening id=${cand.id} ($side, source=${cand.source}) ${size.width}x${size.height}"
    Log.i(TAG, "Opening camera ${cand.id} position=${cand.position} at ${size.width}x${size.height}")
    try {
      manager.openCamera(
          cand.id,
          object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
              if (!wanted) {
                cam.close()
                return
              }
              device = cam
              Log.i(TAG, "Camera ${cam.id} opened")
              createSession(cam, r, encoderSurface, fps, "id=${cand.id} ($side) ${size.width}x${size.height}")
            }

            override fun onDisconnected(cam: CameraDevice) {
              Log.w(TAG, "Camera ${cam.id} disconnected")
              Status.error("Camera ${cam.id} disconnected (another app or system took it)")
              closeAll()
              scheduleReopen()
            }

            override fun onError(cam: CameraDevice, error: Int) {
              val msg =
                  when (error) {
                    ERROR_CAMERA_IN_USE -> "in use"
                    ERROR_MAX_CAMERAS_IN_USE -> "max cameras in use"
                    ERROR_CAMERA_DISABLED -> "disabled by policy"
                    ERROR_CAMERA_DEVICE -> "fatal device error"
                    ERROR_CAMERA_SERVICE -> "fatal service error"
                    else -> "unknown error $error"
                  }
              Log.e(TAG, "Camera ${cam.id} error: $msg")
              Status.error("Camera ${cam.id} error: $msg")
              closeAll()
              scheduleReopen()
            }
          },
          handler,
      )
    } catch (e: Exception) {
      Log.e(TAG, "openCamera failed", e)
      fail("openCamera failed: ${e.javaClass.simpleName}: ${e.message}")
      closeAll()
      scheduleReopen()
    }
  }

  private fun createSession(cam: CameraDevice, r: ImageReader, enc: Surface?, fps: Range<Int>?, label: String) {
    val outputs = listOfNotNull(OutputConfiguration(r.surface), enc?.let { OutputConfiguration(it) })
    try {
      cam.createCaptureSession(
          SessionConfiguration(
              SessionConfiguration.SESSION_REGULAR,
              outputs,
              executor,
              object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                  if (!wanted || device == null) {
                    s.close()
                    return
                  }
                  session = s
                  val template = if (enc != null) CameraDevice.TEMPLATE_RECORD else CameraDevice.TEMPLATE_PREVIEW
                  val req = cam.createCaptureRequest(template).apply {
                    addTarget(r.surface)
                    if (enc != null) addTarget(enc)
                  }
                  if (fps != null) req.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(fps.lower, fps.upper))
                  try {
                    s.setRepeatingRequest(req.build(), null, handler)
                  } catch (e: Exception) {
                    Log.e(TAG, "setRepeatingRequest failed (video=${enc != null})", e)
                    if (enc != null) videoInSession = false // retry without the encoder surface
                    fail("Camera repeating request failed: ${e.message}")
                    closeAll()
                    scheduleReopen()
                    return
                  }
                  openAttempts = 0
                  val v = if (enc != null) " + H.264" else " (JPEG only)"
                  Status.cameraInfo = "camera: streaming $label @ ${fps ?: "default"} fps$v"
                  Log.i(TAG, "Capture session started: $label fps=$fps video=${enc != null}")
                  if (enc != null) watchVideoSession(s)
                }

                override fun onConfigureFailed(s: CameraCaptureSession) {
                  Log.e(TAG, "Session configure failed (video=${enc != null})")
                  if (enc != null) videoInSession = false // retry without the encoder surface
                  fail("Camera session configure failed")
                  closeAll()
                  scheduleReopen()
                }
              },
          ))
    } catch (e: Exception) {
      Log.e(TAG, "createCaptureSession failed", e)
      if (enc != null) videoInSession = false // retry without the encoder surface
      fail("createCaptureSession failed: ${e.message}")
      closeAll()
      scheduleReopen()
    }
  }

  /**
   * Fallback guard: if the session with the encoder surface configured but delivers no camera frames
   * (HAL can't feed both outputs), rebuild it JPEG-only instead of silently capturing nothing.
   */
  private fun watchVideoSession(s: CameraCaptureSession) {
    val framesBefore = Status.framesReceived
    val encodedBefore = Status.videoFramesEncoded
    handler.postDelayed({
      if (!wanted || session !== s) return@postDelayed
      if (Status.framesReceived == framesBefore) {
        Log.e(TAG, "No camera frames 5 s after starting with the encoder surface; retrying JPEG-only")
        videoInSession = false
        fail("No camera frames with video on; retrying without video")
        closeAll()
        open()
      } else if (Status.videoFramesEncoded == encodedBefore) {
        Log.w(TAG, "Camera frames arrive but the encoder produced nothing in 5 s")
        Status.videoInfo = "video: encoder idle (no frames from camera)"
      }
    }, 5_000L)
  }

  private fun onImage(r: ImageReader) {
    val img = try { r.acquireLatestImage() } catch (e: Exception) { null } ?: return
    try {
      Status.framesReceived++
      val now = System.currentTimeMillis()
      // Same sensor timestamp as the encoder's frame for this capture: publish the mapping so
      // video offsets use the same clock as the JPEG timestamps (delivery time).
      video?.clockOffsetMs = now - img.timestamp / 1_000_000L
      if (now - lastKeptMs < intervalMs()) return
      lastKeptMs = now
      val nv21 = toNv21(img)
      val luma = meanLuma(nv21, img.width, img.height)
      val out = ByteArrayOutputStream(256 * 1024)
      YuvImage(nv21, ImageFormat.NV21, img.width, img.height, null)
          .compressToJpeg(Rect(0, 0, img.width, img.height), JPEG_QUALITY, out)
      onFrame(downscale(out.toByteArray(), img.width, img.height), now, luma)
    } catch (e: Exception) {
      Log.e(TAG, "Frame processing failed", e)
      Status.error("Frame processing failed: ${e.message}")
    } finally {
      img.close()
    }
  }

  /** Re-encodes [jpeg] at [MAX_UPLOAD_WIDTH] wide (a no-op for the 1280x960 camera). Falls back to the original. */
  private fun downscale(jpeg: ByteArray, w: Int, h: Int): ByteArray {
    if (w <= MAX_UPLOAD_WIDTH) return jpeg
    return try {
      val src = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return jpeg
      val dst = Bitmap.createScaledBitmap(src, MAX_UPLOAD_WIDTH, h * MAX_UPLOAD_WIDTH / w, true)
      val out = ByteArrayOutputStream(160 * 1024)
      dst.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
      if (dst !== src) dst.recycle()
      src.recycle()
      out.toByteArray()
    } catch (e: Exception) {
      Log.w(TAG, "Downscale failed; keeping full-size frame", e)
      jpeg
    }
  }

  private fun meanLuma(nv21: ByteArray, w: Int, h: Int): Int {
    var sum = 0L
    var n = 0
    val step = 16
    var y = 0
    while (y < h) {
      var x = 0
      while (x < w) {
        sum += nv21[y * w + x].toInt() and 0xFF
        n++
        x += step
      }
      y += step
    }
    return if (n == 0) 0 else (sum / n).toInt()
  }

  /** YUV_420_888 (any strides) -> NV21 (Y plane, then interleaved V/U). */
  private fun toNv21(img: Image): ByteArray {
    val w = img.width
    val h = img.height
    val out = ByteArray(w * h * 3 / 2)
    val yP = img.planes[0]
    val uP = img.planes[1]
    val vP = img.planes[2]
    val yBuf = yP.buffer
    val yRow = yP.rowStride
    var pos = 0
    if (yP.pixelStride == 1 && yRow == w) {
      yBuf.position(0)
      yBuf.get(out, 0, w * h)
      pos = w * h
    } else {
      val row = ByteArray(yRow)
      for (r in 0 until h) {
        yBuf.position(r * yRow)
        val len = minOf(yRow, yBuf.remaining())
        yBuf.get(row, 0, len)
        if (yP.pixelStride == 1) {
          System.arraycopy(row, 0, out, pos, w)
        } else {
          for (c in 0 until w) out[pos + c] = row[c * yP.pixelStride]
        }
        pos += w
      }
    }
    val uBuf = uP.buffer
    val vBuf = vP.buffer
    val uvRow = uP.rowStride
    val uvPix = uP.pixelStride
    val vRow = vP.rowStride
    val vPix = vP.pixelStride
    for (r in 0 until h / 2) {
      for (c in 0 until w / 2) {
        out[pos++] = vBuf.get(r * vRow + c * vPix)
        out[pos++] = uBuf.get(r * uvRow + c * uvPix)
      }
    }
    return out
  }

  private fun fail(msg: String) {
    Status.error(msg)
    Status.cameraInfo = "camera: error"
  }

  private fun scheduleReopen() {
    if (!wanted) return
    val delay = (REOPEN_DELAY_MS * openAttempts.coerceIn(1, 10))
    Status.cameraInfo = "camera: retrying in ${delay / 1000}s"
    handler.postDelayed({ open() }, delay)
  }

  private fun closeAll() {
    runCatching { session?.close() }
    session = null
    runCatching { device?.close() }
    device = null
    runCatching { reader?.close() }
    reader = null
    video?.stopCodec()
  }
}
