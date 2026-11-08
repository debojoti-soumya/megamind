package ai.megamind.quest

enum class CaptureState {
  IDLE,
  RECORDING,
  PAUSED,
}

/** Process-wide capture/upload status. The UI polls it; writers are the service threads. */
object Status {
  @Volatile var state: CaptureState = CaptureState.IDLE

  /** Wall-clock ms when the current recording segment started (0 when not recording). */
  @Volatile var segmentStartMs: Long = 0L

  /** Recorded ms from earlier segments (before a pause). */
  @Volatile var accumulatedMs: Long = 0L

  @Volatile var framesKept: Int = 0
  @Volatile var framesReceived: Long = 0L
  @Volatile var chunksClosed: Int = 0
  @Volatile var chunksSent: Int = 0
  @Volatile var chunksPending: Int = 0
  @Volatile var chunksFailed: Int = 0
  @Volatile var lastJpeg: ByteArray? = null
  @Volatile var lastFrameLuma: Int = -1
  @Volatile var lastFrameWallMs: Long = 0L
  @Volatile var audioBytes: Long = 0L
  @Volatile var audioPeak: Int = 0
  /** True while the system silences the mic (Horizon mutes it when the headset is off-head). */
  @Volatile var micSilenced: Boolean? = null
  @Volatile var cameraInfo: String = "camera: not opened"
  @Volatile var audioInfo: String = "audio: off"
  @Volatile var videoInfo: String = "video: off"
  @Volatile var videoFramesEncoded: Long = 0L
  @Volatile var videoChunks: Int = 0
  /** Clips accepted by POST /v1/video. */
  @Volatile var videoSent: Int = 0
  @Volatile var uploadInfo: String = ""
  @Volatile var lastError: String? = null
  @Volatile var lastErrorAtMs: Long = 0L

  fun elapsedMs(): Long =
      accumulatedMs + (if (state == CaptureState.RECORDING && segmentStartMs > 0) System.currentTimeMillis() - segmentStartMs else 0L)

  fun error(msg: String) {
    lastError = msg
    lastErrorAtMs = System.currentTimeMillis()
  }

  fun resetCounters() {
    accumulatedMs = 0L
    segmentStartMs = 0L
    framesKept = 0
    framesReceived = 0L
    chunksClosed = 0
  }
}
