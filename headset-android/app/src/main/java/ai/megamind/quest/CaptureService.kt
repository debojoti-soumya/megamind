package ai.megamind.quest

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log

/**
 * Foreground service (camera|microphone) that owns capture: passthrough camera frames + headset
 * mic PCM (+ the [VideoRecorder]'s per-chunk H.264 when video is on) go into the current
 * [ChunkWriter]; every chunk-length seconds the chunk is closed and handed to the persistent
 * [Uploader] queue.
 */
class CaptureService : Service() {
  companion object {
    private const val TAG = "MegamindQuest:Service"
    private const val CHANNEL_ID = "megamind_capture"
    private const val NOTIF_ID = 1
    const val ACTION_START = "ai.megamind.quest.START"
    const val ACTION_PAUSE = "ai.megamind.quest.PAUSE"
    const val ACTION_STOP = "ai.megamind.quest.STOP"

    fun send(context: Context, action: String) {
      val i = Intent(context, CaptureService::class.java).setAction(action)
      if (action == ACTION_START) context.startForegroundService(i) else context.startService(i)
    }
  }

  private val main = Handler(Looper.getMainLooper())
  private lateinit var writer: ChunkWriter
  private var camera: PassthroughCamera? = null
  private val video = VideoRecorder()
  private var audio: AudioCapture? = null
  private var wakeLock: PowerManager.WakeLock? = null
  private var foreground = false

  private val tick =
      object : Runnable {
        override fun run() {
          if (Status.state == CaptureState.RECORDING) {
            if (!writer.isOpen) writer.open()
            if (writer.ageMs() >= SettingsStore.get(this@CaptureService).chunkMs) rotateChunk()
            val silenced = audio?.isSilenced()
            if (silenced != Status.micSilenced) {
              Status.micSilenced = silenced
              Log.i(TAG, "Mic silenced by system: $silenced")
            }
          }
          main.postDelayed(this, 250)
        }
      }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onCreate() {
    super.onCreate()
    writer = ChunkWriter(ChunkStore.root(this)).apply {
      audioSampleRate = AudioCapture.SAMPLE_RATE
      video = this@CaptureService.video
    }
    ChunkStore.recoverStale(this)
    Uploader.start(this)
    Uploader.kick()
    val nm = getSystemService(NotificationManager::class.java)
    nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Megamind capture", NotificationManager.IMPORTANCE_LOW))
  }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    when (intent?.action) {
      ACTION_START -> startCapture()
      null -> {
        // START_STICKY restart after the process was killed: resume only if we were recording.
        if (SettingsStore.captureWanted(this)) {
          Log.i(TAG, "Sticky restart: resuming capture")
          startCapture()
        } else {
          stopSelf()
          return START_NOT_STICKY
        }
      }
      ACTION_PAUSE -> pauseCapture()
      ACTION_STOP -> {
        stopCapture()
        return START_NOT_STICKY
      }
    }
    if (Status.state == CaptureState.IDLE) {
      // e.g. Pause pressed while idle, or startForeground was refused: don't linger as a sticky service.
      stopSelf()
      return START_NOT_STICKY
    }
    return START_STICKY
  }

  private fun notification(text: String): Notification {
    val pi =
        PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE)
    return Notification.Builder(this, CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_notification)
        .setContentTitle("Megamind capture")
        .setContentText(text)
        .setOngoing(true)
        .setContentIntent(pi)
        .build()
  }

  private fun goForeground(text: String): Boolean {
    val n = notification(text)
    val types = listOf(
        ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
        ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA,
        ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
    )
    for (t in types) {
      try {
        startForeground(NOTIF_ID, n, t)
        foreground = true
        Log.i(TAG, "startForeground ok (type=$t)")
        return true
      } catch (e: Exception) {
        Log.w(TAG, "startForeground(type=$t) failed: ${e.message}")
      }
    }
    Status.error("Could not start foreground service (missing permissions, or started from background?)")
    return false
  }

  private fun startCapture() {
    if (Status.state == CaptureState.RECORDING) {
      if (!foreground) goForeground("Recording")
      return
    }
    if (!goForeground("Recording")) {
      // Android 14 refuses camera/mic FGS without permission or from the background; capturing
      // without a foreground service would be killed or blocked, so don't start at all.
      Log.e(TAG, "Not starting capture: startForeground refused")
      return
    }
    SettingsStore.setCaptureWanted(this, true)
    if (Status.state == CaptureState.IDLE) Status.resetCounters()
    Status.lastError = null
    if (wakeLock == null) {
      wakeLock = getSystemService(PowerManager::class.java)
          .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "megamind:capture")
          .apply { setReferenceCounted(false); acquire() }
    }
    writer.open()
    Status.segmentStartMs = System.currentTimeMillis()
    Status.state = CaptureState.RECORDING
    val cam =
        camera
            ?: PassthroughCamera(
                    this,
                    { SettingsStore.get(this).frameIntervalMs },
                    video,
                    { SettingsStore.get(this).recordVideo },
                ) { jpeg, wallMs, luma ->
                  if (Status.state != CaptureState.RECORDING) return@PassthroughCamera
                  if (writer.addFrame(jpeg, wallMs)) {
                    Status.framesKept++
                    Status.lastJpeg = jpeg
                    Status.lastFrameLuma = luma
                    Status.lastFrameWallMs = wallMs
                    if (Status.framesKept <= 3 || Status.framesKept % 10 == 0) {
                      Log.i(TAG, "frame #${Status.framesKept} ${jpeg.size} B luma=$luma (received ${Status.framesReceived})")
                    }
                  }
                }
                .also { camera = it }
    cam.start()
    val au = audio ?: AudioCapture(this) { buf, n -> writer.writeAudio(buf, n) }.also { audio = it }
    Thread {
          Status.audioInfo = au.start()
        }
        .start()
    main.removeCallbacks(tick)
    main.post(tick)
    Log.i(TAG, "Capture started")
  }

  private fun rotateChunk(reopen: Boolean = true) {
    val frames = writer.framesInChunk
    if (writer.close()) {
      Status.chunksClosed++
      Log.i(TAG, "Chunk closed ($frames frames, audio total ${Status.audioBytes} B, last peak ${Status.audioPeak}); queued for upload")
      Uploader.kick()
    }
    if (reopen && Status.state == CaptureState.RECORDING) writer.open()
  }

  private fun haltDevices() {
    main.removeCallbacks(tick)
    camera?.stop()
    audio?.stop()
    if (Status.state == CaptureState.RECORDING && Status.segmentStartMs > 0) {
      Status.accumulatedMs += System.currentTimeMillis() - Status.segmentStartMs
    }
    Status.segmentStartMs = 0
    rotateChunk(reopen = false)
  }

  private fun pauseCapture() {
    if (Status.state != CaptureState.RECORDING) return
    SettingsStore.setCaptureWanted(this, false)
    haltDevices()
    Status.state = CaptureState.PAUSED
    Status.micSilenced = null
    Status.cameraInfo = "camera: paused"
    Status.audioInfo = "audio: paused"
    getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification("Paused"))
    Log.i(TAG, "Capture paused")
  }

  private fun stopCapture() {
    SettingsStore.setCaptureWanted(this, false)
    if (Status.state == CaptureState.RECORDING) haltDevices()
    writer.close()
    Status.state = CaptureState.IDLE
    Status.micSilenced = null
    Status.cameraInfo = "camera: off"
    Status.audioInfo = "audio: off"
    Uploader.kick()
    wakeLock?.release()
    wakeLock = null
    stopForeground(STOP_FOREGROUND_REMOVE)
    foreground = false
    stopSelf()
    Log.i(TAG, "Capture stopped")
  }

  override fun onDestroy() {
    if (Status.state == CaptureState.RECORDING) haltDevices()
    writer.close()
    Status.state = CaptureState.IDLE
    camera?.release()
    camera = null
    video.release()
    audio?.stop()
    wakeLock?.release()
    super.onDestroy()
  }
}
