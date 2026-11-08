package ai.megamind.quest

import android.content.Context

data class Settings(
    val backendUrl: String = DEFAULT_BACKEND,
    val frameIntervalSec: Double = 1.0,
    val chunkSec: Int = 5,
    /** `device_id` sent in every /v1/frames and /v1/audio meta. */
    val deviceId: String = DEFAULT_DEVICE_ID,
    /**
     * Record real H.264 video (30 fps) next to the JPEGs and upload one clip per chunk to `/v1/video`.
     * On by default; the camera falls back to JPEG-only by itself if the encoder fails. Applies from the
     * next camera start.
     */
    val recordVideo: Boolean = true,
    /** Server `API_TOKEN` (always set on the Modal deploy, docs/api.md#auth), sent as `Authorization: Bearer`. Empty = none. */
    val apiToken: String = "",
) {
  val frameIntervalMs: Long
    get() = (frameIntervalSec * 1000).toLong().coerceAtLeast(100L)

  val chunkMs: Long
    get() = chunkSec.coerceAtLeast(1) * 1000L

  companion object {
    /** Megamind's uvicorn default port; reach it over USB with `adb reverse tcp:8000 tcp:8000`. */
    const val DEFAULT_BACKEND = "http://localhost:8000"
    const val DEFAULT_DEVICE_ID = "quest3-native"
  }
}

object SettingsStore {
  private const val PREFS = "megamind_quest"

  @Volatile private var cached: Settings? = null

  fun get(context: Context): Settings {
    cached?.let { return it }
    val p = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    val s =
        Settings(
            backendUrl = p.getString("backend_url", Settings.DEFAULT_BACKEND) ?: Settings.DEFAULT_BACKEND,
            frameIntervalSec = p.getFloat("frame_interval_sec", 1.0f).toDouble(),
            chunkSec = p.getInt("chunk_sec", 5),
            deviceId = p.getString("device_id", Settings.DEFAULT_DEVICE_ID)?.takeIf { it.isNotBlank() } ?: Settings.DEFAULT_DEVICE_ID,
            recordVideo = p.getBoolean("record_video", true),
            apiToken = p.getString("api_token", "")?.trim().orEmpty(),
        )
    cached = s
    return s
  }

  /**
   * Whether the user wants capture running. Survives process death so a START_STICKY restart
   * (null intent) resumes recording only if it was recording, never after Pause/Stop.
   */
  fun captureWanted(context: Context): Boolean =
      context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("capture_wanted", false)

  fun setCaptureWanted(context: Context, wanted: Boolean) {
    context.applicationContext
        .getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        .edit()
        .putBoolean("capture_wanted", wanted)
        .apply()
  }

  fun save(context: Context, s: Settings) {
    context.applicationContext
        .getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        .edit()
        .putString("backend_url", s.backendUrl)
        .putFloat("frame_interval_sec", s.frameIntervalSec.toFloat())
        .putInt("chunk_sec", s.chunkSec)
        .putString("device_id", s.deviceId)
        .putBoolean("record_video", s.recordVideo)
        .putString("api_token", s.apiToken)
        .apply()
    cached = s
  }
}
