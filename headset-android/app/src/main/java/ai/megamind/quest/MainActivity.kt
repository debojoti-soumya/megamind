package ai.megamind.quest

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.Locale

/**
 * Plain-Views UI (no Compose/AndroidX, for a fast and dependency-free build).
 *
 * Intent extras (for adb testing):
 *   --ez autostart true          start capture as soon as permissions are granted
 *   --es backend_url http://...  override and save the backend URL
 *   --es device_id quest3-abc    override and save the device_id sent to Megamind
 *   --es api_token <token>       save the server API_TOKEN (sent as Authorization: Bearer; "" clears it)
 *   --ez video false             turn real-video recording (/v1/video) off (true = on, the default)
 *   --es action start|pause|stop control capture
 */
class MainActivity : Activity() {
  companion object {
    private const val TAG = "MegamindQuest:UI"
    private const val REQ_PERMS = 42
    const val HEADSET_CAMERA = "horizonos.permission.HEADSET_CAMERA"
    val PERMISSIONS =
        arrayOf(Manifest.permission.CAMERA, HEADSET_CAMERA, Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS)
  }

  private val main = Handler(Looper.getMainLooper())
  private var autostartPending = false
  private var lastShownJpeg: ByteArray? = null

  private lateinit var dot: View
  private lateinit var stateText: TextView
  private lateinit var elapsedText: TextView
  private lateinit var statsText: TextView
  private lateinit var infoText: TextView
  private lateinit var errorText: TextView
  private lateinit var thumb: ImageView
  private lateinit var urlEdit: EditText
  private lateinit var intervalEdit: EditText
  private lateinit var chunkEdit: EditText
  private lateinit var deviceEdit: EditText
  private lateinit var tokenEdit: EditText
  private lateinit var videoCheck: CheckBox
  private lateinit var settingsMsg: TextView
  private lateinit var pulse: ObjectAnimator

  private val refresh =
      object : Runnable {
        override fun run() {
          render()
          main.postDelayed(this, 500)
        }
      }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    buildUi()
    Uploader.start(this)
    // A recreated activity (config change, task restored from recents) gets its original launch
    // intent again. Its autostart/action extras must not fire twice, or a capture the user
    // stopped would silently restart.
    val replayed =
        savedInstanceState != null ||
            (intent?.flags ?: 0) and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY != 0
    Log.i(TAG, "onCreate (replayed=$replayed, flags=0x${Integer.toHexString(intent?.flags ?: 0)})")
    handleIntent(intent, commands = !replayed)
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    Log.i(TAG, "onNewIntent (flags=0x${Integer.toHexString(intent.flags)})")
    handleIntent(intent, commands = intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY == 0)
  }

  /** Strips one-shot command extras so a later re-delivery of this activity's intent is inert. */
  private fun consumeCommandExtras(intent: Intent) {
    intent.removeExtra("autostart")
    intent.removeExtra("action")
    intent.removeExtra("backend_url")
    intent.removeExtra("device_id")
    intent.removeExtra("api_token")
    intent.removeExtra("video")
    setIntent(intent)
  }

  private fun handleIntent(intent: Intent?, commands: Boolean = true) {
    if (intent == null || !commands) {
      autostartPending = false
      val missing = PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
      if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), REQ_PERMS)
      return
    }
    intent.getStringExtra("backend_url")?.let { url ->
      val s = SettingsStore.get(this).copy(backendUrl = url)
      SettingsStore.save(this, s)
      urlEdit.setText(url)
      Uploader.retryNow()
    }
    intent.getStringExtra("device_id")?.takeIf { it.isNotBlank() }?.let { id ->
      SettingsStore.save(this, SettingsStore.get(this).copy(deviceId = id.trim()))
      deviceEdit.setText(id.trim())
    }
    intent.getStringExtra("api_token")?.let { t ->
      SettingsStore.save(this, SettingsStore.get(this).copy(apiToken = t.trim()))
      tokenEdit.setText(t.trim())
      Uploader.retryNow()
    }
    if (intent.hasExtra("video")) {
      val on = intent.getBooleanExtra("video", true)
      SettingsStore.save(this, SettingsStore.get(this).copy(recordVideo = on))
      videoCheck.isChecked = on
    }
    autostartPending = intent.getBooleanExtra("autostart", false)
    when (intent.getStringExtra("action")) {
      "start" -> autostartPending = true
      "pause" -> CaptureService.send(this, CaptureService.ACTION_PAUSE)
      "stop" -> CaptureService.send(this, CaptureService.ACTION_STOP)
    }
    consumeCommandExtras(intent)
    val missing = PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
    if (missing.isNotEmpty()) {
      Log.i(TAG, "Requesting permissions: $missing")
      requestPermissions(missing.toTypedArray(), REQ_PERMS)
    } else {
      maybeAutostart()
    }
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    permissions.forEachIndexed { i, p -> Log.i(TAG, "permission $p -> ${if (grantResults.getOrNull(i) == 0) "granted" else "denied"}") }
    maybeAutostart()
  }

  private fun cameraPermsOk(): Boolean =
      checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
          checkSelfPermission(HEADSET_CAMERA) == PackageManager.PERMISSION_GRANTED

  private fun maybeAutostart() {
    if (!autostartPending) return
    autostartPending = false
    if (!cameraPermsOk()) Status.error("Camera permission missing (CAMERA + HEADSET_CAMERA); starting anyway")
    Log.i(TAG, "Autostart requested")
    CaptureService.send(this, CaptureService.ACTION_START)
  }

  override fun onResume() {
    super.onResume()
    main.post(refresh)
  }

  override fun onPause() {
    super.onPause()
    main.removeCallbacks(refresh)
  }

  private fun dp(v: Int): Int = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()

  private fun text(size: Float, color: Int = Color.WHITE, bold: Boolean = false) =
      TextView(this).apply {
        setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
        setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
      }

  private fun button(label: String, color: Int, onClick: () -> Unit) =
      Button(this).apply {
        text = label
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
        setTextColor(Color.WHITE)
        isAllCaps = false
        background = GradientDrawable().apply { cornerRadius = dp(12).toFloat(); setColor(color) }
        setPadding(dp(20), dp(12), dp(20), dp(12))
        setOnClickListener { onClick() }
      }

  private fun buildUi() {
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(24), dp(20), dp(24), dp(20))
      setBackgroundColor(Color.rgb(18, 18, 22))
    }

    // Header: pulsing dot + state + elapsed
    val header = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
    dot = View(this).apply {
      background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(Color.rgb(255, 59, 48)) }
    }
    header.addView(dot, LinearLayout.LayoutParams(dp(36), dp(36)).apply { rightMargin = dp(14) })
    stateText = text(34f, bold = true)
    header.addView(stateText)
    elapsedText = text(34f, Color.rgb(200, 200, 210)).apply { typeface = Typeface.MONOSPACE }
    header.addView(elapsedText, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { leftMargin = dp(20) })
    root.addView(header)

    pulse = ObjectAnimator.ofFloat(dot, "alpha", 1f, 0.15f).apply {
      duration = 700
      repeatMode = ValueAnimator.REVERSE
      repeatCount = ValueAnimator.INFINITE
    }

    statsText = text(20f).apply { setPadding(0, dp(10), 0, 0) }
    root.addView(statsText)

    val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(14), 0, dp(10)) }
    val lp = { LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { rightMargin = dp(10) } }
    buttons.addView(button("Start", Color.rgb(200, 40, 40)) {
      if (!cameraPermsOk()) requestPermissions(PERMISSIONS, REQ_PERMS)
      CaptureService.send(this, CaptureService.ACTION_START)
    }, lp())
    buttons.addView(button("Pause", Color.rgb(90, 90, 110)) { CaptureService.send(this, CaptureService.ACTION_PAUSE) }, lp())
    buttons.addView(button("Stop", Color.rgb(60, 60, 75)) { CaptureService.send(this, CaptureService.ACTION_STOP) }, lp())
    root.addView(buttons)

    val mid = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    thumb = ImageView(this).apply {
      setBackgroundColor(Color.rgb(40, 40, 48))
      scaleType = ImageView.ScaleType.FIT_CENTER
      adjustViewBounds = true
    }
    mid.addView(thumb, LinearLayout.LayoutParams(dp(320), dp(240)).apply { rightMargin = dp(16) })
    infoText = text(15f, Color.rgb(180, 180, 190))
    mid.addView(infoText, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
    root.addView(mid)

    errorText = text(16f, Color.rgb(255, 120, 110)).apply { setPadding(0, dp(10), 0, dp(6)) }
    root.addView(errorText)

    // Settings
    root.addView(text(20f, bold = true).apply { text = "Settings"; setPadding(0, dp(12), 0, dp(4)) })
    val s = SettingsStore.get(this)
    fun field(label: String, value: String, type: Int): EditText {
      root.addView(text(14f, Color.rgb(170, 170, 180)).apply { text = label })
      return EditText(this).apply {
        setText(value)
        inputType = type
        setTextColor(Color.WHITE)
        setSingleLine()
        root.addView(this, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
      }
    }
    urlEdit = field("Backend URL (localhost works over USB with adb reverse; use http://<mac-ip>:8000 on Wi-Fi)", s.backendUrl,
        InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)
    intervalEdit = field("Frame interval (seconds)", fmt(s.frameIntervalSec), InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL)
    chunkEdit = field("Chunk length (seconds; also the length of each /v1/audio clip)", s.chunkSec.toString(), InputType.TYPE_CLASS_NUMBER)
    deviceEdit = field("Device ID (device_id sent with every frame and audio clip)", s.deviceId, InputType.TYPE_CLASS_TEXT)
    tokenEdit = field("API token (the server's API_TOKEN, required by the Modal deploy; empty = none)", s.apiToken,
        InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD)
    videoCheck = CheckBox(this).apply {
      text = "Record real video (30 fps H.264 to /v1/video; JPEG-only fallback if the encoder fails; applies from the next Start)"
      setTextColor(Color.WHITE)
      isChecked = s.recordVideo
      root.addView(this, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(6) })
    }
    val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(10), 0, 0) }
    settingsMsg = text(14f, Color.rgb(170, 220, 170))
    row.addView(button("Save", Color.rgb(50, 110, 200)) { saveSettings() }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { rightMargin = dp(10) })
    row.addView(button("Test connection", Color.rgb(60, 60, 75)) {
      val url = urlEdit.text.toString()
      val token = tokenEdit.text.toString().trim()
      settingsMsg.text = "Testing..."
      Thread { val r = Uploader.testConnection(url, token); main.post { settingsMsg.text = r } }.start()
    }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { rightMargin = dp(10) })
    row.addView(button("Retry uploads", Color.rgb(60, 60, 75)) { Uploader.requeueFailed(); Uploader.retryNow() })
    root.addView(row)
    root.addView(settingsMsg)

    setContentView(ScrollView(this).apply { addView(root); setBackgroundColor(Color.rgb(18, 18, 22)) })
  }

  private fun fmt(d: Double) = if (d == Math.floor(d)) d.toInt().toString() else d.toString()

  private fun saveSettings() {
    val interval = intervalEdit.text.toString().toDoubleOrNull()
    val chunk = chunkEdit.text.toString().toIntOrNull()
    val url = urlEdit.text.toString().trim()
    val device = deviceEdit.text.toString().trim()
    if (interval == null || interval < 0.1 || chunk == null || chunk < 1 || chunk > 600 || device.isEmpty() ||
        Uploader.normalizeBaseUrl(url) == null) {
      settingsMsg.text = "Invalid settings (interval >= 0.1 s, chunk 1-600 s, device ID, valid URL)"
      return
    }
    SettingsStore.save(this, Settings(url, interval, chunk, device, videoCheck.isChecked, tokenEdit.text.toString().trim()))
    settingsMsg.text = "Saved"
    Uploader.retryNow()
  }

  private fun render() {
    val st = Status.state
    stateText.text = when (st) { CaptureState.RECORDING -> "REC"; CaptureState.PAUSED -> "PAUSED"; CaptureState.IDLE -> "IDLE" }
    (dot.background as GradientDrawable).setColor(
        when (st) { CaptureState.RECORDING -> Color.rgb(255, 59, 48); CaptureState.PAUSED -> Color.rgb(255, 180, 0); else -> Color.rgb(90, 90, 100) })
    if (st == CaptureState.RECORDING) { if (!pulse.isStarted) pulse.start() } else { pulse.cancel(); dot.alpha = 1f }
    val e = Status.elapsedMs() / 1000
    elapsedText.text = String.format(Locale.US, "%02d:%02d:%02d", e / 3600, (e / 60) % 60, e % 60)
    statsText.text = "Frames: ${Status.framesKept}    Chunks sent: ${Status.chunksSent}    Pending: ${Status.chunksPending}" +
        (if (Status.chunksFailed > 0) "    Failed: ${Status.chunksFailed}" else "")
    val lumaNote = when {
      Status.lastFrameLuma < 0 -> ""
      Status.lastFrameLuma < 8 -> " (frame is BLACK - is the headset on?)"
      else -> ""
    }
    infoText.text = listOf(
        Status.cameraInfo,
        "${Status.videoInfo}  (encoded ${Status.videoFramesEncoded} frames, ${Status.videoChunks} chunks, ${Status.videoSent} uploaded)",
        "${Status.audioInfo}  (peak ${Status.audioPeak}/32767)" +
            (if (Status.micSilenced == true) "\n  MIC MUTED BY SYSTEM - headset off-head/asleep or mic privacy on; frames still captured" else ""),
        "camera frames received: ${Status.framesReceived}",
        "last frame luma: ${Status.lastFrameLuma}$lumaNote",
        "upload: ${Status.uploadInfo}",
        "backend: ${SettingsStore.get(this).backendUrl}  device_id: ${SettingsStore.get(this).deviceId}  api token: " +
            (if (SettingsStore.get(this).apiToken.isEmpty()) "none" else "set"),
    ).joinToString("\n")
    errorText.text = Status.lastError?.let { "Last error: $it" } ?: ""
    val jpeg = Status.lastJpeg
    if (jpeg != null && jpeg !== lastShownJpeg) {
      lastShownJpeg = jpeg
      val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
      BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, opts)?.let { thumb.setImageBitmap(it) }
    }
  }
}
