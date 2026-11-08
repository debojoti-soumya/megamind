package ai.megamind.quest

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * 16 kHz mono PCM16 from the headset mic. Delivers little-endian PCM to [sink] on its own thread.
 * Simplified from the Ray-Ban app's AudioCapture (no Bluetooth SCO routing).
 */
class AudioCapture(context: Context, private val sink: (ByteArray, Int) -> Unit) {
  companion object {
    private const val TAG = "MegamindQuest:Audio"
    const val SAMPLE_RATE = 16000
    private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
  }

  private val context = context.applicationContext
  @Volatile private var active = false
  private var record: AudioRecord? = null
  private var thread: Thread? = null

  @SuppressLint("MissingPermission")
  fun start(): String {
    if (active) return "audio: on"
    if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      return "audio: off (no RECORD_AUDIO permission)"
    }
    val min = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
    if (min <= 0) return "audio: off (16 kHz unsupported)"
    for (source in intArrayOf(MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC)) {
      try {
        val r = AudioRecord(source, SAMPLE_RATE, CHANNEL, ENCODING, maxOf(min * 4, SAMPLE_RATE))
        if (r.state != AudioRecord.STATE_INITIALIZED) {
          r.release()
          continue
        }
        r.startRecording()
        record = r
        active = true
        thread = Thread({ readLoop(r) }, "megamind-audio").apply { start() }
        val label = "audio: headset mic 16 kHz (source $source)"
        Log.i(TAG, label)
        return label
      } catch (e: Exception) {
        Log.w(TAG, "AudioRecord(source=$source) failed", e)
      }
    }
    return "audio: off (microphone unavailable)"
  }

  fun stop() {
    active = false
    // Stop first: while the system mutes the mic (headset doffed/asleep -> mic privacy on),
    // AudioRecord.read() blocks indefinitely, and only stop() unblocks it.
    record?.let { runCatching { it.stop() } }
    thread?.join(1000)
    thread = null
    record?.let { runCatching { it.release() } }
    record = null
  }

  /**
   * True when the system is silencing this recording (on Quest: Horizon turns the microphone
   * privacy toggle on while the headset is off-head/asleep), false when live, null if unknown.
   */
  fun isSilenced(): Boolean? =
      try {
        record?.activeRecordingConfiguration?.isClientSilenced
      } catch (e: Exception) {
        null
      }

  private fun readLoop(r: AudioRecord) {
    val buf = ByteArray(SAMPLE_RATE / 10 * 2) // ~100 ms
    try {
      while (active) {
        val n = r.read(buf, 0, buf.size)
        when {
          n > 0 -> {
            sink(buf, n)
            Status.audioBytes += n
            var peak = 0
            var i = 0
            while (i + 1 < n) {
              val v = kotlin.math.abs(((buf[i + 1].toInt() shl 8) or (buf[i].toInt() and 0xFF)).toShort().toInt())
              if (v > peak) peak = v
              i += 2
            }
            Status.audioPeak = peak
          }
          n == 0 -> Thread.sleep(10)
          else -> {
            Log.w(TAG, "AudioRecord.read error $n")
            Status.error("Audio read error $n")
            break
          }
        }
      }
    } catch (e: Exception) {
      Log.w(TAG, "Audio loop ended", e)
    }
  }
}
