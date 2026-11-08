#!/usr/bin/env bash
# Record what the wearer sees, for the demo video, then caption it with the real questions and answers.
#
#   record-headset.sh pull [DIR]          newest Quest recording (Meta button > Camera > Record video) -> DIR
#   record-headset.sh mirror OUT.mp4      live mirror + record over adb with scrcpy (Ctrl-C to stop)
#   record-headset.sh captions VIDEO [make_captions.py args...]   VIDEO.srt + VIDEO-captioned.mp4
#
# env: ADB_SERIAL (pick a headset), MEGAMIND_SERVER + MEGAMIND_TOKEN (captions), CROP (scrcpy --crop W:H:X:Y,
# one eye out of the side-by-side Quest image; find it once with `scrcpy` and adjust), SHOTS (on-device folder).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
server_dir="$here/../../server"
adb=(adb ${ADB_SERIAL:+-s "$ADB_SERIAL"})
shots="${SHOTS:-/sdcard/Oculus/VideoShots}"

case "${1:-}" in
  pull)
    dir="${2:-.}"
    mkdir -p "$dir"
    latest="$("${adb[@]}" shell "ls -t $shots/*.mp4 2>/dev/null | head -1" | tr -d '\r')"
    [ -n "$latest" ] || { echo "no recordings in $shots (record one first, or set SHOTS)" >&2; exit 1; }
    "${adb[@]}" pull "$latest" "$dir/"
    echo "$dir/$(basename "$latest")"
    ;;
  mirror)
    out="${2:?usage: $0 mirror OUT.mp4}"
    command -v scrcpy >/dev/null || { echo "install scrcpy (brew install scrcpy / apt install scrcpy)" >&2; exit 1; }
    args=(--no-audio-playback --record "$out" --window-title "Megamind headset view")
    [ -n "${CROP:-}" ] && args+=(--crop "$CROP")
    exec scrcpy ${ADB_SERIAL:+--serial "$ADB_SERIAL"} "${args[@]}"
    ;;
  captions)
    video="${2:?usage: $0 captions VIDEO [make_captions.py args...]}"
    shift 2
    python3 "$server_dir/tools/make_captions.py" "$video" --burn "${video%.*}-captioned.mp4" "$@"
    ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
