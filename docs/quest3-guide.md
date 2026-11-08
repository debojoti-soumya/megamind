# Meta Quest 3 guide for Megamind (first-time Quest developers)

This guide is for teammates who have never built for Quest. It covers the headset basics, the
physical links (USB, Wi-Fi, phone, power), and getting the Megamind capture app onto the headset
and talking to the server. [`headset/README.md`](../headset/README.md) is the build reference and
this guide links to it instead of repeating it.

> **Not verified on device.** This guide was written from the repo source and Meta's/Tailscale's
> docs. It has not been tried on a headset or in Unity. Steps that depend on device or Unity
> behaviour are marked **(verify on device)**. Menu names come from the cited Meta pages. Horizon OS
> renames menus often, so trust what you see in the headset over this text, then fix this file.

Contents: [Demo-day quick card](#demo-day-quick-card) · [1. Quest 3 basics](#1-quest-3-basics) ·
[2. Connections](#2-connections) · [3. Developer setup](#3-developer-setup) ·
[4. Getting Megamind on the headset](#4-getting-megamind-on-the-headset) ·
[5. Pointing the app at the server](#5-pointing-the-app-at-the-server) ·
[6. Test runbook](#6-test-runbook) · [7. Troubleshooting](#7-troubleshooting) · [Sources](#sources)

---

## Demo-day quick card

**Night before**
1. Charge the headset to 100% (about 2.5 h on the included 18 W charger [M-charge]) and charge the controllers.
2. Update the headset: Meta button → *Quick controls* → *Settings* → *General* → *Software update*. It must be **v74 or newer** [M-update][D-pca].
3. Install the latest `megamind.apk`: `adb install -r Builds/megamind.apk`. Open it once and **Allow** the camera prompt.

**Morning of (in this order)**
1. **Reboot the headset.** On v74 the camera colours can be wrong after a camera app's first install, and a reboot fixes it [D-pca].
2. Laptop: `cd server && uvicorn megamind.main:app --host 0.0.0.0 --port 8000`.
3. Laptop: `curl http://<server-ip>:8000/v1/health` should return `{"status":"ok",...}`.
4. Headset: confirm it is on the **same network** as the server (Wi-Fi or Tailscale, see [2.3](#23-quest--server-the-network-path)). Open `http://<server-ip>:8000/v1/health` in the Quest Browser (verify on device).
5. Laptop: `adb devices` should list the headset as `device`. Then start `adb logcat -s Unity | grep -i megamind` in a spare terminal.
6. Wipe old memories if you want a clean demo: `curl -X DELETE http://<server-ip>:8000/v1/memories`.
7. Launch: Meta button → *Library* → *Unknown Sources* → **Megamind** [D-deploy].
8. The chat panel header should show a blinking red REC dot and `REC 1 fps · N sent · mic M · url:<source>` with `sent` rising roughly once a second (corner tab → *Debug stats* shows the full stats: resolution, accepted, last reason, latency). "HUD" below means this header strip / Debug view.
9. Start casting for the audience: Navigator → *Quick controls* → *Cast* → phone/computer [M-cast]. **Cast or record, not both**, because Meta lists doing both at once as a known issue [M-cast-ts].
10. Plug the headset into a power bank or charger between runs. Keep it worn or face-up and still. Taking it off can suspend the app (see [7](#7-troubleshooting)).

**If it breaks:** look up the symptom in the [decision table](#27-decision-table-symptom--link--2-minute-check).
Press **A/X**, or pinch thumb + **middle** finger with hand tracking, to pause or resume capture (the header shows `PAUSED`, the REC dot turns grey; Debug: `capture PAUSED (press A/X or middle-pinch)`). Or say **"pause recording"** / **"start recording"** while holding the trigger.
Capture pauses on its own only when the app is really backgrounded (HUD `capture PAUSED (app backgrounded)`, counter
`bg N`). Opening the system menu or a boundary dialog does not pause it (HUD `REC … (unfocused)`, counter `unfocus N`) [D-focus].

---

## 1. Quest 3 basics

| Task | How |
|---|---|
| Open the menu ("Navigator") | Press the **Meta button** on the right controller [M-update]. Hold it to recenter the view [M-record]. |
| Switch to hand tracking | Enable it once: *Settings → Movement tracking → Hand tracking* (older Horizon OS: Navigator → *Quick controls* → *Settings* → *Hands & body* → *Hand and Body Tracking* [M-hands]) (verify on device). Then put the controllers down. In Megamind: **index pinch** = push-to-talk (hold, release to send), **middle pinch** = capture pause/resume (like A/X), and the right hand draws the panel ray (pinch to click) (verify on device). |
| Panel UI ray | The **right controller** draws a ray + cursor on the chat/settings panel; pull the trigger to click (verify on device). |
| Chat panel footer | **Record / Pause** toggles continuous capture (state line: *Recording continuously* / *Paused*). **Hold to talk**: point at it, hold the trigger or index pinch, speak, release. **Ask Megamind…** opens the system keyboard; **Send** asks the typed question. The small **arrow tab** above the panel's top-right corner folds the whole window up behind it and drops it down again (header **Hide** folds it too); only the tab and its blinking REC dot stay visible, and the state survives relaunch. Pointing the ray at the panel and pressing the trigger is a click only — it no longer starts a recording (hold the trigger away from the panel, or use Hold to talk). |
| Chat terminal | Answers stream in as they are written. Thumbnails appear while Megamind searches; tap one to open it large (‹ › next/prev, sticks zoom/pan, B/Y or X to close). Copy / Retry under each answer. Bottom bar: **Hold to talk**, **Ask Megamind…** (keyboard), **Send**; the line above says Listening / Transcribing / Thinking / Speaking. |
| Chat panel drawer | The small **tab** in the panel's top-right corner (three bars + chevron) slides a drawer down the right edge: **+ New chat** and the chats list (Today / Yesterday / Earlier; tap to resume a chat — the same server session, so Claude remembers it; long-press or `…` to rename/delete; chats survive relaunch), then Settings, Debug stats (capture/mic stats, server URL), Pin/Unpin panel, Text size, Controls (all button/hand mappings + voice phrases), Purge memory. Click the tab again or anywhere else to close it. Voice: hold the trigger and say **"open menu"**, **"open settings"**, **"show debug"**, **"close menu"**, **"pause recording"**, **"start recording"**, **"bigger text"**, **"smaller text"**, **"pin panel"**, **"unpin panel"** — short commands are executed instead of being answered (verify on device). Commands are recognised right after speech-to-text (~0.5 s, no answer is generated) and loosely: "could you open the menu" or a slightly misheard "paws recording" work too; a small chip shows what was understood. |
| Chat panel placement | The panel follows you lazily: it sits at eye height 1 m ahead and glides back in front of you when you turn more than ~25° or walk more than 0.5 m. **Pin** (header button) world-locks it, **Unpin** lets it follow again; hold **B/Y** for 0.5 s to drag it with your gaze. **A-** / **A+** in the header make the whole panel (text and thumbnails) 1x–2x; the choice is remembered (verify on device). |
| See your room (passthrough) | Navigator → *Quick settings* → *See your surroundings*. Turning on **Double Tap for Passthrough** in Passthrough settings lets you tap the side of the headset twice [M-passthrough]. |
| Lighting | Meta recommends **≥ 50 lux** for colour passthrough [M-passthrough]. Dim rooms make the camera feed dark and the server rejects frames as `dark`/`blurry`. |
| Boundary | The boundary is the virtual wall that appears near the edge of your play area [M-boundary]. Megamind declares itself a **boundaryless** mixed-reality app (`com.oculus.feature.BOUNDARYLESS_APP` in its manifest), so the boundary should not appear while it runs and you can walk between rooms [D-boundaryless] (verify on device). This only applies to the installed APK, not over Link [D-boundaryless]. Watch where you walk: Megamind shows passthrough only, with no obstacle warnings [D-boundaryless-bp]. |
| Wi-Fi | Navigator → *Quick controls* → *Settings* → *Wi-Fi* [M-wifi]. |
| Battery / charging | USB-C port on the **left** side. LED orange = charging, green = full. Quest 3 needs **18 W** [M-charge]. |
| Cast to phone / browser / PC | Navigator → *Quick controls* → *Cast* → *Mobile*, *Web*, or a nearby device → *Next* [M-cast]. |
| Record a video in the headset | Navigator → *Camera* → *Record video*. A red dot appears while recording; select *Stop* to end. Shortcut: hold the Meta button and then hold the right trigger. Videos are in *Files* (Media/Recents) or *Gallery* [M-record]. |
| Screenshot or record from a laptop | Meta VR Developer Hub (MVDH, formerly MQDH) can capture screenshots and video, and can cast the headset view to the computer [D-mvdh]. |

Megamind draws its own **REC** dot and status strip in the chat panel header (`CaptureHud.cs` is the data source). Your casts and recordings will show it, which is intended.

---

## 2. Connections

Megamind uses up to five physical links. Most problems on the night are one of these links failing,
not a code bug, so find the failing link first using the [decision table](#27-decision-table-symptom--link--2-minute-check).

```
            USB-C (adb, charging)         Wi-Fi / Tailscale (HTTP :8000)
 laptop  <─────────────────────────>  Quest 3  ───────────────────────────>  server (laptop or cloud)
 (Unity,      wireless adb :5555        │  cast (same Wi-Fi + same Meta account)
  adb)   <─────────────────────────>    └──────────────────────────>  phone / browser / PC (demo screen)
```

### 2.1 Quest ↔ laptop over USB-C (adb)

**Cable.** You need a **data-capable** USB-C cable. Many cables, including the ones bundled with
power banks, only charge. Unity's *Build and Run* needs a data cable [D-hw]. Link needs USB 3.2
with at least 5 Gbps [M-link], but adb works on any data cable. If a cable charges the headset
but `adb devices` stays empty, try a different cable before anything else. Prefer a port directly
on the laptop over a hub.

**First connection (each laptop, once):**
1. Developer mode is already on. To confirm it: Meta Horizon phone app → headset → *Headset Settings* → *Developer Mode* toggle should be on [D-setup].
2. Plug in the cable and **put the headset on**. Accept the **"Allow USB debugging"** prompt and tick **"Always allow from this computer"** [D-setup].
3. Run `adb devices` [D-adb]:

| `adb devices` output | Meaning | Fix |
|---|---|---|
| `1WMHH8...  device` | Working | — |
| `...  unauthorized` | You have not accepted the RSA prompt on this laptop yet | Put the headset on and accept the prompt. If it never appears: unplug the cable, restart the headset, plug it back in [D-adb]. Also try `adb kill-server && adb devices`. |
| `...  offline` | adb lost the session (sleep, cable wiggle, or an adb version mismatch) | `adb kill-server && adb devices`. Replug. Close any other adb (Unity, MVDH, or Android Studio can each run their own copy). |
| `... no permissions` (Linux) | No udev rule for the device | Add your user to `plugdev` and install the Android udev rules (on Debian/Ubuntu, `sudo apt install android-sdk-platform-tools-common`), then replug [A-device]. |
| empty list | Charge-only cable, driver missing (Windows), or developer mode off | Use another cable or port. On Windows install the **Oculus ADB Drivers** [D-setup]. Check the Developer Mode toggle. |
| `adb: command not found` | platform-tools are not on your PATH | Install the Android SDK Platform-Tools [A-pt][D-adb]. macOS: `brew install --cask android-platform-tools`. Unity also bundles adb under its Android SDK folder. |

Per-OS notes:
- **Windows:** needs the Oculus ADB driver (linked from [D-setup]). MVDH is available [D-mvdh].
- **macOS:** no driver needed. MVDH is available [D-mvdh]. If the prompt never appears, try a different USB-C port or adapter.
- **Linux:** MVDH is **not** available (Windows and macOS only) [D-mvdh]. Use plain adb and the udev rules [A-device].

If both USB and wireless adb are connected, the headset is listed twice. Target one with `adb -s <serial> ...` [D-adb].

### 2.2 Quest ↔ laptop wireless adb

Use this when you want to walk around untethered and still read logs or reinstall [D-adb]:
1. Connect over USB first (2.1). Put the headset and laptop on the same Wi-Fi.
2. Find the headset IP: `adb shell ip route`. The IP is the address after `src` [D-adb].
3. `adb tcpip 5555`
4. Unplug the cable. Then run `adb connect <headset-ip>:5555`. `adb devices` should now show `<ip>:5555  device`.
5. When you are done, run `adb disconnect`. Wireless adb usually stops after the headset reboots, so repeat steps 1–4 afterwards (verify on device).

Wireless adb needs the laptop to reach the headset directly over the LAN, so **venue Wi-Fi with client isolation will block it** (2.3). Use a laptop or phone hotspot instead.
MVDH can also deploy and show logs, but this guide does not cover its wireless flow because it was not verified against Meta's docs. The MVDH device-setup page says to put the headset on the same Wi-Fi as the computer for casting [D-mvdh-setup].

### 2.3 Quest ↔ server: the network path

The app sends plain HTTP `POST <serverBaseUrl>/v1/frames` (multipart JPEG + JSON) once per second
(`FrameUploader.cs`, [api.md](api.md)). The app does not call `/v1/health`. Health is a manual check.

**The URL comes from (highest priority first)** (`RuntimeConfig.cs`; the device id, API token and ambient-audio toggle follow the same order):
1. Launch-intent extras on `adb shell am start ... --es serverBaseUrl <url>` ([section 5](#5-pointing-the-app-at-the-server)). The app saves them to PlayerPrefs.
2. PlayerPrefs saved earlier: by a previous intent launch, or by the in-app settings panel (the chat UI path in [PR #2](https://github.com/debojoti-soumya/megamind/pull/2)).
3. `megamind.json` in the app's files dir (`/sdcard/Android/data/com.megamind.quest/files/`), only if the app can read it. On Quest 3 / Android 14 the app could not read a file written there by `adb shell` (`Access ... is denied`), so don't rely on it.
4. `Assets/Megamind/MegamindConfig.asset` → `Server Base Url`. This is baked into the APK. The code default is `http://100.64.0.1:8000`.

The HUD's last line shows the active URL and which source it came from (`url:intent|saved|panel|megamind.json|asset`).
Setting the URL is covered in [section 5](#5-pointing-the-app-at-the-server).

**HTTP vs HTTPS.** The project's `AndroidManifest.xml` sets `android:usesCleartextTraffic="true"`, so
both `http://` and `https://` URLs work. Android blocks cleartext HTTP by default when that flag is
missing [A-cleartext]. If someone regenerates the manifest (*Meta → Tools → Update AndroidManifest.xml*),
check that this flag **and** `horizonos.permission.HEADSET_CAMERA` survive [D-pca-unity]. For an
HTTPS URL, leave off the port unless the tunnel uses one, and don't add a trailing path. The app appends `/v1/frames` itself.

**Pick one network option:**

| Option | serverBaseUrl | Works on venue Wi-Fi with client isolation? | Notes |
|---|---|---|---|
| A. Same Wi-Fi, laptop LAN IP | `http://192.168.x.y:8000` | **No.** Client isolation blocks device-to-device traffic. | Simplest at home. Get the laptop IP with `ipconfig getifaddr en0` (macOS), `hostname -I` (Linux), or `ipconfig` (Windows). |
| B. Laptop/phone hotspot | `http://<laptop-ip-on-hotspot>:8000` | Yes, because the hotspot is your own network | Join the Quest to the hotspot [M-wifi]. Phone hotspot: laptop and Quest both join it. |
| C. Tailscale on both ends (recommended) | `http://100.x.y.z:8000` (`tailscale ip -4` on the server) | Usually yes, because traffic is relayed when direct paths are blocked | Quest: sideload the Tailscale Android APK (`tailscale-android-universal-*.apk`, Android 8+) from https://pkgs.tailscale.com/stable/#android with `adb install`. Then open it from *Unknown Sources* and log in to the same tailnet. Signing in inside the headset may be fiddly (verify on device). |
| D. `tailscale serve` on the server | `https://<machine>.<tailnet>.ts.net` | Yes (tailnet only) | Needs HTTPS certificates enabled for the tailnet. The Quest still needs Tailscale. `tailscale serve 8000` [T-serve] (exact CLI syntax varies by version; verify). |
| E. Tailscale Funnel / ngrok / any public HTTPS URL | `https://...` | Yes, and the Quest does **not** need Tailscale | Funnel is public internet, disabled by default, only on ports 443/8443/10000, and DNS can take up to 10 min [T-funnel]. Set `API_TOKEN` on the server and pass it to the app with `--es apiToken <token>` (section 5, [api.md#auth](api.md#auth)). Without it anyone with the URL can read or wipe memories. Turn the tunnel off after the demo. |
| F. Modal (public HTTPS, recommended) | `https://<workspace>--megamind-web.modal.run` | Yes. The server is in the cloud, so no hotspot, Tailscale, firewall rule or client isolation matters | Deploy with [deploy/README.md](../deploy/README.md) (one CPU container, memories on a Modal Volume). Pass the deploy's `API_TOKEN` to the app with `--es apiToken <token>` on `am start` (section 5). The native app (`headset-android/`) takes it as Settings → *API token* or `--es api_token <token>`. HTTPS needs nothing special in `UnityWebRequest`, and the manifest's cleartext flag doesn't affect `https://`. The first request after 10 min idle has a ~10-15 s cold start. The server-side checklist below does not apply. Test with `curl -H "Authorization: Bearer <token>" <url>/v1/health` then the Quest Browser at `<url>/v1/health?token=<token>` (verify on device). |

**Server side checklist:**
1. Bind to all interfaces: `uvicorn megamind.main:app --host 0.0.0.0 --port 8000`. The default `127.0.0.1` is unreachable from the headset.
2. Open port 8000/TCP in the laptop firewall:
   - macOS: when the "accept incoming connections" prompt for Python/uvicorn appears, allow it. Otherwise use *System Settings → Network → Firewall* (verify on your macOS version).
   - Windows (admin PowerShell): `New-NetFirewallRule -DisplayName "megamind 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`. Windows may also put new Wi-Fi networks under the *Public* profile, which blocks inbound connections.
   - Linux (ufw): `sudo ufw allow 8000/tcp`.
3. Test from the outside in, in this order:
   1. Laptop: `curl http://127.0.0.1:8000/v1/health`. Checks that the server is up.
   2. Laptop: `curl http://<serverBaseUrl-host>:8000/v1/health`. Checks the bind address and the firewall to self.
   3. **Phone** on the same Wi-Fi/tailnet: open the same URL in its browser. Checks LAN reachability and client isolation.
   4. **Quest Browser**: open the same URL (verify on device). This is the exact path the app will use.

`adb shell curl` (suggested in older notes) depends on the headset having a `curl` binary. Don't rely
on it (verify on device). `adb shell ping -c 3 <server-ip>` is a cheaper reachability check (verify on device).

### 2.4 Quest ↔ phone (pairing and casting)

- **Pairing / developer mode** is done in the **Meta Horizon** phone app (headset → *Headset Settings* → *Developer Mode*) [D-setup].
- **Casting** needs the phone, computer, and headset on the **same Wi-Fi network and band**, and logged into the **same Meta account** [M-cast].
  - From the headset: Navigator → *Quick controls* → *Cast* → *Mobile* / *Web* / nearby device → *Next*. Stop with *Cast* → *Stop casting* [M-cast].
  - From the phone: Meta Horizon app → *Devices* → your headset → *Cast*. Bluetooth must be on [M-cast].
  - Browser: log in at `oculus.com/casting` with the same account [M-cast-ts].
- Guest, corporate, hotel, or airport networks **often block casting** [M-cast-ts]. On venue Wi-Fi, put the headset and the demo laptop/phone on your own hotspot (option B above). Casting also uses more battery [M-cast-ts].
- For a big screen, cast to a laptop browser and mirror that laptop to the projector.
- Alternative that skips casting entirely: MVDH casts the headset display to the computer over its own connection [D-mvdh].

### 2.5 Quest ↔ PC (Link / Air Link): mostly skip it

Link and Air Link are **Windows-only** [M-link][D-before]. Link needs a USB 3.2 cable (≥ 5 Gbps). Air Link
needs 5 GHz Wi-Fi [M-link]. They let you press Play in Unity and see the scene in the headset, but
Megamind's value is the **passthrough camera feed**, and Play mode does not give you a reliable
path to that. Meta's PCA docs require a device build, and the XR Simulator does not support PCA [D-before].
The headset README says Horizon Link 2.1+ can work, which is not verified here. **Use
*Megamind → 4. Build and Run on headset* instead.** Link is only useful for checking HUD/panel layout on a
Windows PC.

### 2.6 Power

- Full charge takes about 2.5 h on the included 18 W adapter [M-charge]. Meta has not published a runtime for Megamind's workload (passthrough + camera + Wi-Fi uploads + JPEG encoding). **Assume less than an hour and time it on the night** (verify on device). Casting drains the battery faster [M-cast-ts].
- **Charging while tethered:** a laptop USB port usually supplies far less than 18 W, so the battery can still drain while you develop over USB (verify on device). For long sessions, use the 18 W wall charger or a USB-C PD power bank of at least 18 W, and use wireless adb (2.2) for logs.
- **Accessories:** Meta sells charging docks [M-docks] and the **Meta Quest 3 Elite Strap with Battery** [M-elite]. A power bank in your pocket with a long USB-C cable also works.
- Keep the dock's magnetic pins dry [M-battery]. If the headset gets hot, stop and let it cool (see 7).

### 2.7 Decision table: symptom → link → 2-minute check

| Symptom | Likely link at fault | 2-minute check |
|---|---|---|
| `adb devices` is empty | USB cable/driver (2.1) | Swap cable/port. Windows: Oculus ADB driver. Check the Developer Mode toggle in the phone app. |
| `adb devices` shows `unauthorized` | USB trust prompt (2.1) | Put the headset on and accept the *Allow USB debugging* prompt. Otherwise restart the headset and replug. |
| `adb devices` shows `offline` | adb server (2.1) | `adb kill-server && adb devices`. Quit Unity/MVDH's adb. |
| `adb connect <ip>:5555` times out | Wireless adb / LAN (2.2) | Same Wi-Fi? Re-run `adb tcpip 5555` over USB. Venue Wi-Fi isolation → use a hotspot. |
| HUD `NET ERR: ... Cannot connect to destination host` and `last net` | Quest → server (2.3) | Can you open `<url>/v1/health` in the Quest Browser? If not: is the server bound to `0.0.0.0`, is the firewall open, is Tailscale connected on the Quest? |
| Laptop can reach `/v1/health` but phone/Quest can't | Firewall or client isolation (2.3) | Try from the phone. If the phone also fails, switch to a hotspot or Tailscale. |
| HUD shows the wrong URL or `url:asset` with an old IP | Config (2.3, 5) | `adb logcat -s Unity \| grep "\[Megamind\] config"` shows each value and its source. Force-stop, then relaunch with `--es serverBaseUrl <url>` (section 5). A pushed `megamind.json` is likely unreadable on Android 14 (`could not read megamind.json: Access ... is denied`). |
| `https://` URL fails but `http://` works (or the reverse) | TLS / tunnel (2.3) | `curl -v <url>/v1/health` from the laptop. Funnel DNS can take 10 min [T-funnel]. Check that the URL has no `:8000` and no trailing path. |
| Cast list is empty or casting fails | Phone ↔ Quest Wi-Fi (2.4) | Same Wi-Fi, same band, same Meta account? Venue network blocks casting → hotspot [M-cast-ts]. |
| Battery drains while plugged into the laptop | Power (2.6) | Switch to the 18 W wall charger or a PD bank. Use wireless adb. |
| App relaunches or freezes when you take the headset off | Proximity sensor / sleep (7) | Keep it on your head, or turn off the proximity sensor in MVDH [D-mvdh]. |

---

## 3. Developer setup

1. **Pairing:** each headset is paired with a phone in the **Meta Horizon** app, and that app holds the Developer Mode toggle [D-setup]. Developer mode requires a verified developer account [D-setup].
2. **Verify developer mode:** phone app → headset → *Headset Settings* → *Developer Mode* should be on. In the headset, the *Settings* → *Developer* section should exist (verify on device) [D-setup].
3. **Install adb** (any OS): Android SDK Platform-Tools [A-pt]. Check with `adb version`.
4. **Optional: MVDH** (Windows/macOS). It provides a device manager, logs, screenshots/video, APK deploy by drag-and-drop, cast to computer, and toggles to disable the proximity sensor and boundary for development [D-mvdh][D-mvdh-start].
5. **Connect:** follow [2.1](#21-quest--laptop-over-usb-c-adb) and finish with `adb devices` showing `device`.

---

## 4. Getting Megamind on the headset

Exact versions (from the repo): **Unity `6000.0.66f2`** (`headset/Megamind/ProjectSettings/ProjectVersion.txt`),
Meta XR Core SDK **85.0.0**, MR Utility Kit **85.0.0**, OpenXR 1.15.1 (`Packages/manifest.json`). Package
`com.megamind.quest`, ARM64, IL2CPP, min SDK 32. Full details are in [`headset/README.md` → Build + sideload](../headset/README.md#build--sideload-step-by-step).
None of this was run here, so all Unity steps are **(verify on device)**.

**Shortcut: no Unity at all.** Download a prebuilt APK from GitHub (Actions → *headset-apk* → newest green run →
*Artifacts*, or a `headset-v*` [Release](https://github.com/debojoti-soumya/megamind/releases); details and the
one-time licence setup in [`headset/README.md` → Get the APK](../headset/README.md#get-the-apk-no-unity-needed)), then:
```sh
gh run download -R debojoti-soumya/megamind -n megamind.apk     # or unzip the artifact you downloaded
adb devices                                                     # headset must show as "device" (2.1)
adb uninstall com.megamind.quest                                # only if a build from another machine is installed
adb install -r megamind.apk
adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity
```
Then point it at your server with the launch-intent extras in section 5 and continue at step 7 below. The capture-only native
app is `megamind-native.apk` (`adb install -r megamind-native.apk`, package `ai.megamind.quest`, see
[`headset-android/README.md`](../headset-android/README.md)). Against a token-protected server (option F, Modal)
the native app takes the token in Settings → *API token* or `adb shell am start -n ai.megamind.quest/.MainActivity
--es api_token <token>`; a wrong or missing token pauses uploads with "check API token" instead of retrying
(verify on device). No CI-built APK has been installed on a headset yet
**(verify on device)**.


1. **Unity Hub** → *Installs* → *Install Editor* → `6000.0.66f2`. Add the **Android Build Support** module, including **OpenJDK** and **Android SDK & NDK Tools** [U-android].
2. Hub → *Add* → open the folder `headset/Megamind`. The first import downloads the Meta packages.
3. *File → Build Profiles* (Unity 6) → **Android** → *Switch Platform* [D-unity-build].
4. Open the Meta **Project Setup Tool** (*Window → Meta → Tools → Project Setup Tool*) and resolve the **required** issues on the Android tab [D-unity-setup][D-pca-unity].
5. Run the project's menu items in order (`Assets/Megamind/Editor/MegamindSceneBuilder.cs`):
   - **Megamind → 1. Create Capture Scene** (answer **Overwrite** if asked). Creates the scene, `MegamindConfig.asset`, and `CaptureSettings.asset`, adds the tracked hands (`OVRHandPrefab`) and the right controller/hand UI rays, and sets *Hand Tracking Support* = Controllers And Hands.
   - **Megamind → 1b. Add Chat Panel To Current Scene**. Adds the chat/settings panel and wires the hands and the ray into it. Re-run it after every *1.*, because *1.* replaces the scene.
   - **Megamind → Check Project**. Every line should read `OK`.
   - **Megamind → Select Config**. Set `Server Base Url` (see 2.3).
   - **Megamind → 2. Apply Android Player Settings**
   - **Megamind → 3. Build APK** → `Builds/megamind.apk`. Or run **Megamind → 4. Build and Run on headset** to build, install, and launch over USB. The first IL2CPP build takes about 10 min.
6. **Sideload** an APK someone else built:
   - adb: `adb install -r Builds/megamind.apk` (`-r` keeps the app data and permissions) [D-adb].
   - MVDH: *Device Manager* → *Apps* → drag in the APK, or use *Add Build* [D-deploy].
7. **Launch:** in the headset, Meta button → *Library* → **Unknown Sources** → **Megamind** [D-deploy]. From the laptop:
   ```sh
   adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity
   ```
   The manifest declares `UnityPlayerGameActivity` (Unity 6 GameActivity entry point, `androidApplicationEntry: 2`).
   Older notes used `UnityPlayerActivity`, which does not match the manifest (verify on device).
8. **Two permission prompts (one request by `MegamindPermissions`):** on first launch the app requests the passthrough-camera permission (`horizonos.permission.HEADSET_CAMERA`, for frames) and the microphone (`android.permission.RECORD_AUDIO`, for push-to-talk **and** ambient audio memory) in a single request through `OVRPermissionsRequester` [D-pca-unity]; Android asks for each permission separately, so expect **two prompts**, camera then microphone (verify on device: order and wording). Select **Allow** on both. The HUD then changes from `waiting for camera permission…` to `starting passthrough camera…` to `REC ...`, and its `mic:` line from `(waiting for mic permission)` to `(on -NN dB)`.
   - **If you hit Deny:** the app does not ask again in that session. Fix it either way:
     - Headset: *Settings → Apps → Megamind → Permissions* → allow Camera (verify on device, path from the headset README).
     - Laptop: `adb shell pm grant com.megamind.quest horizonos.permission.HEADSET_CAMERA`, then force-stop and relaunch (verify on device; runtime-permission behaviour per [D-pca-native]).
     - Microphone denied: same, with *Microphone* / `android.permission.RECORD_AUDIO` (verify on device). Without it push-to-talk and ambient audio are both off; frames still work.
9. **OS requirement:** the Passthrough Camera API needs **Quest 3/3S on Horizon OS v74+** [D-pca]. Check the version under Navigator → *Quick controls* → *Settings* → *General* → *Software update*, which shows the current version and offers an update when one is available [M-update]. On an old OS the HUD shows `Passthrough Camera API unsupported (Quest 3 + v74 required)`.

---

## 5. Pointing the app at the server

Pick the URL using [2.3](#23-quest--server-the-network-path). Then use **one** of:

- **No rebuild (adb, recommended):** relaunch the app with the values as launch-intent extras. The app saves them, so
  later launches from the Library keep them. Use the editor menu *Megamind → Push config to headset (adb)*, or:
  ```sh
  adb shell am force-stop com.megamind.quest
  adb shell am start -n com.megamind.quest/com.unity3d.player.UnityPlayerGameActivity \
    --es serverBaseUrl http://100.101.102.103:8000 --es deviceId quest3-1
  ```
  Force-stop first: the activity is `singleTask`, so `am start` on a running app doesn't re-read the extras (verify on device).
  Give each headset a unique `deviceId` (`quest3-1`, `quest3-2`, ...) so memories stay separate.
  For a server with `API_TOKEN` set (option F, Modal), add `--es apiToken <token>` with
  `--es serverBaseUrl https://<workspace>--megamind-web.modal.run`.
  The app then sends `Authorization: Bearer <token>` on uploads, chat, health, thumbnails, audio and purge. A 401 in
  logcat means the token is missing or wrong (verify on device).
  Ambient audio uses the same URL and `deviceId`. To switch it off without rebuilding, add `--es ambientAudio off`
  (`on` turns it back on; never passed = `CaptureSettings.ambientAudio`, default on).
  `--es <key> "''"` (empty) removes one saved value, `--ez resetConfig true` removes all of them.
  logcat prints `[Megamind] config: serverBaseUrl=<url> (intent) deviceId=<id> (intent) apiToken=(set, N chars) (intent) ambientAudio=on (asset)`.
  The token itself is never logged. Requires an APK that includes `RuntimeConfig.cs`.
- **`megamind.json` (unreliable on Android 14):** the app still reads `megamind.json` from its files dir *if it can*,
  with the same keys (`serverBaseUrl`, `deviceId`, `apiToken`, `ambientAudio`, plus `captureFps`). On a Quest 3 on
  Horizon OS / Android 14 a file written there with `adb shell 'echo ... > /sdcard/Android/data/com.megamind.quest/files/megamind.json'`
  was not readable by the app (`could not read megamind.json: Access ... is denied`), even after deleting the folder
  and letting the app recreate it. `adb push` writes it as the same shell user (not tested). `adb shell run-as` only works
  for debuggable builds, and the CI/`BuildApk` APK is a release build. Shared folders like `/sdcard/Download` aren't
  readable by this app under scoped storage. Use the intent extras instead.
- **Baked in (fallback that worked on Android 14):** *Megamind → Select Config* → `Server Base Url`, `Device Id` and
  `Api Token`, then rebuild. Don't commit a real token.
- **In-app settings panel** (PR #2 chat UI path): writes PlayerPrefs (URL and device id). It overrides `megamind.json` and the asset, and is itself replaced by the next intent-extra launch. *Reset* (or `--ez resetConfig true`) clears every saved value. `adb shell pm clear com.megamind.quest` also resets it, together with the permissions.

Confirm the result: logcat prints `[Megamind] capture: server=<url> device=<id> fps=1 url-source=<source>` (and the `[Megamind] config:` line above), and the HUD's last line shows the same.

---

## 6. Test runbook

Keep a log terminal open the whole time. Unity writes `Debug.Log` under the logcat tag **`Unity`** (Unity convention; verify on device), and
every Megamind line starts with `[Megamind]` (`-s`, `-c`, `> file` usage per [D-logcat]):
```sh
adb logcat -c                                  # clear old logs
adb logcat -s Unity | grep -i "\[megamind\]"   # -s filters by tag; grep filters the message
adb logcat > session.log                       # full log for crashes; search for "backtrace:"
```
On the server, watch the uvicorn output for `POST /v1/frames ... 200` and (ambient audio, every ~15 s while someone talks) `POST /v1/audio ... 200`.

| # | Step | Success looks like | Failure looks like → go to |
|---|---|---|---|
| 0 | Sanity: Meta's unmodified `CameraViewer` sample ([samples repo](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples)) | Shows the camera image | No image → OS/permission problem, not Megamind |
| 1 | `adb devices` | `<serial>  device` | 2.1 table |
| 2 | Server health from the laptop and the Quest Browser | `{"status":"ok","store_count":N,...}` | 2.3 |
| 3 | Install + launch | App is under *Unknown Sources*. The REC dot and HUD appear | 7: "not in library", "App not installed" |
| 4 | Permissions (verify on device) | Two prompts: camera, then microphone. Log: `[Megamind] permissions: camera=False mic=False (requested 2)` then `[Megamind] permission granted: ...` for each. Relaunches show `camera=True mic=True` and no prompt | Two *camera* dialogs → see headset README troubleshooting. HUD stuck on `waiting for camera permission…` or `mic: … (waiting for mic permission)` → 4.8 |
| 5 | Config | `[Megamind] capture: server=http://... device=quest3-1 fps=1 url-source=...` with the URL you expect | Wrong URL → 5 |
| 6 | Camera | `[Megamind] camera: os=... device=... requested=1280x960 supported=[...]` then **`[Megamind] camera playing: 1280x960 maxFps=30`**. HUD: `REC 1 fps  1280x960` | `Passthrough Camera API unsupported...` → OS < v74. `no PassthroughCameraAccess in scene` → re-run *1. Create Capture Scene*. If `supported=[...]` lacks `1280x960`, pick one from the list for `CaptureSettings.cameraResolution` |
| 7 | Upload | HUD `sent` and `accepted` rise about once per second. `last ok` with a latency in ms and the size in KB | `last net` + `NET ERR: ...` and the log line `[Megamind] frame upload failed (Nx): ... url=...` → 2.7 |
| 8 | Server filter | Some `last duplicate` / `too_soon` while still is normal (the filter keeps roughly 10–30% at 1 fps) | Mostly `dark`/`blurry` → 7 |
| 9 | Pause | A/X (or middle pinch with hand tracking, verify on device) → `capture PAUSED (press A/X or middle-pinch)`. Press again to resume | Pinches do nothing → hand tracking off on the headset (section 1), or the scene predates the hands: re-run *1.* then *1b.* |
| 10 | Image check | Open a stored thumbnail (`/v1/thumbs/{id}.jpg`, see [api.md](api.md)). Colours should be right, upright, not mirrored | Blue skin → reboot the headset [D-pca] |
| 11 | Ambient audio (verify on device) | Log `[Megamind] ambient audio: mic started 16000 Hz x1, 15 s chunks`. Talk near the headset: HUD `mic: chunks sent N  accepted M  (on -NN dB)` rises about every 15 s; in silence only `silent N` rises (chunks below −50 dBFS are not sent). Hold the trigger: state shows `push-to-talk`, then back to `on`. A/X: `paused` | `waiting for mic permission` → 4.8. `ERR …` → same network checks as step 7 (2.7). Only `silent` rises while talking → 7 |
| 12 | "What did I hear?" (verify on device) | After ~30 s of talking, hold the trigger and ask *"what did I hear?"*. The answer cites a speech memory (`Speech heard nearby: "…"`; with mock STT the words are `[mock transcript]`, real words need `AUDIO_STT_PROVIDER=openai` or `faster-whisper` on the server). Laptop check: `curl 'http://<server>:8000/v1/memories/recent?kind=transcript&k=5'` | `not_found` / only frames → no `POST /v1/audio 200` yet (step 11), or a different `deviceId` than the chat uses |

Expected camera numbers: the app requests **1280×960** with a max of **30 fps** from the camera (`CaptureSettings.cs`).
It then samples and uploads only **1 frame/s**, as a full-size 1280×960 JPEG at quality 80. 1280×960 is
one of the resolutions Meta documents for PCA [D-pca-unity]. Meta documents about 20–40 ms latency [D-pca].

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| **Black passthrough / black camera image** | Is passthrough on? Re-run *Megamind → Check Project*, because PCA requires passthrough to be enabled [D-pca]. Is the room bright enough (≥ 50 lux [M-passthrough])? Make sure nothing is covering the front cameras. If the HUD shows `0x0`, the camera has not started. Check step 6 of the runbook. |
| **No permission prompt** | It was already answered. Run `adb shell dumpsys package com.megamind.quest \| grep -i camera` to check (verify on device). The app asks once per session only (`MegamindPermissions.cs`). Grant it manually (4.8). After regenerating the manifest, re-add `HEADSET_CAMERA` [D-pca-unity]. |
| **App not in the library** | Look under *Library → Unknown Sources*, not the main grid [D-deploy]. Check that `adb shell pm list packages \| grep megamind` prints `package:com.megamind.quest`. |
| **`App not installed` / `INSTALL_FAILED_UPDATE_INCOMPATIBLE`** | A build signed with a different key (another teammate's debug keystore) is installed. Run `adb uninstall com.megamind.quest` and reinstall. This deletes the app data (including the saved server URL / token from intent extras or the settings panel) and permission grants. |
| **Boundary / "Guardian" blocks the view or pauses capture** | Megamind is boundaryless [D-boundaryless], so the boundary should not appear (verify on device). If it still does, or logcat shows `app backgrounded (OnApplicationPause): pausing capture` / the HUD `bg N` rises as you cross it: check *Megamind → Check Project* reports `BOUNDARYLESS_APP`, and that you launched the installed APK, not a Link session [D-boundaryless]. Fallbacks: MVDH can disable the boundary system [D-mvdh]; the headset's *Developer* settings may have a boundary toggle (verify on device); or draw one large roomscale boundary covering the whole walk [M-boundary]. `app focus lost … capture keeps running` in logcat is normal: a system overlay had focus and capture continued [D-focus]. |
| **Headset falls asleep or the app restarts when removed** | The proximity sensor suspends apps when the headset is off a head. Disable it in MVDH [D-mvdh], or keep the headset on. The commonly used adb broadcast `adb shell am broadcast -a com.oculus.vrpowermanager.prox_close` is **not** in the Meta docs checked here (verify on device). Power/sleep timeout settings are in the headset's *Settings* (exact path not verified). `SensorLockActivity` in logcat means it was idle: put it on, press power once, then relaunch. |
| **Overheating / throttling** | Stop casting and recording, and take it off the charger and out of direct sun. Let it cool before continuing. This advice is general and not from a Meta doc. |
| **Frames rejected (`accepted` flat, `last <reason>`)** | Reasons come from `server/megamind/ingest/filter.py` ([api.md](api.md)). `dark` means more light. `blurry` means move your head slower; the camera auto-exposes slowly after start. `duplicate` / `too_soon` are expected when the scene is static, so walk around. `budget` means the per-device kept-frames cap was hit (`filter_max_per_min`). `client:motion` / `client:backlog` / `client:backoff` (server unreachable, retrying with exponential backoff up to 30 s) are skipped on the headset and never sent. |
| **HUD `mic:` shows only `silent N` while people talk** | Every 15 s chunk was quieter than `CaptureSettings.audioSilenceDb` (−50 dBFS): mic muted by the system (headset off-head) or speech too far away. Lower the gate (e.g. −60, or −120 to send everything) and rebuild (verify on device). |
| **Colours wrong (blue skin)** | Reboot the headset. This is a known issue on v74 after a camera app's first install [D-pca]. |
| **Wipe memories** | `curl -X DELETE http://<server>:8000/v1/memories` wipes **all** devices. Add `?device_id=quest3-1` to wipe one device, or `&since=<ts>` to wipe from a time onwards ([api.md](api.md)). |

---

## 8. Memory Theater (/xr) in the Quest browser

A WebXR view of your memories that needs no Unity build: open `/xr` on the server in the Quest browser. A large
screen replays the cited moment (`replay_url`, or the cited thumbnail when no replay is buffered), the answer sits on
a panel to the left, suggested questions are buttons on the right, and the newest memories form an arc of cards
below (cited ones glow). Point with a controller and pull the trigger, or pinch with hand tracking, to select. Tap
the screen to pause, resume or replay. On a laptop the same page is an inline 3D preview: drag to look, click to
select, type in the bar to ask.

**Immersive mode needs a secure context** (the browser only exposes WebXR on `https://` or `localhost`). Plain
`http://<laptop-ip>:8000/xr` loads the inline preview but has no *Enter passthrough* button. Options:

| Option | How | Notes |
|---|---|---|
| USB (simplest) | `adb reverse tcp:8000 tcp:8000`, then open `http://localhost:8000/xr` in the Quest browser | Port-forwards the headset's localhost to the laptop (see 2.1). Redo after reconnecting the cable. |
| Tailscale serve | open `https://<machine>.<tailnet>.ts.net/xr` (option D in 2.3) | Real HTTPS, tailnet only. |
| Modal | `https://<workspace>--megamind-web.modal.run/xr?token=<token>` (option F in 2.3) | Real HTTPS from any network. The `?token=` sets a cookie, so the page's own requests are authorized (verify on device). |
| Public HTTPS tunnel | Funnel / ngrok URL + `/xr` (option E in 2.3) | Set `API_TOKEN` and open `/xr?token=<token>`; without it anyone with the URL can read or wipe memories. |

Then press **Enter passthrough** (`immersive-ar`; falls back to **Enter VR** / `immersive-vr` where AR is
unsupported) and allow the permission prompt. *Exit* or the Meta button leaves immersive mode.

- **Voice questions.** The round mic button next to the ask box: hold it and release to send, or tap once to start and
  tap again to stop (Space / Enter toggles it from the keyboard). In immersive mode use the *Ask by voice (tap)* button
  above the suggestions: trigger once to start, again to send. The clip goes to `POST /v1/chat` as 16 kHz WAV and is
  transcribed by the server's configured STT (`STT_PROVIDER`, the same as the /ui push-to-talk; `mock` answers
  "[mock transcript]"). The mic is released after each question. The mic also needs the secure context above: on
  `http://<laptop-ip>:8000/xr` the button explains that instead of recording.
- `?device=quest3-1` scopes cards, chat and replays to one device; `?session=<id>` keeps one chat session across reloads.
- Replays only exist for the last `REPLAY_WINDOW_S` (15 min) of frames; older moments show the still thumbnail.
- The page loads three.js from `cdn.jsdelivr.net`, so the headset needs internet (or the page shows "Failed to start").
- It works alongside the Unity app: the app captures frames, the browser page browses them.

---

## Sources

Meta developer docs (all accessed while writing this guide):
- [D-setup] Device setup, developer mode, USB debugging, Windows driver: https://developers.meta.com/vr/documentation/native/android/mobile-device-setup/
- [D-adb] adb on Quest (devices, wireless, install, `ip route`): https://developers.meta.com/vr/documentation/native/android/ts-adb/
- [D-logcat] Logcat: https://developers.meta.com/vr/documentation/native/android/ts-logcat/
- [D-mvdh] Meta VR Developer Hub overview: https://developers.meta.com/vr/documentation/unity/ts-mqdh/
- [D-mvdh-start] MVDH getting started: https://developers.meta.com/vr/documentation/unity/ts-mqdh-getting-started/
- [D-mvdh-setup] MVDH device setup: https://developers.meta.com/vr/documentation/unity/ts-mqdh-device-setup/
- [D-deploy] Deploy a build with MVDH (Unknown Sources): https://developers.meta.com/vr/documentation/unity/ts-odh-deploy-build/
- [D-pca] Passthrough Camera API overview (v74+, Quest 3/3S, known issues): https://developers.meta.com/vr/documentation/unity/unity-pca-overview/
- [D-pca-unity] PCA Unity docs (resolutions, permissions, manifest): https://developers.meta.com/vr/documentation/unity/unity-pca-documentation/
- [D-pca-native] PCA permissions (native): https://developers.meta.com/vr/documentation/native/android/pca-native-documentation/
- [D-unity-setup] Unity project setup: https://developers.meta.com/vr/documentation/unity/unity-project-setup/
- [D-unity-build] Unity build configuration: https://developers.meta.com/vr/documentation/unity/unity-build/
- [D-hw] Unity device setup (data cable): https://developers.meta.com/vr/documentation/unity/unity-env-device-setup/
- [D-boundaryless] Boundaryless Mode for Mixed Reality in Unity (`BOUNDARYLESS_APP`, Boundary API, Link/6DOF limits): https://developers.meta.com/horizon/documentation/unity/unity-boundaryless/
- [D-boundaryless-bp] Boundaryless and contextual-boundaryless safety best practices: https://developers.meta.com/horizon/design/boundaryless-best-practices/
- [D-focus] Focus awareness (input focus, `OVRManager.InputFocusLost`): https://developers.meta.com/horizon/documentation/unity/unity-focus-awareness/
- [D-before] Before you begin (Link Windows-only, simulator): https://developers.meta.com/vr/documentation/unity/unity-before-you-begin/

Meta Quest Help Center:
- [M-update] Software update: https://www.meta.com/help/quest/540602136930952/
- [M-passthrough] Passthrough: https://www.meta.com/help/quest/617966963105359/
- [M-hands] Hand tracking: https://www.meta.com/help/quest/290147772643252/
- [M-boundary] Boundary: https://www.meta.com/help/quest/463504908043519/
- [M-wifi] Wi-Fi: https://www.meta.com/help/quest/1816744325172615/ and troubleshooting https://www.meta.com/help/quest/517103729284781/
- [M-cast] Casting: https://www.meta.com/help/quest/192719842695017/
- [M-cast-ts] Casting troubleshooting: https://www.meta.com/help/quest/214966974548157/
- [M-record] Record video: https://www.meta.com/help/quest/516228089650875/
- [M-charge] Charging: https://www.meta.com/help/quest/3482884435359892/
- [M-battery] Battery life: https://www.meta.com/help/quest/1275691429562711/
- [M-docks] Charging docks: https://www.meta.com/help/quest/1571448446949054/
- [M-elite] Elite Strap with Battery: https://www.meta.com/help/quest/556883619437824/
- [M-link] Link / Air Link: https://www.meta.com/help/quest/509273027107091/

Android / Tailscale:
- [U-android] Unity Android environment setup: https://docs.unity3d.com/6000.0/Documentation/Manual/android-sdksetup.html
- [A-pt] Platform-Tools: https://developer.android.com/tools/releases/platform-tools
- [A-device] Run on a hardware device (Linux udev, plugdev): https://developer.android.com/studio/run/device
- [A-cleartext] Network security config / cleartext: https://developer.android.com/privacy-and-security/security-config
- [T-serve] Tailscale Serve: https://tailscale.com/kb/1312/serve
- [T-funnel] Tailscale Funnel: https://tailscale.com/kb/1223/funnel
- Tailscale Android APKs: https://pkgs.tailscale.com/stable/#android
