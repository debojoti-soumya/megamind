#!/usr/bin/env sh
# Meta XR Core SDK 85.0.0 does not compile in a Linux editor: Editor/MetaXRSimulator/Installer.cs only declares
# `downloadedInstallerPath` under UNITY_EDITOR_WIN / UNITY_EDITOR_OSX (error CS0103). The XR Simulator is not
# available on Linux anyway, so give the variable a value there. Re-run after any package re-resolve
# (Library/PackageCache is regenerated). Harmless on Windows/macOS. Usage: sh headset/tools/patch-metaxr-linux.sh
set -e
cd "$(dirname "$0")/../Megamind"
for f in Library/PackageCache/com.meta.xr.sdk.core@*/Editor/MetaXRSimulator/Installer.cs; do
  [ -f "$f" ] || continue
  grep -q 'meta_xr_simulator.bin' "$f" && { echo "already patched: $f"; continue; }
  python3 - "$f" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
old = '''#elif UNITY_EDITOR_OSX
            var downloadedInstallerPath =
                            Path.Combine(XRSimConstants.DownloadFolderPath, $"meta_xr_simulator.dmg");
#endif'''
new = old.replace('#endif', '''#else
            var downloadedInstallerPath =
                            Path.Combine(XRSimConstants.DownloadFolderPath, $"meta_xr_simulator.bin");
#endif''')
assert old in s, "unexpected Installer.cs layout"
open(p, 'w').write(s.replace(old, new, 1))
print("patched:", p)
PY
done
