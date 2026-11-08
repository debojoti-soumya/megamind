"""Demo window in one command: keep one Megamind container warm on Modal, then flip back. No rebuild, no redeploy.

    python deploy/keep_warm.py on            # min_containers=1 now, boot it, wait until warmup is done
    python deploy/keep_warm.py on --hours 3  # same, then flip back to 0 after 3 h (keep this terminal open)
    python deploy/keep_warm.py off           # back to min_containers=0 (idle cost 0 again)
    python deploy/keep_warm.py status        # containers running + /v1/health (warmup state)

Uses `Function.update_autoscaler` on the deployed `megamind` app (needs `pip install modal` and `modal token new`).
The override lasts until `off` or the next `modal deploy`, which resets it to the decorator's `min_containers`
(0 unless deployed with MEGAMIND_MIN_CONTAINERS=1), so forgetting `off` costs at most until the next deploy.
A warm container costs ~1 vCPU-hour per hour. `/v1/health` needs the API token: set MEGAMIND_API_TOKEN (or
--token); without it the script still flips the setting and just skips the health wait.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

APP, FUNCTION = "megamind", "web"


def _function(env: str | None):
    import modal

    return modal.Function.from_name(APP, FUNCTION, environment_name=env)


def _health(url: str, token: str, timeout: float = 30.0) -> dict | None:
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/health", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  (our own https URL)
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  health: {e}")
        return None


def wait_warm(url: str, token: str, limit_s: float) -> bool:
    """Poll /v1/health (this request also boots the container) until warmup reports done."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        h = _health(url, token)
        if h is not None:
            state = h.get("warmup", "(server predates warmup)")
            t = time.monotonic() - t0
            print(f"  {t:5.1f} s  status={h.get('status')} store={h.get('store_count')} warmup={state}")
            if state != "running":
                return True
        time.sleep(2)
    print(f"  still not warm after {limit_s:.0f} s")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["on", "off", "status"])
    ap.add_argument("--hours", type=float, default=0.0, help="with on: flip back to 0 after this long (stays running)")
    ap.add_argument("--token", default=os.environ.get("MEGAMIND_API_TOKEN", ""), help="API token for /v1/health")
    ap.add_argument("--url", default=os.environ.get("MEGAMIND_URL", ""), help="default: the deployed web URL")
    ap.add_argument("--env", default=None, help="Modal environment (default: the profile's)")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for warmup after on")
    a = ap.parse_args()

    fn = _function(a.env)
    url = a.url or fn.get_web_url()
    if a.action in ("on", "off"):
        settings = fn.update_autoscaler(min_containers=1 if a.action == "on" else 0)
        print(f"{APP}/{FUNCTION}: min_containers={settings.min_containers} ({url})")
    stats = fn.get_current_stats()
    print(f"  containers running: {stats.num_total_runners}, queued inputs: {stats.backlog}")
    if a.action == "off":
        return 0
    if not a.token:
        print("  (set MEGAMIND_API_TOKEN or --token to check /v1/health and wait for warmup)")
    elif a.action == "on":
        wait_warm(url, a.token, a.wait)
    else:
        h = _health(url, a.token)
        print(json.dumps(h, indent=2) if h else "  health: unreachable")
    if a.action == "on" and a.hours > 0:
        print(f"  keeping warm for {a.hours:g} h; Ctrl-C leaves it on (run `off` later)")
        try:
            time.sleep(a.hours * 3600)
        except KeyboardInterrupt:
            print("  left on")
            return 0
        fn.update_autoscaler(min_containers=0)
        print(f"{APP}/{FUNCTION}: min_containers=0 (demo window over)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
