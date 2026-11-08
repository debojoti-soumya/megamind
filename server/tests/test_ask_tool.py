"""tools/ask.py: default --device-id = most recent device from GET /v1/devices (httpx.MockTransport, no server)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import ask  # noqa: E402

CHAT = {"transcript": "keys?", "answer": "On the counter.", "cited": [], "tool_calls": []}


def _transport(devices: dict | None, status: int = 200, chat: dict = CHAT):
    sent: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/devices":
            return httpx.Response(status, json=devices)
        assert req.url.path == "/v1/chat"
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=chat)

    return httpx.MockTransport(handler), sent


DEVICES = {
    "devices": [
        {"device_id": "quest3-1", "count": 9, "last_ts": 2.0, "last_frame_ts": 2.0, "last_audio_ts": None},
        {"device_id": "browser-x", "count": 1, "last_ts": 1.0, "last_frame_ts": 1.0, "last_audio_ts": None},
    ]
}


def test_defaults_to_most_recent_device(capsys):
    transport, sent = _transport(DEVICES)
    ask.main(["keys?"], transport=transport)
    assert sent[0]["device_id"] == "quest3-1"
    assert "device: quest3-1 (most recent, from /v1/devices)" in capsys.readouterr().out


def test_explicit_device_skips_lookup_and_empty_means_all(capsys):
    transport, sent = _transport(None, status=500)  # /v1/devices must not be called
    ask.main(["keys?", "--device-id", "browser-x"], transport=transport)
    ask.main(["keys?", "--device-id", ""], transport=transport)
    assert [s["device_id"] for s in sent] == ["browser-x", None]
    assert "device:" not in capsys.readouterr().out


@pytest.mark.parametrize("devices,status", [({"devices": []}, 200), ({"detail": "Not Found"}, 404)])
def test_no_devices_or_old_server_searches_all(capsys, devices, status):
    transport, sent = _transport(devices, status)
    ask.main(["keys?"], transport=transport)
    assert sent[0]["device_id"] is None
    assert "device: all devices" in capsys.readouterr().out


def test_prints_server_warning(capsys):
    warning = "device 'browser-x' has no memories; active devices: quest3-1"
    transport, _ = _transport(DEVICES, chat={**CHAT, "warning": warning})
    ask.main(["keys?", "--device-id", "browser-x"], transport=transport)
    assert f"warning: {warning}" in capsys.readouterr().out
