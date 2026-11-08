"""3D from memory: geometry + file formats (pure numpy) and the endpoint with fake models."""

import json
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from megamind.agent import model3d
from megamind.agent.model3d import Mesh, relief_mesh, strip_3d_framing, to_glb, to_splat_ply, wants_3d


def _blob(h=40, w=60):
    v, u = np.mgrid[0:h, 0:w]
    mask = (u - w / 2) ** 2 / (w / 3) ** 2 + (v - h / 2) ** 2 / (h / 3) ** 2 < 1
    disp = np.where(mask, 1.0 - 0.3 * ((u - w / 2) ** 2 + (v - h / 2) ** 2) / (w * w / 4), 0.0)  # a dome
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = 200
    return rgb, mask, disp


def test_relief_mesh_is_the_masked_object_on_the_floor():
    rgb, mask, disp = _blob()
    m = relief_mesh(rgb, mask, disp, size_m=0.4)
    assert len(m.positions) < mask.sum() and len(m.faces) > 100  # eroded silhouette, background excluded
    assert m.faces.max() < len(m.positions)
    lo, hi = m.positions.min(0), m.positions.max(0)
    assert abs(lo[1]) < 1e-6 and abs((hi - lo).max() - 0.4) < 1e-5  # sits on y=0, longest side 0.4 m
    assert abs(lo[0] + hi[0]) < 1e-5 and abs(lo[2] + hi[2]) < 1e-5  # centred
    assert (m.colors[:, 0] == 200).all()


def test_relief_mesh_rejects_an_empty_mask():
    rgb, mask, disp = _blob()
    with pytest.raises(ValueError):
        relief_mesh(rgb, np.zeros_like(mask), disp)


def test_glb_is_valid_gltf():
    m = relief_mesh(*_blob())
    data = to_glb(m)
    magic, version, total = struct.unpack("<4sII", data[:12])
    assert (magic, version, total) == (b"glTF", 2, len(data))
    jlen, jtype = struct.unpack("<II", data[12:20])
    assert jtype == 0x4E4F534A
    doc = json.loads(data[20 : 20 + jlen])
    blen, btype = struct.unpack("<II", data[20 + jlen : 28 + jlen])
    assert btype == 0x004E4942 and blen >= doc["buffers"][0]["byteLength"] and blen % 4 == 0
    pos, col, ind = doc["accessors"]
    assert pos["count"] == col["count"] == len(m.positions) and ind["count"] == 3 * len(m.faces)
    assert doc["materials"][0]["extensions"] == {"KHR_materials_unlit": {}}
    end = max(v["byteOffset"] + v["byteLength"] for v in doc["bufferViews"])
    assert end == doc["buffers"][0]["byteLength"]


def test_splat_ply_header_and_payload():
    m = Mesh(np.zeros((5, 3), np.float32), np.full((5, 3), 255, np.uint8), np.array([[0, 1, 2]], np.uint32))
    data = to_splat_ply(m)
    head, body = data.split(b"end_header\n", 1)
    assert b"element vertex 5" in head and b"property float f_dc_0" in head and b"property float rot_3" in head
    assert len(body) == 5 * 14 * 4
    row = np.frombuffer(body, "<f4").reshape(5, 14)[0]
    assert row[3] == pytest.approx((1 - 0.5) / 0.28209479) and row[10] == 1.0


@pytest.mark.parametrize(
    ("q", "subject"),
    [
        ("make a 3D model of my mug", "my mug"),
        ("can you create a 3d model of the red car?", "the red car"),
        ("show me the dog in 3d", "show me the dog"),
        ("gaussian splat of my keyboard", "my keyboard"),
        ("hologram of the giraffe", "the giraffe"),
        ("my mug", "my mug"),
        ("show me my mug in 3D from yesterday", "show me my mug from yesterday"),
    ],
)
def test_strip_3d_framing(q, subject):
    assert strip_3d_framing(q) == subject


def test_wants_3d():
    assert wants_3d("make a 3D model of my mug") and wants_3d("splat of the car")
    assert not wants_3d("pictures of my dog") and not wants_3d("where did I leave my keys")


class FakeModels(model3d.LocalModels):
    """Detector finds the box only in the 2nd photo; SAM returns an ellipse; depth a dome."""

    calls: list = []

    @staticmethod
    def missing():
        return None

    def detect(self, img, prompt):
        FakeModels.calls.append(("detect", prompt, img.size))
        if img.size == (64, 48):
            return [8.0, 8.0, 56.0, 40.0], 0.8
        return [0.0, 0.0, 10.0, 10.0], 0.0

    def segment(self, img, box):
        w, h = img.size
        v, u = np.mgrid[0:h, 0:w]
        return (u - w / 2) ** 2 / (w / 3) ** 2 + (v - h / 2) ** 2 / (h / 3) ** 2 < 1

    def disparity(self, img):
        w, h = img.size
        v, u = np.mgrid[0:h, 0:w]
        return 1.0 - ((u - w / 2) ** 2 + (v - h / 2) ** 2) / (w * w)


@pytest.fixture
def app3d(tmp_path, monkeypatch):
    import io

    from PIL import Image

    from megamind.agent import collage
    from megamind.agent.smart_montage import Pick
    from megamind.models import MemoryToken

    def jpeg(size):
        buf = io.BytesIO()
        Image.new("RGB", size, (120, 60, 30)).save(buf, format="JPEG")
        return buf.getvalue()

    toks = [
        MemoryToken(
            id=f"t{i}",
            device_id="d",
            ts=1731000000.0 + i,
            caption="a mug",
            thumb_path=str(tmp_path / f"{i}.jpg"),
            frame_hash=f"{i:016x}",
        )
        for i in range(2)
    ]
    for t, size in zip(toks, [(80, 60), (64, 48)], strict=True):
        open(t.thumb_path, "wb").write(jpeg(size))

    async def fake_find(st, query, t0=None, t1=None, device_id=None):
        if "elephant" in query:
            raise collage.CollageError(404, "I couldn't find any photos of elephants in your memories.")
        return collage.Found(
            None if not query else object(), query, query.title(), toks, [Pick(t, "", 0.5, False) for t in toks]
        )

    monkeypatch.setattr(model3d, "find_photos", fake_find)
    monkeypatch.setattr(model3d, "LocalModels", FakeModels)
    from fastapi import FastAPI

    app = FastAPI()

    class S:
        thumb_dir = str(tmp_path / "thumbs")
        wearer_tz = "UTC"

    app.state.settings = S()
    model3d.install(app, S(), model3d.Model3DSettings(cache_dir=tmp_path / "m3d", max_side=64))
    FakeModels.calls = []
    return app


def test_model3d_endpoint_picks_the_detected_frame_and_serves_glb(app3d):
    c = TestClient(app3d)
    r = c.post("/v1/model3d", json={"query": "make a 3D model of my mug"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_id"] == "t1" and body["detection_score"] == 0.8  # the frame the detector was sure of
    assert body["format"] == "glb" and body["kind"] == "mesh" and body["full_3d"] is False
    assert body["viewer_url"] == f"/ui/3d?m={body['id']}&f=glb" and "side you saw" in body["narration"]
    assert [c[1] for c in FakeModels.calls] == ["mug", "mug"]
    g = c.get(body["url"])
    assert g.status_code == 200 and g.content[:4] == b"glTF" and g.headers["content-type"] == "model/gltf-binary"
    assert c.head(body["url"]).status_code == 200
    again = c.post("/v1/model3d", json={"query": "make a 3D model of my mug"}).json()
    assert again["id"] == body["id"] and len(FakeModels.calls) == 4  # cached: no second lift
    app3d.state.model3d_cache.clear()
    assert c.get(body["url"]).status_code == 404


def test_model3d_absent_subject_and_missing_deps(app3d, monkeypatch):
    c = TestClient(app3d)
    r = c.post("/v1/model3d", json={"query": "3d model of an elephant"})
    assert r.status_code == 404 and "elephants" in r.json()["detail"]
    monkeypatch.setattr(FakeModels, "missing", staticmethod(lambda: "3D needs torch"))
    r = c.post("/v1/model3d", json={"query": "3d model of my mug"})
    assert r.status_code == 503 and "torch" in r.json()["detail"]


def test_model3d_splat_format(app3d):
    app3d.state.model3d_settings.local_format = "splat"
    c = TestClient(app3d)
    body = c.post("/v1/model3d", json={"query": "splat of my mug"}).json()
    assert body["format"] == "ply" and body["kind"] == "splat" and body["viewer_url"].endswith("&f=ply")
    assert c.get(body["url"]).content.startswith(b"ply\nformat binary_little_endian")


def test_model3d_remote_backend(app3d, monkeypatch):
    import httpx

    cfg = app3d.state.model3d_settings
    cfg.backend, cfg.remote_url, cfg.remote_token = "remote", "http://gpu.example/sam3d", "tok"
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = req.content
        return httpx.Response(200, content=b"ply\nformat binary_little_endian 1.0\nproperty float f_dc_0\nend_header\n")

    real = httpx.AsyncClient
    monkeypatch.setattr(model3d.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    body = TestClient(app3d).post("/v1/model3d", json={"query": "3d model of my mug"}).json()
    assert body["kind"] == "splat" and body["full_3d"] is True and body["backend"] == "remote"
    assert seen["auth"] == "Bearer tok" and b'name="mask"' in seen["body"] and b'name="prompt"' in seen["body"]
