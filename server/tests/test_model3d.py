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
        ("turn my mug into a 3D model", "my mug"),
        ("show me my mug in 3D from yesterday", "show me my mug from yesterday"),
    ],
)
def test_strip_3d_framing(q, subject):
    assert strip_3d_framing(q) == subject


def test_wants_3d():
    assert wants_3d("make a 3D model of my mug") and wants_3d("splat of the car")
    assert not wants_3d("pictures of my dog") and not wants_3d("where did I leave my keys")
    assert not wants_3d("where did I leave my 3D printer?") and not wants_3d("the 3d-printed vase")


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
    monkeypatch.setattr(model3d.mv.VGGTRunner, "missing", staticmethod(lambda: "no vggt"))
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


# ---- multi-view (VGGT) ----


def _scene(n=6, h=28, w=40, seed=0):
    """Per-pixel world points of a table (top half of each photo) with a box on it (bottom half)."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, h, w, 3), np.float32)
    v, u = np.mgrid[0:h, 0:w]
    for i in range(n):
        table = v < h // 2
        pts[i, ..., 0] = np.where(table, u / w * 2 - 1, (u / w - 0.5) * 0.4)
        pts[i, ..., 2] = np.where(table, v / h * 2 - 1, (i / n - 0.5) * 0.3)
        pts[i, ..., 1] = np.where(table, 0.0, -(0.1 + (v / h - 0.5) * 0.6))  # VGGT: y down
    pts += rng.normal(0, 1e-4, pts.shape).astype(np.float32)
    rgb = np.zeros((n, h, w, 3), np.uint8)
    rgb[..., 1] = np.where(v >= h // 2, 200, 50)[None]
    return pts, np.ones((n, h, w), np.float32), rgb


def test_clean_points_drops_the_table_and_stands_the_object_up():
    from megamind.agent.multiview import clean_points

    pts, conf, rgb = _scene()
    q, c, info = clean_points(pts, conf, rgb)
    assert info["plane_removed"] and (c[:, 1] == 200).mean() > 0.95  # the box's points, not the table's
    assert q[:, 1].min() > 0.05 and 0.3 < q[:, 1].max() < 0.45  # y up, standing on the table plane
    assert abs(np.median(q[:, 0])) < 1e-3 and abs(np.median(q[:, 2])) < 1e-3


def test_clean_points_box_masks_and_too_few_points():
    from megamind.agent.multiview import clean_points

    pts, conf, rgb = _scene()
    with pytest.raises(ValueError):
        clean_points(pts, conf, rgb, np.zeros(conf.shape, bool))


def test_prepare_maps_boxes_like_vggt_crop():
    import io

    from PIL import Image

    from megamind.agent.multiview import SIDE, frame_size, prepare

    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (255, 0, 0)).save(buf, format="JPEG")
    arr, masks = prepare([buf.getvalue()] * 2, [[200.0, 150.0, 600.0, 450.0], None])
    nh, top, ch = frame_size(800, 600)
    assert arr.shape == (2, 3, ch, SIDE) and ch % 14 == 0 and arr[0, 0].mean() > 0.9
    assert masks[1].all() and 0.2 < masks[0].mean() < 0.45 and masks[0][ch // 2, SIDE // 2]


def test_points_ply_roundtrip():
    from megamind.agent.multiview import ply_vertex_count, to_points_ply

    data = to_points_ply(np.ones((7, 3), np.float32), np.full((7, 3), 9, np.uint8))
    head, body = data.split(b"end_header\n", 1)
    assert b"binary_little_endian" in head and b"property uchar red" in head and len(body) == 7 * 15
    assert ply_vertex_count(data) == 7


class FakeRunner:
    runs: list = []

    @staticmethod
    def missing():
        return None

    def run(self, images):
        FakeRunner.runs.append(images.shape)
        n, _, h, w = images.shape
        pts, conf, _ = _scene(n, h, w)
        return pts, conf


def test_model3d_multiview_builds_in_the_background(app3d, tmp_path):
    import io
    import time

    from PIL import Image

    from megamind.models import MemoryToken

    near = []
    for i in range(7):
        p = tmp_path / f"mv{i}.jpg"
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (30 * i, 60, 30)).save(buf, format="JPEG")
        p.write_bytes(buf.getvalue())
        near.append(
            MemoryToken(
                id=f"mv{i}",
                device_id="d",
                ts=1731000000.0 + 5 * i,
                caption="",
                thumb_path=str(p),
                frame_hash=f"{i:016x}",
            )
        )

    class Store:
        async def range(self, t0, t1, limit=100, *, device_id=None, kinds=None):
            return [t for t in near if t0 <= t.ts <= t1]

    app3d.state.store = Store()
    app3d.state.model3d_mv = FakeRunner()
    FakeRunner.runs = []
    with TestClient(app3d) as c:
        body = c.post("/v1/model3d", json={"query": "make a 3D model of my mug"}).json()
        assert body["status"] == "building" and body["kind"] == "points" and body["full_3d"] is True
        assert body["viewer_url"].endswith("&f=ply&k=points") and body["views"] == 9  # 7 nearby + the 2 found
        assert "few minutes" in body["narration"]
        for _ in range(100):
            r = c.get(body["url"])
            if r.status_code != 202:
                break
            time.sleep(0.05)
        assert r.status_code == 200, r.text
        assert r.content.startswith(b"ply\nformat binary_little_endian") and len(FakeRunner.runs) == 1
        done = c.post("/v1/model3d", json={"query": "make a 3D model of my mug"}).json()
        assert done["id"] == body["id"] and done["status"] == "ready" and done["vertices"] > 0
        assert len(FakeRunner.runs) == 1  # cached


def test_model3d_multiview_failure_is_reported(app3d, tmp_path):
    import time

    from megamind.models import MemoryToken

    near = [
        MemoryToken(
            id=f"x{i}",
            device_id="d",
            ts=1731000000.0 + i,
            caption="",
            thumb_path=str(tmp_path / "0.jpg"),
            frame_hash=f"{i:016x}",
        )
        for i in range(6)
    ]  # 80x60 photos: the fake detector never sees the mug in them

    class Store:
        async def range(self, t0, t1, limit=100, *, device_id=None, kinds=None):
            return near

    app3d.state.store = Store()
    app3d.state.model3d_mv = FakeRunner()
    with TestClient(app3d) as c:
        body = c.post("/v1/model3d", json={"query": "3d model of my mug"}).json()
        for _ in range(100):
            r = c.get(body["url"])
            if r.status_code != 202:
                break
            time.sleep(0.05)
        assert r.status_code == 422 and "photos clearly show it" in r.json()["detail"]


def test_model3d_multiview_purged_while_building_keeps_nothing(app3d, tmp_path):
    import io
    import time

    from PIL import Image

    from megamind.models import MemoryToken

    near = []
    for i in range(6):
        p = tmp_path / f"pg{i}.jpg"
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (30 * i, 60, 30)).save(buf, format="JPEG")
        p.write_bytes(buf.getvalue())
        near.append(
            MemoryToken(
                id=f"pg{i}", device_id="d", ts=1731000000.0 + i, caption="", thumb_path=str(p), frame_hash=f"{i:016x}"
            )
        )

    class Store:
        async def range(self, t0, t1, limit=100, *, device_id=None, kinds=None):
            return near

    class PurgingRunner(FakeRunner):
        def run(self, images):
            app3d.state.model3d_cache.clear()  # DELETE /v1/memories mid-build
            return super().run(images)

    app3d.state.store = Store()
    app3d.state.model3d_mv = PurgingRunner()
    with TestClient(app3d) as c:
        body = c.post("/v1/model3d", json={"query": "make a 3D model of my mug"}).json()
        for _ in range(100):
            r = c.get(body["url"])
            if r.status_code != 202:
                break
            time.sleep(0.05)
        assert r.status_code == 404 and "mug" not in r.text


def test_gather_views_one_visit_inside_the_time_range():
    import asyncio
    from types import SimpleNamespace

    from megamind.agent.model3d import Model3DSettings, gather_views
    from megamind.models import MemoryToken

    def tok(i, ts):
        return MemoryToken(id=f"g{i}", device_id="d", ts=ts, caption="", thumb_path="x.jpg", frame_hash=f"{i:016x}")

    day1 = [tok(i, 1000.0 + 10 * i) for i in range(10)]  # 1000..1090
    day2 = [tok(100 + i, 90000.0 + 10 * i) for i in range(5)]
    store = SimpleNamespace(range=None)

    async def rng(t0, t1, limit=100, *, device_id=None, kinds=None):
        return [t for t in day1 + day2 if t0 <= t.ts <= t1]

    store.range = rng
    st = SimpleNamespace(store=store)
    picks = [SimpleNamespace(token=day1[5], score=0.9), SimpleNamespace(token=day2[2], score=0.8)]
    cfg = Model3DSettings()
    got = asyncio.run(gather_views(st, picks, cfg))
    assert {t.id for t in got} == {t.id for t in day1}  # the other day's sighting is a different visit
    got = asyncio.run(gather_views(st, picks, cfg, t0=1030.0, t1=1060.0))
    assert [t.ts for t in got] == [1030.0, 1040.0, 1050.0, 1060.0]


def test_clean_points_confidence_is_judged_inside_the_box():
    from megamind.agent.multiview import clean_points

    pts, conf, rgb = _scene()
    obj = np.zeros(conf.shape, bool)
    obj[:, conf.shape[1] // 2 :] = True
    conf = np.where(obj, 0.2, 5.0).astype(np.float32)  # the table is far more confident than the object
    q, c, _ = clean_points(pts, conf, rgb, obj)
    assert len(q) > 0 and (c[:, 1] == 200).all()


def test_prepare_crops_a_portrait_photo_instead_of_stretching_it():
    import io

    from PIL import Image, ImageDraw

    from megamind.agent.multiview import prepare

    def photo(size, sq):
        img = Image.new("RGB", size, (0, 0, 0))
        ImageDraw.Draw(img).rectangle(sq, fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    sq = (200, 300, 399, 499)  # a 200 px white square in the middle of a 600x800 portrait photo
    arr, masks = prepare([photo((800, 600), (0, 0, 1, 1)), photo((600, 800), sq)], [None, list(map(float, sq))])
    white = arr[1, 0] > 0.5
    ys, xs = np.nonzero(white)
    assert 0.85 < (np.ptp(xs) + 1) / (np.ptp(ys) + 1) < 1.15  # still square, not squashed
    my, mx = np.nonzero(masks[1])
    assert my.min() <= ys.min() and my.max() >= ys.max() and mx.min() <= xs.min() and mx.max() >= xs.max()
    assert masks[1].mean() < 0.5
