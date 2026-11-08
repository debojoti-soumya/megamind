"""Memory Theater (GET /xr): the page is served, every asset it references exists, and the JS parses."""

import re
import shutil
import subprocess

import pytest
from httpx import ASGITransport, AsyncClient

from megamind.config import Settings
from megamind.main import create_app
from megamind.xr import STATIC_DIR, XRSettings, install


async def test_xr_page_served(client):
    c, _ = client
    for path in ("/xr", "/xr/"):
        r = await c.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/html")
        assert "<title>Memory Theater</title>" in r.text
        assert '<script type="importmap">' in r.text


async def test_xr_referenced_assets_exist(client):
    c, _ = client
    html = (await c.get("/xr")).text
    local = re.findall(r'(?:src|href)="(/xr/static/[^"]+)"', html)
    assert {"/xr/static/app.js", "/xr/static/xr.css"} <= set(local)
    for path in local:
        r = await c.get(path)
        assert r.status_code == 200, path
        assert r.content
    # app.js imports theater.js relatively; that must be served too
    app_js = (await c.get("/xr/static/app.js")).text
    for rel in re.findall(r'from "\./([^"]+)"', app_js):
        r = await c.get(f"/xr/static/{rel}")
        assert r.status_code == 200, rel
        assert "javascript" in r.headers["content-type"]
    assert (await c.get("/xr/static/nope.js")).status_code == 404


async def test_xr_three_pinned_in_importmap(client):
    c, _ = client
    html = (await c.get("/xr")).text
    # pinned exact version (no floating "three@latest" / "three@0") with an SRI hash for it
    url = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"
    assert f'"three": "{url}"' in html
    assert re.search(re.escape(f'"{url}": "sha384-') + r"[A-Za-z0-9+/=]{64}", html)


async def test_xr_uses_only_v1_endpoints_that_exist(client):
    c, app = client
    js = "".join(p.read_text() for p in STATIC_DIR.glob("*.js"))
    used = set(re.findall(r'"(/v1/[a-z0-9_./]+)"', js))
    assert used >= {"/v1/memories/recent", "/v1/chat", "/v1/replay.mp4"}
    routes = set(app.openapi()["paths"])
    for u in used:
        assert u in routes, u


async def test_xr_can_be_disabled(tmp_path):
    from fastapi import FastAPI

    app = FastAPI()
    install(app, XRSettings(enabled=False))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/xr")).status_code == 404


async def test_xr_does_not_shadow_api(tmp_path):
    settings = Settings(
        thumb_dir=tmp_path / "thumbs", audio_dir=tmp_path / "audio", consolidate_enabled=False, _env_file=None
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/v1/health")).status_code == 200


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", sorted(p.name for p in STATIC_DIR.glob("*.js")))
def test_xr_js_syntax(name, tmp_path):
    # node --check treats .js as CommonJS unless told otherwise; copy to .mjs so ES module syntax is checked
    src = tmp_path / (name[:-3] + ".mjs")
    src.write_text((STATIC_DIR / name).read_text())
    r = subprocess.run(["node", "--check", str(src)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
