from httpx import ASGITransport, AsyncClient
from test_api import _noisy_jpeg, _post_frame

from megamind.config import Settings
from megamind.main import create_app

TOKEN = "s3cret-token"


def _settings(tmp_path, **kw) -> Settings:
    return Settings(
        thumb_dir=tmp_path / "thumbs",
        audio_dir=tmp_path / "audio",
        embed_provider="mock",
        enrich_provider="mock",
        filter_min_gap_s=0.0,
        consolidate_enabled=False,
        _env_file=None,
        **kw,
    )


async def _client(app, **kw):
    return AsyncClient(transport=ASGITransport(app=app), base_url="https://test", **kw)


async def test_no_token_configured_means_no_auth(tmp_path):
    app = create_app(_settings(tmp_path))
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            assert (await c.get("/v1/health")).status_code == 200
            assert (await c.get("/ui")).status_code == 200


async def test_routes_require_token(tmp_path):
    app = create_app(_settings(tmp_path, api_token=TOKEN))
    async with app.router.lifespan_context(app):
        async with await _client(app) as c:
            for path in ("/v1/health", "/v1/memories?q=x", "/v1/stats", "/ui", "/xr", "/v1/thumbs/nope.jpg"):
                r = await c.get(path)
                assert r.status_code == 401, path
                assert r.headers["www-authenticate"] == "Bearer"
            assert (await c.delete("/v1/memories")).status_code == 401
            assert (await c.post("/v1/chat/stream", json={"session_id": "s", "text": "hi"})).status_code == 401
            bad = {"Authorization": "Bearer wrong"}
            assert (await c.get("/v1/health", headers=bad)).status_code == 401
            assert (await c.get("/v1/health?token=wrong")).status_code == 401
            # CORS preflight is never blocked (browsers send it without credentials)
            pre = await c.options(
                "/v1/health",
                headers={"Origin": "https://x.example", "Access-Control-Request-Method": "GET"},
            )
            assert pre.status_code == 200


async def test_bearer_query_and_cookie_paths(tmp_path):
    app = create_app(_settings(tmp_path, api_token=TOKEN))
    async with app.router.lifespan_context(app):
        hdr = {"Authorization": f"Bearer {TOKEN}"}
        async with await _client(app, headers=hdr) as c:
            assert (await c.get("/v1/health")).json()["status"] == "ok"
            r = await _post_frame(c, _noisy_jpeg(1), 1731000000.0)
            assert r.status_code == 202 and r.json()["accepted"]
            await app.state.worker.queue.join()
            chat = await c.post("/v1/chat", json={"session_id": "s", "text": "what did I see?"})
            assert chat.status_code == 200
            stream = await c.post("/v1/chat/stream", json={"session_id": "s", "text": "what did I see?"})
            assert stream.status_code == 200 and "event: final" in stream.text
        async with await _client(app) as c:
            # media URL with ?token= works without a header and sets the cookie for the page's later requests
            r = await c.get(f"/ui?token={TOKEN}")
            assert r.status_code == 200
            assert "megamind_token=" in r.headers["set-cookie"]
            assert "HttpOnly" in r.headers["set-cookie"] and "Secure" in r.headers["set-cookie"]
            assert (await c.get("/v1/health")).status_code == 200  # cookie carried by the client jar
            assert (await c.get("/xr")).status_code == 200
            assert (await c.get("/xr/static/app.js")).status_code == 200
        async with await _client(app, cookies={"megamind_token": "wrong"}) as c:
            assert (await c.get("/v1/health")).status_code == 401
