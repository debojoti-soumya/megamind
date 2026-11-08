"""Optional shared-token auth (`API_TOKEN`), for a server reachable from the public internet (deploy/modal_app.py).

Empty token (default) = no auth. With a token every route except CORS preflights needs one of:
- `Authorization: Bearer <token>` (headset, tools, fetch calls),
- `?token=<token>` (links and media URLs where a header can't be set, e.g. `<img src>` built by another origin),
- the `megamind_token` cookie, set on any request authenticated by `?token=`, so opening `/ui?token=...` or
  `/xr?token=...` once makes the page's own same-origin `<img>`/`<video>`/`<audio>`/fetch requests work unchanged.
"""

from __future__ import annotations

import hmac
import json
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

COOKIE = "megamind_token"


def _matches(candidate: str | None, token: str) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())


class TokenAuth:
    """Pure-ASGI middleware (keeps SSE streaming untouched)."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        bearer = auth[7:].strip() if auth[:7].lower() == "bearer " else None
        query = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("token", [None])[0]
        cookie = SimpleCookie(headers.get("cookie", "")).get(COOKIE)
        if _matches(bearer, self.token) or (cookie is not None and _matches(cookie.value, self.token)):
            await self.app(scope, receive, send)
            return
        if not _matches(query, self.token):
            await _unauthorized(scope, send)
            return
        secure = scope.get("scheme") in ("https", "wss") or headers.get("x-forwarded-proto") == "https"
        set_cookie = f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")

        async def send_with_cookie(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (b"set-cookie", set_cookie.encode())]}
            await send(message)

        await self.app(scope, receive, send_with_cookie)


async def _unauthorized(scope, send) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps({"detail": "missing or invalid API token"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
                (b"access-control-allow-origin", b"*"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
