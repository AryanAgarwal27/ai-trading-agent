"""Thin async httpx client over the Freqtrade REST API (BRD §7.2).

Scope is deliberately small: the endpoints the orchestrator actually calls
during paper/live monitoring and gate evaluation. Wider Freqtrade endpoints
(`/forceexit`, `/start`, `/reload_config`, etc.) are out of scope until a
later stage proves they are needed.

Auth follows Freqtrade's JWT flow:
- ``POST /api/v1/token/login`` with HTTP Basic auth returns an
  ``access_token`` (short-lived) and a ``refresh_token`` (long-lived).
- Subsequent calls send ``Authorization: Bearer <access_token>``.
- On 401 we transparently refresh once via ``POST /api/v1/token/refresh``
  using ``Authorization: Bearer <refresh_token>``; on a second 401 we
  re-login from username/password and surface the error if that also fails.

Credentials are loaded explicitly from caller code (a Freqtrade ``config.json``
under ``freqtrade/user_data/configs/``). They are NEVER read from environment
variables here — that would let a misconfigured paper instance accidentally
share keys with live, which BRD §1.1 rule 5 forbids.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import httpx


class FreqtradeAPIError(RuntimeError):
    """Raised when a Freqtrade REST call fails after retry/refresh.

    Carries the HTTP status code (or 0 for transport errors) and the response
    body where available, so the caller can log a useful diagnostic.
    """

    def __init__(self, message: str, *, status: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True, slots=True)
class FreqtradeCredentials:
    """Username/password pair from a Freqtrade ``config.json``.

    The orchestrator reads ``api_server.username`` and ``api_server.password``
    from the same config file that is bind-mounted into the Freqtrade
    container, so the credentials match by construction.
    """

    username: str
    password: str


class FreqtradeAPI:
    """Async REST client for one Freqtrade instance.

    Instances are cheap; create one per ``(strategy_id, base_url)`` pair.
    Re-use of an httpx.AsyncClient across many calls is the point — connection
    pooling and a single TLS handshake matter when the paper monitor pulls
    several endpoints per wake.

    Usage::

        async with FreqtradeAPI(
            base_url="http://127.0.0.1:8081",
            credentials=FreqtradeCredentials("orchestrator", "..."),
        ) as client:
            await client.ping()
            profit = await client.profit()

    The class is intentionally NOT a context-manager-only API; long-lived
    callers (the paper subgraph) can hold a reference and call ``aclose()``
    on teardown. The ``async with`` form is sugar around that.
    """

    # Endpoint constants — keeps a typo from silently routing to the wrong
    # action. Kept colocated with the methods rather than in a separate module
    # because this is the only place they're used.
    _PATH_PING = "/api/v1/ping"
    _PATH_STATUS = "/api/v1/status"
    _PATH_PROFIT = "/api/v1/profit"
    _PATH_TRADES = "/api/v1/trades"
    _PATH_PERFORMANCE = "/api/v1/performance"
    _PATH_STOPBUY = "/api/v1/stopbuy"
    _PATH_STOP = "/api/v1/stop"
    _PATH_LOGIN = "/api/v1/token/login"
    _PATH_REFRESH = "/api/v1/token/refresh"

    def __init__(
        self,
        base_url: str,
        credentials: FreqtradeCredentials,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._creds = credentials
        # Timeout covers connect + read; 10s is generous for the local
        # network the orchestrator and Freqtrade share. Kill switch logic
        # must NOT depend on this client (BRD §11) — it has its own
        # subprocess invocation of /stop. So a slow client here can never
        # block the kill switch.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout_s,
            headers={"Accept": "application/json"},
        )
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        # Serialize concurrent first-time logins. Without this lock, two
        # parallel monitor calls on a fresh client would both fire a login
        # request and the second one's tokens would clobber the first's.
        self._auth_lock = asyncio.Lock()

    # ────────────────────────── lifecycle ──────────────────────────

    async def __aenter__(self) -> FreqtradeAPI:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ─────────────────────── public endpoints ───────────────────────

    async def ping(self) -> dict[str, Any]:
        """``GET /api/v1/ping``. Unauthenticated; returns ``{"status": "pong"}``.

        Used by ``paper_spawn`` to confirm a freshly started Freqtrade
        container is accepting requests before the orchestrator transitions
        the thread state.
        """
        # /ping is unauthenticated per Freqtrade API; skip the auth round.
        resp = await self._client.get(self._PATH_PING)
        return cast(dict[str, Any], self._parse(resp, expected=200))

    async def status(self) -> list[dict[str, Any]]:
        """``GET /api/v1/status``. List of currently-open trades."""
        return cast(list[dict[str, Any]], await self._authed_get(self._PATH_STATUS))

    async def profit(self) -> dict[str, Any]:
        """``GET /api/v1/profit``. Cumulative P&L + drawdown summary."""
        return cast(dict[str, Any], await self._authed_get(self._PATH_PROFIT))

    async def trades(self, limit: int = 50) -> dict[str, Any]:
        """``GET /api/v1/trades?limit=N``. Recent closed trades."""
        return cast(
            dict[str, Any],
            await self._authed_get(self._PATH_TRADES, params={"limit": limit}),
        )

    async def performance(self) -> list[dict[str, Any]]:
        """``GET /api/v1/performance``. Per-pair aggregate stats."""
        return cast(list[dict[str, Any]], await self._authed_get(self._PATH_PERFORMANCE))

    async def stopbuy(self) -> dict[str, Any]:
        """``POST /api/v1/stopbuy`` — graceful: stop new entries, let opens run.

        Used by the daily-loss-limit APScheduler job (BRD §11). Does NOT
        force-exit existing positions.
        """
        return cast(dict[str, Any], await self._authed_post(self._PATH_STOPBUY))

    async def stop(self) -> dict[str, Any]:
        """``POST /api/v1/stop`` — full stop, kill switch path.

        BRD §11 mandates that the kill switch be out-of-band of the LLM and
        operate even if the orchestrator is down. The kill-switch APScheduler
        job invokes this method *directly*, not via the graph. The graph
        learns about the stop via Redis pubsub on its next wake.
        """
        return cast(dict[str, Any], await self._authed_post(self._PATH_STOP))

    # ────────────────────────── internals ──────────────────────────

    async def _authed_get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self._authed("GET", path, params=params, json_body=None)

    async def _authed_post(self, path: str) -> Any:
        return await self._authed("POST", path, params=None, json_body=None)

    async def _authed(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> Any:
        # First call: ensure we have an access token.
        await self._ensure_access_token()
        resp = await self._send(method, path, params=params, json_body=json_body)
        if resp.status_code == 401:
            # Try refresh first, then full re-login. Both paths cost 1
            # round-trip; we don't bother distinguishing token-expired vs
            # secret-rotated because the result is the same.
            await self._refresh_or_relogin()
            resp = await self._send(method, path, params=params, json_body=json_body)
        return self._parse(resp, expected=200)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._access_token}"} if self._access_token else {}
        try:
            return await self._client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise FreqtradeAPIError(
                f"transport error calling {method} {path}: {exc}", status=0
            ) from exc

    async def _ensure_access_token(self) -> None:
        if self._access_token is not None:
            return
        async with self._auth_lock:
            if self._access_token is not None:  # double-check after lock
                return
            await self._login()

    async def _login(self) -> None:
        # Freqtrade expects Basic auth on /token/login and returns
        # {access_token, refresh_token}.
        try:
            resp = await self._client.post(
                self._PATH_LOGIN,
                auth=(self._creds.username, self._creds.password),
            )
        except httpx.HTTPError as exc:
            raise FreqtradeAPIError(f"transport error during login: {exc}", status=0) from exc
        body = self._parse(resp, expected=200)
        self._access_token = body["access_token"]
        self._refresh_token = body["refresh_token"]

    async def _refresh_or_relogin(self) -> None:
        async with self._auth_lock:
            if self._refresh_token is None:
                self._access_token = None
                await self._login()
                return
            try:
                resp = await self._client.post(
                    self._PATH_REFRESH,
                    headers={"Authorization": f"Bearer {self._refresh_token}"},
                )
            except httpx.HTTPError:
                # Refresh transport failure — fall back to full login.
                self._access_token = None
                self._refresh_token = None
                await self._login()
                return
            if resp.status_code != 200:
                # Refresh token rejected (expired / rotated). Full re-login.
                self._access_token = None
                self._refresh_token = None
                await self._login()
                return
            body = resp.json()
            self._access_token = body["access_token"]
            # Some Freqtrade versions rotate the refresh token; keep up.
            if "refresh_token" in body:
                self._refresh_token = body["refresh_token"]

    @staticmethod
    def _parse(resp: httpx.Response, *, expected: int) -> Any:
        if resp.status_code != expected:
            raise FreqtradeAPIError(
                f"unexpected status {resp.status_code} for {resp.request.method} "
                f"{resp.request.url.path}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        # Freqtrade /ping returns JSON; /stop and /stopbuy return JSON too.
        # If a future endpoint returns empty body, the json() call here will
        # raise — fix at that endpoint, not in this generic helper.
        return resp.json()
