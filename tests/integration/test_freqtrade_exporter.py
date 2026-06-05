"""Tests for the in-orchestrator Freqtrade exporter (Stage 10e.2, BRD §14).

Fully stubbed — no real Docker, no real Postgres: a fake DB connection
yields the container rows and a fake/raising client stands in for
``FreqtradeAPI``. Covers the three contracts the brief requires:

1. profit/status map to the per-strategy gauges;
2. an unreachable container is SKIPPED (``ait_freqtrade_up`` → 0), not fatal;
3. ``GET /metrics`` still returns 200 when one container is down.

Marked ``integration`` (drives the FastAPI ``/metrics`` surface end-to-end
with the exporter wired), though it needs no live services.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

from orchestrator.main import app
from orchestrator.observability.freqtrade_exporter import collect_freqtrade_metrics
from orchestrator.tools.freqtrade_api import FreqtradeCredentials

pytestmark = pytest.mark.integration


# ─── fakes (no real DB / no real Freqtrade container) ──────────────────


class _FakeCursor:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self._rows = rows

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        return None

    async def fetchall(self) -> list[tuple[str, str, str]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)

    async def close(self) -> None:
        return None


def _connect(rows: list[tuple[str, str, str]]) -> Any:
    async def _fn() -> Any:
        return _FakeConn(rows)

    return _fn


class _FakeClient:
    def __init__(self, profit: dict[str, Any], status: list[dict[str, Any]]) -> None:
        self._profit = profit
        self._status = status

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def profit(self) -> dict[str, Any]:
        return self._profit

    async def status(self) -> list[dict[str, Any]]:
        return self._status


class _DownClient:
    async def __aenter__(self) -> _DownClient:
        raise ConnectionError("container unreachable")

    async def __aexit__(self, *exc: object) -> None:
        return None


# ─── (1) profit/status map to metrics ──────────────────────────────────


async def test_collect_maps_profit_and_status_to_metrics() -> None:
    rows = [("exp_sid_a", "paper", "http://127.0.0.1:8101")]
    client = _FakeClient({"profit_closed_percent": 5.5, "max_drawdown": 0.12}, [{}, {}, {}])

    await collect_freqtrade_metrics(
        connect_fn=_connect(rows), client_factory=lambda url, creds: client
    )

    assert REGISTRY.get_sample_value("ait_freqtrade_up", {"strategy_id": "exp_sid_a"}) == 1.0
    assert (
        REGISTRY.get_sample_value(
            "ait_freqtrade_profit_closed_percent", {"strategy_id": "exp_sid_a"}
        )
        == 5.5
    )
    assert (
        REGISTRY.get_sample_value("ait_freqtrade_max_drawdown", {"strategy_id": "exp_sid_a"})
        == 0.12
    )
    assert (
        REGISTRY.get_sample_value("ait_freqtrade_open_trades", {"strategy_id": "exp_sid_a"}) == 3.0
    )


# ─── (2) an unreachable container is skipped, not fatal ────────────────


async def test_unreachable_container_is_skipped_not_fatal() -> None:
    rows = [("exp_sid_down", "live", "http://127.0.0.1:8201")]

    # Must NOT raise — a down container is swallowed + logged.
    await collect_freqtrade_metrics(
        connect_fn=_connect(rows), client_factory=lambda url, creds: _DownClient()
    )

    assert REGISTRY.get_sample_value("ait_freqtrade_up", {"strategy_id": "exp_sid_down"}) == 0.0


# ─── (3) /metrics stays 200 when one container is down ─────────────────


async def test_metrics_endpoint_stays_200_with_one_container_down() -> None:
    rows = [("exp_up", "paper", "url_up"), ("exp_down", "live", "url_down")]
    up_client = _FakeClient({"profit_closed_percent": 1.0, "max_drawdown": 0.0}, [])

    def factory(url: str, creds: FreqtradeCredentials) -> Any:
        return up_client if url == "url_up" else _DownClient()

    collector = partial(
        collect_freqtrade_metrics, connect_fn=_connect(rows), client_factory=factory
    )
    app.state.freqtrade_collector = collector
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/metrics")
    finally:
        del app.state.freqtrade_collector

    assert resp.status_code == 200
    text = resp.text
    # Up container reported up=1; down container reported up=0 — one bad
    # container did not break the scrape or the endpoint.
    assert 'ait_freqtrade_up{strategy_id="exp_up"} 1.0' in text
    assert 'ait_freqtrade_up{strategy_id="exp_down"} 0.0' in text
