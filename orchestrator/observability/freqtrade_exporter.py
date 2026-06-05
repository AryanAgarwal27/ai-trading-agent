"""In-orchestrator Freqtrade exporter (Stage 10e.2, BRD §14).

BRD §14 calls for "Each Freqtrade container's /api/v1/profit, /status via a
small exporter SIDECAR." This implements the same observable outcome as an
IN-ORCHESTRATOR COLLECTOR rather than a separate sidecar container — the
deviation + rationale is recorded in the Stage 10e.2 commit message
(summary: BRD §14 is observability guidance, not a §1.1/§6 non-negotiable;
the orchestrator already holds the FreqtradeAPI JWT client + the registry of
running containers, so one collector exposing per-strategy-labelled metrics
on the existing /metrics endpoint delivers the same per-container profit/
status to Prometheus with far less surface on the single-host deployment of
BRD §3 — assessed as WITHIN BRD §14's intent).

On each /metrics scrape the orchestrator lists the non-archived
``strategy_registry`` rows that carry a ``freqtrade_api_url`` (the running
paper/live containers) and scrapes each one's ``/api/v1/profit`` +
``/api/v1/status`` via the existing
:class:`orchestrator.tools.freqtrade_api.FreqtradeAPI` client (REUSED, not
re-implemented), CONCURRENTLY, into per-strategy gauges.

Graceful degradation (mandatory — mirrors the kill-switch swallow-and-log):
a container that is down / unreachable is SKIPPED and logged, its
``ait_freqtrade_up`` set to 0, and it NEVER raises. A DB failure listing the
containers is likewise swallowed (the collector returns, updating nothing).
The /metrics endpoint additionally wraps the whole collection in try/except,
so a scrape failure can never break /metrics.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import psycopg

from orchestrator.observability.events import _connect_app_db
from orchestrator.observability.metrics import (
    FREQTRADE_MAX_DRAWDOWN,
    FREQTRADE_OPEN_TRADES,
    FREQTRADE_PROFIT_CLOSED_PCT,
    FREQTRADE_UP,
)
from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeCredentials

logger = logging.getLogger(__name__)

# Injection seams (tests pass fakes; defaults are the real DB + real client).
ConnectFn = Callable[[], Awaitable[psycopg.AsyncConnection]]
# Returns an async-context-manager Freqtrade client (the real FreqtradeAPI, or
# a test double). Typed loosely so a stub need not subclass FreqtradeAPI.
ClientFactory = Callable[[str, FreqtradeCredentials], Any]


def _default_client_factory(base_url: str, creds: FreqtradeCredentials) -> FreqtradeAPI:
    # 5s timeout (tighter than the client's 10s default) so a hung container
    # can't push a /metrics scrape past Prometheus's scrape timeout — combined
    # with concurrent scraping this bounds /metrics latency to one slow client.
    return FreqtradeAPI(base_url=base_url, credentials=creds, timeout_s=5.0)


def _password_for_stage(stage: str) -> str:
    """Freqtrade REST password by stage (BRD §1.1 rule 5 paper/live separation).

    Live containers authenticate with ``BINANCE_LIVE_API_PASSWORD``, paper with
    ``PAPER_API_PASSWORD`` — the same split the paper/live build-context helpers
    use. Read from env (never the registry); an empty value just yields a failed
    login that the per-container guard skips.
    """
    if stage == "live":
        return os.environ.get("BINANCE_LIVE_API_PASSWORD", "")
    return os.environ.get("PAPER_API_PASSWORD", "")


async def _active_containers(connect_fn: ConnectFn) -> list[tuple[str, str, str]]:
    """``(strategy_id, stage, freqtrade_api_url)`` for each non-archived registry
    row that has a URL — i.e. the running paper/live containers. Caller-agnostic;
    opens + closes its own connection."""
    conn = await connect_fn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT strategy_id, stage, freqtrade_api_url FROM strategy_registry "
                "WHERE freqtrade_api_url IS NOT NULL AND stage != 'archived'"
            )
            rows = await cur.fetchall()
    finally:
        await conn.close()
    return [(str(sid), str(stage), str(url)) for sid, stage, url in rows]


def _record_container(
    strategy_id: str, profit: dict[str, Any], status: list[dict[str, Any]]
) -> None:
    p = profit if isinstance(profit, dict) else {}
    FREQTRADE_PROFIT_CLOSED_PCT.labels(strategy_id=strategy_id).set(
        float(p.get("profit_closed_percent", 0.0) or 0.0)
    )
    FREQTRADE_MAX_DRAWDOWN.labels(strategy_id=strategy_id).set(
        float(p.get("max_drawdown", 0.0) or 0.0)
    )
    open_trades = len(status) if isinstance(status, list) else 0
    FREQTRADE_OPEN_TRADES.labels(strategy_id=strategy_id).set(open_trades)


async def _scrape_one(
    strategy_id: str, stage: str, url: str, client_factory: ClientFactory
) -> None:
    """Scrape one container into its gauges. NEVER raises — a down container is
    logged and marked ``ait_freqtrade_up=0``."""
    try:
        creds = FreqtradeCredentials(username="freqtrader", password=_password_for_stage(stage))
        async with client_factory(url, creds) as client:
            profit = await client.profit()
            status = await client.status()
    except Exception as exc:  # noqa: BLE001 — a down container is skipped, not fatal
        logger.warning(
            "freqtrade exporter: scrape failed sid=%s stage=%s url=%s err=%s; skipping",
            strategy_id,
            stage,
            url,
            exc,
        )
        FREQTRADE_UP.labels(strategy_id=strategy_id).set(0)
        return
    _record_container(strategy_id, profit, status)
    FREQTRADE_UP.labels(strategy_id=strategy_id).set(1)


async def collect_freqtrade_metrics(
    *,
    connect_fn: ConnectFn = _connect_app_db,
    client_factory: ClientFactory = _default_client_factory,
) -> None:
    """Scrape every running container's /profit + /status into per-strategy
    gauges, concurrently. NEVER raises (see module docstring)."""
    try:
        rows = await _active_containers(connect_fn)
    except Exception as exc:  # noqa: BLE001 — a DB failure must not break the scrape
        logger.warning("freqtrade exporter: could not list containers: %s; skipping", exc)
        return
    if not rows:
        return
    # Concurrent — each _scrape_one swallows its own failure, so gather never
    # propagates; wall-clock is bounded by the single slowest container.
    await asyncio.gather(
        *(_scrape_one(sid, stage, url, client_factory) for sid, stage, url in rows)
    )
