"""Stage 8f tests — out-of-band kill switch + daily-loss job (BRD §1.1 rule 7, §11).

The kill switch is INDEPENDENT of the graph: it polls Freqtrade REST directly
and calls /stop (kill) or /stopbuy (daily-loss soft halt) on threshold breach.
These tests inject the external effects as seams (REST client factory, DB
record writer, the already-fired guard, the Redis publisher), so the job logic
runs with no Docker / no real LLM. The idempotency-guard, migration, and
live_spawn-anchor tests use real Postgres (mirroring the other integration
suites).

All ``integration``-marked (real Postgres available); none ``freqtrade``.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import psycopg
import pytest

from orchestrator.scheduler import daily_loss_job, kill_switch_poll_job
from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeCredentials

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DUMMY_STRATEGY = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


# ───────────────────────── DB helpers / fixtures ─────────────────────────


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)


@pytest.fixture
async def cleanup_strategy_ids() -> Any:
    ids: list[str] = []
    yield ids
    if not ids:
        return
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM kill_switch_events WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,))
        await conn.commit()


async def _seed_live_row(strategy_id: str, *, live_started_at: datetime) -> None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO strategy_registry
                  (strategy_id, thread_id, name, template, stage, pairs, timeframe,
                   freqtrade_api_url, live_started_at, started_at, last_updated)
                VALUES (%s, %s, %s, 'mean_reversion_template', 'live',
                        '["BTC/USDT"]', '5m', 'http://127.0.0.1:8200', %s, now(), now())
                """,
                (strategy_id, f"strategy_{strategy_id}", f"t-{strategy_id}", live_started_at),
            )
        await conn.commit()


async def _seed_stop_event(strategy_id: str, *, fired_at: datetime) -> None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO kill_switch_events (strategy_id, fired_at, reason, metrics, action_taken) "
                "VALUES (%s, %s, 'drawdown_12pct_exceeded', '{}', 'POST /api/v1/stop')",
                (strategy_id, fired_at),
            )
        await conn.commit()


# ───────────────────────── stubs ─────────────────────────


class _StubClient:
    """Stub Freqtrade REST client — async context manager, records calls."""

    def __init__(
        self,
        *,
        profit: dict[str, Any] | None = None,
        trades: dict[str, Any] | None = None,
        daily: dict[str, Any] | None = None,
        stop_exc: Exception | None = None,
        stopbuy_exc: Exception | None = None,
        calls: dict[str, int] | None = None,
    ) -> None:
        self._profit = profit or {}
        self._trades = trades or {"trades": []}
        self._daily = daily or {"data": []}
        self._stop_exc = stop_exc
        self._stopbuy_exc = stopbuy_exc
        self._calls = calls if calls is not None else {}

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def profit(self) -> dict[str, Any]:
        return self._profit

    async def trades(self, limit: int = 50) -> dict[str, Any]:
        return self._trades

    async def daily(self, timescale: int = 1) -> dict[str, Any]:
        return self._daily

    async def stop(self) -> dict[str, Any]:
        self._calls["stop"] = self._calls.get("stop", 0) + 1
        if self._stop_exc is not None:
            raise self._stop_exc
        return {"status": "stopped"}

    async def stopbuy(self) -> dict[str, Any]:
        self._calls["stopbuy"] = self._calls.get("stopbuy", 0) + 1
        if self._stopbuy_exc is not None:
            raise self._stopbuy_exc
        return {"status": "stopbuy"}


def _one_live(strategy_id: str, url: str = "http://127.0.0.1:8200") -> Any:
    async def _fn() -> list[tuple[str, str]]:
        return [(strategy_id, url)]

    return _fn


async def _never_fired(_strategy_id: str) -> bool:
    return False


def _capture_events() -> tuple[list[dict[str, Any]], Any]:
    events: list[dict[str, Any]] = []

    async def _rec(
        *, strategy_id: str, reason: str, metrics: dict[str, Any], action_taken: str
    ) -> int:
        events.append({"strategy_id": strategy_id, "reason": reason, "action_taken": action_taken})
        return 1

    return events, _rec


async def _noop_publish(_strategy_id: str, _payload: dict[str, Any]) -> None:
    return None


# ───────────────────────── kill_switch_poll_job — fire paths ──────────────


async def test_kill_switch_drawdown_threshold_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = f"ks-{uuid.uuid4().hex[:8]}"
    calls: dict[str, int] = {}
    stub = _StubClient(profit={"max_drawdown": 0.13}, trades={"trades": []}, calls=calls)
    events, rec = _capture_events()
    fake = SimpleNamespace(publish=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr("orchestrator.observability.events._redis_client", lambda: fake)

    await kill_switch_poll_job(
        list_live_fn=_one_live(sid),
        already_fired_fn=_never_fired,
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
    )

    assert calls.get("stop") == 1, "/stop must be called once on a drawdown breach"
    assert events and events[0]["reason"] == "drawdown_12pct_exceeded"
    assert events[0]["action_taken"] == "POST /api/v1/stop"
    fake.publish.assert_awaited_once()
    channel, body = fake.publish.await_args.args
    assert channel == f"ai-trading-agent:kill_switch:{sid}"
    payload = json.loads(body)
    assert payload["reason"] == "drawdown_12pct_exceeded"
    assert "fired_at" in payload and "metrics_summary" in payload
    # 8g Change 1: action_taken flows through the publish payload, matching
    # what was written to kill_switch_events (no subscription-side default).
    assert payload["action_taken"] == "POST /api/v1/stop"
    assert payload["action_taken"] == events[0]["action_taken"]


async def test_kill_switch_consecutive_losses_threshold_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = f"ks-{uuid.uuid4().hex[:8]}"
    calls: dict[str, int] = {}
    trades = {"trades": [{"profit_ratio": -0.01} for _ in range(10)]}
    stub = _StubClient(profit={"max_drawdown": 0.0}, trades=trades, calls=calls)
    events, rec = _capture_events()
    fake = SimpleNamespace(publish=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr("orchestrator.observability.events._redis_client", lambda: fake)

    await kill_switch_poll_job(
        list_live_fn=_one_live(sid),
        already_fired_fn=_never_fired,
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
    )

    assert calls.get("stop") == 1
    assert events[0]["reason"] == "consecutive_losses_10_exceeded"
    channel, body = fake.publish.await_args.args
    assert channel == f"ai-trading-agent:kill_switch:{sid}"
    assert json.loads(body)["reason"] == "consecutive_losses_10_exceeded"


async def test_kill_switch_no_fire_under_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = f"ks-{uuid.uuid4().hex[:8]}"
    calls: dict[str, int] = {}
    trades = {"trades": [{"profit_ratio": -0.01} for _ in range(9)]}  # 9 < 10
    stub = _StubClient(profit={"max_drawdown": 0.11}, trades=trades, calls=calls)  # 11% < 12%
    events, rec = _capture_events()
    fake = SimpleNamespace(publish=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr("orchestrator.observability.events._redis_client", lambda: fake)

    await kill_switch_poll_job(
        list_live_fn=_one_live(sid),
        already_fired_fn=_never_fired,
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
    )

    assert calls.get("stop") is None, "no /stop under thresholds"
    assert events == []
    assert fake.publish.await_count == 0


async def test_kill_switch_freqtrade_unresponsive_logs_and_continues() -> None:
    """A /stop hang on strategy 1 records stop_call_timeout and the loop
    proceeds to strategy 2 (doesn't crash the scheduler)."""
    sid1 = f"ks-{uuid.uuid4().hex[:8]}"
    sid2 = f"ks-{uuid.uuid4().hex[:8]}"
    calls1: dict[str, int] = {}
    calls2: dict[str, int] = {}
    stub1 = _StubClient(
        profit={"max_drawdown": 0.13},
        trades={"trades": []},
        stop_exc=httpx.TimeoutException("timed out"),
        calls=calls1,
    )
    stub2 = _StubClient(profit={"max_drawdown": 0.0}, trades={"trades": []}, calls=calls2)
    events, rec = _capture_events()
    published: list[tuple[str, dict[str, Any]]] = []

    async def _capture_publish(strategy_id: str, payload: dict[str, Any]) -> None:
        published.append((strategy_id, payload))

    async def _two_live() -> list[tuple[str, str]]:
        return [(sid1, "u1"), (sid2, "u2")]

    await kill_switch_poll_job(
        list_live_fn=_two_live,
        already_fired_fn=_never_fired,
        rest_client_factory=lambda url: stub1 if url == "u1" else stub2,
        record_event_fn=rec,
        publish_fn=_capture_publish,
    )

    assert calls1.get("stop") == 1  # attempted, raised
    sid1_events = [e for e in events if e["strategy_id"] == sid1]
    assert sid1_events and sid1_events[0]["action_taken"] == "stop_call_timeout"
    assert not [e for e in events if e["strategy_id"] == sid2], "sid2 had no breach"
    # 8g Change 1: the non-default action_taken ("stop_call_timeout") flows
    # through the publish payload too, matching the kill_switch_events row.
    sid1_published = [p for s, p in published if s == sid1]
    assert sid1_published and sid1_published[0]["action_taken"] == "stop_call_timeout"
    assert sid1_published[0]["action_taken"] == sid1_events[0]["action_taken"]


# ───────────────────────── daily_loss_job ─────────────────────────


async def test_daily_loss_job_stopbuy_at_threshold() -> None:
    sid = f"dl-{uuid.uuid4().hex[:8]}"
    calls: dict[str, int] = {}
    stub = _StubClient(daily={"data": [{"date": "2026-06-03", "rel_profit": -0.035}]}, calls=calls)
    events, rec = _capture_events()

    await daily_loss_job(
        list_live_fn=_one_live(sid),
        already_fired_fn=_never_fired,
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
    )

    assert calls.get("stopbuy") == 1, "/stopbuy must be called"
    assert calls.get("stop") is None, "daily-loss must NOT call /stop (soft halt)"
    assert events[0]["action_taken"] == "POST /api/v1/stopbuy"
    assert events[0]["reason"] == "daily_loss_3pct_exceeded"


async def test_daily_loss_job_no_fire_under_threshold() -> None:
    sid = f"dl-{uuid.uuid4().hex[:8]}"
    calls: dict[str, int] = {}
    stub = _StubClient(daily={"data": [{"rel_profit": -0.025}]}, calls=calls)  # -2.5% > -3%
    events, rec = _capture_events()

    await daily_loss_job(
        list_live_fn=_one_live(sid),
        already_fired_fn=_never_fired,
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
    )

    assert calls.get("stopbuy") is None
    assert events == []


# ───────────────────────── daily() REST client unit ──────────────────────


async def test_daily_client_method_calls_correct_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_get(path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        captured["path"] = path
        captured["params"] = params
        return {"data": [{"date": "2026-06-03", "rel_profit": -0.01}]}

    async with FreqtradeAPI("http://x", FreqtradeCredentials("u", "p")) as client:
        monkeypatch.setattr(client, "_authed_get", _fake_get)
        result = await client.daily(timescale=1)

    assert captured["path"] == "/api/v1/daily"
    assert captured["params"] == {"timescale": 1}
    assert result["data"][0]["rel_profit"] == -0.01


# ───────────────────────── idempotency guard (real DB) ───────────────────


async def test_kill_switch_does_not_refire_within_same_live_run(
    cleanup_strategy_ids: list[str],
) -> None:
    """A stop already fired AFTER live_started_at suppresses the next poll."""
    sid = f"ks-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    t0 = datetime.now(UTC) - timedelta(hours=1)
    await _seed_live_row(sid, live_started_at=t0)
    await _seed_stop_event(sid, fired_at=t0 + timedelta(minutes=1))  # after live_started_at

    calls: dict[str, int] = {}
    stub = _StubClient(profit={"max_drawdown": 0.20}, trades={"trades": []}, calls=calls)
    events, rec = _capture_events()

    # Real already_fired guard (default) hits the seeded rows → skip.
    await kill_switch_poll_job(
        list_live_fn=_one_live(sid),
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
        publish_fn=_noop_publish,
    )

    assert calls.get("stop") is None, "guard must suppress the re-fire"
    assert events == [], "no duplicate kill_switch_events row"


async def test_kill_switch_refires_after_fresh_live_spawn(
    cleanup_strategy_ids: list[str],
) -> None:
    """A fresh live_spawn bumps live_started_at past the old kill's fired_at,
    so the guard no longer suppresses — the kill can fire again. This is the
    property 5d adds (vs 5a, which would suppress forever). NOTE: this proves
    the FRESH-SPAWN un-suppression, not the pause-resume path (which doesn't
    re-spawn — see DEFERRED.md D-5)."""
    sid = f"ks-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    t1 = datetime.now(UTC)  # current live run
    await _seed_live_row(sid, live_started_at=t1)
    # Old kill from a PRIOR run (before this live_started_at).
    await _seed_stop_event(sid, fired_at=t1 - timedelta(hours=1))

    calls: dict[str, int] = {}
    stub = _StubClient(profit={"max_drawdown": 0.20}, trades={"trades": []}, calls=calls)
    events, rec = _capture_events()

    await kill_switch_poll_job(
        list_live_fn=_one_live(sid),
        rest_client_factory=lambda _url: stub,
        record_event_fn=rec,
        publish_fn=_noop_publish,
    )

    assert calls.get("stop") == 1, "old kill (before live_started_at) must not suppress"
    assert events and events[0]["reason"] == "drawdown_12pct_exceeded"


# ───────────────────────── migration + live_spawn anchor (real DB) ────────


async def test_migration_0003_adds_live_started_at_column() -> None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'strategy_registry' AND column_name = 'live_started_at'"
            )
            row = await cur.fetchone()
    assert row is not None, "migration 0003 must add strategy_registry.live_started_at"


async def test_live_spawn_sets_live_started_at_on_each_spawn(
    cleanup_strategy_ids: list[str],
) -> None:
    """_write_live_registry populates live_started_at on insert AND bumps it on
    a conflict-update (re-spawn) — the anchor the guard depends on."""
    from orchestrator.subgraphs.live import live_spawn

    sid = f"ks-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)

    async def _stub_spawn(*, port: int, **_k: Any) -> str:
        return f"http://127.0.0.1:{port}"

    state: dict[str, Any] = {
        "strategy_id": sid,
        "name": f"t-{sid}",
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "stake_amount": 125.0,
        "stage": "live",
        "params": {},
        "agent_votes": [],
        "gate_decisions": {},
        "artifacts": {"generated_strategy_path": str(DUMMY_STRATEGY)},
    }
    config = {"configurable": {"thread_id": f"strategy_{sid}"}}

    async def _live_started_at() -> datetime | None:
        async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT live_started_at FROM strategy_registry WHERE strategy_id = %s", (sid,)
                )
                row = await cur.fetchone()
        return row[0] if row else None

    await live_spawn(state, config, spawn_live_container_fn=_stub_spawn)  # type: ignore[arg-type]
    ts1 = await _live_started_at()
    assert ts1 is not None, "live_started_at set on first spawn"

    await live_spawn(state, config, spawn_live_container_fn=_stub_spawn)  # type: ignore[arg-type]
    ts2 = await _live_started_at()
    assert ts2 is not None and ts2 >= ts1, "live_started_at bumped on re-spawn (conflict-update)"
