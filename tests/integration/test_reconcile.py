"""Tests for on-startup reconciliation (Stage 10f, BRD §16).

Fully stubbed — no real Docker, no real Postgres, no real Redis. A fake DB
connection yields the ``strategy_registry`` container rows, a reachable/raising
client stands in for ``FreqtradeAPI.ping``, and the live_pause transition is
driven through injected seams (a spy ``kill_event_writer_fn`` + a spy
``record_kill_switch_event`` for the orchestration tests, and the REAL
``make_kill_event_writer`` over a stub graph for the end-to-end drive test).

Covers the brief's four required contracts plus the paper-vs-live decision and
graceful DB degradation:

1. reachable container → kept (no row, no transition);
2. unreachable LIVE container → live_pause transition + a ``kill_switch_events``
   row with the EXACT reason ``orchestrator_restart_no_freqtrade``;
3. mixed set handled (reachable-live + unreachable-live + unreachable-paper);
4. empty registry is a no-op;
5. an unreachable NON-live (paper) container is logged but NOT transitioned —
   ``live_pause`` is a live-subgraph-only node (the 9f kill writer's Fork-2
   guard enforces the same invariant);
6. a DB-list failure logs and does NOT crash startup.

Marked ``integration`` (exercises the DR/startup-reconciliation surface,
sibling to ``test_freqtrade_exporter`` / ``test_kill_subscription``), though it
needs no live services.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import orchestrator.kill_subscription as kill_sub
from orchestrator.ops.reconcile import (
    RECONCILE_ACTION,
    RECONCILE_REASON,
    reconcile_on_startup,
)

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


def _connect_boom() -> Any:
    async def _fn() -> Any:
        raise ConnectionError("postgres down")

    return _fn


class _PingClient:
    """A reachable container — ``ping`` returns the Freqtrade pong."""

    async def __aenter__(self) -> _PingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def ping(self) -> dict[str, Any]:
        return {"status": "pong"}


class _DownClient:
    """An unreachable container — entering the client raises (mirrors the
    exporter's down-client; a transport error on connect)."""

    async def __aenter__(self) -> _DownClient:
        raise ConnectionError("container unreachable")

    async def __aexit__(self, *exc: object) -> None:
        return None


def _factory_all_up(url: str, creds: Any) -> Any:
    return _PingClient()


def _factory_all_down(url: str, creds: Any) -> Any:
    return _DownClient()


class _SpyWriter:
    """Records every (strategy_id, event) the reconciler drives to live_pause."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, strategy_id: str, event: dict[str, Any]) -> None:
        self.calls.append((strategy_id, event))


class _SpyRecord:
    """Records every kill_switch_events row the reconciler writes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, strategy_id: str, reason: str, metrics: dict[str, Any], action_taken: str
    ) -> int:
        self.calls.append(
            {
                "strategy_id": strategy_id,
                "reason": reason,
                "metrics": metrics,
                "action_taken": action_taken,
            }
        )
        return len(self.calls)


# ─── (1) reachable container is kept ───────────────────────────────────


async def test_reachable_container_is_kept() -> None:
    writer, record = _SpyWriter(), _SpyRecord()
    rows = [("rec_live_up", "live", "http://127.0.0.1:8201")]

    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect(rows),
        client_factory=_factory_all_up,
        record_kill_switch_event_fn=record,
    )

    assert summary["checked"] == 1
    assert summary["reachable"] == ["rec_live_up"]
    assert summary["transitioned"] == []
    # A reachable container keeps its state: no row, no transition.
    assert writer.calls == []
    assert record.calls == []


# ─── (2) unreachable LIVE → live_pause + kill_switch_events row ─────────


async def test_unreachable_live_transitions_and_records_row() -> None:
    writer, record = _SpyWriter(), _SpyRecord()
    rows = [("rec_live_down", "live", "http://127.0.0.1:8202")]

    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect(rows),
        client_factory=_factory_all_down,
        record_kill_switch_event_fn=record,
    )

    assert summary["transitioned"] == ["rec_live_down"]

    # The durable kill_switch_events row carries the EXACT BRD §16 reason and a
    # NON-stop action_taken (no /stop was issued — the container is unreachable).
    assert len(record.calls) == 1
    row = record.calls[0]
    assert row["strategy_id"] == "rec_live_down"
    assert row["reason"] == RECONCILE_REASON == "orchestrator_restart_no_freqtrade"
    assert row["action_taken"] == RECONCILE_ACTION
    assert "POST /api/v1/stop" not in row["action_taken"]

    # The thread was driven to live_pause via the 9f kill writer (reuse, not a
    # new path): the event carries the same reason so _route_after_live_wait
    # routes to live_pause.
    assert len(writer.calls) == 1
    sid, event = writer.calls[0]
    assert sid == "rec_live_down"
    assert event["reason"] == RECONCILE_REASON


# ─── (3) mixed set handled ─────────────────────────────────────────────


async def test_mixed_set_handled() -> None:
    writer, record = _SpyWriter(), _SpyRecord()
    rows = [
        ("rec_up", "live", "url_up"),
        ("rec_down_live", "live", "url_down_live"),
        ("rec_down_paper", "paper", "url_down_paper"),
    ]

    def factory(url: str, creds: Any) -> Any:
        return _PingClient() if url == "url_up" else _DownClient()

    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect(rows),
        client_factory=factory,
        record_kill_switch_event_fn=record,
    )

    assert summary["checked"] == 3
    assert summary["reachable"] == ["rec_up"]
    assert summary["transitioned"] == ["rec_down_live"]
    assert summary["skipped_non_live"] == ["rec_down_paper"]

    # Only the unreachable LIVE thread gets a row + transition.
    assert [c["strategy_id"] for c in record.calls] == ["rec_down_live"]
    assert [sid for sid, _ in writer.calls] == ["rec_down_live"]


# ─── (4) empty registry is a no-op ─────────────────────────────────────


async def test_empty_registry_is_noop() -> None:
    writer, record = _SpyWriter(), _SpyRecord()

    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect([]),
        client_factory=_factory_all_down,
        record_kill_switch_event_fn=record,
    )

    assert summary["checked"] == 0
    assert summary["reachable"] == []
    assert summary["transitioned"] == []
    assert writer.calls == []
    assert record.calls == []


# ─── (5) unreachable PAPER is skipped (paper-vs-live decision) ──────────


async def test_unreachable_paper_is_skipped_not_transitioned() -> None:
    """live_pause is a live-subgraph-only node: an unreachable paper container is
    logged but never routed to live_pause and writes no kill_switch_events row.
    The paper subgraph's own wake re-attaches it; its gate is HITL regardless."""
    writer, record = _SpyWriter(), _SpyRecord()
    rows = [("rec_paper_down", "paper", "http://127.0.0.1:8101")]

    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect(rows),
        client_factory=_factory_all_down,
        record_kill_switch_event_fn=record,
    )

    assert summary["skipped_non_live"] == ["rec_paper_down"]
    assert summary["transitioned"] == []
    assert writer.calls == []
    assert record.calls == []


# ─── (6) a DB-list failure does not crash startup ──────────────────────


async def test_db_list_failure_does_not_raise() -> None:
    writer, record = _SpyWriter(), _SpyRecord()

    # Must NOT raise — a DB failure listing containers is swallowed + logged so
    # the lifespan keeps booting (degrade gracefully, BRD §17 #12 spirit).
    summary = await reconcile_on_startup(
        kill_event_writer_fn=writer,
        connect_fn=_connect_boom(),
        client_factory=_factory_all_up,
        record_kill_switch_event_fn=record,
    )

    assert summary["checked"] == 0
    assert writer.calls == []
    assert record.calls == []


# ─── (7) end-to-end: the REAL 9f writer drives the live_pause resume ────


class _Snap:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class _ResumeGraph:
    """Minimal stub of the compiled graph the real make_kill_event_writer drives:
    records aupdate_state + astream (an empty async generator). Mirrors the
    test_live_wake.py fixture so this test proves reconcile reuses the SAME 9f
    mechanism end-to-end (artifacts.kill_switch_event write → direct resume)."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.astream_calls: list[tuple[Any, Any]] = []

    async def aget_state(self, config: dict[str, Any]) -> _Snap:
        return _Snap(self._values)

    async def aupdate_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        self.updates.append((config, values))

    def astream(self, command: Any, config: Any = None) -> AsyncIterator[Any]:
        self.astream_calls.append((command, config))
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Any]:
        if False:  # pragma: no cover — empty async generator
            yield None


async def _drain_kill_resume_tasks() -> None:
    await asyncio.gather(*list(kill_sub._KILL_RESUME_TASKS))


async def test_real_writer_drives_live_pause_resume() -> None:
    graph = _ResumeGraph({"stage": "live", "artifacts": {}})
    real_writer = kill_sub.make_kill_event_writer(graph)
    record = _SpyRecord()
    rows = [("rec_e2e", "live", "http://127.0.0.1:8203")]

    summary = await reconcile_on_startup(
        kill_event_writer_fn=real_writer,
        connect_fn=_connect(rows),
        client_factory=_factory_all_down,
        record_kill_switch_event_fn=record,
    )
    await _drain_kill_resume_tasks()

    assert summary["transitioned"] == ["rec_e2e"]
    # The 9f writer wrote artifacts.kill_switch_event (the live_pause route
    # trigger) and then DIRECTLY resumed the parked live_wait thread.
    assert len(graph.updates) == 1
    _config, update = graph.updates[0]
    assert update["artifacts"]["kill_switch_event"]["reason"] == RECONCILE_REASON
    assert len(graph.astream_calls) == 1
    assert graph.astream_calls[0][1]["configurable"]["thread_id"] == "strategy_rec_e2e"
