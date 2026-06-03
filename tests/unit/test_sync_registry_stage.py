"""Unit tests for ``orchestrator.supervisor.sync_registry_stage`` (Stage 9a).

Closes the SPEC 2026-05-27 (Stage 6f) reconciliation item:
``strategy_registry.stage`` does not auto-update from LangGraph state, so
the supervisor reconciles it. These tests prove the reconciliation logic
against a stubbed graph + a mocked ``psycopg.AsyncConnection`` — no real
Postgres, no real checkpointer (same mocked-conn pattern as
``tests/unit/test_events.py``).

Covered:
  - a stale registry row is UPDATEd to match graph state;
  - an in-sync row is left untouched (no UPDATE);
  - a thread whose ``aget_state`` raises (no checkpoint) is skipped;
  - a thread whose state carries no ``stage`` (seeded row, graph not run)
    is skipped;
  - a mixed sweep updates exactly the stale row;
  - the function does NOT commit (caller owns the transaction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from orchestrator.supervisor import sync_registry_stage

# Sentinel for "aget_state raises for this thread" in the stub graph.
_RAISE = object()


class _StubGraph:
    """Minimal compiled-graph stand-in exposing only ``aget_state``.

    ``states`` maps ``thread_id`` → one of:
      - a stage string  → ``snapshot.values == {"stage": <stage>}``
      - ``None``        → ``snapshot.values == {}`` (no stage in state)
      - ``_RAISE``      → ``aget_state`` raises (no checkpoint)
    A thread_id absent from the map also raises (defensive: a registry row
    pointing at a thread the graph has never seen).
    """

    def __init__(self, states: dict[str, Any]) -> None:
        self._states = states

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        thread_id = config["configurable"]["thread_id"]
        if thread_id not in self._states:
            raise RuntimeError(f"no checkpoint for {thread_id}")
        value = self._states[thread_id]
        if value is _RAISE:
            raise RuntimeError(f"aget_state boom for {thread_id}")
        return SimpleNamespace(values={"stage": value} if value is not None else {})


def _make_conn(
    registry_rows: list[tuple[str, str, str]],
) -> tuple[AsyncMock, list[tuple[Any, ...]]]:
    """Build a mocked async conn whose SELECT returns ``registry_rows``.

    Returns ``(conn, executed)`` where ``executed`` accumulates every
    ``(sql, params)`` the code runs — the first entry is the SELECT, any
    later entries are UPDATEs. ``fetchall`` always returns the registry
    rows (only the SELECT consumes them; UPDATEs don't fetch).
    """
    executed: list[tuple[Any, ...]] = []

    cur = AsyncMock()

    async def _execute(sql: str, params: tuple[Any, ...] | None = None) -> None:
        executed.append((sql, params))

    cur.execute.side_effect = _execute
    cur.fetchall.return_value = registry_rows

    cursor_cm = AsyncMock()
    cursor_cm.__aenter__.return_value = cur
    cursor_cm.__aexit__.return_value = None

    conn = AsyncMock()
    conn.cursor = lambda: cursor_cm
    return conn, executed


def _updates(executed: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """The UPDATE statements among the executed SQL (drops the SELECT)."""
    return [(sql, params) for sql, params in executed if sql.strip().upper().startswith("UPDATE")]


# ─── stale row → UPDATE ────────────────────────────────────────────────


async def test_stale_registry_row_is_updated_to_graph_stage() -> None:
    """Registry says 'paper', graph says 'archived' → one UPDATE, summary records it."""
    conn, executed = _make_conn([("s1", "strategy_s1", "paper")])
    graph = _StubGraph({"strategy_s1": "archived"})

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    sql, params = updates[0]
    assert "strategy_registry" in sql
    assert "last_updated = now()" in sql
    assert params == ("archived", "s1")

    assert result["checked"] == 1
    assert result["updated"] == [{"strategy_id": "s1", "from": "paper", "to": "archived"}]
    assert result["skipped"] == []


# ─── in-sync row → no UPDATE ───────────────────────────────────────────


async def test_in_sync_row_is_not_updated() -> None:
    """Registry stage already matches graph stage → no UPDATE issued."""
    conn, executed = _make_conn([("s2", "strategy_s2", "live")])
    graph = _StubGraph({"strategy_s2": "live"})

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == []
    assert result["checked"] == 1


# ─── no checkpoint / aget_state raises → skip ──────────────────────────


async def test_thread_with_no_checkpoint_is_skipped() -> None:
    """A thread whose aget_state raises is skipped, not updated."""
    conn, executed = _make_conn([("s3", "strategy_s3", "research")])
    graph = _StubGraph({"strategy_s3": _RAISE})

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == ["s3"]


# ─── state without a stage → skip ──────────────────────────────────────


async def test_thread_with_no_stage_in_state_is_skipped() -> None:
    """A seeded registry row whose graph state carries no stage is left as-is."""
    conn, executed = _make_conn([("s4", "strategy_s4", "paper")])
    graph = _StubGraph({"strategy_s4": None})  # snapshot.values == {}

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == ["s4"]


# ─── mixed sweep → exactly the stale row updates ───────────────────────


async def test_mixed_sweep_updates_only_the_stale_row() -> None:
    """Three rows: stale, in-sync, no-checkpoint → exactly one UPDATE."""
    conn, executed = _make_conn(
        [
            ("stale", "strategy_stale", "validation"),
            ("synced", "strategy_synced", "paper"),
            ("missing", "strategy_missing", "research"),
        ]
    )
    graph = _StubGraph(
        {
            "strategy_stale": "paper",  # moved validation -> paper
            "strategy_synced": "paper",  # already in sync
            "strategy_missing": _RAISE,  # no checkpoint
        }
    )

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    assert updates[0][1] == ("paper", "stale")

    assert result["checked"] == 3
    assert result["updated"] == [{"strategy_id": "stale", "from": "validation", "to": "paper"}]
    assert result["skipped"] == ["missing"]


# ─── caller owns the transaction → no commit ───────────────────────────


async def test_sync_does_not_commit() -> None:
    """The caller owns the connection + transaction; sync must NOT commit."""
    conn, _executed = _make_conn([("s1", "strategy_s1", "paper")])
    graph = _StubGraph({"strategy_s1": "archived"})

    await sync_registry_stage(graph, conn)

    conn.commit.assert_not_called()
