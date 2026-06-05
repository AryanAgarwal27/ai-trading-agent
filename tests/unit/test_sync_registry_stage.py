"""Unit tests for ``orchestrator.supervisor.sync_registry_stage`` (Stage 9a;
generalized to the registry↔graph-state MIRROR CONTRACT in Stage 10f / D-10).

Closes the SPEC 2026-05-27 (Stage 6f) reconciliation item AND D-10: the
registry mirrors graph-derived fields (``stage`` + ``template``) per a single
declared contract (``MIRRORED_FIELDS`` + the overwrite-vs-skip rule
``_mirror_value``), not a stage-only / one-off-template patch. These tests prove
the reconciliation logic against a stubbed graph + a mocked
``psycopg.AsyncConnection`` — no real Postgres, no real checkpointer (same
mocked-conn pattern as ``tests/unit/test_events.py``).

Covered:
  - the overwrite-vs-skip RULE asserted directly (real value wins; seed
    sentinel / empty / equal value skip);
  - the declared CONTRACT membership (stage + template IN; pairs/timeframe OUT);
  - a stale ``stage`` row is UPDATEd to graph state;
  - a real ``template`` (registry "pending") is mirrored to the registry row;
  - a still-seed ``template`` ("pending" in graph state) is NOT mirrored;
  - a row whose stage AND template both moved updates BOTH columns in one UPDATE
    and reports one change record per field;
  - an in-sync row is left untouched (no UPDATE);
  - a thread whose ``aget_state`` raises (no checkpoint) is skipped;
  - a thread whose state is empty (seeded row, graph not run) is skipped;
  - a mixed sweep updates exactly the stale rows;
  - the function does NOT commit (caller owns the transaction).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from orchestrator.supervisor import (
    MIRRORED_FIELDS,
    MirrorField,
    _mirror_value,
    sync_registry_stage,
)

# Sentinel for "aget_state raises for this thread" in the stub graph.
_RAISE = object()


class _StubGraph:
    """Minimal compiled-graph stand-in exposing only ``aget_state``.

    ``states`` maps ``thread_id`` → one of:
      - a dict       → ``snapshot.values == <dict>`` (graph state values)
      - ``None``     → ``snapshot.values == {}`` (graph thread not run)
      - ``_RAISE``   → ``aget_state`` raises (no checkpoint)
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
        return SimpleNamespace(values=value if value is not None else {})


def _make_conn(
    registry_rows: list[tuple[str, str, str, str]],
) -> tuple[AsyncMock, list[tuple[Any, ...]]]:
    """Build a mocked async conn whose SELECT returns ``registry_rows``.

    Rows are ``(strategy_id, thread_id, stage, template)`` — the mirrored
    columns the generalized SELECT reads. Returns ``(conn, executed)`` where
    ``executed`` accumulates every ``(sql, params)`` the code runs — the first
    entry is the SELECT, any later entries are UPDATEs.
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


# ═══════════════════════════════════════════════════════════════════════
# The DECLARED contract: rule + membership (D-10's actual deliverable)
# ═══════════════════════════════════════════════════════════════════════


def test_overwrite_vs_skip_rule_is_graph_wins_once_real() -> None:
    """D-10's open question, asserted directly: graph state WINS for a mirrored
    field once it holds a real (truthy, non-seed) value differing from the
    registry; otherwise the registry seed is kept (skip)."""
    template = MirrorField(name="template", seed_sentinels=frozenset({"pending"}))

    # Real graph value, differs from the registry seed → overwrite.
    assert _mirror_value(template, registry_value="pending", graph_value="mr_template") is True
    # Graph value is still the seed sentinel → keep registry (skip).
    assert _mirror_value(template, registry_value="pending", graph_value="pending") is False
    # Graph value empty / None → never clobber a registry value with nothing.
    assert _mirror_value(template, registry_value="real", graph_value="") is False
    assert _mirror_value(template, registry_value="real", graph_value=None) is False
    # Graph value equals registry value → no-op (skip).
    assert _mirror_value(template, registry_value="mr", graph_value="mr") is False

    # stage has no seed sentinel — every value is real, so any truthy diff wins.
    stage = MirrorField(name="stage", seed_sentinels=frozenset())
    assert _mirror_value(stage, registry_value="paper", graph_value="live") is True
    assert _mirror_value(stage, registry_value="paper", graph_value="paper") is False


def test_contract_membership_stage_and_template_in_pairs_timeframe_out() -> None:
    """The declared mirror set: stage + template (graph-derived) IN; pairs /
    timeframe / name (spawn inputs, registry is their source of truth) OUT."""
    names = {f.name for f in MIRRORED_FIELDS}
    assert names == {"stage", "template"}
    # template carries the "pending" seed sentinel; stage has none.
    by_name = {f.name: f for f in MIRRORED_FIELDS}
    assert "pending" in by_name["template"].seed_sentinels
    assert by_name["stage"].seed_sentinels == frozenset()
    # Spawn-input fields are deliberately NOT mirrored.
    assert {"pairs", "timeframe", "name"}.isdisjoint(names)


# ═══════════════════════════════════════════════════════════════════════
# stage mirroring (the 9a behavior, now via the generic contract)
# ═══════════════════════════════════════════════════════════════════════


async def test_stale_registry_stage_is_updated_to_graph_stage() -> None:
    """Registry says 'paper', graph says 'archived' → one UPDATE, one change record."""
    conn, executed = _make_conn([("s1", "strategy_s1", "paper", "mr")])
    graph = _StubGraph({"strategy_s1": {"stage": "archived", "template": "mr"}})

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    sql, params = updates[0]
    assert "strategy_registry" in sql
    assert "last_updated = now()" in sql
    assert params == ("archived", "s1")  # only stage changed

    assert result["checked"] == 1
    assert result["updated"] == [
        {"strategy_id": "s1", "field": "stage", "from": "paper", "to": "archived"}
    ]
    assert result["skipped"] == []


# ═══════════════════════════════════════════════════════════════════════
# template mirroring (D-10 closure)
# ═══════════════════════════════════════════════════════════════════════


async def test_real_template_is_mirrored_from_graph_state() -> None:
    """The registry seed 'pending' is overwritten once the researcher's chosen
    template is in graph state — the dashboard/audit view no longer shows 'pending'."""
    conn, executed = _make_conn([("s1", "strategy_s1", "research", "pending")])
    graph = _StubGraph(
        {"strategy_s1": {"stage": "research", "template": "mean_reversion_template"}}
    )

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    sql, params = updates[0]
    assert "template = %s" in sql
    assert params == ("mean_reversion_template", "s1")  # only template changed
    assert result["updated"] == [
        {
            "strategy_id": "s1",
            "field": "template",
            "from": "pending",
            "to": "mean_reversion_template",
        }
    ]


async def test_seed_template_is_not_mirrored() -> None:
    """Before the researcher chooses, graph state still carries the 'pending'
    seed — the rule keeps the registry as-is (no spurious UPDATE)."""
    conn, executed = _make_conn([("s1", "strategy_s1", "research", "pending")])
    graph = _StubGraph({"strategy_s1": {"stage": "research", "template": "pending"}})

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == []


async def test_stage_and_template_both_move_in_one_update() -> None:
    """A row whose stage AND template both moved updates BOTH columns in a single
    UPDATE and reports one change record per field (generic, not stage-only)."""
    conn, executed = _make_conn([("s1", "strategy_s1", "research", "pending")])
    graph = _StubGraph({"strategy_s1": {"stage": "validation", "template": "breakout_template"}})

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    sql, params = updates[0]
    # Both columns set in declaration order (stage, then template), then last_updated.
    assert "stage = %s" in sql
    assert "template = %s" in sql
    assert params == ("validation", "breakout_template", "s1")
    assert result["updated"] == [
        {"strategy_id": "s1", "field": "stage", "from": "research", "to": "validation"},
        {"strategy_id": "s1", "field": "template", "from": "pending", "to": "breakout_template"},
    ]


# ═══════════════════════════════════════════════════════════════════════
# in-sync / skip semantics
# ═══════════════════════════════════════════════════════════════════════


async def test_in_sync_row_is_not_updated() -> None:
    """Registry already matches graph for every mirrored field → no UPDATE."""
    conn, executed = _make_conn([("s2", "strategy_s2", "live", "mr")])
    graph = _StubGraph({"strategy_s2": {"stage": "live", "template": "mr"}})

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == []
    assert result["checked"] == 1


async def test_thread_with_no_checkpoint_is_skipped() -> None:
    """A thread whose aget_state raises is skipped, not updated."""
    conn, executed = _make_conn([("s3", "strategy_s3", "research", "pending")])
    graph = _StubGraph({"strategy_s3": _RAISE})

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == ["s3"]


async def test_thread_with_empty_state_is_skipped() -> None:
    """A seeded registry row whose graph state is empty (thread not run) is
    left as-is — never overwrite a seed with nothing."""
    conn, executed = _make_conn([("s4", "strategy_s4", "paper", "mr")])
    graph = _StubGraph({"strategy_s4": None})  # snapshot.values == {}

    result = await sync_registry_stage(graph, conn)

    assert _updates(executed) == []
    assert result["updated"] == []
    assert result["skipped"] == ["s4"]


async def test_mixed_sweep_updates_only_the_stale_rows() -> None:
    """Three rows: stale stage, in-sync, no-checkpoint → exactly one UPDATE."""
    conn, executed = _make_conn(
        [
            ("stale", "strategy_stale", "validation", "mr"),
            ("synced", "strategy_synced", "paper", "mr"),
            ("missing", "strategy_missing", "research", "pending"),
        ]
    )
    graph = _StubGraph(
        {
            "strategy_stale": {"stage": "paper", "template": "mr"},  # stage moved
            "strategy_synced": {"stage": "paper", "template": "mr"},  # in sync
            "strategy_missing": _RAISE,  # no checkpoint
        }
    )

    result = await sync_registry_stage(graph, conn)

    updates = _updates(executed)
    assert len(updates) == 1
    assert updates[0][1] == ("paper", "stale")

    assert result["checked"] == 3
    assert result["updated"] == [
        {"strategy_id": "stale", "field": "stage", "from": "validation", "to": "paper"}
    ]
    assert result["skipped"] == ["missing"]


async def test_sync_does_not_commit() -> None:
    """The caller owns the connection + transaction; sync must NOT commit."""
    conn, _executed = _make_conn([("s1", "strategy_s1", "paper", "mr")])
    graph = _StubGraph({"strategy_s1": {"stage": "archived", "template": "mr"}})

    await sync_registry_stage(graph, conn)

    conn.commit.assert_not_called()
