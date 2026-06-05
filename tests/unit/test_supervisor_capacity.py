"""Unit tests for the Stage 9b supervisor write tools + capacity gate.

All Postgres-free / Docker-free: the app-DB connection is a layered
``AsyncMock`` (the ``tests/unit/test_events.py`` pattern), the graph is a
stub exposing ``aget_state`` / ``aupdate_state``, and the spawn / stop seams
are recording mocks. We exercise the plain ``aspawn_strategy`` /
``aretire_strategy`` impls directly (the @tool shells are thin ContextVar
readers, surface-checked in ``test_write_tools_surface``).

The capacity-gate refusal test is the money-relevant assertion: if a spawn
ever slips past ``active >= MAX_CONCURRENT_STRATEGIES``, the operator's
single-host container budget is gone.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from orchestrator.gates.thresholds import MAX_CONCURRENT_STRATEGIES
from orchestrator.supervisor import (
    aretire_strategy,
    aspawn_strategy,
)


def _make_conn(*, group_by_rows: list[tuple[str, int]]) -> tuple[AsyncMock, list[tuple[Any, ...]]]:
    """Mocked async conn: ``fetchall`` feeds the portfolio GROUP BY; every
    ``execute`` is captured into ``executed`` as ``(sql, params)``."""
    executed: list[tuple[Any, ...]] = []

    cur = AsyncMock()

    async def _execute(sql: str, params: tuple[Any, ...] | None = None) -> None:
        executed.append((sql, params))

    cur.execute.side_effect = _execute
    cur.fetchall.return_value = group_by_rows

    cursor_cm = AsyncMock()
    cursor_cm.__aenter__.return_value = cur
    cursor_cm.__aexit__.return_value = None

    conn = AsyncMock()
    conn.cursor = lambda: cursor_cm
    return conn, executed


def _inserts(executed: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [(s, p) for s, p in executed if s.strip().upper().startswith("INSERT")]


def _updates(executed: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [(s, p) for s, p in executed if s.strip().upper().startswith("UPDATE")]


class _StubGraph:
    """Graph stand-in for retire: records aget_state / aupdate_state calls."""

    def __init__(self, values: dict[str, Any] | None) -> None:
        self._values = values
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(values=self._values)

    async def aupdate_state(self, config: dict[str, Any], update: dict[str, Any]) -> None:
        self.updates.append((config, update))


# ════════════════════════════════════════════════════════════════════════
# aspawn_strategy — capacity gate
# ════════════════════════════════════════════════════════════════════════


async def test_spawn_under_cap_inserts_row_and_kicks_thread_once() -> None:
    """active=3 < cap(4) → registry row inserted + spawn_thread_fn called once."""
    assert MAX_CONCURRENT_STRATEGIES == 4  # guards the fixture's active=3 intent
    conn, executed = _make_conn(
        group_by_rows=[("research", 1), ("paper", 1), ("live", 1), ("archived", 9)]
    )
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn)

    assert result["spawned"] is True
    assert result["stage"] == "research"
    sid = result["strategy_id"]
    assert result["thread_id"] == f"strategy_{sid}"

    # Exactly one registry INSERT at stage='research'.
    inserts = _inserts(executed)
    assert len(inserts) == 1
    insert_sql, insert_params = inserts[0]
    assert "strategy_registry" in insert_sql
    assert "'research'" in insert_sql
    assert insert_params[0] == sid  # strategy_id is the first column

    # spawn_thread_fn called exactly once, with the minted strategy_id.
    spawn_fn.assert_awaited_once_with(sid)


async def test_spawn_at_cap_refused_no_row_no_kick() -> None:
    """active=4 == cap(4) → refused: no INSERT, spawn_thread_fn NOT called.

    This is the money assertion. A breach here burns the host budget.
    """
    conn, executed = _make_conn(
        group_by_rows=[("research", 2), ("paper", 1), ("live", 1), ("archived", 3)]
    )  # active = 2 + 1 + 1 = 4 == MAX_CONCURRENT_STRATEGIES
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn)

    assert result == {
        "spawned": False,
        "reason": "capacity_exceeded",
        "active": 4,
        "limit": 4,
    }
    assert _inserts(executed) == []
    spawn_fn.assert_not_called()


async def test_spawn_over_cap_refused() -> None:
    """Belt-and-braces: active strictly above the cap is also refused."""
    conn, executed = _make_conn(group_by_rows=[("paper", 5), ("archived", 1)])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn)

    assert result["spawned"] is False
    assert result["active"] == 5
    spawn_fn.assert_not_called()


async def test_spawn_passes_explicit_params_into_registry() -> None:
    """Explicit name/template/pairs/timeframe land in the INSERT params."""
    conn, executed = _make_conn(group_by_rows=[])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(
        spawn_fn,
        conn,
        name="momentum-1",
        template="freqai_classifier_template",
        pairs=["BTC/USDT"],
        timeframe="1h",
    )

    _sql, params = _inserts(executed)[0]
    # (strategy_id, thread_id, name, template, pairs_json, timeframe)
    assert params[2] == "momentum-1"
    assert params[3] == "freqai_classifier_template"
    assert params[5] == "1h"
    assert result["spawned"] is True


# ── Live-cap semantic: enforced at the live transition, NOT at spawn ───


async def test_spawn_ignores_live_cap() -> None:
    """A live strategy already at MAX_CONCURRENT_LIVE_STRATEGIES does NOT block
    a research spawn — the live cap is a transition-time constraint, not a
    spawn-time one. With live=1 (at cap) but active=2 (< total cap), the spawn
    proceeds. Proves the chosen fork-A semantic.
    """
    conn, executed = _make_conn(group_by_rows=[("live", 1), ("research", 1), ("archived", 4)])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn)

    assert result["spawned"] is True  # live at cap did NOT block the spawn
    assert len(_inserts(executed)) == 1
    spawn_fn.assert_awaited_once()


# ════════════════════════════════════════════════════════════════════════
# aspawn_strategy — spawn-vocabulary gate (D-16)
# ════════════════════════════════════════════════════════════════════════
#
# The supervisor agent can hallucinate templates/pairs that do not exist (the
# 10d real-trace proposed templates ``stat_arb`` / ``momentum`` and pair
# ``AVAX/USDT``). The gate REJECTS such an action at the spawn boundary — no
# registry row, no thread kick — so a hallucination never seeds an unfillable
# row or a dataless pair. Same no-write contract as the capacity refusal.


async def test_spawn_unknown_template_refused_no_row_no_kick() -> None:
    """A template outside the shipped set is rejected before any write."""
    conn, executed = _make_conn(group_by_rows=[])  # active=0 — capacity is fine
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn, template="stat_arb")

    assert result["spawned"] is False
    assert result["reason"] == "unknown_template"
    assert result["template"] == "stat_arb"
    assert _inserts(executed) == []  # the load-bearing assertion: no registry row
    spawn_fn.assert_not_called()


async def test_spawn_pair_outside_universe_refused_no_row_no_kick() -> None:
    """A pair outside the SPEC §1 Q2 universe is rejected before any write; only
    the offending pair is flagged (BTC/USDT is valid, AVAX/USDT is not)."""
    conn, executed = _make_conn(group_by_rows=[])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn, pairs=["BTC/USDT", "AVAX/USDT"])

    assert result["spawned"] is False
    assert result["reason"] == "pairs_outside_universe"
    assert result["invalid_pairs"] == ["AVAX/USDT"]
    assert _inserts(executed) == []
    spawn_fn.assert_not_called()


async def test_spawn_valid_template_and_pairs_still_spawns() -> None:
    """A fully-valid action (shipped template + in-universe pairs) spawns
    normally — the gate is a precise filter, not a blanket block."""
    conn, executed = _make_conn(group_by_rows=[])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(
        spawn_fn,
        conn,
        template="mean_reversion_template",
        pairs=["BTC/USDT", "ETH/USDT"],
    )

    assert result["spawned"] is True
    assert len(_inserts(executed)) == 1
    spawn_fn.assert_awaited_once()


async def test_spawn_omitted_template_and_pairs_passes_the_gate() -> None:
    """Omitting template/pairs (the common case — researcher picks the template,
    default universe applies) is NOT a vocabulary violation: ``None`` passes the
    gate, so the row seeds with the 'pending' sentinel + default pairs."""
    conn, executed = _make_conn(group_by_rows=[])
    spawn_fn = AsyncMock()

    result = await aspawn_strategy(spawn_fn, conn)  # no template, no pairs

    assert result["spawned"] is True
    assert len(_inserts(executed)) == 1
    spawn_fn.assert_awaited_once()


# ════════════════════════════════════════════════════════════════════════
# aretire_strategy
# ════════════════════════════════════════════════════════════════════════


async def test_retire_paper_thread_archives_state_and_registry_no_container() -> None:
    """A paper-stage thread → aupdate_state archived + registry UPDATE; the
    live-container stop seam is NOT called (only live threads have one)."""
    conn, executed = _make_conn(group_by_rows=[])
    graph = _StubGraph({"stage": "paper", "strategy_id": "s_paper"})
    stop_fn = AsyncMock()

    result = await aretire_strategy(graph, conn, "s_paper", stop_live_container_fn=stop_fn)

    assert result == {"retired": True, "strategy_id": "s_paper", "previous_stage": "paper"}

    # aupdate_state set stage=archived + retire reason.
    assert len(graph.updates) == 1
    _config, update = graph.updates[0]
    assert update["stage"] == "archived"
    assert update["failure_reason"] == "retired_by_supervisor"

    # registry UPDATE to archived.
    updates = _updates(executed)
    assert len(updates) == 1
    upd_sql, upd_params = updates[0]
    assert "stage = 'archived'" in upd_sql
    assert upd_params[-1] == "s_paper"

    stop_fn.assert_not_called()  # not a live thread


async def test_retire_live_thread_stops_container_via_seam() -> None:
    """A live-stage thread → stop_live_container_fn awaited once + archived."""
    conn, executed = _make_conn(group_by_rows=[])
    graph = _StubGraph({"stage": "live", "strategy_id": "s_live"})
    stop_fn = AsyncMock()

    result = await aretire_strategy(graph, conn, "s_live", stop_live_container_fn=stop_fn)

    assert result["retired"] is True
    assert result["previous_stage"] == "live"
    stop_fn.assert_awaited_once_with("s_live")
    assert len(graph.updates) == 1
    assert len(_updates(executed)) == 1


async def test_retire_live_thread_archives_even_if_container_stop_fails() -> None:
    """A stop_live_container failure is logged, not raised; the archive still
    completes (best-effort teardown, operator reviews the orphan)."""
    conn, executed = _make_conn(group_by_rows=[])
    graph = _StubGraph({"stage": "live", "strategy_id": "s_live"})
    stop_fn = AsyncMock(side_effect=RuntimeError("compose down boom"))

    result = await aretire_strategy(graph, conn, "s_live", stop_live_container_fn=stop_fn)

    assert result["retired"] is True
    assert len(graph.updates) == 1  # archived despite the stop failure
    assert len(_updates(executed)) == 1


async def test_retire_unknown_strategy_is_graceful_noop() -> None:
    """No checkpoint (empty values) → no-op named result; no writes, no stop."""
    conn, executed = _make_conn(group_by_rows=[])
    graph = _StubGraph({})  # empty values → unknown
    stop_fn = AsyncMock()

    result = await aretire_strategy(graph, conn, "ghost", stop_live_container_fn=stop_fn)

    assert result == {"retired": False, "reason": "unknown_strategy", "strategy_id": "ghost"}
    assert graph.updates == []
    assert _updates(executed) == []
    stop_fn.assert_not_called()


# ════════════════════════════════════════════════════════════════════════
# caller-owns-commit + tool surface
# ════════════════════════════════════════════════════════════════════════


async def test_spawn_does_not_commit() -> None:
    """The 9c runner owns the transaction; aspawn must NOT commit."""
    conn, _executed = _make_conn(group_by_rows=[])
    await aspawn_strategy(AsyncMock(), conn)
    conn.commit.assert_not_called()


async def test_retire_does_not_commit() -> None:
    """The 9c runner owns the transaction; aretire must NOT commit."""
    conn, _executed = _make_conn(group_by_rows=[])
    graph = _StubGraph({"stage": "paper"})
    await aretire_strategy(graph, conn, "s1", stop_live_container_fn=AsyncMock())
    conn.commit.assert_not_called()


# NOTE: the 9b write @tool shells (spawn_strategy / retire_strategy) + their
# WRITE_TOOLS export were removed in 9c (Arch 2 — the runner calls the plain
# aspawn_strategy / aretire_strategy impls from the structured decision, so the
# agent-facing shells had no caller). The plain impls — exercised by every test
# above — remain. The old test_write_tools_surface was removed with the shells.
