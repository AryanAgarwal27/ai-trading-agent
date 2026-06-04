"""Unit tests for ``run_supervisor`` — the Stage 9c orchestrator (Arch 2).

The runner's seam design (mocked-conn + injected ``agent`` /
``spawn_thread_fn`` / ``audit_writer_fn`` + a stub graph) makes its logic
fully exercisable WITHOUT Postgres, so these run in CI — the money-path runner
gets coverage now rather than waiting for the 9g integration job. A thinner
real-Postgres end-to-end lives in ``tests/integration/test_supervisor_loop.py``
for true-DB fidelity (operator-run).

Covered:
  - empty portfolio + spawn decision → registry INSERT, producer kicked once
    POST-commit, telemetry written, single commit;
  - at capacity → spawn refused by the gate, refusal captured in telemetry,
    producer NOT called, still one commit;
  - retire decision → graph.aupdate_state + registry UPDATE + telemetry;
  - no_op decision → no mutations, telemetry still written (audit-worthy);
  - agent flake (raise) and malformed output → no_op fallback, flake=True,
    run does not crash, commit still happens;
  - read-tool ContextVars reset to defaults after the run (runner-scoped);
  - telemetry payload shape;
  - two spawns in one decision → exactly ONE batch commit;
  - exception mid-run → rollback + re-raise, no commit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langgraph.store.memory import InMemoryStore

from orchestrator.supervisor import (
    SupervisorAction,
    SupervisorDecision,
    _current_portfolio,
    _current_regime,
    _current_store,
    _current_strategies,
    run_supervisor,
)

# ─── fakes ──────────────────────────────────────────────────────────────


class _FakeCursor:
    def __init__(
        self,
        fetchall_map: dict[str, Any],
        fetchone_map: dict[str, Any],
        executed: list[tuple[Any, ...]],
    ) -> None:
        self._fa = fetchall_map
        self._fo = fetchone_map
        self._executed = executed
        self._last = ""

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self._executed.append((sql, params))
        self._last = sql

    async def fetchall(self) -> Any:
        for key, val in self._fa.items():
            if key in self._last:
                return val
        return []

    async def fetchone(self) -> Any:
        for key, val in self._fo.items():
            if key in self._last:
                return val
        return None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False


class _FakeConn:
    """Routes fetch results by SQL substring; counts commit/rollback."""

    def __init__(
        self,
        *,
        fetchall_map: dict[str, Any] | None = None,
        fetchone_map: dict[str, Any] | None = None,
    ) -> None:
        self.executed: list[tuple[Any, ...]] = []
        self._fa = fetchall_map or {}
        self._fo = fetchone_map or {}
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._fa, self._fo, self.executed)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        pass


class _StubGraph:
    def __init__(self, state_values: dict[str, Any] | None = None) -> None:
        self._values = state_values
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(values=self._values)

    async def aupdate_state(self, config: dict[str, Any], update: dict[str, Any]) -> None:
        self.updates.append((config, update))


class _StubAgent:
    def __init__(
        self,
        *,
        decision: SupervisorDecision | None = None,
        raises: Exception | None = None,
        bad_response: bool = False,
    ) -> None:
        self._decision = decision
        self._raises = raises
        self._bad = bad_response
        self.calls: list[Any] = []

    async def ainvoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.calls.append((payload, config))
        if self._raises is not None:
            raise self._raises
        if self._bad:
            return {"structured_response": "garbage-not-a-decision"}
        return {"structured_response": self._decision}


def _audit_capture() -> tuple[Any, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    async def _audit(conn: Any, metrics: dict[str, Any]) -> None:
        captured.append(metrics)

    return _audit, captured


def _inserts(conn: _FakeConn) -> list[tuple[Any, ...]]:
    return [(s, p) for s, p in conn.executed if s.strip().upper().startswith("INSERT")]


def _updates(conn: _FakeConn) -> list[tuple[Any, ...]]:
    return [(s, p) for s, p in conn.executed if s.strip().upper().startswith("UPDATE")]


def _spawn_decision(name: str = "strat_a") -> SupervisorDecision:
    return SupervisorDecision(
        actions=[SupervisorAction(action="spawn", name=name, rationale="free slot, win in regime")],
        overall_rationale="one spawn",
        confidence=0.8,
    )


# ─── empty portfolio + spawn ───────────────────────────────────────────


async def test_spawn_inserts_row_kicks_producer_postcommit_logs_once() -> None:
    conn = _FakeConn()  # all reads empty → active=0, regime unknown, no strategies
    graph = _StubGraph()
    spawn_rec = AsyncMock()
    audit, metrics = _audit_capture()

    decision = await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(decision=_spawn_decision()),
        spawn_thread_fn=spawn_rec,
        audit_writer_fn=audit,
    )

    inserts = _inserts(conn)
    assert len(inserts) == 1  # the registry research row
    sid = inserts[0][1][0]

    spawn_rec.assert_awaited_once_with(sid)  # producer kicked post-commit, with the minted id
    assert conn.commits == 1
    assert conn.rollbacks == 0
    assert len(metrics) == 1
    m = metrics[0]
    assert m["trigger"] == "cron"
    assert m["flake"] is False
    assert m["decision"]["actions"][0]["action"] == "spawn"
    assert m["action_results"][0]["result"]["spawned"] is True
    assert decision.actions[0].action == "spawn"


# ─── at capacity → tool refuses, audit still records the intent ─────────


async def test_spawn_at_capacity_refused_but_logged() -> None:
    # active = 2 + 1 + 1 = 4 == MAX_CONCURRENT_STRATEGIES
    conn = _FakeConn(fetchall_map={"GROUP BY": [("research", 2), ("paper", 1), ("live", 1)]})
    graph = _StubGraph()
    spawn_rec = AsyncMock()
    audit, metrics = _audit_capture()

    await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="event",
        agent=_StubAgent(decision=_spawn_decision("strat_e")),
        spawn_thread_fn=spawn_rec,
        audit_writer_fn=audit,
    )

    assert _inserts(conn) == []  # gate refused → no row
    spawn_rec.assert_not_called()  # nothing enqueued → nothing drained
    assert conn.commits == 1  # still committed (the decision is auditable)
    # The refusal is observable in telemetry — operator audit shows the decision
    # AND that the tool refused it.
    result = metrics[0]["action_results"][0]["result"]
    assert result["spawned"] is False
    assert result["reason"] == "capacity_exceeded"


# ─── retire ─────────────────────────────────────────────────────────────


async def test_retire_updates_state_registry_and_logs() -> None:
    conn = _FakeConn()
    graph = _StubGraph(state_values={"stage": "paper", "strategy_id": "s_x"})
    audit, metrics = _audit_capture()
    # 9e: inject the completion publisher seam (default would hit a real Redis
    # socket) and assert the retire emits thread_completed POST-commit, once.
    completions: list[tuple[str, dict[str, Any]]] = []

    async def _publish(strategy_id: str, payload: dict[str, Any]) -> None:
        completions.append((strategy_id, payload))

    decision = SupervisorDecision(
        actions=[SupervisorAction(action="retire", strategy_id="s_x", rationale="stalled 30d")],
        overall_rationale="free a slot",
        confidence=0.7,
    )

    await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="manual",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=audit,
        completion_publisher_fn=_publish,
    )

    assert len(graph.updates) == 1
    assert graph.updates[0][1]["stage"] == "archived"
    archive_updates = [u for u in _updates(conn) if "stage = 'archived'" in u[0]]
    assert len(archive_updates) == 1
    assert conn.commits == 1
    assert metrics[0]["action_results"][0]["result"]["retired"] is True
    # 9e point C: exactly one thread_completed for the archived strategy, shaped
    # per the publish_thread_completed contract.
    assert len(completions) == 1
    sid, payload = completions[0]
    assert sid == "s_x"
    assert payload["strategy_id"] == "s_x"
    assert payload["final_stage"] == "archived"
    assert payload["completion_reason"] == "retired_by_supervisor"
    assert "completed_at" in payload


async def test_failed_retire_does_not_emit_completion() -> None:
    """9e no-phantom guard: a retire that did NOT archive a row (unknown
    strategy → retired=False) emits no thread_completed event."""
    conn = _FakeConn()
    graph = _StubGraph(state_values={})  # empty → aretire returns retired=False
    audit, _metrics = _audit_capture()
    completions: list[str] = []

    async def _publish(strategy_id: str, payload: dict[str, Any]) -> None:
        completions.append(strategy_id)

    decision = SupervisorDecision(
        actions=[SupervisorAction(action="retire", strategy_id="ghost", rationale="gone")],
        overall_rationale="try retire a phantom",
        confidence=0.5,
    )

    await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="event",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=audit,
        completion_publisher_fn=_publish,
    )

    assert completions == []  # no archived row → no event


# ─── no_op ────────────────────────────────────────────────────────────────


async def test_no_op_decision_mutates_nothing_but_logs() -> None:
    conn = _FakeConn()
    graph = _StubGraph()
    spawn_rec = AsyncMock()
    audit, metrics = _audit_capture()

    decision = SupervisorDecision(
        actions=[], overall_rationale="let existing threads run", confidence=0.6
    )

    await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=spawn_rec,
        audit_writer_fn=audit,
    )

    assert _inserts(conn) == []
    assert graph.updates == []
    spawn_rec.assert_not_called()
    assert conn.commits == 1  # an empty decision is still an audit-worthy run
    assert metrics[0]["decision"]["actions"] == []


# ─── agent flake → no_op fallback (does not crash) ──────────────────────


async def test_agent_raise_falls_back_to_no_op() -> None:
    conn = _FakeConn()
    audit, metrics = _audit_capture()

    decision = await run_supervisor(
        _StubGraph(),
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(raises=RuntimeError("llm 500")),  # stands in for ValidationError too
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=audit,
    )

    assert decision.actions == []  # no_op fallback
    assert conn.commits == 1  # run did NOT crash
    assert metrics[0]["flake"] is True


async def test_agent_malformed_output_falls_back_to_no_op() -> None:
    conn = _FakeConn()
    audit, metrics = _audit_capture()

    decision = await run_supervisor(
        _StubGraph(),
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(bad_response=True),
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=audit,
    )

    assert decision.actions == []
    assert metrics[0]["flake"] is True
    assert conn.commits == 1


# ─── ContextVar isolation (runner-scoped) ──────────────────────────────


async def test_contextvars_reset_after_run() -> None:
    conn = _FakeConn(fetchone_map={"FROM regime_log": ("high_vol_up",)})
    decision = SupervisorDecision(actions=[], overall_rationale="x", confidence=0.5)

    await run_supervisor(
        _StubGraph(),
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=_audit_capture()[0],
    )

    # All read-tool ContextVars back to their module defaults.
    assert _current_store.get() is None
    assert _current_regime.get() == "unknown"
    assert _current_portfolio.get() is None
    assert _current_strategies.get() is None


# ─── telemetry payload shape ────────────────────────────────────────────


async def test_telemetry_payload_shape() -> None:
    conn = _FakeConn()
    audit, metrics = _audit_capture()

    await run_supervisor(
        _StubGraph(),
        InMemoryStore(),
        conn,
        trigger="manual",
        agent=_StubAgent(decision=_spawn_decision()),
        spawn_thread_fn=AsyncMock(),
        audit_writer_fn=audit,
    )

    m = metrics[0]
    assert set(m) >= {"trigger", "flake", "decision", "action_results"}
    assert set(m["decision"]) >= {"actions", "overall_rationale", "confidence"}
    assert m["trigger"] == "manual"


# ─── single batch commit ────────────────────────────────────────────────


async def test_two_spawns_single_commit() -> None:
    conn = _FakeConn()  # active=0 throughout (fake doesn't track inserts) → both pass the gate
    spawn_rec = AsyncMock()
    audit, _metrics = _audit_capture()

    decision = SupervisorDecision(
        actions=[
            SupervisorAction(action="spawn", name="a", rationale="slot 1"),
            SupervisorAction(action="spawn", name="b", rationale="slot 2"),
        ],
        overall_rationale="two candidates",
        confidence=0.7,
    )

    await run_supervisor(
        _StubGraph(),
        InMemoryStore(),
        conn,
        trigger="cron",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=spawn_rec,
        audit_writer_fn=audit,
    )

    assert len(_inserts(conn)) == 2
    assert conn.commits == 1  # ONE batch commit, not one-per-action
    assert spawn_rec.await_count == 2  # both producers kicked post-commit


# ─── exception mid-run → rollback + re-raise ───────────────────────────


async def test_dry_run_skips_writes_and_telemetry() -> None:
    """dry_run=True: read+reason path runs, but no action execution, no
    telemetry, no commit — rollback instead. Decision returned unchanged."""
    conn = _FakeConn()
    # state_values would make the retire action archive a live thread IF executed.
    graph = _StubGraph(state_values={"stage": "paper", "strategy_id": "s_x"})
    spawn_rec = AsyncMock()
    audit, metrics = _audit_capture()

    decision = SupervisorDecision(
        actions=[
            SupervisorAction(action="spawn", name="x", rationale="would spawn"),
            SupervisorAction(action="retire", strategy_id="s_x", rationale="would retire"),
        ],
        overall_rationale="dry run",
        confidence=0.6,
    )

    result = await run_supervisor(
        graph,
        InMemoryStore(),
        conn,
        trigger="manual",
        agent=_StubAgent(decision=decision),
        spawn_thread_fn=spawn_rec,
        audit_writer_fn=audit,
        dry_run=True,
    )

    assert result is decision  # the agent's decision returned unchanged
    assert conn.commits == 0  # NO commit
    assert conn.rollbacks == 1  # rolled back instead
    assert metrics == []  # NO telemetry row
    spawn_rec.assert_not_called()  # spawn action NOT executed
    assert graph.updates == []  # retire action NOT executed (no aupdate_state)
    assert _inserts(conn) == []  # no spawn INSERT
    assert _updates(conn) == []  # no retire registry UPDATE


async def test_exception_rolls_back_and_reraises() -> None:
    conn = _FakeConn()

    async def _audit_boom(c: Any, m: dict[str, Any]) -> None:
        raise RuntimeError("telemetry write boom")

    with pytest.raises(RuntimeError, match="telemetry write boom"):
        await run_supervisor(
            _StubGraph(),
            InMemoryStore(),
            conn,
            trigger="cron",
            agent=_StubAgent(decision=_spawn_decision()),
            spawn_thread_fn=AsyncMock(),
            audit_writer_fn=_audit_boom,
        )

    assert conn.rollbacks == 1
    assert conn.commits == 0  # nothing committed on failure
