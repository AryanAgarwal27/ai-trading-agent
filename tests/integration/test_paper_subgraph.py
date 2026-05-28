"""Stage 7e integration tests for the full paper subgraph (BRD §5.5).

All tests use real Postgres (paper_spawn + paper_teardown write the
strategy_registry) but stub every other external effect: the container
spawn/stop (no Docker), the metric fetch (no live Freqtrade), and the
monitor agent (no real Haiku). The checkpointer is InMemorySaver so
interrupts work without the Postgres checkpointer.

Marked ``integration`` (real app DB), NOT ``freqtrade`` (no Docker).

Coverage:
  - full cycle: spawn → wake → monitor(rearm) → re-park → wake →
    monitor(advance) → divergence_check(pass, elapsed>=30) → live_gate →
    autoapprove → stage="live"
  - reject path: monitor(advance) → live_gate → autoreject →
    paper_teardown → archive
  - kill path: monitor(kill) → paper_teardown → archive
  - divergence_check OVERRIDE: monitor(advance) but elapsed<30 → forced
    back to paper_wait (the deterministic-backstop guarantee)
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.agents.monitors import PaperMonitorContext, PaperMonitorVerdict
from orchestrator.gates.hitl import autoresume_for_test
from orchestrator.subgraphs.paper import build_paper_subgraph

pytestmark = pytest.mark.integration


# ───────────────────────── db cleanup ─────────────────────────


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
            await cur.execute(
                "DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,)
            )
        await conn.commit()


# ───────────────────────── helpers ─────────────────────────


def _minimal_paper_state(strategy_id: str, paper_started_at: str | None) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"generated_strategy_path": "strategy_templates/mean_reversion_template.py"}
    if paper_started_at is not None:
        artifacts["paper_started_at"] = paper_started_at
    return {
        "strategy_id": strategy_id,
        "name": f"test-{strategy_id}",
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "params": {"stake_amount": 25.0},
        "stage": "paper_gate",
        "agent_votes": [],
        "gate_decisions": {},
        "freqtrade_api_url": None,
        "freqtrade_userdir": None,
        "freqtrade_process_id": None,
        "artifacts": artifacts,
        "failure_reason": None,
    }


def _spawn_stub() -> Any:
    async def _stub(*, port: int, **_kwargs: Any) -> str:
        return f"http://127.0.0.1:{port}"

    return _stub


def _ctx_stub(*, max_drawdown: float = 0.04, trades: list[dict[str, Any]] | None = None) -> Any:
    async def _stub(state: dict[str, Any]) -> PaperMonitorContext:
        # elapsed_days mirrors what divergence_check will independently
        # recompute from paper_started_at; the agent stub ignores it anyway.
        return PaperMonitorContext(
            status=[],
            profit={"max_drawdown": max_drawdown},
            trades=trades or [],
            performance=[],
            backtest_returns=[],
            kill_switch_fired=False,
            elapsed_days=0.0,
        )

    return _stub


def _monitor_fixed(decision: str) -> Any:
    async def _stub(_ctx: PaperMonitorContext) -> PaperMonitorVerdict:
        return PaperMonitorVerdict(
            decision=decision,  # type: ignore[arg-type]
            primary_observation=f"obs-{decision}",
            rationale=f"rationale-{decision}",
            confidence=0.9,
        )

    return _stub


def _monitor_sequence(decisions: list[str]) -> Any:
    calls = {"n": 0}

    async def _stub(_ctx: PaperMonitorContext) -> PaperMonitorVerdict:
        idx = min(calls["n"], len(decisions) - 1)
        decision = decisions[idx]
        calls["n"] += 1
        return PaperMonitorVerdict(
            decision=decision,  # type: ignore[arg-type]
            primary_observation=f"obs-{decision}",
            rationale=f"rationale-{decision}",
            confidence=0.9,
        )

    return _stub


def _stop_stub(record: dict[str, Any]) -> Any:
    async def _stub(strategy_id: str) -> None:
        record["stopped"] = strategy_id

    return _stub


async def _parked_kind(graph: Any, config: dict[str, Any]) -> str | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()):
            val = intr.value
            if isinstance(val, dict):
                return val.get("kind")
    return None


# ───────────────────────── tests ─────────────────────────


async def test_full_cycle_rearm_then_advance_to_live(
    cleanup_strategy_ids: list[str], hitl_autoapprove: Any
) -> None:
    """spawn → wake → rearm → wake → advance → live_gate → approve → live."""
    strategy_id = f"pe-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    # 31 days ago so divergence_check sees elapsed >= MIN_PAPER_DAYS.
    started = (datetime.now(UTC) - timedelta(days=31)).isoformat()

    graph = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_sequence(["rearm", "advance"]),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub({}),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}

    # Run to first park.
    async for _ in graph.astream(_minimal_paper_state(strategy_id, started), config=config):
        pass
    assert await _parked_kind(graph, config) == "paper_wait"

    # Wake 1 → monitor(rearm) → divergence_check(rearm) → paper_wait again.
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "paper_wait"

    # Wake 2 → monitor(advance) → divergence_check(advance) → live_gate.
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "live_gate"

    # Approve → END with stage="live".
    await hitl_autoapprove(graph, thread_id)
    final = await graph.aget_state(config)
    assert final.values["stage"] == "live"
    assert final.values["gate_decisions"]["live"]["approved"] is True
    # Two monitor wakes → two votes appended via the reducer.
    votes = [v for v in final.values["agent_votes"] if v["agent"] == "paper_monitor"]
    assert len(votes) == 2


async def test_divergence_check_overrides_premature_advance(
    cleanup_strategy_ids: list[str],
) -> None:
    """monitor votes advance but elapsed<30 → forced back to paper_wait.

    The deterministic-backstop test. The LLM stub unconditionally votes
    advance; the clock is ~0 days; divergence_check MUST override to rearm.
    """
    strategy_id = f"pe-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    # paper_started_at = now → elapsed ~0, well below MIN_PAPER_DAYS.
    started = datetime.now(UTC).isoformat()

    graph = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed("advance"),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub({}),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_minimal_paper_state(strategy_id, started), config=config):
        pass
    assert await _parked_kind(graph, config) == "paper_wait"

    # Wake → monitor(advance) → divergence_check OVERRIDE → paper_wait.
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "paper_wait", (
        "divergence_check failed to override premature advance; thread should "
        "be re-parked at paper_wait, not advanced to live_gate"
    )

    snap = await graph.aget_state(config)
    dc = snap.values["gate_decisions"]["divergence_check"]
    assert dc["monitor_decision"] == "advance"
    assert dc["effective_decision"] == "rearm"
    assert any("premature_advance" in o for o in dc["overrides"])


async def test_reject_path_tears_down_and_archives(
    cleanup_strategy_ids: list[str], hitl_autoreject: Any
) -> None:
    """monitor(advance) → live_gate → reject → paper_teardown → archive."""
    strategy_id = f"pe-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    started = (datetime.now(UTC) - timedelta(days=31)).isoformat()

    stop_rec: dict[str, Any] = {}
    graph = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed("advance"),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub(stop_rec),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_minimal_paper_state(strategy_id, started), config=config):
        pass
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "live_gate"

    await hitl_autoreject(graph, thread_id, notes="not convinced")
    final = await graph.aget_state(config)
    assert final.values["stage"] == "archived"
    assert final.values["failure_reason"].startswith("live_gate_rejected:")
    assert stop_rec.get("stopped") == strategy_id

    # Registry row should be marked archived by paper_teardown.
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT stage FROM strategy_registry WHERE strategy_id = %s", (strategy_id,)
            )
            row = await cur.fetchone()
    assert row is not None and row[0] == "archived"


async def test_kill_path_tears_down_and_archives(
    cleanup_strategy_ids: list[str],
) -> None:
    """monitor(kill) → divergence_check(kill) → paper_teardown → archive."""
    strategy_id = f"pe-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    started = (datetime.now(UTC) - timedelta(days=10)).isoformat()

    stop_rec: dict[str, Any] = {}
    graph = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed("kill"),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub(stop_rec),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_minimal_paper_state(strategy_id, started), config=config):
        pass
    await autoresume_for_test(graph, thread_id, {"wake": True})

    final = await graph.aget_state(config)
    assert final.values["stage"] == "archived"
    assert "paper_divergence_kill" in final.values["failure_reason"]
    assert stop_rec.get("stopped") == strategy_id


async def test_hard_drawdown_overrides_to_kill(
    cleanup_strategy_ids: list[str],
) -> None:
    """monitor votes rearm but drawdown >= KILL_SWITCH_DRAWDOWN → forced kill.

    Second deterministic-backstop test: the LLM is content to keep going,
    but the hard drawdown threshold forces a kill regardless.
    """
    strategy_id = f"pe-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    started = (datetime.now(UTC) - timedelta(days=10)).isoformat()

    stop_rec: dict[str, Any] = {}
    graph = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed("rearm"),
        build_context_fn=_ctx_stub(max_drawdown=0.15),  # > 0.12 kill threshold
        stop_container_fn=_stop_stub(stop_rec),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_minimal_paper_state(strategy_id, started), config=config):
        pass
    await autoresume_for_test(graph, thread_id, {"wake": True})

    final = await graph.aget_state(config)
    assert final.values["stage"] == "archived"
    dc = final.values["gate_decisions"]["divergence_check"]
    assert dc["monitor_decision"] == "rearm"
    assert dc["effective_decision"] == "kill"
    assert any("hard_kill" in o for o in dc["overrides"])
    assert stop_rec.get("stopped") == strategy_id
