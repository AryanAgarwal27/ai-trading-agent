"""Stage 7g integration tests — the composed parent graph.

Proves the parent graph composes research → validation → paper and that
a strategy flows from a research idea all the way to live-gate-approved
through the REAL parent graph (not just isolated subgraph tests). The
subgraph INTERNALS are covered by test_validation_subgraph /
test_paper_subgraph; here we stub the subgraphs down to what's needed to
exercise parent ROUTING + nested-interrupt surfacing:

  - research_subgraph: a passthrough (stage stays non-archived → routes
    to validation).
  - validation_subgraph: START → real paper_gate node → END (the real
    interrupt payload, kind="paper_gate", surfacing through nesting).
  - paper_subgraph: the REAL paper subgraph with stubbed leaves (spawn /
    monitor / context / stop) — same approach as test_paper_subgraph.

Marked ``integration`` — paper_spawn writes the strategy_registry.

The second test closes the loop opened by the Stage 6f finding: a real
parent-graph thread parked at paper_gate is now visible to the
``GET /threads`` endpoint path WITHOUT the smoke env override.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from orchestrator.agents.monitors import PaperMonitorContext, PaperMonitorVerdict
from orchestrator.gates.hitl import autoresume_for_test
from orchestrator.graph import build_per_strategy_graph
from orchestrator.main import app
from orchestrator.state import StrategyState
from orchestrator.subgraphs.paper import build_paper_subgraph
from orchestrator.subgraphs.validation import ValidationState, paper_gate

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-7g-parent"


# ───────────────────────── stub leaves ─────────────────────────


def _spawn_stub() -> Any:
    async def _stub(*, port: int, **_kwargs: Any) -> str:
        return f"http://127.0.0.1:{port}"

    return _stub


def _ctx_stub() -> Any:
    async def _stub(_state: Any) -> PaperMonitorContext:
        return PaperMonitorContext(profit={"max_drawdown": 0.04})

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


def _stop_stub(rec: dict[str, Any]) -> Any:
    async def _stub(strategy_id: str) -> None:
        rec["stopped"] = strategy_id

    return _stub


# ───────────────────────── stub-composed parent graph ──────────────────


def _research_passthrough(_state: StrategyState) -> dict[str, Any]:
    return {}


_LIVE_CONTINUE_SNAP = {
    "max_drawdown": 0.0,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [0.01, 0.02],
    "paper_returns": [0.01, 0.02],
    "current_regime": "mid_vol_up",
    "approval_regime": "mid_vol_up",
}
# risk_check -> pause (drawdown), regime_check -> pause (regime shift): 2 pause
# majority -> coordinator pause -> live_pause.
_LIVE_PAUSE_SNAP = {
    "max_drawdown": 0.15,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [],
    "paper_returns": [],
    "current_regime": "high_vol_down",
    "approval_regime": "low_vol_up",
}


def _stub_live_subgraph(*, snapshot: dict[str, Any] | None = None) -> Any:
    """Real live subgraph (8e topology) with every external effect stubbed.

    registry_writer_fn keeps its real default (real Postgres registry row from
    live_spawn — the parent tests run against real PG and clean up by id). All
    Docker / LLM / REST / start-trading / bump effects are stubbed so the
    composed parent flows through live_spawn → live_wait without infra.
    """
    from orchestrator.agents.coordinator import LiveVerdict
    from orchestrator.subgraphs.live import build_live_subgraph

    snap = snapshot or _LIVE_CONTINUE_SNAP

    async def _spawn(*, port: int, **_k: Any) -> str:
        return f"http://127.0.0.1:{port}"

    async def _stop_container(_sid: str) -> None:
        return None

    async def _rationale(_facts: dict[str, Any]) -> str:
        return "stub rationale"

    async def _review(_c: dict[str, Any]) -> Any:
        return LiveVerdict(verdict="pause", rationale="perf", confidence=0.8)

    async def _merge(_v: Any) -> Any:
        return LiveVerdict(verdict="pause", rationale="merge", confidence=0.7)

    async def _arbitrate(_v: Any, _m: Any) -> Any:
        return LiveVerdict(verdict="fail", rationale="arb", confidence=0.9)

    async def _noop_state(_s: Any) -> None:
        return None

    async def _noop_bump(_sid: str) -> None:
        return None

    async def _snapshot_fn(_s: Any) -> dict[str, Any]:
        return dict(snap)

    return build_live_subgraph(
        spawn_live_container_fn=_spawn,
        stop_live_container_fn=_stop_container,
        build_snapshot_fn=_snapshot_fn,
        rationale_fn=_rationale,
        review_fn=_review,
        merge_fn=_merge,
        arbitrate_fn=_arbitrate,
        stop_trading_fn=_noop_state,
        start_trading_fn=_noop_state,
        live_started_bump_fn=_noop_bump,
    )


def _build_stub_composed_graph(
    saver: Any,
    *,
    monitor_decision: str = "advance",
    live_snapshot: dict[str, Any] | None = None,
) -> Any:
    """Compose the parent with stub research + paper_gate-only validation
    + real-paper-subgraph-with-stubbed-leaves + stubbed-live-subgraph (8g)."""
    rb: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = StateGraph(
        StrategyState
    )
    rb.add_node("research_pass", _research_passthrough)
    rb.add_edge(START, "research_pass")
    rb.add_edge("research_pass", END)
    research = rb.compile()

    vb: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    vb.add_node("paper_gate", paper_gate)
    vb.add_edge(START, "paper_gate")
    vb.add_edge("paper_gate", END)
    validation = vb.compile()

    paper = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed(monitor_decision),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub({}),
    )

    return build_per_strategy_graph(
        saver,
        None,
        research_subgraph=research,
        validation_subgraph=validation,
        paper_subgraph=paper,
        live_subgraph=_stub_live_subgraph(snapshot=live_snapshot),
    )


def _initial_state(strategy_id: str, *, paper_started_days_ago: int = 31) -> dict[str, Any]:
    started = (datetime.now(UTC) - timedelta(days=paper_started_days_ago)).isoformat()
    return {
        "strategy_id": strategy_id,
        "name": f"test-{strategy_id}",
        "hypothesis": "parent-graph composition test",
        "template": "mean_reversion_template",
        "params": {"stake_amount": 25.0},
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "stage": "research",
        "gate_decisions": {},
        "agent_votes": [],
        "artifacts": {
            "generated_strategy_path": "strategy_templates/mean_reversion_template.py",
            "paper_started_at": started,
        },
    }


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
            # Children first (FK to strategy_registry): live_pause / coordinator
            # may write gate_audits; the 8g live path may write kill_switch_events.
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute(
                "DELETE FROM kill_switch_events WHERE strategy_id = ANY(%s)", (ids,)
            )
            await cur.execute(
                "DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,)
            )
        await conn.commit()


async def _parked_kind(graph: Any, config: dict[str, Any]) -> str | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()):
            val = intr.value
            if isinstance(val, dict):
                return val.get("kind")
    return None


# ───────────────────────── test 1: full pipeline ────────────────────────


async def test_full_pipeline_research_to_live(
    cleanup_strategy_ids: list[str], hitl_autoapprove: Any
) -> None:
    """research → validation(paper_gate) → approve → paper → wake →
    monitor(advance) → live_gate → approve → stage='live'."""
    strategy_id = f"pg-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"

    graph = _build_stub_composed_graph(InMemorySaver(), monitor_decision="advance")
    config = {"configurable": {"thread_id": thread_id}}

    # research (passthrough) → validation → paper_gate interrupt.
    async for _ in graph.astream(_initial_state(strategy_id), config=config):
        pass
    assert await _parked_kind(graph, config) == "paper_gate", (
        "parent graph should park at the nested validation paper_gate"
    )

    # Approve paper_gate → validation completes (stage=paper) → parent
    # routes to paper_subgraph → paper_spawn → schedule_wake → paper_wait.
    await hitl_autoapprove(graph, thread_id)
    assert await _parked_kind(graph, config) == "paper_wait"

    # Wake → paper_monitor(advance) → divergence_check(elapsed>=30) → live_gate.
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "live_gate"

    # Approve live_gate → stage="live" → parent routes into live_subgraph →
    # live_spawn runs → parks at live_wait (8g paper→live handoff).
    await hitl_autoapprove(graph, thread_id)
    final = await graph.aget_state(config)
    assert final.values["stage"] == "live"
    assert final.values["gate_decisions"]["live"]["approved"] is True
    assert await _parked_kind(graph, config) == "live_wait", (
        "8g: after live_gate approve the parent must route into the live "
        "subgraph and park at live_wait (live_spawn ran)"
    )
    # live_spawn wrote the registry row at stage='live'.
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT stage FROM strategy_registry WHERE strategy_id = %s", (strategy_id,)
            )
            row = await cur.fetchone()
    assert row is not None and row[0] == "live", "live_spawn must register stage='live'"


async def test_live_interrupts_propagate_to_parent(
    cleanup_strategy_ids: list[str], hitl_autoapprove: Any
) -> None:
    """8g: the live subgraph's live_wait + live_pause interrupts surface to the
    parent graph (mirrors 7g's nested-interrupt-surfacing lesson for paper).

    A pause-inducing snapshot drives coordinator→pause→live_pause so BOTH live
    interrupts are exercised through the composed parent.
    """
    strategy_id = f"pg-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"

    graph = _build_stub_composed_graph(
        InMemorySaver(), monitor_decision="advance", live_snapshot=_LIVE_PAUSE_SNAP
    )
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_initial_state(strategy_id), config=config):
        pass
    await hitl_autoapprove(graph, thread_id)  # paper_gate
    await autoresume_for_test(graph, thread_id, {"wake": True})  # paper_wait → live_gate
    await hitl_autoapprove(graph, thread_id)  # live_gate → live_subgraph

    # 1. live_wait interrupt surfaces through the parent composition.
    assert await _parked_kind(graph, config) == "live_wait", (
        "live_wait interrupt did not surface to the parent"
    )

    # 2. Wake → live_evaluate → coordinator(pause) → live_pause interrupt
    # surfaces through the parent composition.
    await autoresume_for_test(graph, thread_id, {"wake": True})
    assert await _parked_kind(graph, config) == "live_pause_review", (
        "live_pause interrupt did not surface to the parent"
    )


# ───────────────────────── test 2: /threads visibility (6f loop) ─────────


@asynccontextmanager
async def _app_with_composed_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        app.state.graph = _build_stub_composed_graph(app.state.saver)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_paper_gate_visible_via_threads_endpoint(
    cleanup_strategy_ids: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close the 6f loop: a parent-graph thread parked at the nested
    paper_gate reports has_pending_interrupt via GET /threads — no smoke
    env override.
    """
    strategy_id = f"pg-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"

    async with _app_with_composed_graph(monkeypatch) as client:
        graph = app.state.graph

        # Seed a registry row so GET /threads enumerates this thread.
        async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO strategy_registry "
                    "(strategy_id, thread_id, name, template, stage, pairs, timeframe) "
                    "VALUES (%s, %s, %s, 'mean_reversion_template', 'paper_gate', "
                    "'[\"BTC/USDT\"]', '5m')",
                    (strategy_id, thread_id, f"test-{strategy_id}"),
                )
            await conn.commit()

        # Drive the composed graph to the nested paper_gate interrupt.
        config = {"configurable": {"thread_id": thread_id}}
        async for _ in graph.astream(_initial_state(strategy_id), config=config):
            pass

        resp = await client.get("/threads")
        assert resp.status_code == 200
        rows = {r["thread_id"]: r for r in resp.json()}
        assert thread_id in rows
        row = rows[thread_id]
        assert row["has_pending_interrupt"] is True, (
            "6f regression: parent graph's nested paper_gate not visible to /threads"
        )
        assert row["pending_interrupt_payload"]["kind"] == "paper_gate"
