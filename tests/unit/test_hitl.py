"""Unit tests for :mod:`orchestrator.gates.hitl` (Stage 6b).

Two surfaces under test:

1. :func:`build_interrupt_payload` — payload shape per SPEC §4.1, with
   stub-friendly behaviour when upstream nodes (Stage 7 / Stage 8) have
   not yet written their rationale into ``gate_decisions``.
2. :func:`autoresume_for_test` — refuses to resume a non-interrupted
   thread (assertion error), drives an interrupted one to the next
   state.

The autoresume tests use ``InMemorySaver`` and a deliberately tiny
``StateGraph`` — no Postgres, no FastAPI. The integration-level resume
flow against the real per-strategy graph + Postgres saver lands as a
6c-or-later integration test.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from orchestrator.gates.hitl import (
    autoresume_for_test,
    build_interrupt_payload,
)

# ─── build_interrupt_payload — paper_gate ──────────────────────────────


def test_build_interrupt_payload_paper_gate_includes_risk_analyst_rationale() -> None:
    """SPEC §4.1: paper_gate must surface the risk_analyst rationale as
    the primary rationale source; backtest + robustness metrics are
    secondary."""
    state: dict[str, Any] = {
        "strategy_id": "test_paper_001",
        "gate_decisions": {
            "risk_analyst": {
                "decision": "approve",
                "primary_concern": "fee-stress sensitivity at 3x",
                "rationale": "Sharpe holds above 1.2 OOS across all 6 folds; "
                "fee-stress 3x degradation 52% (under 60% cap).",
                "confidence": 0.78,
            },
            "backtest": {
                "sharpe_is": 1.91,
                "oos_ratio": 0.71,
                "max_dd": 0.14,
            },
            "robustness": {
                "score": 0.78,
                "passed": True,
                "mc_p5_return": 0.04,
                "regimes_passed": 3,
            },
        },
    }
    payload = build_interrupt_payload(state, "paper_gate")
    assert payload["kind"] == "paper_gate"
    assert payload["strategy_id"] == "test_paper_001"
    # Primary rationale — full dict shape preserved so dashboard can
    # show verdict/confidence/primary_concern.
    risk = payload["summary"]["risk_analyst"]
    assert risk["decision"] == "approve"
    assert risk["confidence"] == 0.78
    assert "fee-stress" in risk["rationale"]
    # Secondary metrics under summary.metrics — keys verbatim.
    assert payload["summary"]["metrics"]["backtest"]["sharpe_is"] == 1.91
    assert payload["summary"]["metrics"]["robustness"]["score"] == 0.78


def test_build_interrupt_payload_does_not_mutate_state() -> None:
    """Payload builder must be a pure read on state — gate nodes call it
    inside the interrupt(), and a mutation would make resume replays
    non-deterministic."""
    state: dict[str, Any] = {
        "strategy_id": "x",
        "gate_decisions": {"risk_analyst": {"confidence": 0.5}},
    }
    before = {
        "strategy_id": state["strategy_id"],
        "gate_decisions": dict(state["gate_decisions"]),
    }
    _ = build_interrupt_payload(state, "paper_gate")
    assert state["strategy_id"] == before["strategy_id"]
    assert state["gate_decisions"] == before["gate_decisions"]


# ─── build_interrupt_payload — live_gate stub case ─────────────────────


def test_build_interrupt_payload_handles_missing_optional_sources() -> None:
    """live_gate's rationale comes from paper_monitor, which Stage 7
    produces. Until then the payload must still build cleanly with
    paper_monitor=None — Stage 6 can't depend on Stage 7 having
    landed."""
    state: dict[str, Any] = {
        "strategy_id": "test_live_gate_stub",
        "gate_decisions": {},  # no paper_monitor key yet
    }
    payload = build_interrupt_payload(state, "live_gate")
    assert payload["kind"] == "live_gate"
    assert payload["strategy_id"] == "test_live_gate_stub"
    # Stub — explicit None, not missing key. Dashboard renderer can
    # then show a "monitor has not run yet" placeholder rather than
    # a KeyError.
    assert payload["summary"]["paper_monitor"] is None
    assert payload["summary"]["metrics"]["paper"] is None


def test_build_interrupt_payload_tolerates_missing_gate_decisions_entirely() -> None:
    """State at the start of research has no gate_decisions key at all.
    The builder must not KeyError — defensive read, since gate nodes
    can fire from a partially-populated state during stub/test flows."""
    state: dict[str, Any] = {"strategy_id": "bare"}
    payload = build_interrupt_payload(state, "paper_gate")
    assert payload["summary"]["risk_analyst"] is None
    assert payload["summary"]["metrics"]["backtest"] is None
    assert payload["summary"]["metrics"]["robustness"] is None


# ─── build_interrupt_payload — live_pause_review ───────────────────────


def test_build_interrupt_payload_live_pause_review_carries_kill_switch_when_present() -> None:
    """SPEC §4.1 kill-switch row: when artifacts.kill_switch_event is
    set, the live_pause_review payload takes the kill-switch path —
    no coordinator rationale (the out-of-band APScheduler kill switch
    does not vote, per BRD §5.6)."""
    state: dict[str, Any] = {
        "strategy_id": "test_pause_ks",
        "gate_decisions": {
            # Coordinator data is present but must be IGNORED on the
            # kill-switch path — the path field is the discriminator.
            "coordinator": {"rationale": "should be ignored", "verdict": "pause"},
        },
        "artifacts": {
            "kill_switch_event": {
                "reason": "drawdown_12pct_exceeded",
                "metrics": {"drawdown": 0.131, "consecutive_losses": 4},
                "action_taken": "POST /api/v1/stop",
                "fired_at": "2026-05-27T14:32:11Z",
            },
            "drawdown_trajectory": [0.04, 0.07, 0.09, 0.11, 0.131],
            "recent_trades": [{"pair": "BTC/USDT", "pnl": -42.10}],
        },
    }
    payload = build_interrupt_payload(state, "live_pause_review")
    assert payload["kind"] == "live_pause_review"
    s = payload["summary"]
    assert s["path"] == "kill_switch"
    # Coordinator must be None on the kill-switch path — the dashboard
    # uses this to flag the gate distinctly (SPEC §4.1).
    assert s["coordinator"] is None
    assert s["reviewer_votes"] is None
    # Kill-switch row carried verbatim.
    assert s["kill_switch_event"]["reason"] == "drawdown_12pct_exceeded"
    assert s["kill_switch_event"]["metrics"]["drawdown"] == 0.131
    # Secondary metrics (drawdown trajectory + trades around fired_at).
    assert s["metrics"]["drawdown"] == [0.04, 0.07, 0.09, 0.11, 0.131]
    assert s["metrics"]["recent_trades"][0]["pair"] == "BTC/USDT"


def test_build_interrupt_payload_live_pause_review_coordinator_path_when_no_kill_switch() -> None:
    """Without a kill_switch_event artifact, live_pause_review takes
    the coordinator path — coordinator + per-reviewer votes are the
    primary surface."""
    state: dict[str, Any] = {
        "strategy_id": "test_pause_coord",
        "gate_decisions": {
            "coordinator": {
                "verdict": "pause",
                "rationale": "performance drift exceeds threshold",
                "confidence": 0.82,
            },
            "risk_check": {"verdict": "continue", "confidence": 0.6},
            "performance_check": {"verdict": "pause", "confidence": 0.9},
            "regime_check": {"verdict": "continue", "confidence": 0.55},
        },
        "artifacts": {
            "current_drawdown": 0.08,
            "daily_pnl": -0.024,
            "consecutive_losses": 3,
        },
    }
    payload = build_interrupt_payload(state, "live_pause_review")
    s = payload["summary"]
    assert s["path"] == "coordinator"
    assert s["coordinator"]["verdict"] == "pause"
    assert s["reviewer_votes"]["risk_check"]["verdict"] == "continue"
    assert s["reviewer_votes"]["performance_check"]["verdict"] == "pause"
    assert s["reviewer_votes"]["regime_check"]["verdict"] == "continue"
    assert s["metrics"]["current_drawdown"] == 0.08


# ─── autoresume_for_test ────────────────────────────────────────────────


# NOTE: bare ``dict[str, Any]`` as a StateGraph schema causes LangGraph to
# REPLACE the values dict on each node return instead of merging per-key —
# intermediate writes (``approved``/``notes`` from a gate) get dropped by the
# next node. A TypedDict (even total=False) restores per-key merge semantics.
class _TestState(TypedDict, total=False):
    """Minimal TypedDict schema for the autoresume test graphs."""

    strategy_id: str
    touched: bool
    approved: bool
    notes: str
    stage: str


def _build_passthrough_graph() -> Any:
    """Tiny graph that runs START → noop → END with NO interrupt."""

    def noop(state: _TestState) -> dict[str, Any]:
        return {"touched": True}

    g = StateGraph(_TestState)
    g.add_node("noop", noop)
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g.compile(checkpointer=InMemorySaver())


def _build_one_interrupt_graph() -> Any:
    """Tiny graph that interrupts once at ``gate`` then advances to ``after``.

    Mirrors the real gate-node pattern: ``interrupt()`` returns the
    resume payload, the node merges it into state, the next node sees
    the merged value.
    """

    def gate(state: _TestState) -> dict[str, Any]:
        decision = interrupt({"kind": "paper_gate", "strategy_id": state.get("strategy_id")})
        return {"approved": decision["approved"], "notes": decision.get("notes", "")}

    def after(state: _TestState) -> dict[str, Any]:
        return {"stage": "paper" if state.get("approved") else "archived"}

    g = StateGraph(_TestState)
    g.add_node("gate", gate)
    g.add_node("after", after)
    g.add_edge(START, "gate")
    g.add_edge("gate", "after")
    g.add_edge("after", END)
    return g.compile(checkpointer=InMemorySaver())


async def test_autoresume_raises_when_thread_not_interrupted() -> None:
    """A thread that ran to END is NOT a valid resume target. The helper
    must refuse — silently treating it as a no-op would mask the bug
    where the test set up the wrong fixture."""
    graph = _build_passthrough_graph()
    config = {"configurable": {"thread_id": "tid_no_interrupt"}}
    # Drive the graph all the way through; no interrupt fires.
    async for _ in graph.astream({"strategy_id": "x"}, config=config):
        pass

    with pytest.raises(AssertionError, match="not parked at an interrupt"):
        await autoresume_for_test(graph, "tid_no_interrupt", {"approved": True, "notes": ""})


async def test_autoresume_advances_an_interrupted_thread() -> None:
    """Drive a graph to an interrupt, then call the helper to resume.
    Assert state advances past the gate and the resume payload was
    consumed by the gate node."""
    graph = _build_one_interrupt_graph()
    thread_id = "tid_one_interrupt"
    config = {"configurable": {"thread_id": thread_id}}

    # First astream parks at the gate's interrupt.
    async for _ in graph.astream({"strategy_id": "abc"}, config=config):
        pass
    snapshot = await graph.aget_state(config)
    assert any(t.interrupts for t in snapshot.tasks), "fixture broken: gate did not interrupt"

    final = await autoresume_for_test(
        graph,
        thread_id,
        {"approved": True, "notes": "looks good"},
    )

    # Final event payload — should reflect the post-resume state. After
    # resume, the gate node re-runs (interrupt replay semantics, BRD
    # §6.2) and the after node sets stage=paper.
    assert final, "autoresume returned no event"
    post = await graph.aget_state(config)
    # No more interrupts pending.
    assert not any(t.interrupts for t in post.tasks)
    # State advanced through the gate.
    assert post.values.get("approved") is True
    assert post.values.get("notes") == "looks good"
    assert post.values.get("stage") == "paper"


async def test_autoresume_reject_path_advances_to_archived() -> None:
    """Belt-and-braces: a reject resume drives the same graph to
    stage='archived'. Confirms the helper passes the decision through
    verbatim — not just the happy path."""
    graph = _build_one_interrupt_graph()
    thread_id = "tid_reject"
    config = {"configurable": {"thread_id": thread_id}}
    async for _ in graph.astream({"strategy_id": "xyz"}, config=config):
        pass

    await autoresume_for_test(
        graph,
        thread_id,
        {"approved": False, "notes": "robustness too thin"},
    )

    post = await graph.aget_state(config)
    assert post.values.get("approved") is False
    assert post.values.get("stage") == "archived"
