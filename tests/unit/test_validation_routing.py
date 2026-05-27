"""Stage 4e routing tests — risk_analyst → paper_gate | archive.

These tests are offline: no Docker, no Postgres, no LLM. They use the
real :func:`risk_analyst_node`'s downstream nodes (``paper_gate``,
``archive``) but inject a stubbed agent that returns deterministic
``Command`` objects via :func:`verdict_to_command`. The operator's
manual Opus smoke check exercises the real agent path before tagging.

Tests live under ``tests/unit/`` (not ``tests/integration/``) because
they don't talk to any external system — they exercise pure LangGraph
state-machine routing with an ``InMemorySaver``.

Coverage:

  1. approve verdict → graph reaches ``paper_gate`` and fires
     ``interrupt()`` with a payload containing backtest + robustness +
     risk_analyst summaries (SPEC §4.1 dashboard contract).
  2. reject verdict → graph routes to ``archive``; state has
     ``stage="archived"`` and ``failure_reason`` starting with
     ``"risk_analyst_reject:"``.
  3. paper_gate resume with ``approved=True`` → state advances to
     ``stage="paper"``; gate_decisions["paper_gate"] records the
     human's decision.
  4. paper_gate resume with ``approved=False`` → state archives with
     ``failure_reason`` starting with ``"paper_gate_rejected:"``.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from orchestrator.agents.risk_analyst import RiskVerdict, verdict_to_command
from orchestrator.subgraphs.validation import (
    ValidationState,
    archive,
    paper_gate,
)


def _build_4e_routing_subgraph(stub_verdict: RiskVerdict) -> Any:
    """Minimal graph: ``START → risk_analyst (stub) → paper_gate | archive``.

    The stubbed risk_analyst calls :func:`verdict_to_command` on the
    pre-supplied verdict — same translation the real agent path uses,
    so the routing decision exercised here matches production exactly.
    """

    async def stubbed_risk_analyst(state: ValidationState) -> Command[Any]:
        # Same shape as the real risk_analyst_node — spreads existing
        # gate_decisions through so paper_gate's interrupt payload sees
        # backtest + robustness summaries.
        return verdict_to_command(
            stub_verdict,
            existing_gates=state.get("gate_decisions") or {},
        )

    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("risk_analyst", stubbed_risk_analyst)
    builder.add_node("paper_gate", paper_gate)
    builder.add_node("archive", archive)
    builder.add_edge(START, "risk_analyst")
    builder.add_edge("paper_gate", END)
    builder.add_edge("archive", END)
    return builder.compile(checkpointer=InMemorySaver())


def _seeded_state() -> ValidationState:
    """A passing-pipeline state shaped like what gate_robustness would emit.

    The risk_analyst node reads ``gate_decisions["robustness"]`` for its
    tool; the paper_gate node reads ``gate_decisions["backtest"]`` and
    ``gate_decisions["risk_analyst"]`` to build its interrupt payload
    (SPEC §4.1).
    """
    return {
        "strategy_id": "test_strategy_4e",
        "gate_decisions": {
            "backtest": {"passed": True, "best_param_set_id": "ps_1"},
            "robustness": {
                "passed": True,
                "monte_carlo": {"pct_5_final_equity": 1.10},
                "regime": {"regimes_passed": 3},
                "fee_stress": {"degradation_2x": 0.20, "degradation_3x": 0.40},
            },
        },
    }


# ─── 1. approve → paper_gate interrupt ──────────────────────────────────


@pytest.mark.asyncio
async def test_approve_reaches_paper_gate_interrupt_with_full_payload() -> None:
    verdict = RiskVerdict(
        decision="approve",
        primary_concern="Robustness is strong across all three checks.",
        rationale="MC pct_5 = 1.10 well above 1.0. All 3 regimes pass. Fee stress modest.",
        confidence=0.85,
    )
    graph = _build_4e_routing_subgraph(verdict)
    config: RunnableConfig = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}

    # Invoke; expect graph to pause at paper_gate's interrupt().
    result = await graph.ainvoke(_seeded_state(), config=config)

    # LangGraph 1.x surfaces the interrupt via the result's __interrupt__
    # field (when the graph paused) and the checkpoint state has the
    # interrupt's payload retrievable via aget_state.
    snapshot = await graph.aget_state(config)
    interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else ()
    assert interrupts, (
        "expected paper_gate to fire an interrupt() but the graph did not pause. "
        f"final result keys: {list(result.keys())}"
    )

    payload = interrupts[0].value
    assert payload["kind"] == "paper_gate"
    assert payload["strategy_id"] == "test_strategy_4e"
    # Stage 6e: payload shape is the build_interrupt_payload contract —
    # rationale primary at summary.risk_analyst, metrics secondary under
    # summary.metrics.{backtest, robustness} (SPEC §4.1).
    assert payload["summary"]["metrics"]["backtest"]["passed"] is True
    assert payload["summary"]["metrics"]["robustness"]["passed"] is True
    assert payload["summary"]["risk_analyst"]["decision"] == "approve"
    assert payload["summary"]["risk_analyst"]["confidence"] == 0.85


# ─── 2. reject → archive ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_routes_to_archive_with_failure_reason() -> None:
    verdict = RiskVerdict(
        decision="reject",
        primary_concern="Regime n_folds=1 for the only volatile bucket — single data point.",
        rationale="The strategy hasn't been tested under sufficient regime variety.",
        confidence=0.75,
    )
    graph = _build_4e_routing_subgraph(verdict)
    config: RunnableConfig = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}

    final = await graph.ainvoke(_seeded_state(), config=config)

    # No interrupt; graph reached archive cleanly.
    snapshot = await graph.aget_state(config)
    assert (
        not snapshot.tasks
    ), "reject path should not interrupt — should run to END through archive"

    assert final["stage"] == "archived"
    assert final["failure_reason"].startswith("risk_analyst_reject:")
    assert "single data point" in final["failure_reason"]
    # Vote was recorded.
    [vote] = final["agent_votes"]
    assert vote["agent"] == "risk_analyst"
    assert vote["verdict"] == "fail"


# ─── 3. paper_gate resume — approved ────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_gate_resume_approved_advances_to_paper_stage() -> None:
    """Resume the paused graph with ``approved=True`` and assert the
    state transitions to ``stage="paper"`` and records the human note.
    """
    verdict = RiskVerdict(
        decision="approve",
        primary_concern="Strong robustness across the board.",
        rationale="MC + regime + fee stress all comfortable above thresholds.",
        confidence=0.9,
    )
    graph = _build_4e_routing_subgraph(verdict)
    config: RunnableConfig = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}

    # First invocation pauses at paper_gate.
    await graph.ainvoke(_seeded_state(), config=config)

    # Resume with the human's approval payload (Stage 6 will source this
    # from the Streamlit dashboard's POST /threads/{tid}/approve handler).
    final = await graph.ainvoke(
        Command(resume={"approved": True, "notes": "looks ready for 30-day paper"}),
        config=config,
    )

    assert final["stage"] == "paper"
    # Stage 6e: human approval lands under gate_decisions["paper"] (was
    # "paper_gate" in 4e's stub — node name vs lifecycle stage).
    paper_block = final["gate_decisions"]["paper"]
    assert paper_block["approved"] is True
    assert paper_block["by"] == "human"
    assert "30-day paper" in paper_block["notes"]


# ─── 4. paper_gate resume — rejected ────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_gate_resume_rejected_archives_with_human_reason() -> None:
    verdict = RiskVerdict(
        decision="approve",
        primary_concern="Robustness gate cleared.",
        rationale="All checks pass.",
        confidence=0.7,
    )
    graph = _build_4e_routing_subgraph(verdict)
    config: RunnableConfig = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}

    await graph.ainvoke(_seeded_state(), config=config)

    # Human looks at the dashboard and says no.
    final = await graph.ainvoke(
        Command(resume={"approved": False, "notes": "BTC regime feels off this week"}),
        config=config,
    )

    assert final["stage"] == "archived"
    assert final["failure_reason"].startswith("paper_gate_rejected:")
    assert "BTC regime feels off" in final["failure_reason"]
    # Stage 6e: gate_decisions key is "paper" (the lifecycle stage),
    # not "paper_gate" (the node name).
    paper_block = final["gate_decisions"]["paper"]
    assert paper_block["approved"] is False
