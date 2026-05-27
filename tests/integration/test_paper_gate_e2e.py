"""Stage 6e end-to-end tests for the real ``paper_gate`` node.

Exercises the publish-then-interrupt-then-resume contract against the
real ``AsyncPostgresSaver`` opened by the FastAPI lifespan. The tests
build a minimal ``START → paper_gate → END`` graph that wires the
PRODUCTION ``paper_gate`` node from
:mod:`orchestrator.subgraphs.validation` — so the publish call,
interrupt payload shape, and gate_decisions writes all match what the
real validation subgraph will emit once Stage 7 lands.

Why not the real parent graph? As of Stage 5e the parent graph is
``research → archive → END`` (no validation/paper_gate wiring). Stage
7+ extends it. For 6e we test the gate node in isolation against the
real saver; integration with the parent graph topology lands later.

All tests are ``@pytest.mark.integration`` and require Postgres
reachable at the env URIs.

The 6d ``POST /threads/{tid}/approve`` resume path is exercised by
``tests/integration/test_main_endpoints.py``. Here we use
:func:`autoresume_for_test` (the 6b helper) to drive resume directly —
this bypasses the endpoint, so the ``gate_advanced`` publish (which
fires from the endpoint, not the node) is intentionally not asserted.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestrator.gates.hitl import autoresume_for_test
from orchestrator.main import app
from orchestrator.subgraphs import validation as validation_mod
from orchestrator.subgraphs.validation import ValidationState, paper_gate

pytestmark = pytest.mark.integration


# ─── Helpers ───────────────────────────────────────────────────────────


def _build_paper_gate_only_graph(
    saver: Any,
) -> CompiledStateGraph[ValidationState, ValidationState, ValidationState, ValidationState]:
    """Minimal graph: ``START → paper_gate (real) → END``.

    The schema is ``ValidationState`` so the real ``paper_gate`` node
    sees the field shape it expects.
    """
    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("paper_gate", paper_gate)
    builder.add_edge(START, "paper_gate")
    builder.add_edge("paper_gate", END)
    return builder.compile(checkpointer=saver)


def _seeded_state(strategy_id: str) -> dict[str, Any]:
    """A state shaped like what ``risk_analyst`` would emit on approve."""
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {
            "backtest": {"passed": True, "best_param_set_id": "ps_1", "sharpe_is": 1.8},
            "robustness": {
                "passed": True,
                "monte_carlo": {"pct_5_final_equity": 1.08},
                "regime": {"regimes_passed": 3},
                "fee_stress": {"degradation_2x": 0.22},
            },
            "risk_analyst": {
                "decision": "approve",
                "rationale": "Sharpe holds OOS; robustness clean.",
                "confidence": 0.82,
                "primary_concern": "minor fee-stress at 2x",
            },
        },
    }


@asynccontextmanager
async def _real_saver_app() -> AsyncIterator[Any]:
    """Drive the FastAPI lifespan to get a real AsyncPostgresSaver, yield it.

    Same pattern as 6d's test fixture — minus the test-graph swap and
    httpx client (the e2e tests drive the graph directly, not via
    HTTP)."""
    async with app.router.lifespan_context(app):
        yield app.state.saver


# ─── 1. Approve happy path ─────────────────────────────────────────────


async def test_paper_gate_approve_flow_advances_to_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: park at paper_gate, verify publish, autoresume with
    approved=True, assert advance to stage='paper'."""
    publish_mock = AsyncMock()
    monkeypatch.setattr(validation_mod, "publish_gate_pending", publish_mock)

    strategy_id = f"sid_approve_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{strategy_id}"

    async with _real_saver_app() as saver:
        graph = _build_paper_gate_only_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}

        # First astream parks at paper_gate's interrupt.
        async for _ in graph.astream(_seeded_state(strategy_id), config=config):
            pass

        # Publish fired on the gate_pending channel for this thread.
        publish_mock.assert_awaited()
        called_tid, called_payload = publish_mock.call_args.args
        assert called_tid == thread_id
        assert called_payload["kind"] == "paper_gate"
        assert called_payload["strategy_id"] == strategy_id
        # SPEC §4.1 contract — rationale primary, metrics secondary.
        assert called_payload["summary"]["risk_analyst"]["confidence"] == 0.82
        assert called_payload["summary"]["metrics"]["backtest"]["passed"] is True

        # Resume via the 6b helper (intentionally bypasses the 6d
        # endpoint — see file docstring).
        await autoresume_for_test(
            graph, thread_id, {"approved": True, "notes": "looks good"}
        )

        post = await graph.aget_state(config)
        assert post.values.get("stage") == "paper"
        paper_block = post.values["gate_decisions"]["paper"]
        assert paper_block["approved"] is True
        assert paper_block["by"] == "human"
        assert paper_block["notes"] == "looks good"


# ─── 2. Reject path ────────────────────────────────────────────────────


async def test_paper_gate_reject_flow_archives_with_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume with ``approved=False, notes="regime concerns"`` → state
    archives with ``failure_reason`` containing the notes verbatim."""
    monkeypatch.setattr(validation_mod, "publish_gate_pending", AsyncMock())

    strategy_id = f"sid_reject_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{strategy_id}"

    async with _real_saver_app() as saver:
        graph = _build_paper_gate_only_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}

        async for _ in graph.astream(_seeded_state(strategy_id), config=config):
            pass

        await autoresume_for_test(
            graph, thread_id, {"approved": False, "notes": "regime concerns"}
        )

        post = await graph.aget_state(config)
        assert post.values.get("stage") == "archived"
        failure = post.values.get("failure_reason", "")
        assert "paper_gate_rejected" in failure
        assert "regime concerns" in failure
        paper_block = post.values["gate_decisions"]["paper"]
        assert paper_block["approved"] is False
        assert paper_block["notes"] == "regime concerns"


# ─── 3. Order: publish BEFORE interrupt ────────────────────────────────


async def test_paper_gate_publishes_before_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict ordering test: ``publish_gate_pending`` must complete
    BEFORE ``interrupt()`` halts the node — otherwise the dashboard
    never sees the gate_pending event for the first parking."""
    call_order: list[str] = []

    async def recording_publish(tid: str, payload: dict[str, Any]) -> None:
        call_order.append("publish")

    # Wrap the real interrupt so the graph still actually pauses. The
    # validation module imported ``interrupt`` by name; patch THAT
    # binding, not the langgraph.types one.
    real_interrupt = validation_mod.interrupt

    def recording_interrupt(payload: dict[str, Any]) -> Any:
        call_order.append("interrupt")
        return real_interrupt(payload)

    monkeypatch.setattr(validation_mod, "publish_gate_pending", recording_publish)
    monkeypatch.setattr(validation_mod, "interrupt", recording_interrupt)

    strategy_id = f"sid_order_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{strategy_id}"

    async with _real_saver_app() as saver:
        graph = _build_paper_gate_only_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        async for _ in graph.astream(_seeded_state(strategy_id), config=config):
            pass

    assert call_order[:2] == ["publish", "interrupt"], (
        f"publish must precede interrupt; got order={call_order!r}"
    )


# ─── 4. Idempotency: publish replays on resume ─────────────────────────


async def test_paper_gate_replays_publish_on_resume_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LangGraph replays this node from its start on resume (BRD §6.2).
    The publish_gate_pending call therefore fires on EVERY replay. This
    test verifies the contract — and codifies that re-publishing is the
    expected (idempotent-by-design) behavior, not a bug."""
    publish_mock = AsyncMock()
    monkeypatch.setattr(validation_mod, "publish_gate_pending", publish_mock)

    strategy_id = f"sid_replay_{uuid.uuid4().hex[:8]}"
    thread_id = f"thread_{strategy_id}"

    async with _real_saver_app() as saver:
        graph = _build_paper_gate_only_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}

        # First park → call #1.
        async for _ in graph.astream(_seeded_state(strategy_id), config=config):
            pass
        assert publish_mock.await_count == 1

        # Resume → node replays from its start → call #2 → node
        # completes past interrupt() (which now returns the resume
        # value) → graph advances to END.
        await autoresume_for_test(
            graph, thread_id, {"approved": True, "notes": "replay-check"}
        )
        assert publish_mock.await_count == 2, (
            "expected publish_gate_pending to fire on the resume replay; "
            f"got await_count={publish_mock.await_count}"
        )

        # Sanity: graph really did advance, not just replay again.
        post = await graph.aget_state(config)
        assert post.values.get("stage") == "paper"
        assert not any(t.interrupts for t in post.tasks)
