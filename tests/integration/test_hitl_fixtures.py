"""Stage 6i — demonstration tests for the hitl_autoapprove /
hitl_autoreject fixtures.

These two tests are intentionally minimal and exist as
documentation-by-example: a future Stage 7+ author writing a paper-
subgraph integration test should be able to read this file and see
exactly how to skip the operator HITL prompt.

Tests use ``InMemorySaver`` (no Postgres) and mock the Redis publish
seam in ``orchestrator.observability.events`` — so they verify the
fixture mechanics without standing up infrastructure. The full
end-to-end resume cycle (against real Postgres + real publish path)
is covered by ``tests/integration/test_hitl_resume.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.main import _build_paper_gate_only_graph_for_smoke
from orchestrator.observability import events

# Not marked integration — the test exercises real LangGraph + the real
# paper_gate node, but ALL external I/O (Postgres saver, Redis client)
# is replaced. Adding a marker would imply infra requirements that
# don't exist here.


def _seeded(strategy_id: str) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {
            "risk_analyst": {
                "decision": "approve",
                "rationale": "demo",
                "confidence": 0.8,
            },
            "backtest": {"passed": True},
            "robustness": {"passed": True},
        },
    }


async def test_hitl_autoapprove_fixture_works(
    hitl_autoapprove: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pattern for Stage 7+ tests: park at HITL gate, call
    ``hitl_autoapprove(graph, thread_id, notes=...)``, assert the
    state advanced past the gate.
    """
    monkeypatch.setattr(events, "_redis_client", lambda: AsyncMock())

    saver = InMemorySaver()
    graph = _build_paper_gate_only_graph_for_smoke(saver)
    thread_id = "demo_autoapprove"
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_seeded(thread_id), config=config):
        pass  # parks at paper_gate's interrupt

    await hitl_autoapprove(graph, thread_id, notes="demo approve")

    snap = await graph.aget_state(config)
    assert snap.values.get("stage") == "paper"
    paper_block = snap.values["gate_decisions"]["paper"]
    assert paper_block["approved"] is True
    assert paper_block["notes"] == "demo approve"
    assert paper_block["by"] == "human"


async def test_hitl_autoreject_fixture_works(
    hitl_autoreject: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same pattern for the rejection path — useful for archive-on-
    reject path tests in Stage 7+ (and the equivalent live-pause
    coverage in Stage 8).
    """
    monkeypatch.setattr(events, "_redis_client", lambda: AsyncMock())

    saver = InMemorySaver()
    graph = _build_paper_gate_only_graph_for_smoke(saver)
    thread_id = "demo_autoreject"
    config = {"configurable": {"thread_id": thread_id}}

    async for _ in graph.astream(_seeded(thread_id), config=config):
        pass

    await hitl_autoreject(graph, thread_id, notes="regime concerns")

    snap = await graph.aget_state(config)
    assert snap.values.get("stage") == "archived"
    failure = snap.values.get("failure_reason", "")
    assert "paper_gate_rejected" in failure
    assert "regime concerns" in failure


async def test_hitl_autoapprove_raises_when_graph_not_parked(
    hitl_autoapprove: Any,
) -> None:
    """Edge case for Stage 7+ authors: if the test fixture forgot to
    drive the graph TO the interrupt, the fixture refuses (via
    autoresume_for_test's assertion) rather than silently no-op'ing.

    Catches the "I thought my test parked but the gate never fired"
    bug class loud, before the test passes for the wrong reason."""
    saver = InMemorySaver()
    graph = _build_paper_gate_only_graph_for_smoke(saver)

    with pytest.raises(AssertionError, match="not parked at an interrupt"):
        await hitl_autoapprove(graph, "thread_never_ran", notes="should fail")
