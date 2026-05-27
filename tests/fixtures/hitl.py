"""HITL fixtures for Stage 7+ integration tests.

The Stage 6 lifecycle gates (``paper_gate``, ``live_gate``,
``live_pause_review``) all use LangGraph's dynamic ``interrupt()`` to
park execution until an operator decides via the dashboard. That's
correct production behaviour — but it means **any** integration test
covering a flow that crosses a gate will block forever waiting on a
human.

These fixtures inject a programmatic decision so tests for the paper
subgraph (Stage 7), live subgraph (Stage 8), supervisor (Stage 9), and
the full lifecycle (Stage 10) can drive past gate interrupts without
stubbing the FastAPI endpoint or the dashboard.

Pattern (for a future test author reading this):

.. code-block:: python

    async def test_paper_subgraph_advances_after_approval(hitl_autoapprove):
        graph = build_paper_subgraph(saver)
        config = {"configurable": {"thread_id": "t1"}}
        async for _ in graph.astream(initial_state, config=config):
            pass  # parks at paper_gate

        await hitl_autoapprove(graph, "t1", notes="auto for test")

        post = await graph.aget_state(config)
        assert post.values["stage"] == "paper"

Both fixtures wrap :func:`orchestrator.gates.hitl.autoresume_for_test`,
which refuses to resume a thread that is NOT parked at an interrupt
(silent no-op masking is its own assertion failure). That's the
contract — a test that calls the fixture against an un-interrupted
thread fails LOUD with ``AssertionError``, not silently.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from orchestrator.gates.hitl import autoresume_for_test

HitlApprover = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
def hitl_autoapprove() -> HitlApprover:
    """Returns a callable that auto-approves any HITL gate it sees.

    Signature::

        await hitl_autoapprove(graph, thread_id, notes="…")

    Asserts via :func:`autoresume_for_test` that the thread IS parked
    at an interrupt before resuming — a flow that ran past the gate
    silently raises ``AssertionError`` rather than the test passing
    for the wrong reason.
    """

    async def _approver(
        graph: Any,
        thread_id: str,
        notes: str = "test_autoapprove",
    ) -> dict[str, Any]:
        return await autoresume_for_test(
            graph,
            thread_id,
            {"approved": True, "notes": notes},
        )

    return _approver


@pytest.fixture
def hitl_autoreject() -> HitlApprover:
    """Returns a callable that auto-rejects any HITL gate it sees.

    Same contract as :func:`hitl_autoapprove` but with
    ``approved=False`` — for tests of the archive-on-rejection path
    (e.g. paper_gate's ``paper_gate_rejected: <notes>`` failure_reason
    encoding).
    """

    async def _rejecter(
        graph: Any,
        thread_id: str,
        notes: str = "test_autoreject",
    ) -> dict[str, Any]:
        return await autoresume_for_test(
            graph,
            thread_id,
            {"approved": False, "notes": notes},
        )

    return _rejecter
