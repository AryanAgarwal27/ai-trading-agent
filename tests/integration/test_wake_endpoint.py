"""Stage 7f integration tests for POST /threads/{tid}/wake.

Drives the FastAPI lifespan (real saver, real scheduler with the
in-memory jobstore via conftest) and swaps in a small test graph that
parks at the real ``paper_wait`` node. Covers token auth + the
node-name guard that prevents a wake from corrupting a HITL gate.

Marked ``integration`` (lifespan opens the Postgres saver/store).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from orchestrator.main import app
from orchestrator.subgraphs.paper import paper_wait

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-7f-wake"


class _WaitState(TypedDict, total=False):
    strategy_id: str
    stage: str


def _build_paper_wait_test_graph(saver: Any) -> Any:
    """Graph: START → paper_wait(real interrupt) → END."""
    g: StateGraph[_WaitState, _WaitState, _WaitState, _WaitState] = StateGraph(_WaitState)
    g.add_node("paper_wait", paper_wait)  # type: ignore[arg-type]
    g.add_edge(START, "paper_wait")
    g.add_edge("paper_wait", END)
    return g.compile(checkpointer=saver)


def _build_paper_gate_test_graph(saver: Any) -> Any:
    """Graph parking at a node named 'paper_gate' (NOT paper_wait).

    Used to prove /wake refuses to resume a HITL gate.
    """

    def paper_gate(state: _WaitState) -> dict[str, Any]:
        interrupt({"kind": "paper_gate", "strategy_id": state.get("strategy_id")})
        return {"stage": "paper"}

    g: StateGraph[_WaitState, _WaitState, _WaitState, _WaitState] = StateGraph(_WaitState)
    g.add_node("paper_gate", paper_gate)
    g.add_edge(START, "paper_gate")
    g.add_edge("paper_gate", END)
    return g.compile(checkpointer=saver)


@asynccontextmanager
async def _app_with_graph(
    monkeypatch: pytest.MonkeyPatch, graph_builder: Any
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        app.state.graph = graph_builder(app.state.saver)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _park(graph: Any, thread_id: str, strategy_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async for _ in graph.astream({"strategy_id": strategy_id}, config=config):
        pass


# ─── auth ───────────────────────────────────────────────────────────────


async def test_wake_without_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app_with_graph(monkeypatch, _build_paper_wait_test_graph) as client:
        resp = await client.post("/threads/any_tid/wake")
    assert resp.status_code == 401


async def test_wake_with_wrong_token_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app_with_graph(monkeypatch, _build_paper_wait_test_graph) as client:
        resp = await client.post("/threads/any_tid/wake", headers={"X-Operator-Token": "wrong"})
    assert resp.status_code == 403


# ─── happy path ───────────────────────────────────────────────────────


async def test_wake_resumes_paper_wait_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app_with_graph(monkeypatch, _build_paper_wait_test_graph) as client:
        graph = app.state.graph
        thread_id = f"wake-{uuid.uuid4().hex[:8]}"
        await _park(graph, thread_id, "s1")

        # Parked at paper_wait before the wake.
        snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert any(getattr(t, "interrupts", ()) for t in snap.tasks)

        resp = await client.post(
            f"/threads/{thread_id}/wake",
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 200
        assert resp.json()["woke"] is True

        # No longer parked — the thread ran through to END.
        post = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert not any(getattr(t, "interrupts", ()) for t in post.tasks)


async def test_wake_on_uninterrupted_thread_returns_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _app_with_graph(monkeypatch, _build_paper_wait_test_graph) as client:
        resp = await client.post(
            "/threads/never-ran/wake",
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
    assert resp.status_code == 409


async def test_wake_refuses_to_wake_a_hitl_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """/wake must NOT resume a thread parked at paper_gate (would corrupt it)."""
    async with _app_with_graph(monkeypatch, _build_paper_gate_test_graph) as client:
        graph = app.state.graph
        thread_id = f"gate-{uuid.uuid4().hex[:8]}"
        await _park(graph, thread_id, "s1")

        resp = await client.post(
            f"/threads/{thread_id}/wake",
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 409
        assert "paper_wait" in resp.json()["detail"]

        # Thread is still parked at paper_gate — not corrupted/advanced.
        snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert any(getattr(t, "interrupts", ()) for t in snap.tasks)
