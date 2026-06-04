"""Stage 9f integration e2e — D-6 production live-wake (real saver + endpoint).

Drives the FastAPI lifespan (real AsyncPostgresSaver via conftest) and swaps in
a MINIMAL live graph that uses the REAL ``live_wait`` + REAL
``_route_after_live_wait`` with stub ``live_evaluate`` / ``live_pause`` leaves,
so the split-by-trigger routing is exercised end-to-end without the reviewer
fan-out, LLMs, Docker, or live keys.

Three cases, mapping to the D-6 closure:

1. **Kill-path auto-resume** — ``make_kill_event_writer`` writes the kill event
   AND directly resumes (no manual Command(resume) in the test); the thread
   routes to ``live_pause`` purely from production code.
2. **Periodic /wake on live_wait (no kill)** — routes to ``live_evaluate``.
3. **Regression guard** — a kill-written thread (parent interrupt cleared by
   aupdate_state) refuses a periodic /wake with 409. The kill path is the ONLY
   resumer for kill-written threads.

Marked ``integration`` (opens the Postgres saver/store); excluded from the unit
CI job, runs locally / in Stage 9g.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

import orchestrator.kill_subscription as kill_sub
from orchestrator.kill_subscription import make_kill_event_writer
from orchestrator.main import app
from orchestrator.subgraphs.live import _route_after_live_wait, live_wait

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-9f-live-wake"


class _LiveWaitState(TypedDict, total=False):
    strategy_id: str
    stage: str
    artifacts: dict[str, Any]


def _build_live_wait_test_graph(saver: Any) -> Any:
    """START → live_wait → route → {live_evaluate-stub | live_pause-stub} → END.

    Real ``live_wait`` + real ``_route_after_live_wait``; stub leaves so the kill
    vs no-kill routing is provable without the reviewer fan-out / LLMs.
    """

    def live_evaluate_stub(state: _LiveWaitState) -> dict[str, Any]:
        arts = dict(state.get("artifacts") or {})
        arts["reached"] = "live_evaluate"
        return {"artifacts": arts}

    def live_pause_stub(state: _LiveWaitState) -> dict[str, Any]:
        # The real live_pause emits a live_pause_review interrupt; the stub does
        # the same so the test can assert the thread parked there post-kill.
        interrupt({"kind": "live_pause_review", "strategy_id": state.get("strategy_id")})
        return {}

    g: StateGraph[_LiveWaitState, _LiveWaitState, _LiveWaitState, _LiveWaitState] = StateGraph(
        _LiveWaitState
    )
    g.add_node("live_wait", live_wait)  # type: ignore[arg-type]
    g.add_node("live_evaluate", live_evaluate_stub)
    g.add_node("live_pause", live_pause_stub)
    g.add_edge(START, "live_wait")
    g.add_conditional_edges("live_wait", _route_after_live_wait, ["live_evaluate", "live_pause"])
    g.add_edge("live_evaluate", END)
    g.add_edge("live_pause", END)
    return g.compile(checkpointer=saver)


@asynccontextmanager
async def _app_with_live_graph(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        app.state.graph = _build_live_wait_test_graph(app.state.saver)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _park_at_live_wait(graph: Any, thread_id: str, strategy_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async for _ in graph.astream(
        {"strategy_id": strategy_id, "stage": "live", "artifacts": {}}, config=config
    ):
        pass


def _interrupt_kinds(snapshot: Any) -> list[str]:
    kinds: list[str] = []
    for task in snapshot.tasks:
        for itr in getattr(task, "interrupts", ()):
            val = itr.value
            if isinstance(val, dict) and "kind" in val:
                kinds.append(val["kind"])
    return kinds


# ─── 1. kill-path auto-resume ────────────────────────────────────────────


async def test_kill_event_auto_resumes_to_live_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app_with_live_graph(monkeypatch):
        graph = app.state.graph
        thread_id = f"strategy_{uuid.uuid4().hex[:8]}"
        strategy_id = thread_id.removeprefix("strategy_")
        await _park_at_live_wait(graph, thread_id, strategy_id)
        assert _interrupt_kinds(
            await graph.aget_state({"configurable": {"thread_id": thread_id}})
        ) == ["live_wait"]

        # Production kill writer: writes the event AND directly resumes — no
        # manual Command(resume) here.
        writer = make_kill_event_writer(graph)
        await writer(
            strategy_id,
            {"reason": "drawdown_12pct_exceeded", "fired_at": "t", "metrics_summary": {}},
        )
        await asyncio.gather(*list(kill_sub._KILL_RESUME_TASKS))

        # Routed straight to live_pause (kill branch) — now parked there.
        post = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert _interrupt_kinds(post) == ["live_pause_review"]


# ─── 2. periodic /wake on live_wait (no kill) → live_evaluate ─────────────


async def test_periodic_live_wake_routes_to_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app_with_live_graph(monkeypatch) as client:
        graph = app.state.graph
        thread_id = f"strategy_{uuid.uuid4().hex[:8]}"
        await _park_at_live_wait(graph, thread_id, thread_id.removeprefix("strategy_"))

        resp = await client.post(
            f"/threads/{thread_id}/wake",
            params={"kind": "live_wait"},
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 200
        assert resp.json()["woke"] is True

        post = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert _interrupt_kinds(post) == []  # ran to END through live_evaluate
        assert (post.values.get("artifacts") or {}).get("reached") == "live_evaluate"


# ─── 3. regression guard: periodic /wake refuses a kill-written thread ────


async def test_periodic_wake_refuses_kill_written_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill-path / periodic-path independence (the 8h finding). After the kill
    subscription's aupdate_state writes the event, the parent interrupt is
    cleared, so a periodic /wake (kind=live_wait) must 409 — the kill path is the
    only resumer for kill-written threads."""
    async with _app_with_live_graph(monkeypatch) as client:
        graph = app.state.graph
        thread_id = f"strategy_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        await _park_at_live_wait(graph, thread_id, thread_id.removeprefix("strategy_"))

        # Simulate the kill write WITHOUT the direct-resume: aupdate_state clears
        # the parent-visible interrupt (the 8h finding).
        await graph.aupdate_state(config, {"artifacts": {"kill_switch_event": {"reason": "dd"}}})
        assert _interrupt_kinds(await graph.aget_state(config)) == []  # parent interrupt cleared

        resp = await client.post(
            f"/threads/{thread_id}/wake",
            params={"kind": "live_wait"},
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 409
