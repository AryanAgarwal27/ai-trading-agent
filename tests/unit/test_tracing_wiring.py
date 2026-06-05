"""Stage 10d wiring tests — graceful degradation + one-id threading (BRD §14).

These exercise the *wiring contract* the 4 execution-entry sites use
(supervisor spawn, /wake, /approve, kill-resume), without a real
LangSmith call:

1. **Graceful degradation** — a real LangGraph runs cleanly through a
   ``trace_config``-augmented config with ``LANGSMITH_TRACING`` off: no
   crash, correct output, identical to a plain (un-traced) run. Tracing
   off is a normal supported mode (no hard dependency).
2. **One id, not two** — the exact ``run_id`` string ``run_context``
   yields (and binds into structlog contextvars, so the log lines carry
   it) is the SAME string threaded into ``trace_config``; it surfaces as
   ``metadata.run_id`` (hex) and as the LangSmith root ``config["run_id"]``
   (its ``uuid.UUID``). One id, three views — this is the wiring the entry
   sites perform (``with run_context(...) as run_id: trace_config(...,
   run_id=run_id)``).

NOT covered here (CANNOT be, without a live LangSmith): that LangSmith
actually ACCEPTS ``config["run_id"]`` as the trace's root id. That is
operator-verified by inspecting a real trace.
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

import pytest
import structlog
from langgraph.graph import END, START, StateGraph

from orchestrator.observability import tracing
from orchestrator.observability.log import run_context
from orchestrator.observability.tracing import trace_config


class _S(TypedDict, total=False):
    x: int
    out: int


def _build_graph() -> Any:
    def node(state: _S) -> dict[str, Any]:
        return {"out": state["x"] + 1}

    builder: StateGraph[_S, _S, _S, _S] = StateGraph(_S)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile()


@pytest.fixture(autouse=True)
def _clean_contextvars() -> None:
    structlog.contextvars.clear_contextvars()


async def test_graph_runs_cleanly_with_tracing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LANGSMITH_TRACING off (and no API key) → graph runs identically."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing.tracing_enabled() is False  # we are genuinely on the OFF path

    graph = _build_graph()
    config = {"configurable": {"thread_id": "t"}}

    with run_context(strategy_id="sid", thread_id="t") as run_id:
        traced = trace_config(
            config, strategy_id="sid", thread_id="t", run_id=run_id, stage="research"
        )
        result = await graph.ainvoke({"x": 1}, config=traced)

    assert result["out"] == 2  # ran correctly through the augmented config, no crash


async def test_trace_config_does_not_change_graph_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No behaviour change: traced vs plain config yield the same result."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    graph = _build_graph()

    plain = await graph.ainvoke({"x": 5}, config={"configurable": {"thread_id": "a"}})

    with run_context(strategy_id="sid", thread_id="b") as run_id:
        traced = trace_config(
            {"configurable": {"thread_id": "b"}},
            strategy_id="sid",
            thread_id="b",
            run_id=run_id,
        )
        traced_out = await graph.ainvoke({"x": 5}, config=traced)

    assert plain["out"] == traced_out["out"] == 6


def test_run_context_run_id_is_the_same_id_threaded_into_trace_config() -> None:
    """The id is threaded as ONE string: run_context yield == structlog
    contextvar == trace metadata (hex) == trace root run_id (UUID)."""
    config = {"configurable": {"thread_id": "t"}}

    with run_context(strategy_id="sid", thread_id="t") as run_id:
        # What the structlog lines carry (merge_contextvars reads this).
        bound = structlog.contextvars.get_contextvars()["run_id"]
        traced = trace_config(config, strategy_id="sid", thread_id="t", run_id=run_id)

    # Same string object's value across log + trace surfaces — not a parallel mint.
    assert bound == run_id
    assert traced["metadata"]["run_id"] == run_id
    assert traced["run_id"] == uuid.UUID(run_id)
    assert traced["run_id"].hex == run_id
