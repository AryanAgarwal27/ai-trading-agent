"""Stage 10c contextvar-propagation verification (BRD §14).

The Stage 10c brief calls out two propagation paths that must be VERIFIED,
not assumed, because they cross an execution boundary the entry-site
binding has to survive:

1. **Send fan-out workers** (BRD §6.3) — LangGraph creates a task per
   ``Send``; task creation copies the current context, so a ``run_id`` /
   ``strategy_id`` bound BEFORE ``astream`` propagates into each worker.
2. **The ``asyncio.to_thread`` backtest_runner** (SPEC 2026-05-27 Stage
   3c) — ``asyncio.to_thread`` copies the calling context into the worker
   thread since py3.9, so the same bindings survive the thread boundary.

This test drives BOTH through the REAL machinery (a real LangGraph Send
fan-out + the real ``backtest_runner._run_subprocess`` ``to_thread``
call, with ``subprocess.run`` stubbed so no Docker is needed) inside a
single ``run_context``, and asserts the Send-fan-out worker's log line
AND the to_thread backtest_runner's log line both carry ``run_id`` +
``strategy_id`` — and the SAME ``run_id`` (one graph execution).
"""

from __future__ import annotations

import json
import subprocess
from operator import add
from typing import Annotated, Any, TypedDict

import pytest
import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from orchestrator.observability import log as log_mod
from orchestrator.tools import backtest_runner


class _FanState(TypedDict, total=False):
    items: list[int]
    results: Annotated[list[int], add]


class _DoneProcess:
    """Stand-in for ``subprocess.CompletedProcess`` — no Docker required."""

    stdout = b""
    stderr = b""
    returncode = 0


@pytest.fixture(autouse=True)
def _clean_contextvars() -> None:
    structlog.contextvars.clear_contextvars()


async def test_send_fanout_and_to_thread_both_inherit_run_context(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_mod.configure_logging()

    # Stub subprocess.run so backtest_runner._run_subprocess_sync returns
    # immediately. It still emits its node="backtest_runner" line from INSIDE
    # the to_thread worker thread before this stub is reached.
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _DoneProcess())

    def _plan(state: _FanState) -> list[Send]:
        return [Send("worker", {"_i": i}) for i in state["items"]]

    async def _worker(payload: dict[str, Any]) -> dict[str, Any]:
        # node="backtest_worker" = the Send fan-out worker line.
        log_mod.get_logger("backtest_worker").info("fanout", payload={"i": payload["_i"]})
        # Exercise the REAL asyncio.to_thread in backtest_runner — its
        # _run_subprocess_sync emits node="backtest_runner" from the thread.
        await backtest_runner._run_subprocess(["noop"], 5)
        return {"results": [payload["_i"]]}

    builder: StateGraph[_FanState, _FanState, _FanState, _FanState] = StateGraph(_FanState)
    builder.add_node("worker", _worker)  # type: ignore[call-overload]
    builder.add_conditional_edges(START, _plan, ["worker"])
    builder.add_edge("worker", END)
    graph = builder.compile()

    with log_mod.run_context(strategy_id="strategy_propagation", thread_id="strategy_propagation"):
        await graph.ainvoke({"items": [1]})

    records = [
        json.loads(line)
        for line in capsys.readouterr().out.strip().splitlines()
        if line.startswith("{")
    ]
    by_node = {rec["node"]: rec for rec in records if "node" in rec}

    # (1) Send fan-out worker inherited the entry-bound contextvars.
    fanout = by_node.get("backtest_worker")
    assert fanout is not None, f"no Send-fan-out (backtest_worker) log line in {records!r}"
    assert fanout["strategy_id"] == "strategy_propagation"
    assert fanout["run_id"], "Send fan-out worker dropped run_id"

    # (2) asyncio.to_thread backtest_runner inherited them across the thread.
    threaded = by_node.get("backtest_runner")
    assert threaded is not None, f"no to_thread (backtest_runner) log line in {records!r}"
    assert threaded["strategy_id"] == "strategy_propagation"
    assert threaded["run_id"], "to_thread backtest_runner dropped run_id"

    # Both belong to ONE graph execution → identical run_id.
    assert fanout["run_id"] == threaded["run_id"]
