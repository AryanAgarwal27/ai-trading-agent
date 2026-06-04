"""Stage 7h — research → validation real-handoff verification (Flag 1).

This test does NOT stub the seam between research and validation. It
runs the REAL research subgraph (with stubbed LLM fns that emit the
REAL research output shape — params dict + strategy_path +
artifacts.generated_strategy_path) composed into the parent graph, and
feeds that into the REAL validation entry (plan_backtests + the
_default_backtest_worker_fn contract), with ONLY the Freqtrade
subprocess replaced by a recording stub.

The assertion is the contract: validation's plan_backtests must be able
to read ``param_sets[*].id`` + ``folds[*].{timerange,fold_id}`` +
``strategy_path`` from what research actually emits. If research's
output shape doesn't satisfy validation's input contract, the recording
worker is never invoked (plan_backtests fans out zero Sends) and this
test fails with a clear shape-mismatch message.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.graph import build_per_strategy_graph
from orchestrator.state import BacktestResult
from orchestrator.subgraphs.research import build_research_subgraph

pytestmark = pytest.mark.integration


# ─── Stub LLM fns that emit the REAL research output shape ──────────────
# (Same shapes the real researcher / generator / critic write — see
# orchestrator/agents/{researcher,generator,critic}.py. We stub the LLM
# call, NOT the output contract.)


async def _stub_researcher(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("artifacts") or {}
    return {
        "hypothesis": "handoff-test hypothesis",
        "template": "mean_reversion_template",
        "artifacts": {
            **existing,
            "research_proposal": {
                "hypothesis": "handoff-test hypothesis",
                "template_name": "mean_reversion_template",
                "regime_thesis": "stub",
                "suggested_param_ranges": {},
                "confidence": 0.8,
            },
        },
        "agent_votes": [
            {"agent": "researcher", "verdict": "continue", "rationale": "stub", "confidence": 0.8}
        ],
    }


async def _stub_generator(state: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(tempfile.gettempdir()) / "ai_trading_agent_handoff_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state['strategy_id']}.py"
    out_path.write_text("# stub generated strategy\n", encoding="utf-8")
    existing = state.get("artifacts") or {}
    # This is EXACTLY what generator_node emits: params (dict),
    # strategy_path (str), artifacts.generated_strategy_path.
    return {
        "params": {"rsi_buy_threshold": 30, "ema_fast": 12},
        "strategy_path": str(out_path),
        "artifacts": {**existing, "generated_strategy_path": str(out_path)},
        "agent_votes": [
            {"agent": "generator", "verdict": "pass", "rationale": "stub", "confidence": 1.0}
        ],
    }


async def _stub_pass_critic(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("artifacts") or {}
    prior = list(existing.get("critic_verdicts") or [])
    prior.append(
        {
            "verdict": "pass",
            "primary_concern": "stub",
            "rationale": "stub",
            "revision_guidance": "",
            "confidence": 0.9,
        }
    )
    return {
        "agent_votes": [
            {"agent": "critic", "verdict": "pass", "rationale": "stub", "confidence": 0.9}
        ],
        "critic_notes": [],
        "artifacts": {**existing, "critic_verdicts": prior},
    }


async def _stub_pass_lookahead(
    strategy_path: Any, *, pairs: Any, timeframe: Any, timerange: Any
) -> dict[str, Any]:
    return {
        "passed": True,
        "details": "stub: no look-ahead bias",
        "returncode": 0,
        "worker_dir": "/tmp/stub_la",
        "stderr_tail": "",
        "stdout_tail": "",
    }


def _initial_state(strategy_id: str) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "name": f"handoff-{strategy_id}",
        "hypothesis": "",
        "template": "",
        "params": {},
        "freqai_config": None,
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "stage": "research",
        "backtest_results": [],
        "robustness_results": [],
        "agent_votes": [],
        "revision_count": 0,
        "critic_notes": [],
        "gate_decisions": {},
        "freqtrade_userdir": None,
        "freqtrade_process_id": None,
        "freqtrade_api_url": None,
        "artifacts": {},
        "started_at": "2026-05-01T00:00:00+00:00",
        "last_updated": "2026-05-01T00:00:00+00:00",
        "failure_reason": None,
    }


async def test_research_output_satisfies_validation_input_contract() -> None:
    """The recording worker MUST be invoked with payloads carrying the
    fields validation reads — proving research's output shape satisfies
    validation's input contract through the composed parent graph.
    """
    captured: list[dict[str, Any]] = []

    async def recording_worker(payload: dict[str, Any]) -> BacktestResult:
        captured.append(payload)
        # Return a deliberately-failing result so gate_backtest archives
        # right after the fan-out — we only care that the worker was
        # reached with a valid payload (the seam), not that validation
        # passes its gates.
        return BacktestResult(
            param_set_id=str(payload.get("_param_set", {}).get("id", "?")),
            pair="BTC/USDT",
            timeframe="5m",
            fold_id=str(payload.get("_fold", {}).get("fold_id", "?")),
            is_sharpe=-1.0,
            oos_sharpe=0.0,
            profit_factor=0.5,
            max_dd=0.5,
            trades=1,
            raw_zip_path="",
        )

    research = build_research_subgraph(
        store=None,
        researcher_fn=_stub_researcher,
        generator_fn=_stub_generator,
        critic_fn=_stub_pass_critic,
        lookahead_runner=_stub_pass_lookahead,
    )
    graph = build_per_strategy_graph(
        InMemorySaver(),
        None,
        research_subgraph=research,
        worker_fn=recording_worker,
    )

    strategy_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": f"strategy_{strategy_id}"}}
    await graph.ainvoke(cast(Any, _initial_state(strategy_id)), config=config)

    assert captured, (
        "research → validation handoff produced ZERO backtest Sends: "
        "plan_backtests read empty param_sets/folds from research's output. "
        "Research emits {params: dict, strategy_path, artifacts."
        "generated_strategy_path} but validation needs {param_sets: "
        "list[{id,...}], folds: list[{timerange,fold_id}]}. Contract gap."
    )

    payload = captured[0]
    assert "strategy_path" in payload, "worker payload missing strategy_path"
    assert (
        "_param_set" in payload and "id" in payload["_param_set"]
    ), f"worker payload _param_set lacks 'id': {payload.get('_param_set')!r}"
    assert "_fold" in payload and {"timerange", "fold_id"} <= set(
        payload["_fold"]
    ), f"worker payload _fold lacks timerange/fold_id: {payload.get('_fold')!r}"
