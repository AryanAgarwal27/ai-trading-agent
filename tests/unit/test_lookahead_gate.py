"""Unit tests for the lookahead_gate node (Stage 5e, BRD §8 rule 5).

Tests the routing logic in isolation with a stubbed
``lookahead_runner``. The real Freqtrade-subprocess runner is
exercised by the integration test (``test_research_subgraph_e2e``)
with a stub too — actually invoking docker is the operator's manual
verification at DoD time, not CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

from orchestrator.subgraphs.research import (
    DEFAULT_LOOKAHEAD_TIMERANGE,
    make_lookahead_gate,
)

# ─── helpers ───────────────────────────────────────────────────────────


def _make_runner_returning(passed: bool, details: str = "stub"):
    """Build a stub runner that returns a canned LookaheadResult."""

    async def stub(
        strategy_path: Path,
        *,
        pairs: list[str],
        timeframe: str,
        timerange: str,
    ) -> dict[str, Any]:
        # Record the call args on the stub for the test to inspect.
        stub.last_call = {  # type: ignore[attr-defined]
            "strategy_path": strategy_path,
            "pairs": pairs,
            "timeframe": timeframe,
            "timerange": timerange,
        }
        return {
            "passed": passed,
            "details": details,
            "returncode": 0,
            "worker_dir": "/tmp/stub_la",
            "stderr_tail": "",
            "stdout_tail": "",
        }

    stub.last_call = None  # type: ignore[attr-defined]
    return stub


def _state_with_strategy(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a minimal state with a real strategy file on disk."""
    strategy_path = tmp_path / "rendered.py"
    strategy_path.write_text("# minimal\n", encoding="utf-8")
    base = {
        "strategy_id": "test_la_001",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "artifacts": {"generated_strategy_path": str(strategy_path)},
    }
    base.update(overrides)
    return base


# ─── pass-route tests ──────────────────────────────────────────────────


async def test_lookahead_gate_pass_routes_to_lookahead_gate_target(
    tmp_path: Path,
) -> None:
    """passed=True → Command(goto='lookahead_gate' ??)... actually no goto
    — the builder edges from lookahead_gate to END. We assert the
    Command returns no goto (falls through) and writes the artifacts +
    vote."""
    runner = _make_runner_returning(passed=True, details="clean")
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(tmp_path)
    cmd = await gate(state)
    assert isinstance(cmd, Command)
    # Pass path uses no goto — END is a terminal sentinel so there's
    # no add_edge collision risk (see research.py edge comment).
    assert cmd.goto is None or cmd.goto == ()
    assert cmd.update is not None
    assert cmd.update["artifacts"]["lookahead_analysis"]["passed"] is True
    assert cmd.update["artifacts"]["lookahead_analysis"]["details"] == "clean"
    assert cmd.update["agent_votes"] == [
        {
            "agent": "lookahead_gate",
            "verdict": "pass",
            "rationale": "clean",
            "confidence": 1.0,
        }
    ]
    # Pass path does NOT set stage/failure_reason.
    assert "stage" not in (cmd.update or {})
    assert "failure_reason" not in (cmd.update or {})


# ─── fail-route tests ──────────────────────────────────────────────────


async def test_lookahead_gate_fail_routes_to_archive_with_canonical_prefix(
    tmp_path: Path,
) -> None:
    """passed=False → goto='archive' with failure_reason starting
    'lookahead_bias:' (BRD §8 rule 5 canonical prefix)."""
    runner = _make_runner_returning(
        passed=False, details="forward shift in feature column"
    )
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(tmp_path)
    cmd = await gate(state)
    assert cmd.goto == "archive"
    assert cmd.update is not None
    assert cmd.update["stage"] == "archived"
    assert cmd.update["failure_reason"].startswith("lookahead_bias:")
    assert "forward shift in feature column" in cmd.update["failure_reason"]
    # Vote recorded as fail.
    assert cmd.update["agent_votes"][0]["verdict"] == "fail"


# ─── input-validation tests ────────────────────────────────────────────


async def test_lookahead_gate_missing_path_raises(tmp_path: Path) -> None:
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = {
        "strategy_id": "x",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "artifacts": {},
    }
    with pytest.raises(ValueError, match="generated_strategy_path"):
        await gate(state)


async def test_lookahead_gate_missing_file_raises(tmp_path: Path) -> None:
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = {
        "strategy_id": "x",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "artifacts": {"generated_strategy_path": str(tmp_path / "does_not_exist.py")},
    }
    with pytest.raises(FileNotFoundError, match="generated strategy file missing"):
        await gate(state)


# ─── runner-args wiring tests ─────────────────────────────────────────


async def test_lookahead_gate_passes_state_pairs_and_timeframe_to_runner(
    tmp_path: Path,
) -> None:
    """Runner must receive the state's pair list and timeframe verbatim."""
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(
        tmp_path,
        pairs=["ETH/USDT", "SOL/USDT"],
        timeframe="1h",
    )
    await gate(state)
    assert runner.last_call is not None
    assert runner.last_call["pairs"] == ["ETH/USDT", "SOL/USDT"]
    assert runner.last_call["timeframe"] == "1h"


async def test_lookahead_gate_uses_default_timerange_when_state_silent(
    tmp_path: Path,
) -> None:
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(tmp_path)
    await gate(state)
    assert runner.last_call["timerange"] == DEFAULT_LOOKAHEAD_TIMERANGE


async def test_lookahead_gate_uses_state_override_timerange(tmp_path: Path) -> None:
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(tmp_path)
    state["artifacts"]["lookahead_timerange"] = "20231201-20231215"
    await gate(state)
    assert runner.last_call["timerange"] == "20231201-20231215"


async def test_lookahead_gate_preserves_existing_artifacts(tmp_path: Path) -> None:
    runner = _make_runner_returning(passed=True)
    gate = make_lookahead_gate(lookahead_runner=runner)
    state = _state_with_strategy(tmp_path)
    state["artifacts"]["research_proposal"] = {"hypothesis": "h"}
    state["artifacts"]["critic_verdicts"] = [{"verdict": "pass"}]
    cmd = await gate(state)
    artifacts = cmd.update["artifacts"]
    assert artifacts["research_proposal"] == {"hypothesis": "h"}
    assert artifacts["critic_verdicts"] == [{"verdict": "pass"}]
    assert artifacts["generated_strategy_path"] == state["artifacts"]["generated_strategy_path"]
    assert "lookahead_analysis" in artifacts
