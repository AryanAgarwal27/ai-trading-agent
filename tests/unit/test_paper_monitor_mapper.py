"""Unit tests for paper_monitor's pure verdict → Command mapper (7d).

No markers — runs in the default CI invocation. The real Haiku agent is
not exercised here; only the deterministic routing mapper.
"""

from __future__ import annotations

from orchestrator.agents.monitors import PaperMonitorVerdict, verdict_to_command


def _verdict(decision: str) -> PaperMonitorVerdict:
    return PaperMonitorVerdict(
        decision=decision,  # type: ignore[arg-type]
        primary_observation=f"obs for {decision}",
        rationale="rationale text",
        confidence=0.8,
    )


def test_rearm_routes_to_paper_wait() -> None:
    cmd = verdict_to_command(_verdict("rearm"))
    assert cmd.goto == "paper_wait"
    assert "stage" not in cmd.update  # rearm does not change lifecycle stage
    vote = cmd.update["agent_votes"][0]
    assert vote["agent"] == "paper_monitor"
    assert vote["verdict"] == "continue"


def test_advance_routes_to_live_gate() -> None:
    cmd = verdict_to_command(_verdict("advance"))
    assert cmd.goto == "live_gate"
    assert cmd.update["agent_votes"][0]["verdict"] == "pass"
    assert cmd.update["gate_decisions"]["paper_monitor"]["decision"] == "advance"


def test_kill_routes_to_archive_with_failure_reason() -> None:
    cmd = verdict_to_command(_verdict("kill"))
    assert cmd.goto == "archive"
    assert cmd.update["stage"] == "archived"
    assert cmd.update["failure_reason"].startswith("paper_monitor_kill:")
    assert "obs for kill" in cmd.update["failure_reason"]
    assert cmd.update["agent_votes"][0]["verdict"] == "fail"


def test_existing_gate_decisions_preserved() -> None:
    """The mapper must not drop prior gate_decisions keys."""
    existing = {
        "backtest": {"passed": True, "sharpe_is": 1.9},
        "risk_analyst": {"decision": "approve"},
    }
    cmd = verdict_to_command(_verdict("advance"), existing_gates=existing)
    gd = cmd.update["gate_decisions"]
    assert gd["backtest"]["sharpe_is"] == 1.9
    assert gd["risk_analyst"]["decision"] == "approve"
    assert gd["paper_monitor"]["decision"] == "advance"


def test_confidence_and_rationale_carried_through() -> None:
    v = PaperMonitorVerdict(
        decision="rearm",
        primary_observation="too early",
        rationale="only 3 days elapsed, metrics nominal",
        confidence=0.42,
    )
    cmd = verdict_to_command(v)
    pm = cmd.update["gate_decisions"]["paper_monitor"]
    assert pm["confidence"] == 0.42
    assert pm["rationale"] == "only 3 days elapsed, metrics nominal"
