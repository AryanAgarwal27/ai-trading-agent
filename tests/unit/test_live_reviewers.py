"""Stage 8d unit tests — the three live-review reviewer nodes (BRD §5.6).

risk_check (plain + small LLM rationale), performance_check (Sonnet agent),
regime_check (plain, no LLM). The LLM calls (risk rationale, performance
review) are injected as stubs so these run in CI with no Anthropic key.

Each reviewer returns a state-update appending exactly one AgentVote whose
verdict is in {continue, pause, fail} (the live-monitoring subset of the
AgentVote vocabulary in orchestrator/state.py).
"""

from __future__ import annotations

from typing import Any, cast

from orchestrator.agents.coordinator import LiveVerdict
from orchestrator.subgraphs.live import performance_check, regime_check, risk_check


def _snapshot(**kw: Any) -> dict[str, Any]:
    base = {
        "max_drawdown": 0.0,
        "daily_pnl_pct": 0.0,
        "consecutive_losses": 0,
        "live_returns": [],
        "paper_returns": [],
        "current_regime": "mid_vol_up",
        "approval_regime": "mid_vol_up",
    }
    base.update(kw)
    return {"strategy_id": "s1", "artifacts": {}, "live_snapshot": base}


def _only_vote(update: dict[str, Any]) -> dict[str, Any]:
    votes = update["agent_votes"]
    assert len(votes) == 1
    return cast(dict[str, Any], votes[0])


async def _stub_rationale(facts: dict[str, Any]) -> str:
    return "stub rationale"


# ─── risk_check (deterministic verdict; LLM only writes the rationale) ──


async def test_risk_check_healthy_continues() -> None:
    update = await risk_check(_snapshot(), rationale_fn=_stub_rationale)
    vote = _only_vote(update)
    assert vote["agent"] == "risk_check"
    assert vote["verdict"] == "continue"


async def test_risk_check_drawdown_breach_pauses() -> None:
    # max_drawdown >= KILL_SWITCH_DRAWDOWN (0.12).
    update = await risk_check(_snapshot(max_drawdown=0.15), rationale_fn=_stub_rationale)
    assert _only_vote(update)["verdict"] == "pause"


async def test_risk_check_daily_loss_breach_pauses() -> None:
    # daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT (-0.03).
    update = await risk_check(_snapshot(daily_pnl_pct=-0.05), rationale_fn=_stub_rationale)
    assert _only_vote(update)["verdict"] == "pause"


async def test_risk_check_consecutive_losses_breach_pauses() -> None:
    update = await risk_check(_snapshot(consecutive_losses=10), rationale_fn=_stub_rationale)
    assert _only_vote(update)["verdict"] == "pause"


async def test_risk_check_uses_injected_rationale() -> None:
    async def _rat(facts: dict[str, Any]) -> str:
        return "drawdown 15% exceeds kill threshold"

    update = await risk_check(_snapshot(max_drawdown=0.15), rationale_fn=_rat)
    assert _only_vote(update)["rationale"] == "drawdown 15% exceeds kill threshold"


# ─── regime_check (pure: compare current vs approval-time regime) ───────


def test_regime_check_same_regime_continues() -> None:
    update = regime_check(_snapshot(current_regime="low_vol_up", approval_regime="low_vol_up"))
    assert _only_vote(update)["verdict"] == "continue"


def test_regime_check_shift_pauses() -> None:
    update = regime_check(_snapshot(current_regime="high_vol_down", approval_regime="low_vol_up"))
    assert _only_vote(update)["verdict"] == "pause"


def test_regime_check_missing_approval_regime_continues_low_confidence() -> None:
    update = regime_check(_snapshot(current_regime="mid_vol_up", approval_regime=None))
    vote = _only_vote(update)
    assert vote["verdict"] == "continue"
    assert vote["confidence"] < 0.5  # can't compare → low confidence


# ─── performance_check (Sonnet agent injected as a stub) ────────────────


async def test_performance_check_uses_review_fn_verdict() -> None:
    async def _review(comparison: dict[str, Any]) -> LiveVerdict:
        return LiveVerdict(verdict="pause", rationale="KS p=0.01 diverged", confidence=0.85)

    update = await performance_check(
        _snapshot(live_returns=[0.1, -0.2, 0.05], paper_returns=[0.01, 0.02, 0.0]),
        review_fn=_review,
    )
    vote = _only_vote(update)
    assert vote["agent"] == "performance_check"
    assert vote["verdict"] == "pause"
    assert vote["confidence"] == 0.85


async def test_performance_check_passes_comparison_to_review_fn() -> None:
    seen: dict[str, Any] = {}

    async def _review(comparison: dict[str, Any]) -> LiveVerdict:
        seen.update(comparison)
        return LiveVerdict(verdict="continue", rationale="tracks paper", confidence=0.7)

    await performance_check(
        _snapshot(live_returns=[0.1, 0.2, 0.1, 0.0], paper_returns=[0.1, 0.15, 0.1, 0.05]),
        review_fn=_review,
    )
    # The node computes the KS/Sharpe comparison and hands it to the agent.
    assert "p_value" in seen
    assert "diverged" in seen
    assert seen["n_paper"] == 4
