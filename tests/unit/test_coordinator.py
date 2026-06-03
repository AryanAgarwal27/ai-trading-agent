"""Stage 8d unit tests — coordinator escalation logic + gate_audits row.

These are pure-unit (no Postgres, no real LLM): the Sonnet merge, the Opus
arbitration, and the gate_audits DB write are all injected as stubs. What we
pin down here is the three design decisions made in code:

  Decision 1 — needs_escalation predicate (majority resolves; only a 3-way
               split escalates to Opus).
  Decision 2 — arbitrate (Opus reads the reviewer votes + Sonnet's merge
               attempt and produces the final verdict).
  Decision 3 — the gate_audits row written on every escalation carries the
               full reconstruction payload.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.types import Command

from orchestrator.agents.coordinator import (
    VERDICT_TO_AUDIT_DECISION,
    LiveVerdict,
    coordinator,
    needs_escalation,
)
from orchestrator.state import AgentVote


def _vote(agent: str, verdict: str, *, rationale: str = "r", confidence: float = 0.8) -> AgentVote:
    return cast(
        AgentVote,
        {"agent": agent, "verdict": verdict, "rationale": rationale, "confidence": confidence},
    )


def _three(risk: str, perf: str, regime: str) -> list[AgentVote]:
    return [
        _vote("risk_check", risk),
        _vote("performance_check", perf),
        _vote("regime_check", regime),
    ]


def _state(votes: list[AgentVote], strategy_id: str = "s1") -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "agent_votes": votes,
        "gate_decisions": {},
        "artifacts": {},
    }


# ─── Decision 1: needs_escalation predicate ─────────────────────────────


def test_unanimous_does_not_escalate() -> None:
    assert needs_escalation(_three("continue", "continue", "continue")) is False


def test_two_one_majority_does_not_escalate() -> None:
    # 2 continue / 1 pause — clear plurality, Sonnet merge resolves.
    assert needs_escalation(_three("continue", "continue", "pause")) is False
    # 2 pause / 1 fail.
    assert needs_escalation(_three("pause", "pause", "fail")) is False


def test_three_way_split_escalates() -> None:
    # continue / pause / fail — no plurality, escalate to Opus.
    assert needs_escalation(_three("continue", "pause", "fail")) is True


# ─── stubs ──────────────────────────────────────────────────────────────


def _merge_stub(verdict: str) -> Any:
    async def _fn(votes: list[AgentVote]) -> LiveVerdict:
        return LiveVerdict(verdict=verdict, rationale="sonnet merge", confidence=0.6)

    return _fn


def _arbitrate_stub(verdict: str, captured: dict[str, Any] | None = None) -> Any:
    async def _fn(votes: list[AgentVote], merge: LiveVerdict) -> LiveVerdict:
        if captured is not None:
            captured["called"] = True
            captured["merge_seen"] = merge
        return LiveVerdict(verdict=verdict, rationale="opus arbitration", confidence=0.9)

    return _fn


def _audit_capture(captured: dict[str, Any]) -> Any:
    async def _fn(**kwargs: Any) -> int:
        captured["audit"] = kwargs
        return 1

    return _fn


# ─── non-escalation path: majority routes, no Opus, no audit ────────────


async def test_majority_continue_routes_to_live_wait_without_opus() -> None:
    arb: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    cmd = await coordinator(
        _state(_three("continue", "continue", "pause")),
        merge_fn=_merge_stub("continue"),
        arbitrate_fn=_arbitrate_stub("fail", arb),
        gate_audit_writer_fn=_audit_capture(audit),
    )
    assert isinstance(cmd, Command)
    assert cmd.goto == "live_wait"
    assert arb == {}, "arbitrate must NOT run on a resolvable majority"
    assert audit == {}, "no gate_audits row on a non-escalation"


async def test_majority_pause_routes_to_live_pause() -> None:
    cmd = await coordinator(
        _state(_three("pause", "pause", "continue")),
        merge_fn=_merge_stub("pause"),
        arbitrate_fn=_arbitrate_stub("continue"),
        gate_audit_writer_fn=_audit_capture({}),
    )
    assert cmd.goto == "live_pause"


async def test_majority_routing_is_deterministic_not_merge_verdict() -> None:
    """On a clear majority the route follows the DETERMINISTIC majority, not
    whatever the Sonnet merge stub returns — the merge only supplies rationale."""
    cmd = await coordinator(
        _state(_three("continue", "continue", "pause")),
        merge_fn=_merge_stub("fail"),  # merge disagrees with the majority
        arbitrate_fn=_arbitrate_stub("fail"),
        gate_audit_writer_fn=_audit_capture({}),
    )
    assert cmd.goto == "live_wait"  # majority=continue wins, not merge's "fail"


# ─── escalation path: Opus arbitrates, gate_audits written ──────────────


async def test_three_way_split_escalates_and_opus_verdict_wins() -> None:
    arb: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    cmd = await coordinator(
        _state(_three("continue", "pause", "fail")),
        merge_fn=_merge_stub("pause"),  # Sonnet's merge attempt
        arbitrate_fn=_arbitrate_stub("fail", arb),  # Opus arbitrates -> fail
        gate_audit_writer_fn=_audit_capture(audit),
    )
    assert arb.get("called") is True, "arbitrate must run on a 3-way split"
    # Decision 2 (arbitrate): Opus sees Sonnet's merge attempt.
    assert isinstance(arb["merge_seen"], LiveVerdict)
    assert arb["merge_seen"].verdict == "pause"
    # Opus's verdict (fail) wins -> archive.
    assert cmd.goto == "archive"
    assert cmd.update is not None
    assert cmd.update["stage"] == "archived"
    assert audit, "escalation MUST write a gate_audits row"


async def test_gate_audits_row_content_is_reconstructable() -> None:
    audit: dict[str, Any] = {}
    await coordinator(
        _state(_three("continue", "pause", "fail")),
        merge_fn=_merge_stub("pause"),
        arbitrate_fn=_arbitrate_stub("pause"),
        gate_audit_writer_fn=_audit_capture(audit),
    )
    row = audit["audit"]
    assert row["gate"] == "live"
    assert row["actor"] == "coordinator:opus"
    # pause -> auto_pause (requires migration 0002).
    assert row["decision"] == VERDICT_TO_AUDIT_DECISION["pause"] == "auto_pause"

    payload = row["payload"]
    assert payload["event_type"] == "coordinator_escalation"
    assert "timestamp" in payload
    assert len(payload["reviewer_votes"]) == 3
    assert {v["agent"] for v in payload["reviewer_votes"]} == {
        "risk_check",
        "performance_check",
        "regime_check",
    }
    assert payload["sonnet_merge_attempt"]["verdict"] == "pause"
    assert payload["opus_verdict"]["verdict"] == "pause"
    assert payload["final_decision"] == "pause"
    assert payload["disagreement_reason"] == "3-way_split"


async def test_escalation_decision_mapping_covers_all_verdicts() -> None:
    assert VERDICT_TO_AUDIT_DECISION == {
        "continue": "auto_continue",
        "pause": "auto_pause",
        "fail": "auto_fail",
    }


async def test_coordinator_appends_its_own_vote() -> None:
    cmd = await coordinator(
        _state(_three("continue", "continue", "continue")),
        merge_fn=_merge_stub("continue"),
        arbitrate_fn=_arbitrate_stub("continue"),
        gate_audit_writer_fn=_audit_capture({}),
    )
    assert cmd.update is not None
    votes = cmd.update["agent_votes"]
    assert any(v["agent"] == "coordinator" for v in votes)
