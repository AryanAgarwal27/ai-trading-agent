"""Tests for orchestrator.agents.critic (Stage 5d).

Unit tests for the deterministic pieces (verdict_to_state_update,
revise_or_proceed routing). The real Opus 4.7 agent is exercised by
scripts/smoke_researcher.py at the operator's discretion — same pattern
as risk_analyst (Stage 4e).

Loop-level tests live in tests/unit/test_critic_loop.py.
"""

from __future__ import annotations

import pytest
from langgraph.constants import END
from pydantic import ValidationError

from orchestrator.agents.critic import (
    MAX_REVISIONS,
    CriticVerdict,
    revise_or_proceed,
    verdict_to_state_update,
)

# ─── CriticVerdict structural sanity ──────────────────────────────────


def test_critic_verdict_accepts_pass() -> None:
    v = CriticVerdict(
        verdict="pass",
        primary_concern="strong hypothesis-encoding",
        rationale="bb_std=2.7 and rsi_buy_threshold=18 both pull toward edges that match the 'aggressive oversold on stretched BB' hypothesis.",
        revision_guidance="",
        confidence=0.85,
    )
    assert v.verdict == "pass"


def test_critic_verdict_accepts_revise() -> None:
    v = CriticVerdict(
        verdict="revise",
        primary_concern="default-hugging on RSI thresholds",
        rationale="rsi_buy_threshold=30 is textbook; hypothesis demands aggressive oversold (≤20).",
        revision_guidance="rsi_buy_threshold should be ≤20 to encode 'aggressive'.",
        confidence=0.9,
    )
    assert v.verdict == "revise"


def test_critic_verdict_rejects_bogus_verdict() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(
            verdict="abstain",  # type: ignore[arg-type]
            primary_concern="x",
            rationale="y",
            revision_guidance="z",
            confidence=0.5,
        )


def test_critic_verdict_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(
            verdict="pass",
            primary_concern="x",
            rationale="y",
            revision_guidance="",
            confidence=0.5,
            sneaky_extra="not allowed",  # type: ignore[call-arg]
        )


# ─── verdict_to_state_update pure-function tests ──────────────────────


def test_verdict_to_state_update_pass_writes_clean_vote() -> None:
    verdict = CriticVerdict(
        verdict="pass",
        primary_concern="strong",
        rationale="r",
        revision_guidance="",
        confidence=0.7,
    )
    update = verdict_to_state_update(verdict)
    assert update["agent_votes"] == [
        {
            "agent": "critic",
            "verdict": "pass",
            "rationale": "r",
            "confidence": 0.7,
        }
    ]
    # Empty revision_guidance → no critic_notes append (avoids feeding
    # the next generator pass an empty-string note).
    assert update["critic_notes"] == []
    # critic_verdicts artifact populated with the dump.
    assert update["artifacts"]["critic_verdicts"] == [verdict.model_dump()]


def test_verdict_to_state_update_revise_appends_critic_note() -> None:
    verdict = CriticVerdict(
        verdict="revise",
        primary_concern="default-hug",
        rationale="midpoints everywhere",
        revision_guidance="bb_std should be ≥ 2.6",
        confidence=0.85,
    )
    update = verdict_to_state_update(verdict)
    assert update["agent_votes"][0]["verdict"] == "revise"
    assert update["critic_notes"] == ["bb_std should be ≥ 2.6"]


def test_verdict_to_state_update_preserves_existing_artifacts() -> None:
    verdict = CriticVerdict(
        verdict="revise",
        primary_concern="x",
        rationale="y",
        revision_guidance="z",
        confidence=0.5,
    )
    existing = {
        "generated_strategy_path": "/tmp/s.py",
        "research_proposal": {"hypothesis": "h"},
        "critic_verdicts": [{"verdict": "revise", "from": "prior_call"}],
    }
    update = verdict_to_state_update(verdict, existing_artifacts=existing)
    # Existing artifacts are preserved.
    assert update["artifacts"]["generated_strategy_path"] == "/tmp/s.py"
    assert update["artifacts"]["research_proposal"] == {"hypothesis": "h"}
    # Prior verdicts list extended, not replaced.
    assert len(update["artifacts"]["critic_verdicts"]) == 2
    assert update["artifacts"]["critic_verdicts"][0] == {
        "verdict": "revise",
        "from": "prior_call",
    }
    assert update["artifacts"]["critic_verdicts"][1] == verdict.model_dump()


# ─── revise_or_proceed routing tests ──────────────────────────────────


def _state_with_critic_vote(verdict: str, *, revision_count: int = 0) -> dict:
    """Build a minimal state with one critic vote of the given verdict."""
    return {
        "revision_count": revision_count,
        "agent_votes": [
            {"agent": "researcher", "verdict": "continue", "confidence": 0.8},
            {"agent": "generator", "verdict": "pass", "confidence": 1.0},
            {"agent": "critic", "verdict": verdict, "confidence": 0.7},
        ],
        "artifacts": {
            "critic_verdicts": [
                {
                    "verdict": verdict,
                    "revision_guidance": "specific guidance text",
                    "primary_concern": "x",
                    "rationale": "y",
                    "confidence": 0.7,
                }
            ],
        },
    }


def test_revise_or_proceed_pass_returns_no_goto() -> None:
    """verdict=pass → Command with empty update and no goto, letting the
    builder's explicit add_edge(revise_or_proceed, END) take effect."""
    cmd = revise_or_proceed(_state_with_critic_vote("pass"))
    assert cmd.goto is None or cmd.goto == ()
    # update is either empty dict or None; either is fine.
    assert not cmd.update or cmd.update == {}


def test_revise_or_proceed_revise_under_cap_routes_to_generator() -> None:
    """verdict=revise + count < MAX_REVISIONS → goto generator, count++."""
    cmd = revise_or_proceed(_state_with_critic_vote("revise", revision_count=0))
    assert cmd.goto == "generator"
    assert cmd.update == {"revision_count": 1}


def test_revise_or_proceed_revise_at_boundary_increments_to_max() -> None:
    """Edge case: count = MAX_REVISIONS - 1 → still allowed; count becomes MAX_REVISIONS."""
    cmd = revise_or_proceed(
        _state_with_critic_vote("revise", revision_count=MAX_REVISIONS - 1)
    )
    assert cmd.goto == "generator"
    assert cmd.update == {"revision_count": MAX_REVISIONS}


def test_revise_or_proceed_revise_at_cap_routes_to_archive() -> None:
    """verdict=revise + count >= MAX_REVISIONS → archive with failure_reason."""
    cmd = revise_or_proceed(
        _state_with_critic_vote("revise", revision_count=MAX_REVISIONS)
    )
    assert cmd.goto == "archive"
    assert cmd.update is not None
    assert cmd.update["stage"] == "archived"
    assert cmd.update["failure_reason"].startswith("critic_loop_exhausted:")
    # Last guidance carried into failure_reason for the operator log.
    assert "specific guidance text" in cmd.update["failure_reason"]
    assert f"MAX_REVISIONS={MAX_REVISIONS}" in cmd.update["failure_reason"]


def test_revise_or_proceed_no_critic_vote_raises() -> None:
    state = {
        "revision_count": 0,
        "agent_votes": [
            {"agent": "researcher", "verdict": "continue", "confidence": 0.8},
        ],
    }
    with pytest.raises(RuntimeError, match="critic.*agent_votes"):
        revise_or_proceed(state)


def test_revise_or_proceed_bogus_verdict_raises() -> None:
    state = {
        "revision_count": 0,
        "agent_votes": [
            {"agent": "critic", "verdict": "abstain", "confidence": 0.5},
        ],
    }
    with pytest.raises(RuntimeError, match="unexpected critic verdict"):
        revise_or_proceed(state)


def test_revise_or_proceed_uses_latest_critic_vote() -> None:
    """When multiple critic votes accumulate across revisions, the
    router must read the LATEST one."""
    state = {
        "revision_count": 1,
        "agent_votes": [
            {"agent": "critic", "verdict": "revise", "confidence": 0.6},
            {"agent": "generator", "verdict": "pass", "confidence": 1.0},
            {"agent": "critic", "verdict": "pass", "confidence": 0.9},
        ],
        "artifacts": {"critic_verdicts": []},
    }
    cmd = revise_or_proceed(state)
    # Latest is pass — no revise, no archive.
    assert cmd.goto is None or cmd.goto == ()


def test_max_revisions_constant_is_three() -> None:
    """BRD §5.3: 'if vote=\"revise\" and revision_count < 3'. The constant
    is the source of truth — this test is a contract check that nobody
    bumps MAX_REVISIONS without explicit operator sign-off."""
    assert MAX_REVISIONS == 3


# END is imported just to confirm the import works in this test module —
# the routing semantics use the Command(no goto) pattern, not END
# directly, but verifying the symbol resolves catches refactors that
# break the langgraph public-API surface.
def test_end_symbol_resolves() -> None:
    assert END is not None
