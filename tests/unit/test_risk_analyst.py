"""Stage 4e unit tests — risk_analyst tool + verdict → Command mapping.

Three groups, all offline (no LLM, no Docker):

1. ``test_read_robustness_summary_*`` — the ``@tool``-decorated function
   reads from the module-level ContextVar that the node function sets
   before invoking the agent.

2. ``test_verdict_to_command_*`` — the pure helper that translates a
   ``RiskVerdict`` into a routing ``Command``. Approve → paper_gate
   with vote + gate_decisions update. Reject → archive with stage=
   archived + failure_reason from primary_concern.

3. ``test_risk_verdict_pydantic_*`` — schema validation: ``decision``
   must be one of the literals, ``confidence`` must be in [0, 1].
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from orchestrator.agents.risk_analyst import (
    RiskVerdict,
    _current_robustness_summary,
    read_robustness_summary,
    verdict_to_command,
)

# ─── 1. read_robustness_summary tool ────────────────────────────────────


def test_read_robustness_summary_returns_contextvar_default_when_unset() -> None:
    """Outside any context, the tool returns ``"{}"`` so the agent's
    first ``read_robustness_summary`` call doesn't crash even on a state
    where ``gate_decisions["robustness"]`` was never populated."""
    out = read_robustness_summary.invoke({})
    assert out == "{}"


def test_read_robustness_summary_returns_contextvar_value_when_set() -> None:
    """The risk_analyst_node sets this ContextVar before invoking the
    agent. The tool then returns whatever the node set."""
    payload = json.dumps({"monte_carlo": {"pct_5_final_equity": 1.05}})
    token = _current_robustness_summary.set(payload)
    try:
        out = read_robustness_summary.invoke({})
    finally:
        _current_robustness_summary.reset(token)
    assert out == payload


def test_read_robustness_summary_is_decorated_as_tool() -> None:
    """The agent invokes this via ``@tool`` dispatch. Confirm the
    decorator wired it up — name + docstring become the LLM-visible
    metadata.

    Stage 4 handoff precision #4: 'tool must be a real @tool-decorated
    function the agent can invoke, not a name-only reference.'
    """
    # langchain @tool gives the function a `.invoke` method and a `.name`.
    assert hasattr(read_robustness_summary, "invoke")
    assert read_robustness_summary.name == "read_robustness_summary"
    assert read_robustness_summary.description
    assert "robustness" in read_robustness_summary.description.lower()


# ─── 2. verdict_to_command ──────────────────────────────────────────────


def _v(decision: str = "approve", confidence: float = 0.85) -> RiskVerdict:
    # cast literal at call site: RiskVerdict's Literal is enforced at runtime
    # by Pydantic; mypy sees `str` here only because the helper signature is
    # generic over both legal values.
    if decision not in ("approve", "reject"):
        raise ValueError(decision)
    return RiskVerdict(
        decision=decision,
        primary_concern="The strategy shows marginal regime coverage.",
        rationale="Three folds in mid_vol_flat, one each in down and up — small n in tail regimes.",
        confidence=confidence,
    )


def test_verdict_to_command_approve_routes_to_paper_gate() -> None:
    cmd = verdict_to_command(_v(decision="approve"))
    assert cmd.goto == "paper_gate"
    update = cmd.update
    assert update is not None
    # agent_votes appended via the reducer (one new vote).
    [vote] = update["agent_votes"]
    assert vote["agent"] == "risk_analyst"
    assert vote["verdict"] == "pass"
    # gate_decisions["risk_analyst"] populated for the dashboard (SPEC §4.1).
    assert update["gate_decisions"]["risk_analyst"]["decision"] == "approve"
    # No stage/failure_reason on approve — those belong to the reject path only.
    assert "stage" not in update
    assert "failure_reason" not in update


def test_verdict_to_command_reject_routes_to_archive_with_failure_reason() -> None:
    cmd = verdict_to_command(_v(decision="reject"))
    assert cmd.goto == "archive"
    update = cmd.update
    assert update is not None
    [vote] = update["agent_votes"]
    assert vote["verdict"] == "fail"
    assert update["stage"] == "archived"
    assert update["failure_reason"].startswith("risk_analyst_reject:")
    # primary_concern is carried into the failure_reason string for audit visibility.
    assert "marginal regime coverage" in update["failure_reason"]


def test_verdict_to_command_preserves_confidence_in_vote_and_dashboard() -> None:
    cmd = verdict_to_command(_v(confidence=0.42))
    update = cmd.update
    assert update is not None
    assert update["agent_votes"][0]["confidence"] == 0.42
    assert update["gate_decisions"]["risk_analyst"]["confidence"] == 0.42


# ─── 3. RiskVerdict Pydantic schema ─────────────────────────────────────


def test_risk_verdict_rejects_invalid_decision_literal() -> None:
    # Build via __pydantic_validator__ to bypass mypy's literal narrowing
    # at the call site — we WANT Pydantic to reject this at runtime.
    with pytest.raises(ValidationError):
        RiskVerdict.model_validate(
            {
                "decision": "maybe",
                "primary_concern": "x",
                "rationale": "y",
                "confidence": 0.5,
            }
        )


def test_risk_verdict_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        RiskVerdict(decision="approve", primary_concern="x", rationale="y", confidence=1.5)
    with pytest.raises(ValidationError):
        RiskVerdict(decision="approve", primary_concern="x", rationale="y", confidence=-0.1)


def test_risk_verdict_accepts_both_decision_literals() -> None:
    assert (
        RiskVerdict(decision="approve", primary_concern="x", rationale="y", confidence=1.0).decision
        == "approve"
    )
    assert (
        RiskVerdict(decision="reject", primary_concern="x", rationale="y", confidence=0.0).decision
        == "reject"
    )
