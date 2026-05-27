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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from orchestrator.agents.risk_analyst import (
    RiskVerdict,
    _build_risk_analyst_agent,
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


# ─── 4. Regression: no temperature kwarg in Opus 4.7 construction ───────


def test_build_risk_analyst_agent_does_not_pass_temperature_to_opus_4_7() -> None:
    """Regression guard for the Stage 4e smoke-check fix.

    Opus 4.7 rejects ``temperature``/``top_p``/``top_k`` with a 400 error
    ("Setting temperature, top_p, or top_k to any non-default value on
    Claude Opus 4.7 returns a 400 error.") per the official Anthropic
    migration guide:
    https://platform.claude.com/docs/en/about-claude/models/migration-guide

    The first smoke check failed with::

        anthropic.BadRequestError: Error code: 400 - ... 'message':
        '`temperature` is deprecated for this model.'

    This test mocks ``ChatAnthropic`` and ``create_agent`` so no real
    LLM call happens, then asserts the constructor invocation does NOT
    include any of the three sampling kwargs. Sonnet 4.6 and Haiku 4.5
    still accept ``temperature``; if a future agent constructs those
    models, the constraint here applies only to Opus.
    """
    captured_kwargs: dict[str, Any] = {}

    def fake_chat_anthropic(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return MagicMock(name="ChatAnthropicMock")

    # The imports are local to _build_risk_analyst_agent (so the module can
    # be imported without ANTHROPIC_API_KEY), so we patch them at their
    # source modules rather than as attributes of risk_analyst.
    with (
        patch(
            "langchain_anthropic.ChatAnthropic",
            side_effect=fake_chat_anthropic,
        ),
        patch(
            "langchain.agents.create_agent",
            return_value=MagicMock(name="AgentMock"),
        ),
    ):
        _build_risk_analyst_agent()

    # The Opus model must be selected.
    assert (
        captured_kwargs.get("model") == "claude-opus-4-7"
    ), f"agent must construct Opus 4.7, got {captured_kwargs.get('model')!r}"
    # The three sampling parameters Opus 4.7 rejects must NOT appear.
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in captured_kwargs, (
            f"{forbidden!r} kwarg was passed to ChatAnthropic for Opus 4.7; "
            "this triggers a 400 deprecation error. See "
            "https://platform.claude.com/docs/en/about-claude/models/migration-guide"
        )
