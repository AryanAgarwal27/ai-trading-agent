"""Risk Analyst agent — final automated check before HITL paper_gate (BRD §5.4).

Pipeline placement::

    gate_robustness ──pass──> risk_analyst ──approve──> paper_gate (interrupt)
                                          └──reject──> archive

The robustness gate already eliminated the obvious failures with the
cheap deterministic check (Monte Carlo bootstrap, regime coverage, fee
stress thresholds). The risk_analyst's job is NOT to redo that check —
it's to find the subtle, gateable-but-marginal reasons not to spend a
30-day paper-trade slot on this strategy.

Implementation contract (per Stage 4e operator precisions):

  1. ``read_robustness_summary`` is a real ``@tool``-decorated function
     the agent invokes via ``create_agent``. Not a name-only reference.
     The tool reads from a context-local variable that the node function
     sets before calling the agent — keeps the tool's signature
     argument-free (so the LLM has a trivially correct invocation) and
     avoids the InjectedState complexity for a single-fact lookup.

  2. The agent uses ``response_format=RiskVerdict`` so the final output
     is a structured Pydantic instance, not natural language. No fragile
     regex parsing.

  3. The node function is injectable via ``build_validation_subgraph``;
     ``tests/integration/test_validation_subgraph.py`` passes a stubbed
     callable so the real Opus invocation is the operator's manual
     smoke check, not part of the CI gate.

  4. Lazy agent construction: ``_build_risk_analyst_agent`` is only
     called inside the node function, so importing this module on a
     machine without ``ANTHROPIC_API_KEY`` does not fail.

The prompt is deliberately opinionated (BRD §17 #10): a friendly
prompt rubber-stamps; an adversarial one catches edge cases.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

# ─── Context-local robustness summary (used by the tool) ────────────────
# The node function sets this before invoking the agent; the tool reads
# from here. ContextVar is per-asyncio-task so concurrent risk_analyst
# invocations (Stage 8+ live coordinator parallel paths) don't clobber
# each other's summaries.
_current_robustness_summary: ContextVar[str] = ContextVar(
    "risk_analyst.robustness_summary", default="{}"
)


@tool
def read_robustness_summary() -> str:
    """Return the robustness gate's full summary as a JSON string.

    Contains the Monte Carlo bootstrap distribution, per-regime Sharpe
    breakdown, and fee-stress degradation numbers. The strategy already
    cleared the cheap deterministic thresholds in BRD §10; your job is
    to find the subtle red flags those thresholds miss.

    Returns
    -------
    str
        JSON-encoded summary. Example fields:

        - ``monte_carlo.pct_5_final_equity``: 5th-pct bootstrap equity.
          Clearing 1.0 is the threshold; values just above 1.0 are
          marginal and should be scrutinized.
        - ``regime.by_regime``: per-regime mean Sharpe + fold counts.
          A regime with n_folds=1 means the strategy was tested under
          that regime exactly once — barely a sample.
        - ``fee_stress.degradation_2x / 3x``: relative Sharpe drop at
          higher fees. Strategies clearing 0.40 / 0.60 thresholds but
          sitting near them are fragile in a future fee hike.
    """
    return _current_robustness_summary.get()


# ─── Structured verdict ─────────────────────────────────────────────────


class RiskVerdict(BaseModel):
    """Structured output the agent emits via ``response_format``.

    Using a Pydantic class with ``response_format`` removes the need to
    parse free-form text for "APPROVE"/"REJECT" keywords. The agent
    must populate every field, and Pydantic validates the literals.
    """

    decision: Literal["approve", "reject"] = Field(
        description=(
            "approve: strategy advances to paper_gate (30-day dry-run). "
            "reject: strategy archives. Bias toward reject when in doubt — "
            "the operator's $500 capital is small and paper-trade slots "
            "are limited."
        ),
    )
    primary_concern: str = Field(
        description=(
            "One sentence describing the single most important reason for "
            "your decision. For approve: the strongest signal. For reject: "
            "the specific subtle weakness gate_robustness missed."
        ),
    )
    rationale: str = Field(
        description=(
            "2–4 sentences elaborating. Refer to specific numbers from "
            "read_robustness_summary (e.g. 'pct_5_equity=1.02 is marginal' "
            "or 'regime n_folds=1 for high_vol_down')."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the verdict, 0.0 to 1.0.",
    )


# ─── Prompt ─────────────────────────────────────────────────────────────


_RISK_ANALYST_PROMPT = """\
You are the Risk Analyst — the final automated check before this strategy
consumes a 30-day human-monitored paper-trading slot.

Read these rules and follow them strictly:

1. The cheap deterministic gate (Monte Carlo 5th-percentile, regime
   coverage, fee-stress degradation) has ALREADY passed. You are NOT
   here to confirm that pass. You are here to find the subtle reasons
   the strategy should NOT advance.

2. Call read_robustness_summary to get the full JSON of the robustness
   results. Look specifically for:

   - **Marginal threshold clears.** A metric that barely cleared
     (e.g. pct_5_equity = 1.01, regimes_passed = 2 of 3 with one
     regime borderline) means the strategy is fragile to small
     parameter drift or regime shifts.
   - **Asymmetric trade distribution.** If you can infer from the MC
     payload that a few big winners are masking many small losers,
     flag it.
   - **Fold-to-fold inconsistency.** High variance in per-fold Sharpe
     even with a positive mean is overfit-prone.
   - **Regime under-coverage.** A regime with n_folds = 1 is a single
     data point; a "passing" Sharpe there is noise.
   - **Fee fragility at the threshold edge.** 35% degradation at 2x
     clears the 40% threshold but predicts disaster at 2.5x.

3. The operator's capital cap is $500 (SPEC §1 Q3). Paper-trade slots
   are finite. Reject marginal strategies. Approve only when the
   robustness evidence is decisively strong.

4. Emit a RiskVerdict with: decision (approve/reject), primary_concern
   (one sentence), rationale (2–4 sentences citing specific numbers),
   confidence (0.0–1.0).

Do not approve strategies just because gate_robustness passed. The gate
is a floor, not a ceiling.
"""


# ─── Real agent ─────────────────────────────────────────────────────────


def _build_risk_analyst_agent() -> Any:
    """Construct the real ``create_agent``-backed risk analyst.

    Lazy import + lazy construction so this module can be imported on a
    machine without ``ANTHROPIC_API_KEY`` (e.g. importing inside a
    pytest collection pass that's about to skip the integration test).

    Returns the agent runnable; caller invokes via ``ainvoke``.
    """
    # Imports kept local so a missing langchain_anthropic install or a
    # missing API key only fails when the real agent is actually used.
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    # BRD §4 pins Opus 4.7 for risk_analyst (SPEC §1 Q6 confirms).
    # NOTE: temperature/top_p/top_k are intentionally omitted. Claude
    # Opus 4.7 rejects non-default values for these sampling parameters
    # with a 400 error per the official migration guide
    # (https://platform.claude.com/docs/en/about-claude/models/migration-guide
    # — "Setting temperature, top_p, or top_k to any non-default value on
    # Claude Opus 4.7 returns a 400 error. The safest migration path is
    # to omit these parameters entirely from request payloads.").
    # Sonnet 4.6 and Haiku 4.5 still accept temperature, so this caveat
    # is Opus-only — if a future agent module constructs Sonnet/Haiku,
    # passing temperature is fine there. Do NOT re-add temperature here.
    model = ChatAnthropic(
        model="claude-opus-4-7",
        timeout=60.0,
        stop=None,
    )
    return create_agent(
        model=model,
        tools=[read_robustness_summary],
        system_prompt=_RISK_ANALYST_PROMPT,
        response_format=RiskVerdict,
    )


# ─── Node function (used by the validation subgraph) ───────────────────


async def risk_analyst_node(
    state: dict[str, Any],
) -> Command[Literal["paper_gate", "archive"]]:
    """Run the risk_analyst agent and return ``Command(goto, update)``.

    Per BRD §5.4: "reads aggregated robustness; returns
    Command(goto='paper_gate', update={...}) or Command(goto='archive',
    update={'failure_reason':...})".

    Sets the ``_current_robustness_summary`` ContextVar so the
    ``read_robustness_summary`` tool returns real data, invokes the
    agent with a single ``HumanMessage`` kickoff, then maps the agent's
    structured ``RiskVerdict`` to a state update and routes via Command.

    The update writes:
    - ``agent_votes`` (Annotated[list, add] reducer — one new vote appended)
    - ``gate_decisions["risk_analyst"]`` carrying the agent's verdict for
      the dashboard to display (SPEC §4.1 layout requirement).
    - On reject: ``stage="archived"`` + ``failure_reason``.
    """
    summary = json.dumps(state.get("gate_decisions", {}).get("robustness", {}))
    token = _current_robustness_summary.set(summary)
    try:
        agent = _build_risk_analyst_agent()
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Read the robustness summary and emit your verdict.")
                ]
            }
        )
    finally:
        _current_robustness_summary.reset(token)

    verdict: RiskVerdict = result["structured_response"]
    return verdict_to_command(verdict, existing_gates=state.get("gate_decisions") or {})


def verdict_to_command(
    verdict: RiskVerdict,
    *,
    existing_gates: dict[str, Any] | None = None,
) -> Command[Literal["paper_gate", "archive"]]:
    """Translate a :class:`RiskVerdict` into a routing ``Command``.

    Pure function — used by both the real agent path and the stubbed
    test path. Unit tests verify the mapping without needing the agent;
    the integration test injects a stub that returns one of these
    Commands directly.

    Parameters
    ----------
    verdict
        The Pydantic-validated agent output.
    existing_gates
        The CURRENT value of ``state["gate_decisions"]``. We spread these
        into the update so downstream nodes (``paper_gate``) still see
        ``backtest`` and ``robustness`` summaries. Without this, the
        Command's dict update REPLACES the gate_decisions field outright
        — LangGraph does not deep-merge by default.
    """
    base_gates = existing_gates or {}
    base_update: dict[str, Any] = {
        "agent_votes": [
            {
                "agent": "risk_analyst",
                "verdict": "pass" if verdict.decision == "approve" else "fail",
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            },
        ],
        "gate_decisions": {
            **base_gates,
            "risk_analyst": {
                "decision": verdict.decision,
                "primary_concern": verdict.primary_concern,
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            },
        },
    }

    if verdict.decision == "approve":
        return Command(goto="paper_gate", update=base_update)

    return Command(
        goto="archive",
        update={
            **base_update,
            "stage": "archived",
            "failure_reason": f"risk_analyst_reject: {verdict.primary_concern}",
        },
    )
