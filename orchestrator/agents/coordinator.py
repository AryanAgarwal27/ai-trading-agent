"""Coordinator agent — merges the three live reviewers' votes (BRD §5.6).

Pipeline placement (the Send fan-out + node wiring lands in 8e)::

    live_evaluate ──Send──> risk_check ┐
                  ──Send──> performance_check ├─> coordinator ─> live_wait
                  ──Send──> regime_check ┘                     ├─> live_pause
                                                               └─> archive

This 8d commit ships the coordinator's DECISION LOGIC + the three reviewer
nodes (in orchestrator/subgraphs/live.py). The three Stage-8 design decisions
are encoded here:

DECISION 1 — what counts as "disagreement" (escalation to Opus)?
    **Majority sufficient; only a 3-way split escalates** (operator option 2).
    With three reviewers and verdicts in {continue, pause, fail}, any 2/3
    plurality is a clear resolution the cheap Sonnet merge can apply. Only a
    1-1-1 split (all three distinct) has no plurality, so only that escalates
    to Opus. Reasoning: the coordinator is the SOFT safety layer; the HARD
    floor is the out-of-band kill switch (BRD §1.1 rule 7, §11, Stage 8f),
    which stops the bot on a real drawdown/consecutive-loss breach regardless
    of any coordinator verdict. Because catastrophic risk is independently
    backstopped, the coordinator need not burn Opus on every dissent — escalate
    only the genuinely unresolvable split. (Rejects option 3 "any pause
    escalates": one noisy reviewer would drag every cycle through Opus.)

DECISION 2 — Opus re-vote vs. arbitrate?
    **Arbitrate.** On escalation, Sonnet still runs (its merge attempt is
    captured), then Opus reads the reviewer votes + rationales + Sonnet's merge
    attempt and produces the final verdict. Reasoning: Opus gets maximal
    context (re-vote would discard the Sonnet reasoning we already paid for),
    and the audit trail is richer — a post-mortem sees what Opus DID with the
    merge, not just an independent second opinion.

DECISION 3 — gate_audits row on every escalation.
    Written via the existing ``record_gate_audit`` writer with gate='live',
    actor='coordinator:opus', decision = the mapped final verdict, and a
    payload carrying everything needed to reconstruct the escalation months
    later (reviewer votes, Sonnet merge attempt, Opus verdict, final decision,
    disagreement reason). The decision column needed two new values
    (auto_continue, auto_pause) — added in migration 0002, since the original
    CHECK only allowed auto_pass/auto_fail/human_*.

Non-escalation routing is DETERMINISTIC: the final verdict is the computed
majority (not whatever the Sonnet merge returns), so a clear 2/3 split always
routes to the majority outcome. Sonnet supplies the human-readable rationale.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langgraph.types import Command
from pydantic import BaseModel, Field

from orchestrator.observability.events import record_gate_audit
from orchestrator.state import AgentVote

logger = logging.getLogger(__name__)

# The three reviewer agent names whose votes the coordinator merges.
REVIEWER_AGENTS: tuple[str, ...] = ("risk_check", "performance_check", "regime_check")

# Final-verdict → routing target (the live subgraph nodes 8e wires).
VERDICT_TO_GOTO: dict[str, str] = {
    "continue": "live_wait",
    "pause": "live_pause",
    "fail": "archive",
}

# Final-verdict → gate_audits.decision value. continue/pause need the values
# added in migration 0002 (the original CHECK had only auto_pass/auto_fail).
VERDICT_TO_AUDIT_DECISION: dict[str, str] = {
    "continue": "auto_continue",
    "pause": "auto_pause",
    "fail": "auto_fail",
}


class LiveVerdict(BaseModel):
    """Structured verdict shared by the reviewers' LLM outputs + the coordinator.

    The verdict vocabulary is the live-monitoring subset of AgentVote: continue
    (keep trading), pause (halt → HITL live_pause_review), fail (archive).
    """

    verdict: Literal["continue", "pause", "fail"] = Field(
        description=(
            "continue: metrics nominal, keep trading. pause: halt and route to "
            "HITL review (recoverable). fail: archive the strategy (terminal)."
        ),
    )
    rationale: str = Field(description="2-4 sentences citing the reviewer signals.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence.")


# Injection seams (8e wires the real agents; tests + 8d pass stubs).
MergeFn = Callable[[list[AgentVote]], Awaitable[LiveVerdict]]
ArbitrateFn = Callable[[list[AgentVote], LiveVerdict], Awaitable[LiveVerdict]]
GateAuditWriterFn = Callable[..., Awaitable[int]]


# ─── Decision 1: the escalation predicate ───────────────────────────────


def latest_reviewer_votes(agent_votes: list[AgentVote]) -> list[AgentVote]:
    """Return the most-recent vote per reviewer agent, in REVIEWER_AGENTS order.

    ``agent_votes`` accumulates across the whole strategy lifecycle (critic,
    risk_analyst, paper_monitor, …); the coordinator only cares about the
    current live cycle's three reviewers. Taking the LAST vote per reviewer
    name is robust across multiple live wake-cycles.
    """
    latest: dict[str, AgentVote] = {}
    for vote in agent_votes:
        if vote.get("agent") in REVIEWER_AGENTS:
            latest[vote["agent"]] = vote
    return [latest[name] for name in REVIEWER_AGENTS if name in latest]


def _majority_verdict(verdicts: list[str]) -> str | None:
    """Return the strictly-majority verdict, or None when there is no majority.

    "Majority" = more than half. For three votes that means ≥ 2 agreeing; a
    1-1-1 split returns None (the escalation trigger).
    """
    if not verdicts:
        return None
    verdict, count = Counter(verdicts).most_common(1)[0]
    return verdict if count * 2 > len(verdicts) else None


def needs_escalation(votes: list[AgentVote]) -> bool:
    """DECISION 1: escalate only when three reviewers split three ways.

    True iff there are exactly three reviewer votes with no majority verdict
    (i.e. all three distinct). A unanimous or 2-1 outcome resolves on the
    cheap Sonnet merge.
    """
    verdicts = [str(v["verdict"]) for v in votes]
    if len(verdicts) < 3:
        # Incomplete fan-out — not a three-way split; let the coordinator's
        # defensive path handle it (no Opus escalation).
        return False
    return _majority_verdict(verdicts) is None


# ─── Real Sonnet merge / Opus arbitrate (defaults; 8e/prod path) ────────


def _build_merge_agent() -> Any:
    """Lazy-build the Sonnet 4.6 merge agent (BRD §4)."""
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-sonnet-4-6", timeout=60.0, stop=None)
    return create_agent(
        model=model,
        tools=[],
        system_prompt=_MERGE_PROMPT,
        response_format=LiveVerdict,
    )


def _build_arbitrate_agent() -> Any:
    """Lazy-build the Opus 4.7 arbitration agent (BRD §4; escalation only)."""
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    # Opus 4.7 rejects non-default temperature/top_p/top_k (see risk_analyst.py).
    model = ChatAnthropic(model="claude-opus-4-7", timeout=60.0, stop=None)
    return create_agent(
        model=model,
        tools=[],
        system_prompt=_ARBITRATE_PROMPT,
        response_format=LiveVerdict,
    )


_MERGE_PROMPT = """\
You are the live-trade Coordinator. Three reviewers (risk_check,
performance_check, regime_check) have each voted continue / pause / fail on a
live strategy. Merge their votes into a single verdict and a concise rationale
that cites what each reviewer found. Prefer pause over fail unless a reviewer
presents decisive evidence the strategy should be archived. Emit a LiveVerdict.
"""

_ARBITRATE_PROMPT = """\
You are the senior live-trade Coordinator, invoked ONLY when the three
reviewers split three ways (continue / pause / fail) with no majority. You are
given the reviewer votes + rationales AND the junior coordinator's merge
attempt. Arbitrate: weigh the evidence, treat the merge attempt as one signal
(not binding), and produce the final verdict. Bias toward pause (route to human
review) over fail (archive) unless the evidence for archiving is decisive. Emit
a LiveVerdict.
"""


async def _default_merge_fn(votes: list[AgentVote]) -> LiveVerdict:
    from langchain_core.messages import HumanMessage

    agent = _build_merge_agent()
    kickoff = f"Reviewer votes: {[dict(v) for v in votes]}. Merge them."
    result = await agent.ainvoke({"messages": [HumanMessage(content=kickoff)]})
    return cast(LiveVerdict, result["structured_response"])


async def _default_arbitrate_fn(votes: list[AgentVote], merge: LiveVerdict) -> LiveVerdict:
    from langchain_core.messages import HumanMessage

    agent = _build_arbitrate_agent()
    kickoff = (
        f"Reviewer votes: {[dict(v) for v in votes]}. "
        f"Junior merge attempt: verdict={merge.verdict}, rationale={merge.rationale}. "
        "Arbitrate and emit the final verdict."
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=kickoff)]})
    return cast(LiveVerdict, result["structured_response"])


# ─── Coordinator node ────────────────────────────────────────────────────


async def coordinator(
    state: dict[str, Any],
    config: Any = None,
    *,
    merge_fn: MergeFn | None = None,
    arbitrate_fn: ArbitrateFn | None = None,
    gate_audit_writer_fn: GateAuditWriterFn | None = None,
) -> Command[Any]:
    """Merge reviewer votes → routing Command (BRD §5.6).

    Non-escalation: route on the deterministic majority verdict; Sonnet supplies
    the rationale. Escalation (3-way split): Sonnet's merge attempt is captured,
    Opus arbitrates, the final verdict is Opus's, and a gate_audits row is
    written (DECISION 3). Routing: continue→live_wait, pause→live_pause,
    fail→archive.
    """
    merge = merge_fn or _default_merge_fn
    arbitrate = arbitrate_fn or _default_arbitrate_fn
    write_audit = gate_audit_writer_fn or record_gate_audit

    votes = latest_reviewer_votes(state.get("agent_votes") or [])
    verdicts = [str(v["verdict"]) for v in votes]
    escalate = needs_escalation(votes)

    # Sonnet merge always runs — it is the merge on the resolvable path and the
    # logged "merge attempt" on the escalation path.
    merge_attempt = await merge(votes)

    if not escalate:
        majority = _majority_verdict(verdicts)
        if majority is None:
            # Defensive: incomplete fan-out (<3 votes, no majority). Pause to
            # HITL rather than guess — should not happen once 8e wires all three.
            final = LiveVerdict(
                verdict="pause",
                rationale="incomplete reviewer fan-out; pausing for human review",
                confidence=0.3,
            )
        else:
            final = LiveVerdict(
                verdict=cast(Literal["continue", "pause", "fail"], majority),
                rationale=merge_attempt.rationale,
                confidence=merge_attempt.confidence,
            )
        actor = "coordinator:sonnet"
    else:
        opus = await arbitrate(votes, merge_attempt)
        final = opus
        actor = "coordinator:opus"
        await _write_escalation_audit(
            state=state,
            votes=votes,
            merge_attempt=merge_attempt,
            opus_verdict=opus,
            disagreement_reason="3-way_split",
            write_audit=write_audit,
        )

    return _route(state, final, actor)


async def _write_escalation_audit(
    *,
    state: dict[str, Any],
    votes: list[AgentVote],
    merge_attempt: LiveVerdict,
    opus_verdict: LiveVerdict,
    disagreement_reason: str,
    write_audit: GateAuditWriterFn,
) -> None:
    """DECISION 3: durable, reconstructable gate_audits row for an escalation."""
    payload: dict[str, Any] = {
        "event_type": "coordinator_escalation",
        "timestamp": datetime.now(UTC).isoformat(),
        "reviewer_votes": [dict(v) for v in votes],
        "sonnet_merge_attempt": {
            "verdict": merge_attempt.verdict,
            "rationale": merge_attempt.rationale,
            "confidence": merge_attempt.confidence,
        },
        "opus_verdict": {
            "verdict": opus_verdict.verdict,
            "rationale": opus_verdict.rationale,
            "confidence": opus_verdict.confidence,
        },
        "final_decision": opus_verdict.verdict,
        "disagreement_reason": disagreement_reason,
    }
    try:
        await write_audit(
            strategy_id=str(state.get("strategy_id", "")),
            gate="live",
            decision=VERDICT_TO_AUDIT_DECISION[opus_verdict.verdict],
            actor="coordinator:opus",
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001 — audit write must not strand routing
        logger.error(
            "coordinator gate_audits write failed strategy_id=%s exc=%s",
            state.get("strategy_id"),
            exc,
        )


def _route(state: dict[str, Any], final: LiveVerdict, actor: str) -> Command[Any]:
    """Build the routing Command + state update from the final verdict."""
    existing = state.get("gate_decisions") or {}
    update: dict[str, Any] = {
        "agent_votes": [
            {
                "agent": "coordinator",
                "verdict": final.verdict,
                "rationale": final.rationale,
                "confidence": final.confidence,
            }
        ],
        "gate_decisions": {
            **existing,
            "coordinator": {
                "verdict": final.verdict,
                "rationale": final.rationale,
                "confidence": final.confidence,
                "by": actor,
            },
        },
    }
    if final.verdict == "fail":
        update["stage"] = "archived"
        update["failure_reason"] = f"coordinator_archive: {final.rationale}"
    return Command(goto=VERDICT_TO_GOTO[final.verdict], update=update)
