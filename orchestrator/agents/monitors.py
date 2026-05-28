"""Paper monitor agent — wake-cycle paper-trade evaluation (BRD §5.5).

Pipeline placement (the node wiring lands in 7e)::

    paper_wait ──wake──> paper_monitor ──rearm──> paper_wait
                                       ├──advance──> divergence_check ──> live_gate
                                       └──kill────> archive

This 7d commit ships the AGENT, its TOOLS, the structured verdict, the
``run_paper_monitor`` invocation core, and the pure ``verdict_to_command``
mapper. The LangGraph node that fetches metrics from the live Freqtrade
container, builds the context, and routes the Command lands in 7e.

Design notes:

1. **Snapshot tools, not live calls.** The six BRD §5.5 tools
   (``ft_status``, ``ft_profit``, ``ft_trades``, ``ft_performance``,
   ``compare_to_backtest``, ``check_kill_switch``) read from a
   pre-fetched :class:`PaperMonitorContext` held in a ContextVar. The
   7e node fetches the metrics ONCE at wake and stuffs them into the
   context before invoking the agent. This gives the agent a consistent
   snapshot (metrics don't shift mid-reasoning), avoids async-tool
   plumbing, and collapses what could be many REST round-trips into one
   fetch. The tools are sync and argument-free, so the LLM's invocations
   are trivially correct.

2. **Model: Haiku 4.5** (BRD §4, §12 — routine paper monitoring is the
   cheap-model path; ~120 wakes per lifecycle at Haiku pricing).

3. **30-day clock is NOT enforced here.** ``check_kill_switch`` surfaces
   ``elapsed_days`` + ``MIN_PAPER_DAYS`` so the agent won't recommend
   "advance" prematurely, but the HARD gate is the deterministic
   ``divergence_check`` node (7e) downstream — belt and braces per
   BRD §5.5. The agent is advisory on timing; deterministic code
   enforces it.

4. **Injection seam.** 7e's node accepts a ``paper_monitor_fn`` that
   defaults to :func:`run_paper_monitor`; tests pass a stub (see
   ``tests/fixtures/monitors.py``) so CI never makes a real Haiku call.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from orchestrator.gates.thresholds import MIN_PAPER_DAYS
from orchestrator.tools.compare import compare_paper_to_backtest

# ─── Context snapshot (read by the tools) ──────────────────────────────


@dataclass
class PaperMonitorContext:
    """Pre-fetched paper metrics + backtest baseline for one wake.

    The 7e node builds this from the live Freqtrade REST API; the smoke
    probe + tests build it synthetically. Held in a ContextVar so the
    argument-free tools can read it. ContextVar (not a global) keeps
    concurrent monitor invocations — Stage 8+ may run several paper
    threads' wakes overlapping — from clobbering each other's snapshot.
    """

    status: list[dict[str, Any]] = field(default_factory=list)
    profit: dict[str, Any] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    performance: list[dict[str, Any]] = field(default_factory=list)
    backtest_returns: list[float] = field(default_factory=list)
    kill_switch_fired: bool = False
    elapsed_days: float = 0.0


_monitor_ctx: ContextVar[PaperMonitorContext] = ContextVar(
    "paper_monitor.context", default=PaperMonitorContext()
)


# ─── Tools (BRD §5.5) ──────────────────────────────────────────────────


@tool
def ft_status() -> str:
    """Return the list of currently-open paper trades as JSON.

    Each element is one open position with entry price, current rate,
    unrealized profit, stake, and duration. An empty list means no
    open positions right now.
    """
    return json.dumps(_monitor_ctx.get().status)


@tool
def ft_profit() -> str:
    """Return the cumulative P&L + drawdown summary as JSON.

    Key fields: ``profit_closed_percent`` (realized return so far),
    ``profit_all_percent`` (incl. unrealized), ``max_drawdown``,
    ``trade_count``, ``winning_trades`` / ``losing_trades``.
    """
    return json.dumps(_monitor_ctx.get().profit)


@tool
def ft_trades() -> str:
    """Return recent closed paper trades as JSON.

    Each trade carries ``profit_ratio`` (per-trade return), open/close
    timestamps, pair, and exit reason. This is the raw material for
    judging the per-trade return distribution.
    """
    return json.dumps(_monitor_ctx.get().trades)


@tool
def ft_performance() -> str:
    """Return per-pair aggregate performance as JSON.

    One entry per traded pair: total profit, count, and average. Useful
    for spotting a single pair carrying (or dragging) the whole result.
    """
    return json.dumps(_monitor_ctx.get().performance)


@tool
def compare_to_backtest() -> str:
    """Compare live paper per-trade returns to the backtest distribution.

    Runs a two-sample Kolmogorov-Smirnov test + a relative Sharpe
    deviation check (BRD §10 thresholds) and returns JSON with:
    ``p_value`` (KS; < 0.05 means the distributions differ),
    ``sharpe_deviation`` (relative; > 0.30 means performance drift),
    and ``diverged`` (True if EITHER fires). A diverged result is the
    primary evidence for a "kill" verdict — the live behaviour no
    longer matches what justified advancing the strategy.
    """
    ctx = _monitor_ctx.get()
    paper_returns = [
        float(t["profit_ratio"])
        for t in ctx.trades
        if isinstance(t, dict) and t.get("profit_ratio") is not None
    ]
    return json.dumps(compare_paper_to_backtest(paper_returns, ctx.backtest_returns))


@tool
def check_kill_switch() -> str:
    """Report kill-switch state + the 30-day paper clock as JSON.

    Fields:
    - ``kill_switch_fired``: True if the out-of-band kill switch tripped
      during this paper run (drawdown / consecutive-loss breach). If
      True, vote "kill".
    - ``elapsed_days``: days the strategy has been paper-trading.
    - ``min_paper_days``: the minimum required before "advance" is
      permissible. Do NOT vote "advance" if ``elapsed_days`` is below
      this — the deterministic gate downstream will reject it anyway.
    """
    ctx = _monitor_ctx.get()
    return json.dumps(
        {
            "kill_switch_fired": ctx.kill_switch_fired,
            "elapsed_days": ctx.elapsed_days,
            "min_paper_days": MIN_PAPER_DAYS,
        }
    )


# ─── Structured verdict ────────────────────────────────────────────────


class PaperMonitorVerdict(BaseModel):
    """Structured output the agent emits via ``response_format``."""

    decision: Literal["rearm", "advance", "kill"] = Field(
        description=(
            "rearm: keep paper-trading, re-arm the wake cycle (metrics fine "
            "but not yet ready to graduate, or not enough elapsed days). "
            "advance: paper performance matches backtest and >= min_paper_days "
            "elapsed; route to live_gate for human approval. "
            "kill: distribution diverged, kill switch fired, or performance "
            "decisively bad; archive the strategy."
        ),
    )
    primary_observation: str = Field(
        description=(
            "One sentence: the single most important fact behind your "
            "decision (e.g. 'KS p=0.01, paper returns diverged from backtest' "
            "or 'only 4 of 30 paper days elapsed, metrics nominal')."
        ),
    )
    rationale: str = Field(
        description=(
            "2-4 sentences citing specific numbers from your tools "
            "(profit_closed_percent, max_drawdown, KS p_value, "
            "sharpe_deviation, elapsed_days)."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the verdict, 0.0 to 1.0.",
    )


# ─── Prompt ─────────────────────────────────────────────────────────────


_PAPER_MONITOR_PROMPT = """\
You are the Paper Monitor — you evaluate a dry-run (paper) trading
strategy on each wake of its 30-day paper-trade cycle and decide whether
to keep going, graduate it to live, or kill it.

Procedure on each wake:

1. Call check_kill_switch FIRST. If kill_switch_fired is true, your
   decision is "kill" — the out-of-band safety system already stopped
   the bot; you are confirming the archive. Note elapsed_days and
   min_paper_days for step 4.

2. Call ft_profit and ft_status to see realized/unrealized P&L,
   drawdown, and open positions.

3. Call compare_to_backtest. This is the decisive signal:
   - If diverged is true (KS p_value < 0.05 OR sharpe_deviation > 0.30),
     the live paper behaviour no longer matches the backtest that
     justified this strategy. Strong evidence for "kill" — unless the
     sample is tiny (very few paper trades so far), in which case
     "rearm" and wait for more data.
   - If diverged is false, the strategy is tracking its backtest.

4. Decide:
   - "kill": kill switch fired, OR distributions diverged on a
     meaningful sample, OR drawdown is severe.
   - "advance": NOT diverged, metrics healthy, AND
     elapsed_days >= min_paper_days. Never advance before the minimum
     elapsed days — a deterministic gate will reject it and you'll have
     wasted the recommendation.
   - "rearm": the default. Metrics are fine but it's too early to
     graduate (elapsed_days < min_paper_days), or the sample is still
     too small to trust the comparison.

5. Emit a PaperMonitorVerdict: decision, primary_observation (one
   sentence), rationale (2-4 sentences citing specific numbers),
   confidence (0.0-1.0).

Bias toward "rearm" when uncertain. Advancing prematurely risks real
capital; killing prematurely wastes a strategy that might still prove
out. Only "advance" on strong, sufficiently-sampled evidence.
"""


# ─── Real agent ─────────────────────────────────────────────────────────


def _build_paper_monitor_agent() -> Any:
    """Construct the real ``create_agent``-backed paper monitor (Haiku 4.5).

    Lazy import + construction so importing this module on a machine
    without ``ANTHROPIC_API_KEY`` doesn't fail (e.g. a pytest collection
    pass about to skip the integration test).
    """
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    # BRD §4 pins Haiku 4.5 for routine paper monitoring. Haiku accepts
    # temperature, but we omit it for consistency with the project's
    # other agents — determinism isn't needed and the default is fine.
    model = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        timeout=60.0,
        stop=None,
    )
    return create_agent(
        model=model,
        tools=[
            ft_status,
            ft_profit,
            ft_trades,
            ft_performance,
            compare_to_backtest,
            check_kill_switch,
        ],
        system_prompt=_PAPER_MONITOR_PROMPT,
        response_format=PaperMonitorVerdict,
    )


# ─── Invocation core (used by the 7e node + the smoke probe) ────────────


async def run_paper_monitor(ctx: PaperMonitorContext) -> PaperMonitorVerdict:
    """Set the context, invoke the agent, return the structured verdict.

    This is the agent-invocation core — NOT the LangGraph node. The 7e
    node builds ``ctx`` from the live Freqtrade API + strategy state,
    calls this, then maps the verdict via :func:`verdict_to_command`.
    The smoke probe builds ``ctx`` synthetically and calls this directly.
    """
    token = _monitor_ctx.set(ctx)
    try:
        agent = _build_paper_monitor_agent()
        kickoff = (
            f"Evaluate this paper-trading strategy. It has been running for "
            f"{ctx.elapsed_days:.1f} days (minimum {MIN_PAPER_DAYS} required "
            "before it can advance to live). Use your tools, then emit your verdict."
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=kickoff)]})
    finally:
        _monitor_ctx.reset(token)
    verdict: PaperMonitorVerdict = result["structured_response"]
    return verdict


# ─── Pure verdict → routing mapper ─────────────────────────────────────


# AgentVote.verdict is Literal["pass","fail","revise","pause","continue"].
# Map the monitor's decision onto that vocabulary for the shared
# agent_votes trail.
_VOTE_VERDICT: dict[str, str] = {
    "rearm": "continue",
    "advance": "pass",
    "kill": "fail",
}


def verdict_to_command(
    verdict: PaperMonitorVerdict,
    *,
    existing_gates: dict[str, Any] | None = None,
) -> Command[Literal["paper_wait", "live_gate", "archive"]]:
    """Translate a :class:`PaperMonitorVerdict` into a routing ``Command``.

    Pure function — used by both the real agent path (7e node) and the
    stubbed test path. Routing per BRD §5.5:

    - rearm  → ``paper_wait`` (re-arm the wake cycle)
    - advance → ``live_gate`` (HITL approval to go live)
    - kill   → ``archive`` (stage=archived + failure_reason)

    ``existing_gates`` is spread into the update so the gate_decisions
    field keeps its prior keys — LangGraph dict updates REPLACE, they do
    not deep-merge (same caveat as risk_analyst.verdict_to_command).
    """
    base_gates = existing_gates or {}
    base_update: dict[str, Any] = {
        "agent_votes": [
            {
                "agent": "paper_monitor",
                "verdict": _VOTE_VERDICT[verdict.decision],
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            },
        ],
        "gate_decisions": {
            **base_gates,
            "paper_monitor": {
                "decision": verdict.decision,
                "primary_observation": verdict.primary_observation,
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            },
        },
    }

    if verdict.decision == "rearm":
        return Command(goto="paper_wait", update=base_update)
    if verdict.decision == "advance":
        return Command(goto="live_gate", update=base_update)
    return Command(
        goto="archive",
        update={
            **base_update,
            "stage": "archived",
            "failure_reason": f"paper_monitor_kill: {verdict.primary_observation}",
        },
    )
