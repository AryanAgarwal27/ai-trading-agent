"""Researcher agent — Sonnet 4.6 ReAct (BRD §5.3).

Pipeline placement::

    load_context ──> researcher ──> generator ──> (critic, 5d) ──>
                                                  (lookahead_gate, 5e)

The researcher proposes a market-belief hypothesis, selects ONE of the
shipped strategy templates, names the regime fit, and suggests narrowed
parameter ranges for each SLOT. Concrete parameter values are produced
by the deterministic generator (BRD §5.3 says "generator: plain") via a
``with_structured_output(schema_cls)`` call — the researcher is not on
that path, so the agent's free-form output can't ever land in a strategy
file.

Implementation contract (mirrors :mod:`orchestrator.agents.risk_analyst`
patterns established in Stage 4e):

  1. Tools (``query_store``, ``get_market_regime``, ``read_template``,
     ``get_pair_stats``) are real ``@tool``-decorated functions the agent
     invokes via ``create_agent``. State-dependent tools (``query_store``,
     ``get_market_regime``) read from ContextVars the node sets before
     invoking the agent, avoiding ``InjectedState`` boilerplate.

  2. The agent uses ``response_format=ResearchProposal`` so the final
     output is a structured Pydantic instance — no regex parsing.

  3. The node function is injectable via ``build_research_subgraph``;
     unit tests pass a stubbed callable so the real Sonnet invocation
     is the operator's manual smoke check (scripts/smoke_researcher.py),
     not part of the CI gate.

  4. Lazy agent construction: ``_build_researcher_agent`` is only called
     inside the node function, so importing this module on a machine
     without ``ANTHROPIC_API_KEY`` does not fail.

The prompt is opinionated (BRD §17 #10) — the researcher must justify a
template choice against the current regime AND past failures, not just
emit a plausible-looking suggestion.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field

from orchestrator.tools.store_queries import aget_failures, aget_wins

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"

# The three v1 templates (BRD §8.1). Literal type narrows the agent's
# response_format and forces it to pick one of these — there is no
# "freeform name" escape.
TemplateName = Literal[
    "mean_reversion_template",
    "freqai_classifier_template",
    "freqai_regressor_template",
]


# ─── Context-local tool inputs ─────────────────────────────────────────
# Set by ``researcher_node`` before invoking the agent; read by the
# state-dependent tools (``query_store``, ``get_market_regime``). The
# ContextVar pattern (per-asyncio-task) prevents concurrent research runs
# from clobbering each other once the supervisor (Stage 9) starts
# spawning parallel threads.

_current_store: ContextVar[BaseStore | None] = ContextVar(
    "researcher.store", default=None
)
_current_regime: ContextVar[str] = ContextVar(
    "researcher.regime", default="unknown"
)
_current_pairs: ContextVar[tuple[str, ...]] = ContextVar(
    "researcher.pairs", default=()
)


# ─── Tools ─────────────────────────────────────────────────────────────


@tool
async def query_store(
    category: Literal["failures", "wins"],
    regime: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return past ``failures`` or ``wins`` from the long-term Store (BRD §5.9).

    Args:
        category: ``"failures"`` for strategies that lost money / failed
            a gate, ``"wins"`` for strategies that completed a live cycle
            profitably.
        regime: Regime label to scope the lookup (e.g. ``"low_vol_up"``).
            Defaults to the current market regime (see
            ``get_market_regime``).
        limit: Max records to return. Default 10.

    Returns:
        A list of dicts, each carrying the archived strategy's hypothesis,
        params, failure_reason (failures only), and live_metrics_summary
        (wins only). An empty list means nothing has been recorded for
        this regime yet — common on a fresh install; do not treat as an
        error.
    """
    store = _current_store.get()
    if store is None:
        return []
    target_regime = regime or _current_regime.get()
    if category == "failures":
        return await aget_failures(store, target_regime, limit=limit)
    return await aget_wins(store, target_regime, limit=limit)


@tool
def get_market_regime() -> str:
    """Return the current market regime label (BRD §5.7).

    Composite ``{vol}_{trend}`` label from
    :mod:`orchestrator.tools.regime` — values are ``low_vol``/``mid_vol``/
    ``high_vol`` × ``up``/``flat``/``down`` (nine combinations) plus the
    sentinel ``"unknown"`` when no recent classification is available.

    The regime is fixed for the duration of this research call; use it
    to anchor the hypothesis to the regime that will actually be live
    when the strategy starts paper-trading.
    """
    return _current_regime.get()


@tool
def read_template(template_name: TemplateName) -> str:
    """Return the source code of one of the v1 strategy templates.

    Args:
        template_name: One of ``mean_reversion_template``,
            ``freqai_classifier_template``, ``freqai_regressor_template``.

    Returns:
        The full ``.py`` source. The SLOT comments
        (``# SLOT: <name> (type, range)``) are the parameters you may
        propose narrowed ranges for; everything outside SLOT lines is
        structural shell and cannot be tuned.
    """
    path = TEMPLATES_DIR / f"{template_name}.py"
    if not path.exists():
        return f"ERROR: template {template_name!r} not found at {path}"
    return path.read_text(encoding="utf-8")


@tool
def get_pair_stats(pair: str) -> dict[str, Any]:
    """Return recent realized-vol and mean daily volume for ``pair``.

    Args:
        pair: Exchange pair symbol (e.g. ``"BTC/USDT"``).

    Returns:
        A dict with ``annualized_vol`` (float, fraction — 0.5 = 50%),
        ``mean_daily_volume_quote`` (float, in quote currency units),
        ``closes_used`` (int), and ``timeframe`` (str). Returns
        ``{"error": "<reason>"}`` if the cached feather is missing or
        empty — do NOT treat that as a strategy-blocker; pick a pair
        with data instead.
    """
    # Local imports keep this tool importable on machines without the
    # feather-reading deps (mainly pytest collection on dev laptops).
    import math
    import statistics

    from orchestrator.tools.backtest_runner import SHARED_DATA_DIR

    timeframe = "5m"
    feather_path = (
        SHARED_DATA_DIR / "binance" / f"{pair.replace('/', '_')}-{timeframe}.feather"
    )
    if not feather_path.exists():
        return {"error": f"no_cached_feather_for_{pair}_{timeframe}"}

    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        return {"error": f"pyarrow_unavailable: {exc}"}

    table = feather.read_table(feather_path)  # type: ignore[no-untyped-call]
    df = table.to_pandas()
    if "close" not in df.columns or "volume" not in df.columns:
        return {"error": "feather_missing_close_or_volume_columns"}

    # 30 days of 5m candles = 8640 rows; clip to whatever is available.
    window = min(len(df), 30 * 24 * 12)
    closes = df["close"].tail(window).tolist()
    volumes = df["volume"].tail(window).tolist()
    if len(closes) < 30:
        return {"error": f"insufficient_closes: {len(closes)}"}

    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    per_candle_std = statistics.stdev(log_returns) if len(log_returns) >= 2 else 0.0
    candles_per_year = 365 * 24 * 12  # 5m candles
    annualized_vol = per_candle_std * math.sqrt(candles_per_year)

    # Mean daily quote volume = mean(volume * close) over the window * candles/day.
    candles_per_day = 24 * 12
    quote_per_candle = [v * c for v, c in zip(volumes, closes, strict=True)]
    mean_quote_per_candle = (
        statistics.fmean(quote_per_candle) if quote_per_candle else 0.0
    )

    return {
        "pair": pair,
        "timeframe": timeframe,
        "annualized_vol": annualized_vol,
        "mean_daily_volume_quote": mean_quote_per_candle * candles_per_day,
        "closes_used": len(closes),
    }


# ─── Structured proposal ───────────────────────────────────────────────


class ResearchProposal(BaseModel):
    """Structured output the researcher emits via ``response_format``.

    The generator (BRD §5.3, deterministic node) consumes this and runs
    a Pydantic-schema-enforced parameter extraction — concrete values
    live in the generator's structured-output call, not here. The
    ``suggested_param_ranges`` field is guidance to the generator, not
    a hard constraint (the schema's ``Field(ge=, le=)`` is the hard
    constraint).
    """

    # extra="forbid" guards against silent field drift if the prompt is
    # ever changed in a way that asks the agent for additional fields.
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(
        description=(
            "One-paragraph statement of the market belief this strategy "
            "encodes. Should name the regime, the pair characteristics, "
            "and the specific edge — not a generic 'mean reversion works'."
        ),
    )
    template_name: TemplateName = Field(
        description=(
            "The chosen template — exactly one of the three v1 templates. "
            "Justify the choice against the regime in `regime_thesis`."
        ),
    )
    regime_thesis: str = Field(
        description=(
            "2–4 sentences on why this template fits the CURRENT regime "
            "(from get_market_regime) better than the alternatives. Cite "
            "any past failures or wins that informed the choice."
        ),
    )
    suggested_param_ranges: dict[str, str] = Field(
        description=(
            "For each SLOT in the chosen template, a NARROWED range "
            "(e.g. {'rsi_period': '10-18', 'bb_std': '2.0-2.4'}). The "
            "generator extracts a concrete value within this range AND "
            "the template's Pydantic schema. Use string ranges so the "
            "agent doesn't have to commit to int-vs-float for every slot."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the proposal, 0.0 to 1.0.",
    )


# ─── Prompt ────────────────────────────────────────────────────────────


_RESEARCHER_PROMPT = """\
You are the Researcher — the first step in the per-strategy lifecycle.
Your job is to propose ONE concrete strategy hypothesis grounded in the
current market regime and the project's history of past failures and wins.

Read these rules and follow them strictly:

1. Call get_market_regime FIRST. The regime label is the anchor for
   every subsequent decision. If it returns "unknown", proceed as if it
   were "mid_vol_flat" but lower your confidence to reflect the
   uncertainty.

2. Call query_store("failures") for the current regime. If non-empty,
   you MUST cite at least one past failure in your regime_thesis and
   explain how this proposal differs. Repeating a known failure is the
   single most expensive mistake you can make — every failure record
   represents a real $500 capital paper-trade slot that did not produce.

3. Call query_store("wins") for the current regime. If non-empty, prefer
   building on a working pattern; if empty, draw from the v1 baseline
   intuition (mean reversion as the cheap baseline, FreqAI templates
   when there is evidence of learnable structure).

4. Call read_template on at most TWO of the three v1 templates to see
   their SLOT structure. Do not read all three by reflex; the third call
   is wasted tokens if the first two clearly determine the choice.

5. Call get_pair_stats on at least one pair from the configured pair
   list. Use the realized vol to inform threshold-sensitive slot ranges
   (RSI thresholds, ATR multipliers).

6. Emit a ResearchProposal with: hypothesis, template_name (one of
   mean_reversion_template / freqai_classifier_template /
   freqai_regressor_template), regime_thesis (citing failures/wins),
   suggested_param_ranges (one entry per SLOT in the chosen template),
   confidence (0.0–1.0).

7. Be opinionated. A vague hypothesis like "the market mean-reverts" is
   a failure mode — name the regime, the pair characteristics, and the
   specific edge you expect to capture.

Do not propose multiple strategies. Do not propose a template not in
the v1 list. Do not skip the failures lookup.
"""


# ─── Real agent ─────────────────────────────────────────────────────────


def _build_researcher_agent() -> Any:
    """Construct the real ``create_agent``-backed researcher.

    Lazy import + lazy construction so this module can be imported on a
    machine without ``ANTHROPIC_API_KEY`` (e.g. pytest collection on a
    laptop that's about to skip the integration test).
    """
    # Imports kept local so a missing langchain_anthropic install or a
    # missing API key only fails when the real agent is actually used.
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    # BRD §4 pins Sonnet 4.6 for researcher. Sonnet accepts ``temperature``
    # (unlike Opus 4.7 which rejects it; see risk_analyst.py:188-200), but
    # we omit it here intentionally — keeping every ChatAnthropic
    # construction in the codebase uniform avoids the same regression
    # entry point that hit risk_analyst (someone reading the Anthropic
    # docs adds temperature on Opus and gets a 400 only at runtime).
    #
    # ``stop=None`` is the defensive default — explicit None tells the
    # SDK "no stop sequences", preventing accidental sequence-truncation
    # in tool-calling agents where some intermediate token could otherwise
    # match a global stop sequence and cut the agent loop short mid-tool.
    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        timeout=60.0,
        stop=None,
    )
    return create_agent(
        model=model,
        tools=[query_store, get_market_regime, read_template, get_pair_stats],
        system_prompt=_RESEARCHER_PROMPT,
        response_format=ResearchProposal,
    )


# ─── Node function (used by the research subgraph) ─────────────────────


async def researcher_node(
    state: dict[str, Any],
    *,
    store: BaseStore | None = None,
) -> dict[str, Any]:
    """Run the researcher agent and return a state update with the proposal.

    Per BRD §5.3: the researcher proposes a hypothesis, candidate
    template, and parameter ranges. This node populates:

      - ``hypothesis`` (str) — for downstream nodes + dashboard.
      - ``template`` (str) — the chosen template name, consumed by
        ``generator_node``.
      - ``artifacts["research_proposal"]`` — full ``ResearchProposal``
        dict, kept for the critic loop (5d) and the audit trail.
      - ``agent_votes`` — appends one ``{"agent": "researcher", "verdict":
        "continue", ...}`` vote so the per-strategy graph's audit log
        shows the researcher's confidence.

    Parameters
    ----------
    state
        The current ``StrategyState`` (BRD §5.7). Required fields:
        ``strategy_id``, ``pairs``. Optional: ``current_regime``
        (defaults to ``"unknown"`` if not set).
    store
        The long-term Store handle. Required for the ``query_store`` tool
        to return anything useful; if ``None`` the tool returns an empty
        list (acceptable on a fresh install with no archived strategies).
    """
    regime = state.get("current_regime") or "unknown"
    pairs = tuple(state.get("pairs") or ())

    token_store = _current_store.set(store)
    token_regime = _current_regime.set(regime)
    token_pairs = _current_pairs.set(pairs)
    try:
        agent = _build_researcher_agent()
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Propose one strategy for the current regime. "
                            "Follow the protocol in the system prompt."
                        )
                    )
                ]
            }
        )
    finally:
        _current_store.reset(token_store)
        _current_regime.reset(token_regime)
        _current_pairs.reset(token_pairs)

    proposal: ResearchProposal = result["structured_response"]
    return proposal_to_state_update(proposal, existing_artifacts=state.get("artifacts") or {})


def proposal_to_state_update(
    proposal: ResearchProposal,
    *,
    existing_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a ResearchProposal into a state-update dict.

    Pure function — used by both the real agent path and the stubbed
    test path. Unit tests verify the mapping without needing the agent.
    """
    base_artifacts = existing_artifacts or {}
    return {
        "hypothesis": proposal.hypothesis,
        "template": proposal.template_name,
        "artifacts": {
            **base_artifacts,
            "research_proposal": proposal.model_dump(),
        },
        "agent_votes": [
            {
                "agent": "researcher",
                "verdict": "continue",
                "rationale": proposal.regime_thesis,
                "confidence": proposal.confidence,
            },
        ],
    }
