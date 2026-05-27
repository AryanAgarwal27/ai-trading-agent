"""Critic agent — Opus 4.7 adversarial reviewer (BRD §5.3).

Pipeline placement::

    generator ──> critic ──> revise_or_proceed ──┬── (revise, count < 3) ──> generator (back-edge)
                                                 ├── (revise, count ≥ 3) ──> archive
                                                 └── (pass) ──> lookahead_gate (Stage 5e)

The critic is the FIRST adversarial gate on a generated strategy.
Per BRD §5.3: "adversarial review: 'find the look-ahead bias', 'find the
indicator reading future data', 'find the position sizing compounding
losses'." It runs on Opus 4.7 because the marginal cost is justified by
the fact that this is the LAST chance to reject a bad strategy before
it consumes:

  - the entire validation subgraph (10s of backtests = real wall-clock),
  - a 30-day human-monitored paper-trade slot,
  - and eventually $500 of operator capital.

Two additional charges per Stage 5c finding (SPEC §6 change log):

  1. **Default-hugging detection.** The generator's Sonnet extractor
     partially default-hugs — three of eight params encode the
     hypothesis, five hug textbook/midpoint values. The critic is the
     enforcement point for "params actually encode the thesis". If the
     concrete parameter set looks like a generic textbook strategy
     rather than something the stated hypothesis demands, vote revise
     with concrete guidance on which slot is the offender and which
     edge of the range the hypothesis pulls toward.

  2. **Researcher template-choice variance.** The Sonnet researcher's
     ReAct tool-call ordering is non-deterministic; the same regime
     can yield different template selections. The critic must NOT
     re-litigate the template choice (the validation subgraph gates on
     strategy quality, not researcher reproducibility); only critique
     the chosen template's parameter set + structural integrity.

Implementation contract (mirrors :mod:`orchestrator.agents.risk_analyst`
patterns established in Stage 4e and :mod:`orchestrator.agents.researcher`
in Stage 5c):

  1. Tools (``read_generated_strategy``, ``read_template``) are real
     ``@tool``-decorated functions. ``read_generated_strategy`` reads
     from a ContextVar populated by the node before agent invocation;
     ``read_template`` reads directly from disk (no per-invocation
     state).
  2. ``response_format=CriticVerdict`` — structured Pydantic, no regex.
  3. Lazy agent construction so importing this module doesn't require
     ``ANTHROPIC_API_KEY``.
  4. Node function is injectable via ``build_research_subgraph``;
     unit/loop tests pass a stubbed critic so CI doesn't burn Opus
     tokens.

The prompt is deliberately opinionated (BRD §17 #10) — a friendly
critic rubber-stamps and the entire loop becomes performative theatre.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"

# The three v1 templates (same as researcher's TemplateName) — re-imported
# here as a Literal because critic.py shouldn't depend on researcher.py
# (would create a cycle once researcher imports anything from critic).
TemplateName = Literal[
    "mean_reversion_template",
    "freqai_classifier_template",
    "freqai_regressor_template",
]


# Hard ceiling on revisions (BRD §5.3). After this many revisions, the
# strategy archives with failure_reason="critic_loop_exhausted".
# Exposed as a module constant so tests can reference it without
# magic-numbering and the bounded-loop assertion stays in sync.
MAX_REVISIONS = 3


# ─── Context-local tool inputs ─────────────────────────────────────────

_current_generated_source: ContextVar[str] = ContextVar(
    "critic.generated_source", default=""
)


# ─── Tools ─────────────────────────────────────────────────────────────


@tool
def read_generated_strategy() -> str:
    """Return the source of the strategy file the generator just rendered.

    This is the CONCRETE strategy with substituted parameter values —
    not the un-rendered template. Use this for parameter-sensibility
    critiques (does the chosen `bb_std` actually encode the
    hypothesis's "stretched BB"?) and for full-file structural reviews
    (look-ahead bias, suspicious shifts in feature_engineering_*).

    Returns the raw ``.py`` source as a string.
    """
    return _current_generated_source.get()


@tool
def read_template(template_name: TemplateName) -> str:
    """Return the source code of the un-rendered template.

    Args:
        template_name: One of ``mean_reversion_template``,
            ``freqai_classifier_template``, ``freqai_regressor_template``.

    Returns:
        The full template ``.py`` source with default SLOT values.
        Useful for diffing against ``read_generated_strategy`` to see
        exactly which literals the generator changed — a parameter set
        that differs from the template only at one or two slots while
        leaving the rest at textbook defaults is a default-hugging
        signal.
    """
    path = TEMPLATES_DIR / f"{template_name}.py"
    if not path.exists():
        return f"ERROR: template {template_name!r} not found at {path}"
    return path.read_text(encoding="utf-8")


# ─── Structured verdict ────────────────────────────────────────────────


class CriticVerdict(BaseModel):
    """Structured output the critic emits via ``response_format``.

    Two-way verdict (pass / revise). On revise, the generator re-runs
    with the critic's ``revision_guidance`` prepended to the
    parameter-extractor's user message (see
    :func:`orchestrator.agents.generator._default_params_extractor`).
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["pass", "revise"] = Field(
        description=(
            "pass: the strategy advances to lookahead_gate (Stage 5e) "
            "or, for 5d, the subgraph END. "
            "revise: send the strategy back to the generator with "
            "`revision_guidance` for the parameter extractor to address. "
            "Bias toward revise when in doubt — the validation subgraph "
            "and 30-day paper trade are far more expensive than another "
            "generator pass."
        ),
    )
    primary_concern: str = Field(
        description=(
            "One sentence describing the single most important issue with "
            "this strategy. For pass: name the strongest signal that this "
            "strategy is grounded. For revise: name the specific defect "
            "the generator must address."
        ),
    )
    rationale: str = Field(
        description=(
            "2–4 sentences elaborating, citing SPECIFIC code lines or "
            "parameter values from read_generated_strategy. Generic "
            "rationale like 'looks reasonable' or 'needs improvement' "
            "is a failure mode — quote the offending line, name the "
            "specific slot value, point at the indicator chain."
        ),
    )
    revision_guidance: str = Field(
        description=(
            "Concrete, slot-specific feedback the generator's extractor "
            "will see on the next pass. For verdict=revise: name each "
            "slot that needs changing AND which edge of its range the "
            "hypothesis demands (e.g. 'bb_std should be ≥ 2.6 to encode "
            "\\\"stretched\\\"; current value 2.2 is generic'). For "
            "verdict=pass: may be empty string. Do NOT re-propose the "
            "template choice — that decision is upstream and final."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the verdict, 0.0 to 1.0.",
    )


# ─── Prompt ────────────────────────────────────────────────────────────


_CRITIC_PROMPT = """\
You are the Critic — the LAST adversarial gate before this strategy
consumes the validation subgraph (10s of backtests), and downstream of
that a 30-day paper-trade slot. Approve only what survives scrutiny.

Read these rules and follow them strictly:

1. Call read_generated_strategy to see the CONCRETE rendered strategy
   with substituted parameter values. This is what will actually run.

2. Optionally call read_template on the same template_name to see the
   un-rendered original. Diff the two mentally — which slots actually
   changed? Quantitative test for default-hugging: if 5+ of N slots
   are within 10% of the template default OR within 10% of the schema
   midpoint, that's the 5c default-hugging signal — vote REVISE with
   slot-specific guidance naming which params are textbook and what
   the hypothesis would imply instead. (Per SPEC §6 5c finding.)

3. Adversarial attacks you MUST consider (the first three are named in
   BRD §5.3 verbatim; the last two are inferred from the 5c finding
   and the schema-cant-span-fields gap):

   - **Look-ahead bias.** Does any indicator computation use future
     data? Check for forward shifts (`shift(-N)`) outside of
     set_freqai_targets — only the label may use them. Check for
     rolling windows without min_periods= which produce partial-window
     leakage at the start.
   - **Indicator reading future data.** Are entry/exit columns derived
     from columns that themselves use future data? Trace the data flow.
   - **Position-sizing compounding losses.** Does the stoploss interact
     badly with the typical move size? A stoploss tighter than 1× ATR
     or 1× typical BB-band-width whipsaws out of normal noise.
   - **Cross-field inconsistencies.** For templates with paired slots
     (ema_fast/ema_slow), is fast actually less than slow? Pydantic's
     field-level constraints can't span fields — this is on you.
   - **Hypothesis–parameter mismatch.** Read the hypothesis from the
     generator's proposal (passed in the human message). Does each
     concrete parameter value actually encode that hypothesis? If the
     hypothesis says "aggressive oversold capture" and
     rsi_buy_threshold=30, that's textbook, not aggressive — vote
     REVISE.

4. The critic is the enforcement point for "params encode thesis"
   (see SPEC §6 5c finding). The generator's Sonnet extractor is known
   to partially default-hug; you must catch it. A strategy whose params
   don't encode its own hypothesis is not a passable strategy.

5. Do NOT re-litigate the template choice. The researcher selected the
   template upstream; you are critiquing the rendered output, not the
   architecture decision. Saying "should have been freqai_classifier
   instead" is out of scope — vote on the strategy as rendered.
   If after critiquing the rendered params you believe the template is
   fundamentally wrong for the hypothesis (not just suboptimal — wrong),
   vote REVISE with revision_guidance="hypothesis-template mismatch:
   <one-sentence-why>". The generator can't switch templates, so this
   revision will fail; the audit trail captures the issue for the
   operator's review at archive.

6. Emit a CriticVerdict with: verdict (pass/revise), primary_concern
   (one sentence), rationale (2–4 sentences citing specific code lines
   or parameter values), revision_guidance (slot-specific feedback for
   the generator on revise; may be empty on pass), confidence (0.0–1.0).

The operator's capital cap is $500 (SPEC §1 Q3). Paper-trade slots are
finite and revision passes are cheap by comparison. Bias toward REVISE
when in doubt; bias toward PASS only when the strategy is decisively
grounded in its stated hypothesis.
"""


# ─── Real agent ─────────────────────────────────────────────────────────


def _build_critic_agent() -> Any:
    """Construct the real ``create_agent``-backed critic.

    Lazy import + lazy construction so this module can be imported on a
    machine without ``ANTHROPIC_API_KEY`` (e.g. pytest collection on a
    laptop that's about to skip the integration test).
    """
    # Imports kept local — see risk_analyst.py:_build_risk_analyst_agent
    # for the same reasoning.
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic

    # BRD §4 pins Opus 4.7 for critic (adversarial review of marginal
    # strategies — the same reasoning depth that risk_analyst applies
    # to marginal robustness profiles).
    # NOTE: temperature/top_p/top_k are intentionally omitted. Claude
    # Opus 4.7 rejects non-default values for these sampling parameters
    # with a 400 error per the official migration guide; see
    # risk_analyst.py:188-200 for the full citation. Sonnet 4.6 and
    # Haiku 4.5 still accept temperature, so this caveat is Opus-only.
    #
    # ``stop=None`` is the defensive default — explicit None tells the
    # SDK "no stop sequences", preventing accidental sequence-truncation
    # in tool-calling agents where some intermediate token could
    # otherwise match a global stop sequence and cut the agent loop
    # short mid-tool.
    model = ChatAnthropic(
        model="claude-opus-4-7",
        timeout=60.0,
        stop=None,
    )
    return create_agent(
        model=model,
        tools=[read_generated_strategy, read_template],
        system_prompt=_CRITIC_PROMPT,
        response_format=CriticVerdict,
    )


# ─── Node function (used by the research subgraph) ─────────────────────


async def critic_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run the critic agent against the rendered strategy.

    Returns a state update that:
      - Appends a ``"critic"`` entry to ``agent_votes`` (BRD §5.7
        reducer).
      - Appends the critic's ``revision_guidance`` to ``critic_notes``
        (BRD §5.7 reducer) so the next generator pass sees it AND so
        the operator's audit log shows the full sequence of critiques
        across all revisions.
      - Writes the structured verdict under
        ``artifacts["critic_verdicts"]`` as a list (one entry per
        critic call across the revision loop).

    The routing decision (revise vs pass, with bounded count) is made
    by :func:`revise_or_proceed` downstream — keeps the critic node
    side-effect-free w.r.t. graph edges, which mirrors the
    risk_analyst → paper_gate split.

    Reads the rendered strategy from
    ``state["artifacts"]["generated_strategy_path"]`` and sets the
    ContextVar before invoking the agent.
    """
    strategy_path_str = (state.get("artifacts") or {}).get("generated_strategy_path")
    if not strategy_path_str:
        raise ValueError(
            "critic_node requires state['artifacts']['generated_strategy_path']"
        )
    strategy_path = Path(strategy_path_str)
    if not strategy_path.exists():
        raise FileNotFoundError(
            f"critic_node: generated strategy file missing: {strategy_path}"
        )
    rendered_source = strategy_path.read_text(encoding="utf-8")

    proposal = (state.get("artifacts") or {}).get("research_proposal") or {}
    hypothesis = proposal.get("hypothesis", "<no hypothesis>")
    regime_thesis = proposal.get("regime_thesis", "<no regime thesis>")
    template_name = state.get("template", "<unknown>")
    revision_count = int(state.get("revision_count") or 0)

    token = _current_generated_source.set(rendered_source)
    try:
        agent = _build_critic_agent()
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Strategy under review (revision {revision_count} of "
                            f"{MAX_REVISIONS} allowed):\n\n"
                            f"Template: {template_name}\n"
                            f"Hypothesis: {hypothesis}\n"
                            f"Regime thesis: {regime_thesis}\n\n"
                            f"Call read_generated_strategy to see the rendered "
                            f"source, then emit your verdict per the protocol "
                            f"in the system prompt."
                        )
                    )
                ]
            }
        )
    finally:
        _current_generated_source.reset(token)

    verdict: CriticVerdict = result["structured_response"]
    return verdict_to_state_update(
        verdict,
        existing_artifacts=state.get("artifacts") or {},
    )


def verdict_to_state_update(
    verdict: CriticVerdict,
    *,
    existing_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a CriticVerdict into a state-update dict.

    Pure function — used by both the real agent path and stubbed test
    paths. Does NOT include routing (Command) — the downstream
    revise_or_proceed node owns that.
    """
    base_artifacts = existing_artifacts or {}
    prior_verdicts = list(base_artifacts.get("critic_verdicts") or [])
    prior_verdicts.append(verdict.model_dump())

    return {
        "agent_votes": [
            {
                "agent": "critic",
                # BRD §5.7 AgentVote literal includes "revise"; map pass
                # to "pass" verbatim, revise to "revise". confidence
                # carries through.
                "verdict": verdict.verdict,
                "rationale": verdict.rationale,
                "confidence": verdict.confidence,
            },
        ],
        "critic_notes": (
            [verdict.revision_guidance] if verdict.revision_guidance else []
        ),
        "artifacts": {
            **base_artifacts,
            "critic_verdicts": prior_verdicts,
        },
    }


# ─── revise_or_proceed router ──────────────────────────────────────────


def revise_or_proceed(
    state: dict[str, Any],
) -> Command[Literal["generator", "archive", "lookahead_gate"]]:
    """Route based on the latest critic verdict and the revision count.

    BRD §5.3:
      - vote="revise" AND revision_count < MAX_REVISIONS → goto generator
        (revision_count is incremented in the returned Command's update)
      - vote="revise" AND revision_count >= MAX_REVISIONS → goto archive
        with failure_reason="critic_loop_exhausted: <last guidance>"
      - vote="pass" → goto lookahead_gate (Stage 5e)

    All three targets are explicit ``Command(goto=...)`` returns rather
    than a mix of Command + falls-through-to-add_edge. This is a
    deliberate post-5d correction: when a node returns Command without
    a goto AND the builder has an add_edge to a downstream node, Pregel
    schedules both nodes in the SAME superstep, which conflicts on
    state-channel writes that don't have an ``add`` reducer (the
    ``artifacts`` field is one such — see the InvalidUpdateError that
    surfaced when 5e's lookahead_gate was wired downstream of a
    Command(no-goto) pass route).

    Reads:
      - ``state["agent_votes"]`` — finds the latest ``"critic"`` vote.
      - ``state["revision_count"]`` — current count (default 0).

    Writes (on revise back-edge):
      - ``revision_count`` incremented by 1.
    Writes (on archive):
      - ``stage="archived"``.
      - ``failure_reason="critic_loop_exhausted: <last_guidance>"``.
    """
    votes = state.get("agent_votes") or []
    critic_vote = next(
        (v for v in reversed(votes) if v.get("agent") == "critic"),
        None,
    )
    if critic_vote is None:
        raise RuntimeError(
            "revise_or_proceed needs at least one 'critic' entry in "
            "agent_votes; got none"
        )

    revision_count = int(state.get("revision_count") or 0)
    verdict = critic_vote.get("verdict")

    if verdict == "pass":
        # Explicit goto to lookahead_gate (BRD §5.3). Pregel schedules
        # the gate in the next superstep, not the same one — avoids
        # the artifacts-channel double-write that no-goto + add_edge
        # produced.
        return Command(goto="lookahead_gate")

    if verdict != "revise":
        raise RuntimeError(
            f"revise_or_proceed: unexpected critic verdict {verdict!r} "
            f"(expected 'pass' or 'revise')"
        )

    # verdict == "revise" — bounded loop.
    if revision_count >= MAX_REVISIONS:
        # Pull the last critic guidance from artifacts for the
        # failure_reason — operator sees exactly what the final critic
        # asked for that the generator couldn't deliver.
        last_guidance = ""
        verdicts = (state.get("artifacts") or {}).get("critic_verdicts") or []
        if verdicts:
            last_guidance = verdicts[-1].get("revision_guidance", "")
        reason = (
            f"critic_loop_exhausted: hit MAX_REVISIONS={MAX_REVISIONS} "
            f"with verdict=revise. Last guidance: {last_guidance or '<empty>'}"
        )
        return Command(
            goto="archive",
            update={"stage": "archived", "failure_reason": reason},
        )

    return Command(
        goto="generator",
        update={"revision_count": revision_count + 1},
    )
