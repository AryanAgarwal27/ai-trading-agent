"""Generator node — deterministic template renderer (BRD §5.3).

Pipeline placement::

    researcher ──> generator ──> (critic, 5d) ──> (lookahead_gate, 5e)

The generator is DETERMINISTIC in the BRD's sense: no ReAct loop, no
free-form output. It does call Sonnet 4.6 once via
``ChatAnthropic.with_structured_output(schema_cls)`` to extract concrete
parameter values for the chosen template — but the schema's Pydantic
``Field(ge=, le=)`` bounds are the hard constraint, so the LLM cannot
emit anything the schema rejects.

Pipeline inside this node:

  1. Look up the chosen template's schema class from the proposal.
  2. Call Sonnet 4.6 with ``response_format=schema_cls`` (i.e.
     ``model.with_structured_output(schema_cls)``) to extract concrete
     SLOT values within the researcher's suggested ranges.
  3. Render the template via line-oriented SLOT substitution — the same
     pattern :mod:`tests.unit.test_template_filling` smoke-tests in 5b.
  4. Write the rendered source to
     ``strategy_templates/_generated/<strategy_id>.py``.
  5. Run the Stage 5a AST validator on the written file. On failure:
     return ``{"stage": "archived", "failure_reason": "ast_validator: ..."}``
     without raising — the strategy archives and the graph continues.

Why split the LLM call out of the researcher: BRD §5.3 says the
generator is "plain (deterministic)" — meaning the generator's behavior
is reproducible from a fixed proposal. The Sonnet call here is a
single-shot structured-output extraction, not a tool-loop; the same
proposal will produce parameter sets within the same hard ranges every
time. This separation also lets unit tests inject a stub extractor
without mocking the entire researcher.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from orchestrator.security.ast_validator import (
    ASTValidationError,
    validate_strategy_source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"
GENERATED_DIR = TEMPLATES_DIR / "_generated"


# Schema class name suffix convention (BRD §8 rule 3 + 5b implementation):
# template ``<stem>.py`` pairs with ``<stem>_schema.py`` exporting
# ``<CamelCase>Params``.
_SCHEMA_CLASS_NAMES: dict[str, str] = {
    "mean_reversion_template": "MeanReversionParams",
    "freqai_classifier_template": "FreqaiClassifierParams",
    "freqai_regressor_template": "FreqaiRegressorParams",
}


# Matches a SLOT line: `<name>: <type> = <literal>  # SLOT: <slot_name> (...)`.
# Same pattern as tests/unit/test_template_filling.py — kept in sync by
# the test_rendered_template_passes_ast_validator parameterized case,
# which renders every template via THIS regex and AST-validates the result.
SLOT_LINE_RE = re.compile(
    r"^(?P<lead>\s*)"
    r"(?P<name>\w+)\s*:\s*\w+\s*=\s*"
    r"(?P<value>[^#]+?)\s*"
    r"#\s*SLOT:\s*(?P<slot>\w+)\b"
)


# ─── Injection seam for the LLM call ───────────────────────────────────
# Tests pass a stub that returns a pre-built Pydantic instance; the real
# default builds the Sonnet 4.6 structured-output call below.
ParamsExtractor = Callable[
    [dict[str, Any], str, type[BaseModel]],
    Awaitable[BaseModel],
]


# Schema class cache. spec_from_file_location creates a fresh module on
# every call — repeated invocations produce sibling classes with the
# same definition but distinct identity (so isinstance() across calls
# fails). Caching also avoids re-running module init on every generation
# in production. Keyed by template_name; entries are immortal for the
# process lifetime since schemas are part of the deployed artifact.
_SCHEMA_CACHE: dict[str, type[BaseModel]] = {}


def load_schema(template_name: str) -> type[BaseModel]:
    """Load (and cache) the Pydantic schema class for the named template.

    Uses ``importlib.util.spec_from_file_location`` instead of a package
    import so ``strategy_templates/`` doesn't need an ``__init__.py``
    (and so the template ``.py`` files — which import ``freqtrade`` —
    don't get imported as a side effect of importing the schema).
    The first call per ``template_name`` loads the module; subsequent
    calls return the cached class so identity is stable across
    callers (critical for ``isinstance`` checks in callers/tests).
    """
    cached = _SCHEMA_CACHE.get(template_name)
    if cached is not None:
        return cached

    class_name = _SCHEMA_CLASS_NAMES.get(template_name)
    if class_name is None:
        raise KeyError(
            f"Unknown template {template_name!r}; expected one of "
            f"{sorted(_SCHEMA_CLASS_NAMES.keys())}"
        )
    schema_path = TEMPLATES_DIR / f"{template_name}_schema.py"
    spec = importlib.util.spec_from_file_location(
        f"_generator_schema_{template_name}", schema_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build module spec for {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    _SCHEMA_CACHE[template_name] = cls
    return cls


def render_template(template_source: str, params: dict[str, Any]) -> str:
    """Render a template by replacing each ``# SLOT: <name>`` RHS literal.

    Same algorithm as :func:`tests.unit.test_template_filling._render` —
    line-oriented regex match, ``repr(value)`` for the substituted
    literal, comment kept intact. Lines without a SLOT comment pass
    through unchanged.
    """
    out: list[str] = []
    for line in template_source.splitlines(keepends=True):
        m = SLOT_LINE_RE.match(line)
        if m and m.group("slot") in params:
            slot = m.group("slot")
            new_value = repr(params[slot])
            head, _ = line.split("=", 1)
            comment_at = line.index("#")
            tail = line[comment_at:]
            line = f"{head}= {new_value}  {tail}"
        out.append(line)
    return "".join(out)


async def _default_params_extractor(
    proposal: dict[str, Any],
    template_source: str,
    schema_cls: type[BaseModel],
) -> BaseModel:
    """Real Sonnet 4.6 structured-output call.

    Lazy import + lazy construction so this module is importable without
    ``ANTHROPIC_API_KEY`` (test path injects a stub extractor and never
    reaches this function).
    """
    # Imports kept local — see researcher.py:_build_researcher_agent
    # docstring for the same reasoning.
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    # BRD §4 pins Sonnet 4.6 for the generator's parameter extraction.
    # ``temperature``/``top_p``/``top_k`` omitted intentionally — Sonnet
    # accepts them but we keep every ChatAnthropic construction uniform
    # to avoid the same regression entry point that hit risk_analyst
    # (see risk_analyst.py:188-200).
    #
    # ``stop=None`` is the defensive default — explicit None tells the
    # SDK "no stop sequences", preventing accidental sequence-truncation
    # mid-emit (especially important when the schema includes free-form
    # rationale fields that could contain a token matching a stale
    # global stop sequence).
    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        timeout=60.0,
        stop=None,
    )
    structured = model.with_structured_output(schema_cls)

    # The system prompt is deliberately directive about avoiding midpoint
    # default-hugging. Without this framing Sonnet returns parameter sets
    # clustered around textbook RSI/BB values regardless of the hypothesis,
    # which collapses the researcher → generator chain into "render the
    # template's own defaults with tiny perturbations". The 5d critic loop
    # is meant for substantive design errors (look-ahead bias, leverage
    # compounding), not calibration — letting default-hugging through to
    # the critic burns Opus tokens on a Sonnet-fixable problem.
    system = (
        "You are the parameter extractor for an LLM-orchestrated trading "
        "agent. The researcher proposed a strategy hypothesis and a set of "
        "narrowed parameter ranges. Your job: emit ONE concrete parameter "
        "set that encodes the hypothesis.\n\n"
        "RULES:\n\n"
        "1. The hypothesis is load-bearing. Each parameter you pick must "
        "be defensible as 'I picked this value because the hypothesis "
        "says X.'\n\n"
        "2. Midpoint or textbook values are an ANTI-PATTERN. They produce "
        "a generic strategy that does not encode any particular thesis. "
        "Examples:\n"
        "   - If the hypothesis says 'mean-reversion on stretched BB in a "
        "flat regime', bb_std=2.0 is generic but bb_std=2.6+ encodes "
        "'stretched'.\n"
        "   - If the hypothesis says 'aggressive oversold capture', "
        "rsi_buy_threshold=30 is textbook but 18 encodes 'aggressive'.\n"
        "   - If the hypothesis says 'fast retrain on regime-change risk', "
        "label_period_candles=12 is generic but 6 encodes 'fast'.\n\n"
        "3. Read the hypothesis and regime_thesis carefully before each "
        "value. Pick toward the EDGE of the range that the hypothesis "
        "demands. Avoid midpoint values unless you can articulate why "
        "the midpoint is itself the position the hypothesis requires "
        "(rare — most hypotheses pull toward one edge).\n\n"
        "4. The researcher's narrowed ranges are guidance. The Pydantic "
        "schema's ge/le bounds are the hard constraint. If guidance and "
        "schema conflict, prefer the schema. If guidance is silent on a "
        "slot, pick a value with conviction toward the edge of the "
        "schema range that the hypothesis demands — do NOT default to "
        "the schema midpoint.\n\n"
        "5. If the hypothesis is genuinely too vague to ground a "
        "parameter choice, pick a value with conviction toward one edge "
        "anyway — a strategy with a bad-but-decisive parameter set is "
        "diagnostically more useful than one with a midpoint-defaulted "
        "set, because the backtest will surface where the conviction was "
        "wrong."
    )
    suggested_ranges = proposal.get("suggested_param_ranges") or {}
    # Critic feedback from prior revision passes (5d). The generator's
    # node bundles state["critic_notes"] (accumulated across the bounded
    # loop) into the proposal under "critic_feedback" before invoking
    # the extractor. Empty on the initial pass; non-empty on revisions.
    critic_feedback = proposal.get("critic_feedback") or []
    critic_section = ""
    if critic_feedback:
        joined = "\n".join(f"  - {note}" for note in critic_feedback)
        critic_section = (
            f"\n\nPRIOR CRITIC FEEDBACK (this is a revision pass — "
            f"address EVERY item below; the critic will check):\n{joined}\n"
        )
    user_msg = (
        f"Hypothesis: {proposal.get('hypothesis', '')}\n\n"
        f"Regime thesis: {proposal.get('regime_thesis', '')}\n\n"
        f"Suggested parameter ranges (from researcher): {suggested_ranges}"
        f"{critic_section}\n\n"
        f"Template source (read the # SLOT: comments for each field):\n\n"
        f"```python\n{template_source}\n```\n\n"
        f"Emit the parameter set. Remember: each value must encode the "
        f"hypothesis, not the textbook midpoint."
    )
    params_instance = await structured.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user_msg)]
    )

    # SMOKE_DEBUG=1 in env → print the extracted params for smoke-probe
    # diagnostics. Guarded so production paths (and unit tests, which
    # don't reach this function — they inject a stub extractor) don't
    # get spammed. Set via `$env:SMOKE_DEBUG="1"` (PowerShell) before
    # running scripts/smoke_researcher.py.
    if os.environ.get("SMOKE_DEBUG"):
        print("[generator._default_params_extractor] DEBUG extracted params:")
        print(f"    schema={schema_cls.__name__}")
        print(f"    params={params_instance.model_dump()}")

    return params_instance


# ─── Node function (used by the research subgraph) ─────────────────────


async def generator_node(
    state: dict[str, Any],
    *,
    params_extractor: ParamsExtractor | None = None,
    generated_dir: Path | None = None,
) -> dict[str, Any]:
    """Render the researcher's proposal into a concrete strategy file.

    Parameters
    ----------
    state
        Current ``StrategyState``. Required fields: ``strategy_id``,
        ``template``, ``artifacts["research_proposal"]`` (the dict form
        of ``ResearchProposal`` written by ``researcher_node``).
    params_extractor
        Optional override for the Sonnet structured-output call. Default
        is :func:`_default_params_extractor`. Unit tests pass a stub
        returning a pre-built ``BaseModel`` so the CI gate doesn't burn
        LLM tokens.
    generated_dir
        Optional override for the output directory. Default is
        ``strategy_templates/_generated/``; tests pass a ``tmp_path``.

    Returns
    -------
    dict
        State update. On success: writes ``params``, ``strategy_path``,
        ``artifacts["generated_strategy_path"]``, and appends a
        ``"generator"`` agent vote. On AST failure: writes
        ``stage="archived"`` + ``failure_reason`` with the violations.
    """
    extractor = params_extractor or _default_params_extractor
    out_dir = generated_dir or GENERATED_DIR

    strategy_id = state.get("strategy_id")
    if not strategy_id:
        raise ValueError("generator_node requires state['strategy_id']")
    template_name = state.get("template")
    if not template_name:
        raise ValueError("generator_node requires state['template']")
    proposal = (state.get("artifacts") or {}).get("research_proposal") or {}
    # Stage 5d: bundle accumulated critic_notes into the proposal so the
    # extractor sees them on revision passes. Empty on the initial pass.
    critic_notes = state.get("critic_notes") or []
    if critic_notes:
        proposal = {**proposal, "critic_feedback": list(critic_notes)}

    template_path = TEMPLATES_DIR / f"{template_name}.py"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_source = template_path.read_text(encoding="utf-8")

    schema_cls = load_schema(template_name)
    params_instance = await extractor(proposal, template_source, schema_cls)
    params = params_instance.model_dump()

    rendered = render_template(template_source, params)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{strategy_id}.py"
    out_path.write_text(rendered, encoding="utf-8")

    base_artifacts = state.get("artifacts") or {}

    try:
        validate_strategy_source(rendered, filename=str(out_path))
    except ASTValidationError as exc:
        return {
            "params": params,
            "stage": "archived",
            "failure_reason": f"ast_validator: {exc}",
            "artifacts": {
                **base_artifacts,
                "generated_strategy_path": str(out_path),
                "ast_violations": list(exc.violations),
            },
            "agent_votes": [
                {
                    "agent": "generator",
                    "verdict": "fail",
                    "rationale": (
                        f"AST validation rejected the rendered template: "
                        f"{len(exc.violations)} violation(s)"
                    ),
                    "confidence": 1.0,
                },
            ],
        }

    return {
        "params": params,
        "strategy_path": str(out_path),
        "artifacts": {
            **base_artifacts,
            "generated_strategy_path": str(out_path),
        },
        "agent_votes": [
            {
                "agent": "generator",
                "verdict": "pass",
                "rationale": (
                    f"Rendered {template_name} with "
                    f"{len(params)} param(s) → {out_path.name} "
                    f"({len(rendered.encode('utf-8'))} bytes); AST-clean."
                ),
                "confidence": 1.0,
            },
        ],
    }
