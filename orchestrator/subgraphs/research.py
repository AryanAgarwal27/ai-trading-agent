"""Research subgraph (BRD §5.3) — 5e (lookahead gate added).

Topology (this commit, 5e)::

    START ──> load_context ──> researcher ──> generator ──> critic
                                                  ▲           │
                                                  │           ▼
                                                  │     revise_or_proceed
                                                  │      │      │      │
                                                  │      │      │      └── pass ──> lookahead_gate
                                                  │      │      │                     │      │
                                                  │      │      │                     │ pass │ fail
                                                  │      │      │                     ▼      ▼
                                                  │      │      │                    END   archive ──> END
                                                  │      │      └── revise (count ≥ 3) ──> archive ──> END
                                                  │      │
                                                  └──────┘  revise (count < 3)
                                                            (revision_count incremented)

5e adds the ``lookahead_gate`` node downstream of the critic-pass route.
Per BRD §8 rule 5: "Every generated strategy is run through
``freqtrade lookahead-analysis`` before backtest. Failures route to
``archive`` with ``failure_reason='lookahead_bias'``." The gate is the
LAST automated check inside the research subgraph; passing here means
the strategy is ready for the validation subgraph (Stage 4, separate).

State additions versus 5d: none. The lookahead gate writes its result
to ``artifacts["lookahead_analysis"]`` (reuses the artifacts dict) and
appends one ``"lookahead_gate"`` vote to ``agent_votes``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from orchestrator.agents.critic import critic_node, revise_or_proceed
from orchestrator.agents.generator import generator_node
from orchestrator.agents.researcher import researcher_node
from orchestrator.tools.lookahead import (
    LookaheadResult,
    _default_lookahead_runner,
)
from orchestrator.tools.store_queries import aget_failures, aget_wins

CheckpointSaver = BaseCheckpointSaver[Any]

# ─── Node injection seams ──────────────────────────────────────────────
# Same pattern as build_validation_subgraph (Stage 4e): factory takes
# optional callable overrides so unit tests can inject deterministic
# stubs without touching the production code path.
ResearcherFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
GeneratorFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
CriticFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
# Lookahead runner signature: (path, *, pairs, timeframe, timerange) → result.
# Kept as a loose Any to avoid the same closure-async generic noise as
# the validation subgraph's BacktestWorkerFn — the gate node enforces
# the result shape at call time.
LookaheadRunnerFn = Any


class ResearchState(TypedDict, total=False):
    """Workspace state for the research subgraph.

    ``total=False`` so the parent graph can pass a partial dict; nodes
    populate fields as they run. Fields here are a subset of
    :class:`orchestrator.state.StrategyState`'s identity + research-stage
    columns — the subgraph never touches lifecycle-later fields.
    """

    # Identity (passed in by parent graph or test fixture)
    strategy_id: str
    pairs: list[str]
    timeframe: str

    # Regime context — populated by load_context. May be "unknown" on a
    # fresh install with no regime_log rows.
    current_regime: str

    # Researcher outputs
    hypothesis: str
    template: str

    # Generator outputs
    params: dict[str, Any]
    strategy_path: str

    # Loaded context (failures / wins for the current regime)
    artifacts: dict[str, Any]

    # Agent vote trail. ``Annotated[..., add]`` reducer (BRD §6.3) so
    # researcher, generator, and critic all append rather than the
    # latter overwriting the former — same contract as
    # :class:`orchestrator.state.StrategyState.agent_votes` and
    # ``ValidationState.agent_votes``.
    agent_votes: Annotated[list[dict[str, Any]], add]

    # Critic loop accounting (5d, BRD §5.7).
    # ``revision_count`` has no reducer — latest write wins, which is
    # the right semantics for "how many revisions have we done so far".
    # ``critic_notes`` uses the add reducer so the generator on a
    # revision pass sees the FULL history of critic guidance, not just
    # the latest call's output.
    revision_count: int
    critic_notes: Annotated[list[str], add]

    # Lifecycle (set on AST failure or critic_loop_exhausted)
    stage: str
    failure_reason: str


# ─── load_context node ────────────────────────────────────────────────


def make_load_context(
    *,
    store: BaseStore | None,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Build the ``load_context`` node bound to a Store handle.

    Per BRD §5.3 the node "pulls (failures, regime) and (wins, regime)
    from Store". The factory pattern lets the subgraph builder pass the
    same Store the FastAPI lifespan opened (BRD §6.5) without making
    every node aware of the lifespan.

    When ``store`` is ``None`` (unit tests that don't care about Store
    state), the node populates empty lists rather than failing — the
    researcher prompt explicitly tolerates an empty failures/wins
    history (fresh-install case).
    """

    async def load_context(state: ResearchState) -> dict[str, Any]:
        regime = state.get("current_regime") or "unknown"
        if store is None:
            failures: list[dict[str, Any]] = []
            wins: list[dict[str, Any]] = []
        else:
            failures = await aget_failures(store, regime)
            wins = await aget_wins(store, regime)
        base_artifacts = state.get("artifacts") or {}
        return {
            "current_regime": regime,
            "artifacts": {
                **base_artifacts,
                "loaded_context": {
                    "regime": regime,
                    "failures_count": len(failures),
                    "wins_count": len(wins),
                    # Carry the raw records so the researcher's
                    # query_store tool returns the same data even if the
                    # Store handle becomes unavailable mid-call.
                    "failures": failures,
                    "wins": wins,
                },
            },
        }

    return load_context


# ─── lookahead_gate node (5e) ─────────────────────────────────────────


# Default lookahead window for the gate when the state doesn't specify
# one. A 1-week window of cached BTC/USDT 5m data is enough to surface
# look-ahead bias in indicators; longer windows just burn wall-clock
# without changing the verdict.
DEFAULT_LOOKAHEAD_TIMERANGE = "20240501-20240508"


def make_lookahead_gate(
    *,
    lookahead_runner: LookaheadRunnerFn | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the ``lookahead_gate`` node bound to a (possibly-stub) runner.

    Per BRD §8 rule 5: "Every generated strategy is run through
    ``freqtrade lookahead-analysis`` before backtest. Failures route to
    ``archive`` with ``failure_reason='lookahead_bias'``."

    Routing is via ``Command(goto, update)`` — pass-path returns no
    goto (falls through to the builder's add_edge → END), fail-path
    routes to ``"archive"`` with ``failure_reason`` prefixed
    ``lookahead_bias:`` and the Freqtrade-emitted ``details`` carried
    in the suffix for the operator log.

    ``lookahead_runner`` defaults to the real Freqtrade-subprocess
    runner from :mod:`orchestrator.tools.lookahead`. Unit tests + the
    integration test pass a stub returning a synthetic
    :class:`LookaheadResult` so the CI gate doesn't need Docker.
    """
    from langgraph.types import Command  # local to keep module import lean

    runner = lookahead_runner or _default_lookahead_runner

    async def lookahead_gate(state: dict[str, Any]) -> Command[Any]:
        strategy_path_str = (state.get("artifacts") or {}).get(
            "generated_strategy_path"
        )
        if not strategy_path_str:
            raise ValueError(
                "lookahead_gate requires state['artifacts']['generated_strategy_path']"
            )
        from pathlib import Path as _Path

        strategy_path = _Path(strategy_path_str)
        if not strategy_path.exists():
            raise FileNotFoundError(
                f"lookahead_gate: generated strategy file missing: {strategy_path}"
            )

        pairs = list(state.get("pairs") or ["BTC/USDT"])
        timeframe = state.get("timeframe") or "5m"
        # The gate uses a fixed lookahead window — the integration is
        # about catching forward shifts, not about regime coverage.
        # State can override via artifacts.lookahead_timerange if a
        # future caller wants pair-specific cached data.
        timerange = (
            (state.get("artifacts") or {}).get("lookahead_timerange")
            or DEFAULT_LOOKAHEAD_TIMERANGE
        )

        result: LookaheadResult = await runner(
            strategy_path,
            pairs=pairs,
            timeframe=timeframe,
            timerange=timerange,
        )

        existing_artifacts = state.get("artifacts") or {}
        base_update: dict[str, Any] = {
            "artifacts": {
                **existing_artifacts,
                "lookahead_analysis": dict(result),
            },
            "agent_votes": [
                {
                    "agent": "lookahead_gate",
                    "verdict": "pass" if result["passed"] else "fail",
                    "rationale": result.get("details", ""),
                    "confidence": 1.0,
                },
            ],
        }

        if result["passed"]:
            # Falls through to the builder's add_edge(lookahead_gate, END).
            return Command(update=base_update)

        # BRD §8 rule 5: archive with failure_reason="lookahead_bias".
        # We extend with the runner's details so the operator log isn't
        # opaque, but keep the canonical prefix for grep-ability.
        return Command(
            goto="archive",
            update={
                **base_update,
                "stage": "archived",
                "failure_reason": f"lookahead_bias: {result.get('details', '<no details>')}",
            },
        )

    return lookahead_gate


# ─── archive terminal sink (5d) ────────────────────────────────────────


def archive(state: ResearchState) -> dict[str, Any]:
    """Terminal sink for archive routes (critic_loop_exhausted, AST fail).

    Mirrors :func:`orchestrator.subgraphs.validation.archive` — stamps
    ``stage="archived"`` and preserves ``failure_reason`` if upstream
    nodes already set one. The router (``revise_or_proceed``) and the
    generator already set both fields on archive paths; this node is the
    routing destination, ensuring there's a single sink edge to END.
    """
    return {
        "stage": "archived",
        "failure_reason": (
            state.get("failure_reason") or "research_archived_without_reason"
        ),
    }


# ─── Subgraph builder ──────────────────────────────────────────────────


def build_research_subgraph(
    *,
    store: BaseStore | None = None,
    researcher_fn: ResearcherFn | None = None,
    generator_fn: GeneratorFn | None = None,
    critic_fn: CriticFn | None = None,
    lookahead_runner: LookaheadRunnerFn | None = None,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[ResearchState, ResearchState, ResearchState, ResearchState]:
    """Compile the Stage 5e research subgraph (critic loop + lookahead gate).

    Topology::

        START
          ↓
        load_context
          ↓
        researcher (Sonnet 4.6 ReAct)
          ↓
        generator (deterministic + Sonnet 4.6 structured output)
          ↓ ←─────────────────┐
        critic (Opus 4.7 ReAct)│
          ↓                    │
        revise_or_proceed      │ revise (count < MAX_REVISIONS)
          ├── revise ──────────┘   (revision_count incremented)
          ├── revise (count ≥ MAX_REVISIONS) ──> archive ──> END
          └── pass ──> lookahead_gate
                          ├── pass ──> END
                          └── fail ──> archive ──> END

    Parameters
    ----------
    store
        Long-term Store handle for ``load_context`` (BRD §5.9). Unit
        tests pass ``InMemoryStore`` or ``None``; production wires the
        ``AsyncPostgresStore`` from the FastAPI lifespan.
    researcher_fn, generator_fn, critic_fn
        Optional overrides for the agent-backed nodes. Defaults invoke
        the real Sonnet 4.6 (researcher, generator) or Opus 4.7
        (critic) agents. Unit tests pass deterministic stubs so the CI
        gate doesn't burn LLM tokens; the bounded-loop test in
        particular passes a critic stub that always returns "revise"
        to verify the MAX_REVISIONS bound.
    lookahead_runner
        Optional override for the Freqtrade lookahead-analysis
        subprocess. Defaults to the real Docker-backed runner. Unit
        tests + the research-subgraph integration test pass a stub
        returning a synthetic LookaheadResult so the CI gate doesn't
        need Docker.
    checkpointer
        Optional saver. Unit tests pass ``InMemorySaver``; production
        wires ``AsyncPostgresSaver`` from the FastAPI lifespan.
    """

    async def _default_researcher(state: dict[str, Any]) -> dict[str, Any]:
        return await researcher_node(state, store=store)

    async def _default_generator(state: dict[str, Any]) -> dict[str, Any]:
        return await generator_node(state)

    async def _default_critic(state: dict[str, Any]) -> dict[str, Any]:
        return await critic_node(state)

    researcher = researcher_fn or _default_researcher
    generator = generator_fn or _default_generator
    critic = critic_fn or _default_critic
    load_context = make_load_context(store=store)
    lookahead_gate = make_lookahead_gate(lookahead_runner=lookahead_runner)

    builder: StateGraph[ResearchState, ResearchState, ResearchState, ResearchState] = (
        StateGraph(ResearchState)
    )
    # Closure-async nodes hit the same false-positive LangGraph generic
    # mismatch as validation.py's backtest_worker / risk_analyst.
    builder.add_node("load_context", load_context)  # type: ignore[arg-type]
    builder.add_node("researcher", researcher)  # type: ignore[arg-type]
    builder.add_node("generator", generator)  # type: ignore[arg-type]
    builder.add_node("critic", critic)  # type: ignore[arg-type]
    # revise_or_proceed returns Command(goto=...) — sync function, no
    # closure-generic issue. It's a router, not an agent.
    builder.add_node("revise_or_proceed", revise_or_proceed)
    builder.add_node("lookahead_gate", lookahead_gate)  # type: ignore[arg-type]
    builder.add_node("archive", archive)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "researcher")
    builder.add_edge("researcher", "generator")
    builder.add_edge("generator", "critic")
    builder.add_edge("critic", "revise_or_proceed")
    # revise_or_proceed and lookahead_gate route via Command(goto=...)
    # exclusively — see the post-5d correction in
    # orchestrator.agents.critic.revise_or_proceed docstring for why
    # we don't mix Command(no-goto) + add_edge. Targets:
    #   - revise_or_proceed → "generator" (revise back-edge) |
    #     "archive" (count cap) | "lookahead_gate" (pass).
    #   - lookahead_gate → "archive" (fail) | END (pass, via
    #     Command(update=...) — END is a terminal sentinel so no
    #     add_edge-collision risk, unlike a real downstream node).
    builder.add_edge("lookahead_gate", END)
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=checkpointer)
