"""Research subgraph (BRD §5.3) — 5c skeleton.

Topology (this commit)::

    START ──> load_context ──> researcher ──> generator ──> END

Topology after 5d/5e (next commits)::

    START ──> load_context ──> researcher ──> generator ──> critic
                                                  ▲          │
                                                  │     ┌────┴────┐
                                                  │     │         │
                                                  └─revise/    pass/exhausted
                                                       (revision_count < 3)
                                                          │
                                                          ▼
                                                    lookahead_gate
                                                       │      │
                                                       │      └──> archive
                                                       ▼
                                                  (validation entry)

The 5c skeleton wires three real nodes (``load_context``, ``researcher``,
``generator``) and stops at END. The critic loop and lookahead gate land
in 5d and 5e respectively; their absence is why the 5c subgraph builder
takes no critic injection seam (yet).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from orchestrator.agents.generator import generator_node
from orchestrator.agents.researcher import researcher_node
from orchestrator.tools.store_queries import aget_failures, aget_wins

CheckpointSaver = BaseCheckpointSaver[Any]

# ─── Node injection seams ──────────────────────────────────────────────
# Same pattern as build_validation_subgraph (Stage 4e): factory takes
# optional callable overrides so unit tests can inject deterministic
# stubs without touching the production code path.
ResearcherFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
GeneratorFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


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
    # researcher and generator both append rather than the latter
    # overwriting the former — same contract as
    # :class:`orchestrator.state.StrategyState.agent_votes` and
    # ``ValidationState.agent_votes``.
    agent_votes: Annotated[list[dict[str, Any]], add]

    # Lifecycle (set on AST failure)
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


# ─── Subgraph builder ──────────────────────────────────────────────────


def build_research_subgraph(
    *,
    store: BaseStore | None = None,
    researcher_fn: ResearcherFn | None = None,
    generator_fn: GeneratorFn | None = None,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[ResearchState, ResearchState, ResearchState, ResearchState]:
    """Compile the Stage 5c research subgraph.

    Topology::

        START
          ↓
        load_context
          ↓
        researcher (Sonnet 4.6 ReAct)
          ↓
        generator (deterministic + Sonnet 4.6 structured output)
          ↓
        END

    The critic loop + lookahead gate land in 5d/5e. The builder
    signature accepts ``critic_fn`` etc. only when those stages add
    them — no premature parameter additions.

    Parameters
    ----------
    store
        Long-term Store handle for ``load_context`` (BRD §5.9). Unit
        tests pass ``InMemoryStore`` or ``None``; production wires the
        ``AsyncPostgresStore`` from the FastAPI lifespan.
    researcher_fn
        Optional override for the researcher node. Default invokes the
        real Sonnet 4.6 ReAct agent. Tests pass a stub.
    generator_fn
        Optional override for the generator node. Default invokes the
        real Sonnet structured-output extractor + AST validator. Tests
        pass a stub.
    checkpointer
        Optional saver. Unit tests pass ``InMemorySaver``; production
        wires ``AsyncPostgresSaver`` from the FastAPI lifespan.
    """

    async def _default_researcher(state: dict[str, Any]) -> dict[str, Any]:
        return await researcher_node(state, store=store)

    async def _default_generator(state: dict[str, Any]) -> dict[str, Any]:
        return await generator_node(state)

    researcher = researcher_fn or _default_researcher
    generator = generator_fn or _default_generator
    load_context = make_load_context(store=store)

    builder: StateGraph[ResearchState, ResearchState, ResearchState, ResearchState] = (
        StateGraph(ResearchState)
    )
    # Closure-async nodes hit the same false-positive LangGraph generic
    # mismatch as validation.py's backtest_worker / risk_analyst.
    builder.add_node("load_context", load_context)  # type: ignore[arg-type]
    builder.add_node("researcher", researcher)  # type: ignore[arg-type]
    builder.add_node("generator", generator)  # type: ignore[arg-type]

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "researcher")
    builder.add_edge("researcher", "generator")
    builder.add_edge("generator", END)

    return builder.compile(checkpointer=checkpointer)
