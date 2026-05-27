"""Research subgraph (BRD §5.3) — 5d (critic + bounded loop).

Topology (this commit, 5d)::

    START ──> load_context ──> researcher ──> generator ──> critic
                                                  ▲           │
                                                  │           ▼
                                                  │     revise_or_proceed
                                                  │      │      │      │
                                                  │      │      │      └── pass ──> END
                                                  │      │      └── revise (count ≥ 3) ──> archive ──> END
                                                  │      │
                                                  └──────┘  revise (count < 3)
                                                            (revision_count incremented)

Topology after 5e (next commit)::

    ... critic → revise_or_proceed --pass--> lookahead_gate --pass--> END
                                                              \\--fail--> archive

5d wires three new pieces: ``critic_node`` (Opus 4.7 ReAct),
``revise_or_proceed`` (deterministic ``Command(goto, update)`` router),
and the back-edge from ``revise_or_proceed`` → ``generator`` for bounded
re-generation. The bounded loop is capped at
:data:`orchestrator.agents.critic.MAX_REVISIONS` = 3 per BRD §5.3.

State additions versus 5c:
  - ``revision_count: int`` — incremented by ``revise_or_proceed`` on
    each revise back-edge; capped check uses ``MAX_REVISIONS``.
  - ``critic_notes: Annotated[list[str], add]`` — BRD §5.7 reducer
    field; appended by each critic call. The generator's extractor
    reads the full accumulated list on revision passes so it can
    address EVERY prior concern, not just the latest.
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
from orchestrator.tools.store_queries import aget_failures, aget_wins

CheckpointSaver = BaseCheckpointSaver[Any]

# ─── Node injection seams ──────────────────────────────────────────────
# Same pattern as build_validation_subgraph (Stage 4e): factory takes
# optional callable overrides so unit tests can inject deterministic
# stubs without touching the production code path.
ResearcherFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
GeneratorFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
CriticFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


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
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[ResearchState, ResearchState, ResearchState, ResearchState]:
    """Compile the Stage 5d research subgraph (critic + bounded loop).

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
          └── pass ──> END

    The lookahead gate (BRD §5.3) lands in Stage 5e and is wired in
    place of the pass→END edge.

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
    builder.add_node("archive", archive)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "researcher")
    builder.add_edge("researcher", "generator")
    builder.add_edge("generator", "critic")
    builder.add_edge("critic", "revise_or_proceed")
    # revise_or_proceed routes via Command(goto=...) to one of:
    #   - "generator" (revise back-edge), or
    #   - "archive" (revise count ≥ MAX_REVISIONS), or
    #   - END via Command(update={}) with no goto (pass-through).
    # No explicit conditional edges needed for the Command paths; we
    # add the explicit pass-through edge so LangGraph knows the
    # terminal route exists.
    builder.add_edge("revise_or_proceed", END)
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=checkpointer)
