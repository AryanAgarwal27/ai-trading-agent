"""Per-strategy parent graph (BRD §5.2).

Composes the four lifecycle subgraphs in sequence. Stage 5e wires the
research subgraph as the first stage; validation / paper / live land in
later stages (their subgraphs already exist as Stage 4 / future work,
but the parent-graph composition is incremental).

Topology (this commit, post-5e)::

    START ──> research_subgraph ──> archive ──> END

The ``research_subgraph`` carries its own internal topology (BRD §5.3,
see :mod:`orchestrator.subgraphs.research`). If it terminates with
``stage="archived"`` (critic_loop_exhausted, ast_validator failure,
lookahead_bias), the parent graph's archive sink preserves the
``failure_reason``. If it terminates without an archive — i.e. the
strategy passed the lookahead gate — the parent graph stops here for
now; Stage 6+ will route to ``validation_subgraph`` in place of the
archive sink.

The ``research_subgraph`` is injectable: production wires the real
agent-backed subgraph; integration tests pass a stub-agent subgraph
so the parent-graph wiring can be exercised without real LLM calls.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.state import StrategyState
from orchestrator.subgraphs.research import build_research_subgraph


def archive(state: StrategyState) -> dict[str, Any]:
    """Terminal sink (BRD §5.3, §5.9 — failures land here, wins land here too).

    Mirrors :func:`orchestrator.subgraphs.validation.archive` and
    :func:`orchestrator.subgraphs.research.archive`. Stamps
    ``stage="archived"`` and preserves the upstream
    ``failure_reason`` if any (research subgraph's lookahead_bias,
    critic_loop_exhausted, ast_validator paths all set it; a passing
    research run terminates with no failure_reason and the sink reports
    the placeholder ``research_complete_no_validation_subgraph_yet``).
    """
    return {
        "stage": "archived",
        "failure_reason": (
            state.get("failure_reason")
            or "research_complete_no_validation_subgraph_yet"
        ),
    }


def build_per_strategy_graph(
    saver: AsyncPostgresSaver | Any,
    store: AsyncPostgresStore | Any,
    *,
    research_subgraph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
) -> CompiledStateGraph[StrategyState, StrategyState, StrategyState, StrategyState]:
    """Compile the per-strategy parent graph (BRD §5.2).

    Parameters
    ----------
    saver, store
        The LangGraph saver + Store handles opened in the FastAPI
        lifespan (BRD §6.5). Production passes Postgres-backed
        implementations; integration tests pass InMemory variants.
    research_subgraph
        Optional pre-built research subgraph. When None, builds the
        default (real-agent) subgraph passing the same Store handle so
        load_context can read failures/wins from the long-term Store.
        Integration tests pass a stub-agent subgraph so the parent
        graph can round-trip without invoking real LLMs.

    Topology:
      START → research_subgraph → archive → END
    """
    research = research_subgraph or build_research_subgraph(store=store)

    builder: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = (
        StateGraph(StrategyState)
    )
    # The compiled research subgraph is itself a node — LangGraph
    # supports nested compiled graphs as nodes directly. The subgraph's
    # state schema (ResearchState, total=False) is a subset of
    # StrategyState's field names, so passthrough works without
    # explicit channel mapping.
    builder.add_node("research_subgraph", research)
    builder.add_node("archive", archive)

    builder.add_edge(START, "research_subgraph")
    builder.add_edge("research_subgraph", "archive")
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=saver, store=store)
