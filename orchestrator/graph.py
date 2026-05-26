"""Per-strategy skeleton graph (BRD §13 Stage 2).

Two-node skeleton (`research_stub` → `archive`) used to validate that the
PostgresSaver round-trips checkpoint rows under a stable `thread_id`. Stages
5+ replace `research_stub` with the real research subgraph; the `archive`
node persists as a terminal sink across the full lifecycle (BRD §5.3, §5.9).
"""

from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.state import StrategyState


def research_stub(state: StrategyState) -> dict:
    return {"stage": "archived"}


def archive(state: StrategyState) -> dict:
    return {
        "failure_reason": state.get("failure_reason") or "stage_2_skeleton_run",
    }


def build_per_strategy_graph(
    saver: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(StrategyState)
    builder.add_node("research_stub", research_stub)
    builder.add_node("archive", archive)
    builder.add_edge(START, "research_stub")
    builder.add_edge("research_stub", "archive")
    builder.add_edge("archive", END)
    return builder.compile(checkpointer=saver, store=store)
