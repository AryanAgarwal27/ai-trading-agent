"""Stage 2 integration test — skeleton graph round-trips checkpoint rows.

BRD §13 Stage 2 DoD: minimal graph (research_stub → archive) round-trips
checkpoint rows; thread_id persists.

What this test asserts:
1. Single `await graph.ainvoke(initial_state, config={...})` runs the
   skeleton graph from START to END.
2. `await graph.aget_state(config)` returns a StateSnapshot whose
   `values["stage"]` is "archived" (the research_stub's transition).
3. A direct psycopg query against langgraph_checkpoints.checkpoints
   confirms at least one persisted row keyed on the thread_id —
   verifying actual on-disk checkpoint persistence (BRD §1.1 rule 6),
   not just in-memory state propagation.

We deliberately do NOT use the `ainvoke(None, config=...)` resume idiom
here: that is the HITL-resume-after-interrupt pattern, and this skeleton
has no `interrupt()` call. Exercising it would test resume semantics
that don't apply.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import psycopg
import pytest

from orchestrator.main import app
from orchestrator.state import StrategyState


def _initial_state(strategy_id: str) -> StrategyState:
    now = datetime.now(UTC).isoformat()
    return {
        "strategy_id": strategy_id,
        "name": "stage-2-skeleton",
        "hypothesis": "skeleton round-trip",
        "template": "n/a",
        "params": {},
        "freqai_config": None,
        "pairs": [],
        "timeframe": "5m",
        "stage": "research",
        "backtest_results": [],
        "robustness_results": [],
        "agent_votes": [],
        "revision_count": 0,
        "critic_notes": [],
        "gate_decisions": {},
        "freqtrade_userdir": None,
        "freqtrade_process_id": None,
        "freqtrade_api_url": None,
        "artifacts": {},
        "started_at": now,
        "last_updated": now,
        "failure_reason": None,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skeleton_graph_persists_checkpoint_under_thread_id() -> None:
    async with app.router.lifespan_context(app):
        graph = app.state.graph
        assert graph is not None, "build_per_strategy_graph was not stashed on app.state"

        strategy_id = str(uuid.uuid4())
        thread_id = f"strategy_{strategy_id}"
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await graph.ainvoke(_initial_state(strategy_id), config=config)
        assert final_state["stage"] == "archived"

        snapshot = await graph.aget_state(config)
        assert snapshot is not None
        assert snapshot.values, "aget_state returned empty StateSnapshot.values"
        assert snapshot.values["stage"] == "archived"
        assert snapshot.values["strategy_id"] == strategy_id
        assert snapshot.values["failure_reason"] == "stage_2_skeleton_run"

        checkpoint_uri = os.environ["LANGGRAPH_CHECKPOINT_URI"]
        async with await psycopg.AsyncConnection.connect(checkpoint_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()

        assert row is not None
        assert (
            row[0] >= 1
        ), f"expected at least one checkpoints row for thread_id={thread_id}, got {row[0]}"
