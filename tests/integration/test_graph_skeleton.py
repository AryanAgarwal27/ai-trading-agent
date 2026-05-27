"""Integration test — parent per-strategy graph round-trips Postgres checkpoint.

History: this test originated as the Stage 2 DoD check
(``research_stub → archive`` round-tripping checkpoint rows under a
stable ``thread_id``). Stage 5e replaced ``research_stub`` with the
real :mod:`orchestrator.subgraphs.research` subgraph; this test was
updated to build the parent graph manually with a stub-agent research
subgraph so it doesn't hit Anthropic + Opus during CI.

What this test asserts (the contract is unchanged from Stage 2):

  1. ``await graph.ainvoke(initial_state, config)`` runs the parent
     graph from START to END.
  2. ``aget_state(config)`` returns a StateSnapshot reflecting the
     archive sink's writes.
  3. A direct psycopg query against ``langgraph_checkpoints.checkpoints``
     confirms ≥1 persisted row keyed on the thread_id — proving
     on-disk checkpoint persistence (BRD §1.1 rule 6), not just
     in-memory state propagation.

We deliberately do NOT use ``ainvoke(None, config=...)`` here: that's
the HITL-resume-after-interrupt idiom, and this graph has no
``interrupt()`` calls. Exercising it would test resume semantics that
don't apply.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from orchestrator.graph import build_per_strategy_graph
from orchestrator.state import StrategyState
from orchestrator.subgraphs.research import build_research_subgraph


def _initial_state(strategy_id: str) -> StrategyState:
    now = datetime.now(UTC).isoformat()
    return {
        "strategy_id": strategy_id,
        "name": "stage-5e-parent-roundtrip",
        "hypothesis": "skeleton round-trip",
        "template": "n/a",
        "params": {},
        "freqai_config": None,
        "pairs": ["BTC/USDT"],
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


# ─── Stub agent functions for the research subgraph ───────────────────


async def _stub_researcher(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("artifacts") or {}
    return {
        "hypothesis": "skeleton-test hypothesis",
        "template": "mean_reversion_template",
        "artifacts": {
            **existing,
            "research_proposal": {
                "hypothesis": "skeleton-test hypothesis",
                "template_name": "mean_reversion_template",
                "regime_thesis": "stub",
                "suggested_param_ranges": {},
                "confidence": 0.8,
            },
        },
        "agent_votes": [
            {
                "agent": "researcher",
                "verdict": "continue",
                "rationale": "stub",
                "confidence": 0.8,
            }
        ],
    }


async def _stub_generator(state: dict[str, Any]) -> dict[str, Any]:
    """Stub generator that pretends to render a strategy file.

    Writes a tiny file under the test's tmp path so lookahead_gate's
    existence check passes. Path lives in the OS temp dir to keep the
    integration test hermetic from the repo's strategy_templates/."""
    import tempfile
    from pathlib import Path as _Path

    out_dir = _Path(tempfile.gettempdir()) / "ai_trading_agent_test_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state['strategy_id']}.py"
    out_path.write_text("# stub generated strategy\n", encoding="utf-8")

    existing = state.get("artifacts") or {}
    return {
        "params": {"stub": True},
        "strategy_path": str(out_path),
        "artifacts": {**existing, "generated_strategy_path": str(out_path)},
        "agent_votes": [
            {
                "agent": "generator",
                "verdict": "pass",
                "rationale": "stub",
                "confidence": 1.0,
            }
        ],
    }


async def _stub_pass_critic(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("artifacts") or {}
    prior = list(existing.get("critic_verdicts") or [])
    prior.append(
        {
            "verdict": "pass",
            "primary_concern": "stub",
            "rationale": "stub",
            "revision_guidance": "",
            "confidence": 0.9,
        }
    )
    return {
        "agent_votes": [
            {"agent": "critic", "verdict": "pass", "rationale": "stub", "confidence": 0.9}
        ],
        "critic_notes": [],
        "artifacts": {**existing, "critic_verdicts": prior},
    }


async def _stub_pass_lookahead(strategy_path, *, pairs, timeframe, timerange):
    return {
        "passed": True,
        "details": "stub: no look-ahead bias",
        "returncode": 0,
        "worker_dir": "/tmp/stub_la",
        "stderr_tail": "",
        "stdout_tail": "",
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parent_graph_persists_checkpoint_under_thread_id() -> None:
    """Build the parent graph manually with stub-agent research subgraph
    and verify thread-keyed checkpoint persistence in Postgres."""
    checkpoint_uri = os.environ["LANGGRAPH_CHECKPOINT_URI"]
    store_uri = os.environ["LANGGRAPH_STORE_URI"]

    async with (
        AsyncPostgresSaver.from_conn_string(checkpoint_uri) as saver,
        AsyncPostgresStore.from_conn_string(store_uri) as store,
    ):
        await saver.setup()
        await store.setup()

        research = build_research_subgraph(
            store=store,
            researcher_fn=_stub_researcher,
            generator_fn=_stub_generator,
            critic_fn=_stub_pass_critic,
            lookahead_runner=_stub_pass_lookahead,
        )
        graph = build_per_strategy_graph(
            saver, store, research_subgraph=research
        )

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
        # Pass-through path (no archive in research) lands the
        # placeholder failure_reason from the parent graph's archive
        # sink (Stage 6 will replace with validation_subgraph routing).
        assert "research_complete_no_validation_subgraph_yet" in (
            snapshot.values.get("failure_reason") or ""
        )

        async with await psycopg.AsyncConnection.connect(checkpoint_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()

        assert row is not None
        assert row[0] >= 1, (
            f"expected at least one checkpoints row for thread_id={thread_id}, got {row[0]}"
        )
