"""Tests for orchestrator.subgraphs.research load_context wiring.

Originally Stage 5c skeleton tests; updated for Stage 5d's new topology
(START → load_context → researcher → generator → critic →
revise_or_proceed → END). Each test passes a critic stub that
unconditionally votes pass so the loop terminates in one iteration —
the bounded-loop behavior is covered by :mod:`tests.unit.test_critic_loop`.

These tests retain value as load_context-specific contract checks
(regime propagation, default-regime fallback, Store seeding) that are
narrower than the loop tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from orchestrator.subgraphs.research import build_research_subgraph
from orchestrator.tools.store_queries import aput_failure, aput_win


@pytest.fixture
def loaded_store() -> InMemoryStore:
    """An InMemoryStore pre-seeded with one failure + one win in low_vol_up.

    Tests assert load_context surfaces these into
    ``artifacts.loaded_context``."""
    store = InMemoryStore()
    return store


def _make_pass_critic_stub() -> Any:
    """Critic stub that always votes pass — collapses the 5d bounded
    loop into a single iteration so these tests stay focused on
    load_context behavior, not loop behavior."""

    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        existing = state.get("artifacts") or {}
        verdict_dump = {
            "verdict": "pass",
            "primary_concern": "stub",
            "rationale": "stub",
            "revision_guidance": "",
            "confidence": 0.9,
        }
        prior = list(existing.get("critic_verdicts") or [])
        prior.append(verdict_dump)
        return {
            "agent_votes": [
                {
                    "agent": "critic",
                    "verdict": "pass",
                    "rationale": "stub",
                    "confidence": 0.9,
                }
            ],
            "critic_notes": [],
            "artifacts": {**existing, "critic_verdicts": prior},
        }

    return stub


async def test_subgraph_runs_end_to_end_with_stubs(tmp_path: Path) -> None:
    """Topology test: stubbed researcher + generator round-trip through
    START → load_context → researcher → generator → END. Asserts every
    expected state field is populated."""

    async def stub_researcher(state: dict[str, Any]) -> dict[str, Any]:
        # Mimics the real researcher_node's output contract.
        existing_artifacts = state.get("artifacts") or {}
        return {
            "hypothesis": "stubbed hypothesis for topology test",
            "template": "mean_reversion_template",
            "artifacts": {
                **existing_artifacts,
                "research_proposal": {
                    "hypothesis": "stubbed hypothesis",
                    "template_name": "mean_reversion_template",
                    "regime_thesis": "stubbed thesis",
                    "suggested_param_ranges": {"bb_period": "15-25"},
                    "confidence": 0.7,
                },
            },
            "agent_votes": [
                {
                    "agent": "researcher",
                    "verdict": "continue",
                    "rationale": "stub",
                    "confidence": 0.7,
                }
            ],
        }

    async def stub_generator(state: dict[str, Any]) -> dict[str, Any]:
        # Mimics the real generator_node happy path.
        out_path = tmp_path / f"{state['strategy_id']}.py"
        out_path.write_text("# stubbed strategy\n", encoding="utf-8")
        existing_artifacts = state.get("artifacts") or {}
        return {
            "params": {"bb_period": 20},
            "strategy_path": str(out_path),
            "artifacts": {
                **existing_artifacts,
                "generated_strategy_path": str(out_path),
            },
            "agent_votes": [
                {
                    "agent": "generator",
                    "verdict": "pass",
                    "rationale": "stub",
                    "confidence": 1.0,
                }
            ],
        }

    graph = build_research_subgraph(
        store=None,
        researcher_fn=stub_researcher,
        generator_fn=stub_generator,
        critic_fn=_make_pass_critic_stub(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_topology_001"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_topo_001",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "low_vol_up",
        },
        config=config,
    )

    # load_context populated regime + loaded_context block.
    assert final["current_regime"] == "low_vol_up"
    assert "loaded_context" in final["artifacts"]
    assert final["artifacts"]["loaded_context"]["regime"] == "low_vol_up"
    # No store passed → empty failures/wins.
    assert final["artifacts"]["loaded_context"]["failures_count"] == 0
    assert final["artifacts"]["loaded_context"]["wins_count"] == 0

    # researcher populated hypothesis + template + research_proposal artifact.
    assert final["hypothesis"] == "stubbed hypothesis for topology test"
    assert final["template"] == "mean_reversion_template"
    assert final["artifacts"]["research_proposal"]["template_name"] == "mean_reversion_template"

    # generator populated params + strategy_path + generated_strategy_path artifact.
    assert final["params"] == {"bb_period": 20}
    assert final["strategy_path"] == str(tmp_path / "test_topo_001.py")
    assert final["artifacts"]["generated_strategy_path"] == str(tmp_path / "test_topo_001.py")

    # Agent votes accumulated through the reducer-less list (not yet
    # using add reducer at this subgraph state level — 5c keeps it
    # simple, parent graph state is where the reducer lives).
    assert any(v["agent"] == "researcher" for v in final["agent_votes"])
    assert any(v["agent"] == "generator" for v in final["agent_votes"])


async def test_load_context_reads_from_store(tmp_path: Path) -> None:
    """When a real Store is passed, load_context surfaces the failures
    and wins for the current regime into artifacts.loaded_context."""
    store = InMemoryStore()
    await aput_failure(
        store,
        regime="low_vol_up",
        strategy_id="past_failure_001",
        payload={"hypothesis": "old bad", "failure_reason": "lost money"},
    )
    await aput_win(
        store,
        regime="low_vol_up",
        strategy_id="past_win_001",
        payload={"hypothesis": "old good", "live_metrics_summary": {"sharpe": 1.4}},
    )

    async def stub_researcher(state: dict[str, Any]) -> dict[str, Any]:
        existing_artifacts = state.get("artifacts") or {}
        return {
            "hypothesis": "h",
            "template": "mean_reversion_template",
            "artifacts": {**existing_artifacts, "research_proposal": {}},
            "agent_votes": [],
        }

    async def stub_generator(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "params": {},
            "strategy_path": str(tmp_path / "x.py"),
            "artifacts": state.get("artifacts") or {},
            "agent_votes": [],
        }

    graph = build_research_subgraph(
        store=store,
        researcher_fn=stub_researcher,
        generator_fn=stub_generator,
        critic_fn=_make_pass_critic_stub(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_load_ctx_001"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_load_001",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "low_vol_up",
        },
        config=config,
    )

    ctx = final["artifacts"]["loaded_context"]
    assert ctx["failures_count"] == 1
    assert ctx["wins_count"] == 1
    # Raw records carried in artifacts (so a tool call falling back to
    # this dict gets the same data the Store search returned).
    assert ctx["failures"][0]["key"] == "past_failure_001"
    assert ctx["failures"][0]["failure_reason"] == "lost money"
    assert ctx["wins"][0]["key"] == "past_win_001"


async def test_load_context_defaults_regime_when_missing(tmp_path: Path) -> None:
    """No current_regime in state → load_context sets it to 'unknown'."""

    async def stub_researcher(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "hypothesis": "h",
            "template": "mean_reversion_template",
            "artifacts": {**(state.get("artifacts") or {}), "research_proposal": {}},
            "agent_votes": [],
        }

    async def stub_generator(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "params": {},
            "strategy_path": str(tmp_path / "x.py"),
            "artifacts": state.get("artifacts") or {},
            "agent_votes": [],
        }

    graph = build_research_subgraph(
        store=None,
        researcher_fn=stub_researcher,
        generator_fn=stub_generator,
        critic_fn=_make_pass_critic_stub(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_no_regime_001"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_no_regime",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
        },
        config=config,
    )
    assert final["current_regime"] == "unknown"
    assert final["artifacts"]["loaded_context"]["regime"] == "unknown"
