"""Bounded-loop tests for the research subgraph (Stage 5d).

End-to-end runs of build_research_subgraph with stubbed researcher /
generator / critic nodes, asserting:

  - A critic that always votes "revise" causes the loop to bound at
    MAX_REVISIONS and archive with failure_reason="critic_loop_exhausted".
  - A critic that votes "pass" on the Nth call terminates the loop
    cleanly after N-1 revisions.
  - revision_count increments correctly and is visible in final state.
  - critic_notes accumulate across revisions (reducer contract).
  - The generator sees prior critic_notes on each revision pass
    (verified by stub that records what state["critic_notes"] looked
    like at its invocation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.agents.critic import MAX_REVISIONS
from orchestrator.subgraphs.research import build_research_subgraph


def _make_stub_researcher() -> Any:
    """Researcher stub: deterministic single proposal, no LLM."""

    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        existing = state.get("artifacts") or {}
        return {
            "hypothesis": "h",
            "template": "mean_reversion_template",
            "artifacts": {
                **existing,
                "research_proposal": {
                    "hypothesis": "h",
                    "template_name": "mean_reversion_template",
                    "regime_thesis": "rt",
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

    return stub


def _make_recording_generator(tmp_path: Path, observed: list[list[str]]) -> Any:
    """Generator stub that records the value of state['critic_notes'] at
    each invocation, then writes a trivial strategy file."""

    call_count = {"n": 0}

    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        # Snapshot the critic_notes the generator sees on this pass.
        # The first pass sees an empty list (or absent); subsequent
        # passes see the accumulated critic guidance.
        observed.append(list(state.get("critic_notes") or []))
        out_path = tmp_path / f"{state['strategy_id']}_call_{call_count['n']}.py"
        out_path.write_text("# stub strategy\n", encoding="utf-8")
        existing = state.get("artifacts") or {}
        return {
            "params": {"x": call_count["n"]},
            "strategy_path": str(out_path),
            "artifacts": {
                **existing,
                "generated_strategy_path": str(out_path),
            },
            "agent_votes": [
                {
                    "agent": "generator",
                    "verdict": "pass",
                    "rationale": f"call {call_count['n']}",
                    "confidence": 1.0,
                }
            ],
        }

    return stub


def _make_always_revise_critic(call_count: list[int]) -> Any:
    """Critic stub that always votes revise. ``call_count`` is a
    single-element list (mutable counter)."""

    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        call_count[0] += 1
        n = call_count[0]
        existing = state.get("artifacts") or {}
        prior = list(existing.get("critic_verdicts") or [])
        verdict_dump = {
            "verdict": "revise",
            "primary_concern": f"call_{n}_concern",
            "rationale": f"call_{n}_rationale",
            "revision_guidance": f"call_{n}_guidance",
            "confidence": 0.7,
        }
        prior.append(verdict_dump)
        return {
            "agent_votes": [
                {
                    "agent": "critic",
                    "verdict": "revise",
                    "rationale": f"call_{n}_rationale",
                    "confidence": 0.7,
                }
            ],
            "critic_notes": [f"call_{n}_guidance"],
            "artifacts": {**existing, "critic_verdicts": prior},
        }

    return stub


def _make_pass_on_nth_critic(pass_on: int, call_count: list[int]) -> Any:
    """Critic stub that votes revise on calls 1..(pass_on-1) and pass on
    call ``pass_on``."""

    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        call_count[0] += 1
        n = call_count[0]
        verdict_str = "pass" if n == pass_on else "revise"
        existing = state.get("artifacts") or {}
        prior = list(existing.get("critic_verdicts") or [])
        verdict_dump = {
            "verdict": verdict_str,
            "primary_concern": f"call_{n}",
            "rationale": f"call_{n}_rationale",
            "revision_guidance": "" if verdict_str == "pass" else f"call_{n}_guidance",
            "confidence": 0.8,
        }
        prior.append(verdict_dump)
        return {
            "agent_votes": [
                {
                    "agent": "critic",
                    "verdict": verdict_str,
                    "rationale": f"call_{n}_rationale",
                    "confidence": 0.8,
                }
            ],
            "critic_notes": (
                [] if verdict_str == "pass" else [f"call_{n}_guidance"]
            ),
            "artifacts": {**existing, "critic_verdicts": prior},
        }

    return stub


# ─── Bounded-loop tests ────────────────────────────────────────────────


async def test_loop_bounds_at_max_revisions_with_always_revise(tmp_path: Path) -> None:
    """Critic that always votes revise → loop terminates at MAX_REVISIONS.

    Expected sequence (MAX_REVISIONS=3):
      - generator call 1 (initial, count=0) → critic call 1 (revise, count<3) → generator
      - generator call 2 (count=1) → critic call 2 (revise, count<3) → generator
      - generator call 3 (count=2) → critic call 3 (revise, count<3) → generator
      - generator call 4 (count=3) → critic call 4 (revise, count NOT<3) → archive

    So: 4 generator calls, 4 critic calls, archived with critic_loop_exhausted.
    """
    observed_critic_notes_on_gen_call: list[list[str]] = []
    critic_calls: list[int] = [0]

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(),
        generator_fn=_make_recording_generator(tmp_path, observed_critic_notes_on_gen_call),
        critic_fn=_make_always_revise_critic(critic_calls),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_loop_max"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_loop_max",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    # Generator called once initially + MAX_REVISIONS times for each revision pass.
    assert len(observed_critic_notes_on_gen_call) == MAX_REVISIONS + 1
    # Critic called once per generator output.
    assert critic_calls[0] == MAX_REVISIONS + 1
    # Archived with the canonical failure_reason.
    assert final["stage"] == "archived"
    assert final["failure_reason"].startswith("critic_loop_exhausted:")
    assert f"MAX_REVISIONS={MAX_REVISIONS}" in final["failure_reason"]
    # Last critic guidance carried into the failure_reason.
    assert f"call_{MAX_REVISIONS + 1}_guidance" in final["failure_reason"]
    # revision_count reached MAX_REVISIONS (the count incremented to
    # MAX_REVISIONS on the last successful revise edge before the cap).
    assert final["revision_count"] == MAX_REVISIONS


async def test_loop_terminates_cleanly_when_critic_passes(tmp_path: Path) -> None:
    """Critic that votes pass on call 2 → 1 revision then clean exit."""
    observed: list[list[str]] = []
    critic_calls: list[int] = [0]

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(),
        generator_fn=_make_recording_generator(tmp_path, observed),
        critic_fn=_make_pass_on_nth_critic(pass_on=2, call_count=critic_calls),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_loop_pass_2nd"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_loop_pass_2nd",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    # 2 generator calls (initial + 1 revision).
    assert len(observed) == 2
    assert critic_calls[0] == 2
    # NOT archived — pass route.
    assert "stage" not in final or final.get("stage") != "archived"
    assert "failure_reason" not in final or not final.get("failure_reason")
    # 1 revision performed.
    assert final.get("revision_count") == 1


async def test_loop_terminates_immediately_when_critic_passes_first_call(
    tmp_path: Path,
) -> None:
    """Critic that votes pass on the first call → 0 revisions, clean exit."""
    observed: list[list[str]] = []
    critic_calls: list[int] = [0]

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(),
        generator_fn=_make_recording_generator(tmp_path, observed),
        critic_fn=_make_pass_on_nth_critic(pass_on=1, call_count=critic_calls),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_loop_pass_1st"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_loop_pass_1st",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    assert len(observed) == 1
    assert critic_calls[0] == 1
    assert final.get("stage") != "archived"
    # revision_count never incremented (still 0 or absent).
    assert (final.get("revision_count") or 0) == 0


async def test_generator_sees_accumulated_critic_notes(tmp_path: Path) -> None:
    """The generator on revision pass N must see all N-1 critic notes
    accumulated in state['critic_notes'] (add reducer)."""
    observed: list[list[str]] = []
    critic_calls: list[int] = [0]

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(),
        generator_fn=_make_recording_generator(tmp_path, observed),
        critic_fn=_make_always_revise_critic(critic_calls),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_critic_notes_accum"}}
    await graph.ainvoke(
        {
            "strategy_id": "test_critic_notes_accum",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    # 4 generator calls total (initial + 3 revisions before cap).
    # Initial pass sees no critic notes.
    assert observed[0] == []
    # 1st revision (generator call 2) sees critic call 1's guidance.
    assert observed[1] == ["call_1_guidance"]
    # 2nd revision sees calls 1 + 2.
    assert observed[2] == ["call_1_guidance", "call_2_guidance"]
    # 3rd revision sees calls 1 + 2 + 3.
    assert observed[3] == ["call_1_guidance", "call_2_guidance", "call_3_guidance"]


async def test_critic_verdicts_artifact_accumulates(tmp_path: Path) -> None:
    """artifacts['critic_verdicts'] grows with one entry per critic call."""
    observed: list[list[str]] = []
    critic_calls: list[int] = [0]

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(),
        generator_fn=_make_recording_generator(tmp_path, observed),
        critic_fn=_make_always_revise_critic(critic_calls),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_critic_artifacts"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_critic_artifacts",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    verdicts = (final.get("artifacts") or {}).get("critic_verdicts") or []
    # MAX_REVISIONS + 1 critic calls happened, each appending one verdict.
    assert len(verdicts) == MAX_REVISIONS + 1
    # All revise.
    assert all(v["verdict"] == "revise" for v in verdicts)
    # Each has a unique guidance string (verifies appender works
    # call-order correctly).
    guidances = [v["revision_guidance"] for v in verdicts]
    assert guidances == [f"call_{i + 1}_guidance" for i in range(MAX_REVISIONS + 1)]
