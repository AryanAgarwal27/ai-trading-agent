"""Integration test for the Stage 5e research subgraph.

End-to-end run of build_research_subgraph against the REAL templates
(BRD §8.1), the REAL Stage 5a AST validator, and the REAL line-oriented
SLOT substitution from generator.render_template — only the LLM-backed
nodes (researcher, generator extractor, critic) and the Freqtrade
lookahead subprocess are stubbed.

Why this lives in tests/integration/ even though no Docker / Postgres
is required: the test exercises a 6-node subgraph end-to-end (load
→ researcher → generator → critic → revise_or_proceed → lookahead_gate)
and writes a real strategy file under tmp_path that the AST validator
runs against. It's heavier than the unit tests, and a failure here
typically points at the subgraph wiring rather than at one node's
behavior, which matches the integration-test marker semantics.

BRD §13 Stage 5 DoD pinned by this test:
  - "a research run produces a strategy file that passes
    ``freqtrade lookahead-analysis``": exercised here with a
    pass-stub for the lookahead runner; the actual subprocess is the
    operator's manual verification at DoD time.
  - "and contains no disallowed imports": the AST validator (Stage 5a)
    runs inside generator_node on the rendered output.
  - "critic loop bounded at 3": covered by test_critic_loop.py at
    unit-test level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from orchestrator.agents.generator import load_schema
from orchestrator.security.ast_validator import validate_strategy_source
from orchestrator.subgraphs.research import build_research_subgraph

pytestmark = pytest.mark.integration


# ─── helpers ───────────────────────────────────────────────────────────


def _midpoint_params(schema_cls: type[BaseModel]) -> BaseModel:
    raw: dict[str, Any] = {}
    for name, field in schema_cls.model_fields.items():
        ge = le = None
        for c in field.metadata:
            if hasattr(c, "ge"):
                ge = c.ge
            if hasattr(c, "le"):
                le = c.le
        assert ge is not None and le is not None
        if field.annotation is int:
            raw[name] = int((ge + le) // 2)
        else:
            raw[name] = (ge + le) / 2.0
    return schema_cls(**raw)


def _make_stub_researcher(template_name: str):
    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        existing = state.get("artifacts") or {}
        return {
            "hypothesis": "Integration test hypothesis",
            "template": template_name,
            "artifacts": {
                **existing,
                "research_proposal": {
                    "hypothesis": "Integration test hypothesis",
                    "template_name": template_name,
                    "regime_thesis": "stub thesis",
                    "suggested_param_ranges": {},
                    "confidence": 0.85,
                },
            },
            "agent_votes": [
                {
                    "agent": "researcher",
                    "verdict": "continue",
                    "rationale": "stub",
                    "confidence": 0.85,
                }
            ],
        }

    return stub


def _make_stub_pass_critic():
    async def stub(state: dict[str, Any]) -> dict[str, Any]:
        existing = state.get("artifacts") or {}
        prior = list(existing.get("critic_verdicts") or [])
        prior.append(
            {
                "verdict": "pass",
                "primary_concern": "stub-pass",
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

    return stub


def _make_stub_pass_lookahead():
    async def stub(strategy_path, *, pairs, timeframe, timerange):
        return {
            "passed": True,
            "details": "stub: no look-ahead bias",
            "returncode": 0,
            "worker_dir": "/tmp/la_stub",
            "stderr_tail": "",
            "stdout_tail": "",
        }

    return stub


def _make_stub_fail_lookahead():
    async def stub(strategy_path, *, pairs, timeframe, timerange):
        return {
            "passed": False,
            "details": "Found a problem: forward shift in feature column rsi",
            "returncode": 0,
            "worker_dir": "/tmp/la_stub_fail",
            "stderr_tail": "",
            "stdout_tail": "Found a problem: forward shift in feature column rsi",
        }

    return stub


# ─── end-to-end happy path ─────────────────────────────────────────────


async def test_e2e_research_subgraph_passes_with_real_template_and_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full subgraph against real templates: stub researcher selects
    mean_reversion, real generator (with stub extractor) renders it,
    real AST validator runs, stub critic + lookahead pass → END.

    Asserts the strategy file exists, parses, AST-validates, and the
    final state carries the lookahead_analysis artifact + critic and
    lookahead_gate votes."""
    template_name = "mean_reversion_template"
    schema_cls = load_schema(template_name)
    params = _midpoint_params(schema_cls)

    # Stub the extractor (avoid LLM call) but keep the real
    # generator_node pipeline so we exercise render_template + AST
    # validator + file write.
    async def stub_extractor(proposal, template_source, schema_cls):
        return params

    from orchestrator.agents import generator as gen_mod

    monkeypatch.setattr(gen_mod, "_default_params_extractor", stub_extractor)
    monkeypatch.setattr(gen_mod, "GENERATED_DIR", tmp_path / "_generated")

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(template_name),
        critic_fn=_make_stub_pass_critic(),
        lookahead_runner=_make_stub_pass_lookahead(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_e2e_happy"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_e2e_happy",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    # Strategy file lands under the monkey-patched GENERATED_DIR.
    out_path = tmp_path / "_generated" / "test_e2e_happy.py"
    assert out_path.exists(), f"expected rendered file at {out_path}"

    # Re-run the AST validator on the file to confirm it's clean.
    validate_strategy_source(out_path.read_text(encoding="utf-8"))

    # Lookahead analysis artifact present and passed.
    la = final["artifacts"]["lookahead_analysis"]
    assert la["passed"] is True

    # No archive route taken.
    assert final.get("stage") != "archived"
    assert not final.get("failure_reason")

    # All four agent_votes flavors present (researcher, generator,
    # critic, lookahead_gate).
    agents = {v["agent"] for v in final["agent_votes"]}
    assert agents == {"researcher", "generator", "critic", "lookahead_gate"}


async def test_e2e_research_subgraph_archives_on_lookahead_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookahead fail → archive with failure_reason='lookahead_bias: ...'."""
    template_name = "mean_reversion_template"
    schema_cls = load_schema(template_name)
    params = _midpoint_params(schema_cls)

    async def stub_extractor(proposal, template_source, schema_cls):
        return params

    from orchestrator.agents import generator as gen_mod

    monkeypatch.setattr(gen_mod, "_default_params_extractor", stub_extractor)
    monkeypatch.setattr(gen_mod, "GENERATED_DIR", tmp_path / "_generated")

    graph = build_research_subgraph(
        researcher_fn=_make_stub_researcher(template_name),
        critic_fn=_make_stub_pass_critic(),
        lookahead_runner=_make_stub_fail_lookahead(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test_e2e_la_fail"}}
    final = await graph.ainvoke(
        {
            "strategy_id": "test_e2e_la_fail",
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
            "current_regime": "mid_vol_flat",
        },
        config=config,
    )

    assert final["stage"] == "archived"
    assert final["failure_reason"].startswith("lookahead_bias:")
    assert "forward shift" in final["failure_reason"]
    # Strategy file was still written (operator inspects it under
    # _generated/ alongside the archive record).
    assert (tmp_path / "_generated" / "test_e2e_la_fail.py").exists()
