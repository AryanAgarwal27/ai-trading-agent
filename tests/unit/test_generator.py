"""Tests for orchestrator.agents.generator (Stage 5c).

Stubs the Sonnet structured-output call with a callable that returns a
pre-built Pydantic instance (BRD §17 #10 FakeListChatModel pattern in
spirit — we don't fake the model itself, we inject one layer up at the
``params_extractor`` seam, which keeps the test isolated from the
langchain_anthropic dep entirely).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from orchestrator.agents.generator import (
    generator_node,
    load_schema,
    render_template,
)
from orchestrator.security.ast_validator import validate_strategy_source

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"


# ─── Helpers ───────────────────────────────────────────────────────────


def _midpoint_params(schema_cls: type[BaseModel]) -> BaseModel:
    """Mid-range Pydantic instance — same approach as test_template_filling."""
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


def _make_stub_extractor(params_instance: BaseModel):
    """Build a ``params_extractor`` that ignores its inputs and returns
    a pre-built Pydantic instance. Mirrors the LLM call's contract
    (same return type) without needing a real model."""

    async def stub(
        proposal: dict[str, Any],
        template_source: str,
        schema_cls: type[BaseModel],
    ) -> BaseModel:
        # Sanity: the seam should be invoked with the schema that matches
        # the proposal's template — catches accidental schema mix-ups.
        assert isinstance(params_instance, schema_cls), (
            f"stub_extractor received schema_cls={schema_cls.__name__} "
            f"but was preloaded with {type(params_instance).__name__}"
        )
        return params_instance

    return stub


# ─── render_template / load_schema unit tests ──────────────────────────


@pytest.mark.parametrize(
    "template_name",
    [
        "mean_reversion_template",
        "freqai_classifier_template",
        "freqai_regressor_template",
    ],
)
def test_load_schema_returns_pydantic_class(template_name: str) -> None:
    cls = load_schema(template_name)
    assert issubclass(cls, BaseModel)
    assert cls.model_fields, f"{template_name} schema has no fields"


def test_load_schema_unknown_template_raises() -> None:
    with pytest.raises(KeyError, match="Unknown template"):
        load_schema("nonexistent_template")


def test_render_template_substitutes_slot_lines() -> None:
    source = (
        "class X:\n"
        "    bb_period: int = 20  # SLOT: bb_period (int, 10-50)\n"
        "    bb_std: float = 2.0  # SLOT: bb_std (float, 1.5-3.0)\n"
        "    untouched: int = 99  # not a SLOT line\n"
    )
    rendered = render_template(source, {"bb_period": 35, "bb_std": 2.5})
    assert "bb_period: int = 35  # SLOT: bb_period" in rendered
    assert "bb_std: float = 2.5  # SLOT: bb_std" in rendered
    assert "untouched: int = 99" in rendered


# ─── generator_node tests ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "template_name",
    [
        "mean_reversion_template",
        "freqai_classifier_template",
        "freqai_regressor_template",
    ],
)
async def test_generator_node_writes_ast_clean_file(
    tmp_path: Path, template_name: str
) -> None:
    """End-to-end happy path: stub extractor → render → write → AST validates.

    Parameterized over all three v1 templates so a regression in any one
    template's SLOT layout fails this test in isolation."""
    schema_cls = load_schema(template_name)
    params_instance = _midpoint_params(schema_cls)
    state = {
        "strategy_id": "test_gen_001",
        "template": template_name,
        "artifacts": {
            "research_proposal": {
                "hypothesis": "test hypothesis",
                "template_name": template_name,
                "regime_thesis": "test thesis",
                "suggested_param_ranges": {},
                "confidence": 0.8,
            }
        },
    }
    update = await generator_node(
        state,
        params_extractor=_make_stub_extractor(params_instance),
        generated_dir=tmp_path,
    )

    # File landed where promised.
    out_path = tmp_path / "test_gen_001.py"
    assert out_path.exists(), f"expected file at {out_path}"
    assert update["strategy_path"] == str(out_path)
    assert update["artifacts"]["generated_strategy_path"] == str(out_path)

    # Rendered source parses + AST-validates clean (the real Stage 5a
    # validator runs inside generator_node; this re-runs to confirm
    # the file on disk matches).
    rendered = out_path.read_text(encoding="utf-8")
    ast.parse(rendered, filename=str(out_path))
    validate_strategy_source(rendered, filename=str(out_path))

    # State update shape — happy path doesn't set stage / failure_reason.
    assert "stage" not in update
    assert "failure_reason" not in update
    assert update["params"] == params_instance.model_dump()

    # Agent vote recorded as pass.
    assert len(update["agent_votes"]) == 1
    vote = update["agent_votes"][0]
    assert vote["agent"] == "generator"
    assert vote["verdict"] == "pass"


async def test_generator_node_archives_on_ast_failure(tmp_path: Path) -> None:
    """If the rendered output contains a forbidden import, the generator
    must archive (not raise) — the per-strategy graph keeps running."""
    schema_cls = load_schema("mean_reversion_template")
    params_instance = _midpoint_params(schema_cls)

    # Render the template normally, then inject a forbidden import to
    # simulate a SLOT extractor that somehow produced a poisoned literal
    # (in practice the schema bounds prevent this — we're exercising the
    # archive-on-failure path defensively).
    async def poisoning_extractor(
        proposal: dict[str, Any],
        template_source: str,
        schema_cls: type[BaseModel],
    ) -> BaseModel:
        return params_instance

    # Monkey-patch the generator's render_template via a wrapper that
    # appends a forbidden import. Cleanest way to exercise the AST path
    # without contorting the schema or the templates.
    import orchestrator.agents.generator as gen_mod

    original_render = gen_mod.render_template

    def render_with_poison(template_source: str, params: dict[str, Any]) -> str:
        rendered = original_render(template_source, params)
        return rendered + "\nimport os  # synthetic poison for AST test\n"

    gen_mod.render_template = render_with_poison
    try:
        state = {
            "strategy_id": "test_gen_poison",
            "template": "mean_reversion_template",
            "artifacts": {"research_proposal": {}},
        }
        update = await generator_node(
            state,
            params_extractor=poisoning_extractor,
            generated_dir=tmp_path,
        )
    finally:
        gen_mod.render_template = original_render

    assert update["stage"] == "archived"
    assert update["failure_reason"].startswith("ast_validator:")
    assert "os" in update["failure_reason"]
    assert update["artifacts"]["ast_violations"]
    assert update["agent_votes"][0]["verdict"] == "fail"
    # File still got written even though it failed validation — that's
    # the contract: the file on disk + the violations list together
    # let the operator and the critic loop see exactly what was rejected.
    out_path = tmp_path / "test_gen_poison.py"
    assert out_path.exists()


async def test_generator_node_requires_strategy_id() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        await generator_node(
            {"template": "mean_reversion_template", "artifacts": {}},
            params_extractor=_make_stub_extractor(
                _midpoint_params(load_schema("mean_reversion_template"))
            ),
        )


async def test_generator_node_requires_template() -> None:
    with pytest.raises(ValueError, match="template"):
        await generator_node(
            {"strategy_id": "x", "artifacts": {}},
            params_extractor=_make_stub_extractor(
                _midpoint_params(load_schema("mean_reversion_template"))
            ),
        )
