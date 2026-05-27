"""Tests for template SLOT/schema alignment + AST cleanliness (BRD §8 rules 2-4).

For each shipped template:
  1. SLOT names and schema field names must agree byte-for-byte — no drift.
  2. The template's literal default values must satisfy its co-located
     Pydantic schema (Stage 3/5 smoke-test runs the un-rendered template).
  3. Rendering the template with a synthetic Pydantic-valid param set must
     produce source that parses cleanly AND passes Stage 5a's AST validator.

The substitution helper used here is intentionally minimal — a regex over
SLOT lines, just enough to detect schema/slot drift in CI. The full
generator (renderer + ``ChatAnthropic.with_structured_output`` + cache-busted
``freqtrade lookahead-analysis``) lands in Stage 5c.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from orchestrator.security.ast_validator import validate_strategy_source

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"

# (template_module_stem, schema_class_name) for every template that ships
# with a generator-compatible schema in v1.
TEMPLATE_SPECS: list[tuple[str, str]] = [
    ("mean_reversion_template", "MeanReversionParams"),
    ("freqai_classifier_template", "FreqaiClassifierParams"),
    ("freqai_regressor_template", "FreqaiRegressorParams"),
]


# Matches a SLOT line: `<name>: <type> = <literal>  # SLOT: <slot_name> (...)`.
# The literal must not contain `#` (no inline comments inside the value).
SLOT_LINE_RE = re.compile(
    r"^(?P<lead>\s*)"
    r"(?P<name>\w+)\s*:\s*\w+\s*=\s*"
    r"(?P<value>[^#]+?)\s*"
    r"#\s*SLOT:\s*(?P<slot>\w+)\b"
)


def _load_schema(stem: str, class_name: str) -> type[BaseModel]:
    """Load a schema module by file path.

    Avoids requiring ``strategy_templates/__init__.py`` (which would also
    mark the template ``.py`` files as importable, but they ``import
    freqtrade`` which isn't installed in the test venv)."""
    path = TEMPLATES_DIR / f"{stem}_schema.py"
    spec = importlib.util.spec_from_file_location(f"_test_schema_{stem}", path)
    assert spec is not None and spec.loader is not None, (
        f"Could not build module spec for {path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _extract_slots(source: str) -> dict[str, str]:
    """Map SLOT name → literal value string from template source."""
    out: dict[str, str] = {}
    for line in source.splitlines():
        m = SLOT_LINE_RE.match(line)
        if m:
            out[m.group("slot")] = m.group("value").strip()
    return out


def _midpoint_value(field: Any) -> int | float | None:
    """Mid-range value for a Pydantic field whose constraints carry ge + le.

    Returns ``None`` if the field type or constraints are not int/float with
    both bounds set — the synthetic-params helper turns that into a clear
    test failure with the field name attached.
    """
    ge = le = None
    for constraint in field.metadata:
        if hasattr(constraint, "ge"):
            ge = constraint.ge
        if hasattr(constraint, "le"):
            le = constraint.le
    if ge is None or le is None:
        return None
    if field.annotation is int:
        return int((ge + le) // 2)
    if field.annotation is float:
        return (ge + le) / 2.0
    return None


def _synthetic_params(schema_cls: type[BaseModel]) -> dict[str, Any]:
    """Build a mid-range, Pydantic-valid param dict for the schema."""
    raw: dict[str, Any] = {}
    for name, field in schema_cls.model_fields.items():
        value = _midpoint_value(field)
        assert value is not None, (
            f"Cannot generate synthetic mid-range value for "
            f"{schema_cls.__name__}.{name} "
            f"(type={field.annotation}, metadata={field.metadata})"
        )
        raw[name] = value
    # Calling the model validates the dict against the schema — extra=forbid,
    # range checks, etc.
    return schema_cls(**raw).model_dump()


def _render(source: str, params: dict[str, Any]) -> str:
    """Naive SLOT substituter; minimal-viable generator stub for the test.

    For each SLOT line whose slot name is in ``params``, replaces the RHS
    literal with ``repr(params[slot])`` and keeps the trailing ``# SLOT: ...``
    comment intact. Other lines pass through unchanged.
    """
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        m = SLOT_LINE_RE.match(line)
        if m and m.group("slot") in params:
            slot = m.group("slot")
            new_value = repr(params[slot])
            head, _ = line.split("=", 1)
            comment_at = line.index("#")
            tail = line[comment_at:]
            line = f"{head}= {new_value}  {tail}"
        out.append(line)
    return "".join(out)


@pytest.fixture(params=TEMPLATE_SPECS, ids=lambda spec: spec[0])
def template_pair(request: pytest.FixtureRequest) -> tuple[Path, type[BaseModel]]:
    stem, class_name = request.param
    template_path = TEMPLATES_DIR / f"{stem}.py"
    schema_cls = _load_schema(stem, class_name)
    return template_path, schema_cls


# ─── Alignment / contract tests ────────────────────────────────────────────


def test_slot_names_match_schema_fields(
    template_pair: tuple[Path, type[BaseModel]],
) -> None:
    """SLOT/schema drift detector — the contract from BRD §8 rule 3."""
    template_path, schema_cls = template_pair
    source = template_path.read_text(encoding="utf-8")
    slot_names = set(_extract_slots(source).keys())
    schema_names = set(schema_cls.model_fields.keys())
    extra_slots = slot_names - schema_names
    extra_fields = schema_names - slot_names
    assert not extra_slots, (
        f"{template_path.name}: SLOTs without matching schema field: {extra_slots}"
    )
    assert not extra_fields, (
        f"{template_path.name}: schema fields without matching SLOT: {extra_fields}"
    )


def test_template_defaults_satisfy_schema(
    template_pair: tuple[Path, type[BaseModel]],
) -> None:
    """The literal defaults in each SLOT line must validate against the schema.

    Required so the un-rendered template is itself a runnable strategy (Stage
    3 / Stage 5 smoke tests backtest the un-rendered file directly)."""
    template_path, schema_cls = template_pair
    source = template_path.read_text(encoding="utf-8")
    defaults = {
        name: ast.literal_eval(value)
        for name, value in _extract_slots(source).items()
    }
    schema_cls(**defaults)  # raises ValidationError on drift


# ─── AST validation of rendered output ─────────────────────────────────────


def test_rendered_template_passes_ast_validator(
    template_pair: tuple[Path, type[BaseModel]],
) -> None:
    """A render with mid-range synthetic params must:
       (a) parse as Python, and
       (b) pass Stage 5a's AST allowlist validator.

    Catches: a template introducing a forbidden import, a SLOT line whose
    type doesn't match the schema (producing a malformed literal under
    substitution), or any future widening of the template that would slip
    past the allowlist."""
    template_path, schema_cls = template_pair
    source = template_path.read_text(encoding="utf-8")
    params = _synthetic_params(schema_cls)
    rendered = _render(source, params)
    # (a) Parseable.
    ast.parse(rendered, filename=str(template_path))
    # (b) Allowlist-clean.
    validate_strategy_source(rendered, filename=str(template_path))


def test_rendered_template_actually_substituted(
    template_pair: tuple[Path, type[BaseModel]],
) -> None:
    """Sanity check on the substitution helper itself — if every slot value
    happens to equal the template's default the test would pass trivially.
    Verify at least one slot's rendered literal differs from the default."""
    template_path, schema_cls = template_pair
    source = template_path.read_text(encoding="utf-8")
    defaults = {
        name: ast.literal_eval(value)
        for name, value in _extract_slots(source).items()
    }
    params = _synthetic_params(schema_cls)
    rendered = _render(source, params)
    rendered_defaults = {
        name: ast.literal_eval(value)
        for name, value in _extract_slots(rendered).items()
    }
    assert rendered_defaults == params, (
        f"{template_path.name}: rendered SLOT values "
        f"{rendered_defaults} != requested params {params}"
    )
    differing = [k for k in params if params[k] != defaults.get(k)]
    assert differing, (
        f"{template_path.name}: synthetic mid-range params accidentally "
        f"match every default — substitution helper is untested for this template."
    )
