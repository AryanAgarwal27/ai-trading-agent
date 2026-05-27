"""Tests for orchestrator.security.ast_validator (BRD §8 rule 4).

Negative cases cover the BRD-named forbidden modules and names plus the
v1 hardening additions (``open``, ``input``, ``importlib``, relative
imports, attribute-chain escapes). Positive cases include the shipped
mean_reversion template as the regression canary that the allowlist still
accepts what real templates need.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.security.ast_validator import (
    ALLOWED_TOP_LEVEL_IMPORTS,
    FORBIDDEN_NAMES,
    ASTValidationError,
    validate_strategy_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MEAN_REVERSION_TEMPLATE = (
    REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"
)


# ─── Positive cases ─────────────────────────────────────────────────────────


def test_mean_reversion_template_validates_clean() -> None:
    """Regression canary: the shipped template must pass. If this fails
    after an allowlist tweak, the tweak removed something a real template
    needs."""
    source = MEAN_REVERSION_TEMPLATE.read_text(encoding="utf-8")
    validate_strategy_source(source, filename=str(MEAN_REVERSION_TEMPLATE))


def test_minimal_allowlisted_imports_pass() -> None:
    source = (
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "def foo(df):\n"
        "    return df\n"
    )
    validate_strategy_source(source)


def test_freqtrade_and_talib_submodule_imports_pass() -> None:
    source = (
        "import talib.abstract as ta\n"
        "from freqtrade.strategy import IStrategy\n"
        "from freqtrade.vendor.qtpylib import indicators as qtpylib\n"
    )
    validate_strategy_source(source)


def test_constants_are_frozensets() -> None:
    """Frozen by design — runtime code must not mutate the allowlist."""
    assert isinstance(ALLOWED_TOP_LEVEL_IMPORTS, frozenset)
    assert isinstance(FORBIDDEN_NAMES, frozenset)


# ─── Forbidden imports — parameterized over BRD + hardening list ────────────


FORBIDDEN_IMPORT_CASES = [
    "os",
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "http.client",
    "smtplib",
    "ctypes",
    "importlib",
]


@pytest.mark.parametrize("mod", FORBIDDEN_IMPORT_CASES)
def test_rejects_import_statement(mod: str) -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(f"import {mod}\n")
    msg = str(exc_info.value)
    top = mod.split(".", 1)[0]
    assert top in msg
    assert "allowlist" in msg


@pytest.mark.parametrize("mod", FORBIDDEN_IMPORT_CASES)
def test_rejects_from_import_statement(mod: str) -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(f"from {mod} import something\n")
    msg = str(exc_info.value)
    top = mod.split(".", 1)[0]
    assert top in msg


def test_rejects_aliased_forbidden_import() -> None:
    """`import os as o` — the alias does not bypass the top-level check."""
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source("import os as o\n")
    assert "os" in str(exc_info.value)


# ─── Forbidden names (eval/exec/compile/__import__/open/input) ──────────────


FORBIDDEN_NAME_CASES = ["eval", "exec", "compile", "__import__", "open", "input"]


@pytest.mark.parametrize("name", FORBIDDEN_NAME_CASES)
def test_rejects_forbidden_name_call(name: str) -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(f"x = {name}('arg')\n")
    assert name in str(exc_info.value)


@pytest.mark.parametrize("name", FORBIDDEN_NAME_CASES)
def test_rejects_forbidden_name_reference(name: str) -> None:
    """`f = open` — deferred-execution escape; reference alone is enough."""
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(f"f = {name}\n")
    assert name in str(exc_info.value)


# ─── Attribute-chain escapes ────────────────────────────────────────────────


def test_rejects_forbidden_attribute_on_allowlisted_module() -> None:
    """`functools.eval` — contrived but exercises the attribute visitor.
    Even when the receiver is allowlisted, a forbidden attribute name on
    the chain is flagged."""
    source = (
        "import functools\n"
        "x = functools.eval\n"
    )
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(source)
    assert "eval" in str(exc_info.value)


def test_rejects_os_system_after_aliased_import() -> None:
    """`import os as o; o.system(...)` — the aliased import is caught at
    import time so the attribute chain never gets to run. We assert the
    rejection mentions ``os`` (the import-level violation)."""
    source = (
        "import os as o\n"
        "o.system('rm -rf /')\n"
    )
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(source)
    assert "os" in str(exc_info.value)


# ─── Relative imports ───────────────────────────────────────────────────────


def test_rejects_bare_dotted_relative_import() -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source("from . import sibling\n")
    assert "relative" in str(exc_info.value).lower()


def test_rejects_dotted_relative_import_with_module() -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source("from ..parent_pkg import sibling\n")
    assert "relative" in str(exc_info.value).lower()


# ─── Collect-all behavior + syntax errors ───────────────────────────────────


def test_collects_all_violations_not_just_first() -> None:
    source = (
        "import os\n"
        "import subprocess\n"
        "x = eval('1+1')\n"
        "f = open\n"
    )
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source(source)
    violations = exc_info.value.violations
    assert len(violations) >= 4, (
        f"expected ≥4 violations, got {len(violations)}: {violations}"
    )


def test_syntax_error_is_wrapped_as_validation_error() -> None:
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source("def broken(:\n")
    assert "syntax error" in str(exc_info.value).lower()


def test_violations_attribute_is_a_list() -> None:
    """Callers iterate `.violations` directly for structured logging —
    keep it a plain list, not a tuple or frozen container."""
    with pytest.raises(ASTValidationError) as exc_info:
        validate_strategy_source("import os\n")
    assert isinstance(exc_info.value.violations, list)
    assert len(exc_info.value.violations) == 1
