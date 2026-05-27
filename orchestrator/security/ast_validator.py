"""AST allowlist validator for LLM-rendered strategy files (BRD §8 rule 4).

This validator runs against every file the generator writes into
``strategy_templates/_generated/`` before that file is ever handed to
Freqtrade. It is the last automated line of defence between an LLM-rendered
``.py`` and ``subprocess.run([..., 'freqtrade', 'backtesting', ...])``.

Rationale — intentionally stricter than BRD §8 step 4
-----------------------------------------------------

BRD §8 step 4 names a minimum surface to block (``import os``,
``import subprocess``, network modules, ``eval``, ``exec``, ``__import__``,
``compile``) and adds: "Reject also if any imported module is not in an
allowlist." This v1 implementation goes further on three axes; each
widening is deliberate and reversible only with explicit evidence that a
template legitimately needs the wider surface.

1. **Allowlist, not blocklist, for imports.** Anything not explicitly in
   :data:`ALLOWED_TOP_LEVEL_IMPORTS` is rejected — including stdlib modules
   the BRD does not name (``pathlib``, ``json``, ``pickle``, ``shutil``,
   ``socket``, ``http``, …). Blocklists rot; allowlists fail closed when a
   new stdlib module ships or a transitive dep adds a new attack surface.
   The cost is that new template needs surface a deliberate edit here.

2. **Bans ``open`` and ``input``** in addition to the BRD-named names.
   ``open`` is a filesystem escape vector — a "pure-compute" strategy has
   no business reading or writing arbitrary paths. ``input`` would block
   the Freqtrade worker waiting on stdin and is never legitimate inside
   strategy code.

3. **Bans ``importlib``** (and the broader closure of "things that can
   bring in a forbidden module at runtime"). The BRD names ``__import__``
   but ``importlib.import_module`` is the obvious bypass; both are blocked.

The :func:`validate_strategy_source` entry point parses with ``ast.parse``
and walks once, collecting *all* violations into a single
:class:`ASTValidationError` so the operator and the critic-loop logs see
every problem in one pass instead of fixing them one at a time.
"""

from __future__ import annotations

import ast

# Top-level module names that may appear in `import X` or
# `from X import ...`. Submodule access (e.g. `talib.abstract`,
# `freqtrade.vendor.qtpylib`) is fine — only the top-level name is checked.
ALLOWED_TOP_LEVEL_IMPORTS: frozenset[str] = frozenset(
    {
        # Strategy framework + ecosystem
        "freqtrade",
        # TA / numerical libraries used by templates
        "talib",
        "pandas",
        "numpy",
        # Stdlib — minimal surface, additions are deliberate
        "__future__",
        "typing",
        "dataclasses",
        "datetime",
        "math",
        "functools",
    }
)


# Names that must never appear in the AST — neither as a call, an
# assignment target, nor as the trailing attribute of an attribute chain
# (`builtins.eval`, `o.system` after `import os as o`, etc.).
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
    }
)


class ASTValidationError(Exception):
    """Raised when a rendered strategy file violates the AST allowlist.

    Carries the full list of violations (not just the first) so callers
    log a single rejection event with every problem visible at once.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations: list[str] = list(violations)
        super().__init__(str(self))

    def __str__(self) -> str:
        if not self.violations:
            return "AST validation failed (no violations recorded)"
        return "AST validation failed:\n  - " + "\n  - ".join(self.violations)


class _StrategyASTVisitor(ast.NodeVisitor):
    """Walks the AST and collects every allowlist violation it finds."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".", 1)[0]
            if top not in ALLOWED_TOP_LEVEL_IMPORTS:
                self.violations.append(
                    f"line {node.lineno}: disallowed import '{alias.name}' "
                    f"(top-level '{top}' not in allowlist)"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative imports — `from . import x`, `from ..pkg import x` —
        # are never allowed in generator output. Generated strategies live
        # in `strategy_templates/_generated/` and have no sibling package
        # to legitimately import from.
        if node.level and node.level > 0:
            target = node.module if node.module else "<relative>"
            self.violations.append(
                f"line {node.lineno}: relative import "
                f"'from {'.' * node.level}{node.module or ''} import ...' "
                f"is not allowed (level={node.level}, target={target})"
            )
            return

        if node.module is None:
            # Defensive: an absolute `from None import ...` is a syntax error
            # so we shouldn't reach this branch, but if we do, fail closed.
            self.violations.append(
                f"line {node.lineno}: import with no module name is not allowed"
            )
            return

        top = node.module.split(".", 1)[0]
        if top not in ALLOWED_TOP_LEVEL_IMPORTS:
            self.violations.append(
                f"line {node.lineno}: disallowed 'from {node.module} import ...' "
                f"(top-level '{top}' not in allowlist)"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Catches direct call (`eval(...)`), reference (`f = open`), and
        # any other context that resolves a forbidden builtin by name.
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(
                f"line {node.lineno}: forbidden name '{node.id}'"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Catches attribute-chain escapes like `builtins.eval(...)` or
        # `some_alias.exec(...)` even when the receiver is allowlisted.
        if node.attr in FORBIDDEN_NAMES:
            self.violations.append(
                f"line {node.lineno}: forbidden attribute access '.{node.attr}'"
            )
        self.generic_visit(node)


def validate_strategy_source(
    source: str, *, filename: str = "<generated>"
) -> None:
    """Validate ``source`` against the strategy allowlist.

    Parses with ``ast.parse`` and walks the tree once, collecting every
    violation. Raises :class:`ASTValidationError` with all violations on
    failure (or wraps :class:`SyntaxError` as one). Returns ``None`` on
    success — generators should treat a clean return as "safe to write".
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ASTValidationError(
            [f"syntax error in {filename}: {exc.msg} (line {exc.lineno})"]
        ) from exc

    visitor = _StrategyASTVisitor()
    visitor.visit(tree)
    if visitor.violations:
        raise ASTValidationError(visitor.violations)
