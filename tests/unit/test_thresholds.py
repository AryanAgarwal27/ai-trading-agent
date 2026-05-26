"""Stage 4a unit tests — ``orchestrator.gates.thresholds`` is the single source of truth.

Two responsibilities:

1. **Concrete value pinning.** Each threshold from BRD §10 has the exact
   documented value. A casual edit to ``thresholds.py`` that changes a
   number will fail this test, forcing the editor to also update the BRD
   diff (which is BRD §0's "propose a BRD diff first" rule).

2. **Single-source enforcement.** No other Python module under
   ``orchestrator/`` defines a constant with the same name as one of the
   threshold symbols. Catches the common drift of copy-pasting a literal
   into a gate node — BRD §10 calls this out by name as a non-negotiable.

The single-source check is grep-based by design: a unit test that just
imports symbols only catches re-imports, not literal copies. Reading the
source tree for ``<NAME>\\s*=`` finds both.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator.gates import thresholds

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_ROOT = REPO_ROOT / "orchestrator"
THRESHOLDS_FILE = ORCHESTRATOR_ROOT / "gates" / "thresholds.py"


# Verbatim BRD §10 expected values. Sourced from BRD.md §10 in the same
# commit that authored thresholds.py — if BRD §10 changes, both this test
# AND thresholds.py must change in lockstep, per SPEC §4.4 rule 3.
EXPECTED: dict[str, int | float] = {
    # Backtest hard gate (in-sample)
    "MIN_TRADES_IS": 150,
    "MIN_OOS_TRADES": 30,
    "MIN_SHARPE_IS": 1.5,
    "MIN_PROFIT_FACTOR_IS": 1.5,
    "MAX_DRAWDOWN_IS": 0.20,
    # OOS / walk-forward gate
    "MIN_OOS_RATIO": 0.6,
    "MIN_OOS_SHARPE_PER_FOLD": 0.0,
    "MIN_OOS_PROFIT_FACTOR": 1.2,
    "MAX_OOS_DRAWDOWN": 0.25,
    # Robustness gate
    "MIN_MC_5TH_PERCENTILE_RETURN": 0.0,
    "MIN_REGIMES_PASSED": 2,
    "MAX_FEE_STRESS_DEGRADATION_2X": 0.40,
    "MAX_FEE_STRESS_DEGRADATION_3X": 0.60,
    # Paper gate (advisory)
    "MIN_PAPER_DAYS": 30,
    "MAX_PAPER_VS_BACKTEST_KS_PVALUE": 0.05,
    "MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION": 0.30,
    # Live monitoring (auto-pause)
    "KILL_SWITCH_DRAWDOWN": 0.12,
    "KILL_SWITCH_CONSECUTIVE_LOSSES": 10,
    "DAILY_LOSS_LIMIT_PCT": 0.03,
    "MAX_OPEN_TRADES": 4,
    "MAX_POSITION_CONCENTRATION": 0.30,
    # Live capital
    "LIVE_CAPITAL_CAP_USD": 500,
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_threshold_value_matches_brd_section_10(name: str, expected: int | float) -> None:
    actual = getattr(thresholds, name)
    assert actual == expected, (
        f"{name}: thresholds.py has {actual!r}, BRD §10 says {expected!r}. "
        "If BRD §10 changed, propose the BRD diff first (BRD §0) and update "
        "both files in lockstep (SPEC §4.4 rule 3)."
    )
    # Type pinning: int constants must be ints, fraction constants must be
    # floats. Drift here (e.g. MIN_SHARPE_IS becoming int 2) would silently
    # cast in some comparisons but not others.
    assert isinstance(
        actual, type(expected)
    ), f"{name}: expected type {type(expected).__name__}, got {type(actual).__name__}"


def test_all_brd_thresholds_are_defined() -> None:
    """Catches a deletion of a threshold from thresholds.py."""
    missing = [name for name in EXPECTED if not hasattr(thresholds, name)]
    assert not missing, (
        f"thresholds.py is missing BRD §10 constants: {missing}. "
        "Removing a threshold requires a BRD §10 edit first (BRD §0)."
    )


def test_no_extra_undocumented_thresholds() -> None:
    """Catches drift the OTHER way — a new constant added without BRD coverage.

    Iterates ``thresholds`` module dunder-safe and flags any uppercase
    symbol that isn't in :data:`EXPECTED`. New thresholds must land in
    BRD §10 first, then here.
    """
    public_consts = {
        name for name in vars(thresholds) if name.isupper() and not name.startswith("_")
    }
    extras = public_consts - set(EXPECTED)
    assert not extras, (
        f"thresholds.py has constants not covered by EXPECTED: {extras}. "
        "Add them to BRD §10 first, then to this test."
    )


# Names that legitimately appear in non-threshold contexts in the BRD/code
# even though they share a token with a threshold name. None today; the list
# is here to make false positives easy to allowlist if Stage 5+ adds one.
_GREP_ALLOWLIST_FILES: set[Path] = set()


def test_no_other_module_defines_a_threshold_name() -> None:
    """Single-source-of-truth grep: BRD §10 is unambiguous about this.

    Walks every ``.py`` under ``orchestrator/`` (excluding ``thresholds.py``
    itself) and asserts no file contains ``<NAME>\\s*=`` for any threshold
    name. Catches copy-paste drift the import-only check would miss.
    """
    bare_assignment = re.compile(
        r"^\s*(?:" + "|".join(re.escape(n) for n in EXPECTED) + r")\s*[:=]",
        re.MULTILINE,
    )

    offenders: list[tuple[Path, str]] = []
    for py in ORCHESTRATOR_ROOT.rglob("*.py"):
        if py.resolve() == THRESHOLDS_FILE.resolve():
            continue
        if py in _GREP_ALLOWLIST_FILES:
            continue
        source = py.read_text(encoding="utf-8")
        for match in bare_assignment.finditer(source):
            offenders.append((py.relative_to(REPO_ROOT), match.group(0).strip()))

    assert not offenders, (
        "Threshold names assigned outside thresholds.py — violates BRD §10 "
        "single-source rule:\n" + "\n".join(f"  {p}: {line!r}" for p, line in offenders)
    )
