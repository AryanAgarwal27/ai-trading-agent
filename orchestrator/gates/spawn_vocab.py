"""Shared spawn-vocabulary guards (D-16, BRD §21.1.1).

Single source of truth for "which templates / pairs may a strategy spawn
name". BOTH the supervisor's ``aspawn_strategy`` (orchestrator/supervisor.py)
AND the manual-injection endpoint (BRD §21.2, ``POST /strategies/validate``)
validate against THIS module, so the two can never drift.

The two checks are exactly the ones D-16 shipped inline in ``aspawn_strategy``:
- ``template`` (when not None) must be a shipped template
  (:data:`orchestrator.agents.generator.SHIPPED_TEMPLATES`, BRD §8.1).
- ``pairs`` (when not None) must all be inside the SPEC §1 Q2 universe
  (:data:`PAIR_UNIVERSE`).

``template=None`` / ``pairs=None`` are the normal "the researcher chooses the
template / the default universe applies" case and PASS the check — they are not
a vocabulary violation. The caller decides what a violation maps to: the
supervisor returns a no-write refusal dict; the endpoint raises HTTP 422.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orchestrator.agents.generator import SHIPPED_TEMPLATES

# SPEC §1 Q2 — the v1 pair universe. This is the canonical definition; the
# spawn-vocabulary whitelist for pairs is derived from it so the two never drift.
# A pair outside this set has no cached OHLCV / Binance whitelist entry.
DEFAULT_PAIRS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")
PAIR_UNIVERSE: frozenset[str] = frozenset(DEFAULT_PAIRS)


@dataclass(frozen=True, slots=True)
class VocabError:
    """A spawn-vocabulary violation. ``reason`` is the machine-readable code
    (``"unknown_template"`` / ``"pairs_outside_universe"``); ``detail`` carries
    the offending value(s) + the allowed set, ready to merge into a refusal dict
    or an HTTP error body."""

    reason: str
    detail: dict[str, Any]


def check_spawn_vocabulary(
    template: str | None,
    pairs: list[str] | None,
) -> VocabError | None:
    """Validate a spawn's ``template`` + ``pairs`` against the shipped vocabulary.

    Returns ``None`` when the action is in-vocabulary (including the all-``None``
    case), or a :class:`VocabError` describing the FIRST violation found
    (template checked before pairs, matching the historic D-16 order). Pure — no
    logging, no I/O, no side effects; the caller maps the result to its own
    error surface.
    """
    if template is not None and template not in SHIPPED_TEMPLATES:
        return VocabError(
            reason="unknown_template",
            detail={"template": template, "allowed_templates": sorted(SHIPPED_TEMPLATES)},
        )
    if pairs is not None:
        invalid_pairs = [p for p in pairs if p not in PAIR_UNIVERSE]
        if invalid_pairs:
            return VocabError(
                reason="pairs_outside_universe",
                detail={"invalid_pairs": invalid_pairs, "allowed_pairs": sorted(PAIR_UNIVERSE)},
            )
    return None
