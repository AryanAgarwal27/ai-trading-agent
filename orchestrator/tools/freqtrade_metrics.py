"""Pure metrics derived from Freqtrade ``/trades`` data.

Shared, side-effect-free helpers over the list-of-trade-dicts shape Freqtrade's
REST ``/api/v1/trades`` returns. Promoted here (Stage 11e, D-4) from a private
``paper.py`` definition that had been copied into ``live.py`` and imported by
``scheduler.py`` — three call sites across the paper monitor, the live reviewer,
and the out-of-band kill-switch poll. One canonical definition removes the
duplication.
"""

from __future__ import annotations

from typing import Any


def trailing_losses(trades: list[dict[str, Any]]) -> int:
    """Count the trailing run of losing closed trades (most-recent-first).

    Freqtrade ``/trades`` returns trades oldest-first, so we walk in
    reverse. A non-dict or missing ``profit_ratio`` ends the run.
    """
    count = 0
    for t in reversed(trades):
        if not isinstance(t, dict):
            break
        pr = t.get("profit_ratio")
        if pr is None:
            break
        if float(pr) < 0:
            count += 1
        else:
            break
    return count
