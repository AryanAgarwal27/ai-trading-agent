"""Paper-monitor test fixtures (Stage 7d).

``paper_monitor_stub`` returns a factory that builds a drop-in
replacement for :func:`orchestrator.agents.monitors.run_paper_monitor`.
The 7e paper subgraph node accepts a ``paper_monitor_fn`` injection
seam (same pattern as Stage 4's ``risk_analyst_fn`` and Stage 5's
``critic_fn``); tests pass ``paper_monitor_stub("advance")`` so the
node routes deterministically without a real Haiku call.

Usage in a 7e/Stage-8 test::

    async def test_paper_advances(paper_monitor_stub):
        node_out = await paper_monitor_node(
            state, config,
            paper_monitor_fn=paper_monitor_stub("advance", confidence=0.9),
        )
        assert node_out.goto == "live_gate"
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import pytest

from orchestrator.agents.monitors import PaperMonitorContext, PaperMonitorVerdict

PaperMonitorFn = Callable[[PaperMonitorContext], Any]


@pytest.fixture
def paper_monitor_stub() -> Callable[..., PaperMonitorFn]:
    """Return a factory building a stub ``run_paper_monitor`` coroutine.

    The factory signature mirrors the fields of
    :class:`PaperMonitorVerdict` so a test can pin every field it cares
    about and accept defaults for the rest.
    """

    def _make(
        decision: Literal["rearm", "advance", "kill"] = "rearm",
        *,
        confidence: float = 0.7,
        primary_observation: str = "stub observation",
        rationale: str = "stub rationale",
    ) -> PaperMonitorFn:
        async def _stub(_ctx: PaperMonitorContext) -> PaperMonitorVerdict:
            return PaperMonitorVerdict(
                decision=decision,
                primary_observation=primary_observation,
                rationale=rationale,
                confidence=confidence,
            )

        return _stub

    return _make
