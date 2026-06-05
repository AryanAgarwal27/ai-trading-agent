"""Unit test for the production entry point (Stage 10d closure, BRD §3).

``orchestrator.main.main()`` must install a ``SelectorEventLoop`` policy on
Windows before serving — otherwise the lifespan's ``AsyncPostgresSaver``
crashes under uvicorn's default ``ProactorEventLoop`` (SPEC 2026-05-27
Stage 3c). The policy install is the whole point of owning the loop.

Stage 11a (D-15): this test now exercises the EXTRACTED
``_install_selector_loop_policy_on_win32`` helper directly, instead of driving
``main()`` and stubbing ``asyncio.run``. The earlier version called the real
``asyncio.run`` inside ``main()``, which created+closed an event loop and
polluted an unrelated later test's loop state (purely as a function of pytest
collection order) — the D-14 instance of the D-15 measurement-integrity bug.
Testing the pure helper means this test NEVER creates a real loop, so it cannot
pollute siblings — a structural fix that subsumes the old ``asyncio.run`` stub.

Platform-agnostic: on win32 (the operator's machine) it asserts the selector
policy was installed; on Linux (CI) it asserts the win32 branch was skipped —
and never references ``WindowsSelectorEventLoopPolicy`` (which does not exist
off-Windows) on that branch.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest


def test_install_selector_loop_policy_per_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.main import _install_selector_loop_policy_on_win32

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        asyncio, "set_event_loop_policy", lambda policy: captured.__setitem__("policy", policy)
    )

    _install_selector_loop_policy_on_win32()

    if sys.platform == "win32":
        # The whole fix: a SelectorEventLoop policy is installed before serving.
        assert isinstance(captured.get("policy"), asyncio.WindowsSelectorEventLoopPolicy)
    else:
        # Linux: the selector loop is already the default — no override, and we
        # must not reference the win-only policy class on this branch.
        assert "policy" not in captured
