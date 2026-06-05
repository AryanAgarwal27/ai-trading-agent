"""Unit test for the production entry point (Stage 10d closure, BRD §3).

``orchestrator.main.main()`` must install a ``SelectorEventLoop`` policy on
Windows before serving — otherwise the lifespan's ``AsyncPostgresSaver``
crashes under uvicorn's default ``ProactorEventLoop`` (SPEC 2026-05-27
Stage 3c). uvicorn is stubbed here so no server actually boots; the test
only asserts the loop-policy install, which is the whole point of owning
the loop.

Platform-agnostic: on win32 (the operator's machine) it asserts the
selector policy was installed; on Linux (CI) it asserts the win32 branch
was skipped — and never references ``WindowsSelectorEventLoopPolicy``
(which does not exist off-Windows) on that branch.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest


class _FakeServer:
    """No-op stand-in for ``uvicorn.Server`` — ``serve`` returns immediately."""

    def __init__(self, config: Any) -> None:
        self.config = config

    async def serve(self) -> None:
        return None


def test_main_installs_selector_loop_policy_per_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from orchestrator.main import main

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        asyncio, "set_event_loop_policy", lambda policy: captured.__setitem__("policy", policy)
    )
    # Stub asyncio.run so main() never creates/closes a REAL event loop — a real
    # asyncio.run here would close the loop and pollute loop state for later
    # tests (the same event-loop-pollution family as D-13). ``coro.close()``
    # disposes the un-awaited serve() coroutine cleanly (no "never awaited"
    # warning, which filterwarnings=error would otherwise fail on).
    monkeypatch.setattr(asyncio, "run", lambda coro: coro.close())
    # Stub uvicorn so main() builds a fake server (no real bind).
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: object())
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)

    main()

    if sys.platform == "win32":
        # The whole fix: a SelectorEventLoop policy is installed before serving.
        assert isinstance(captured.get("policy"), asyncio.WindowsSelectorEventLoopPolicy)
    else:
        # Linux: the selector loop is already the default — no override, and we
        # must not reference the win-only policy class on this branch.
        assert "policy" not in captured
