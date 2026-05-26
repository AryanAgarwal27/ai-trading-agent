"""Shared pytest fixtures for the ai-trading-agent test suite.

The only fixture here today is the event-loop policy override needed on
Windows: psycopg's async mode refuses Python's default `ProactorEventLoop`
on Windows and demands a `SelectorEventLoop`. Production (Linux VPS per
BRD §3) is unaffected; uvicorn already auto-selects the selector loop on
Windows at runtime, so this only matters for pytest's directly-driven
async tests.
"""

from __future__ import annotations

import asyncio
import sys

import pytest


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
