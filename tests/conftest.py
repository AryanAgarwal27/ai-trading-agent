"""Shared pytest fixtures for the ai-trading-agent test suite.

Fixtures defined directly here:
- ``event_loop_policy`` — Windows ``SelectorEventLoop`` override required
  by psycopg async (BRD §4 Python pin + SPEC §6 Stage 3c note).

Fixtures imported from ``tests/fixtures/`` (auto-discovered by pytest
once the names exist in this module — the imports below are the
"plugin" mechanism for non-conftest fixture modules under the test
root):

- ``hitl_autoapprove`` / ``hitl_autoreject`` (Stage 6i) — auto-decide
  for HITL gates so Stage 7+ tests don't block on ``interrupt()``.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

# Re-export topic-grouped fixtures from tests/fixtures/. The F401 is
# the standard pytest pattern for fixture re-export from a topic
# module — pytest discovers fixtures by name in the conftest's
# namespace, so the import alone wires them up.
from tests.fixtures.hitl import hitl_autoapprove, hitl_autoreject  # noqa: F401


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
