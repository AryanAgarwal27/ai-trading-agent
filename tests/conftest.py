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
import os
import sys

import pytest
from dotenv import load_dotenv

# Load .env once at collection time so integration tests pick up
# DATABASE_URL / REDIS_URL / OPERATOR_TOKEN / BINANCE_PAPER_* without
# requiring the operator to manually export them per shell. The
# load is a no-op if .env is absent (e.g. CI with secrets injected
# through the environment directly). Override semantics are NOT
# enabled — env vars set in the shell win over .env values, matching
# uvicorn's startup behaviour in orchestrator/main.py.
load_dotenv()

# Stage 7f: force the APScheduler jobstore to in-memory for ALL tests.
# Any test that drives the FastAPI lifespan starts the scheduler; with
# the production SQLAlchemyJobStore that would (a) create/read the
# apscheduler_jobs table in the app DB and (b) risk loading a leaked
# per-thread wake job left by a real uvicorn run, which could then fire
# an httpx POST mid-test. The memory store isolates the suite from the
# production jobstore entirely. Production (uvicorn) never imports
# conftest, so it keeps the SQLAlchemyJobStore default.
os.environ["AIT_SCHEDULER_JOBSTORE"] = "memory"

# Re-export topic-grouped fixtures from tests/fixtures/. The F401 is
# the standard pytest pattern for fixture re-export from a topic
# module — pytest discovers fixtures by name in the conftest's
# namespace, so the import alone wires them up.
from tests.fixtures.hitl import hitl_autoapprove, hitl_autoreject  # noqa: E402, F401
from tests.fixtures.monitors import paper_monitor_stub  # noqa: E402, F401


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
