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
from collections.abc import AsyncIterator

import pytest
from dotenv import load_dotenv

# Stage 11c (F1): the suite ALWAYS exercises strict-msgpack mode (the §15
# checkpoint-RCE control). LangGraph reads LANGGRAPH_STRICT_MSGPACK at IMPORT
# time, so this must be set BEFORE the orchestrator imports below (importing
# orchestrator pulls in langgraph). load_dotenv (here and in
# orchestrator/__init__.py) does NOT override an already-set env var, so this
# forced value wins; the lifespan's _assert_strict_msgpack_enabled then passes
# under test. setdefault leaves a deliberate shell override in place.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

import orchestrator.kill_subscription as _kill_subscription  # noqa: E402
import orchestrator.supervisor as _supervisor  # noqa: E402

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

# Stage 10d: force LangSmith tracing OFF for the whole test suite, regardless
# of what the operator's .env (loaded above) sets. Two reasons: (1) tests must
# never upload traces to the real LangSmith project (pollution + latency), and
# (2) it makes the graceful-degradation contract — graph runs identically with
# tracing off — the DEFAULT condition every graph-running test exercises. A
# test that wants to assert trace-config behaviour does so by building the
# config directly (test_tracing.py) or monkeypatching this back on; it never
# needs a live trace. Set AFTER load_dotenv so it wins over any .env value.
os.environ["LANGSMITH_TRACING"] = "false"

# Stage 11a (D-15): make the loop policy bulletproof, NOT order-dependent. The
# session-scoped ``event_loop_policy`` fixture below already hands pytest-asyncio
# the Selector policy for every async test, but a test that constructs its OWN
# loop (a stray ``asyncio.run``, or sync code reaching ``get_event_loop``) would
# otherwise get win32's default ``ProactorEventLoop`` — which psycopg async
# rejects (D-14; SPEC 2026-05-27 Stage 3c). Installing the policy at conftest
# IMPORT time makes the Selector loop the default for ANY loop created during
# the session, independent of collection order. No-op off win32 (Selector is
# already the default there).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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


# Module-level registries of fire-and-forget ``asyncio.Task``s in product code.
# They live at MODULE scope (so a task survives the coroutine that spawned it —
# GC protection), which means they also OUTLIVE pytest-asyncio's per-test loop.
_MODULE_TASK_REGISTRIES = (
    _kill_subscription._KILL_RESUME_TASKS,
    _supervisor._BACKGROUND_SPAWN_TASKS,
)


@pytest.fixture(autouse=True)
async def _isolate_module_task_registries() -> AsyncIterator[None]:
    """Stage 11a (D-15) — enforce per-test event-loop isolation suite-wide.

    Root cause of D-13 (and the D-14 commit's test-pollution): pytest-asyncio
    gives each test a fresh function-scoped event loop, but the module-level task
    registries above OUTLIVE that loop. A fire-and-forget task left in a registry
    by test A — created on A's now-closed loop — is then gathered/awaited by a
    later test B on B's loop, raising ``RuntimeError: ... got Future ... attached
    to a different loop``. Because pytest COLLECTION ORDER decides whether A runs
    before B, the suite's green-ness becomes order-dependent — the
    measurement-integrity bug D-15 names.

    The teardown of an async autouse fixture runs on the SAME loop the test used
    (verified for sync AND async tests), so here — before pytest-asyncio closes
    that loop — we cancel and drain any tasks the test leaked, on their OWNING
    loop, then clear the registries. After EVERY test the sets are empty, so no
    task can ever cross into the next test's loop. Collection order can no longer
    change pass/fail. Tests that drain their own tasks (e.g. test_live_wake's
    ``_drain_kill_resume_tasks``) leave nothing for this to do; it is the
    suite-wide safety net that makes the guarantee structural rather than
    per-file.
    """
    yield
    for registry in _MODULE_TASK_REGISTRIES:
        leaked = list(registry)
        registry.clear()
        for task in leaked:
            task.cancel()
        if leaked:
            # Drain on this (the owning) loop so nothing is left pending to leak
            # into the next test, and no "Task was destroyed but it is pending"
            # noise is emitted. Exceptions/cancellations are swallowed — these
            # are leaked fire-and-forget tasks being torn down, not assertions.
            await asyncio.gather(*leaked, return_exceptions=True)
