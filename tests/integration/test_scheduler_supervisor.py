"""Integration test for the Stage 9d supervisor lifespan wiring.

Exercises the REAL FastAPI lifespan (AsyncPostgresSaver/Store, scheduler):
asserts it sets a callable ``app.state.run_supervisor_fn`` and registers the
supervisor cron job, and tears down cleanly. Same Postgres precondition as
``test_postgres_lifecycle``; skips if the DB env is absent. CI runs this from
9g (D-7); excluded from the unit job.

The companion unit tests (registration shape, job-invokes-runner, exception
swallow) live in ``tests/unit/test_scheduler_supervisor.py``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from orchestrator.scheduler import SUPERVISOR_CRON_JOB_ID

pytestmark = pytest.mark.integration


async def test_lifespan_wires_supervisor_runner() -> None:
    """The FastAPI lifespan sets a callable ``app.state.run_supervisor_fn`` and
    registers the supervisor cron job, and tears down cleanly.

    NOTE: we do NOT assert ``not scheduler.running`` post-teardown — the
    existing ``shutdown_scheduler`` deliberately uses ``shutdown(wait=False)``
    (so an in-flight wake httpx call can't stall FastAPI shutdown), and
    AsyncIOScheduler's ``wait=False`` defers the stop to the event loop rather
    than flipping ``running`` synchronously. That is a pre-existing, deliberate
    property — orthogonal to the 9d wiring this test covers. The clean exit of
    the lifespan context (no teardown exception) is the "clean shutdown" signal.
    """
    load_dotenv()
    if not all(os.environ.get(v) for v in ("LANGGRAPH_CHECKPOINT_URI", "LANGGRAPH_STORE_URI")):
        pytest.skip("integration DB env not set")

    from orchestrator.main import app

    async with app.router.lifespan_context(app):
        run_fn: Any = app.state.run_supervisor_fn
        assert callable(run_fn)
        job = app.state.scheduler.get_job(SUPERVISOR_CRON_JOB_ID)
        assert job is not None
        assert isinstance(job.trigger, CronTrigger)
        assert job.id == SUPERVISOR_CRON_JOB_ID
