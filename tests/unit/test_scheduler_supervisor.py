"""Unit tests for the Stage 9d supervisor nightly cron registration + job.

Four unit tests (registration shape, time override, job invokes the runner,
exception is swallowed). They use the in-memory scheduler (conftest sets
``AIT_SCHEDULER_JOBSTORE=memory``) and a monkeypatched ``_connect_app_db`` so
they need neither Postgres nor a real runner. The lifespan-wiring integration
test lives in ``tests/integration/test_scheduler_supervisor.py``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from apscheduler.triggers.cron import CronTrigger

from orchestrator import scheduler as scheduler_mod
from orchestrator.scheduler import (
    SUPERVISOR_CRON_JOB_ID,
    build_scheduler,
    register_supervisor_cron,
    shutdown_scheduler,
    supervisor_cron_job,
)

# ─── registration shape ─────────────────────────────────────────────────


async def test_supervisor_cron_registered_with_cron_trigger() -> None:
    """Registered at id='supervisor_cron' with a CronTrigger at the default
    03:15 UTC; re-registration is idempotent (no error, still one job)."""
    sch = build_scheduler()
    sch.start()
    try:
        run_fn = AsyncMock()
        register_supervisor_cron(sch, run_supervisor_fn=run_fn)

        job = sch.get_job(SUPERVISOR_CRON_JOB_ID)
        assert job is not None
        assert job.id == SUPERVISOR_CRON_JOB_ID
        assert isinstance(job.trigger, CronTrigger)
        assert job.max_instances == 1
        trig = str(job.trigger)
        assert "hour='3'" in trig
        assert "minute='15'" in trig

        # Idempotent re-registration (replace_existing + guarded add_jobstore).
        register_supervisor_cron(sch, run_supervisor_fn=run_fn)
        matching = [j for j in sch.get_jobs() if j.id == SUPERVISOR_CRON_JOB_ID]
        assert len(matching) == 1
    finally:
        await shutdown_scheduler(sch)


async def test_supervisor_cron_time_overridable() -> None:
    """hour/minute params override the 03:15 UTC default."""
    sch = build_scheduler()
    sch.start()
    try:
        register_supervisor_cron(sch, run_supervisor_fn=AsyncMock(), hour=21, minute=45)
        job = sch.get_job(SUPERVISOR_CRON_JOB_ID)
        trig = str(job.trigger)
        assert "hour='21'" in trig
        assert "minute='45'" in trig
    finally:
        await shutdown_scheduler(sch)


# ─── job invokes the runner ─────────────────────────────────────────────


async def test_supervisor_cron_job_invokes_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job opens a conn, calls run_supervisor_fn(conn, trigger='cron'), closes."""
    fake_conn = AsyncMock()

    async def _fake_connect() -> AsyncMock:
        return fake_conn

    monkeypatch.setattr(scheduler_mod, "_connect_app_db", _fake_connect)

    run_fn = AsyncMock(return_value=SimpleNamespace(actions=[], overall_rationale="quiet night"))
    await supervisor_cron_job(run_supervisor_fn=run_fn)

    run_fn.assert_awaited_once()
    assert run_fn.await_args is not None
    assert run_fn.await_args.args[0] is fake_conn  # conn passed positionally
    assert run_fn.await_args.kwargs["trigger"] == "cron"
    fake_conn.close.assert_awaited_once()  # conn always closed


# ─── exception swallowed ────────────────────────────────────────────────


async def test_supervisor_cron_job_swallows_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing supervisor run is logged and SWALLOWED — no raise — and the
    conn is still closed, so the next nightly fire is unaffected."""
    fake_conn = AsyncMock()

    async def _fake_connect() -> AsyncMock:
        return fake_conn

    monkeypatch.setattr(scheduler_mod, "_connect_app_db", _fake_connect)

    run_fn = AsyncMock(side_effect=RuntimeError("supervisor boom"))
    with caplog.at_level(logging.ERROR):
        await supervisor_cron_job(run_supervisor_fn=run_fn)  # must NOT raise

    assert any("supervisor cron run failed" in r.getMessage() for r in caplog.records)
    fake_conn.close.assert_awaited_once()
