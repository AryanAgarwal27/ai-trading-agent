"""Unit tests for the APScheduler wiring (Stage 7f).

No markers — run in the default CI invocation. The scheduler uses the
in-memory jobstore (conftest sets AIT_SCHEDULER_JOBSTORE=memory) and is
started PAUSED so jobs commit to the store for inspection without ever
firing. _fire_wake's httpx call is faked.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.scheduler import (
    KILL_SWITCH_JOB_ID,
    REGIME_JOB_ID,
    WAKE_INTERVAL_HOURS,
    _fire_wake,
    build_scheduler,
    make_schedule_wake_fn,
    make_unschedule_wake_fn,
    register_recurring_jobs,
    shutdown_scheduler,
)


async def test_schedule_wake_fn_registers_per_thread_job() -> None:
    """schedule_wake_fn adds a 6h interval job keyed wake:<thread_id>."""
    scheduler = build_scheduler()
    scheduler.start(paused=True)
    try:
        fn = make_schedule_wake_fn(scheduler, base_url="http://127.0.0.1:8000")
        await fn("strategy_t1", "t1")

        job = scheduler.get_job("wake:strategy_t1")
        assert job is not None
        # APScheduler normalizes args to a tuple.
        assert tuple(job.args) == ("strategy_t1", "http://127.0.0.1:8000")
        # interval trigger at the BRD §5.5 6-hour cadence.
        assert int(job.trigger.interval.total_seconds()) == WAKE_INTERVAL_HOURS * 3600
    finally:
        await shutdown_scheduler(scheduler)


async def test_schedule_wake_fn_replaces_existing_on_respawn() -> None:
    """Re-registering the same thread replaces (not duplicates) the job."""
    scheduler = build_scheduler()
    scheduler.start(paused=True)
    try:
        fn = make_schedule_wake_fn(scheduler, base_url="http://127.0.0.1:8000")
        await fn("strategy_t1", "t1")
        await fn("strategy_t1", "t1")  # respawn / replay

        wake_jobs = [j for j in scheduler.get_jobs() if j.id == "wake:strategy_t1"]
        assert len(wake_jobs) == 1
    finally:
        await shutdown_scheduler(scheduler)


async def test_unschedule_wake_fn_removes_job_and_is_idempotent() -> None:
    """D-12: make_unschedule_wake_fn cancels the paper wake job
    (wake:strategy_<sid>, keyed off thread_id == strategy_<sid>) and a missing
    job is a swallowed no-op (the spawn-failure path / already-removed case)."""
    scheduler = build_scheduler()
    scheduler.start(paused=True)
    try:
        schedule = make_schedule_wake_fn(scheduler, base_url="http://127.0.0.1:8000")
        await schedule("strategy_t1", "t1")
        assert scheduler.get_job("wake:strategy_t1") is not None

        unschedule = make_unschedule_wake_fn(scheduler)
        await unschedule("t1")  # strategy_id → removes wake:strategy_t1
        assert scheduler.get_job("wake:strategy_t1") is None

        # Idempotent: cancelling an already-gone job does not raise.
        await unschedule("t1")
    finally:
        await shutdown_scheduler(scheduler)


async def test_register_recurring_jobs_adds_regime_and_kill_switch() -> None:
    scheduler = build_scheduler()
    scheduler.start(paused=True)
    try:
        register_recurring_jobs(scheduler)
        assert scheduler.get_job(REGIME_JOB_ID) is not None
        assert scheduler.get_job(KILL_SWITCH_JOB_ID) is not None
    finally:
        await shutdown_scheduler(scheduler)


async def test_fire_wake_posts_with_operator_token_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_fire_wake POSTs to /threads/{tid}/wake with X-Operator-Token (SPEC §6)."""
    monkeypatch.setenv("OPERATOR_TOKEN", "tok-7f-abc")

    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200
        text = ""

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            params: dict[str, str] | None = None,
        ) -> _FakeResp:
            captured["url"] = url
            captured["headers"] = headers
            captured["params"] = params
            return _FakeResp()

    monkeypatch.setattr("orchestrator.scheduler.httpx.AsyncClient", _FakeClient)

    await _fire_wake("strategy_t1", "http://127.0.0.1:8000")

    assert captured["url"] == "http://127.0.0.1:8000/threads/strategy_t1/wake"
    assert captured["headers"] == {"X-Operator-Token": "tok-7f-abc"}
    # 9f: default kind is paper_wait (backward compat for Stage-7f paper jobs).
    assert captured["params"] == {"kind": "paper_wait"}


async def test_fire_wake_swallows_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport failure is logged, not raised (next cycle retries)."""
    import httpx

    monkeypatch.setenv("OPERATOR_TOKEN", "tok-7f-abc")

    class _BoomClient:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...

        async def __aenter__(self) -> _BoomClient:
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(self, *_a: Any, **_k: Any) -> Any:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("orchestrator.scheduler.httpx.AsyncClient", _BoomClient)

    # Must NOT raise.
    await _fire_wake("strategy_t1", "http://127.0.0.1:8000")
