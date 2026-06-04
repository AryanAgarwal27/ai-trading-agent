"""Stage 9e DoD e2e — event-driven supervisor trigger (BRD §13 row 9).

Real Redis + real Postgres. Publishes three ``thread_completed`` events in quick
succession (distinct strategy_ids) and asserts the debounce collapses them to
EXACTLY ONE supervisor run at ``trigger="event"`` (last-event-wins trailing
debounce). Also covers the single-event case.

Marked ``@pytest.mark.integration`` so it is excluded from the unit CI job
(``pytest -m "not integration"``); it runs against the 9g integration service
containers (Postgres + Redis) or an operator-local stack. Requires ``REDIS_URL``
+ ``DATABASE_URL`` in the environment (conftest's ``load_dotenv``).

Mechanics: a real AsyncIOScheduler (memory jobstores, started) backs the
debounced one-shot; ``run_supervisor`` is replaced by a spy that records the
trigger (the one-shot still opens + closes a REAL app-DB conn via
``supervisor_event_job`` → ``_connect_app_db``, proving the real conn lifecycle).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
import redis.asyncio as aioredis

from orchestrator.observability.events import publish_thread_completed
from orchestrator.scheduler import SUPERVISOR_JOBSTORE
from orchestrator.supervisor_subscription import (
    cancel_supervisor_subscription,
    make_schedule_supervisor_run,
    run_supervisor_subscription,
)

pytestmark = pytest.mark.integration

_DEBOUNCE_S = 3.0


class _Decision:
    actions: list[Any] = []
    overall_rationale = "event run (e2e spy)"


def _build_started_scheduler() -> Any:
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    sched = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone="UTC")
    sched.add_jobstore(MemoryJobStore(), SUPERVISOR_JOBSTORE)
    sched.start()
    return sched


async def _run_with_published(strategy_ids: list[str]) -> list[str]:
    """Wire scheduler + subscription against real Redis, publish the given
    completions, wait out the debounce, and return the recorded run triggers."""
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = aioredis.from_url(redis_url)

    triggers: list[str] = []

    async def _spy_run(conn: Any, *, trigger: str) -> _Decision:
        triggers.append(trigger)
        return _Decision()

    scheduler = _build_started_scheduler()
    schedule_fn = make_schedule_supervisor_run(scheduler, _spy_run, debounce_seconds=_DEBOUNCE_S)
    sub_task: asyncio.Task[None] = asyncio.create_task(
        run_supervisor_subscription(redis_client, schedule_supervisor_run_fn=schedule_fn)
    )
    try:
        # Let the PSUBSCRIBE land before publishing (pubsub has no backlog).
        await asyncio.sleep(1.0)
        for sid in strategy_ids:
            await publish_thread_completed(
                sid,
                {
                    "strategy_id": sid,
                    "completed_at": "2026-06-04T03:00:00+00:00",
                    "final_stage": "archived",
                    "completion_reason": "retired_by_supervisor",
                },
            )
            await asyncio.sleep(0.2)  # quick succession, well inside the window
        # Wait out the trailing debounce + a margin for the one-shot to fire.
        await asyncio.sleep(_DEBOUNCE_S + 5.0)
    finally:
        await cancel_supervisor_subscription(sub_task)
        scheduler.shutdown(wait=False)
        await redis_client.aclose()

    return triggers


async def test_three_completions_collapse_to_one_event_run() -> None:
    """DoD: three completions in quick succession → exactly ONE supervisor run
    at trigger='event'."""
    triggers = await _run_with_published(["e2e-a", "e2e-b", "e2e-c"])
    assert triggers == ["event"], f"expected one coalesced event run, got {triggers!r}"


async def test_single_completion_triggers_one_run() -> None:
    """DoD: a single completion triggers one supervisor run at trigger='event'."""
    triggers = await _run_with_published(["e2e-solo"])
    assert triggers == ["event"]
