"""Stage 9e tests for the Redis thread-completion subscription.

Layers (mirroring tests/integration/test_kill_subscription.py's shape):

1. **Pure helpers** — channel → strategy_id parsing.
2. **Loop dispatch** (fake Redis, stub scheduler fn) — a well-formed
   ``thread_completed`` message reaches the debounced scheduler with the right
   strategy_id; bytes are decoded; malformed messages + scheduler failures are
   skipped without killing the loop.
3. **Debounce builder** (real AsyncIOScheduler, started paused so nothing fires)
   — N completions collapse to ONE one-shot job at the fixed id; the one-shot,
   when invoked, runs the runner at ``trigger="event"`` exactly once.
4. **Lifecycle** — the loop cancels cleanly and runs cleanup.
5. **Option 1-minimal regression guard** — the graph-only archive sinks
   (research / validation / live_archive) must NOT emit thread_completed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from orchestrator.scheduler import SUPERVISOR_JOBSTORE
from orchestrator.supervisor_subscription import (
    SUPERVISOR_EVENT_ONESHOT_ID,
    THREAD_COMPLETED_PATTERN,
    _strategy_id_from_channel,
    cancel_supervisor_subscription,
    make_schedule_supervisor_run,
    run_supervisor_subscription,
    supervisor_event_job,
)

# ───────────────────────── fakes ─────────────────────────


class _FakePubSub:
    """Minimal stand-in for ``redis.asyncio`` pubsub (mirrors the kill test).

    ``listen`` yields the canned messages then ends so the loop exits and runs
    cleanup, letting the test assert on punsubscribe / aclose.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.psubscribed: list[str] = []
        self.punsubscribed: list[str] = []
        self.closed = False

    async def psubscribe(self, pattern: str) -> None:
        self.psubscribed.append(pattern)

    async def punsubscribe(self, pattern: str) -> None:
        self.punsubscribed.append(pattern)

    async def aclose(self) -> None:
        self.closed = True

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        for message in self._messages:
            yield message


class _BlockingPubSub(_FakePubSub):
    """``listen`` blocks forever (until the task is cancelled) — for the
    lifecycle/cancellation test."""

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        await asyncio.Event().wait()
        yield {}  # pragma: no cover — unreachable, satisfies the async-gen type


class _FakeRedis:
    def __init__(self, pubsub: _FakePubSub) -> None:
        self._pubsub = pubsub

    def pubsub(self) -> _FakePubSub:
        return self._pubsub


def _pmessage(channel: str | bytes, data: Any) -> dict[str, Any]:
    return {"type": "pmessage", "channel": channel, "data": data}


def _completion_payload(strategy_id: str) -> str:
    return json.dumps(
        {
            "strategy_id": strategy_id,
            "completed_at": "2026-06-04T03:00:00+00:00",
            "final_stage": "archived",
            "completion_reason": "retired_by_supervisor",
        }
    )


def _build_test_scheduler() -> AsyncIOScheduler:
    """A real AsyncIOScheduler with the supervisor_memory jobstore the one-shot
    targets. Started PAUSED in tests so add_job coalescing runs immediately
    (state != STOPPED) but jobs never actually fire."""
    sched = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone="UTC")
    sched.add_jobstore(MemoryJobStore(), SUPERVISOR_JOBSTORE)
    return sched


# ───────────────────────── pure helpers ─────────────────────────


def test_strategy_id_from_channel_handles_prefix_colons() -> None:
    assert _strategy_id_from_channel("ai-trading-agent:thread_completed:abc123def") == "abc123def"


# ───────────────────────── loop dispatch ─────────────────────────


async def test_completion_dispatched_to_scheduler_with_strategy_id() -> None:
    """A well-formed thread_completed pmessage reaches the scheduler fn with the
    strategy_id parsed from the channel; cleanup runs on loop exit."""
    pubsub = _FakePubSub(
        [_pmessage("ai-trading-agent:thread_completed:strat-7", _completion_payload("strat-7"))]
    )
    dispatched: list[str] = []

    async def _schedule(strategy_id: str) -> None:
        dispatched.append(strategy_id)

    await run_supervisor_subscription(_FakeRedis(pubsub), schedule_supervisor_run_fn=_schedule)

    assert dispatched == ["strat-7"]
    assert pubsub.psubscribed == [THREAD_COMPLETED_PATTERN]
    assert pubsub.punsubscribed == [THREAD_COMPLETED_PATTERN]
    assert pubsub.closed is True


async def test_bytes_channel_and_data_are_decoded() -> None:
    """Real redis-py yields bytes; the loop decodes both channel and data, and
    the JSON payload parses (payload fields available if a consumer wants them)."""
    payload = _completion_payload("s9").encode("utf-8")
    pubsub = _FakePubSub([_pmessage(b"ai-trading-agent:thread_completed:s9", payload)])
    dispatched: list[str] = []

    async def _schedule(strategy_id: str) -> None:
        dispatched.append(strategy_id)

    await run_supervisor_subscription(_FakeRedis(pubsub), schedule_supervisor_run_fn=_schedule)
    assert dispatched == ["s9"]


async def test_malformed_messages_skipped_safely() -> None:
    """Bad JSON, non-object payloads, and subscribe-confirmation frames are
    skipped without raising; only the well-formed message dispatches."""
    messages = [
        _pmessage("ai-trading-agent:thread_completed:s1", "not json{{"),
        _pmessage("ai-trading-agent:thread_completed:s2", json.dumps([1, 2, 3])),
        {"type": "psubscribe", "channel": "ai-trading-agent:thread_completed:*", "data": 1},
        _pmessage("ai-trading-agent:thread_completed:s3", _completion_payload("s3")),
    ]
    pubsub = _FakePubSub(messages)
    seen: list[str] = []

    async def _schedule(strategy_id: str) -> None:
        seen.append(strategy_id)

    await run_supervisor_subscription(_FakeRedis(pubsub), schedule_supervisor_run_fn=_schedule)
    assert seen == ["s3"], "only the well-formed message should dispatch"
    assert pubsub.closed is True  # cleanup still ran


async def test_scheduler_failure_does_not_abort_loop() -> None:
    """A scheduler fn raising on one event must not stop later events."""
    messages = [
        _pmessage("ai-trading-agent:thread_completed:boom", _completion_payload("boom")),
        _pmessage("ai-trading-agent:thread_completed:ok", _completion_payload("ok")),
    ]
    pubsub = _FakePubSub(messages)
    seen: list[str] = []

    async def _schedule(strategy_id: str) -> None:
        if strategy_id == "boom":
            raise RuntimeError("jobstore exploded")
        seen.append(strategy_id)

    await run_supervisor_subscription(_FakeRedis(pubsub), schedule_supervisor_run_fn=_schedule)
    assert seen == ["ok"]


# ───────────────────────── debounce builder ─────────────────────────


async def test_debounce_coalesces_three_completions_to_one_job() -> None:
    """Three completions in quick succession (distinct strategy_ids) collapse to
    EXACTLY one one-shot job at the fixed id (replace_existing — last-event-wins)."""
    sched = _build_test_scheduler()
    sched.start(paused=True)  # flush pending into the store; never fire
    try:

        async def _run(conn: Any, *, trigger: str) -> Any:  # pragma: no cover — not invoked here
            return None

        schedule = make_schedule_supervisor_run(sched, _run, debounce_seconds=30.0)
        await schedule("s1")
        await schedule("s2")
        await schedule("s3")

        jobs = sched.get_jobs(jobstore=SUPERVISOR_JOBSTORE)
        assert len(jobs) == 1, "N completions must coalesce to one one-shot"
        assert jobs[0].id == SUPERVISOR_EVENT_ONESHOT_ID
        assert jobs[0].func is supervisor_event_job
        # The job carries the runner via kwargs (non-picklable closure → memory store).
        assert jobs[0].kwargs["run_supervisor_fn"] is _run
    finally:
        sched.shutdown(wait=False)


async def test_oneshot_job_invokes_runner_with_event_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Firing the one-shot job runs the runner ONCE at trigger='event', on a
    fresh conn that is closed afterward (mirrors supervisor_cron_job)."""
    import orchestrator.supervisor_subscription as sub_mod

    captured: dict[str, Any] = {}

    class _FakeConn:
        async def close(self) -> None:
            captured["closed"] = True

    async def _fake_connect() -> _FakeConn:
        return _FakeConn()

    monkeypatch.setattr(sub_mod, "_connect_app_db", _fake_connect)

    class _Decision:
        actions: list[Any] = []
        overall_rationale = "let threads run"

    async def _run(conn: Any, *, trigger: str) -> _Decision:
        captured["conn"] = conn
        captured["trigger"] = trigger
        return _Decision()

    await supervisor_event_job(run_supervisor_fn=_run)

    assert captured["trigger"] == "event"
    assert isinstance(captured["conn"], _FakeConn)
    assert captured["closed"] is True


async def test_oneshot_job_swallows_runner_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner failure inside the event job is logged + swallowed (one bad run
    must not kill the scheduler), and the conn is still closed."""
    import orchestrator.supervisor_subscription as sub_mod

    closed = {"value": False}

    class _FakeConn:
        async def close(self) -> None:
            closed["value"] = True

    async def _fake_connect() -> _FakeConn:
        return _FakeConn()

    monkeypatch.setattr(sub_mod, "_connect_app_db", _fake_connect)

    async def _run(conn: Any, *, trigger: str) -> Any:
        raise RuntimeError("runner boom")

    # Does not raise.
    await supervisor_event_job(run_supervisor_fn=_run)
    assert closed["value"] is True


# ───────────────────────── lifecycle ─────────────────────────


async def test_subscription_task_cancels_cleanly() -> None:
    """The lifespan task blocks on listen(), then cancels cleanly and runs
    cleanup (punsubscribe + aclose) — no orphaned task."""
    pubsub = _BlockingPubSub([])

    async def _schedule(strategy_id: str) -> None:  # pragma: no cover — never reached
        pass

    task: asyncio.Task[None] = asyncio.create_task(
        run_supervisor_subscription(_FakeRedis(pubsub), schedule_supervisor_run_fn=_schedule)
    )
    # Let it subscribe + enter listen().
    for _ in range(50):
        await asyncio.sleep(0)
        if pubsub.psubscribed:
            break
    assert pubsub.psubscribed == [THREAD_COMPLETED_PATTERN]
    assert not task.done()

    await cancel_supervisor_subscription(task)

    assert task.done()
    assert pubsub.punsubscribed == [THREAD_COMPLETED_PATTERN]
    assert pubsub.closed is True


# ───────────────────────── Option 1-minimal regression guard ─────────────────


async def test_no_emission_at_graph_only_archive_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option 1-minimal CONTRACT GUARD: the graph-only terminal sinks
    (research / validation / live_archive) must NOT publish thread_completed —
    emission lives ONLY at paper_teardown / live_spawn-fail / supervisor retire,
    with the nightly cron as the backstop for funnel-internal completions. If a
    future change moves emission to one of these sinks, this fails loudly and
    forces a deliberate reconsideration of the 9e design."""
    import orchestrator.observability.events as events_mod
    import orchestrator.subgraphs.live as live_mod
    from orchestrator.subgraphs.research import archive as research_archive
    from orchestrator.subgraphs.validation import archive as validation_archive

    calls: list[str] = []

    async def _spy(strategy_id: str, payload: dict[str, Any]) -> None:
        calls.append(strategy_id)

    # Patch at the origin AND in live.py's namespace (live.py binds the name at
    # import for its live_spawn-failure emission, so a future call inside
    # live_archive would resolve to the bound name).
    monkeypatch.setattr(events_mod, "publish_thread_completed", _spy)
    monkeypatch.setattr(live_mod, "publish_thread_completed", _spy)

    # research / validation archive: pure sync sinks — no emission.
    research_archive({"stage": "archived", "failure_reason": "critic_loop_exhausted"})
    validation_archive({"stage": "archived", "failure_reason": "robustness_gate"})

    # live_archive: terminal teardown (stop only) — no emission.
    async def _noop_stop(strategy_id: str) -> None:
        return None

    await live_mod.live_archive(
        {"strategy_id": "s1", "stage": "archived", "failure_reason": "coordinator_fail"},
        None,
        stop_container_fn=_noop_stop,
    )

    assert (
        calls == []
    ), "graph-only archive sinks must not emit thread_completed (9e Option 1-minimal)"
