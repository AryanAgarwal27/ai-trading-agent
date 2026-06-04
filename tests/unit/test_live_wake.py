"""Stage 9f unit tests — D-6 production live-wake (the split-by-trigger design).

Hermetic (no Postgres, no Redis, no real scheduler firing, no Docker). Covers
both halves of the split + the critical kill-vs-periodic independence guard:

- **_fire_wake** passes ``kind`` through to /wake (paper_wait default, live_wait
  for the live job).
- **make_schedule_live_wake_fn / make_unschedule_live_wake_fn** register +
  cancel ``live_wake:<sid>`` (kwargs kind="live_wait"); unschedule is idempotent.
- **live_spawn** registers the wake on success; **live_archive** cancels it.
- **_route_after_live_wait** routes the no-kill periodic wake to live_evaluate
  (regression guard for the normal live cycle) and the kill wake to live_pause.
- **kill-path direct-resume** (make_kill_event_writer): after the state write,
  the thread is resumed via Command(resume=...); failures are swallowed.
- **/wake endpoint** (wake_thread): accepts kind=live_wait, 409s on kind
  mismatch, 422s on an unknown kind, and — the CRITICAL regression guard —
  409s a kill-written thread whose parent interrupt was cleared (so the kill
  path is the ONLY resumer for kill-written threads).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import HTTPException
from langgraph.types import Command

import orchestrator.kill_subscription as kill_sub
import orchestrator.subgraphs.live as live_mod
from orchestrator.main import WAKEABLE_KINDS, wake_thread
from orchestrator.scheduler import (
    LIVE_WAKE_JOB_PREFIX,
    _fire_wake,
    make_schedule_live_wake_fn,
    make_unschedule_live_wake_fn,
)
from orchestrator.subgraphs.live import _route_after_live_wait, live_archive, live_spawn

# ════════════════════════════════════════════════════════════════════════
# _fire_wake — kind pass-through
# ════════════════════════════════════════════════════════════════════════


async def test_fire_wake_passes_live_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        text = ""

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None: ...

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(
            self, url: str, headers: dict[str, str] | None = None, params: Any = None
        ) -> _Resp:
            captured["url"] = url
            captured["params"] = params
            return _Resp()

    monkeypatch.setattr("orchestrator.scheduler.httpx.AsyncClient", _Client)
    await _fire_wake("strategy_sid1", "http://127.0.0.1:8000", kind="live_wait")

    assert captured["url"] == "http://127.0.0.1:8000/threads/strategy_sid1/wake"
    assert captured["params"] == {"kind": "live_wait"}


# ════════════════════════════════════════════════════════════════════════
# Scheduler: register + unschedule the live-wake job
# ════════════════════════════════════════════════════════════════════════


def _started_paused_scheduler() -> AsyncIOScheduler:
    sched = AsyncIOScheduler(jobstores={"default": MemoryJobStore()}, timezone="UTC")
    sched.start(paused=True)  # flush add_job into the store; never fire
    return sched


async def test_schedule_live_wake_registers_then_unschedule_removes() -> None:
    sched = _started_paused_scheduler()
    try:
        schedule = make_schedule_live_wake_fn(sched, base_url="http://127.0.0.1:8000")
        await schedule("strategy_sid1", "sid1")

        jobs = sched.get_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job.id == f"{LIVE_WAKE_JOB_PREFIX}sid1"
        # kind threaded as a kwarg so _fire_wake posts kind=live_wait; args mirror
        # paper's [thread_id, base_url].
        assert job.kwargs == {"kind": "live_wait"}
        assert tuple(job.args) == ("strategy_sid1", "http://127.0.0.1:8000")

        unschedule = make_unschedule_live_wake_fn(sched)
        await unschedule("sid1")
        assert sched.get_jobs() == []

        # Idempotent: cancelling an already-gone job is a swallowed no-op.
        await unschedule("sid1")
    finally:
        sched.shutdown(wait=False)


# ════════════════════════════════════════════════════════════════════════
# live_spawn registers the wake; live_archive cancels it
# ════════════════════════════════════════════════════════════════════════


async def test_live_spawn_registers_wake_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_mod, "_read_used_ports", AsyncMock(return_value=[]))
    scheduled: list[tuple[str, str]] = []

    async def _spy_schedule(thread_id: str, strategy_id: str) -> None:
        scheduled.append((thread_id, strategy_id))

    async def _stub_spawn(**_kw: Any) -> str:
        return "http://127.0.0.1:8200"

    async def _stub_registry(**_kw: Any) -> None:
        return None

    state: dict[str, Any] = {
        "strategy_id": "sid1",
        "name": "strat-a",
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "artifacts": {"generated_strategy_path": "/tmp/s.py"},
    }
    config = {"configurable": {"thread_id": "strategy_sid1"}}

    result = await live_spawn(
        state,  # type: ignore[arg-type]
        config,  # type: ignore[arg-type]
        spawn_live_container_fn=_stub_spawn,
        registry_writer_fn=_stub_registry,
        secrets_provider=object(),  # type: ignore[arg-type]
        schedule_wake_fn=_spy_schedule,
    )

    assert result["stage"] == "live"
    assert scheduled == [("strategy_sid1", "sid1")]


async def test_live_spawn_failure_does_not_register_wake(monkeypatch: pytest.MonkeyPatch) -> None:
    """A boot failure archives BEFORE the registration line, so no wake job is
    scheduled (nothing to clean up on the fail path)."""
    monkeypatch.setattr(live_mod, "_read_used_ports", AsyncMock(return_value=[]))
    # Avoid the real thread_completed publish (9e) reaching Redis.
    monkeypatch.setattr(live_mod, "publish_thread_completed", AsyncMock())
    scheduled: list[tuple[str, str]] = []

    async def _spy_schedule(thread_id: str, strategy_id: str) -> None:
        scheduled.append((thread_id, strategy_id))

    async def _boom_spawn(**_kw: Any) -> str:
        raise RuntimeError("container boot failed")

    async def _stub_registry(**_kw: Any) -> None:
        return None

    state: dict[str, Any] = {
        "strategy_id": "sid2",
        "name": "strat-b",
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "artifacts": {"generated_strategy_path": "/tmp/s.py"},
    }
    result = await live_spawn(
        state,  # type: ignore[arg-type]
        {"configurable": {"thread_id": "strategy_sid2"}},  # type: ignore[arg-type]
        spawn_live_container_fn=_boom_spawn,
        registry_writer_fn=_stub_registry,
        secrets_provider=object(),  # type: ignore[arg-type]
        schedule_wake_fn=_spy_schedule,
    )

    assert result["stage"] == "archived"
    assert scheduled == []


async def test_live_archive_cancels_wake() -> None:
    unscheduled: list[str] = []

    async def _spy_unschedule(strategy_id: str) -> None:
        unscheduled.append(strategy_id)

    async def _noop_stop(strategy_id: str) -> None:
        return None

    result = await live_archive(
        {"strategy_id": "sid1", "stage": "archived", "failure_reason": "coordinator_fail"},  # type: ignore[arg-type]
        None,
        stop_container_fn=_noop_stop,
        unschedule_wake_fn=_spy_unschedule,
    )

    assert result["stage"] == "archived"
    assert unscheduled == ["sid1"]


# ════════════════════════════════════════════════════════════════════════
# _route_after_live_wait — periodic (no-kill) vs kill routing
# ════════════════════════════════════════════════════════════════════════


def test_route_after_live_wait_no_kill_goes_to_evaluate() -> None:
    """Regression guard for the NORMAL periodic live cycle: a wake with no
    kill_switch_event proceeds to live_evaluate (the reviewer fan-out)."""
    assert _route_after_live_wait({"artifacts": {}}) == "live_evaluate"  # type: ignore[arg-type]
    assert _route_after_live_wait({}) == "live_evaluate"  # type: ignore[arg-type]


def test_route_after_live_wait_kill_goes_to_pause() -> None:
    assert (
        _route_after_live_wait({"artifacts": {"kill_switch_event": {"reason": "dd"}}})  # type: ignore[arg-type]
        == "live_pause"
    )


# ════════════════════════════════════════════════════════════════════════
# Kill-path direct-resume (make_kill_event_writer)
# ════════════════════════════════════════════════════════════════════════


class _Snap:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class _ResumeGraph:
    """Records aupdate_state + astream; astream is an empty async generator."""

    def __init__(self, values: dict[str, Any], *, boom: bool = False) -> None:
        self._values = values
        self._boom = boom
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.astream_calls: list[tuple[Any, Any]] = []

    async def aget_state(self, config: dict[str, Any]) -> _Snap:
        return _Snap(self._values)

    async def aupdate_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        self.updates.append((config, values))

    def astream(self, command: Any, config: Any = None) -> AsyncIterator[Any]:
        self.astream_calls.append((command, config))
        if self._boom:
            raise RuntimeError("graph astream boom")
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Any]:
        if False:  # pragma: no cover — empty async generator
            yield None


async def _drain_kill_resume_tasks() -> None:
    # The writer schedules the direct-resume as a fire-and-forget task tracked in
    # _KILL_RESUME_TASKS; grab + await them so the test is deterministic.
    await asyncio.gather(*list(kill_sub._KILL_RESUME_TASKS))


async def test_kill_writer_direct_resumes_after_state_write() -> None:
    graph = _ResumeGraph({"stage": "live", "artifacts": {}})
    writer = kill_sub.make_kill_event_writer(graph)

    await writer("sid1", {"reason": "drawdown_12pct_exceeded", "metrics_summary": {}})
    await _drain_kill_resume_tasks()

    # State write happened (route trigger) ...
    assert len(graph.updates) == 1
    # ... then the thread was DIRECTLY resumed (not via /wake).
    assert len(graph.astream_calls) == 1
    command, config = graph.astream_calls[0]
    assert config["configurable"]["thread_id"] == "strategy_sid1"
    assert isinstance(command, Command)
    assert command.resume == {"wake": True, "source": "kill_subscription"}


async def test_kill_writer_resume_failure_is_swallowed() -> None:
    """A direct-resume failure must not propagate (the durable kill_switch_events
    row is the recovery path); the state write still landed."""
    graph = _ResumeGraph({"stage": "live", "artifacts": {}}, boom=True)
    writer = kill_sub.make_kill_event_writer(graph)

    await writer("sid1", {"reason": "r"})
    await _drain_kill_resume_tasks()  # must not raise

    assert len(graph.updates) == 1  # state write intact
    assert len(graph.astream_calls) == 1  # resume was attempted


# ════════════════════════════════════════════════════════════════════════
# /wake endpoint — kind acceptance / rejection (called directly, hermetic)
# ════════════════════════════════════════════════════════════════════════


class _Interrupt:
    def __init__(self, kind: str) -> None:
        self.value = {"kind": kind}


class _ParkedTask:
    def __init__(self, kind: str | None) -> None:
        self.interrupts = (_Interrupt(kind),) if kind else ()


class _WakeSnap:
    def __init__(self, tasks: list[Any], values: dict[str, Any], nxt: tuple[Any, ...]) -> None:
        self.tasks = tasks
        self.values = values
        self.next = nxt


class _WakeGraph:
    """aget_state returns a parked snapshot, then a cleared one after astream."""

    def __init__(self, parked_kind: str | None, post_values: dict[str, Any] | None = None) -> None:
        self._parked_kind = parked_kind
        self._post = post_values or {"stage": "live"}
        self._resumed = False

    async def aget_state(self, config: dict[str, Any]) -> _WakeSnap:
        if self._resumed:
            return _WakeSnap([], self._post, ())
        tasks = [_ParkedTask(self._parked_kind)] if self._parked_kind else []
        return _WakeSnap(tasks, {}, ("live_wait",) if self._parked_kind else ())

    def astream(self, command: Any, config: Any = None) -> AsyncIterator[Any]:
        self._resumed = True
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Any]:
        if False:  # pragma: no cover — empty async generator
            yield None


class _FakeReq:
    def __init__(self, graph: Any) -> None:
        self.app = type(
            "_App",
            (),
            {
                "state": type(
                    "_St", (), {"graph": graph, "thread_locks": defaultdict(asyncio.Lock)}
                )()
            },
        )()


async def test_wake_live_wait_resumes_thread() -> None:
    graph = _WakeGraph("live_wait", post_values={"stage": "live"})
    result = await wake_thread("strategy_sid1", _FakeReq(graph), kind="live_wait", token="x")  # type: ignore[arg-type]
    assert result == {"woke": True, "next_stage": "live"}


async def test_wake_live_wait_409_on_kind_mismatch() -> None:
    """A live-wake must refuse a thread parked at paper_wait (and vice versa)."""
    graph = _WakeGraph("paper_wait")
    with pytest.raises(HTTPException) as ei:
        await wake_thread("tid", _FakeReq(graph), kind="live_wait", token="x")  # type: ignore[arg-type]
    assert ei.value.status_code == 409


async def test_wake_invalid_kind_422() -> None:
    graph = _WakeGraph("live_wait")
    with pytest.raises(HTTPException) as ei:
        await wake_thread("tid", _FakeReq(graph), kind="not_a_kind", token="x")  # type: ignore[arg-type]
    assert ei.value.status_code == 422
    assert "live_wait" in WAKEABLE_KINDS  # sanity: the live kind IS wake-able


async def test_periodic_wake_refuses_kill_written_thread() -> None:
    """CRITICAL 9f regression guard — kill-path / periodic-path independence.

    When kill_subscription writes artifacts.kill_switch_event via aupdate_state,
    the parent-visible interrupt surface is CLEARED (the 8h finding). A periodic
    live-wake (kind="live_wait") that fires against such a thread therefore finds
    NO parent interrupt → 409. This pins the contract that the KILL PATH is the
    ONLY resumer for kill-written threads (it direct-resumes via Command(resume=)),
    and the periodic /wake never double-invokes that path. A future refactor that
    "unifies" the two paths — letting /wake resume a kill-written thread — would
    re-introduce the 8h problem and break this test loudly.
    """
    # parked_kind=None models the post-aupdate_state state: interrupt cleared,
    # though the nested live_wait stays parked-and-resumable internally.
    graph = _WakeGraph(None)
    with pytest.raises(HTTPException) as ei:
        await wake_thread("strategy_killed", _FakeReq(graph), kind="live_wait", token="x")  # type: ignore[arg-type]
    assert ei.value.status_code == 409
