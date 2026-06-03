"""Stage 8g tests for the Redis kill-switch subscription.

Three layers:

1. **Loop unit tests** (no Redis, no checkpointer) — drive
   :func:`run_kill_subscription` with a fake pubsub yielding canned messages
   and a stub writer. Covers dispatch + malformed-message safety.
2. **Writer unit tests** (fake graph) — :func:`make_kill_event_writer`'s
   state-write + Fork-2 guard (skip unknown / non-live threads) + the
   ``metrics_summary`` → ``metrics`` normalization.
3. **Lifespan lifecycle** (integration — real lifespan) — the subscription
   task is created on startup and cancelled cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from orchestrator.kill_subscription import (
    KILL_SWITCH_PATTERN,
    _normalize_kill_event,
    _strategy_id_from_channel,
    make_kill_event_writer,
    run_kill_subscription,
)

# ───────────────────────── fakes ─────────────────────────


class _FakePubSub:
    """Minimal stand-in for ``redis.asyncio`` pubsub.

    ``listen`` yields the canned messages then ends (StopAsyncIteration) so
    ``run_kill_subscription`` exits its loop and runs cleanup — letting the test
    assert on the punsubscribe / aclose calls without cancellation plumbing.
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


class _FakeRedis:
    def __init__(self, pubsub: _FakePubSub) -> None:
        self._pubsub = pubsub

    def pubsub(self) -> _FakePubSub:
        return self._pubsub


class _FakeSnapshot:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class _FakeGraph:
    """Records aupdate_state calls; returns a fixed snapshot from aget_state."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values
        self.updates: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def aget_state(self, config: dict[str, Any]) -> _FakeSnapshot:
        return _FakeSnapshot(self._values)

    async def aupdate_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        self.updates.append((config, values))


def _pmessage(channel: str, data: Any) -> dict[str, Any]:
    return {"type": "pmessage", "channel": channel, "data": data}


# ───────────────────────── pure helpers ─────────────────────────


def test_strategy_id_from_channel_handles_prefix_colons() -> None:
    assert (
        _strategy_id_from_channel("ai-trading-agent:kill_switch:live-abc123") == "live-abc123"
    )


def test_normalize_aliases_metrics_and_does_not_inject_action() -> None:
    """8g Change 1: metrics_summary is aliased to metrics, but action_taken is
    NOT defaulted — it now flows through the publish payload (source-fix)."""
    out = _normalize_kill_event(
        {
            "reason": "r",
            "fired_at": "t",
            "action_taken": "POST /api/v1/stop",
            "metrics_summary": {"max_drawdown": 0.13},
        }
    )
    assert out["metrics"] == {"max_drawdown": 0.13}  # aliased
    assert out["metrics_summary"] == {"max_drawdown": 0.13}  # preserved
    assert out["action_taken"] == "POST /api/v1/stop"  # passed through, not injected


def test_normalize_does_not_fabricate_action_when_missing() -> None:
    """A legacy/malformed event without action_taken is left without one — the
    dashboard card's _unknown_ fallback handles it (no fabricated label)."""
    out = _normalize_kill_event({"reason": "r", "metrics_summary": {"max_drawdown": 0.1}})
    assert "action_taken" not in out


# ───────────────────────── loop dispatch ─────────────────────────


async def test_kill_event_dispatched_to_writer() -> None:
    """A well-formed kill pmessage reaches the writer with the parsed event +
    the channel appended; cleanup runs on loop exit."""
    payload = {
        "reason": "drawdown_12pct_exceeded",
        "fired_at": "2026-06-03T14:32:11Z",
        "metrics_summary": {"max_drawdown": 0.131, "consecutive_losses": 4},
    }
    pubsub = _FakePubSub(
        [_pmessage("ai-trading-agent:kill_switch:live-abc", json.dumps(payload))]
    )
    captured: dict[str, Any] = {}

    async def _writer(strategy_id: str, event: dict[str, Any]) -> None:
        captured["sid"] = strategy_id
        captured["event"] = event

    await run_kill_subscription(_FakeRedis(pubsub), kill_event_writer_fn=_writer)

    assert captured["sid"] == "live-abc"
    assert captured["event"]["reason"] == "drawdown_12pct_exceeded"
    assert captured["event"]["channel"] == "ai-trading-agent:kill_switch:live-abc"
    assert pubsub.psubscribed == [KILL_SWITCH_PATTERN]
    assert pubsub.punsubscribed == [KILL_SWITCH_PATTERN]
    assert pubsub.closed is True


async def test_bytes_channel_and_data_are_decoded() -> None:
    """Real redis-py yields bytes; the loop decodes both channel and data."""
    payload = json.dumps({"reason": "consecutive_losses_10_exceeded", "fired_at": "t"})
    pubsub = _FakePubSub(
        [_pmessage(b"ai-trading-agent:kill_switch:s9", payload.encode("utf-8"))]
    )
    captured: dict[str, Any] = {}

    async def _writer(strategy_id: str, event: dict[str, Any]) -> None:
        captured["sid"] = strategy_id
        captured["event"] = event

    await run_kill_subscription(_FakeRedis(pubsub), kill_event_writer_fn=_writer)
    assert captured["sid"] == "s9"
    assert captured["event"]["channel"] == "ai-trading-agent:kill_switch:s9"


async def test_malformed_messages_skipped_safely() -> None:
    """Bad JSON, non-object payloads, and subscribe-confirmation frames are
    skipped without raising; only the well-formed message dispatches."""
    messages = [
        _pmessage("ai-trading-agent:kill_switch:s1", "not json{{"),
        _pmessage("ai-trading-agent:kill_switch:s2", json.dumps([1, 2, 3])),
        {"type": "psubscribe", "channel": "ai-trading-agent:kill_switch:*", "data": 1},
        _pmessage("ai-trading-agent:kill_switch:s3", json.dumps({"reason": "r"})),
    ]
    pubsub = _FakePubSub(messages)
    seen: list[str] = []

    async def _writer(strategy_id: str, event: dict[str, Any]) -> None:
        seen.append(strategy_id)

    await run_kill_subscription(_FakeRedis(pubsub), kill_event_writer_fn=_writer)
    assert seen == ["s3"], "only the well-formed message should dispatch"
    assert pubsub.closed is True  # cleanup still ran


async def test_writer_failure_does_not_abort_loop() -> None:
    """A writer raising on one event must not stop later events processing."""
    messages = [
        _pmessage("ai-trading-agent:kill_switch:boom", json.dumps({"reason": "r1"})),
        _pmessage("ai-trading-agent:kill_switch:ok", json.dumps({"reason": "r2"})),
    ]
    pubsub = _FakePubSub(messages)
    seen: list[str] = []

    async def _writer(strategy_id: str, event: dict[str, Any]) -> None:
        if strategy_id == "boom":
            raise RuntimeError("checkpointer exploded")
        seen.append(strategy_id)

    await run_kill_subscription(_FakeRedis(pubsub), kill_event_writer_fn=_writer)
    assert seen == ["ok"]


# ───────────────────────── writer + Fork-2 guard ─────────────────────────


async def test_writer_writes_normalized_event_for_live_thread() -> None:
    """The default writer merges a normalized kill event into a live thread's
    artifacts (preserving existing keys)."""
    graph = _FakeGraph(
        {"stage": "live", "artifacts": {"live_started_at": "2026-06-03T00:00:00Z"}}
    )
    writer = make_kill_event_writer(graph)
    await writer(
        "live-abc",
        {
            "reason": "drawdown_12pct_exceeded",
            "fired_at": "t",
            "action_taken": "POST /api/v1/stop",
            "metrics_summary": {"max_drawdown": 0.13},
            "channel": "ai-trading-agent:kill_switch:live-abc",
        },
    )
    assert len(graph.updates) == 1
    config, values = graph.updates[0]
    assert config["configurable"]["thread_id"] == "strategy_live-abc"
    artifacts = values["artifacts"]
    assert artifacts["live_started_at"] == "2026-06-03T00:00:00Z"  # preserved
    event = artifacts["kill_switch_event"]
    assert event["reason"] == "drawdown_12pct_exceeded"
    assert event["metrics"] == {"max_drawdown": 0.13}  # normalized alias
    assert event["action_taken"] == "POST /api/v1/stop"  # passed through


async def test_writer_skips_unknown_thread() -> None:
    """Fork-2: a kill event for a thread with no checkpoint must not mint one."""
    graph = _FakeGraph({})  # aget_state → empty values (unknown thread)
    writer = make_kill_event_writer(graph)
    await writer("ghost", {"reason": "r"})
    assert graph.updates == []


async def test_writer_skips_non_live_thread() -> None:
    """Fork-2: a kill event for an already-terminal thread is skipped."""
    graph = _FakeGraph({"stage": "archived", "artifacts": {}})
    writer = make_kill_event_writer(graph)
    await writer("done", {"reason": "r"})
    assert graph.updates == []


# ───────────────────────── lifespan lifecycle ─────────────────────────


@pytest.mark.integration
async def test_lifespan_starts_and_cancels_kill_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan creates the subscription task on startup and cancels it
    cleanly on shutdown (no orphaned task)."""
    import orchestrator.main as main_mod

    started = asyncio.Event()
    cancelled = {"value": False}

    async def _fake_run(
        redis_client: Any,
        *,
        kill_event_writer_fn: Any,
        pattern: str = KILL_SWITCH_PATTERN,
    ) -> None:
        started.set()
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    monkeypatch.setattr(main_mod, "run_kill_subscription", _fake_run)
    monkeypatch.setenv("OPERATOR_TOKEN", "tok-8g-kill")

    app = main_mod.app
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=5.0)
        task = app.state.kill_subscription_task
        assert not task.done(), "subscription task should be running inside the lifespan"

    # After exit, the AsyncExitStack callback cancelled the task.
    assert cancelled["value"] is True, "subscription task was not cancelled on shutdown"
    assert app.state.kill_subscription_task.done()
