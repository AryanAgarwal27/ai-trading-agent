"""Stage 8 end-to-end DoD test (BRD §13 row 8).

DoD verbatim: "end-to-end: paper graduates to live_gate; human approves; live
spawns; synthetic drawdown triggers kill switch in < 5 min and routes thread to
live_pause."

This drives the FULL chain through the composed parent graph
(``build_per_strategy_graph``) with the REAL production plumbing wherever it is
safe to run in CI, stubbing ONLY what is not safely runnable:

REAL (production code, exercised):
  - the LangGraph parent composition + all routing (_route_after_paper,
    _route_after_live_wait, the live_pause payload discriminator);
  - the strategy_registry writes (live_spawn's real registry_writer);
  - kill_switch_poll_job (reason computation, the Fork-5d already-fired guard,
    record_kill_switch_event → real Postgres kill_switch_events row);
  - publish_kill → REAL Redis publish on ai-trading-agent:kill_switch:<sid>;
  - run_kill_subscription → REAL Redis PSUBSCRIBE + make_kill_event_writer
    (aget_state → merge artifacts → aupdate_state);
  - live_pause's stop-trading call + the kill-switch interrupt payload.

STUBBED (not safely runnable in CI — and NOT part of Stage 8's DoD):
  - paper + live container spawn (Docker) → synthetic api_url + container_id;
  - the kill switch's Freqtrade REST client → controlled {"max_drawdown": 0.13};
  - LLM seams (paper monitor, reviewers, coordinator merge/arbitrate) →
    deterministic verdicts (8h validates WIRING, not LLM reasoning);
  - stop_trading / start_trading / live container teardown → recorded mocks.

TIMING: BRD §13 row 8's "< 5 min" is the APScheduler poll interval
(``KILL_SWITCH_INTERVAL_MINUTES = 5`` in scheduler.py), NOT a wall-clock budget.
The job fires synchronously when invoked; this test invokes it directly. Per
DEFERRED.md D-6, the production live-wake mechanism is a later sub-stage, so this
test drives the post-kill wake via ``Command(resume=...)`` — exactly D-6's
"acceptable for tests" pattern. The test asserts CORRECTNESS (the chain routes
to live_pause) and bounds the kill→route wall time to prove there is no
5-minute sleep in the code path — it does NOT assert interval timing.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import psycopg
import pytest
import redis.asyncio as aioredis
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from orchestrator.agents.coordinator import REVIEWER_AGENTS, LiveVerdict
from orchestrator.agents.monitors import PaperMonitorContext, PaperMonitorVerdict
from orchestrator.gates.hitl import autoresume_for_test
from orchestrator.graph import build_per_strategy_graph
from orchestrator.kill_subscription import (
    cancel_kill_subscription,
    make_kill_event_writer,
    run_kill_subscription,
)
from orchestrator.scheduler import kill_switch_poll_job
from orchestrator.state import StrategyState
from orchestrator.subgraphs.paper import build_paper_subgraph
from orchestrator.subgraphs.validation import ValidationState, paper_gate

pytestmark = pytest.mark.integration

STRATEGY_PATH = "strategy_templates/mean_reversion_template.py"


# ───────────────────────── DB helpers / fixtures ─────────────────────────


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)


@pytest.fixture
async def cleanup_strategy_ids() -> Any:
    ids: list[str] = []
    yield ids
    if not ids:
        return
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM kill_switch_events WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,))
        await conn.commit()


async def _registry_row(strategy_id: str) -> dict[str, Any] | None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT stage, freqtrade_api_url, live_started_at FROM strategy_registry "
                "WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"stage": row[0], "freqtrade_api_url": row[1], "live_started_at": row[2]}


async def _kill_switch_event_row(strategy_id: str) -> dict[str, Any] | None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT reason, action_taken FROM kill_switch_events "
                "WHERE strategy_id = %s ORDER BY fired_at DESC LIMIT 1",
                (strategy_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"reason": row[0], "action_taken": row[1]}


async def _gate_audit_count(strategy_id: str) -> int:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT count(*) FROM gate_audits WHERE strategy_id = %s", (strategy_id,)
            )
            row = await cur.fetchone()
    return int(row[0]) if row else 0


# ───────────────────────── stub seams ─────────────────────────


class _StubRestClient:
    """Stub Freqtrade REST client for the kill switch — async ctx mgr."""

    def __init__(self, *, profit: dict[str, Any], trades: dict[str, Any], calls: dict[str, int]):
        self._profit = profit
        self._trades = trades
        self._calls = calls

    async def __aenter__(self) -> _StubRestClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def profit(self) -> dict[str, Any]:
        return self._profit

    async def trades(self, limit: int = 50) -> dict[str, Any]:
        return self._trades

    async def stop(self) -> dict[str, Any]:
        self._calls["stop"] = self._calls.get("stop", 0) + 1
        return {"status": "stopped"}


class _StubProvider:
    """Distinct fixture creds (live vs paper) — passed to live_spawn but never
    exercised because the spawn helper itself is stubbed."""

    _VALUES = {
        "BINANCE_LIVE_API_KEY": "LIVE-key-e2e",
        "BINANCE_LIVE_API_SECRET": "LIVE-secret-e2e",
        "BINANCE_PAPER_API_KEY": "PAPER-key-e2e",
        "BINANCE_PAPER_API_SECRET": "PAPER-secret-e2e",
    }

    def get(self, name: str) -> str | None:
        return self._VALUES.get(name)


def _one_live(strategy_id: str, url: str) -> Callable[[], Awaitable[list[tuple[str, str]]]]:
    """Inject a single-strategy live list (test isolation — see review note)."""

    async def _fn() -> list[tuple[str, str]]:
        return [(strategy_id, url)]

    return _fn


_LIVE_CONTINUE_SNAP = {
    "max_drawdown": 0.0,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [],
    "paper_returns": [],
    "current_regime": "mid_vol_up",
    "approval_regime": "mid_vol_up",
}


def _build_e2e_graph(saver: Any, rec: dict[str, Any]) -> Any:
    """Composed parent graph: stub research + paper_gate-only validation + real
    paper subgraph (stubbed leaves) + real live subgraph (stubbed leaves)."""

    # research passthrough → routes to validation
    def _research_passthrough(_state: StrategyState) -> dict[str, Any]:
        return {}

    rb: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = StateGraph(
        StrategyState
    )
    rb.add_node("research_pass", _research_passthrough)
    rb.add_edge(START, "research_pass")
    rb.add_edge("research_pass", END)
    research = rb.compile()

    vb: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = StateGraph(
        ValidationState
    )
    vb.add_node("paper_gate", paper_gate)
    vb.add_edge(START, "paper_gate")
    vb.add_edge("paper_gate", END)
    validation = vb.compile()

    # ── paper subgraph leaves
    async def _paper_spawn(*, port: int, **_k: Any) -> str:
        rec["paper_spawned"] = True
        return f"http://127.0.0.1:{port}"

    async def _paper_ctx(_state: Any) -> PaperMonitorContext:
        return PaperMonitorContext(profit={"max_drawdown": 0.04})

    async def _paper_monitor_advance(_ctx: PaperMonitorContext) -> PaperMonitorVerdict:
        return PaperMonitorVerdict(
            decision="advance",
            primary_observation="obs-advance",
            rationale="30-day paper window complete; metrics within tolerance",
            confidence=0.9,
        )

    async def _paper_stop(_sid: str) -> None:
        rec["paper_stopped"] = True

    paper = build_paper_subgraph(
        spawn_container_fn=_paper_spawn,
        paper_monitor_fn=_paper_monitor_advance,
        build_context_fn=_paper_ctx,
        stop_container_fn=_paper_stop,
    )

    # ── live subgraph leaves
    async def _live_spawn(*, port: int, **_k: Any) -> tuple[str, str]:
        rec["live_spawned"] = True
        return (f"http://127.0.0.1:{port}", f"container_{port}")

    async def _live_stop_container(_sid: str) -> None:
        rec["live_teardown"] = True

    async def _rationale(_facts: dict[str, Any]) -> str:
        return "stub rationale"

    async def _review(_c: dict[str, Any]) -> LiveVerdict:
        return LiveVerdict(verdict="continue", rationale="perf", confidence=0.8)

    async def _merge(_v: Any) -> LiveVerdict:
        return LiveVerdict(verdict="continue", rationale="merge", confidence=0.7)

    async def _arbitrate(_v: Any, _m: Any) -> LiveVerdict:
        return LiveVerdict(verdict="fail", rationale="arb", confidence=0.9)

    async def _snapshot(_s: Any) -> dict[str, Any]:
        return dict(_LIVE_CONTINUE_SNAP)

    async def _stop_trading(_state: Any) -> None:
        rec["stop_trading_called"] = rec.get("stop_trading_called", 0) + 1

    async def _start_trading(_state: Any) -> None:
        rec["start_trading_called"] = rec.get("start_trading_called", 0) + 1

    async def _bump(_sid: str) -> None:
        rec["bump_called"] = rec.get("bump_called", 0) + 1

    return build_per_strategy_graph(
        saver,
        None,
        research_subgraph=research,
        validation_subgraph=validation,
        paper_subgraph=paper,
        # live subgraph: stub external effects; keep REAL registry_writer +
        # gate_audit_writer (both hit real Postgres).
        live_subgraph=None,
        spawn_live_container_fn=_live_spawn,
        stop_live_container_fn=_live_stop_container,
        stop_trading_fn=_stop_trading,
        start_trading_fn=_start_trading,
        live_started_bump_fn=_bump,
        live_secrets_provider=cast(Any, _StubProvider()),
        build_snapshot_fn=_snapshot,
        rationale_fn=_rationale,
        review_fn=_review,
        merge_fn=_merge,
        arbitrate_fn=_arbitrate,
    )


def _initial_state(strategy_id: str, *, paper_started_days_ago: int = 31) -> dict[str, Any]:
    started = (datetime.now(UTC) - timedelta(days=paper_started_days_ago)).isoformat()
    return {
        "strategy_id": strategy_id,
        "name": f"e2e-{strategy_id}",
        "hypothesis": "stage 8 e2e",
        "template": "mean_reversion_template",
        "params": {"stake_amount": 25.0},
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "stage": "research",
        "gate_decisions": {},
        "agent_votes": [],
        "artifacts": {
            "generated_strategy_path": STRATEGY_PATH,
            "paper_started_at": started,
        },
    }


async def _parked_kind(graph: Any, config: dict[str, Any]) -> str | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()):
            val = intr.value
            if isinstance(val, dict):
                return cast(str | None, val.get("kind"))
    return None


async def _parent_interrupt_payload(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()):
            return cast(dict[str, Any], intr.value)
    return None


async def _wait_until(
    predicate: Callable[[], Awaitable[bool]], *, timeout_s: float = 5.0, interval: float = 0.02
) -> bool:
    import asyncio

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ───────────────────────── the e2e test ─────────────────────────


async def test_stage_8_e2e_paper_to_live_to_killswitch_pause(
    cleanup_strategy_ids: list[str], hitl_autoapprove: Any
) -> None:
    """BRD §13 row 8: paper → live_gate → approve → live spawn → synthetic
    drawdown → kill switch → routed to live_pause (kill-switch path)."""
    import asyncio

    strategy_id = f"e2e-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)
    thread_id = f"strategy_{strategy_id}"
    config = {"configurable": {"thread_id": thread_id}}
    rec: dict[str, Any] = {}

    graph = _build_e2e_graph(InMemorySaver(), rec)

    # ── 1. Drive paper → live_gate ───────────────────────────────────────
    async for _ in graph.astream(_initial_state(strategy_id), config=config):
        pass
    assert await _parked_kind(graph, config) == "paper_gate"

    await hitl_autoapprove(graph, thread_id)  # paper_gate approve → paper subgraph
    assert await _parked_kind(graph, config) == "paper_wait"
    assert rec.get("paper_spawned") is True

    await autoresume_for_test(graph, thread_id, {"wake": True})  # wake → monitor(advance)
    assert await _parked_kind(graph, config) == "live_gate"

    # ── 2. Human approves live_gate → live spawns ────────────────────────
    await hitl_autoapprove(graph, thread_id)
    assert await _parked_kind(graph, config) == "live_wait", "live thread must park at live_wait"
    assert rec.get("live_spawned") is True, "live_spawn must run on graduation"

    row = await _registry_row(strategy_id)
    assert row is not None and row["stage"] == "live"
    assert row["live_started_at"] is not None, "live_spawn must set live_started_at"
    assert row["freqtrade_api_url"] is not None

    # ── 3. Synthetic drawdown → kill switch (real Redis pub + sub) ────────
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    redis_client = aioredis.from_url(redis_url)

    real_writer = make_kill_event_writer(graph)
    writer_captures: list[tuple[str, dict[str, Any]]] = []

    async def _capturing_writer(sid: str, event: dict[str, Any]) -> None:
        writer_captures.append((sid, dict(event)))
        await real_writer(sid, event)

    baseline_numpat = await redis_client.pubsub_numpat()
    sub_task = asyncio.create_task(
        run_kill_subscription(redis_client, kill_event_writer_fn=_capturing_writer)
    )
    elapsed: float
    try:
        # Gate: do not publish until our PSUBSCRIBE has registered (Redis
        # pub/sub drops messages published before subscribe).
        subscribed = await _wait_until(
            lambda: _numpat_above(redis_client, baseline_numpat), timeout_s=5.0
        )
        assert subscribed, "kill subscription did not PSUBSCRIBE in time"

        stop_calls: dict[str, int] = {}
        stub_rest = _StubRestClient(
            profit={"max_drawdown": 0.13}, trades={"trades": []}, calls=stop_calls
        )

        t0 = time.monotonic()
        # REAL kill_switch_poll_job: real already-fired guard, real
        # record_kill_switch_event (Postgres), real publish_kill (Redis).
        # Only the REST client + the live-list are injected (isolation).
        await kill_switch_poll_job(
            list_live_fn=_one_live(strategy_id, row["freqtrade_api_url"]),
            rest_client_factory=lambda _url: stub_rest,
        )
        assert stop_calls.get("stop") == 1, "/api/v1/stop must be called once on the breach"

        # The real subscription receives the real publish and writes state.
        wrote = await _wait_until(lambda: _event_in_state(graph, config), timeout_s=5.0)
        elapsed = time.monotonic() - t0
        assert wrote, "kill event did not propagate to thread state via the subscription"
    finally:
        await cancel_kill_subscription(sub_task)
        await redis_client.aclose()

    # kill_switch_events row (real Postgres) — reason + action_taken
    ks_row = await _kill_switch_event_row(strategy_id)
    assert ks_row is not None
    assert ks_row["reason"] == "drawdown_12pct_exceeded"
    assert ks_row["action_taken"] == "POST /api/v1/stop"

    # The subscription parsed strategy_id from the real channel + carried the
    # published payload (incl action_taken) through.
    assert writer_captures, "subscription writer never fired"
    cap_sid, cap_event = writer_captures[0]
    assert cap_sid == strategy_id, "channel suffix must parse to our strategy_id"
    assert cap_event["action_taken"] == "POST /api/v1/stop"
    assert cap_event["reason"] == "drawdown_12pct_exceeded"

    # Timing: bounds correctness, not the 5-min interval (see module docstring).
    assert elapsed < 30.0, f"kill→state propagation took {elapsed:.2f}s (no 5-min sleep expected)"

    # ── 4. Wake the thread (D-6 test-driven wake) → routes to live_pause ──
    #
    # FRAMEWORK-BEHAVIOR REGRESSION GUARD. The subscription wrote the kill event
    # via a PARENT-level graph.aupdate_state on a thread parked at a NESTED
    # (live_subgraph → live_wait) interrupt. Empirically (8h diagnostic), that
    # parent-level update CLEARS the parent-visible interrupt surface — the
    # nested subgraph stays parked and resumable via Command(resume=...), but
    # aget_state().tasks[*].interrupts goes empty at the parent. The two asserts
    # below lock that in: if a future LangGraph release changes how aupdate_state
    # affects nested-subgraph interrupt visibility, this fails loudly.
    pre_wake = await graph.aget_state(config)
    assert (pre_wake.values.get("artifacts") or {}).get(
        "kill_switch_event"
    ) is not None, "kill event must be present in parent state after the subscription write"
    assert sum(len(getattr(t, "interrupts", ())) for t in pre_wake.tasks) == 0, (
        "parent-level aupdate_state on a nested-interrupt thread clears the "
        "parent-visible interrupt surface (8h finding — see DEFERRED.md D-6)"
    )

    # CONSTRAINT: because the kill event cleared the parent-visible interrupt,
    # this wake CANNOT use autoresume_for_test / hitl_autoapprove — their guard
    # ("is there a parent-visible interrupt?") would false-negative here. The
    # nested live_wait is still parked and resumes correctly via a raw
    # Command(resume=...), which is exactly how the production live-wake
    # mechanism (DEFERRED.md D-6, not yet built) must resume the thread: NOT via
    # the interrupt-presence-gated /wake endpoint. The EARLIER wakes in this test
    # (paper_gate, paper_wait, live_gate) correctly DO use autoresume_for_test /
    # hitl_autoapprove because no aupdate_state intervened — those threads are
    # genuinely parent-visible-parked at the time they are resumed.
    async for _ in graph.astream(Command(resume={"wake": True}), config=config):
        pass

    assert rec.get("stop_trading_called") == 1, "live_pause must POST /stop (halt)"

    payload = await _parent_interrupt_payload(graph, config)
    assert payload is not None and payload["kind"] == "live_pause_review"
    summary = payload["summary"]
    assert summary["path"] == "kill_switch", "kill-switch discriminator must be taken"
    assert summary["coordinator"] is None, "no coordinator vote on the kill path (BRD §5.6)"
    assert summary["reviewer_votes"] is None
    assert summary["kill_switch_event"]["reason"] == "drawdown_12pct_exceeded"
    assert summary["kill_switch_event"]["action_taken"] == "POST /api/v1/stop"

    # The reviewer fan-out (live_evaluate) was skipped on the kill path.
    snap = await graph.aget_state(config)
    reviewer_votes = [
        v for v in snap.values.get("agent_votes", []) if v["agent"] in REVIEWER_AGENTS
    ]
    assert reviewer_votes == [], "live_evaluate must be skipped on the kill path"

    # gate_audits writes nothing for the kill path — the audit happens on the
    # operator's RESUME of live_pause (not driven here), and no coordinator ran.
    assert await _gate_audit_count(strategy_id) == 0


async def _event_in_state(graph: Any, config: dict[str, Any]) -> bool:
    snap = await graph.aget_state(config)
    return bool((snap.values.get("artifacts") or {}).get("kill_switch_event"))


async def _numpat_above(redis_client: Any, baseline: int) -> bool:
    return bool(await redis_client.pubsub_numpat() > baseline)
