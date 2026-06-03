"""Stage 8 live subgraph (BRD §5.6) — live_spawn node only.

This 8c commit ships ONLY ``live_spawn`` + the ``LiveState`` skeleton, mirroring
how 7c shipped only ``paper_spawn`` before 7e wired the rest of the paper
subgraph. The other live nodes (``live_wait``, ``live_evaluate`` Send fan-out,
``risk_check`` / ``performance_check`` / ``regime_check``, ``coordinator``,
``live_pause``) land in 8d/8e.

``live_spawn`` mirrors ``paper_spawn``'s contract (BRD §5.6):

1. Derive the per-trade ``stake_amount`` from operator intent, capped to
   ``LIVE_CAPITAL_CAP_USD`` (SPEC §1 Q3) — belt-and-braces with the cap
   render_live_config also applies (the node knows the strategy's intent; the
   renderer only sees the float it's handed).
2. Allocate a host port from the LIVE range via :func:`next_free_live_port`,
   reading currently-used ports from ``strategy_registry``. The live range is
   disjoint from paper's, so paper-range ports never influence the allocation.
3. **Upsert the registry row BEFORE spawn** (stage='live', userdir set). Same
   orphan-container-prevention contract as paper: a live container placing real
   orders must never be untracked.
4. Spawn via ``spawn_live_container`` (injected for tests). On ANY failure,
   record the failure on the registry row (audit trail — not deleted) and
   return ``{stage: "archived", failure_reason: "live_spawn_failed: …"}`` —
   never raise out of the node.
5. On success: update the registry row with the api_url and return state with
   ``stage="live"``, ``freqtrade_api_url``, ``freqtrade_userdir``,
   ``artifacts.live_started_at`` (UTC ISO 8601), and ``artifacts.live_container_id``
   when the spawn helper returns it.

ASYMMETRY WITH PAPER (flagged, not refactored): ``paper_spawn`` injects only
``spawn_container_fn`` — its registry write is inline and its credentials are
loaded inside ``spawn_paper_container``. ``live_spawn`` additionally exposes a
``registry_writer_fn`` seam and threads a ``secrets_provider`` seam through to
the spawn helper, which makes the live tests DB-/Docker-stubbable and the live
path's credential source injectable. Backporting these seams to paper is a
possible follow-up; this commit does not touch paper.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send, interrupt

from orchestrator.agents.coordinator import (
    ArbitrateFn,
    GateAuditWriterFn,
    LiveVerdict,
    MergeFn,
    coordinator,
    latest_reviewer_votes,
)
from orchestrator.gates.hitl import build_interrupt_payload
from orchestrator.gates.thresholds import (
    DAILY_LOSS_LIMIT_PCT,
    KILL_SWITCH_CONSECUTIVE_LOSSES,
    KILL_SWITCH_DRAWDOWN,
    LIVE_CAPITAL_CAP_USD,
    MAX_OPEN_TRADES,
)
from orchestrator.observability.events import publish_gate_pending, record_gate_audit
from orchestrator.security.secrets import EnvSecretProvider, SecretProvider
from orchestrator.subgraphs.paper import PaperState
from orchestrator.tools.compare import compare_paper_to_backtest
from orchestrator.tools.freqtrade_api import FreqtradeAPI, FreqtradeCredentials
from orchestrator.tools.freqtrade_lifecycle import (
    LIVE_WORKERS_ROOT,
    next_free_live_port,
    spawn_live_container,
    stop_live_container,
)

# Reviewer LLM seams (8e wires the real agents; tests + 8d pass stubs).
RationaleFn = Callable[[dict[str, Any]], Awaitable[str]]
PerformanceReviewFn = Callable[[dict[str, Any]], Awaitable[LiveVerdict]]

logger = logging.getLogger(__name__)

# Injection seams (8c). spawn returns the api_url, or a (api_url, container_id)
# tuple when the helper surfaces a container id.
SpawnLiveContainerFn = Callable[..., Awaitable[str | tuple[str, str]]]
RegistryWriterFn = Callable[..., Awaitable[None]]

# 8e assembly seams.
# build_snapshot_fn is the live analog of paper's build_context_fn — fetches the
# per-cycle metrics from the live container before the reviewer fan-out.
BuildSnapshotFn = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]
# stop_trading_fn POSTs /api/v1/stop only (halt) — NOT a teardown. live_pause
# must leave the container queryable/resumable for the next wake (BRD §5.6).
StopTradingFn = Callable[[Mapping[str, Any]], Awaitable[None]]
StopContainerFn = Callable[[str], Awaitable[None]]

# BaseCheckpointSaver is generic on its serializer; accept any concrete saver
# (InMemorySaver for tests, AsyncPostgresSaver for production) — mirror paper.py.
CheckpointSaver = BaseCheckpointSaver[Any]


# ════════════════════════════════════════════════════════════════════════
# Live subgraph state (skeleton — full topology lands in 8e)
# ════════════════════════════════════════════════════════════════════════


class LiveState(PaperState, total=False):
    """Workspace state for the live subgraph.

    Inherits the PaperState channels (strategy_id, name, template, pairs,
    timeframe, params, stage, agent_votes reducer, gate_decisions,
    freqtrade_api_url / _userdir / _process_id, artifacts, failure_reason) and
    adds the live-specific inputs the research → live handoff carries.

    The live container handle reuses the inherited ``freqtrade_process_id``
    (str since SPEC 7c) rather than introducing a separate field, so the value
    propagates cleanly through the parent StrategyState (BRD §5.7) in 8g; the
    human-friendly container id, when the spawn helper returns one, is stored
    under ``artifacts.live_container_id``.
    """

    # Research handoff input — the generated strategy module. live_spawn falls
    # back to artifacts.generated_strategy_path (the paper source) if unset.
    strategy_path: str
    # Operator stake intent (pre-cap); capped to LIVE_CAPITAL_CAP_USD at spawn.
    stake_amount: float
    # Per-cycle live metrics snapshot the 8e live_evaluate node fetches and the
    # three reviewer nodes read (drawdown / daily P&L / consecutive losses,
    # live-vs-paper return samples, current + approval-time regime). Transient
    # (re-fetched each wake), not a durable channel.
    live_snapshot: dict[str, Any]


# ───────────────────────── db helpers (mirror paper.py) ─────────────────────────


def _libpq_dsn(sqlalchemy_url: str) -> str:
    """Strip the ``postgresql+psycopg://`` SQLAlchemy prefix for libpq."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _connect_app_db() -> psycopg.AsyncConnection:
    """Open a fresh async psycopg connection to the app DB (mirror paper.py)."""
    return await psycopg.AsyncConnection.connect(_libpq_dsn(os.environ["DATABASE_URL"]))


def _used_ports_from_urls(urls: list[Any]) -> list[int]:
    """Extract port numbers from ``freqtrade_api_url`` values (tolerant)."""
    ports: list[int] = []
    for url in urls:
        if not isinstance(url, str):
            continue
        try:
            parsed = urlparse(url)
            if parsed.port is not None:
                ports.append(parsed.port)
        except ValueError:
            continue
    return ports


def _thread_id_for(config: RunnableConfig, strategy_id: str) -> str:
    """Return the thread_id from config, or the canonical fallback (BRD §5.1)."""
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return f"strategy_{strategy_id}"


async def _read_used_ports() -> list[int]:
    """Read every assigned freqtrade port from the registry (paper + live).

    Returns ALL used ports; :func:`next_free_live_port` filters to the live
    range, so paper-range ports are naturally ignored — the cross-namespace
    isolation lives in the range filter, not this query.
    """
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT freqtrade_api_url FROM strategy_registry "
                "WHERE freqtrade_api_url IS NOT NULL"
            )
            rows = await cur.fetchall()
    finally:
        await conn.close()
    return _used_ports_from_urls([r[0] for r in rows])


async def _write_live_registry(
    *,
    strategy_id: str,
    thread_id: str,
    name: str,
    template: str,
    pairs: list[str],
    timeframe: str,
    stage: str,
    userdir: str | None,
    api_url: str | None = None,
    failure_reason: str | None = None,
) -> None:
    """Default ``registry_writer_fn`` — upsert the live registry row.

    Called by the node (a) before spawn with stage='live' + userdir, (b) on
    success with api_url, (c) on failure with stage='archived' + failure_reason.
    ``COALESCE`` on userdir/api_url preserves an already-set value across the
    multi-call sequence (the pre-spawn upsert passes api_url=None and must not
    null out a later success write on replay).
    """
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO strategy_registry
                  (strategy_id, thread_id, name, template, stage, pairs,
                   timeframe, freqtrade_userdir, freqtrade_api_url,
                   failure_reason, started_at, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (strategy_id) DO UPDATE SET
                  stage = EXCLUDED.stage,
                  freqtrade_userdir = COALESCE(
                      EXCLUDED.freqtrade_userdir, strategy_registry.freqtrade_userdir),
                  freqtrade_api_url = COALESCE(
                      EXCLUDED.freqtrade_api_url, strategy_registry.freqtrade_api_url),
                  failure_reason = EXCLUDED.failure_reason,
                  last_updated = now()
                """,
                (
                    strategy_id,
                    thread_id,
                    name,
                    template,
                    stage,
                    json.dumps(pairs),
                    timeframe,
                    userdir,
                    api_url,
                    failure_reason,
                ),
            )
        await conn.commit()
    finally:
        await conn.close()


# ════════════════════════════════════════════════════════════════════════
# live_spawn node
# ════════════════════════════════════════════════════════════════════════


async def live_spawn(
    state: LiveState,
    config: RunnableConfig,
    *,
    spawn_live_container_fn: SpawnLiveContainerFn | None = None,
    secrets_provider: SecretProvider | None = None,
    registry_writer_fn: RegistryWriterFn | None = None,
) -> dict[str, Any]:
    """First node in the live subgraph — see module docstring for contract.

    Returns a state-update dict. Never raises on spawn failure; failures route
    to ``stage="archived"`` with a ``live_spawn_failed:`` ``failure_reason`` and
    are recorded on the registry row (audit trail).
    """
    spawn_fn: SpawnLiveContainerFn = spawn_live_container_fn or spawn_live_container
    provider: SecretProvider = secrets_provider or EnvSecretProvider()
    write_registry: RegistryWriterFn = registry_writer_fn or _write_live_registry

    strategy_id = state["strategy_id"]
    pairs = state["pairs"]
    artifacts = state.get("artifacts") or {}

    # 1. Stake: operator intent (explicit field, else params, else BRD §10
    # default), capped to LIVE_CAPITAL_CAP_USD (SPEC §1 Q3).
    params = state.get("params") or {}
    stake_intent = float(
        state.get("stake_amount")
        or params.get("stake_amount")
        or (LIVE_CAPITAL_CAP_USD / MAX_OPEN_TRADES)
    )
    capped_stake = min(stake_intent, float(LIVE_CAPITAL_CAP_USD))

    # Generated strategy path: explicit channel, else the research artifact.
    strategy_path_str = state.get("strategy_path") or artifacts.get("generated_strategy_path")
    if not strategy_path_str:
        logger.error("live_spawn missing strategy path strategy_id=%s", strategy_id)
        return {
            "stage": "archived",
            "failure_reason": (
                "live_spawn_missing_strategy_path: no 'strategy_path' nor "
                "artifacts.generated_strategy_path — research must run first"
            ),
        }
    strategy_path = Path(strategy_path_str)

    thread_id = _thread_id_for(config, strategy_id)
    userdir = str(LIVE_WORKERS_ROOT / strategy_id)
    identity = {
        "strategy_id": strategy_id,
        "thread_id": thread_id,
        "name": state["name"],
        "template": state["template"],
        "pairs": pairs,
        "timeframe": state["timeframe"],
    }

    # 2 + 3. Allocate the live port and upsert the registry row BEFORE spawn.
    used = await _read_used_ports()
    port = next_free_live_port(used)
    await write_registry(**identity, stage="live", userdir=userdir)

    logger.info(
        "live_spawn allocated port=%d strategy_id=%s thread_id=%s",
        port,
        strategy_id,
        thread_id,
    )

    # 4. Spawn — any failure archives cleanly AND records on the registry row.
    try:
        result = await spawn_fn(
            strategy_id=strategy_id,
            pair_whitelist=pairs,
            stake_amount=capped_stake,
            strategy_module_path=strategy_path,
            port=port,
            provider=provider,
        )
    except Exception as exc:  # noqa: BLE001 — any spawn failure archives cleanly
        reason = f"live_spawn_failed: {type(exc).__name__}: {str(exc)[:200]}"
        logger.error(
            "live_spawn failed strategy_id=%s port=%d exc_type=%s exc=%s",
            strategy_id,
            port,
            type(exc).__name__,
            exc,
        )
        await write_registry(**identity, stage="archived", userdir=userdir, failure_reason=reason)
        return {"stage": "archived", "failure_reason": reason}

    # Normalize: helper may return just the api_url, or (api_url, container_id).
    if isinstance(result, tuple):
        api_url, container_id = result
    else:
        api_url, container_id = result, None

    # 5. Update registry with the URL and return the success state-update.
    await write_registry(**identity, stage="live", userdir=userdir, api_url=api_url)

    # Preserve an existing live_started_at across re-runs (replay / re-spawn).
    started_iso = artifacts.get("live_started_at") or datetime.now(UTC).isoformat()
    new_artifacts: dict[str, Any] = {**artifacts, "live_started_at": started_iso}
    if container_id:
        new_artifacts["live_container_id"] = container_id

    return {
        "stage": "live",
        "freqtrade_api_url": api_url,
        "freqtrade_userdir": userdir,
        "freqtrade_process_id": container_id or f"live-{strategy_id}",
        "artifacts": new_artifacts,
    }


# ════════════════════════════════════════════════════════════════════════
# Reviewer nodes (Stage 8d) — BRD §5.6 live_evaluate Send fan-out targets
# ════════════════════════════════════════════════════════════════════════
#
# Each reviewer reads the per-cycle ``live_snapshot`` and appends ONE AgentVote
# (verdict ∈ {continue, pause, fail}). They write ONLY ``agent_votes`` (the
# Annotated[list, add] reducer field) — never ``gate_decisions`` — because they
# run in parallel under a Send fan-out, and ``gate_decisions`` has no reducer so
# concurrent writes would clobber. The coordinator (runs after the barrier)
# reads the votes and owns the gate_decisions["coordinator"] write.


def _reviewer_vote(agent: str, verdict: str, rationale: str, confidence: float) -> dict[str, Any]:
    """A reviewer's state-update: one appended vote, reducer-safe."""
    return {
        "agent_votes": [
            {
                "agent": agent,
                "verdict": verdict,
                "rationale": rationale,
                "confidence": confidence,
            }
        ]
    }


# ─── risk_check: plain verdict + small LLM rationale (BRD §5.6) ──────────


async def _default_risk_rationale(facts: dict[str, Any]) -> str:
    """Default ``rationale_fn`` — a small Haiku call summarizing the facts."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    model = ChatAnthropic(model="claude-haiku-4-5-20251001", timeout=30.0, stop=None)
    msg = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "You write a single terse sentence explaining a live-trade "
                    "risk verdict from the given facts. Cite the numbers."
                )
            ),
            HumanMessage(content=json.dumps(facts)),
        ]
    )
    return str(msg.content)


async def risk_check(
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
    *,
    rationale_fn: RationaleFn | None = None,
) -> dict[str, Any]:
    """Reviewer: drawdown / daily P&L / consecutive losses → continue|pause.

    Verdict is DETERMINISTIC against the BRD §10 live thresholds; a small LLM
    (``rationale_fn``) only writes the human rationale. risk_check never votes
    "fail" (archive) — a threshold breach routes to pause/HITL; the hard stop
    is the out-of-band kill switch (Stage 8f), not this advisory vote.
    """
    snap = state.get("live_snapshot") or {}
    max_dd = float(snap.get("max_drawdown", 0.0) or 0.0)
    daily = float(snap.get("daily_pnl_pct", 0.0) or 0.0)
    losses = int(snap.get("consecutive_losses", 0) or 0)

    breaches: list[str] = []
    if max_dd >= KILL_SWITCH_DRAWDOWN:
        breaches.append(f"max_drawdown={max_dd:.3f}>={KILL_SWITCH_DRAWDOWN}")
    if daily <= -DAILY_LOSS_LIMIT_PCT:
        breaches.append(f"daily_pnl_pct={daily:.3f}<=-{DAILY_LOSS_LIMIT_PCT}")
    if losses >= KILL_SWITCH_CONSECUTIVE_LOSSES:
        breaches.append(f"consecutive_losses={losses}>={KILL_SWITCH_CONSECUTIVE_LOSSES}")

    verdict = "pause" if breaches else "continue"
    confidence = 0.9 if breaches else 0.7
    facts = {
        "max_drawdown": max_dd,
        "daily_pnl_pct": daily,
        "consecutive_losses": losses,
        "breaches": breaches,
        "verdict": verdict,
    }
    rat_fn = rationale_fn or _default_risk_rationale
    rationale = await rat_fn(facts)
    return _reviewer_vote("risk_check", verdict, rationale, confidence)


# ─── regime_check: plain lookup + compare (no LLM, BRD §5.6) ────────────


def regime_check(
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Reviewer: current regime vs approval-time regime → continue|pause.

    Pure (no LLM, BRD §5.6). A regime shift from the conditions under which the
    strategy was approved is a pause signal (the edge that justified going live
    may no longer hold). A missing approval regime can't be compared → continue
    at low confidence.
    """
    snap = state.get("live_snapshot") or {}
    current = snap.get("current_regime")
    approval = snap.get("approval_regime")

    if not approval:
        return _reviewer_vote(
            "regime_check",
            "continue",
            "no approval-time regime recorded; cannot compare regimes",
            0.3,
        )
    if current == approval:
        return _reviewer_vote(
            "regime_check",
            "continue",
            f"regime unchanged since approval ({approval})",
            0.8,
        )
    return _reviewer_vote(
        "regime_check",
        "pause",
        f"regime shifted {approval} -> {current} since approval",
        0.8,
    )


# ─── performance_check: Sonnet agent over live-vs-paper drift (BRD §5.6) ─


async def _default_performance_review(comparison: dict[str, Any]) -> LiveVerdict:
    """Default ``review_fn`` — a Sonnet 4.6 agent judging the drift comparison."""
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    model = ChatAnthropic(model="claude-sonnet-4-6", timeout=60.0, stop=None)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=_PERFORMANCE_PROMPT,
        response_format=LiveVerdict,
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=json.dumps(comparison))]})
    return cast(LiveVerdict, result["structured_response"])


_PERFORMANCE_PROMPT = """\
You are the live-trade Performance reviewer. You are given a KS-test + Sharpe
comparison of LIVE per-trade returns vs the PAPER returns that justified going
live. If diverged is true on a meaningful sample (n_paper and n_live both
reasonable), the live behaviour no longer matches paper — vote pause (or fail
if the divergence is severe and the sample is large). If not diverged, vote
continue. With a tiny sample, prefer continue and wait for more data. Emit a
LiveVerdict.
"""


async def performance_check(
    state: Mapping[str, Any],
    config: RunnableConfig | None = None,
    *,
    review_fn: PerformanceReviewFn | None = None,
) -> dict[str, Any]:
    """Reviewer: live-vs-paper return-distribution drift → continue|pause|fail.

    Computes the KS + relative-Sharpe comparison deterministically (reusing the
    Stage 7 two-sample machinery), then hands it to a Sonnet agent for the
    verdict. The agent call is injectable (``review_fn``).
    """
    snap = state.get("live_snapshot") or {}
    live_returns = [float(x) for x in (snap.get("live_returns") or [])]
    paper_returns = [float(x) for x in (snap.get("paper_returns") or [])]
    comparison = dict(compare_paper_to_backtest(live_returns, paper_returns))

    review = review_fn or _default_performance_review
    verdict = await review(comparison)
    return _reviewer_vote(
        "performance_check", verdict.verdict, verdict.rationale, verdict.confidence
    )


# ════════════════════════════════════════════════════════════════════════
# Subgraph assembly (Stage 8e) — BRD §5.6 topology
# ════════════════════════════════════════════════════════════════════════


def _trailing_losses(trades: list[dict[str, Any]]) -> int:
    """Count the trailing run of losing closed trades (mirror paper.py)."""
    count = 0
    for t in reversed(trades):
        if not isinstance(t, dict):
            break
        pr = t.get("profit_ratio")
        if pr is None:
            break
        if float(pr) < 0:
            count += 1
        else:
            break
    return count


# ─── live_evaluate: fetch the snapshot, then fan out to the reviewers ───


async def _build_live_snapshot_from_container(state: Mapping[str, Any]) -> dict[str, Any]:
    """Default ``build_snapshot_fn`` — live analog of paper's build_context_fn.

    Fetches profit/trades/performance from the live container and derives the
    fields the reviewers read. ``paper_returns`` + the regimes come from
    ``artifacts`` (populated by the paper→live handoff / 8g). The
    ``daily_pnl_pct`` here is a cumulative-percent approximation; the real
    rolling-24h figure is the daily_loss_job's job (Stage 8f).
    """
    api_url = state.get("freqtrade_api_url")
    artifacts = state.get("artifacts") or {}
    base: dict[str, Any] = {
        "max_drawdown": 0.0,
        "daily_pnl_pct": 0.0,
        "consecutive_losses": 0,
        "live_returns": [],
        "paper_returns": artifacts.get("paper_returns", []),
        "current_regime": artifacts.get("current_regime"),
        "approval_regime": artifacts.get("approval_regime"),
    }
    if not api_url:
        return base

    creds = FreqtradeCredentials(
        username="freqtrader",
        password=os.environ.get("BINANCE_LIVE_API_PASSWORD", ""),
    )
    async with FreqtradeAPI(base_url=api_url, credentials=creds) as client:
        profit = await client.profit()
        trades_resp = await client.trades(limit=500)
        await client.performance()

    profit = profit if isinstance(profit, dict) else {}
    trades = trades_resp.get("trades", []) if isinstance(trades_resp, dict) else []
    base["max_drawdown"] = float(profit.get("max_drawdown", 0.0) or 0.0)
    # Approximation (cumulative, not rolling-24h) — see docstring.
    base["daily_pnl_pct"] = float(profit.get("profit_closed_percent", 0.0) or 0.0) / 100.0
    base["consecutive_losses"] = _trailing_losses(trades)
    base["live_returns"] = [
        float(t["profit_ratio"])
        for t in trades
        if isinstance(t, dict) and t.get("profit_ratio") is not None
    ]
    return base


async def live_evaluate(
    state: LiveState,
    config: RunnableConfig | None = None,
    *,
    build_snapshot_fn: BuildSnapshotFn | None = None,
) -> dict[str, Any]:
    """Fetch the per-cycle live snapshot into state, then the conditional edge
    (:func:`_fan_out_reviewers`) Send-fans-out to the three reviewers.

    Splitting fetch (here) from fan-out (the edge) means the snapshot is fetched
    ONCE per wake — the three reviewers read a consistent snapshot rather than
    each re-fetching (metrics would shift mid-cycle, and it'd be 3× the REST
    round-trips).
    """
    fn = build_snapshot_fn or _build_live_snapshot_from_container
    snap = await fn(state)
    return {"live_snapshot": snap}


def _fan_out_reviewers(state: LiveState) -> list[Send]:
    """Router: one Send per reviewer with the snapshot slice (BRD §5.6).

    The reviewers append to the ``agent_votes`` reducer (inherited from
    PaperState — ``Annotated[list[AgentVote], add]``), so the three parallel
    writes concatenate rather than clobber.
    """
    snap = state.get("live_snapshot") or {}
    payload = {"strategy_id": state.get("strategy_id"), "live_snapshot": snap}
    return [
        Send("risk_check", payload),
        Send("performance_check", payload),
        Send("regime_check", payload),
    ]


# ─── live_wait: interrupt() park between wake-cycles ────────────────────


def live_wait(state: LiveState) -> dict[str, Any]:
    """Park the live thread until a wake (BRD §5.6); resume → live_evaluate.

    Wakes via (a) the APScheduler wake job calling the FastAPI /wake endpoint,
    or (b) a kill-switch publish. KILL-SWITCH WAKE PATH: SEE 8f — Stage 8f wires
    the out-of-band kill switch, which publishes a kill event and (8f's change)
    makes this resume route conditionally to live_pause when
    ``artifacts.kill_switch_event`` is set. For 8e there is no kill-switch
    wiring: resume always proceeds to live_evaluate (plain edge below). Bare
    interrupt → no side effects on replay.
    """
    interrupt({"kind": "live_wait", "strategy_id": state.get("strategy_id")})
    return {}


# ─── live_pause: POST /stop (halt), HITL review, resume routing ─────────


async def _default_live_stop_trading(state: Mapping[str, Any]) -> None:
    """Default ``stop_trading_fn`` — POST /api/v1/stop only (halt, not teardown).

    A pause must leave the container queryable/resumable for the next wake
    (BRD §5.6), so this halts trading via the REST API and does NOT compose-down
    the container (that is the archive path's job via stop_live_container).
    """
    api_url = state.get("freqtrade_api_url")
    if not api_url:
        return
    password = os.environ.get("BINANCE_LIVE_API_PASSWORD", "")
    if not password:
        logger.warning("BINANCE_LIVE_API_PASSWORD unset; skipping live /stop")
        return
    creds = FreqtradeCredentials(username="freqtrader", password=password)
    async with FreqtradeAPI(base_url=api_url, credentials=creds, timeout_s=5.0) as client:
        await client.stop()


def _project_reviewer_gate_decisions(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project the latest reviewer votes into ``gate_decisions`` for the payload.

    The reviewers write only ``agent_votes`` (reducer-safe under the fan-out);
    ``build_interrupt_payload``'s live_pause_review shape reads
    ``gate_decisions[<reviewer>]``. Coordinator already wrote
    ``gate_decisions["coordinator"]``; here we add the per-reviewer entries so
    the dashboard payload is complete. Does not mutate state.
    """
    gates = dict(state.get("gate_decisions") or {})
    for v in latest_reviewer_votes(list(state.get("agent_votes") or [])):
        gates[v["agent"]] = {
            "verdict": v["verdict"],
            "rationale": v["rationale"],
            "confidence": v["confidence"],
        }
    return gates


async def live_pause(
    state: LiveState,
    config: RunnableConfig | None = None,
    *,
    stop_trading_fn: StopTradingFn | None = None,
    gate_audit_writer_fn: GateAuditWriterFn | None = None,
) -> Command[Any]:
    """Halt the live bot, surface live_pause_review HITL, route on resume.

    POST /api/v1/stop (best-effort halt; container survives), then
    interrupt() with the SPEC §4.1 live_pause_review payload (coordinator
    branch; the kill-switch branch is taken automatically by
    build_interrupt_payload when 8f sets artifacts.kill_switch_event). On
    resume: approve → live_wait (re-arm), reject → archive. A gate_audits row
    (gate='live_pause', human_approve/human_reject) is written on resume.

    NOTE (diverges from paper's gate pattern): paper_gate/live_gate do NOT write
    gate_audits in-node — the FastAPI /approve endpoint does. live_pause writes
    it in-node so the subgraph's HITL decision is durably recorded without a
    web layer (a dedicated live_pause-resume endpoint is 8g/dashboard territory).
    """
    stop_fn = stop_trading_fn or _default_live_stop_trading
    write_audit = gate_audit_writer_fn or record_gate_audit
    sid = state["strategy_id"]
    thread_id = _thread_id_for(config, sid) if config is not None else f"strategy_{sid}"

    # Best-effort halt — mirror paper_teardown's swallow-and-log; a re-stop on
    # interrupt replay is harmless (the container is already halted).
    try:
        await stop_fn(state)
    except Exception as exc:  # noqa: BLE001 — a stop failure must not strand HITL
        logger.error("live_pause stop failed strategy_id=%s exc=%s", sid, exc)

    gates = _project_reviewer_gate_decisions(state)
    payload = build_interrupt_payload({**state, "gate_decisions": gates}, "live_pause_review")
    await publish_gate_pending(thread_id, payload)

    decision = interrupt(payload)
    existing = state.get("gate_decisions") or {}

    if not isinstance(decision, dict) or "approved" not in decision:
        await write_audit(
            strategy_id=sid,
            gate="live_pause",
            decision="human_reject",
            actor="human",
            payload={"error": "invalid_decision_payload"},
        )
        return Command(
            goto="archive",
            update={
                "stage": "archived",
                "failure_reason": "live_pause_invalid_decision_payload",
                "gate_decisions": {
                    **existing,
                    "live_pause": {
                        "approved": False,
                        "by": "human",
                        "error": "invalid_decision_payload",
                    },
                },
            },
        )

    approved = bool(decision.get("approved", False))
    notes = str(decision.get("notes", ""))
    # Stage 8g may move this to a /live_pause/resume endpoint; until then
    # in-node write is the durable HITL audit record. If 8g adds an endpoint
    # write, drop the in-node call to avoid duplicate gate_audits rows.
    await write_audit(
        strategy_id=sid,
        gate="live_pause",
        decision="human_approve" if approved else "human_reject",
        actor="human",
        payload={"approved": approved, "notes": notes},
    )

    if approved:
        return Command(
            goto="live_wait",
            update={
                "gate_decisions": {
                    **existing,
                    "live_pause": {"approved": True, "notes": notes, "by": "human"},
                }
            },
        )
    return Command(
        goto="archive",
        update={
            "stage": "archived",
            "failure_reason": f"live_pause_rejected: {notes}",
            "gate_decisions": {
                **existing,
                "live_pause": {"approved": False, "notes": notes, "by": "human"},
            },
        },
    )


# ─── archive: terminal teardown sink ────────────────────────────────────


async def live_archive(
    state: LiveState,
    config: RunnableConfig | None = None,
    *,
    stop_container_fn: StopContainerFn | None = None,
) -> dict[str, Any]:
    """Terminal sink: tear the live container down, stamp stage + failure_reason.

    Reached from coordinator (fail) and live_pause (reject). Unlike live_pause's
    halt, archive is terminal → full teardown (stop + compose-down) via
    stop_live_container. Best-effort: a teardown failure is logged, not raised
    (a leaked container is an ops concern surfaced via logs, not a reason to
    strand the graph). Mirrors paper_teardown folded into the terminal node.
    """
    stop_fn = stop_container_fn or stop_live_container
    sid = str(state.get("strategy_id", ""))
    try:
        await stop_fn(sid)
    except Exception as exc:  # noqa: BLE001 — teardown must not strand the thread
        logger.error(
            "live_archive teardown failed strategy_id=%s exc=%s "
            "(possible leaked container — check docker ps)",
            sid,
            exc,
        )
    return {
        "stage": "archived",
        "failure_reason": state.get("failure_reason") or "live_archived_without_reason",
    }


# ─── builder ─────────────────────────────────────────────────────────────


def _route_after_live_spawn(state: LiveState) -> Literal["live_wait", "archive"]:
    """Route live_spawn's dict output: archived → archive, else → live_wait."""
    return "archive" if state.get("stage") == "archived" else "live_wait"


def build_live_subgraph(
    *,
    spawn_live_container_fn: SpawnLiveContainerFn | None = None,
    stop_live_container_fn: StopContainerFn | None = None,
    secrets_provider: SecretProvider | None = None,
    registry_writer_fn: RegistryWriterFn | None = None,
    build_snapshot_fn: BuildSnapshotFn | None = None,
    rationale_fn: RationaleFn | None = None,
    review_fn: PerformanceReviewFn | None = None,
    merge_fn: MergeFn | None = None,
    arbitrate_fn: ArbitrateFn | None = None,
    gate_audit_writer_fn: GateAuditWriterFn | None = None,
    stop_trading_fn: StopTradingFn | None = None,
    checkpointer: CheckpointSaver | None = None,
) -> CompiledStateGraph[LiveState, LiveState, LiveState, LiveState]:
    """Compile the Stage 8e live subgraph (BRD §5.6).

    Topology::

        START → live_spawn ──fail──> archive ──> END
                    │ stage="live"
                    ▼
                live_wait ──(wake / resume)──> live_evaluate
                    ▲                               │ Send×3
                    │                ┌──────────────┼──────────────┐
                    │                ▼              ▼              ▼
                    │           risk_check  performance_check  regime_check
                    │                └──────────────┼──────────────┘
                    │                               ▼ (join)
                    │                          coordinator
                    ├──── continue ─────────────────┤
                    │                               ├── pause ──> live_pause
                    │                               │               ├─approve─> live_wait
                    │                               │               └─reject──> archive
                    │                               └── fail ───> archive ──> END
                    └───────────────────────────────┘   (continue re-arms live_wait)

    All external-effect seams are injectable so the integration test runs fully
    stubbed (no Docker, no real LLM, no live keys): spawn / stop-teardown /
    stop-trading / secrets / registry / snapshot-fetch / risk-rationale /
    performance-review / coordinator-merge / coordinator-arbitrate /
    gate-audit-writer. KILL-SWITCH wake path: SEE 8f (live_wait note).
    """

    async def _live_spawn(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return await live_spawn(
            state,
            config,
            spawn_live_container_fn=spawn_live_container_fn,
            secrets_provider=secrets_provider,
            registry_writer_fn=registry_writer_fn,
        )

    async def _live_evaluate(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return await live_evaluate(state, config, build_snapshot_fn=build_snapshot_fn)

    async def _risk_check(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return await risk_check(state, config, rationale_fn=rationale_fn)

    async def _performance_check(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return await performance_check(state, config, review_fn=review_fn)

    def _regime_check(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return regime_check(state, config)

    async def _coordinator(state: LiveState, config: RunnableConfig) -> Command[Any]:
        return await coordinator(
            dict(state),
            config,
            merge_fn=merge_fn,
            arbitrate_fn=arbitrate_fn,
            gate_audit_writer_fn=gate_audit_writer_fn,
        )

    async def _live_pause(state: LiveState, config: RunnableConfig) -> Command[Any]:
        return await live_pause(
            state,
            config,
            stop_trading_fn=stop_trading_fn,
            gate_audit_writer_fn=gate_audit_writer_fn,
        )

    async def _archive(state: LiveState, config: RunnableConfig) -> dict[str, Any]:
        return await live_archive(state, config, stop_container_fn=stop_live_container_fn)

    builder: StateGraph[LiveState, LiveState, LiveState, LiveState] = StateGraph(LiveState)
    builder.add_node("live_spawn", _live_spawn)
    builder.add_node("live_wait", live_wait)
    builder.add_node("live_evaluate", _live_evaluate)
    builder.add_node("risk_check", _risk_check)
    builder.add_node("performance_check", _performance_check)
    builder.add_node("regime_check", _regime_check)
    builder.add_node("coordinator", _coordinator)
    builder.add_node("live_pause", _live_pause)
    builder.add_node("archive", _archive)

    builder.add_edge(START, "live_spawn")
    builder.add_conditional_edges("live_spawn", _route_after_live_spawn, ["live_wait", "archive"])
    builder.add_edge("live_wait", "live_evaluate")
    builder.add_conditional_edges(
        "live_evaluate",
        _fan_out_reviewers,
        ["risk_check", "performance_check", "regime_check"],
    )
    builder.add_edge("risk_check", "coordinator")
    builder.add_edge("performance_check", "coordinator")
    builder.add_edge("regime_check", "coordinator")
    # coordinator returns Command(goto=live_wait|live_pause|archive)
    # live_pause returns Command(goto=live_wait|archive)
    builder.add_edge("archive", END)

    return builder.compile(checkpointer=checkpointer)
