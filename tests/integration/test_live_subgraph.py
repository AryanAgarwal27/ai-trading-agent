"""Stage 8c integration tests for the ``live_spawn`` node (spawn-only slice).

Mirrors :mod:`tests.integration.test_paper_spawn` (the 7c paper-spawn contract
tests) but for the LIVE path. All marked ``integration`` (real Postgres) but
NOT ``freqtrade`` — the spawn helper is stubbed via the
``spawn_live_container_fn`` seam, so Docker (and real exchange keys) are not
required.

Contract surface verified (mirrors paper_spawn + the 8c additions):

1. Registry row written BEFORE spawn (orphan-container prevention).
2. Spawn failure → ``stage="archived"`` with ``live_spawn_failed:`` prefix,
   and the registry row records the failure (audit trail, not deleted).
3. Success → ``freqtrade_api_url`` + ``artifacts.live_started_at`` (ISO 8601)
   + ``artifacts.live_container_id`` (when the helper returns it) + stage=live.
4. Spawn invoked with the LIVE secrets provider (not paper).
5. ``stake_amount`` capped to ``LIVE_CAPITAL_CAP_USD`` before reaching spawn.
6. Port allocation reads the LIVE range only — paper-range ports in the
   registry never appear as "used" for a live allocation.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import psycopg
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator.agents.coordinator import REVIEWER_AGENTS, LiveVerdict
from orchestrator.gates.thresholds import LIVE_CAPITAL_CAP_USD
from orchestrator.subgraphs.live import LiveState, build_live_subgraph, live_spawn
from orchestrator.tools.freqtrade_lifecycle import LIVE_WORKERS_ROOT, LiveSpawnError

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DUMMY_STRATEGY = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


# ───────────────────────── fixtures ─────────────────────────


def _libpq_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _dsn() -> str:
    return _libpq_dsn(os.environ["DATABASE_URL"])


@pytest.fixture
async def cleanup_strategy_ids() -> Any:
    ids: list[str] = []
    yield ids
    if not ids:
        return
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            # gate_audits + kill_switch_events FK-reference strategy_registry —
            # delete children first (live_pause / coordinator-escalation write
            # gate_audits; the 8g kill-guard test writes kill_switch_events).
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute(
                "DELETE FROM kill_switch_events WHERE strategy_id = ANY(%s)", (ids,)
            )
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,))
        await conn.commit()


def _minimal_live_state(strategy_id: str) -> LiveState:
    """Minimal LiveState shape the spawn node reads (research handoff)."""
    return cast(
        LiveState,
        {
            "strategy_id": strategy_id,
            "name": f"test-{strategy_id}",
            "template": "mean_reversion_template",
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "5m",
            "stake_amount": 125.0,
            "stage": "live",
            "params": {},
            "agent_votes": [],
            "gate_decisions": {},
            "artifacts": {"generated_strategy_path": str(DUMMY_STRATEGY)},
            "failure_reason": "",
            "freqtrade_api_url": None,
            "freqtrade_userdir": None,
            "freqtrade_process_id": None,
        },
    )


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": "strategy_test"}}


class _StubProvider:
    """SecretProvider stub returning known-distinct live and paper creds."""

    _VALUES = {
        "BINANCE_LIVE_API_KEY": "LIVE-key-distinct",
        "BINANCE_LIVE_API_SECRET": "LIVE-secret-distinct",
        "BINANCE_LIVE_API_PASSWORD": "LIVE-rest-pw",
        "BINANCE_PAPER_API_KEY": "PAPER-key-distinct",
        "BINANCE_PAPER_API_SECRET": "PAPER-secret-distinct",
        "PAPER_API_PASSWORD": "PAPER-rest-pw",
    }

    def get(self, name: str) -> str | None:
        return self._VALUES.get(name)


async def _registry_row(strategy_id: str) -> dict[str, Any] | None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT stage, freqtrade_api_url, freqtrade_userdir, failure_reason "
                "FROM strategy_registry WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {
        "stage": row[0],
        "freqtrade_api_url": row[1],
        "freqtrade_userdir": row[2],
        "failure_reason": row[3],
    }


# ───────────────────────── tests ─────────────────────────


async def test_live_spawn_writes_registry_row_before_spawn(
    cleanup_strategy_ids: list[str],
) -> None:
    """Registry row MUST exist with stage='live' at the moment spawn is called."""
    strategy_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    seen = {"row_at_spawn": False}

    async def stub_spawn(*, port: int, **_k: Any) -> str:
        row = await _registry_row(strategy_id)
        seen["row_at_spawn"] = row is not None and row["stage"] == "live"
        return f"http://127.0.0.1:{port}"

    result = await live_spawn(
        _minimal_live_state(strategy_id),
        _config(),
        spawn_live_container_fn=stub_spawn,
    )
    assert seen["row_at_spawn"], "registry row absent at spawn time — orphan risk"
    assert result["stage"] == "live"


async def test_live_spawn_archives_on_spawn_error(
    cleanup_strategy_ids: list[str],
) -> None:
    """LiveSpawnError → stage='archived', live_spawn_failed: prefix, registry
    records the failure (row not deleted — audit trail)."""
    strategy_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    async def stub_spawn(**_k: Any) -> str:
        raise LiveSpawnError("docker daemon unreachable")

    result = await live_spawn(
        _minimal_live_state(strategy_id),
        _config(),
        spawn_live_container_fn=stub_spawn,
    )
    assert result["stage"] == "archived"
    assert result["failure_reason"].startswith("live_spawn_failed:")

    row = await _registry_row(strategy_id)
    assert row is not None, "registry row deleted on failure — audit trail lost"
    assert row["stage"] == "archived"
    assert row["failure_reason"] and "live_spawn_failed" in row["failure_reason"]


async def test_live_spawn_records_api_url_and_started_at(
    cleanup_strategy_ids: list[str],
) -> None:
    """Success records api_url, ISO live_started_at, container id, stage=live."""
    strategy_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    async def stub_spawn(**_k: Any) -> tuple[str, str]:
        return ("http://127.0.0.1:8200", "container_abc")

    before = datetime.now(UTC)
    result = await live_spawn(
        _minimal_live_state(strategy_id),
        _config(),
        spawn_live_container_fn=stub_spawn,
    )
    after = datetime.now(UTC)

    assert result["stage"] == "live"
    assert result["freqtrade_api_url"] == "http://127.0.0.1:8200"
    assert result["artifacts"]["live_container_id"] == "container_abc"

    started = datetime.fromisoformat(result["artifacts"]["live_started_at"])
    assert before <= started <= after

    row = await _registry_row(strategy_id)
    assert row is not None and row["freqtrade_api_url"] == "http://127.0.0.1:8200"


async def test_live_spawn_passes_live_credentials_not_paper(
    cleanup_strategy_ids: list[str],
) -> None:
    """The secrets provider that flows to spawn resolves the LIVE keys, never
    the paper keys (BRD §1.1 rule 5)."""
    strategy_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    captured: dict[str, Any] = {}

    async def stub_spawn(*, port: int, provider: Any = None, **_k: Any) -> str:
        captured["provider"] = provider
        return f"http://127.0.0.1:{port}"

    await live_spawn(
        _minimal_live_state(strategy_id),
        _config(),
        spawn_live_container_fn=stub_spawn,
        secrets_provider=_StubProvider(),
    )

    provider = captured["provider"]
    assert provider is not None, "live_spawn did not forward the secrets provider to spawn"
    assert provider.get("BINANCE_LIVE_API_KEY") == "LIVE-key-distinct"
    assert provider.get("BINANCE_LIVE_API_KEY") != provider.get("BINANCE_PAPER_API_KEY")


async def test_live_spawn_caps_stake_at_live_capital_cap(
    cleanup_strategy_ids: list[str],
) -> None:
    """An over-cap stake intent (1000 > $500) is capped before reaching spawn."""
    strategy_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    captured: dict[str, Any] = {}

    async def stub_spawn(*, port: int, stake_amount: float, **_k: Any) -> str:
        captured["stake_amount"] = stake_amount
        return f"http://127.0.0.1:{port}"

    state = _minimal_live_state(strategy_id)
    state["stake_amount"] = 1000.0  # over the SPEC §1 Q3 cap
    await live_spawn(state, _config(), spawn_live_container_fn=stub_spawn)

    assert captured["stake_amount"] == LIVE_CAPITAL_CAP_USD


async def test_live_spawn_port_allocation_ignores_paper_ports(
    cleanup_strategy_ids: list[str],
) -> None:
    """With paper containers on 8100/8101 AND a live container on 8200, the new
    live spawn gets 8201 — never 8102. Paper ports are outside the live range
    and must not influence live allocation."""
    paper_a = f"live-occ-{uuid.uuid4().hex[:6]}"
    paper_b = f"live-occ-{uuid.uuid4().hex[:6]}"
    live_occ = f"live-occ-{uuid.uuid4().hex[:6]}"
    cleanup_strategy_ids.extend([paper_a, paper_b, live_occ])

    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            for sid, port in [(paper_a, 8100), (paper_b, 8101), (live_occ, 8200)]:
                await cur.execute(
                    """
                    INSERT INTO strategy_registry
                      (strategy_id, thread_id, name, template, stage, pairs,
                       timeframe, freqtrade_api_url, started_at, last_updated)
                    VALUES (%s, %s, %s, 'mean_reversion_template', 'live',
                            '["BTC/USDT"]', '5m', %s, now(), now())
                    """,
                    (sid, f"strategy_{sid}", f"occ-{sid}", f"http://127.0.0.1:{port}"),
                )
        await conn.commit()

    new_id = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(new_id)
    captured: dict[str, Any] = {}

    async def stub_spawn(*, port: int, **_k: Any) -> str:
        captured["port"] = port
        return f"http://127.0.0.1:{port}"

    await live_spawn(_minimal_live_state(new_id), _config(), spawn_live_container_fn=stub_spawn)

    assert (
        captured["port"] == 8201
    ), f"expected 8201 (8200 taken, paper ports irrelevant), got {captured['port']}"


def test_live_state_inherits_paper_state() -> None:
    """LiveState extends PaperState (clean inheritance path) + live fields.

    TypedDicts forbid issubclass(); inheritance is verified structurally via
    the aggregated key sets (TypedDict rolls inherited keys into
    __required_keys__ / __optional_keys__).
    """
    from orchestrator.subgraphs.live import LiveState
    from orchestrator.subgraphs.paper import PaperState

    paper_keys = PaperState.__required_keys__ | PaperState.__optional_keys__
    live_keys = LiveState.__required_keys__ | LiveState.__optional_keys__
    assert paper_keys <= live_keys, "LiveState must carry every PaperState channel"
    assert {"strategy_path", "stake_amount"} <= live_keys
    # The live userdir root is distinct from paper's (BRD §7.1).
    assert "_live_workers" in str(LIVE_WORKERS_ROOT)


# ════════════════════════════════════════════════════════════════════════
# Stage 8e — subgraph assembly tests
# ════════════════════════════════════════════════════════════════════════
#
# Fully stubbed external effects (no Docker, no real LLM, no live keys), but
# real Postgres for the registry row (live_spawn) + gate_audits (live_pause).
# Mirrors test_paper_subgraph's assembly approach.

_CONTINUE_SNAP = {
    "max_drawdown": 0.0,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [0.01, 0.02, 0.0],
    "paper_returns": [0.01, 0.02, 0.0],
    "current_regime": "mid_vol_up",
    "approval_regime": "mid_vol_up",
}
# risk_check -> pause (drawdown breach), regime_check -> pause (regime shift):
# 2 pause majority.
_PAUSE_SNAP = {
    "max_drawdown": 0.15,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [],
    "paper_returns": [],
    "current_regime": "high_vol_down",
    "approval_regime": "low_vol_up",
}
# risk_check -> pause (drawdown), regime_check -> continue (regime match),
# performance_check -> fail (review stub): 3-way split -> escalation.
_SPLIT_SNAP = {
    "max_drawdown": 0.15,
    "daily_pnl_pct": 0.0,
    "consecutive_losses": 0,
    "live_returns": [],
    "paper_returns": [],
    "current_regime": "mid_vol_up",
    "approval_regime": "mid_vol_up",
}


def _make_graph(
    *,
    review_verdict: Literal["continue", "pause", "fail"] = "continue",
    merge_verdict: Literal["continue", "pause", "fail"] = "continue",
    arbitrate_verdict: Literal["continue", "pause", "fail"] = "fail",
    captured: dict[str, Any] | None = None,
    start_trading_fn: Any = None,
    live_started_bump_fn: Any = None,
) -> Any:
    """Build the live subgraph with all external effects stubbed.

    registry_writer_fn + gate_audit_writer_fn + secrets_provider keep their real
    defaults (the test uses real Postgres for the registry row + gate_audits).
    """

    async def _spawn(*, port: int, **_k: Any) -> str:
        return f"http://127.0.0.1:{port}"

    async def _stop_container(_sid: str) -> None:
        if captured is not None:
            captured["archive_teardown_called"] = True

    async def _rationale(_facts: dict[str, Any]) -> str:
        return "stub rationale"

    async def _review(_comparison: dict[str, Any]) -> LiveVerdict:
        return LiveVerdict(verdict=review_verdict, rationale="perf", confidence=0.8)

    async def _merge(_votes: Any) -> LiveVerdict:
        return LiveVerdict(verdict=merge_verdict, rationale="merge", confidence=0.7)

    async def _arbitrate(_votes: Any, _merge_attempt: Any) -> LiveVerdict:
        return LiveVerdict(verdict=arbitrate_verdict, rationale="arb", confidence=0.9)

    async def _stop_trading(_state: Any) -> None:
        if captured is not None:
            captured["stop_trading_called"] = True

    return build_live_subgraph(
        spawn_live_container_fn=_spawn,
        stop_live_container_fn=_stop_container,
        build_snapshot_fn=lambda _state: _async_snapshot(captured),
        rationale_fn=_rationale,
        review_fn=_review,
        merge_fn=_merge,
        arbitrate_fn=_arbitrate,
        stop_trading_fn=_stop_trading,
        start_trading_fn=start_trading_fn,
        live_started_bump_fn=live_started_bump_fn,
        checkpointer=InMemorySaver(),
    )


async def _async_snapshot(captured: dict[str, Any] | None) -> dict[str, Any]:
    snap = (captured or {}).get("snapshot", _CONTINUE_SNAP)
    return dict(snap)


def _initial_state(strategy_id: str) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "name": f"t-{strategy_id}",
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "stake_amount": 125.0,
        "stage": "live",
        "agent_votes": [],
        "gate_decisions": {},
        "params": {},
        "artifacts": {"generated_strategy_path": str(DUMMY_STRATEGY)},
    }


async def _drain(graph: Any, inp: Any, config: dict[str, Any]) -> None:
    async for _ev in graph.astream(inp, config):
        pass


async def _interrupt_payload(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for itr in getattr(task, "interrupts", ()):
            return cast(dict[str, Any], itr.value)
    return None


def _cfg(strategy_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"strategy_{strategy_id}"}}


async def test_fan_out_produces_three_reviewer_votes(
    cleanup_strategy_ids: list[str],
) -> None:
    """The Send fan-out appends exactly three reviewer votes (reducer concat)."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    captured = {"snapshot": _CONTINUE_SNAP}
    graph = _make_graph(captured=captured)
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)  # parks at live_wait
    await _drain(graph, Command(resume={"wake": True}), cfg)  # wake → fan-out → coordinator

    snap = await graph.aget_state(cfg)
    votes = snap.values.get("agent_votes", [])
    reviewer_votes = [v for v in votes if v["agent"] in REVIEWER_AGENTS]
    assert len(reviewer_votes) == 3
    assert {v["agent"] for v in reviewer_votes} == set(REVIEWER_AGENTS)
    for v in reviewer_votes:
        assert v["verdict"] in {"continue", "pause", "fail"}
        assert isinstance(v["rationale"], str)
        assert 0.0 <= v["confidence"] <= 1.0


async def test_continue_verdict_rearms_live_wait(
    cleanup_strategy_ids: list[str],
) -> None:
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(captured={"snapshot": _CONTINUE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)

    snap = await graph.aget_state(cfg)
    assert snap.values["stage"] == "live"
    # Parked again at live_wait (interrupt pending).
    assert any(getattr(t, "interrupts", ()) for t in snap.tasks)
    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None and payload["kind"] == "live_wait"


async def test_pause_verdict_surfaces_live_pause_review_payload(
    cleanup_strategy_ids: list[str],
) -> None:
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    captured: dict[str, Any] = {"snapshot": _PAUSE_SNAP}
    graph = _make_graph(merge_verdict="pause", captured=captured)
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)  # → coordinator pause → live_pause

    assert captured.get("stop_trading_called") is True, "live_pause must POST /stop"
    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None
    assert payload["kind"] == "live_pause_review"
    summary = payload["summary"]
    assert summary["path"] == "coordinator"
    assert summary["coordinator"] is not None  # coordinator wrote its gate_decisions
    assert set(summary["reviewer_votes"].keys()) == set(REVIEWER_AGENTS)
    assert "metrics" in summary


async def test_fail_verdict_routes_to_archive(
    cleanup_strategy_ids: list[str],
) -> None:
    """A 3-way split escalates; Opus (arbitrate stub) returns fail → archive."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(
        review_verdict="fail", arbitrate_verdict="fail", captured={"snapshot": _SPLIT_SNAP}
    )
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)

    snap = await graph.aget_state(cfg)
    assert snap.values["stage"] == "archived"
    assert snap.values["failure_reason"].startswith("coordinator_archive:")


async def test_live_pause_resume_approve_rearms_live_wait(
    cleanup_strategy_ids: list[str],
) -> None:
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(merge_verdict="pause", captured={"snapshot": _PAUSE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)  # parks at live_pause
    await _drain(graph, Command(resume={"approved": True, "notes": "looks ok"}), cfg)

    snap = await graph.aget_state(cfg)
    assert snap.values["stage"] == "live"
    assert any(getattr(t, "interrupts", ()) for t in snap.tasks)  # re-armed live_wait
    row = await _latest_gate_audit(sid)
    assert row is not None and row["gate"] == "live_pause" and row["decision"] == "human_approve"


async def test_live_pause_resume_reject_routes_to_archive(
    cleanup_strategy_ids: list[str],
) -> None:
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(merge_verdict="pause", captured={"snapshot": _PAUSE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)
    await _drain(graph, Command(resume={"approved": False, "notes": "kill it"}), cfg)

    snap = await graph.aget_state(cfg)
    assert snap.values["stage"] == "archived"
    assert "kill it" in snap.values["failure_reason"]
    row = await _latest_gate_audit(sid)
    assert row is not None and row["gate"] == "live_pause" and row["decision"] == "human_reject"


# ════════════════════════════════════════════════════════════════════════
# Stage 8g — kill-switch wake path + D-5 Option A resume
# ════════════════════════════════════════════════════════════════════════

_KILL_EVENT = {
    "reason": "drawdown_12pct_exceeded",
    "fired_at": "2026-06-03T14:32:11Z",
    "metrics": {"max_drawdown": 0.131, "consecutive_losses": 4},
    "action_taken": "POST /api/v1/stop",
}


async def _inject_kill_event(
    graph: Any, cfg: dict[str, Any], event: dict[str, Any]
) -> None:
    """Simulate the 8g Redis subscription writing the kill event into state.

    Merges into the existing artifacts (channel has no reducer) so live_spawn's
    live_started_at survives.
    """
    snap = await graph.aget_state(cfg)
    artifacts = dict(snap.values.get("artifacts") or {})
    artifacts["kill_switch_event"] = dict(event)
    await graph.aupdate_state(cfg, {"artifacts": artifacts})


async def test_kill_switch_event_routes_live_wait_to_live_pause(
    cleanup_strategy_ids: list[str],
) -> None:
    """artifacts.kill_switch_event set → live_wait wake skips live_evaluate and
    routes straight to live_pause (no reviewer fan-out)."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(captured={"snapshot": _CONTINUE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)  # parks at live_wait
    await _inject_kill_event(graph, cfg, _KILL_EVENT)
    await _drain(graph, Command(resume={"wake": True}), cfg)

    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None and payload["kind"] == "live_pause_review"
    snap = await graph.aget_state(cfg)
    reviewer_votes = [
        v for v in snap.values.get("agent_votes", []) if v["agent"] in REVIEWER_AGENTS
    ]
    assert reviewer_votes == [], "live_evaluate fan-out must be skipped on the kill path"


async def test_kill_path_live_pause_payload_uses_kill_switch_discriminator(
    cleanup_strategy_ids: list[str],
) -> None:
    """On the kill path the live_pause payload takes the kill-switch branch:
    path='kill_switch', coordinator=None, reviewer_votes=None, event populated
    (aligns with test_live_pause_review_kill_switch_path_renders, Stage 6g)."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(captured={"snapshot": _CONTINUE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _inject_kill_event(graph, cfg, _KILL_EVENT)
    await _drain(graph, Command(resume={"wake": True}), cfg)

    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None
    summary = payload["summary"]
    assert summary["path"] == "kill_switch"
    assert summary["coordinator"] is None
    assert summary["reviewer_votes"] is None
    assert summary["kill_switch_event"]["reason"] == "drawdown_12pct_exceeded"
    assert summary["kill_switch_event"]["action_taken"] == "POST /api/v1/stop"


async def test_no_kill_switch_event_routes_to_live_evaluate(
    cleanup_strategy_ids: list[str],
) -> None:
    """Regression guard: without a kill event, live_wait wake → live_evaluate
    (three reviewer votes), and a continue re-arms live_wait — never live_pause."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    graph = _make_graph(captured={"snapshot": _CONTINUE_SNAP})
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)

    snap = await graph.aget_state(cfg)
    reviewer_votes = [
        v for v in snap.values.get("agent_votes", []) if v["agent"] in REVIEWER_AGENTS
    ]
    assert len(reviewer_votes) == 3
    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None and payload["kind"] == "live_wait"


async def test_live_pause_approve_starts_trading_and_bumps_live_started(
    cleanup_strategy_ids: list[str],
) -> None:
    """D-5 Option A: live_pause approve POSTs /start (resume trading) AND bumps
    strategy_registry.live_started_at — both must hold."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    seen: dict[str, Any] = {"started": False, "bumped": None}

    async def _start(_state: Any) -> None:
        seen["started"] = True

    async def _bump(strategy_id: str) -> None:
        seen["bumped"] = strategy_id

    graph = _make_graph(
        merge_verdict="pause",
        captured={"snapshot": _PAUSE_SNAP},
        start_trading_fn=_start,
        live_started_bump_fn=_bump,
    )
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)
    await _drain(graph, Command(resume={"wake": True}), cfg)  # parks at live_pause
    await _drain(graph, Command(resume={"approved": True, "notes": "fixed"}), cfg)

    assert seen["started"] is True, "approve must POST /start to resume trading"
    assert seen["bumped"] == sid, "approve must bump live_started_at"
    snap = await graph.aget_state(cfg)
    assert snap.values["stage"] == "live"


async def test_kill_path_approve_clears_event_and_next_wake_evaluates(
    cleanup_strategy_ids: list[str],
) -> None:
    """On the kill path, live_pause approve CLEARS artifacts.kill_switch_event so
    the next live_wait wake routes to live_evaluate, not back to live_pause."""
    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)

    async def _noop_start(_state: Any) -> None:
        return None

    async def _noop_bump(_sid: str) -> None:
        return None

    graph = _make_graph(
        captured={"snapshot": _CONTINUE_SNAP},
        start_trading_fn=_noop_start,
        live_started_bump_fn=_noop_bump,
    )
    cfg = _cfg(sid)

    await _drain(graph, _initial_state(sid), cfg)  # parks at live_wait
    await _inject_kill_event(graph, cfg, _KILL_EVENT)
    await _drain(graph, Command(resume={"wake": True}), cfg)  # parks at live_pause (kill)
    await _drain(graph, Command(resume={"approved": True, "notes": "ok"}), cfg)  # → live_wait

    snap = await graph.aget_state(cfg)
    assert snap.values["artifacts"].get("kill_switch_event") is None, (
        "approve on the kill path must clear kill_switch_event"
    )

    await _drain(graph, Command(resume={"wake": True}), cfg)  # next wake → live_evaluate
    snap2 = await graph.aget_state(cfg)
    reviewer_votes = [
        v for v in snap2.values.get("agent_votes", []) if v["agent"] in REVIEWER_AGENTS
    ]
    assert len(reviewer_votes) == 3, "next wake after kill-approve must evaluate, not re-pause"
    payload = await _interrupt_payload(graph, cfg)
    assert payload is not None and payload["kind"] == "live_wait"


async def test_kill_guard_unsuppresses_after_live_started_bump(
    cleanup_strategy_ids: list[str],
) -> None:
    """D-5 closure: scheduler._kill_already_fired flips True→False once
    live_started_at is bumped past the prior kill row's fired_at."""
    from datetime import timedelta

    from orchestrator.scheduler import _kill_already_fired
    from orchestrator.subgraphs.live import _default_bump_live_started_at

    sid = f"live-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(sid)
    now = datetime.now(UTC)
    t_live_start = now - timedelta(hours=2)
    t_fired = now - timedelta(hours=1)  # kill fired AFTER the live run began

    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO strategy_registry
                  (strategy_id, thread_id, name, template, stage, pairs,
                   timeframe, live_started_at, started_at, last_updated)
                VALUES (%s, %s, %s, 'mean_reversion_template', 'live',
                        '["BTC/USDT"]', '5m', %s, now(), now())
                """,
                (sid, f"strategy_{sid}", f"t-{sid}", t_live_start),
            )
            await cur.execute(
                """
                INSERT INTO kill_switch_events
                  (strategy_id, fired_at, reason, metrics, action_taken)
                VALUES (%s, %s, 'drawdown_12pct_exceeded', '{}', 'POST /api/v1/stop')
                """,
                (sid, t_fired),
            )
        await conn.commit()

    # Before bump: kill fired after live_started_at → suppressed.
    assert await _kill_already_fired(sid) is True
    # Bump live_started_at to now() (> t_fired) → un-suppressed.
    await _default_bump_live_started_at(sid)
    assert await _kill_already_fired(sid) is False


def test_build_live_subgraph_compiles_with_defaults() -> None:
    """All-defaults build compiles (don't run — defaults touch Docker/DB)."""
    graph = build_live_subgraph()
    nodes = set(graph.get_graph().nodes)
    assert "live_spawn" in nodes


def test_topology_node_set() -> None:
    """Node-list assertion catches accidentally-disconnected nodes."""
    graph = _make_graph()
    nodes = set(graph.get_graph().nodes)
    expected = {
        "live_spawn",
        "live_wait",
        "live_evaluate",
        "risk_check",
        "performance_check",
        "regime_check",
        "coordinator",
        "live_pause",
        "archive",
    }
    assert expected <= nodes, f"missing nodes: {expected - nodes}"


async def _latest_gate_audit(strategy_id: str) -> dict[str, Any] | None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT gate, decision, actor FROM gate_audits "
                "WHERE strategy_id = %s ORDER BY at DESC LIMIT 1",
                (strategy_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"gate": row[0], "decision": row[1], "actor": row[2]}
