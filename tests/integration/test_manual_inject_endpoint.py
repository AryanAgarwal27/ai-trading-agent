"""Stage 12 Feature 1 integration tests — POST /strategies/validate (BRD §21.2).

Manual strategy injection: an operator-supplied strategy definition (template +
explicit params + pairs/timeframe) runs straight through the EXISTING validation
gauntlet, BYPASSING the LLM researcher / generator / critic. These tests prove
the endpoint contract against the REAL FastAPI app + a stub-composed parent graph
(no Docker, no real LLM):

  - research_subgraph: a marker stub that records if it EVER runs — manual
    injection seeds ``as_node="research_subgraph"`` so the parent router skips
    straight to validation, meaning researcher/generator/critic (all INSIDE the
    research subgraph) never execute. The marker list staying empty is the
    "zero LLM creation calls" assertion (BRD §21.2 acceptance #1).
  - validation_subgraph: ``START → paper_gate → END`` — the REAL paper_gate
    interrupt node, so the seed proving "entered validation, parked at the same
    HITL gate any strategy hits" is exercised through the real interrupt path.
  - paper_subgraph: the REAL paper subgraph with stub leaves (spawn / monitor /
    context / stop) — so the paper_gate-approve test proves manual strategies
    spawn paper exactly like an LLM-path strategy (acceptance #2), via the
    EXISTING ``POST /threads/{tid}/approve``.

Marked ``integration`` (writes the real strategy_registry); NOT ``freqtrade`` —
the backtest/LLM effects are stubbed via the composed graph.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.graph import END, START, StateGraph

import orchestrator.main as main_mod
from orchestrator.agents.generator import load_schema
from orchestrator.agents.monitors import PaperMonitorContext, PaperMonitorVerdict
from orchestrator.graph import build_per_strategy_graph
from orchestrator.main import app
from orchestrator.state import StrategyState
from orchestrator.subgraphs.paper import build_paper_subgraph
from orchestrator.subgraphs.validation import ValidationState, paper_gate

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-stage12-f1"


# ───────────────────────── valid params helper ─────────────────────────


def _valid_mean_reversion_params() -> dict[str, Any]:
    """A schema-valid midpoint param set for mean_reversion_template.

    Mirrors tests/unit/test_generator._midpoint_params — builds a value at the
    middle of each ``Field(ge=, le=)`` range, so the body passes
    ``load_schema(template)(**params)`` deterministically.
    """
    schema_cls = load_schema("mean_reversion_template")
    raw: dict[str, Any] = {}
    for name, field in schema_cls.model_fields.items():
        ge = le = None
        for c in field.metadata:
            if hasattr(c, "ge"):
                ge = c.ge
            if hasattr(c, "le"):
                le = c.le
        assert ge is not None and le is not None
        raw[name] = int((ge + le) // 2) if field.annotation is int else (ge + le) / 2.0
    return dict(schema_cls(**raw).model_dump())


# ───────────────────────── stub paper leaves ───────────────────────────


def _spawn_stub() -> Any:
    async def _stub(*, port: int, **_kwargs: Any) -> str:
        return f"http://127.0.0.1:{port}"

    return _stub


def _ctx_stub() -> Any:
    async def _stub(_state: Any) -> PaperMonitorContext:
        return PaperMonitorContext(profit={"max_drawdown": 0.04})

    return _stub


def _monitor_fixed(decision: str) -> Any:
    async def _stub(_ctx: PaperMonitorContext) -> PaperMonitorVerdict:
        return PaperMonitorVerdict(
            decision=decision,
            primary_observation=f"obs-{decision}",
            rationale=f"rationale-{decision}",
            confidence=0.9,
        )

    return _stub


def _stop_stub() -> Any:
    async def _stub(_strategy_id: str) -> None:
        return None

    return _stub


async def _zero_live_count() -> int:
    return 0  # D-9: slot free → live_gate would offer approval (never reached here)


# ───────────────────────── stub-composed parent graph ──────────────────


def _build_manual_inject_graph(saver: Any, *, research_ran: list[str]) -> Any:
    """Parent graph: marker-research + paper_gate-only validation + real paper.

    ``research_ran`` is appended to IFF the research subgraph ever executes —
    which manual injection (as_node="research_subgraph" seed) must never trigger.
    """

    def _research_marker(_state: StrategyState) -> dict[str, Any]:
        research_ran.append("research")
        return {}

    rb: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = StateGraph(
        StrategyState
    )
    rb.add_node("research_marker", _research_marker)  # type: ignore[arg-type]
    rb.add_edge(START, "research_marker")
    rb.add_edge("research_marker", END)
    research = rb.compile()

    vb: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = StateGraph(
        ValidationState
    )
    vb.add_node("paper_gate", paper_gate)
    vb.add_edge(START, "paper_gate")
    vb.add_edge("paper_gate", END)
    validation = vb.compile()

    paper = build_paper_subgraph(
        spawn_container_fn=_spawn_stub(),
        paper_monitor_fn=_monitor_fixed("advance"),
        build_context_fn=_ctx_stub(),
        stop_container_fn=_stop_stub(),
        live_count_fn=_zero_live_count,
    )

    def _live_noop(_state: StrategyState) -> dict[str, Any]:
        return {}

    lb: StateGraph[StrategyState, StrategyState, StrategyState, StrategyState] = StateGraph(
        StrategyState
    )
    lb.add_node("live_noop", _live_noop)  # type: ignore[arg-type]
    lb.add_edge(START, "live_noop")
    lb.add_edge("live_noop", END)
    live = lb.compile()

    return build_per_strategy_graph(
        saver,
        None,
        research_subgraph=research,
        validation_subgraph=validation,
        paper_subgraph=paper,
        live_subgraph=live,
    )


# ───────────────────────── app / db helpers ────────────────────────────


def _dsn() -> str:
    return os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)


@asynccontextmanager
async def _app_with_manual_inject_graph(
    monkeypatch: pytest.MonkeyPatch,
    research_ran: list[str],
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        app.state.graph = _build_manual_inject_graph(app.state.saver, research_ran=research_ran)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _drain_manual_inject_tasks() -> None:
    """Await the endpoint's backgrounded astream task(s) so the thread has
    reached its interrupt before we inspect graph state."""
    tasks = [t for t in main_mod._MANUAL_INJECT_TASKS if not t.done()]
    if tasks:
        await asyncio.gather(*tasks)


async def _parked_kind(graph: Any, config: dict[str, Any]) -> str | None:
    snap = await graph.aget_state(config)
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()):
            val = intr.value
            if isinstance(val, dict):
                kind = val.get("kind")
                return kind if isinstance(kind, str) else None
    return None


async def _registry_row(strategy_id: str) -> tuple[str, str] | None:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT stage, template FROM strategy_registry WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


async def _registry_count() -> int:
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM strategy_registry")
            row = await cur.fetchone()
    return int(row[0]) if row is not None else 0


@pytest.fixture
async def cleanup_strategy_ids() -> Any:
    ids: list[str] = []
    yield ids
    if not ids:
        return
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM gate_audits WHERE strategy_id = ANY(%s)", (ids,))
            await cur.execute("DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,))
        await conn.commit()


# ───────────────────────── tests ───────────────────────────────────────


async def test_manual_inject_runs_validation_with_zero_research_events(
    cleanup_strategy_ids: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-supplied param set enters validation, parks at paper_gate, and the
    research subgraph (researcher/generator/critic) NEVER runs (BRD §21.2 #1)."""
    research_ran: list[str] = []
    async with _app_with_manual_inject_graph(monkeypatch, research_ran) as client:
        body = {
            "template": "mean_reversion_template",
            "params": _valid_mean_reversion_params(),
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "5m",
            "name": "manual-bb",
        }
        resp = await client.post(
            "/strategies/validate",
            json=body,
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        sid = data["strategy_id"]
        cleanup_strategy_ids.append(sid)
        thread_id = data["thread_id"]
        assert thread_id == f"strategy_{sid}"
        assert data["stage"] == "validation"

        await _drain_manual_inject_tasks()

        graph = app.state.graph
        config = {"configurable": {"thread_id": thread_id}}
        assert (
            await _parked_kind(graph, config) == "paper_gate"
        ), "manual injection must enter validation and park at the real paper_gate HITL"

        # The load-bearing assertion: research never executed → zero
        # researcher/generator/critic LLM calls.
        assert research_ran == [], "research subgraph must NOT run for a manual injection"

        # Registry row seeded at stage='validation' with the REAL template.
        row = await _registry_row(sid)
        assert row is not None
        stage, template = row
        assert stage == "validation"
        assert template == "mean_reversion_template"


@pytest.mark.parametrize(
    ("template", "pairs", "params", "expected_reason"),
    [
        ("stat_arb", ["BTC/USDT"], {}, "unknown_template"),
        ("mean_reversion_template", ["BTC/USDT", "AVAX/USDT"], {}, "pairs_outside_universe"),
        ("mean_reversion_template", ["BTC/USDT"], {}, "params_schema_invalid"),
    ],
)
async def test_manual_inject_rejections_write_no_row(
    template: str,
    pairs: list[str],
    params: dict[str, Any],
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown template / out-of-universe pair / schema-invalid params each 422
    and write NO registry row (BRD §21.2 #3 — rejection is no-write)."""
    research_ran: list[str] = []
    async with _app_with_manual_inject_graph(monkeypatch, research_ran) as client:
        before = await _registry_count()
        resp = await client.post(
            "/strategies/validate",
            json={"template": template, "params": params, "pairs": pairs, "timeframe": "5m"},
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["reason"] == expected_reason
        after = await _registry_count()
        assert after == before, "a rejected manual injection must write NO registry row"


async def test_manual_inject_n_folds_without_data_start_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 13: n_folds without data_start is ambiguous → 422, no row written."""
    research_ran: list[str] = []
    async with _app_with_manual_inject_graph(monkeypatch, research_ran) as client:
        before = await _registry_count()
        resp = await client.post(
            "/strategies/validate",
            json={
                "template": "mean_reversion_template",
                "params": _valid_mean_reversion_params(),
                "pairs": ["BTC/USDT"],
                "timeframe": "15m",
                "n_folds": 4,  # no data_start
            },
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["reason"] == "n_folds_requires_data_start"
        assert await _registry_count() == before


async def test_manual_inject_out_of_range_data_start_is_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 13: a data_start the cache can't honor → clear 422, no row written.

    1900-01-01 is before any cached candle (or there's no cached feather at all in
    a fresh CI env) — both resolve to the walk_forward_window_out_of_range reason.
    """
    research_ran: list[str] = []
    async with _app_with_manual_inject_graph(monkeypatch, research_ran) as client:
        before = await _registry_count()
        resp = await client.post(
            "/strategies/validate",
            json={
                "template": "mean_reversion_template",
                "params": _valid_mean_reversion_params(),
                "pairs": ["BTC/USDT"],
                "timeframe": "15m",
                "data_start": "1900-01-01",
            },
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["reason"] == "walk_forward_window_out_of_range"
        assert await _registry_count() == before


async def test_manual_inject_paper_gate_approve_spawns_paper(
    cleanup_strategy_ids: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On paper_gate approve via the EXISTING /approve endpoint, a manual
    strategy advances into the paper subgraph and parks at paper_wait —
    spawning paper exactly like an LLM-path strategy (BRD §21.2 #2)."""
    research_ran: list[str] = []
    async with _app_with_manual_inject_graph(monkeypatch, research_ran) as client:
        headers = {"X-Operator-Token": TEST_OPERATOR_TOKEN}
        body = {
            "template": "mean_reversion_template",
            "params": _valid_mean_reversion_params(),
            "pairs": ["BTC/USDT"],
            "timeframe": "5m",
        }
        resp = await client.post("/strategies/validate", json=body, headers=headers)
        assert resp.status_code == 200, resp.text
        sid = resp.json()["strategy_id"]
        cleanup_strategy_ids.append(sid)
        thread_id = resp.json()["thread_id"]

        await _drain_manual_inject_tasks()

        graph = app.state.graph
        config = {"configurable": {"thread_id": thread_id}}
        assert await _parked_kind(graph, config) == "paper_gate"

        approve = await client.post(
            f"/threads/{thread_id}/approve",
            json={"approved": True, "notes": "manual-inject approve"},
            headers=headers,
        )
        assert approve.status_code == 200, approve.text

        assert (
            await _parked_kind(graph, config) == "paper_wait"
        ), "paper_gate approve must route a manual strategy into paper_spawn → paper_wait"
