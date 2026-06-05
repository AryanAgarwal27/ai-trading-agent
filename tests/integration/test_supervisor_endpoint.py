"""Integration tests for POST /supervisor/run (Stage 10d closure).

The manual supervisor trigger — the on-demand entry the 9d cron / 9e event
subscription reference but never exposed. Drives the real FastAPI lifespan
(real saver/store, in-memory scheduler via conftest) and overrides the
lifespan-bound ``run_supervisor_fn`` so no real Sonnet agent is invoked:

- a RECORDING stub proves the endpoint forwards ``trigger="manual"`` + the
  ``dry_run`` flag on a fresh conn and returns the SupervisorDecision;
- the REAL ``run_supervisor`` with a stub agent that PROPOSES a spawn proves
  ``dry_run=True`` returns that decision yet writes nothing (no registry row).

Marked ``integration`` (lifespan opens the Postgres saver/store; the
spawns-nothing test reads ``strategy_registry``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.main import app
from orchestrator.observability.events import _connect_app_db
from orchestrator.supervisor import SupervisorAction, SupervisorDecision, run_supervisor

pytestmark = pytest.mark.integration

TEST_OPERATOR_TOKEN = "test-operator-token-10d-supervisor"


@asynccontextmanager
async def _app(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPERATOR_TOKEN", TEST_OPERATOR_TOKEN)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _registry_count() -> int:
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM strategy_registry")
            row = await cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        await conn.close()


class _SpawnAgent:
    """Stub supervisor agent — always proposes ONE spawn (no real LLM)."""

    async def ainvoke(self, payload: dict[str, Any], config: Any = None) -> dict[str, Any]:
        return {
            "structured_response": SupervisorDecision(
                actions=[SupervisorAction(action="spawn", rationale="endpoint test spawn")],
                overall_rationale="manual-trigger endpoint test",
                confidence=1.0,
            )
        }


# ─── auth ───────────────────────────────────────────────────────────────


async def test_supervisor_run_without_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app(monkeypatch) as client:
        resp = await client.post("/supervisor/run", json={"dry_run": True})
    assert resp.status_code == 401


async def test_supervisor_run_with_wrong_token_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _app(monkeypatch) as client:
        resp = await client.post(
            "/supervisor/run", json={"dry_run": True}, headers={"X-Operator-Token": "wrong"}
        )
    assert resp.status_code == 403


# ─── wiring: forwards trigger="manual" + dry_run, returns the decision ───


async def test_supervisor_run_forwards_manual_trigger_and_returns_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    async def _recording_fn(
        conn: Any, *, trigger: str, dry_run: bool = False
    ) -> SupervisorDecision:
        recorded["conn_opened"] = conn is not None
        recorded["trigger"] = trigger
        recorded["dry_run"] = dry_run
        return SupervisorDecision(actions=[], overall_rationale="recorded", confidence=0.5)

    async with _app(monkeypatch) as client:
        app.state.run_supervisor_fn = _recording_fn
        resp = await client.post(
            "/supervisor/run",
            json={"dry_run": True},
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )

    assert resp.status_code == 200
    body = resp.json()
    # The endpoint opened a real conn and forwarded trigger="manual" + dry_run.
    assert recorded["conn_opened"] is True
    assert recorded["trigger"] == "manual"
    assert recorded["dry_run"] is True
    # And it returns the SupervisorDecision (plus trigger/dry_run echo).
    assert body["trigger"] == "manual"
    assert body["dry_run"] is True
    assert body["decision"]["overall_rationale"] == "recorded"
    assert body["decision"]["confidence"] == 0.5


# ─── dry_run returns a decision yet spawns nothing ───────────────────────


async def test_supervisor_run_dry_run_returns_decision_but_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _app(monkeypatch) as client:
        # Real run_supervisor, but a stub agent that PROPOSES a spawn. dry_run
        # must run the reasoning path, return the spawn decision, and roll back
        # — never inserting a strategy_registry row.
        app.state.run_supervisor_fn = partial(
            run_supervisor,
            app.state.graph,
            app.state.store,
            agent=_SpawnAgent(),
            unschedule_wake_fn=app.state.unschedule_live_wake_fn,
        )

        before = await _registry_count()
        resp = await client.post(
            "/supervisor/run",
            json={"dry_run": True},
            headers={"X-Operator-Token": TEST_OPERATOR_TOKEN},
        )
        after = await _registry_count()

    assert resp.status_code == 200
    body = resp.json()
    # The proposed spawn IS in the returned decision...
    assert body["decision"]["actions"][0]["action"] == "spawn"
    assert body["dry_run"] is True
    # ...but dry_run executed no writes — registry row count is unchanged.
    assert after == before
