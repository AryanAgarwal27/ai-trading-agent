"""FastAPI app for the ai-trading-agent orchestrator.

Stage 1b brought up the lifespan with the LangGraph
``AsyncPostgresSaver`` / ``AsyncPostgresStore`` and a ``GET /health``
endpoint. Stage 6d extends this with:

- ``GET /threads`` — registry rows + per-thread interrupt state.
- ``POST /threads/{tid}/approve`` — resume an interrupted thread. Token
  gated via ``X-Operator-Token`` (SPEC §6 change log 2026-05-27).
  Serialized per-thread via an ``asyncio.Lock`` so concurrent operator
  taps from a stale tab cannot double-advance.
- ``WS /events`` — Redis pubsub bridge for the dashboard. Unauthenticated
  read-only stream over the same WireGuard / SSH tunnel as the API
  itself (BRD §15).

Connection URIs are read from the environment (``.env``):
- ``LANGGRAPH_CHECKPOINT_URI`` → AsyncPostgresSaver
- ``LANGGRAPH_STORE_URI``      → AsyncPostgresStore
- ``REDIS_URL``                → pubsub client
- ``DATABASE_URL``             → app DB (via observability.events helpers)
- ``OPERATOR_TOKEN``           → shared secret for ``/approve``
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from secrets import compare_digest
from typing import Any

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from orchestrator.graph import build_per_strategy_graph
from orchestrator.observability.events import (
    _connect_app_db,
    publish_gate_advanced,
    record_gate_audit,
)

load_dotenv()

logger = logging.getLogger(__name__)


# ─── Gate-node ↔ gate_audits.gate mapping ──────────────────────────────
# gate_audits.gate has a CHECK constraint per BRD §5.8. The resumable
# LangGraph gate-node names map to those values here. Any node NOT in
# this map is rejected at /approve with 409 (defensive: the test graphs
# in 6d also use these names so they pass through cleanly).

RESUMABLE_GATES: dict[str, str] = {
    "paper_gate": "paper",
    "live_gate": "live",
    "live_pause_review": "live_pause",
}


# ─── Request body shape ────────────────────────────────────────────────


class ApprovalDecisionBody(BaseModel):
    """Body of ``POST /threads/{tid}/approve``.

    Mirrors :class:`orchestrator.gates.hitl.ApprovalDecision` — kept as a
    separate Pydantic model so FastAPI can validate / OpenAPI-document
    the request without leaking the TypedDict into the schema. The two
    must stay in sync; if you add a field here, mirror it there.
    """

    approved: bool
    notes: str = Field(default="")


# ─── Env helpers ───────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


# ─── Operator-token dependency ─────────────────────────────────────────


# ─── Smoke-only graph override (env-gated, throwaway) ────────────────
# Lives here for locality with the lifespan branch that swaps it in.
# The validation subgraph (and its langchain-anthropic transitive deps)
# is imported INSIDE the function so a production startup with
# AIT_SMOKE_PAPER_GATE_GRAPH unset never pays that import cost.
#
# Public-ish (single underscore): scripts/midstage_seed.py imports this
# so the seed and the lifespan use the IDENTICAL graph topology — no
# contract drift between "graph the smoke parks at" and "graph the
# FastAPI endpoint queries".


def _build_paper_gate_only_graph_for_smoke(saver: Any) -> Any:
    """Smoke-only graph: ``START → paper_gate → END``. NOT for production.

    Wired into ``app.state.graph`` ONLY when
    ``AIT_SMOKE_PAPER_GATE_GRAPH`` is set in the env — see the env-gated
    branch at the tail of :func:`lifespan`. The minimal topology matches
    the 6e e2e test fixtures, so the smoke exercises the real interrupt
    + resume + audit paths through the real FastAPI endpoints.

    Operator workflow (Stage 6f mid-stage smoke):

    1. Run ``scripts/midstage_seed.py`` — parks a checkpoint at
       paper_gate against the real ``AsyncPostgresSaver`` + inserts a
       ``strategy_registry`` row.
    2. Restart uvicorn with ``AIT_SMOKE_PAPER_GATE_GRAPH=1`` in the
       env so this branch fires.
    3. Open the dashboard, drive Approve / Reject.
    4. UNSET the env var for any subsequent real run.
    """
    # Lazy imports keep production startup free of the validation
    # subgraph's langchain-anthropic chain when the env var is unset.
    from langgraph.graph import END, START, StateGraph

    from orchestrator.subgraphs.validation import ValidationState, paper_gate

    builder: StateGraph[
        ValidationState, ValidationState, ValidationState, ValidationState
    ] = StateGraph(ValidationState)
    builder.add_node("paper_gate", paper_gate)
    builder.add_edge(START, "paper_gate")
    builder.add_edge("paper_gate", END)
    return builder.compile(checkpointer=saver)


def _hash_token_for_actor(token: str) -> str:
    """Derive an audit-safe actor id from the operator token.

    Never store or log the raw token. ``sha256`` first-12-hex (96 bits)
    is plenty for audit traceability and lets a future operator rotate
    tokens without losing the ability to attribute historical decisions.
    """
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"operator_token:{digest}"


async def _require_operator_token(
    x_operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
) -> str:
    """FastAPI dependency: require a valid ``X-Operator-Token`` header.

    - 401 if the header is missing.
    - 403 if the header is present but does not match ``OPERATOR_TOKEN``.
    - 500 if ``OPERATOR_TOKEN`` is not configured on the server (a
      misconfigured deploy should fail loudly, not silently allow).

    Constant-time comparison via :func:`secrets.compare_digest` avoids
    leaking match progress through timing.
    """
    expected = os.environ.get("OPERATOR_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPERATOR_TOKEN not configured on server",
        )
    if x_operator_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Operator-Token header missing",
        )
    if not compare_digest(x_operator_token, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="X-Operator-Token mismatch",
        )
    return x_operator_token


# ─── Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the LangGraph saver/store, the Redis pubsub client, and
    initialize the per-thread lock map.

    Connection lifecycle uses ``AsyncExitStack`` so a partial setup
    failure still tears down resources opened earlier — the BRD §6.5
    sample skips this but practical operation requires it (see SPEC
    §6 change log 2026-05-27 Stage 1b entry).
    """
    checkpoint_uri = _require_env("LANGGRAPH_CHECKPOINT_URI")
    store_uri = _require_env("LANGGRAPH_STORE_URI")
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(checkpoint_uri)
        )
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(store_uri)
        )
        await saver.setup()
        await store.setup()

        # Redis client is lazy-connect — startup succeeds even if Redis
        # is unreachable; first publish/subscribe surfaces the failure.
        # Matches the "best-effort pubsub" semantics in
        # orchestrator/observability/events.py.
        redis_client = aioredis.from_url(redis_url)
        stack.push_async_callback(redis_client.aclose)

        app.state.saver = saver
        app.state.store = store
        app.state.graph = build_per_strategy_graph(saver, store)
        app.state.redis = redis_client
        # Per-thread asyncio.Lock — created on first access. v1 has ≤20
        # active threads so unbounded growth is not a concern.
        # Stage 10 work item: LRU eviction once we routinely retain
        # > ~50 archived threads in memory.
        app.state.thread_locks = defaultdict(asyncio.Lock)

        # ── Smoke-only graph override (env-gated, off by default). ─────
        # When AIT_SMOKE_PAPER_GATE_GRAPH is set, replace the production
        # parent graph with a paper_gate-only minimal graph for the
        # Stage 6f operator smoke — the real parent doesn't include
        # validation_subgraph yet (per 6e note 1), so an endpoint's
        # aget_state() of a paper_gate-parked checkpoint would otherwise
        # report no pending interrupt. Production behavior is identical
        # when the env var is unset.
        if os.environ.get("AIT_SMOKE_PAPER_GATE_GRAPH"):
            app.state.graph = _build_paper_gate_only_graph_for_smoke(saver)
            logger.warning(
                "AIT_SMOKE_PAPER_GATE_GRAPH=1 — production parent graph "
                "replaced with smoke-only paper_gate graph. UNSET this "
                "env var for any real run."
            )

        yield


app = FastAPI(title="ai-trading-agent orchestrator", lifespan=lifespan)


# ─── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/threads")
async def list_threads(request: Request) -> list[dict[str, Any]]:
    """List every row in ``strategy_registry`` with current interrupt state.

    Returns a list of objects:
    ``{strategy_id, thread_id, stage, last_updated, has_pending_interrupt,
    pending_interrupt_payload}``.

    ``pending_interrupt_payload`` is the dict that was passed to
    ``interrupt(...)`` inside the paused gate node (i.e. the output of
    :func:`orchestrator.gates.hitl.build_interrupt_payload`). It is
    ``None`` when ``has_pending_interrupt`` is False. The Streamlit
    dashboard (Stage 6f) renders rationale + metrics from this field —
    embedding it here avoids a per-thread HTTP round-trip and keeps the
    UI to a single polling endpoint.

    Computed by calling ``graph.aget_state(config)`` per thread and
    inspecting ``snapshot.tasks[*].interrupts[*].value``. O(N) per
    request — fine for v1 (≤20 threads). **Stage 10 work item**: cache
    the snapshot in a Redis hash invalidated by the ``gate_pending`` /
    ``gate_advanced`` channels once N > ~50.
    """
    graph = request.app.state.graph

    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT strategy_id, thread_id, stage, last_updated "
                "FROM strategy_registry ORDER BY last_updated DESC"
            )
            rows = await cur.fetchall()
    finally:
        await conn.close()

    threads: list[dict[str, Any]] = []
    for strategy_id, thread_id, stage, last_updated in rows:
        config = {"configurable": {"thread_id": thread_id}}
        has_pending = False
        pending_payload: dict[str, Any] | None = None
        try:
            snapshot = await graph.aget_state(config)
            for task in snapshot.tasks:
                interrupts = getattr(task, "interrupts", ())
                if interrupts:
                    has_pending = True
                    # First interrupt's value is the
                    # build_interrupt_payload(...) dict the gate node
                    # passed to interrupt(). Multiple interrupts on one
                    # task are not a pattern this project uses — first
                    # wins keeps the contract simple.
                    pending_payload = interrupts[0].value
                    break
        except Exception:
            # Threads with no checkpoint history (registry row but graph
            # never ran) — treat as no pending interrupt rather than
            # failing the whole list.
            pass
        threads.append(
            {
                "strategy_id": strategy_id,
                "thread_id": thread_id,
                "stage": stage,
                "last_updated": last_updated.isoformat() if last_updated else None,
                "has_pending_interrupt": has_pending,
                "pending_interrupt_payload": pending_payload,
            }
        )
    return threads


@app.post("/threads/{thread_id}/approve")
async def approve_thread(
    thread_id: str,
    body: ApprovalDecisionBody,
    request: Request,
    token: str = Depends(_require_operator_token),
) -> dict[str, Any]:
    """Resume an interrupted thread with an :class:`ApprovalDecisionBody`.

    Contract:
    1. Per-thread ``asyncio.Lock`` taken before any state inspection —
       two concurrent approves on the same thread serialize, and the
       loser sees the post-advance state and gets 409.
    2. Refuses (409) if the thread is not parked at an interrupt OR is
       parked at an unexpected gate node (not in
       :data:`RESUMABLE_GATES`).
    3. Streams ``Command(resume={"approved", "notes"})`` through the
       graph until it completes (next interrupt or END).
    4. Writes a ``gate_audits`` row with ``actor="operator_token:<hash>"``
       (the raw token is never persisted).
    5. Fires ``publish_gate_advanced`` (best-effort; failures logged
       and swallowed inside ``events.py``).

    Returns ``{"resumed": True, "next_stage", "audit_id"}``.
    """
    graph = request.app.state.graph
    thread_locks: defaultdict[str, asyncio.Lock] = request.app.state.thread_locks

    lock = thread_locks[thread_id]
    async with lock:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)
        interrupted_tasks = [t for t in snapshot.tasks if getattr(t, "interrupts", ())]

        if not interrupted_tasks:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"thread {thread_id!r} is not parked at an interrupt "
                    f"(next={snapshot.next!r}). Refusing to resume."
                ),
            )

        gate_node = getattr(interrupted_tasks[0], "name", "")
        if gate_node not in RESUMABLE_GATES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"thread {thread_id!r} interrupted at unexpected node "
                    f"{gate_node!r}; expected one of {tuple(RESUMABLE_GATES)}."
                ),
            )

        decision_dict = {"approved": body.approved, "notes": body.notes}
        async for _ in graph.astream(Command(resume=decision_dict), config=config):
            pass

        post_snapshot = await graph.aget_state(config)
        next_stage = post_snapshot.values.get("stage")
        strategy_id = post_snapshot.values.get("strategy_id") or thread_id

        actor = _hash_token_for_actor(token)
        decision_label = "human_approve" if body.approved else "human_reject"
        audit_payload = {
            "thread_id": thread_id,
            "gate_node": gate_node,
            "decision": decision_dict,
            "next_stage": next_stage,
        }

        audit_id = await record_gate_audit(
            strategy_id=strategy_id,
            gate=RESUMABLE_GATES[gate_node],
            decision=decision_label,
            actor=actor,
            payload=audit_payload,
        )

        await publish_gate_advanced(
            thread_id,
            {
                "thread_id": thread_id,
                "decision": decision_dict,
                "next_stage": next_stage,
                "audit_id": audit_id,
            },
        )

    return {"resumed": True, "next_stage": next_stage, "audit_id": audit_id}


# ─── WS /events ────────────────────────────────────────────────────────


@app.websocket("/events")
async def ws_events(ws: WebSocket, thread_id: str | None = None) -> None:
    """Stream Redis pubsub messages to a WebSocket client.

    Unauthenticated by design — the FastAPI port is bound to ``127.0.0.1``
    and reached only via the WireGuard / SSH tunnel (BRD §15). Adding
    a header check to a WebSocket upgrade would mean shipping the token
    in the URL or a custom protocol, which is worse than relying on
    the network boundary.

    Query param:
        thread_id — when provided, subscribe to
        ``ai-trading-agent:*:<thread_id>`` (per-strategy view).
        When absent, subscribe to ``ai-trading-agent:*`` (global view
        for the threads list dashboard).
    """
    await ws.accept()
    redis_client = ws.app.state.redis
    pubsub = redis_client.pubsub()
    pattern = (
        f"ai-trading-agent:*:{thread_id}" if thread_id else "ai-trading-agent:*"
    )

    try:
        await pubsub.psubscribe(pattern)
        async for message in pubsub.listen():
            # ``listen`` yields subscribe-confirmation events too; the
            # pmessage / message types carry payloads we care about.
            mtype = message.get("type")
            if mtype not in ("pmessage", "message"):
                continue
            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            channel = message.get("channel")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            await ws.send_json({"channel": channel, "data": data})
    except WebSocketDisconnect:
        # Normal client disconnect — fall through to cleanup.
        pass
    finally:
        try:
            await pubsub.punsubscribe(pattern)
        except Exception:
            pass
        try:
            await pubsub.aclose()
        except Exception:
            pass
