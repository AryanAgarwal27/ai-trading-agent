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
import json
import logging
import os
import sys
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from secrets import compare_digest
from typing import Any

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestrator.agents.generator import load_schema, render_and_validate
from orchestrator.gates.spawn_vocab import check_spawn_vocabulary
from orchestrator.graph import build_per_strategy_graph
from orchestrator.kill_subscription import (
    cancel_kill_subscription,
    make_kill_event_writer,
    run_kill_subscription,
)
from orchestrator.observability.events import (
    _connect_app_db,
    publish_gate_advanced,
    record_gate_audit,
)
from orchestrator.observability.freqtrade_exporter import collect_freqtrade_metrics
from orchestrator.observability.log import configure_logging, get_logger, run_context
from orchestrator.observability.tracing import langsmith_project, trace_config, tracing_enabled
from orchestrator.ops.reconcile import reconcile_on_startup
from orchestrator.scheduler import (
    build_scheduler,
    make_schedule_live_wake_fn,
    make_schedule_wake_fn,
    make_unschedule_live_wake_fn,
    make_unschedule_wake_fn,
    register_recurring_jobs,
    register_supervisor_cron,
    shutdown_scheduler,
)
from orchestrator.security.ast_validator import ASTValidationError
from orchestrator.supervisor import _insert_registry_row, run_supervisor
from orchestrator.supervisor_subscription import (
    cancel_supervisor_subscription,
    make_schedule_supervisor_run,
    run_supervisor_subscription,
)

# .env is loaded in orchestrator/__init__.py (Stage 11c F1) — BEFORE any langgraph
# import, so LANGGRAPH_STRICT_MSGPACK (read at langgraph import time) is honoured.
# Loading it here (after the langgraph imports above) would be too late.

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


def _interrupt_kind(task: Any) -> str | None:
    """Return the ``kind`` of a parked task's first interrupt payload.

    The gate identity lives in the interrupt payload's ``"kind"`` field
    (set by ``build_interrupt_payload`` / the wake-park), NOT in the
    task's node name. When a gate runs inside a nested compiled subgraph
    (the Stage 7g parent graph: paper_gate inside ``validation_subgraph``,
    paper_wait + live_gate inside ``paper_subgraph``), the parent task
    name is the SUBGRAPH node name — so a node-name check would reject a
    legitimate gate. Keying off the payload ``kind`` is nesting-invariant
    and matches the build_interrupt_payload contract.
    """
    interrupts = getattr(task, "interrupts", ())
    if not interrupts:
        return None
    value = interrupts[0].value
    if isinstance(value, dict):
        kind = value.get("kind")
        return kind if isinstance(kind, str) else None
    return None


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


class SupervisorRunBody(BaseModel):
    """Body of ``POST /supervisor/run`` (the manual supervisor trigger).

    ``dry_run=True`` runs the full read + reasoning path (sync → snapshot →
    regime → strategies → agent → decision) but SKIPS every write — no
    spawn/retire, no telemetry row, ``conn.rollback()`` instead of commit
    (the ``run_supervisor`` dry-run contract). It is the safe way to exercise
    the trigger without mutating the portfolio. Default ``False`` matches
    ``run_supervisor``'s own default (a manual run is a real run unless asked
    otherwise).
    """

    dry_run: bool = Field(default=False)


class ManualValidateBody(BaseModel):
    """Body of ``POST /strategies/validate`` (Stage 12 Feature 1, BRD §21.2).

    A complete, operator-supplied strategy definition that runs straight through
    the validation gauntlet — no researcher / generator / critic LLM calls. The
    ``params`` are validated against the template's co-located Pydantic schema
    (``load_schema``), the SAME hard constraint the generator's structured output
    enforces (BRD §8 rule 3 / §1.1 rule 1) — so a hand-supplied param set cannot
    smuggle a value the LLM path couldn't.
    """

    model_config = ConfigDict(extra="forbid")

    template: str = Field(description="A shipped template name (∈ SHIPPED_TEMPLATES).")
    params: dict[str, Any] = Field(
        description="Concrete SLOT values; validated by load_schema(template).",
    )
    pairs: list[str] = Field(description="Trading pairs; a non-empty subset of PAIR_UNIVERSE.")
    timeframe: str = Field(description="Freqtrade timeframe, e.g. '5m'.")
    name: str | None = Field(default=None, description="Optional human label.")


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

    Operator workflow:

    1. Run ``scripts/seed_dashboard_data.py --shape={clean_pass,marginal}``
       — parks a checkpoint at paper_gate against the real
       ``AsyncPostgresSaver`` + inserts a ``strategy_registry`` row.
    2. Restart uvicorn with ``AIT_SMOKE_PAPER_GATE_GRAPH=1`` in the
       env so this branch fires.
    3. Open the dashboard, drive Approve / Reject.
    4. UNSET the env var for any subsequent real run.
    """
    # Lazy imports keep production startup free of the validation
    # subgraph's langchain-anthropic chain when the env var is unset.
    from langgraph.graph import END, START, StateGraph

    from orchestrator.subgraphs.validation import ValidationState, paper_gate

    builder: StateGraph[ValidationState, ValidationState, ValidationState, ValidationState] = (
        StateGraph(ValidationState)
    )
    builder.add_node("paper_gate", paper_gate)
    builder.add_edge(START, "paper_gate")
    builder.add_edge("paper_gate", END)
    return builder.compile(checkpointer=saver)


def _build_live_pause_review_only_graph_for_smoke(saver: Any) -> Any:
    """Smoke-only graph: ``START → live_pause_review → END``. NOT for production.

    Stage 8 will land the REAL ``live_pause_review`` node inside the
    live subgraph; this 6j smoke helper provides a STANDALONE synthetic
    gate node so the kill-switch dashboard rendering can be exercised
    before Stage 8 ships. Mirrors
    :func:`_build_paper_gate_only_graph_for_smoke` in shape: minimal
    one-node graph against the same saver, gated by an env var.

    The node calls :func:`build_interrupt_payload` with
    ``kind="live_pause_review"`` — the payload's ``summary.path``
    discriminator (``"kill_switch"`` vs ``"coordinator"``) is decided
    entirely by whether the caller pre-populated
    ``state.artifacts.kill_switch_event``. The seed script
    (``scripts/seed_dashboard_data.py --shape=kill_switch``) drops a
    synthetic event in there; with that absent the same helper would
    produce a coordinator-path payload (no Stage 8 dependency needed).

    Lazy imports + a local ``TypedDict`` so production startup remains
    free of these symbols when the env var is unset.
    """
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt
    from typing_extensions import TypedDict

    from orchestrator.gates.hitl import build_interrupt_payload
    from orchestrator.observability.events import publish_gate_pending

    class _LivePauseSmokeState(TypedDict, total=False):
        # Minimal schema — ValidationState lacks the ``artifacts`` key
        # build_interrupt_payload reads from for the kill-switch path,
        # so a local TypedDict keeps the smoke helper self-contained
        # rather than mutating the validation subgraph's contract.
        strategy_id: str
        gate_decisions: dict[str, Any]
        artifacts: dict[str, Any]
        stage: str
        failure_reason: str

    async def live_pause_review(
        state: _LivePauseSmokeState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        payload = build_interrupt_payload(state, "live_pause_review")  # type: ignore[arg-type]
        thread_id = (config.get("configurable") or {}).get("thread_id", "")
        # publish-before-interrupt — same idempotency contract as the
        # real paper_gate node (BRD §6.2 replay; events.py swallow).
        await publish_gate_pending(thread_id, payload)
        decision = interrupt(payload)

        existing_gates = state.get("gate_decisions") or {}
        if not isinstance(decision, dict) or "approved" not in decision:
            return {
                "stage": "archived",
                "failure_reason": "live_pause_review_invalid_decision_payload",
                "gate_decisions": {
                    **existing_gates,
                    "live_pause": {
                        "approved": False,
                        "notes": "",
                        "by": "human",
                        "error": "invalid_decision_payload",
                    },
                },
            }
        notes = str(decision.get("notes", ""))
        approved = bool(decision.get("approved", False))
        if approved:
            # Operator chose resume — Stage 8 will own the actual
            # respawn / state-transition logic; for the smoke we
            # stamp ``stage="live"`` and let the test inspect.
            return {
                "stage": "live",
                "gate_decisions": {
                    **existing_gates,
                    "live_pause": {"approved": True, "notes": notes, "by": "human"},
                },
            }
        return {
            "stage": "archived",
            "failure_reason": f"live_pause_review_archived: {notes}",
            "gate_decisions": {
                **existing_gates,
                "live_pause": {"approved": False, "notes": notes, "by": "human"},
            },
        }

    builder: StateGraph[
        _LivePauseSmokeState,
        _LivePauseSmokeState,
        _LivePauseSmokeState,
        _LivePauseSmokeState,
    ] = StateGraph(_LivePauseSmokeState)
    builder.add_node("live_pause_review", live_pause_review)
    builder.add_edge(START, "live_pause_review")
    builder.add_edge("live_pause_review", END)
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


def _assert_strict_msgpack_enabled() -> None:
    """Refuse to start unless LangGraph's strict-msgpack deserialization is in
    effect (BRD §6.6 / §15 — Stage 11c F1).

    Reads LangGraph's RESOLVED flag (``_msgpack.STRICT_MSGPACK_ENABLED``), NOT the
    env var, deliberately: the flag is captured once at langgraph IMPORT time, so
    an env value that lands too late (e.g. ``load_dotenv`` after the import) leaves
    strict mode silently OFF. Asserting the value actually in effect catches that
    and any future load-order regression — the env var reading "true" is not proof
    the protection is on. Strict mode restricts checkpoint deserialization to a
    safe type set, blocking code execution from a compromised checkpoint DB.
    """
    from langgraph.checkpoint.serde._msgpack import STRICT_MSGPACK_ENABLED

    if not STRICT_MSGPACK_ENABLED:
        raise RuntimeError(
            "LANGGRAPH_STRICT_MSGPACK is not in effect — refusing to start "
            "(BRD §6.6/§15: strict msgpack blocks code execution from a "
            "compromised checkpoint database). The flag is read at langgraph "
            "import time, so it must be set in the process environment BEFORE "
            "Python imports langgraph; a .env value is loaded by "
            "orchestrator/__init__.py to guarantee that ordering. Set "
            "LANGGRAPH_STRICT_MSGPACK=true (see .env.example)."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the LangGraph saver/store, the Redis pubsub client, and
    initialize the per-thread lock map.

    Connection lifecycle uses ``AsyncExitStack`` so a partial setup
    failure still tears down resources opened earlier — the BRD §6.5
    sample skips this but practical operation requires it (see SPEC
    §6 change log 2026-05-27 Stage 1b entry).
    """
    # Stage 10c (BRD §14): install the structlog processor chain ONCE at process
    # startup, before any node runs. The orchestrator process owns every graph
    # execution (FastAPI endpoints + APScheduler jobs + kill/supervisor
    # subscriptions + post-commit spawn tasks all run here), so this single call
    # configures structured logging for all of them. JSON to stdout is the prod
    # default; AIT_LOG_CONSOLE swaps in the human renderer for local dev.
    configure_logging()

    # Stage 11c (F1, BRD §6.6/§15): fail loudly NOW if strict-msgpack checkpoint
    # deserialization is not actually in effect — before opening the saver that
    # would deserialize checkpoints. Guards against the env var being set too late
    # to be honoured (the silent-off hole this stage closed).
    _assert_strict_msgpack_enabled()

    # Stage 10d (BRD §14, SPEC §1 Q4): LangSmith auto-instruments
    # LangGraph/LangChain from the env (LANGSMITH_TRACING / LANGSMITH_API_KEY,
    # loaded by load_dotenv — the key is never hardcoded). Log the resolved
    # state once at startup so the operator can see whether traces will flow,
    # without depending on a successful first trace to find out. Tracing being
    # off is a normal, supported mode — the graph runs identically either way.
    get_logger("startup").info(
        "langsmith_tracing",
        payload={"enabled": tracing_enabled(), "project": langsmith_project()},
    )

    checkpoint_uri = _require_env("LANGGRAPH_CHECKPOINT_URI")
    store_uri = _require_env("LANGGRAPH_STORE_URI")
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

    async with AsyncExitStack() as stack:
        saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(checkpoint_uri))
        store = await stack.enter_async_context(AsyncPostgresStore.from_conn_string(store_uri))
        await saver.setup()
        await store.setup()

        # Redis client is lazy-connect — startup succeeds even if Redis
        # is unreachable; first publish/subscribe surfaces the failure.
        # Matches the "best-effort pubsub" semantics in
        # orchestrator/observability/events.py.
        # from_url untyped across redis-py 5.x (Stage 10a; see events.py
        # _redis_client) — returns a Redis client at runtime. Targeted ignore.
        redis_client = aioredis.from_url(redis_url)  # type: ignore[no-untyped-call]
        stack.push_async_callback(redis_client.aclose)

        app.state.saver = saver
        app.state.store = store
        app.state.redis = redis_client
        # Per-thread asyncio.Lock — created on first access. v1 has ≤20
        # active threads so unbounded growth is not a concern.
        # Stage 10 work item: LRU eviction once we routinely retain
        # > ~50 archived threads in memory.
        app.state.thread_locks = defaultdict(asyncio.Lock)

        # Stage 10e.2: the in-orchestrator Freqtrade exporter. /metrics calls
        # this at scrape time to refresh per-container profit/status gauges.
        # Wired here so unit tests that hit /metrics without the lifespan simply
        # skip it (no DB / no container scrape). It is self-degrading: per-
        # container + DB failures are swallowed inside the collector.
        app.state.freqtrade_collector = collect_freqtrade_metrics

        # ── APScheduler (Stage 7f). ────────────────────────────────────
        # Wake jobs (6h) + regime job (1h) + kill-switch poll placeholder
        # (5m). SQLAlchemyJobStore in production so a parked paper
        # thread's wake survives restart (BRD §4); tests select an
        # in-memory store via AIT_SCHEDULER_JOBSTORE=memory. Built BEFORE
        # the graph so the real schedule_wake_fn can be threaded into the
        # paper subgraph (Stage 7g composition).
        scheduler = build_scheduler()
        register_recurring_jobs(scheduler)
        scheduler.start()
        stack.push_async_callback(shutdown_scheduler, scheduler)
        app.state.scheduler = scheduler
        app.state.schedule_wake_fn = make_schedule_wake_fn(scheduler)
        # D-12: cancel a paper thread's recurring wake job at archive (mirrors the
        # live cleanup below) so an archived paper thread's 6h wake stops firing.
        app.state.unschedule_wake_fn = make_unschedule_wake_fn(scheduler)
        # 9f (D-6 periodic live-wake): the live analog of schedule_wake_fn +
        # its cleanup. schedule registers live_wake:<sid> at live_spawn (6h,
        # kind="live_wait"); unschedule cancels it at live_archive + supervisor
        # retire. Exposed on app.state so the supervisor partial can reuse the
        # cleanup fn for aretire.
        app.state.schedule_live_wake_fn = make_schedule_live_wake_fn(scheduler)
        app.state.unschedule_live_wake_fn = make_unschedule_live_wake_fn(scheduler)

        # The production parent graph composes research → validation →
        # paper (Stage 7g) → live (Stage 8g), with the APScheduler-backed
        # paper + live wake seams threaded into the paper/live subgraphs. All
        # other leaf seams default to their real implementations.
        app.state.graph = build_per_strategy_graph(
            saver,
            store,
            schedule_wake_fn=app.state.schedule_wake_fn,
            unschedule_wake_fn=app.state.unschedule_wake_fn,
            schedule_live_wake_fn=app.state.schedule_live_wake_fn,
            unschedule_live_wake_fn=app.state.unschedule_live_wake_fn,
        )

        # ── Smoke-only graph overrides (env-gated, off by default). ────
        # When AIT_SMOKE_PAPER_GATE_GRAPH is set, replace the production
        # parent graph with a paper_gate-only minimal graph for the
        # Stage 6f operator smoke — the real parent doesn't include
        # validation_subgraph yet (per 6e note 1), so an endpoint's
        # aget_state() of a paper_gate-parked checkpoint would otherwise
        # report no pending interrupt. AIT_SMOKE_LIVE_PAUSE_REVIEW_GRAPH
        # is the equivalent override for the kill-switch dashboard
        # iteration (Stage 6j). Production behavior is identical when
        # both env vars are unset.
        #
        # If BOTH env vars are set, the second branch wins (last write
        # to app.state.graph) — the warning makes the live state
        # observable. Operators should set only one at a time.
        if os.environ.get("AIT_SMOKE_PAPER_GATE_GRAPH"):
            app.state.graph = _build_paper_gate_only_graph_for_smoke(saver)
            logger.warning(
                "AIT_SMOKE_PAPER_GATE_GRAPH=1 — production parent graph "
                "replaced with smoke-only paper_gate graph. UNSET this "
                "env var for any real run."
            )
        if os.environ.get("AIT_SMOKE_LIVE_PAUSE_REVIEW_GRAPH"):
            app.state.graph = _build_live_pause_review_only_graph_for_smoke(saver)
            logger.warning(
                "AIT_SMOKE_LIVE_PAUSE_REVIEW_GRAPH=1 — production parent "
                "graph replaced with smoke-only live_pause_review graph. "
                "UNSET this env var for any real run."
            )

        # ── Redis kill-switch subscription (Stage 8g, BRD §5.6). ───────
        # Long-running task: PSUBSCRIBE ai-trading-agent:kill_switch:*, and on
        # each event write artifacts.kill_switch_event into the matching
        # thread's state so its next live_wait wake routes to live_pause. Wired
        # AFTER the smoke-override branches so the writer closes over the FINAL
        # app.state.graph. The writer is exposed on app.state so a future
        # /live_pause/resume endpoint or test can reuse it; the task is
        # cancel-on-shutdown via the AsyncExitStack callback.
        #
        # NOTE: kill events written to state via this subscription require a live-wake
        # mechanism to actually trigger live_pause routing in production. Tests drive
        # wakes via Command(resume=...) directly. See DEFERRED.md D-6.
        app.state.kill_event_writer_fn = make_kill_event_writer(app.state.graph)
        kill_task: asyncio.Task[None] = asyncio.create_task(
            run_kill_subscription(redis_client, kill_event_writer_fn=app.state.kill_event_writer_fn)
        )
        app.state.kill_subscription_task = kill_task
        stack.push_async_callback(cancel_kill_subscription, kill_task)

        # ── Supervisor nightly cron (Stage 9d, BRD §5.1, §13 row 9). ────
        # Bind the runner over the FINAL app.state.graph (after the smoke
        # overrides) + store; the cron job opens a fresh conn per run and
        # calls this. Exposed on app.state so a future manual-trigger admin
        # endpoint or the 9e event subscription can reuse the same bound
        # runner without rebuilding the closure. The job lives in a dedicated
        # in-memory jobstore (the closure isn't picklable) — see
        # register_supervisor_cron.
        # 9f: bind the live-wake cleanup so a supervisor retire of a LIVE thread
        # cancels its recurring wake job (aretire_strategy calls it only when the
        # retired thread was in the live stage).
        app.state.run_supervisor_fn = partial(
            run_supervisor,
            app.state.graph,
            store,
            unschedule_wake_fn=app.state.unschedule_live_wake_fn,
        )
        register_supervisor_cron(scheduler, run_supervisor_fn=app.state.run_supervisor_fn)

        # ── Supervisor event-driven trigger (Stage 9e, BRD §5.1, §13 row 9). ──
        # The other half of the supervisor's dual trigger (9d = nightly cron).
        # Long-running task: PSUBSCRIBE ai-trading-agent:thread_completed:*; each
        # completion (a thread reaching terminal archived in the REGISTRY —
        # paper_teardown, live_spawn failure, supervisor retire) schedules a
        # single COALESCED supervisor run via a debounced one-shot. Wired AFTER
        # register_supervisor_cron so the supervisor_memory jobstore the one-shot
        # targets already exists, and over the SAME lifespan-bound
        # run_supervisor_fn (final graph + store). The debounced scheduler fn is
        # exposed on app.state for a future manual-trigger endpoint / debugging;
        # the task is cancel-on-shutdown via the AsyncExitStack callback.
        app.state.schedule_supervisor_run_fn = make_schedule_supervisor_run(
            scheduler, app.state.run_supervisor_fn
        )
        supervisor_sub_task: asyncio.Task[None] = asyncio.create_task(
            run_supervisor_subscription(
                redis_client,
                schedule_supervisor_run_fn=app.state.schedule_supervisor_run_fn,
            )
        )
        app.state.supervisor_subscription_task = supervisor_sub_task
        stack.push_async_callback(cancel_supervisor_subscription, supervisor_sub_task)

        # ── On-startup DR reconciliation (Stage 10f, BRD §16, §17 #12). ──
        # Now that the saver/store, the FINAL app.state.graph, and the 9f kill
        # writer are all live, reconcile the registry against reality: scan the
        # non-archived containers, ping each freqtrade_api_url, and for any
        # unreachable LIVE container write a kill_switch_events row + drive the
        # thread to live_pause via the SAME kill_event_writer_fn the Redis kill
        # subscription uses (reuse 9f's direct-resume path — /wake would 409 a
        # kill-written thread). A single unreachable container is the NORMAL case
        # this handles; a DB failure is swallowed inside reconcile. Wrapped here
        # too (belt-and-suspenders) so a reconcile error can NEVER crash startup —
        # BRD §17 #12: reconcile must not depend on the graph being awake.
        try:
            await reconcile_on_startup(kill_event_writer_fn=app.state.kill_event_writer_fn)
        except Exception:  # noqa: BLE001 — reconcile failure must not abort the lifespan
            logger.exception("startup reconciliation failed; continuing to serve")

        yield


app = FastAPI(title="ai-trading-agent orchestrator", lifespan=lifespan)


# ─── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape endpoint (Stage 10e, BRD §14).

    Serializes the prometheus_client DEFAULT registry — the metrics defined in
    ``orchestrator.observability.metrics`` (kill-switch fires, per-stage
    strategy gauge, supervisor runs), registered at import time via the
    ``orchestrator.supervisor`` + ``orchestrator.kill_subscription`` modules
    this app already imports. No auth by design: the orchestrator binds
    ``127.0.0.1`` only (BRD §15 — ``orchestrator.main.main()`` /
    ``ORCHESTRATOR_HOST`` default 127.0.0.1), so the scrape endpoint is reached
    only over the same loopback / WireGuard tunnel as the rest of the API; a
    token on a Prometheus scrape would just be a shared secret in the scrape
    config, not a real boundary.

    Stage 10e.2: if the lifespan wired the in-orchestrator Freqtrade exporter
    (``app.state.freqtrade_collector``), refresh the per-container profit/status
    gauges at scrape time first. Belt-and-braces try/except — the exporter
    already swallows per-container + DB failures, and this guarantees even an
    unexpected error there can NEVER break /metrics (a down container, an
    unreachable DB, etc. still return the push metrics with a 200).
    """
    collector = getattr(request.app.state, "freqtrade_collector", None)
    if collector is not None:
        try:
            await collector()
        except Exception as exc:  # noqa: BLE001 — a scrape failure must never break /metrics
            logger.warning(
                "freqtrade exporter collection failed: %s; serving push metrics only", exc
            )
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/threads")
async def list_threads(request: Request) -> list[dict[str, Any]]:
    """List every row in ``strategy_registry`` with current interrupt state.

    Returns a list of objects:
    ``{strategy_id, thread_id, stage, last_updated, has_pending_interrupt,
    pending_interrupt_payload, live_gate_status}``. ``live_gate_status`` is
    ``"live_slot_occupied"`` when the thread is parked at ``paper_wait`` because
    ``live_gate`` refused to offer approval — the live slot is full (D-9) —
    else ``None``.

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
        live_gate_status: str | None = None
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
            # D-9: surface "live slot occupied" — a paper-staged thread parked at
            # paper_wait whose last live_gate verdict was a capacity BLOCK (the
            # gate refused to offer approval because MAX_CONCURRENT_LIVE_STRATEGIES
            # is full). Keyed off the parked interrupt KIND (paper_wait) + the
            # recorded status, so a thread actually parked AT live_gate (offering
            # approval, kind="live_gate") never reads as blocked. A strategy that
            # was blocked then degraded to rearm shows blocked for one wake cycle
            # until the next live_gate re-evaluates — a cosmetic, self-healing blip.
            if (pending_payload or {}).get("kind") == "paper_wait":
                lg = ((snapshot.values or {}).get("gate_decisions") or {}).get("live_gate") or {}
                if lg.get("status") == "live_slot_occupied":
                    live_gate_status = "live_slot_occupied"
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
                "live_gate_status": live_gate_status,
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

        # Identify the gate by the interrupt payload's "kind" (nesting-
        # invariant — see _interrupt_kind), not the task node name.
        gate_kind = _interrupt_kind(interrupted_tasks[0])
        if gate_kind is None or gate_kind not in RESUMABLE_GATES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"thread {thread_id!r} interrupted at unexpected gate "
                    f"kind={gate_kind!r}; expected one of {tuple(RESUMABLE_GATES)}. "
                    "(A paper_wait park is resumed via /wake, not /approve.)"
                ),
            )

        decision_dict = {"approved": body.approved, "notes": body.notes}
        # Stage 10c: this resume is one GRAPH EXECUTION — mint a fresh
        # execution-scoped run_id and bind {run_id, strategy_id, thread_id} for
        # its duration so every node it drives logs under the same execution id.
        # Stage 10d: thread that SAME run_id (the string run_context yields and
        # binds into structlog contextvars) into trace_config so the LangSmith
        # trace and the structlog lines share one id — not a parallel mint.
        resume_sid = str((snapshot.values or {}).get("strategy_id") or thread_id)
        resume_stage = (snapshot.values or {}).get("stage")
        with run_context(strategy_id=resume_sid, thread_id=thread_id) as run_id:
            traced_config = trace_config(
                config,
                strategy_id=resume_sid,
                thread_id=thread_id,
                run_id=run_id,
                stage=resume_stage,
            )
            async for _ in graph.astream(Command(resume=decision_dict), config=traced_config):
                pass

        post_snapshot = await graph.aget_state(config)
        next_stage = post_snapshot.values.get("stage")
        strategy_id = post_snapshot.values.get("strategy_id") or thread_id

        actor = _hash_token_for_actor(token)
        decision_label = "human_approve" if body.approved else "human_reject"
        # ``notes`` is intentionally written at TWO paths:
        #   - ``payload.decision.notes`` (canonical structured shape;
        #     mirrors the ApprovalDecision TypedDict the resume Command
        #     carried — keeps the audit blob symmetric with the input).
        #   - ``payload.notes`` (convenience path for ad-hoc SQL — e.g.
        #     ``SELECT payload->>'notes' FROM gate_audits WHERE ...``
        #     drops the nested ``->'decision'`` dereference).
        # Do NOT "clean up the redundancy" — Stage 6f smoke surfaced
        # operators reaching for the top-level path on instinct, and
        # losing audit visibility because of a JSON-path mismatch is
        # exactly the failure mode this duplication prevents.
        audit_payload = {
            "thread_id": thread_id,
            "gate_kind": gate_kind,
            "decision": decision_dict,
            "next_stage": next_stage,
            "notes": body.notes,
        }

        audit_id = await record_gate_audit(
            strategy_id=strategy_id,
            gate=RESUMABLE_GATES[gate_kind],
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


# Wake kinds /wake will resume. paper_wait (Stage 7f) + live_wait (Stage 9f
# D-6 periodic re-eval). A HITL gate kind (paper_gate / live_gate /
# live_pause_review) is NEVER wake-able — that is /approve's job.
WAKEABLE_KINDS: frozenset[str] = frozenset({"paper_wait", "live_wait"})


@app.post("/threads/{thread_id}/wake")
async def wake_thread(
    thread_id: str,
    request: Request,
    kind: str = "paper_wait",
    token: str = Depends(_require_operator_token),
) -> dict[str, Any]:
    """Wake a thread parked at ``paper_wait`` (Stage 7f) or ``live_wait`` (9f).

    Called by the APScheduler wake job (loopback, X-Operator-Token — SPEC §6
    d6736ba) and resumes the thread with ``Command(resume={"wake": True}})`` so
    it proceeds (paper → ``paper_monitor``; live → ``live_evaluate`` via
    ``_route_after_live_wait`` on the no-kill path).

    The ``kind`` query param (default ``"paper_wait"`` so the Stage-7f paper
    wake jobs — which send no kind — keep working) declares which park the
    caller intends. The endpoint REFUSES (409) unless the thread's parent-
    visible interrupt kind equals ``kind``. Consequences:

    - Distinct from ``/approve``: a wake is NOT a HITL decision (no
      ``gate_audits`` row). Feeding ``{"wake": True}`` into a HITL gate's
      decision validation would archive the strategy, so a HITL-gate park is
      never wake-able.
    - **9f kill-path independence (regression-guarded):** a kill-written thread
      has had its parent interrupt CLEARED by ``kill_subscription``'s
      ``aupdate_state`` (the 8h finding), so ``interrupted_tasks`` is empty →
      409 here. That is the desired contract: the kill path is the ONLY resumer
      for kill-written threads (it direct-resumes via ``Command(resume=...)``).
      The periodic ``kind="live_wait"`` wake must never operate on such a thread.

    Same per-thread ``asyncio.Lock`` as /approve so a wake and an operator
    approve can't race on the same thread.
    """
    if kind not in WAKEABLE_KINDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"kind={kind!r} is not wake-able; expected one of {tuple(WAKEABLE_KINDS)}.",
        )

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
                    f"(next={snapshot.next!r}). Nothing to wake. (A kill-written "
                    "thread has no parent interrupt — it is resumed by the kill "
                    "subscription, not /wake.)"
                ),
            )

        # Identify the parked gate by the interrupt payload "kind" (the
        # paper_wait / live_wait interrupt carries it). Under the composed parent
        # graph the parked task is the SUBGRAPH node, so a task-name check would
        # never match; the payload kind is nesting-invariant (same reasoning as
        # /approve's _interrupt_kind). The wake must match the requested kind so a
        # live-wake can never resume a paper park (or vice versa) and neither can
        # resume a HITL gate.
        parked_kind = _interrupt_kind(interrupted_tasks[0])
        if parked_kind != kind:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"thread {thread_id!r} is parked at kind={parked_kind!r}, not the "
                    f"requested wake kind={kind!r}; /wake only resumes a matching "
                    "paper_wait/live_wait park. Use /approve for HITL gates."
                ),
            )

        # Stage 10c: a wake-resume is one GRAPH EXECUTION — fresh run_id bound
        # for its scope (mirrors /approve + the kill direct-resume). Stage 10d:
        # the same run_id is threaded into trace_config (one shared id).
        resume_sid = str((snapshot.values or {}).get("strategy_id") or thread_id)
        resume_stage = (snapshot.values or {}).get("stage")
        with run_context(strategy_id=resume_sid, thread_id=thread_id) as run_id:
            traced_config = trace_config(
                config,
                strategy_id=resume_sid,
                thread_id=thread_id,
                run_id=run_id,
                stage=resume_stage,
            )
            async for _ in graph.astream(Command(resume={"wake": True}), config=traced_config):
                pass

        post_snapshot = await graph.aget_state(config)
        next_stage = post_snapshot.values.get("stage")

    return {"woke": True, "next_stage": next_stage}


# ─── POST /supervisor/run (manual trigger) ─────────────────────────────


@app.post("/supervisor/run")
async def run_supervisor_endpoint(
    body: SupervisorRunBody,
    request: Request,
    token: str = Depends(_require_operator_token),
) -> dict[str, Any]:
    """Manually trigger ONE supervisor run (Stage 10d closure).

    This is the on-demand trigger the 9d cron / 9e event-subscription wiring
    references but never exposed as an endpoint — for ops + testing, and the
    permanent Stage 10d trace-verification path: a non-dry run spawns a
    strategy whose per-strategy graph execution carries the trace_config tags
    + the run_id mirror (so a real LangSmith trace can be inspected on demand).

    OPERATOR_TOKEN-guarded (``X-Operator-Token``), exactly like /approve and
    /wake. Invokes the lifespan-bound ``run_supervisor_fn`` (graph + store
    already bound) at ``trigger="manual"`` on a fresh app-DB connection — the
    SAME per-call connection lifecycle the cron + event jobs use; the runner
    owns its own transaction (commit on a real run, rollback on dry_run), this
    endpoint only opens/closes the connection. ``dry_run`` (body) is forwarded
    so the trigger can be exercised without spawning. Returns the
    ``SupervisorDecision`` (model_dump) plus the trigger + dry_run echo.
    """
    run_supervisor_fn = request.app.state.run_supervisor_fn
    conn = await _connect_app_db()
    try:
        decision = await run_supervisor_fn(conn, trigger="manual", dry_run=body.dry_run)
    finally:
        await conn.close()

    return {
        "trigger": "manual",
        "dry_run": body.dry_run,
        "decision": decision.model_dump(),
    }


# ─── POST /strategies/validate (manual strategy injection) ─────────────
# Background manual-injection graph runs — held so the fire-and-forget validation
# task is not GC'd mid-run (mirrors supervisor._BACKGROUND_SPAWN_TASKS).
_MANUAL_INJECT_TASKS: set[asyncio.Task[None]] = set()


@app.post("/strategies/validate")
async def validate_strategy_manual(
    body: ManualValidateBody,
    request: Request,
    token: str = Depends(_require_operator_token),
) -> dict[str, Any]:
    """Manually inject a strategy straight into the validation gauntlet (Feature 1).

    BRD §21.2: accept a complete strategy definition (template + explicit params +
    pairs/timeframe) and run it through the EXISTING validation subgraph —
    ``prepare_validation_inputs → backtests → gate_backtest → robustness →
    risk_analyst → paper_gate``, then the paper subgraph on approve — with ZERO
    researcher / generator / critic LLM calls. This is NOT a gauntlet or HITL
    bypass: the strategy hits the same BRD §10 gates, the same ``paper_gate``
    interrupt (resumed via ``POST /threads/{tid}/approve``), and the same 30-day
    paper clock as an LLM-path strategy.

    Validation order (ALL before any write — a rejection writes NO registry row,
    BRD §21.1.1/.3):

      1. ``template ∈ SHIPPED_TEMPLATES`` then ``pairs ⊆ PAIR_UNIVERSE`` — the
         shared D-16 spawn-vocabulary guard
         (:func:`orchestrator.gates.spawn_vocab.check_spawn_vocabulary`). 422 on
         a violation (``unknown_template`` / ``pairs_outside_universe``), plus a
         422 for an empty pair list.
      2. ``load_schema(template)(**params)`` — 422 + the Pydantic error on a
         missing slot or out-of-range value (BRD §1.1 rule 1 for manual params).
      3. ``render_and_validate`` (deterministic, no LLM) writes
         ``_generated/<sid>.py`` and runs the AST allowlist. 422 if it
         (defensively) fails — impossible for a shipped template + schema-valid
         params, but the check stays.

    Then: insert a ``stage="validation"`` registry row
    (:func:`orchestrator.supervisor._insert_registry_row`), seed the parent graph
    ``as_node="research_subgraph"`` so its router (``_route_after_research``)
    advances the non-archived state into the validation subgraph (BRD §21.1.5),
    and drive the run in a backgrounded ``astream(None)`` wrapped in
    ``run_context`` + ``trace_config`` (one ``run_id``; BRD §21.1.8). Returns
    immediately with ``{strategy_id, thread_id, stage}``; the operator watches
    ``GET /threads`` and approves at ``paper_gate``.

    Token-gated (``X-Operator-Token``) exactly like /approve, /wake, /supervisor/run.
    """
    # ── 1. Spawn-vocabulary (shared D-16 guard) — template + pairs. ──
    if not body.pairs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": "empty_pairs",
                "message": "pairs must be a non-empty subset of the universe",
            },
        )
    vocab_error = check_spawn_vocabulary(body.template, body.pairs)
    if vocab_error is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": vocab_error.reason, **vocab_error.detail},
        )

    # ── 2. Param schema — the same hard constraint the generator enforces. ──
    # template ∈ SHIPPED_TEMPLATES was just confirmed, so load_schema cannot KeyError.
    schema_cls = load_schema(body.template)
    try:
        validated = schema_cls(**body.params)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "params_schema_invalid", "errors": json.loads(exc.json())},
        ) from exc
    params = validated.model_dump()

    # ── 3. Deterministic render + AST allowlist (no LLM). ──
    strategy_id = uuid.uuid4().hex
    thread_id = f"strategy_{strategy_id}"
    try:
        strategy_path = render_and_validate(strategy_id, body.template, params)
    except ASTValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "ast_validation_failed", "violations": list(exc.violations)},
        ) from exc

    resolved_name = body.name or f"strategy_{strategy_id[:8]}"

    # ── 4. Registry row (stage="validation") — shared seam; endpoint owns commit. ──
    conn = await _connect_app_db()
    try:
        await _insert_registry_row(
            conn,
            strategy_id=strategy_id,
            thread_id=thread_id,
            name=resolved_name,
            template=body.template,
            pairs=body.pairs,
            timeframe=body.timeframe,
            stage="validation",
        )
        await conn.commit()
    finally:
        await conn.close()

    # ── 5. Seed the parent graph at validation + drive it backgrounded. ──
    graph = request.app.state.graph
    thread_locks: defaultdict[str, asyncio.Lock] = request.app.state.thread_locks
    config = {"configurable": {"thread_id": thread_id}}
    now_iso = datetime.now(UTC).isoformat()
    # Seed only real StrategyState channels (BRD §5.7). ``strategy_path`` is NOT a
    # parent channel — the validation subgraph derives it from
    # ``artifacts.generated_strategy_path`` in prepare_validation_inputs, exactly
    # as the research→validation handoff does (the generator sets the same artifact).
    seed_state: dict[str, Any] = {
        "strategy_id": strategy_id,
        "name": resolved_name,
        "template": body.template,
        "params": params,
        "pairs": body.pairs,
        "timeframe": body.timeframe,
        # Non-archived → _route_after_research advances to validation_subgraph (BRD §21.1.5).
        "stage": "validation",
        "started_at": now_iso,
        "last_updated": now_iso,
        "artifacts": {"generated_strategy_path": str(strategy_path)},
    }

    async with thread_locks[thread_id]:
        # Seed as_node="research_subgraph": the checkpoint records this as research's
        # output, so resuming with astream(None) makes the parent router advance the
        # (non-archived) state straight into validation_subgraph — research is skipped,
        # the whole downstream gauntlet + HITL path is reused unchanged (BRD §21.1.5).
        await graph.aupdate_state(config, seed_state, as_node="research_subgraph")

        async def _drive() -> None:
            try:
                # One graph execution → one run_id shared by structlog + LangSmith
                # (BRD §21.1.8), mirroring the spawn / approve / wake paths.
                with run_context(strategy_id=strategy_id, thread_id=thread_id) as run_id:
                    traced_config = trace_config(
                        config,
                        strategy_id=strategy_id,
                        thread_id=thread_id,
                        run_id=run_id,
                        stage="validation",
                    )
                    async for _ in graph.astream(None, config=traced_config):
                        pass
            except Exception as exc:  # noqa: BLE001 — background run; surfaces via logs + archived thread
                logger.error(
                    "manual-inject: graph run failed strategy_id=%s exc=%s", strategy_id, exc
                )

        task = asyncio.create_task(_drive())
        _MANUAL_INJECT_TASKS.add(task)
        task.add_done_callback(_MANUAL_INJECT_TASKS.discard)

    return {"strategy_id": strategy_id, "thread_id": thread_id, "stage": "validation"}


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
    pattern = f"ai-trading-agent:*:{thread_id}" if thread_id else "ai-trading-agent:*"

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


# ─── Production entry point ────────────────────────────────────────────


def _install_selector_loop_policy_on_win32() -> None:
    """Install the ``SelectorEventLoop`` policy on Windows (no-op elsewhere).

    Extracted from :func:`main` (Stage 11a / D-15) so the loop-policy install —
    the whole point of owning the loop — is unit-testable WITHOUT creating a real
    event loop: the earlier ``test_main_entrypoint`` that drove the real
    ``asyncio.run`` below created+closed a loop and polluted sibling tests' loop
    state (D-14). psycopg async requires the Selector loop on win32; uvicorn
    would otherwise force ``ProactorEventLoop`` (Stage 10d). On Linux the
    Selector loop is already the default, so this is a no-op and the VPS prod
    target is unaffected.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main() -> None:
    """Launch the orchestrator, owning the event loop so psycopg async gets a
    SelectorEventLoop on Windows (Stage 10d closure; BRD §3 supports a local
    machine, so a Windows-native launch is a supported config).

    Why not ``uvicorn.run()`` / ``uvicorn --loop asyncio``? Both FORCE uvicorn's
    ``ProactorEventLoop`` on win32. psycopg's async mode rejects Proactor and
    requires ``SelectorEventLoop`` (SPEC 2026-05-27 Stage 3c), so the lifespan's
    ``AsyncPostgresSaver.from_conn_string`` crashes under it. Setting the policy
    and THEN calling ``uvicorn.run()`` does NOT help — uvicorn replaces the loop.
    The fix is to OWN the loop: install the selector policy on win32, then run a
    ``uvicorn.Server`` via ``asyncio.run`` (which honours the policy). On Linux
    the selector loop is already the default, so the win32 branch is skipped and
    behaviour is unchanged (the VPS prod target is unaffected).

    ``tests/conftest.py`` installs the same selector policy as a fixture, which
    is why the 87 integration tests pass — but there was no PRODUCTION launch
    path with the equivalent until this entry point. Host/port come from
    ``ORCHESTRATOR_HOST`` / ``ORCHESTRATOR_PORT`` (loaded by ``load_dotenv``;
    default ``127.0.0.1:8000`` per BRD §15 loopback-only).
    """
    import uvicorn

    _install_selector_loop_policy_on_win32()

    host = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCHESTRATOR_PORT", "8000"))
    server = uvicorn.Server(uvicorn.Config("orchestrator.main:app", host=host, port=port))
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
