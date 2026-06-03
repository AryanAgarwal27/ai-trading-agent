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
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg
from langchain_core.runnables import RunnableConfig

from orchestrator.gates.thresholds import LIVE_CAPITAL_CAP_USD, MAX_OPEN_TRADES
from orchestrator.security.secrets import EnvSecretProvider, SecretProvider
from orchestrator.subgraphs.paper import PaperState
from orchestrator.tools.freqtrade_lifecycle import (
    LIVE_WORKERS_ROOT,
    next_free_live_port,
    spawn_live_container,
)

logger = logging.getLogger(__name__)

# Injection seams (8c). spawn returns the api_url, or a (api_url, container_id)
# tuple when the helper surfaces a container id.
SpawnLiveContainerFn = Callable[..., Awaitable[str | tuple[str, str]]]
RegistryWriterFn = Callable[..., Awaitable[None]]


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
        await write_registry(
            **identity, stage="archived", userdir=userdir, failure_reason=reason
        )
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
