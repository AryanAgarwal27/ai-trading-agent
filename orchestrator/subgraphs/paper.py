"""Stage 7 paper subgraph (BRD §5.5) — paper_spawn node only.

This 7c commit ships ONLY ``paper_spawn``. The other nodes
(``schedule_wake``, ``paper_wait``, ``paper_monitor``,
``divergence_check``, ``live_gate``, ``paper_teardown``) land in 7e
once the spawn + monitor cycle has a real Freqtrade container to
talk to.

``paper_spawn`` is the first node in the paper subgraph. Pipeline:

1. Derive ``stake_amount`` from ``state.params`` (or fall back to
   ``LIVE_CAPITAL_CAP_USD / MAX_OPEN_TRADES`` per BRD §10).
2. Query ``strategy_registry`` for ports already assigned to active
   paper threads; call :func:`next_free_paper_port` for the next free.
3. **Upsert the registry row BEFORE calling spawn.** This is the
   orphan-container-prevention contract: if spawn succeeds but the
   registry write fails, the orchestrator loses track of a live
   container that's placing dry-run orders. Order matters.
4. Call ``spawn_container_fn`` (injected for tests; defaults to the
   real :func:`spawn_paper_container`). Catch ``PaperSpawnTimeout``
   and any other exception, return ``{stage: "archived", ...}`` —
   never raise. A spawn failure should archive the thread cleanly,
   not crash the graph thread.
5. On success: UPDATE the registry row with the resolved API URL.
6. Return state update including ``artifacts.paper_started_at`` —
   this is the 30-day clock SPEC §6 (d6736ba) locked. The
   ``divergence_check`` node (7e) enforces
   ``elapsed_days >= MIN_PAPER_DAYS`` against this timestamp before
   the monitor can vote "advance".
"""

from __future__ import annotations

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
from orchestrator.state import StrategyState
from orchestrator.tools.freqtrade_lifecycle import (
    WORKERS_ROOT,
    PaperSpawnTimeout,
    next_free_paper_port,
    spawn_paper_container,
)

logger = logging.getLogger(__name__)

# Injection seam — same shape as Stage 5's ``critic_fn`` and Stage 4's
# ``risk_analyst_fn``. Tests pass a stub; production uses the default.
SpawnContainerFn = Callable[..., Awaitable[str]]


# ───────────────────────── db helpers ─────────────────────────


def _libpq_dsn(sqlalchemy_url: str) -> str:
    """Strip the ``postgresql+psycopg://`` SQLAlchemy prefix for libpq."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _connect_app_db() -> psycopg.AsyncConnection:
    """Open a fresh async psycopg connection to the app DB.

    Same shape as :func:`orchestrator.observability.events._connect_app_db`.
    Not reused from there to keep the paper subgraph's import surface
    independent of observability — the two modules ship and evolve at
    different cadences.
    """
    return await psycopg.AsyncConnection.connect(_libpq_dsn(os.environ["DATABASE_URL"]))


def _used_ports_from_urls(urls: list[Any]) -> list[int]:
    """Extract port numbers from ``freqtrade_api_url`` row values.

    Tolerant of None and malformed URLs — a row without a parseable
    port simply doesn't contribute to the in-use set. The allocator
    is the authoritative source for collision-avoidance; this is a
    best-effort prune of the search space.
    """
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
    """Return the thread_id from config, or the canonical fallback.

    LangGraph passes thread_id via ``config["configurable"]["thread_id"]``.
    Tests that invoke the node directly may omit this — falling back to
    ``strategy_<id>`` matches the convention used elsewhere in the
    codebase (BRD §5.1).
    """
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return f"strategy_{strategy_id}"


async def _upsert_registry_row(
    conn: psycopg.AsyncConnection,
    state: StrategyState,
    thread_id: str,
) -> None:
    """Upsert the strategy_registry row for ``state`` with ``stage='paper'``.

    On conflict, only ``stage`` and ``last_updated`` are touched — the
    immutable identity fields (thread_id, name, template, pairs,
    timeframe) are not overwritten. ``freqtrade_api_url`` is left NULL
    here and filled by a follow-up UPDATE after spawn returns.

    Idempotent: re-running the node (replay after interrupt, restart
    after crash, smoke test re-run) leaves the registry in the same
    state regardless of prior runs.
    """
    import json as _json  # local import keeps top-level surface tidy

    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO strategy_registry
              (strategy_id, thread_id, name, template, stage, pairs,
               timeframe, started_at, last_updated)
            VALUES (%s, %s, %s, %s, 'paper', %s, %s, now(), now())
            ON CONFLICT (strategy_id) DO UPDATE SET
              stage = 'paper',
              last_updated = now()
            """,
            (
                state["strategy_id"],
                thread_id,
                state["name"],
                state["template"],
                _json.dumps(state["pairs"]),
                state["timeframe"],
            ),
        )


# ───────────────────────── node ─────────────────────────


async def paper_spawn(
    state: StrategyState,
    config: RunnableConfig,
    *,
    spawn_container_fn: SpawnContainerFn | None = None,
) -> dict[str, Any]:
    """First node in the paper subgraph — see module docstring for contract.

    Returns a state-update dict (LangGraph node convention). Never
    raises on spawn failure; failures route to ``stage="archived"``
    with a descriptive ``failure_reason``.
    """
    spawn_fn: SpawnContainerFn = spawn_container_fn or spawn_paper_container

    strategy_id = state["strategy_id"]
    pairs = state["pairs"]
    params = state.get("params") or {}
    artifacts = state.get("artifacts") or {}

    # Stake amount: explicit param wins; otherwise BRD §10 default.
    default_stake = LIVE_CAPITAL_CAP_USD / MAX_OPEN_TRADES
    stake_amount = float(params.get("stake_amount", default_stake))

    # Generated strategy path is written by Stage 5's generator node
    # into state.artifacts. Missing it is a research-subgraph contract
    # violation — archive cleanly with a diagnostic.
    strategy_path_str = artifacts.get("generated_strategy_path")
    if not strategy_path_str:
        logger.error(
            "paper_spawn missing generated_strategy_path strategy_id=%s",
            strategy_id,
        )
        return {
            "stage": "archived",
            "failure_reason": (
                "paper_spawn_missing_strategy_path: state.artifacts has no "
                "'generated_strategy_path' — research subgraph must run first"
            ),
        }
    strategy_path = Path(strategy_path_str)

    thread_id = _thread_id_for(config, strategy_id)

    # 2 + 3: query used ports, allocate, upsert row BEFORE spawn.
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT freqtrade_api_url FROM strategy_registry "
                "WHERE freqtrade_api_url IS NOT NULL"
            )
            rows = await cur.fetchall()
        used = _used_ports_from_urls([r[0] for r in rows])
        port = next_free_paper_port(used)

        await _upsert_registry_row(conn, state, thread_id)
        await conn.commit()
    finally:
        await conn.close()

    logger.info(
        "paper_spawn allocated port=%d strategy_id=%s thread_id=%s",
        port,
        strategy_id,
        thread_id,
    )

    # 4. Call spawn — both branches archive cleanly, neither raises.
    try:
        api_url = await spawn_fn(
            strategy_id=strategy_id,
            pair_whitelist=pairs,
            stake_amount=stake_amount,
            strategy_module_path=strategy_path,
            port=port,
        )
    except PaperSpawnTimeout as exc:
        logger.error(
            "paper_spawn timeout strategy_id=%s port=%d exc=%s",
            strategy_id,
            port,
            exc,
        )
        return {
            "stage": "archived",
            "failure_reason": f"paper_spawn_timeout: {strategy_id}",
        }
    except Exception as exc:  # noqa: BLE001 — any spawn failure archives cleanly
        logger.error(
            "paper_spawn failed strategy_id=%s port=%d exc_type=%s exc=%s",
            strategy_id,
            port,
            type(exc).__name__,
            exc,
        )
        type_name = type(exc).__name__
        return {
            "stage": "archived",
            "failure_reason": f"paper_spawn_failed: {type_name}: {str(exc)[:200]}",
        }

    # 5. UPDATE registry with the URL.
    conn = await _connect_app_db()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE strategy_registry SET freqtrade_api_url = %s, "
                "last_updated = now() WHERE strategy_id = %s",
                (api_url, strategy_id),
            )
        await conn.commit()
    finally:
        await conn.close()

    # 6. State update — paper_started_at is the 30-day clock anchor
    # SPEC §6 (d6736ba) locked. divergence_check (7e) enforces
    # elapsed_days >= MIN_PAPER_DAYS against this timestamp.
    started_at_iso = datetime.now(UTC).isoformat()
    worker_dir = WORKERS_ROOT / strategy_id
    return {
        "stage": "paper",
        "freqtrade_api_url": api_url,
        "freqtrade_userdir": str(worker_dir),
        "freqtrade_process_id": f"paper-{strategy_id}",
        "artifacts": {
            **artifacts,
            "paper_started_at": started_at_iso,
        },
    }
