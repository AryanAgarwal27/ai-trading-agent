"""Stage 7c integration tests for the ``paper_spawn`` node.

Five tests, all marked ``integration`` (real Postgres) but NOT
``freqtrade`` — the spawn helper is stubbed via the
``spawn_container_fn`` injection seam, so Docker is not required.

What we verify here is the node's CONTRACT SURFACE:

1. Registry row written BEFORE spawn (orphan-container prevention).
2. Success path returns ``stage="paper"`` with the spawned URL +
   ``paper_started_at`` ISO timestamp.
3. PaperSpawnTimeout archives with ``failure_reason`` prefix.
4. Generic exception archives with exact ``failure_reason`` format.
5. Port allocator sees already-allocated ports from the registry.

The integration test in
:mod:`tests.integration.test_freqtrade_lifecycle` covers the
real-Docker spawn lifecycle separately.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from orchestrator.subgraphs.paper import paper_spawn
from orchestrator.tools.freqtrade_lifecycle import PaperSpawnTimeout

pytestmark = pytest.mark.integration


# ───────────────────────── fixtures ─────────────────────────


def _libpq_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _dsn() -> str:
    return _libpq_dsn(os.environ["DATABASE_URL"])


@pytest.fixture
async def cleanup_strategy_ids() -> Any:
    """Yields a list the test appends strategy_ids to; deletes them after."""
    ids: list[str] = []
    yield ids
    if not ids:
        return
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)",
                (ids,),
            )
        await conn.commit()


def _minimal_state(strategy_id: str, generated_strategy_path: str) -> dict[str, Any]:
    """Build the minimal StrategyState shape paper_spawn expects."""
    return {
        "strategy_id": strategy_id,
        "name": f"test-{strategy_id}",
        "hypothesis": "synthetic test hypothesis",
        "template": "mean_reversion_template",
        "params": {"stake_amount": 25.0, "rsi_buy_threshold": 30},
        "freqai_config": None,
        "pairs": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "5m",
        "stage": "paper_gate",
        "backtest_results": [],
        "robustness_results": [],
        "agent_votes": [],
        "revision_count": 0,
        "critic_notes": [],
        "gate_decisions": {},
        "freqtrade_userdir": None,
        "freqtrade_process_id": None,
        "freqtrade_api_url": None,
        "artifacts": {"generated_strategy_path": generated_strategy_path},
        "started_at": datetime.now(UTC).isoformat(),
        "last_updated": datetime.now(UTC).isoformat(),
        "failure_reason": None,
    }


def _empty_config() -> dict[str, Any]:
    """RunnableConfig without an explicit thread_id — node falls back to default."""
    return {"configurable": {}}


# A real-looking strategy module path; the stubbed spawn_fn doesn't open it.
# We point at the actual mean_reversion_template so the path is plausible
# and existence checks pass in case the helper grows one later.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DUMMY_STRATEGY = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"


# ───────────────────────── tests ─────────────────────────


async def test_paper_spawn_writes_registry_row_before_spawn_call(
    cleanup_strategy_ids: list[str],
) -> None:
    """Registry row MUST exist at the moment spawn_fn is called.

    Orphan-container prevention contract: if spawn succeeds but the
    registry write fails (or hasn't happened yet), the orchestrator
    loses track of a live container. The fix is to write the registry
    row FIRST, even though it means we have a row that promises a
    container that doesn't exist yet for the duration of spawn.
    """
    strategy_id = f"sp-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    seen_row_at_spawn = {"value": False}

    async def stub_spawn(*, strategy_id: str, **_kwargs: Any) -> str:
        # Open a fresh connection from this stub and check that the
        # registry already has the row at the moment we get called.
        async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT stage FROM strategy_registry WHERE strategy_id = %s",
                    (strategy_id,),
                )
                row = await cur.fetchone()
        seen_row_at_spawn["value"] = row is not None and row[0] == "paper"
        return "http://127.0.0.1:8123"

    state = _minimal_state(strategy_id, str(DUMMY_STRATEGY))
    result = await paper_spawn(state, _empty_config(), spawn_container_fn=stub_spawn)

    assert seen_row_at_spawn["value"], (
        "registry row absent at spawn-time — orphan-container risk; "
        "paper_spawn must upsert the registry BEFORE calling spawn"
    )
    assert result["stage"] == "paper"


async def test_paper_spawn_returns_paper_stage_on_success(
    cleanup_strategy_ids: list[str],
) -> None:
    """Successful spawn yields the documented state-update shape."""
    strategy_id = f"sp-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    async def stub_spawn(*, port: int, **_kwargs: Any) -> str:
        return f"http://127.0.0.1:{port}"

    before = datetime.now(UTC)
    state = _minimal_state(strategy_id, str(DUMMY_STRATEGY))
    result = await paper_spawn(state, _empty_config(), spawn_container_fn=stub_spawn)
    after = datetime.now(UTC)

    assert result["stage"] == "paper"
    assert result["freqtrade_api_url"].startswith("http://127.0.0.1:")
    assert result["freqtrade_process_id"] == f"paper-{strategy_id}"
    assert result["freqtrade_userdir"].endswith(strategy_id.replace("/", os.sep))

    # paper_started_at: parseable ISO timestamp within the test window.
    started_at = datetime.fromisoformat(result["artifacts"]["paper_started_at"])
    assert (
        before <= started_at <= after
    ), f"paper_started_at={started_at!r} outside test window [{before}, {after}]"

    # And the registry row should now carry the URL.
    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT freqtrade_api_url FROM strategy_registry WHERE strategy_id = %s",
                (strategy_id,),
            )
            row = await cur.fetchone()
    assert row is not None
    assert row[0] == result["freqtrade_api_url"]


async def test_paper_spawn_archives_on_timeout(
    cleanup_strategy_ids: list[str],
) -> None:
    """PaperSpawnTimeout → archived with timeout-prefix failure_reason."""
    strategy_id = f"sp-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    async def stub_spawn(**_kwargs: Any) -> str:
        raise PaperSpawnTimeout("container did not /ping within budget")

    state = _minimal_state(strategy_id, str(DUMMY_STRATEGY))
    result = await paper_spawn(state, _empty_config(), spawn_container_fn=stub_spawn)

    assert result["stage"] == "archived"
    assert result["failure_reason"].startswith("paper_spawn_timeout:")
    assert strategy_id in result["failure_reason"]


async def test_paper_spawn_archives_on_generic_failure(
    cleanup_strategy_ids: list[str],
) -> None:
    """Generic exception → archived with exact failure_reason format."""
    strategy_id = f"sp-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(strategy_id)

    async def stub_spawn(**_kwargs: Any) -> str:
        raise RuntimeError("docker daemon unreachable")

    state = _minimal_state(strategy_id, str(DUMMY_STRATEGY))
    result = await paper_spawn(state, _empty_config(), spawn_container_fn=stub_spawn)

    assert result["stage"] == "archived"
    assert result["failure_reason"] == (
        "paper_spawn_failed: RuntimeError: docker daemon unreachable"
    )


async def test_paper_spawn_handles_port_collision(
    cleanup_strategy_ids: list[str],
) -> None:
    """Already-allocated ports surface to next_free_paper_port.

    Seed two registry rows with URLs claiming ports 8100 and 8101. The
    third spawn (no other live threads) should land on 8102.
    """
    # Two existing strategies with allocated ports.
    occupant_a = f"sp-occ-{uuid.uuid4().hex[:6]}"
    occupant_b = f"sp-occ-{uuid.uuid4().hex[:6]}"
    cleanup_strategy_ids.extend([occupant_a, occupant_b])

    async with await psycopg.AsyncConnection.connect(_dsn()) as conn:
        async with conn.cursor() as cur:
            for sid, port in [(occupant_a, 8100), (occupant_b, 8101)]:
                await cur.execute(
                    """
                    INSERT INTO strategy_registry
                      (strategy_id, thread_id, name, template, stage, pairs,
                       timeframe, freqtrade_api_url, started_at, last_updated)
                    VALUES (%s, %s, %s, 'mean_reversion_template', 'paper',
                            '["BTC/USDT"]', '5m', %s, now(), now())
                    """,
                    (sid, f"strategy_{sid}", f"occ-{sid}", f"http://127.0.0.1:{port}"),
                )
        await conn.commit()

    # Now spawn — capture the port arg the stub receives.
    new_id = f"sp-{uuid.uuid4().hex[:8]}"
    cleanup_strategy_ids.append(new_id)
    received: dict[str, Any] = {}

    async def stub_spawn(*, port: int, **_kwargs: Any) -> str:
        received["port"] = port
        return f"http://127.0.0.1:{port}"

    state = _minimal_state(new_id, str(DUMMY_STRATEGY))
    result = await paper_spawn(state, _empty_config(), spawn_container_fn=stub_spawn)

    assert received["port"] == 8102, (
        f"expected port 8102 with [8100, 8101] taken, got {received['port']}; "
        "next_free_paper_port not seeing registry-allocated ports"
    )
    assert result["stage"] == "paper"
    # Sanity check the URL the node returned matches the allocated port.
    assert re.match(r"^http://127\.0\.0\.1:8102$", result["freqtrade_api_url"])
