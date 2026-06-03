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
from typing import Any

import psycopg
import pytest

from orchestrator.gates.thresholds import LIVE_CAPITAL_CAP_USD
from orchestrator.subgraphs.live import live_spawn
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
            await cur.execute(
                "DELETE FROM strategy_registry WHERE strategy_id = ANY(%s)", (ids,)
            )
        await conn.commit()


def _minimal_live_state(strategy_id: str) -> dict[str, Any]:
    """Minimal LiveState shape the spawn node reads (research handoff)."""
    return {
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
    }


def _config() -> dict[str, Any]:
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

    await live_spawn(
        _minimal_live_state(new_id), _config(), spawn_live_container_fn=stub_spawn
    )

    assert captured["port"] == 8201, (
        f"expected 8201 (8200 taken, paper ports irrelevant), got {captured['port']}"
    )


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
