"""Unit tests for the Stage 9a supervisor read tools + DB reads.

Two layers, both Postgres-free / API-key-free:

  - The DB-read functions (``aget_portfolio_snapshot``,
    ``aget_current_regime``) against a mocked ``psycopg.AsyncConnection``
    (same pattern as ``tests/unit/test_events.py``).
  - The tool implementations (``_view_portfolio_impl`` /
    ``_get_market_regime_impl`` / ``_query_store_impl``) against the
    ContextVars the Stage 9c runner will set. We call the ``_impl``
    functions directly (not via langchain tool-invocation) so the tests
    are free of cross-thread ContextVar quirks; the @tool wrappers are
    one-line shells over these, surface-checked in
    ``test_read_tools_surface``.

Each test sets every ContextVar it reads (including the default cases),
so test order can never leak state between cases.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from langgraph.store.memory import InMemoryStore

from orchestrator import supervisor
from orchestrator.supervisor import (
    READ_TOOLS,
    SUPERVISOR_TOOLS,
    _current_portfolio,
    _current_regime,
    _current_store,
    _current_strategies,
    _get_market_regime_impl,
    _list_strategies_impl,
    _query_store_impl,
    _view_portfolio_impl,
    aget_current_regime,
    aget_portfolio_snapshot,
    alist_strategies,
)


def _conn_with(*, fetchall: Any = None, fetchone: Any = None) -> AsyncMock:
    """A mocked async conn whose cursor returns the given fetch results."""
    cur = AsyncMock()
    cur.fetchall.return_value = fetchall
    cur.fetchone.return_value = fetchone

    cursor_cm = AsyncMock()
    cursor_cm.__aenter__.return_value = cur
    cursor_cm.__aexit__.return_value = None

    conn = AsyncMock()
    conn.cursor = lambda: cursor_cm
    return conn


# ─── aget_portfolio_snapshot ───────────────────────────────────────────


async def test_portfolio_snapshot_counts_by_stage() -> None:
    """active = sum of non-archived counts; live = the 'live' bucket."""
    conn = _conn_with(fetchall=[("research", 2), ("paper", 1), ("live", 1), ("archived", 5)])

    snapshot = await aget_portfolio_snapshot(conn)

    assert snapshot["by_stage"] == {"research": 2, "paper": 1, "live": 1, "archived": 5}
    assert snapshot["active"] == 4  # 2 + 1 + 1, archived excluded
    assert snapshot["live"] == 1


async def test_portfolio_snapshot_empty_registry() -> None:
    """No rows → a valid all-zero snapshot, not an error."""
    conn = _conn_with(fetchall=[])

    snapshot = await aget_portfolio_snapshot(conn)

    assert snapshot == {"by_stage": {}, "active": 0, "live": 0}


async def test_portfolio_snapshot_no_live_bucket() -> None:
    """live defaults to 0 when no live thread exists."""
    conn = _conn_with(fetchall=[("research", 3), ("archived", 1)])

    snapshot = await aget_portfolio_snapshot(conn)

    assert snapshot["active"] == 3
    assert snapshot["live"] == 0


# ─── aget_current_regime ───────────────────────────────────────────────


async def test_current_regime_returns_latest_label() -> None:
    conn = _conn_with(fetchone=("high_vol_up",))

    assert await aget_current_regime(conn) == "high_vol_up"


async def test_current_regime_empty_log_returns_unknown() -> None:
    """Fresh install, no regime_log rows → 'unknown' sentinel."""
    conn = _conn_with(fetchone=None)

    assert await aget_current_regime(conn) == "unknown"


# ─── view_portfolio tool ───────────────────────────────────────────────


def test_view_portfolio_returns_contextvar_snapshot() -> None:
    snap = {"by_stage": {"paper": 2}, "active": 2, "live": 0}
    _current_portfolio.set(snap)

    assert _view_portfolio_impl() == snap


def test_view_portfolio_unset_returns_zeroed_copy() -> None:
    """Unset ContextVar → a fresh zeroed snapshot, and a COPY (not the constant)."""
    _current_portfolio.set(None)

    result = _view_portfolio_impl()
    assert result == {"by_stage": {}, "active": 0, "live": 0}

    # Mutating the result must not corrupt the module-level default.
    result["active"] = 99
    assert supervisor._EMPTY_SNAPSHOT["active"] == 0


# ─── get_market_regime tool ────────────────────────────────────────────


def test_get_market_regime_reads_contextvar() -> None:
    _current_regime.set("low_vol_down")
    assert _get_market_regime_impl() == "low_vol_down"


def test_get_market_regime_default_unknown() -> None:
    _current_regime.set("unknown")
    assert _get_market_regime_impl() == "unknown"


# ─── query_store tool ──────────────────────────────────────────────────


async def test_query_store_no_store_returns_empty() -> None:
    """Unset store ContextVar → [] (fresh install / no runner), not an error."""
    _current_store.set(None)

    assert await _query_store_impl("failures", None, 10) == []


async def test_query_store_failures_scoped_to_current_regime() -> None:
    """With no explicit regime, scopes to _current_regime and returns failures."""
    store = InMemoryStore()
    await store.aput(
        ("failures", "mid_vol_flat"),
        "strategy_old",
        {"hypothesis": "rsi mean reversion", "failure_reason": "lookahead_bias"},
    )
    _current_store.set(store)
    _current_regime.set("mid_vol_flat")

    out = await _query_store_impl("failures", None, 10)

    assert len(out) == 1
    assert out[0]["key"] == "strategy_old"
    assert out[0]["failure_reason"] == "lookahead_bias"


async def test_query_store_explicit_regime_overrides_contextvar() -> None:
    """An explicit regime arg wins over the current-regime ContextVar."""
    store = InMemoryStore()
    await store.aput(("wins", "high_vol_up"), "winner", {"hypothesis": "breakout"})
    _current_store.set(store)
    _current_regime.set("low_vol_flat")  # different from the explicit arg

    out = await _query_store_impl("wins", "high_vol_up", 10)

    assert len(out) == 1
    assert out[0]["key"] == "winner"


async def test_query_store_wins_empty_for_unseen_regime() -> None:
    store = InMemoryStore()
    _current_store.set(store)
    _current_regime.set("mid_vol_flat")

    assert await _query_store_impl("wins", None, 10) == []


# ─── tool surface ──────────────────────────────────────────────────────


def test_read_tools_surface() -> None:
    """READ_TOOLS exposes exactly the three 9a read tools by name (frozen surface)."""
    names = {t.name for t in READ_TOOLS}
    assert names == {"view_portfolio", "query_store", "get_market_regime"}


# ─── alist_strategies (DB read) + list_strategies tool (9c) ────────────


async def test_alist_strategies_maps_rows() -> None:
    """Maps registry rows → per-thread dicts; last_updated → ISO string."""
    from datetime import UTC, datetime

    ts = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    conn = _conn_with(
        fetchall=[
            ("s1", "paper", "strat-1", 12.34, ts),
            ("s2", "research", "strat-2", 0.5, ts),
        ]
    )

    out = await alist_strategies(conn)

    assert out == [
        {
            "strategy_id": "s1",
            "stage": "paper",
            "name": "strat-1",
            "age_days": 12.3,  # rounded to 1 dp
            "last_transition_at": ts.isoformat(),
        },
        {
            "strategy_id": "s2",
            "stage": "research",
            "name": "strat-2",
            "age_days": 0.5,
            "last_transition_at": ts.isoformat(),
        },
    ]


async def test_alist_strategies_empty() -> None:
    conn = _conn_with(fetchall=[])
    assert await alist_strategies(conn) == []


def test_list_strategies_impl_reads_contextvar() -> None:
    rows = [{"strategy_id": "s1", "stage": "live", "name": "x", "age_days": 3.0}]
    _current_strategies.set(rows)
    assert _list_strategies_impl() == rows


def test_list_strategies_impl_unset_returns_empty() -> None:
    _current_strategies.set(None)
    assert _list_strategies_impl() == []


def test_supervisor_tools_surface() -> None:
    """SUPERVISOR_TOOLS = the 3 read tools + list_strategies (Arch 2: no write tools)."""
    names = {t.name for t in SUPERVISOR_TOOLS}
    assert names == {"view_portfolio", "query_store", "get_market_regime", "list_strategies"}
