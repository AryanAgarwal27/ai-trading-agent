"""scripts/seed_dashboard_data.py — operator-run seed for dashboard development.

UNLIKE ``scripts/midstage_seed.py`` (throwaway, deleted after each
mid-stage smoke), this script SHIPS WITH THE REPO. Used during
dashboard iteration across Stage 6 → Stage 10 — any future operator
working on a card's layout, copy, or rendering should reach for this
script first.

Three preset shapes:

- ``--shape clean_pass``    — obvious-approve paper_gate. Backtest
  Sharpe 2.1, OOS ratio 0.78, fee_3x degradation 0.31 (well under
  cap). risk_analyst rationale enthusiastic, confidence 0.88. The
  "I'd approve this in five seconds" case for testing approve-button
  ergonomics.

- ``--shape marginal``      — on-the-fence paper_gate. Backtest just
  clears thresholds (Sharpe 1.55 vs MIN 1.5, OOS 0.62 vs MIN 0.6).
  risk_analyst rationale wary, confidence 0.58, recommending caution.
  The "actually read the rationale before clicking" case.

- ``--shape kill_switch``   — live_pause_review with the kill-switch
  path. Synthesizes a ``kill_switch_event`` artifact (drawdown 13.1%,
  ``POST /api/v1/stop``). No coordinator rationale — the dashboard's
  red distinct-render branch is the test.

Each shape:

1. Parks a checkpoint at the appropriate interrupt against the real
   ``AsyncPostgresSaver`` (using the smoke-graph helpers from
   :mod:`orchestrator.main`).
2. Inserts a ``strategy_registry`` row (upsert — re-runnable).
3. Prints the strategy_id, the uvicorn env var to set, the dashboard
   URL, and (unless ``--keep-on-exit``) the cleanup ``psql`` commands.

Operator workflow::

    .venv/Scripts/python.exe scripts/seed_dashboard_data.py --shape clean_pass

    # New shell — set the env var for THIS shape (see printed output)
    $env:AIT_SMOKE_PAPER_GATE_GRAPH = "1"
    .venv/Scripts/python.exe -m uvicorn orchestrator.main:app --reload --host 127.0.0.1 --port 8000

    # Another shell:
    .venv/Scripts/streamlit run dashboard/app.py --server.address 127.0.0.1

    # Open http://127.0.0.1:8501; iterate on the card; click Approve or Reject
    # when satisfied (or just Ctrl+C uvicorn/streamlit and re-seed for the
    # next iteration).

After the smoke, run the printed cleanup commands to remove the seeded
rows from ``strategy_registry``, ``gate_audits`` (if any landed), and
``langgraph_checkpoints.*``. Then UNSET the env var.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

# Windows event-loop convention (see scripts/README.md). MUST be set
# BEFORE any psycopg / langgraph import.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import psycopg  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402

from orchestrator.main import (  # noqa: E402
    _build_live_pause_review_only_graph_for_smoke,
    _build_paper_gate_only_graph_for_smoke,
)
from orchestrator.observability.events import _libpq_dsn  # noqa: E402

# ─── Preset payload generators ─────────────────────────────────────────


def _state_clean_pass(strategy_id: str) -> dict[str, Any]:
    """Obvious-approve paper_gate state. Every metric well above floor."""
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {
            "backtest": {
                "passed": True,
                "best_param_set_id": "ps_clean",
                "sharpe_is": 2.1,
                "oos_ratio": 0.78,
                "max_dd": 0.09,
                "trades_is": 412,
                "trades_oos": 96,
                "profit_factor_is": 2.55,
            },
            "robustness": {
                "passed": True,
                "monte_carlo": {"pct_5_final_equity": 1.18, "n_trades": 280},
                "regime": {
                    "regimes_passed": 3,
                    "by_regime": {
                        "low_vol_up": {"mean_sharpe": 1.9, "n_folds": 2},
                        "mid_vol_flat": {"mean_sharpe": 2.4, "n_folds": 3},
                        "high_vol_down": {"mean_sharpe": 1.1, "n_folds": 1},
                    },
                },
                "fee_stress": {"degradation_2x": 0.18, "degradation_3x": 0.31},
            },
            "risk_analyst": {
                "decision": "approve",
                "primary_concern": "None — all gates clear by comfortable margins.",
                "rationale": (
                    "All gates passed cleanly. Walk-forward Sharpe holds at 2.1 "
                    "in-sample with a 0.78 OOS ratio — the strongest combination "
                    "across the v1 corpus. All three vol-regime buckets clear "
                    "their per-regime Sharpe gates with positive returns; even "
                    "high-vol-down posts 1.1.\n\n"
                    "Monte Carlo 5th-percentile final equity at 1.18× capital "
                    "leaves a real margin against the 1.0 breakeven floor — "
                    "rare in crypto-spot strategies. Fee stress 2x degrades "
                    "only 18% (cap 40%); 3x degrades 31% (cap 60%). Plenty of "
                    "headroom for fee-tier drift.\n\n"
                    "Recommend approval. This is the cleanest research output "
                    "since the v1 thresholds were tuned."
                ),
                "confidence": 0.88,
            },
        },
    }


def _state_marginal(strategy_id: str) -> dict[str, Any]:
    """On-the-fence paper_gate state. Metrics within 5% of thresholds."""
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {
            "backtest": {
                "passed": True,
                "best_param_set_id": "ps_marginal",
                "sharpe_is": 1.55,
                "oos_ratio": 0.62,
                "max_dd": 0.19,
                "trades_is": 168,
                "trades_oos": 34,
                "profit_factor_is": 1.62,
            },
            "robustness": {
                "passed": True,
                "monte_carlo": {"pct_5_final_equity": 1.02, "n_trades": 145},
                "regime": {
                    "regimes_passed": 2,
                    "by_regime": {
                        "low_vol_up": {"mean_sharpe": 1.7, "n_folds": 2},
                        "mid_vol_flat": {"mean_sharpe": 1.4, "n_folds": 3},
                        "high_vol_down": {"mean_sharpe": -0.1, "n_folds": 1},
                    },
                },
                "fee_stress": {"degradation_2x": 0.38, "degradation_3x": 0.58},
            },
            "risk_analyst": {
                "decision": "approve",
                "primary_concern": (
                    "Multiple metrics sit within 5% of their thresholds — "
                    "the strategy clears every gate but with no margin for "
                    "live-trading slip."
                ),
                "rationale": (
                    "This strategy clears every threshold by a thin margin and "
                    "deserves careful operator review. Walk-forward IS Sharpe "
                    "1.55 just above the 1.5 floor; OOS ratio 0.62 just above "
                    "the 0.6 floor. One regime bucket (high-vol-down) posts a "
                    "slight loss — the gate passed on 2-of-3 (the minimum), "
                    "not 3-of-3.\n\n"
                    "Fee stress is the most worrying axis: 2x degrades 38% "
                    "(cap 40%, margin 2 pp); 3x degrades 58% (cap 60%, margin "
                    "2 pp). If real Binance fees drift even marginally during "
                    "the paper window, the metric crosses the line.\n\n"
                    "Monte Carlo 5th-percentile final equity at 1.02× is "
                    "barely positive — a 1000-iter bootstrap puts the bottom "
                    "5% of outcomes at near-breakeven.\n\n"
                    "Recommend approval ONLY if the operator has bandwidth to "
                    "monitor closely during week 1 of paper. The dashboard's "
                    "Approve button should not be a reflex on this card."
                ),
                "confidence": 0.58,
            },
        },
    }


def _state_kill_switch(strategy_id: str) -> dict[str, Any]:
    """Kill-switch path of live_pause_review.

    Populates ``artifacts.kill_switch_event`` so
    :func:`build_interrupt_payload` discriminates to ``path="kill_switch"``.
    """
    return {
        "strategy_id": strategy_id,
        "gate_decisions": {},
        "artifacts": {
            "kill_switch_event": {
                "reason": "drawdown_12pct_exceeded",
                "metrics": {
                    "drawdown": 0.131,
                    "consecutive_losses": 4,
                    "running_peak_equity": 487.40,
                    "current_equity": 423.62,
                },
                "action_taken": "POST /api/v1/stop",
                "fired_at": "2026-05-27T14:32:11Z",
            },
            "drawdown_trajectory": [0.02, 0.04, 0.07, 0.09, 0.11, 0.124, 0.131],
            "recent_trades": [
                {"pair": "BTC/USDT", "pnl": -28.40, "closed_at": "2026-05-27T14:18:02Z"},
                {"pair": "ETH/USDT", "pnl": -19.10, "closed_at": "2026-05-27T14:24:55Z"},
                {"pair": "BTC/USDT", "pnl": -42.10, "closed_at": "2026-05-27T14:31:47Z"},
            ],
        },
    }


# ─── Shape configuration ───────────────────────────────────────────────


_SHAPE_CONFIG: dict[str, dict[str, Any]] = {
    "clean_pass": {
        "state_fn": _state_clean_pass,
        "graph_builder": _build_paper_gate_only_graph_for_smoke,
        "env_var": "AIT_SMOKE_PAPER_GATE_GRAPH",
        "stage": "validation",
        "template": "mean_reversion_template",
    },
    "marginal": {
        "state_fn": _state_marginal,
        "graph_builder": _build_paper_gate_only_graph_for_smoke,
        "env_var": "AIT_SMOKE_PAPER_GATE_GRAPH",
        "stage": "validation",
        "template": "freqai_classifier_template",
    },
    "kill_switch": {
        "state_fn": _state_kill_switch,
        "graph_builder": _build_live_pause_review_only_graph_for_smoke,
        "env_var": "AIT_SMOKE_LIVE_PAUSE_REVIEW_GRAPH",
        "stage": "live",
        "template": "freqai_classifier_template",
    },
}


# ─── Graph + DB operations ─────────────────────────────────────────────


async def _park_at_interrupt(
    shape: str, strategy_id: str, thread_id: str
) -> bool:
    """Drive the shape's graph to its interrupt against a fresh saver
    context. Returns True iff the checkpoint actually parked."""
    cfg = _SHAPE_CONFIG[shape]
    state = cfg["state_fn"](strategy_id)
    checkpoint_uri = os.environ["LANGGRAPH_CHECKPOINT_URI"]
    async with AsyncPostgresSaver.from_conn_string(checkpoint_uri) as saver:
        await saver.setup()
        graph = cfg["graph_builder"](saver)
        graph_config = {"configurable": {"thread_id": thread_id}}
        async for _ in graph.astream(state, config=graph_config):
            pass
        snapshot = await graph.aget_state(graph_config)
        return any(getattr(t, "interrupts", ()) for t in snapshot.tasks)


async def _upsert_registry_row(strategy_id: str, thread_id: str, shape: str) -> None:
    """Upsert into strategy_registry; idempotent on re-runs."""
    cfg = _SHAPE_CONFIG[shape]
    libpq_dsn = _libpq_dsn(os.environ["DATABASE_URL"])
    conn = await psycopg.AsyncConnection.connect(libpq_dsn)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO strategy_registry
                  (strategy_id, thread_id, name, template, stage, pairs,
                   timeframe, started_at, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (strategy_id) DO UPDATE SET
                  stage = EXCLUDED.stage,
                  last_updated = now()
                """,
                (
                    strategy_id,
                    thread_id,
                    f"dashboard-seed-{shape}",
                    cfg["template"],
                    cfg["stage"],
                    json.dumps(["BTC/USDT", "ETH/USDT"]),
                    "5m",
                ),
            )
        await conn.commit()
    finally:
        await conn.close()


# ─── CLI ────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "Seed dashboard data.",
    )
    p.add_argument(
        "--shape",
        required=True,
        choices=sorted(_SHAPE_CONFIG.keys()),
        help="Which preset payload to seed.",
    )
    p.add_argument(
        "--strategy-id",
        default=None,
        help="Override the strategy_id (default: seed-<uuid-prefix>).",
    )
    p.add_argument(
        "--keep-on-exit",
        action="store_true",
        help=(
            "Don't print cleanup psql commands on exit. Default false: "
            "the script prints DELETE statements the operator can run "
            "after the smoke. With this flag, exit silently and leave "
            "the seeded rows in place (useful for iterating multiple "
            "dashboard refreshes on the same seed)."
        ),
    )
    return p.parse_args()


def _print_cleanup_commands(strategy_id: str, thread_id: str) -> None:
    print()
    print("─── Cleanup commands (run after the smoke completes) ───")
    print()
    print("  # App DB:")
    print(
        f"  psql $env:DATABASE_URL -c \""
        f"DELETE FROM gate_audits WHERE strategy_id = '{strategy_id}';\""
    )
    print(
        f"  psql $env:DATABASE_URL -c \""
        f"DELETE FROM strategy_registry WHERE strategy_id = '{strategy_id}';\""
    )
    print()
    print("  # LangGraph checkpoint DB (thread is identified by thread_id):")
    print(
        f"  psql $env:LANGGRAPH_CHECKPOINT_URI -c \""
        f"DELETE FROM checkpoint_writes WHERE thread_id = '{thread_id}'; "
        f"DELETE FROM checkpoint_blobs WHERE thread_id = '{thread_id}'; "
        f"DELETE FROM checkpoints WHERE thread_id = '{thread_id}';\""
    )
    print()
    print(
        "  # And UNSET the smoke env var in your uvicorn shell when done."
    )


async def _main(args: argparse.Namespace) -> int:
    shape: str = args.shape
    strategy_id: str = args.strategy_id or f"seed-{uuid.uuid4().hex[:8]}"
    thread_id: str = strategy_id
    cfg = _SHAPE_CONFIG[shape]

    # Set the smoke env var IN THIS PROCESS so the imports of paper_gate /
    # build_interrupt_payload that happen lazily inside the graph helpers
    # see consistent state. The operator MUST set this in the uvicorn
    # shell too — env vars don't cross processes.
    os.environ[cfg["env_var"]] = "1"

    print(f"Seeding shape={shape!r}, strategy_id={strategy_id!r}...")
    parked = await _park_at_interrupt(shape, strategy_id, thread_id)
    if not parked:
        print(
            f"ERROR: graph for shape={shape!r} did NOT park at an interrupt. "
            "The seed payload may not route through the gate node anymore.",
            file=sys.stderr,
        )
        return 1

    await _upsert_registry_row(strategy_id, thread_id, shape)

    print()
    print(
        f"✓ Seeded shape={shape!r}. strategy_id={strategy_id!r} parked at "
        f"interrupt; strategy_registry row upserted."
    )
    print()
    print("─── Next steps ───")
    print()
    print("1. Restart uvicorn with the smoke env var for this shape:")
    print()
    print(
        f"   PowerShell:  $env:{cfg['env_var']}=\"1\"; "
        ".venv\\Scripts\\python.exe -m uvicorn orchestrator.main:app --reload "
        "--host 127.0.0.1 --port 8000"
    )
    print(
        f"   Bash:        {cfg['env_var']}=1 "
        ".venv/Scripts/python.exe -m uvicorn orchestrator.main:app --reload "
        "--host 127.0.0.1 --port 8000"
    )
    print()
    print("2. In another shell, launch Streamlit:")
    print()
    print(
        "   .venv\\Scripts\\streamlit run dashboard/app.py "
        "--server.address 127.0.0.1"
    )
    print()
    print("3. Open http://127.0.0.1:8501 — the seeded thread appears with a "
          "Review button.")

    if not args.keep_on_exit:
        _print_cleanup_commands(strategy_id, thread_id)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse_args())))
