"""scripts/smoke_paper_monitor.py — operator smoke for the paper monitor agent.

SHIPS WITH THE REPO (like smoke_researcher / smoke_critic /
smoke_risk_analyst). Exercises the REAL Haiku 4.5 agent + all six tools
against a SYNTHETIC Freqtrade snapshot — so it needs ANTHROPIC_API_KEY
but NOT exchange credentials or a running container.

Three shapes via --shape:

  --shape healthy_early   metrics fine, only ~4 days elapsed → expect "rearm"
  --shape graduate        metrics match backtest, 31 days elapsed → expect "advance"
  --shape diverged        paper returns diverged from backtest    → expect "kill"
  --shape kill_switch     kill_switch_fired=True                  → expect "kill"

Usage::

    .venv/Scripts/python.exe scripts/smoke_paper_monitor.py --shape graduate

Prints the agent's verdict + the Command goto the mapper produces. The
verdict is LLM-driven, so the printed decision is the agent's judgement,
not a hard assertion — the expectations above are the design intent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from orchestrator.agents.monitors import (  # noqa: E402
    PaperMonitorContext,
    run_paper_monitor,
    verdict_to_command,
)


def _build_context(shape: str) -> PaperMonitorContext:
    # A backtest baseline of mildly-positive per-trade returns.
    backtest_returns = [0.012, 0.02, -0.005, 0.018, 0.009, -0.011, 0.022, 0.014] * 20

    if shape == "healthy_early":
        return PaperMonitorContext(
            status=[{"pair": "BTC/USDT", "profit_ratio": 0.004}],
            profit={"profit_closed_percent": 1.2, "max_drawdown": 0.03, "trade_count": 9},
            trades=[{"pair": "BTC/USDT", "profit_ratio": r} for r in backtest_returns[:9]],
            performance=[{"pair": "BTC/USDT", "profit": 1.2, "count": 9}],
            backtest_returns=backtest_returns,
            kill_switch_fired=False,
            elapsed_days=4.2,
        )
    if shape == "graduate":
        return PaperMonitorContext(
            status=[],
            profit={"profit_closed_percent": 6.1, "max_drawdown": 0.05, "trade_count": 160},
            trades=[{"pair": "BTC/USDT", "profit_ratio": r} for r in backtest_returns],
            performance=[{"pair": "BTC/USDT", "profit": 6.1, "count": 160}],
            backtest_returns=backtest_returns,
            kill_switch_fired=False,
            elapsed_days=31.0,
        )
    if shape == "diverged":
        # Paper returns shifted strongly negative vs the backtest baseline.
        paper = [-0.03, -0.02, -0.04, 0.005, -0.025, -0.018, -0.031, -0.012] * 20
        return PaperMonitorContext(
            status=[{"pair": "BTC/USDT", "profit_ratio": -0.02}],
            profit={"profit_closed_percent": -7.8, "max_drawdown": 0.11, "trade_count": 160},
            trades=[{"pair": "BTC/USDT", "profit_ratio": r} for r in paper],
            performance=[{"pair": "BTC/USDT", "profit": -7.8, "count": 160}],
            backtest_returns=backtest_returns,
            kill_switch_fired=False,
            elapsed_days=18.0,
        )
    if shape == "kill_switch":
        return PaperMonitorContext(
            status=[],
            profit={"profit_closed_percent": -12.5, "max_drawdown": 0.13, "trade_count": 40},
            trades=[{"pair": "BTC/USDT", "profit_ratio": r} for r in backtest_returns[:40]],
            performance=[{"pair": "BTC/USDT", "profit": -12.5, "count": 40}],
            backtest_returns=backtest_returns,
            kill_switch_fired=True,
            elapsed_days=12.0,
        )
    raise SystemExit(f"unknown shape: {shape}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        choices=["healthy_early", "graduate", "diverged", "kill_switch"],
        default="graduate",
    )
    args = parser.parse_args()

    ctx = _build_context(args.shape)
    print(f"=== smoke_paper_monitor --shape {args.shape} ===")
    print(f"elapsed_days     = {ctx.elapsed_days}")
    print(f"kill_switch_fired= {ctx.kill_switch_fired}")
    print(f"profit           = {ctx.profit}")
    print()
    print("Invoking real Haiku 4.5 agent...")
    print()

    verdict = await run_paper_monitor(ctx)
    cmd = verdict_to_command(verdict)

    print("=== verdict ===")
    print(f"decision           = {verdict.decision}")
    print(f"primary_observation= {verdict.primary_observation}")
    print(f"rationale          = {verdict.rationale}")
    print(f"confidence         = {verdict.confidence}")
    print()
    print(f"mapped Command.goto = {cmd.goto}")


if __name__ == "__main__":
    asyncio.run(main())
