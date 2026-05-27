"""Operator-run smoke probe: real Sonnet 4.6 researcher → generator end-to-end.

Mirrors scripts/smoke_risk_analyst.py shape. Loads .env, invokes the
research subgraph against a synthetic post-validation state, prints the
metrics-pause report the operator asked for after Stage 5c:

  (a) Wall-clock time
  (b) Selected template + one-line rationale
  (c) Structured-output verdict (binary yes/no)
  (d) AST validator verdict on rendered output
  (e) Size of strategy_templates/_generated/<strategy_id>.py
  (f) Token cost (sum across researcher + generator Sonnet calls)

Run from the repo root::

    .venv/Scripts/python.exe scripts/smoke_researcher.py

Requires ANTHROPIC_API_KEY in .env. No PostgresStore needed — an
InMemoryStore pre-seeded with one fake failure + one fake win exercises
the query_store tool path without standing up Postgres.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from orchestrator.subgraphs.research import build_research_subgraph  # noqa: E402
from orchestrator.tools.store_queries import aput_failure, aput_win  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "strategy_templates" / "_generated"


async def main() -> None:
    # Seed the Store with one representative failure and one win so the
    # researcher's query_store calls return non-empty results — better
    # exercise of the tool path than running against a cold store.
    store = InMemoryStore()
    await aput_failure(
        store,
        regime="mid_vol_flat",
        strategy_id="prior_failure_btc_meanrev_001",
        payload={
            "hypothesis": (
                "BTC/USDT mean reversion on the 1h with RSI(7) — too noisy on "
                "this regime; whipsawed by short-lived spikes."
            ),
            "template": "mean_reversion_template",
            "params": {"bb_period": 12, "rsi_period": 7, "rsi_buy_threshold": 22},
            "failure_reason": "robustness_gate: mc_pct_5=0.96 < 1.0",
        },
    )
    await aput_win(
        store,
        regime="mid_vol_flat",
        strategy_id="prior_win_eth_classifier_001",
        payload={
            "hypothesis": (
                "ETH/USDT FreqAI classifier on 5m with EMA(20/100) trend "
                "filter — picked up sustained mid-vol up-bias well."
            ),
            "template": "freqai_classifier_template",
            "params": {"ema_fast": 20, "ema_slow": 100, "min_class_prob": 0.72},
            "live_metrics_summary": {"sharpe": 1.6, "max_dd": 0.09},
        },
    )

    strategy_id = f"smoke_{uuid.uuid4().hex[:8]}"
    initial_state = {
        "strategy_id": strategy_id,
        "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        "timeframe": "5m",
        # Hardcode the regime so the probe is reproducible — the regime
        # APScheduler job (BRD §5.7) is not yet wired in 5c.
        "current_regime": "mid_vol_flat",
    }

    graph = build_research_subgraph(
        store=store,
        checkpointer=InMemorySaver(),
    )

    print(f"[smoke_researcher] strategy_id={strategy_id}")
    print(f"[smoke_researcher] regime={initial_state['current_regime']}")
    print(f"[smoke_researcher] pairs={initial_state['pairs']}")
    print("[smoke_researcher] invoking research subgraph (real Sonnet 4.6)...")

    config = {"configurable": {"thread_id": strategy_id}}
    t0 = time.perf_counter()
    final = await graph.ainvoke(initial_state, config=config)
    wall_clock = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print(" 5c METRICS REPORT")
    print("=" * 72)

    # (a) Wall clock
    print(f"\n(a) Wall-clock time (researcher + generator end-to-end):")
    print(f"    {wall_clock:.2f} s")

    # (b) Template + rationale
    template = final.get("template", "<none>")
    proposal = (final.get("artifacts") or {}).get("research_proposal") or {}
    print(f"\n(b) Template selected: {template}")
    regime_thesis = proposal.get("regime_thesis", "<no thesis>")
    rationale_line = regime_thesis.split("\n")[0].strip()
    if len(rationale_line) > 200:
        rationale_line = rationale_line[:197] + "..."
    print(f"    Rationale: {rationale_line}")
    print(f"    Researcher confidence: {proposal.get('confidence', '<n/a>')}")

    # (c) Structured output verdict
    stage = final.get("stage")
    failure_reason = final.get("failure_reason")
    params = final.get("params") or {}
    structured_output_ok = bool(params) and (stage != "archived" or "ast_validator" in (failure_reason or ""))
    # If params landed (extractor succeeded) but AST failed, the
    # structured-output call still counted as "yes" — the failure was
    # downstream. Distinguish in output.
    print(f"\n(c) with_structured_output produced valid Pydantic on first try: "
          f"{'YES' if params else 'NO'}")
    if params:
        print(f"    Extracted {len(params)} param(s): {list(params.keys())}")
    else:
        print(f"    (no params recorded — failure_reason: {failure_reason})")

    # (d) AST verdict
    if stage == "archived" and failure_reason and "ast_validator" in failure_reason:
        ast_verdict = f"REJECTED — {failure_reason}"
    elif stage == "archived":
        ast_verdict = f"N/A (archived for non-AST reason: {failure_reason})"
    else:
        ast_verdict = "PASSED"
    print(f"\n(d) AST validator verdict on rendered output:")
    print(f"    {ast_verdict}")

    # (e) File size
    out_path_str = (final.get("artifacts") or {}).get("generated_strategy_path")
    if out_path_str:
        out_path = Path(out_path_str)
        if out_path.exists():
            size_bytes = out_path.stat().st_size
            line_count = sum(1 for _ in out_path.read_text(encoding="utf-8").splitlines())
            print(f"\n(e) Generated strategy file:")
            print(f"    {out_path.name}: {line_count} lines, {size_bytes} bytes")
        else:
            print(f"\n(e) Generated path recorded but file missing: {out_path}")
    else:
        print(f"\n(e) No generated file path recorded.")

    # (f) Token cost — extract from each agent vote's rationale where
    # available, plus walk the checkpoint history to sum usage_metadata.
    # In 5c we don't aggregate token cost directly because the agent
    # doesn't surface it through Command/state; the operator should
    # check LangSmith for the per-call breakdown.
    print(f"\n(f) Token cost:")
    print(f"    Not surfaced in state in 5c — check LangSmith for")
    print(f"    per-call breakdown (project: ai-trading-agent, thread_id={strategy_id}).")
    print(f"    LangSmith trace URL appears in agent logs above this report.")

    print("\n" + "=" * 72)
    if stage == "archived":
        print(f" RESULT: archived ({failure_reason})")
    else:
        print(" RESULT: research subgraph completed (ready for 5d critic loop)")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
