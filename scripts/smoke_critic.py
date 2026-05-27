"""Operator-run smoke probe: real Opus 4.7 critic vs. a default-hugging strategy.

Mirrors scripts/smoke_risk_analyst.py + scripts/smoke_researcher.py
shape. Constructs a synthetic state with:

  - An aggressive-oversold-on-stretched-BB hypothesis (the kind of
    hypothesis a competent researcher might emit for a low_vol_flat
    regime).
  - A rendered strategy file made of the mean_reversion_template's
    TEXTBOOK DEFAULTS (bb_std=2.0, rsi_buy_threshold=30, ...).

The critic is invoked directly. Expected behavior under the Stage 5d
prompt (cb92c67 + critic-tightening): vote REVISE with
revision_guidance flagging at least bb_std and rsi_buy_threshold as
default-hugging. If the critic instead votes PASS, that's a real
signal about the SPEC §6 5c-finding enforcement — flag it for review.

Run from the repo root::

    .venv/Scripts/python.exe scripts/smoke_critic.py

Requires ANTHROPIC_API_KEY in .env. One Opus 4.7 call, ~$0.10–$0.30
depending on tool-loop depth.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when this script is invoked directly (vs as a module).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

# Windows event-loop convention (see scripts/README.md). MUST be set
# BEFORE any psycopg / langgraph import, including transitive ones via
# the dotenv / orchestrator imports below.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time  # noqa: E402
import uuid  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from orchestrator.agents.critic import critic_node  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"
TEMP_OUT = REPO_ROOT / "strategy_templates" / "_generated"


# Aggressive-oversold hypothesis the critic should see is NOT encoded by
# the template defaults — that's the whole point of the probe.
SYNTHETIC_HYPOTHESIS = (
    "BTC/USDT mean reversion on 5m in a low-vol flat regime — capture "
    "AGGRESSIVE oversold dips at extremely STRETCHED Bollinger Band "
    "deviations. The thesis is that the market maker bids return price "
    "to the midline within 1–3 candles after a stretched dip, but only "
    "from genuinely-oversold zones. Generic mean-reversion entries get "
    "knifed in this regime by the slow drift; only deep dips are real."
)
SYNTHETIC_REGIME_THESIS = (
    "low-vol flat regime: the BB midline is roughly horizontal, so any "
    "stretched dip below the lower band has a strong mechanical pull "
    "back to the mean. The hypothesis demands tight oversold filtering "
    "(rsi well below textbook 30) and wider-than-default BB envelopes "
    "(stds > textbook 2.0) so we only fire on dips that are genuinely "
    "outliers in this regime."
)


def _copy_template_defaults_to_synthetic_file(strategy_id: str) -> Path:
    """Copy mean_reversion_template.py verbatim into _generated/ as the
    synthetic 'rendered' strategy. Template defaults (bb_std=2.0,
    rsi_buy_threshold=30, rsi_period=14, etc.) are what the critic
    should flag as default-hugging."""
    src = TEMPLATES_DIR / "mean_reversion_template.py"
    TEMP_OUT.mkdir(parents=True, exist_ok=True)
    dest = TEMP_OUT / f"{strategy_id}.py"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


async def main() -> None:
    strategy_id = f"smoke_critic_{uuid.uuid4().hex[:8]}"
    strategy_path = _copy_template_defaults_to_synthetic_file(strategy_id)

    state = {
        "strategy_id": strategy_id,
        "template": "mean_reversion_template",
        "pairs": ["BTC/USDT"],
        "timeframe": "5m",
        "current_regime": "low_vol_flat",
        "revision_count": 0,
        "agent_votes": [],
        "critic_notes": [],
        "artifacts": {
            "generated_strategy_path": str(strategy_path),
            "research_proposal": {
                "hypothesis": SYNTHETIC_HYPOTHESIS,
                "template_name": "mean_reversion_template",
                "regime_thesis": SYNTHETIC_REGIME_THESIS,
                "suggested_param_ranges": {
                    "bb_std": "2.4-2.9",
                    "rsi_buy_threshold": "12-20",
                },
                "confidence": 0.8,
            },
        },
    }

    print(f"[smoke_critic] strategy_id={strategy_id}")
    print(f"[smoke_critic] strategy_path={strategy_path}")
    print(
        "[smoke_critic] synthetic state: aggressive-oversold hypothesis "
        "vs. template-default params (default-hugging on purpose)"
    )
    print("[smoke_critic] invoking real Opus 4.7 critic...")

    t0 = time.perf_counter()
    update = await critic_node(state)
    wall_clock = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print(" 5d CRITIC METRICS REPORT")
    print("=" * 72)

    print(f"\nWall-clock time: {wall_clock:.2f} s")

    verdict_dump = (
        (update.get("artifacts") or {})
        .get("critic_verdicts", [{}])[-1]
    )
    vote = (update.get("agent_votes") or [{}])[0]

    print(f"\nVerdict: {verdict_dump.get('verdict', '<missing>').upper()}")
    print(f"Confidence: {verdict_dump.get('confidence', '<missing>')}")
    print(f"\nPrimary concern: {verdict_dump.get('primary_concern', '<missing>')}")
    print(f"\nRationale: {verdict_dump.get('rationale', '<missing>')}")
    print(
        f"\nRevision guidance: {verdict_dump.get('revision_guidance', '<empty>')}"
    )

    print("\n" + "=" * 72)
    if verdict_dump.get("verdict") == "revise":
        guidance = verdict_dump.get("revision_guidance", "")
        flagged_slots = [
            slot
            for slot in ("bb_std", "rsi_buy_threshold", "bb_period", "rsi_period")
            if slot in guidance.lower()
        ]
        if flagged_slots:
            print(
                f" RESULT: critic caught default-hugging (flagged slots: "
                f"{flagged_slots}) ✓"
            )
        else:
            print(
                " RESULT: critic voted REVISE but didn't name the default-"
                "hugging slots specifically — review the rationale above."
            )
    elif verdict_dump.get("verdict") == "pass":
        print(
            " RESULT: critic voted PASS on a default-hugging strategy. "
            "This is a real signal about SPEC §6 5c-finding enforcement "
            "— consider tightening the critic prompt."
        )
    else:
        print(f" RESULT: unexpected verdict {verdict_dump.get('verdict')!r}")
    print("=" * 72)

    print(
        f"\n(Generated strategy file kept at {strategy_path} for inspection. "
        f"Delete manually when done.)"
    )


if __name__ == "__main__":
    asyncio.run(main())
