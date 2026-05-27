import sys
from pathlib import Path

# Make the project importable when this script is invoked directly (vs as a module).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from orchestrator.agents.risk_analyst import risk_analyst_node  # noqa: E402

state = {
    "gate_decisions": {
        "robustness": {
            "passed": True,
            "monte_carlo": {"pct_5_final_equity": 1.10, "n_trades": 200},
            "regime": {
                "regimes_passed": 3,
                "by_regime": {
                    "low_vol_up":   {"mean_sharpe": 1.3, "n_folds": 2},
                    "mid_vol_flat": {"mean_sharpe": 1.8, "n_folds": 3},
                    "high_vol_down":{"mean_sharpe": 0.4, "n_folds": 1},
                },
            },
            "fee_stress": {"degradation_2x": 0.30, "degradation_3x": 0.55},
        },
    },
}

cmd = asyncio.run(risk_analyst_node(state))
print("GOTO:", cmd.goto)
print("DECISION:", cmd.update["gate_decisions"]["risk_analyst"])