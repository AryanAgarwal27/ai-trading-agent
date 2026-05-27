import asyncio
from dotenv import load_dotenv
load_dotenv()

from orchestrator.agents.risk_analyst import risk_analyst_node

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