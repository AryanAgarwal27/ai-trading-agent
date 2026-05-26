"""Per-strategy state schema (BRD §5.7).

Verbatim translation of the BRD §5.7 schema to runtime types. Send-fan-out
fields use Annotated[..., operator.add] reducers (BRD §6.3) — without these,
parallel workers' writes overwrite each other.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, Optional, TypedDict


class BacktestResult(TypedDict):
    param_set_id: str
    pair: str
    timeframe: str
    fold_id: str
    is_sharpe: float
    oos_sharpe: float
    profit_factor: float
    max_dd: float
    trades: int
    raw_zip_path: str


class RobustnessResult(TypedDict):
    kind: Literal["monte_carlo", "regime", "fee_stress", "walk_forward"]
    payload: dict


class AgentVote(TypedDict):
    agent: str
    verdict: Literal["pass", "fail", "revise", "pause", "continue"]
    rationale: str
    confidence: float


Stage = Literal[
    "research",
    "validation",
    "paper_gate",
    "paper",
    "live_gate",
    "live",
    "archived",
]


class StrategyState(TypedDict):
    # Identity
    strategy_id: str
    name: str
    hypothesis: str
    template: str
    params: dict
    freqai_config: Optional[dict]
    pairs: list[str]
    timeframe: str

    # Lifecycle
    stage: Stage

    # Reducers — Send workers append here (BRD §6.3).
    backtest_results: Annotated[list[BacktestResult], add]
    robustness_results: Annotated[list[RobustnessResult], add]
    agent_votes: Annotated[list[AgentVote], add]

    # Critic / reflection
    revision_count: int
    critic_notes: list[str]

    # Gate audit
    gate_decisions: dict

    # Execution
    freqtrade_userdir: Optional[str]
    freqtrade_process_id: Optional[int]
    freqtrade_api_url: Optional[str]
    artifacts: dict

    # Timestamps
    started_at: str
    last_updated: str

    # Terminal
    failure_reason: Optional[str]
