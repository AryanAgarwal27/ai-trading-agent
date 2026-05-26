"""Per-strategy state schema (BRD §5.7).

Verbatim translation of the BRD §5.7 schema to runtime types. Send-fan-out
fields use Annotated[..., operator.add] reducers (BRD §6.3) — without these,
parallel workers' writes overwrite each other.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


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
    payload: dict[str, Any]


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
    params: dict[str, Any]
    freqai_config: dict[str, Any] | None
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
    gate_decisions: dict[str, Any]

    # Execution
    freqtrade_userdir: str | None
    freqtrade_process_id: int | None
    freqtrade_api_url: str | None
    artifacts: dict[str, Any]

    # Timestamps
    started_at: str
    last_updated: str

    # Terminal
    failure_reason: str | None
