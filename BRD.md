# BRD — Autonomous AI Crypto Trading Agent

> **Single source of truth for this project. For Claude Code: read this file end-to-end at the start of every session.** This BRD overrides any conflicting content in `freq_langGraph.md` or any earlier research artifacts. Those files are reference-only.

**Repository:** https://github.com/AryanAgarwal27/ai-trading-agent
**Operator:** Aryan Agarwal (single-operator deployment)

---

## 0. How to use this BRD

1. **At session start**, read this file end-to-end.
2. Run `git log --oneline -20` and check the **Stage Table** in §13 to find the current stage.
3. **Never skip stages.** Each stage has explicit *Definition of Done* criteria. Do not advance until they are met and committed.
4. Use `TodoWrite` to track within-stage work; each TODO maps to a "Files to create" or "Tests" item in the stage spec.
5. If something is ambiguous, **ASK the user**; do not invent. If you find a better way, propose it as a diff to this BRD *before* changing code.
6. Use `/agents` to load the right subagent (see §22) before focused work.

---

## 1. What we are building

**Product.** A LangGraph-orchestrated autonomous agent that proposes, validates, paper-trades, and (with human approval) live-trades crypto-spot algorithmic strategies. Freqtrade is the execution layer; FreqAI is its ML prediction layer. LangGraph is the brain.

**Target user.** A single operator running this on a personal VPS or local machine, with capped real capital ($500 to start) and a human-in-the-loop on every move from paper → live.

**Markets.** Crypto **spot only** on Binance, Bybit, Kraken, or OKX (operator picks one in Stage 0). No futures, no margin, no leverage in v1.

**LLM autonomy mode.** *Propose-and-approve.* The LLM autonomously researches, generates parameter sets, runs backtests, runs paper-trade monitoring, and runs live-trade monitoring. A human approves every transition at `paper_gate` and `live_gate`, and reviews every `live_pause`.

### 1.1 Non-negotiable rules

These are not preferences. Each maps to a real failure mode that has destroyed real money. Violations of any of these are blockers.

1. **The LLM never writes free-form strategy code.** Strategy synthesis loads a vetted template from `strategy_templates/` and fills *parameter slots only*, validated against a Pydantic schema co-located with the template. Free-form generation produces look-ahead bias, broken index alignment, and silent division-by-zero. If you find yourself reaching for an LLM call to produce a `.py` strategy file from scratch, STOP and add a template instead.
2. **The LLM never executes trades.** Freqtrade is the only process that touches exchange APIs. The orchestrator can start, stop, and query Freqtrade instances — that is all.
3. **HITL gates are real gates.** `paper_gate`, `live_gate`, and `live_pause_review` are dynamic `interrupt()` calls. There is no auto-approve path, no env-var skip, no "just for testing" bypass. Approval is human-only via the dashboard.
4. **Paper trade ≥30 days before live.** Backtests overstate live performance by 2–5× routinely. The 30-day dry-run is the cheapest insurance you can buy.
5. **Separate API keys for paper and live.** Live keys must be on a subaccount with capped balance and **withdrawals disabled at the exchange**. Never share keys between dry-run and live Freqtrade instances.
6. **Postgres checkpointer and Postgres Store from day one.** A strategy thread may live in the graph for 30+ days during paper trading. SQLite is for unit tests only.
7. **Kill switch is out-of-band of the LLM.** A separate APScheduler job polls the Freqtrade REST API every 5 minutes and calls `/api/v1/stop` directly when global drawdown ≥ 12% or consecutive losses ≥ 10. It does NOT wait for the graph to wake.

---

## 2. System overview

```
                              ┌────────────────────────┐
                              │  Supervisor agent      │
                              │  (cron + event driven) │
                              └─────────┬──────────────┘
                                        │  spawn_strategy
                                        ▼
        ┌───────────────────────────────────────────────────────┐
        │  Per-strategy thread  (thread_id = strategy_id)        │
        │                                                        │
        │   research ──▶ validation ──▶ paper_gate (HITL)        │
        │                                  │                     │
        │                                  ▼                     │
        │                              paper ──▶ live_gate (HITL)│
        │                                              │         │
        │                                              ▼         │
        │                                            live        │
        └───────────────────────────────────────────────────────┘
                                        │
                  ┌─────────────────────┼─────────────────────┐
                  ▼                     ▼                     ▼
       ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
       │ Freqtrade pool  │   │ Postgres 15     │   │ Redis 7         │
       │ (1 per strategy)│   │ checkpointer +  │   │ pubsub + APS    │
       └─────────────────┘   │ store + app DB  │   │ jobstore        │
                             └─────────────────┘   └─────────────────┘
```

**What Freqtrade does:** runs strategies, downloads OHLCV, runs backtests, trains FreqAI models, places exchange orders, exposes a REST API for the orchestrator.

**What FreqAI does (optional per strategy):** when a strategy template enables it, FreqAI trains a classifier, regressor, or RL agent per pair, on features the template defines, and exposes prediction columns the template's entry/exit logic reads. **FreqAI is a smart indicator, not a strategy generator.**

**What LangGraph does:** owns the strategy lifecycle as a stateful graph. Runs ReAct agents for ideation/criticism/monitoring. Owns Send-based parallelism for backtests and robustness tests. Owns HITL via `interrupt()`. Owns long-term memory of past failures/wins via PostgresStore.

---

## 3. Prerequisites — operator must have before Stage 0

The operator (human) is responsible for:

- A Linux machine or VPS (Ubuntu 22.04+ recommended), **8+ GB RAM, 4+ CPU cores, 100 GB disk**. Local dev on macOS is fine for stages 0–5; Linux strongly preferred from Stage 6 onward.
- **Docker Engine 24+** and **Docker Compose v2** installed.
- **Python 3.11, 3.12, or 3.13** installed (3.12 recommended).
- **Git** and a private GitHub repo to push to.
- **Anthropic API key** with billing enabled. Budget: ≈$15–25 per strategy lifecycle in LLM calls (see §17).
- **One exchange account** (Binance, Bybit, Kraken, or OKX) with:
  - The ability to create *subaccounts* and *API keys with withdrawals disabled*.
  - $500 USDT (or equivalent) on the live subaccount when ready for live stage.
- **A WireGuard or SSH tunnel** for remote dashboard access. (No public ports.)
- **An off-box backup target** (Backblaze B2, S3, or similar) for `pg_dump`.

Claude Code is responsible for everything else: installing Freqtrade, FreqAI, LangGraph, all Python deps, all Docker images, all migrations.

---

## 4. Tech stack — pin these exact versions

| Component | Version | Notes |
|---|---|---|
| Python | 3.12.x | also OK: 3.11, 3.13. Pin in `pyproject.toml`. |
| Freqtrade | **2026.4** | latest stable; monthly release cadence. |
| FreqAI | bundled in Freqtrade 2026.4 | install via the `freqtradeorg/freqtrade:stable_freqai` Docker image (or `:stable_freqairl` for RL). |
| LangGraph | **1.2.x** (≥1.2.0) | functional API GA; `interrupt()` dynamic form is the recommended HITL primitive. |
| langgraph-checkpoint-postgres | **3.1.x** | required for the PostgresSaver **and** the PostgresStore — `AsyncPostgresStore` (and its sync counterpart) ships in `langgraph.store.postgres[.aio]` inside this package. There is no separate `langgraph-store-postgres` distribution on PyPI (verified against the official LangChain langgraph docs install snippet). |
| LangChain | **1.3.x** (≥1.3.1) | use `langchain.agents.create_agent` — NOT the deprecated `langgraph.prebuilt.create_react_agent`. |
| langchain-anthropic | **1.3.3+** | `ChatAnthropic` integration. |
| Anthropic Claude models | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` | Opus for ideation/critic/risk; Sonnet for researcher/coordinator; Haiku for routine paper monitor. |
| Postgres | 15.x | single cluster, three logical DBs (see §5.6). Enable `pgvector` extension. |
| Redis | 7.x | APScheduler jobstore + pubsub. |
| APScheduler | 3.10+ | use `SQLAlchemyJobStore` against the app DB so wakes survive restart. |
| FastAPI | 0.115+ | resume endpoint, dashboard backend, websocket events. |
| Streamlit | 1.40+ | HITL UI. |
| Docker Compose | v2 | orchestration of all services. |

**Forbidden:**
- `langgraph.prebuilt.create_react_agent` — deprecated in LangGraph 1.0.
- Static `compile(interrupt_before=[...])` — use dynamic `interrupt()` inside the gate node.
- Running multiple Freqtrade backtest workers against a shared `user_data/` directory — each worker MUST get its own `--userdir` and pass `--cache none`.

---

## 5. Architecture

### 5.1 Top-level

The orchestrator runs **two graph kinds**:

1. **Supervisor graph** — one thread, `thread_id="supervisor"`. ReAct agent (Sonnet 4.6) with portfolio-level tools. Runs on APScheduler cron + event triggers.
2. **Per-strategy graph** — one thread per strategy, `thread_id="strategy_<uuid>"`. Composes four subgraphs in sequence.

Both graphs share a single `PostgresSaver` checkpointer and a single `PostgresStore`.

### 5.2 Per-strategy thread — four subgraphs

| Subgraph | Purpose | Key primitives |
|---|---|---|
| **Research** | propose hypothesis, fill template, critic loop | `create_agent`, structured output, bounded reflection (≤3) |
| **Validation** | parallel backtests + robustness + risk verdict | `Send` fan-out × 2, reducer, `Command(goto, update)` |
| **Paper** | spawn dry-run, 30-day wake-cycle monitoring | `interrupt()`, APScheduler wake job, ReAct monitor |
| **Live** | spawn live, parallel multi-agent review, kill switch | `Send` fan-out, coordinator agent, **out-of-band kill switch** |

### 5.3 Research subgraph — nodes

| Node | Primitive | Role |
|---|---|---|
| `load_context` | plain | pulls `("failures", regime)` and `("wins", regime)` from Store. |
| `researcher` | `create_agent` (Sonnet 4.6) + tools `query_store`, `get_market_regime`, `read_template`, `get_pair_stats` | proposes hypothesis, candidate template, parameter ranges. |
| `generator` | plain (deterministic) | renders the proposal into the chosen template using Pydantic-validated params; runs AST validation (no `import os`, `subprocess`, network, `eval`, `exec`); writes `.py` to `strategy_templates/_generated/<strategy_id>.py`. |
| `critic` | `create_agent` (Opus 4.7) + tool `read_template` | adversarial review: "find the look-ahead bias", "find the indicator reading future data", "find the position sizing compounding losses". Appends to `agent_votes`. |
| `revise_or_proceed` | `Command(goto)` router | if vote="revise" and `revision_count < 3` → goto `generator`. If exhausted → goto `archive` with `failure_reason="critic_loop_exhausted"`. If vote="pass" → goto `lookahead_gate`. |
| `lookahead_gate` | plain | runs `freqtrade lookahead-analysis`. On failure → `archive` with `failure_reason="lookahead_bias"`. On pass → validation entry. |

### 5.4 Validation subgraph — nodes

| Node | Primitive | Role |
|---|---|---|
| `prepare_data` | plain | runs `freqtrade download-data` incrementally for the strategy's pairs + timeframes. Data dir is shared read-only across workers. |
| `plan_backtests` | router returns `list[Send]` | one `Send("backtest_worker", {...})` per `(param_set × pair_group × walk_forward_fold)`. Uses **anchored 6-fold walk-forward** (4 months train / 1 month test, sliding by 1 month). |
| `backtest_worker` | plain | creates isolated `--userdir user_data/_workers/<wid>` (symlink-tree to shared OHLCV), runs `freqtrade backtesting --cache none --timerange <fold>`, parses zip into a `BacktestResult`. Returns `{"backtest_results": [BacktestResult]}` — reducer concatenates. |
| `aggregate_results` | plain | per param set: IS Sharpe, OOS Sharpe, OOS/IS ratio, profit factor, max DD, trade count. Writes `gate_decisions["backtest"]`. |
| `gate_backtest` | `Command(goto)` | if any hard threshold fails → `archive`. Else → `plan_robustness`. |
| `plan_robustness` | router returns `list[Send]` | three Sends in parallel: `monte_carlo_worker`, `regime_worker`, `fee_stress_worker`. |
| `monte_carlo_worker` | plain | **trade-level bootstrap**: resample realized trades with replacement, 1000 iterations, 5th-percentile final equity. |
| `regime_worker` | plain | slices the period into 3 vol regimes (low/mid/high BTC realized vol); reports per-regime Sharpe. |
| `fee_stress_worker` | plain | re-runs best param set with `--fee 0.002` (2× exchange default) and `--fee 0.003` (3×); reports % Sharpe degradation. |
| `risk_analyst` | `create_agent` (Opus 4.7) + tool `read_robustness_summary` | reads aggregated robustness; returns `Command(goto="paper_gate", update={...})` or `Command(goto="archive", update={"failure_reason":...})`. |
| `paper_gate` | dynamic `interrupt({"kind":"paper_gate","summary":...})` | pauses. Resumes with `Command(resume={"approved": bool, "notes": str})`. On approve: `stage="paper"` → paper subgraph. |

### 5.5 Paper subgraph — nodes

| Node | Primitive | Role |
|---|---|---|
| `paper_spawn` | plain | picks next free port, writes `config-paper.json` with `dry_run: true` and `dry_run_wallet`, spawns Freqtrade dry-run as a Docker Compose service (one container per strategy), waits for `/api/v1/ping` 200. Updates `freqtrade_api_url`, `freqtrade_process_id`. |
| `schedule_wake` | plain | registers APScheduler interval job (every 6h) calling `POST /threads/{tid}/wake`. |
| `paper_wait` | `interrupt()` | parks until wake or kill event. |
| `paper_monitor` | `create_agent` (Haiku 4.5) + tools `ft_status`, `ft_profit`, `ft_trades`, `ft_performance`, `compare_to_backtest`, `check_kill_switch` | on wake: pulls live paper metrics, KS-tests per-trade returns vs backtest, decides: re-arm (goto `paper_wait`), advance (goto `live_gate`), or kill (goto `archive`). |
| `divergence_check` | `Command(goto)` | belt-and-braces threshold check downstream of monitor. |
| `live_gate` | dynamic `interrupt()` | second HITL gate. |
| `paper_teardown` | plain | `POST /api/v1/stop` to dry-run instance; remove container. |

### 5.6 Live subgraph — nodes

| Node | Primitive | Role |
|---|---|---|
| `live_spawn` | plain | new userdir, `config-live.json` with **separate live API keys** from secrets store, `dry_run: false`, `stake_amount` capped to `LIVE_CAPITAL_CAP_USD`. Spawns container. |
| `live_wait` | `interrupt()` | parks between wake-cycles. |
| `live_evaluate` | router returns `list[Send]` | three Sends: `risk_check`, `performance_check`, `regime_check`. |
| `risk_check` | plain + small LLM call | current drawdown, daily P&L, consecutive losses → vote. |
| `performance_check` | `create_agent` (Sonnet 4.6) | live vs paper distribution drift → vote. |
| `regime_check` | plain | compares current regime to approval-time regime → vote. |
| `coordinator` | `create_agent` (Sonnet 4.6; escalate to Opus 4.7 on vote disagreement) | merges votes → `Command(goto="live_wait")` (continue), `live_pause`, or `archive`. |
| `live_pause` | plain | calls `POST /api/v1/stop`, then `interrupt({"kind":"live_pause_review"})` for human review. |
| **Out-of-band**: `kill_switch_handler` | APScheduler job, NOT a node | polls `/api/v1/profit` every 5 min for every live thread; if dd ≥ 12% or losses ≥ 10 → `POST /api/v1/stop` directly + publishes `redis:kill:<sid>`. The orchestrator routes the thread to `live_pause` on next wake. |

### 5.7 State schema

```python
# orchestrator/state.py
from typing import TypedDict, Annotated, Literal, Optional
from operator import add

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
    kind: Literal["monte_carlo","regime","fee_stress","walk_forward"]
    payload: dict

class AgentVote(TypedDict):
    agent: str
    verdict: Literal["pass","fail","revise","pause","continue"]
    rationale: str
    confidence: float

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
    stage: Literal["research","validation","paper_gate","paper",
                   "live_gate","live","archived"]

    # Reducers — Send workers append here
    backtest_results:     Annotated[list[BacktestResult],   add]
    robustness_results:   Annotated[list[RobustnessResult], add]
    agent_votes:          Annotated[list[AgentVote],        add]

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
```

### 5.8 Postgres schema

Three logical DBs in one cluster:

1. **`langgraph_checkpoints`** — managed by `PostgresSaver`; created by `checkpointer.setup()`.
2. **`langgraph_store`** — managed by `PostgresStore`; created by `store.setup()`. Use `pgvector` for semantic search namespace.
3. **`app`** — owned by us. Schema (Alembic migrations):

```sql
CREATE TABLE strategy_registry (
    strategy_id TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    template    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    pairs       JSONB NOT NULL,
    timeframe   TEXT NOT NULL,
    freqtrade_userdir   TEXT,
    freqtrade_api_url   TEXT,
    freqtrade_pid       INT,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
    failure_reason TEXT
);
CREATE INDEX ON strategy_registry(stage);

CREATE TABLE gate_audits (
    id BIGSERIAL PRIMARY KEY,
    strategy_id TEXT REFERENCES strategy_registry(strategy_id),
    gate        TEXT NOT NULL CHECK (gate IN ('backtest','paper','live','live_pause')),
    decision    TEXT NOT NULL CHECK (decision IN ('auto_pass','auto_fail','human_approve','human_reject','human_revise')),
    actor       TEXT NOT NULL,
    payload     JSONB NOT NULL,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON gate_audits(strategy_id, at DESC);

CREATE TABLE telemetry (
    id BIGSERIAL PRIMARY KEY,
    strategy_id TEXT REFERENCES strategy_registry(strategy_id),
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage       TEXT,
    metrics     JSONB NOT NULL,
    source      TEXT NOT NULL
);
CREATE INDEX ON telemetry(strategy_id, snapshot_at DESC);

CREATE TABLE kill_switch_events (
    id BIGSERIAL PRIMARY KEY,
    strategy_id TEXT REFERENCES strategy_registry(strategy_id),
    fired_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason      TEXT NOT NULL,
    metrics     JSONB NOT NULL,
    action_taken TEXT NOT NULL
);

CREATE TABLE regime_log (
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    regime      TEXT NOT NULL,
    features    JSONB NOT NULL,
    detector    TEXT NOT NULL,
    PRIMARY KEY (at, detector)
);
```

### 5.9 Long-term Store namespaces

| Namespace | Key | Value | Writer | Reader |
|---|---|---|---|---|
| `("failures", regime)` | `<strategy_id>` | `{hypothesis, params, failure_reason, metrics}` | any `archive` node | `researcher` |
| `("wins", regime)` | `<strategy_id>` | `{hypothesis, params, live_metrics_summary}` | post-live archive when metrics positive | `researcher` |
| `("postmortems", strategy_id)` | uuid | `{written_by_agent, narrative, root_cause}` | dedicated post-mortem node | `researcher`, `critic` |
| `("regime_log",)` | ISO timestamp | `{regime, features}` | regime APScheduler job + `regime_worker` | `supervisor`, `regime_check`, `researcher` |

---

## 6. LangGraph v1 patterns — MANDATORY

This project uses LangGraph 1.x and LangChain 1.x. The following patterns are required; the deprecated ones are forbidden.

### 6.1 ReAct agents — use `create_agent`

```python
# CORRECT (v1):
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_market_regime() -> dict:
    """Return current market regime features."""
    ...

researcher = create_agent(
    model=ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[get_market_regime, query_store, read_template, get_pair_stats],
    prompt="You propose trading hypotheses grounded in regime and past failures.",
)

# FORBIDDEN (deprecated):
# from langgraph.prebuilt import create_react_agent
```

### 6.2 HITL — use dynamic `interrupt()`

```python
# CORRECT (v1):
from langgraph.types import interrupt, Command

def paper_gate(state: StrategyState) -> dict:
    decision = interrupt({
        "kind": "paper_gate",
        "strategy_id": state["strategy_id"],
        "summary": {
            "sharpe_is": state["gate_decisions"]["backtest"]["sharpe_is"],
            "sharpe_oos_ratio": state["gate_decisions"]["backtest"]["oos_ratio"],
            "max_dd": state["gate_decisions"]["backtest"]["max_dd"],
            "robustness": state["gate_decisions"].get("robustness"),
        },
    })
    if not decision["approved"]:
        return {"stage": "archived", "failure_reason": f"paper_gate_rejected: {decision.get('notes','')}"}
    return {"stage": "paper", "gate_decisions": {**state["gate_decisions"],
            "paper": {"approved": True, "notes": decision.get("notes",""), "by": "human"}}}

# Resume from FastAPI:
async for ev in graph.astream(
    Command(resume={"approved": True, "notes": "..."}),
    config={"configurable": {"thread_id": strategy_id}},
):
    ...

# FORBIDDEN:
# graph.compile(checkpointer=cp, interrupt_before=["paper_gate"])  # static form is legacy
```

**Important property of `interrupt()`:** when resumed, the node *replays from its start*. Keep gate nodes side-effect-free; do the spawn in the *next* node.

### 6.3 Parallel fan-out — `Send` + reducer

```python
from typing import Annotated
from operator import add
from langgraph.types import Send

class State(TypedDict):
    backtest_results: Annotated[list[BacktestResult], add]  # REDUCER REQUIRED

def plan_backtests(state) -> list[Send]:
    sends = []
    for ps in state["param_sets"]:
        for fold in state["folds"]:
            sends.append(Send("backtest_worker", {**state, "_ps": ps, "_fold": fold}))
    return sends

def backtest_worker(state) -> dict:
    result = run_one_backtest(state["_ps"], state["_fold"])
    return {"backtest_results": [result]}  # reducer appends
```

Without `Annotated[..., add]`, parallel writes overwrite each other.

### 6.4 Dynamic routing — `Command(goto, update)`

```python
from langgraph.types import Command
from typing import Literal

def risk_analyst(state: StrategyState) -> Command[Literal["paper_gate","archive"]]:
    score = compute_risk_score(state["robustness_results"])
    if score >= 0.7:
        return Command(goto="paper_gate", update={
            "gate_decisions": {**state["gate_decisions"], "robustness": {"score": score, "passed": True}}
        })
    return Command(goto="archive", update={
        "stage": "archived",
        "failure_reason": f"robustness_score={score:.2f}_below_threshold",
    })
```

### 6.5 Compile with checkpointer + store

```python
# CORRECT: lifecycle owned by FastAPI lifespan
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

@asynccontextmanager
async def lifespan(app):
    async with AsyncPostgresSaver.from_conn_string(PG_URI) as saver, \
               AsyncPostgresStore.from_conn_string(PG_URI) as store:
        await saver.setup()
        await store.setup()
        app.state.saver = saver
        app.state.store = store
        app.state.graph = build_per_strategy_graph(saver, store)
        yield
```

### 6.6 Security: enforce strict deserialization

In `.env`:
```
LANGGRAPH_STRICT_MSGPACK=true
```

This prevents code execution from a compromised checkpoint database.

---

## 7. Freqtrade integration rules

### 7.1 Worker isolation

**Every Freqtrade invocation (backtest, paper, live) MUST have its own `--userdir`.** Never share. The shared OHLCV data directory may be mounted read-only into each worker's userdir via symlink, but `backtest_results/`, `models/`, `logs/` are per-worker.

```bash
# CORRECT:
freqtrade backtesting \
  --userdir freqtrade/user_data/_workers/<worker_id> \
  --strategy GenStrategy_<strategy_id> \
  --timerange 20240101-20240401 \
  --cache none

# FORBIDDEN: shared userdir across parallel backtests; reusing cached results.
```

### 7.2 REST API client

`orchestrator/tools/freqtrade_api.py` is a thin httpx client over Freqtrade's REST API. Endpoints used:

- `GET  /api/v1/ping` — health
- `GET  /api/v1/status` — open trades
- `GET  /api/v1/profit` — cumulative P&L + drawdown
- `GET  /api/v1/trades?limit=N` — recent trades
- `GET  /api/v1/performance` — per-pair stats
- `POST /api/v1/stopbuy` — graceful (stop new entries, let opens run)
- `POST /api/v1/stop` — full stop
- JWT auth from `config-*.json`; passwords from secrets, never in code.

### 7.3 FreqAI config pins

These FreqAI parameters MUST be set explicitly in every FreqAI strategy's `freqai_config`:

```python
{
  "train_period_days": 30,         # required; no default
  "backtest_period_days": 7,       # required; no default
  "live_retrain_hours": 24,        # set to 24-48; default 0 = retrain constantly
  "expiration_hours": 72,          # set to 72; default 0 = never expire (stale models trade)
  "purge_old_models": 2,
  "feature_parameters": {
    "DI_threshold": 0.9,           # activate; default 0 = off
    "use_SVM_to_remove_outliers": True,  # OR use_DBSCAN_to_remove_outliers: True
  },
}
```

### 7.4 Worker placement: subprocess vs Docker

- **Backtests** — short-lived (minutes), ephemeral userdirs → subprocess driver (`asyncio.create_subprocess_exec`).
- **Paper and live** — long-lived (days to weeks) → one Docker Compose service per strategy. Resource caps via `mem_limit` and `cpus` in compose.

### 7.5 Required CLI flags

- Backtests: `--cache none` always. Cache reuse with a freshly generated strategy is a stale-result hazard.
- Downloads: incremental by default (`download-data` is already incremental in 2026.4); pass `--timerange` to scope.

---

## 8. Strategy template authoring contract

Every file in `strategy_templates/` MUST conform to this contract:

1. **Structural shell is hand-written and untouchable.** Class name, `populate_indicators`, `populate_entry_trend`, `populate_exit_trend`, `stoploss`, `timeframe`, `process_only_new_candles`, FreqAI `feature_engineering_*` / `set_freqai_targets` (if applicable). The LLM never edits these.
2. **Slots marked explicitly.** `# SLOT: <name> (type, range)` inline comments mark every variable the LLM may fill. Example:
   ```python
   # SLOT: rsi_buy_threshold (int, 10-40)
   # SLOT: ema_fast (int, 5-50)
   # SLOT: label_period_candles (int, 4-24)  # FreqAI templates only
   ```
3. **Pydantic schema co-located.** For `freqai_classifier_template.py` add `freqai_classifier_template_schema.py`:
   ```python
   from pydantic import BaseModel, Field
   class FreqaiClassifierParams(BaseModel):
       rsi_buy_threshold: int = Field(ge=10, le=40)
       ema_fast: int = Field(ge=5, le=50)
       label_period_candles: int = Field(ge=4, le=24)
       label_threshold_pct: float = Field(ge=0.1, le=2.0)
       min_class_prob: float = Field(ge=0.55, le=0.85)
       # ...
   ```
   The generator calls `ChatAnthropic(...).with_structured_output(FreqaiClassifierParams)`. Free-form params are impossible.
4. **AST validation.** Generator parses the rendered file with `ast.parse`, walks the tree, rejects `import os`, `import subprocess`, network modules, `eval`, `exec`, `__import__`, `compile`. Reject also if any imported module is not in an allowlist.
5. **Lookahead test.** Every generated strategy is run through `freqtrade lookahead-analysis` before backtest. Failures route to `archive` with `failure_reason="lookahead_bias"`.
6. **Smoke test fixture.** Every template ships `tests/test_<template>_smoke.py` that backtests it with default params on 1 month of cached BTC/USDT data and asserts `trades > 0`. CI runs these.
7. **README** in `strategy_templates/<template_name>_README.md` describing the strategy hypothesis (what market belief it encodes). The critic uses this to argue against it.

### 8.1 v1 templates to ship in Stage 3

- `mean_reversion_template.py` — pure TA. RSI + Bollinger Bands. No FreqAI. Baseline.
- `freqai_classifier_template.py` — LightGBM classifier predicts {up, flat, down}. Entry on `up` with `prob >= min_class_prob`. DI threshold + SVM outlier rejection.
- `freqai_regressor_template.py` — LightGBM regressor predicts return over `label_period_candles`. Entry when `predicted_return > k * ATR`.

Optional later (v2): `freqai_rl_template.py` using `BaseReinforcementLearner` + stable_baselines3. Out of scope for v1.

---

## 9. Repository structure

```
ai-trading-agent/
├── BRD.md                          # this file — SOURCE OF TRUTH
├── README.md
├── SPEC.md                         # Stage-0 output: operator-specific choices
├── .env.example
├── .gitignore                      # excludes secrets/, freqtrade/user_data/data/
├── pyproject.toml                  # pinned deps
├── alembic.ini
├── docker-compose.yml              # postgres + redis + orchestrator + dashboard
├── docker-compose.freqtrade.yml    # per-strategy worker services (templated)
│
├── orchestrator/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app + lifespan
│   ├── graph.py                    # parent per-strategy graph
│   ├── supervisor.py               # supervisor ReAct agent + tools
│   ├── state.py                    # StrategyState TypedDict
│   ├── scheduler.py                # APScheduler setup (wakes, regime job, kill switch)
│   │
│   ├── subgraphs/
│   │   ├── __init__.py
│   │   ├── research.py
│   │   ├── validation.py
│   │   ├── paper.py
│   │   └── live.py
│   │
│   ├── agents/
│   │   ├── researcher.py
│   │   ├── critic.py
│   │   ├── risk_analyst.py
│   │   ├── monitors.py             # paper + live monitor agents
│   │   ├── coordinator.py
│   │   └── prompts/                # system prompts as .md
│   │
│   ├── tools/
│   │   ├── freqtrade_api.py        # httpx REST client
│   │   ├── backtest_runner.py      # subprocess driver
│   │   ├── regime.py               # vol+trend bucketing (+HMM optional)
│   │   ├── store_queries.py        # typed wrappers over PostgresStore
│   │   └── compare.py              # paper-vs-backtest KS test
│   │
│   ├── gates/
│   │   ├── thresholds.py           # ALL gate thresholds in ONE file (see §10)
│   │   └── hitl.py                 # interrupt + resume helpers
│   │
│   └── security/
│       ├── secrets.py              # secret loading (env/sops/1password)
│       └── ast_validator.py        # AST allowlist for generated strategies
│
├── strategy_templates/
│   ├── README.md                   # contract from §8
│   ├── mean_reversion_template.py
│   ├── mean_reversion_template_schema.py
│   ├── mean_reversion_template_README.md
│   ├── freqai_classifier_template.py
│   ├── freqai_classifier_template_schema.py
│   ├── freqai_classifier_template_README.md
│   ├── freqai_regressor_template.py
│   ├── freqai_regressor_template_schema.py
│   ├── freqai_regressor_template_README.md
│   └── _generated/                 # LLM-rendered strategies land here
│
├── freqtrade/
│   ├── user_data/
│   │   ├── data/                   # SHARED OHLCV (read-only mount into workers)
│   │   ├── strategies/             # symlinked from _generated/
│   │   ├── _workers/               # ephemeral per-worker userdirs
│   │   └── configs/                # base config templates
│   └── README.md
│
├── dashboard/
│   ├── app.py                      # Streamlit UI
│   └── components/                 # cards, charts, tables
│
├── db/
│   ├── migrations/                 # alembic
│   └── schema.sql                  # reference dump
│
├── tests/
│   ├── unit/
│   │   ├── test_thresholds.py
│   │   ├── test_template_filling.py
│   │   ├── test_ast_validator.py
│   │   ├── test_send_fanout.py
│   │   ├── test_critic_loop.py
│   │   ├── test_kill_switch.py
│   │   └── test_graph_routing.py
│   ├── integration/
│   │   ├── test_research_to_validation.py
│   │   ├── test_hitl_resume.py
│   │   ├── test_freqtrade_subprocess.py
│   │   └── test_postgres_lifecycle.py
│   └── fixtures/
│       └── btc_usdt_5m_1week.feather
│
├── ops/
│   ├── backup.sh                   # pg_dump to off-box
│   ├── restore.sh
│   └── reconcile.py                # on-startup attach/cleanup
│
└── .github/
    └── workflows/
        └── ci.yml                  # lint + unit + integration smoke
```

---

## 10. Gate thresholds — final values

All thresholds live in `orchestrator/gates/thresholds.py`. **Do not put thresholds anywhere else.**

```python
# orchestrator/gates/thresholds.py

# Backtest hard gate (in-sample, anchored 6-fold walk-forward)
MIN_TRADES_IS = 150
MIN_OOS_TRADES = 30
MIN_SHARPE_IS = 1.5
MIN_PROFIT_FACTOR_IS = 1.5
MAX_DRAWDOWN_IS = 0.20

# OOS / walk-forward gate
MIN_OOS_RATIO = 0.6                 # mean OOS Sharpe / IS Sharpe
MIN_OOS_SHARPE_PER_FOLD = 0.0       # no fold may lose money
MIN_OOS_PROFIT_FACTOR = 1.2
MAX_OOS_DRAWDOWN = 0.25

# Robustness gate
MIN_MC_5TH_PERCENTILE_RETURN = 0.0  # 5th-pct bootstrap final equity must be positive
MIN_REGIMES_PASSED = 2              # of 3 (low/mid/high vol)
MAX_FEE_STRESS_DEGRADATION_2X = 0.40
MAX_FEE_STRESS_DEGRADATION_3X = 0.60

# Paper gate (advisory — human decides)
MIN_PAPER_DAYS = 30
MAX_PAPER_VS_BACKTEST_KS_PVALUE = 0.05  # KS test on per-trade returns
MAX_PAPER_VS_BACKTEST_SHARPE_DEVIATION = 0.30

# Live monitoring — AUTO PAUSE, not advisory
KILL_SWITCH_DRAWDOWN = 0.12
KILL_SWITCH_CONSECUTIVE_LOSSES = 10
DAILY_LOSS_LIMIT_PCT = 0.03         # -3% in any rolling 24h → stopbuy (graceful)
MAX_OPEN_TRADES = 4
MAX_POSITION_CONCENTRATION = 0.30   # one pair ≤ 30% of equity

# Live capital
LIVE_CAPITAL_CAP_USD = 500
```

**These values are operator-tunable in `SPEC.md` during Stage 0.** Defaults above stand otherwise. Re-tune after the first 10 strategies have completed a lifecycle.

---

## 11. Kill switch — layered safeguards

The single 8% drawdown trigger in the original brief is too tight for $500 capital. Replace with the following layered design, all implemented in `orchestrator/scheduler.py` + Freqtrade config:

| Layer | Threshold | Action | Implemented by |
|---|---|---|---|
| Per-trade stop | strategy-defined `stoploss` | Freqtrade exits position | Freqtrade strategy code |
| Daily loss limit | -3% rolling 24h | `POST /api/v1/stopbuy` (graceful) | APScheduler job, every 15 min |
| Position concentration | one pair > 30% equity | reject new entry | Freqtrade `protection` |
| Max concurrent positions | 4 | `max_open_trades` | Freqtrade config |
| Global drawdown | 12% from running peak | `POST /api/v1/stop` (full) | APScheduler kill-switch job, every 5 min |
| Consecutive losses | 10 | `POST /api/v1/stop` (full) | same job |
| LangGraph response | on any of the above | thread → `live_pause` (HITL review) | Redis pubsub → orchestrator |

The kill-switch job operates *independently of the graph*. It MUST run even if the orchestrator is down.

---

## 12. Cost budget

Per strategy lifecycle (research → live for 90 days):

| Phase | Model | Cost (USD) |
|---|---|---|
| Researcher (3 turns) | Sonnet 4.6 | ~$0.20 |
| Generator (structured output) | Sonnet 4.6 | ~$0.04 |
| Critic (≤3 iterations) | Opus 4.7 | ~$0.70 |
| Risk analyst | Opus 4.7 | ~$0.17 |
| Paper monitor (120 wakes) | Haiku 4.5 | ~$0.96 |
| Live coordinator + 3 reviewers (360 cycles) | Sonnet/Haiku mix | ~$10–$20 |
| Supervisor share | Sonnet 4.6 | ~$1.50/week background |
| **Total per strategy lifecycle** | | **~$15–$25** |

At 4–6 strategies per quarter through the full pipeline: **~$100–$200/quarter** in LLM costs.

---

## 13. Build sequence — Stage Table

**Do not skip stages.** Each stage commits to git. Tag with `stage-N-complete` when DoD is met.

| Stage | Goal | DoD |
|---|---|---|
| 0 | Spec + tooling | `SPEC.md` written; `pyproject.toml` pinned; `docker compose up` brings up Postgres + Redis |
| 1 | App skeleton + migrations | FastAPI starts; Postgres has 3 logical DBs; `checkpointer.setup()` + `store.setup()` succeed; Alembic head applied |
| 2 | StrategyState + skeleton graph | minimal graph (research stub → archive stub) round-trips checkpoint rows; thread_id persists |
| 3 | Freqtrade integration tools | subprocess driver runs a 1-week backtest on cached BTC/USDT and parses the zip into a `BacktestResult`; REST client pings a dry-run instance |
| 4 | Validation subgraph | 5 parallel Send workers produce 5 BacktestResults that the reducer concatenates; `gate_backtest` routes to archive on a failing strategy; trade-level bootstrap implemented |
| 5 | Research subgraph | a research run produces a strategy file that passes `freqtrade lookahead-analysis` and contains no disallowed imports; critic loop bounded at 3 |
| 6 | HITL + dashboard + FastAPI | approve/reject from Streamlit advances or archives the thread; `gate_audits` rows written |
| 7 | Paper subgraph + APScheduler | paper instance runs for 24h, gets woken every 6h, monitor compares to backtest and either re-arms or escalates |
| 8 | Live subgraph + kill switch | end-to-end: paper graduates to live_gate; human approves; live spawns; synthetic drawdown triggers kill switch in < 5 min and routes thread to `live_pause` |
| 9 | Supervisor | runs nightly + on every thread completion; spawns up to capacity; logs decisions |
| 10 | Observability + DR | LangSmith on; Prometheus scraping Freqtrade APIs; nightly `pg_dump` to off-box; reconciliation script on orchestrator startup |
| 11 | Hardening | AST validator, structured output, daily loss limit, concentration enforcement, secrets review, port audit |

### Stage 0 — Spec + tooling

**Goal:** lock operator choices; install local deps; bring up infra.

**Install commands (run from a fresh `ai-trading-agent/` directory):**
```bash
# System (operator-side prereqs assumed installed: docker, docker compose, python 3.12, git)
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# Initial deps (more in later stages)
pip install \
  "langgraph==1.2.*" \
  "langgraph-checkpoint-postgres==3.1.*" \
  "langchain==1.3.*" \
  "langchain-anthropic==1.3.*" \
  "fastapi==0.115.*" "uvicorn[standard]" \
  "streamlit==1.40.*" \
  "apscheduler==3.10.*" \
  "httpx==0.27.*" \
  "pydantic==2.9.*" \
  "alembic==1.13.*" "psycopg[binary]==3.2.*" "sqlalchemy==2.0.*" \
  "redis==5.0.*" \
  "python-dotenv==1.0.*"

pip install --group dev \
  "pytest==8.*" "pytest-asyncio==0.24.*" \
  "ruff==0.7.*" "mypy==1.13.*" "pre-commit==4.*"
```

**Files to create:**
- `BRD.md` (this file)
- `SPEC.md` — operator answers: exchange choice, initial pair list, capital cap, threshold overrides
- `.env.example` — `DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `LANGGRAPH_STRICT_MSGPACK=true`, exchange API keys (commented out)
- `.gitignore` — `.env`, `secrets/`, `freqtrade/user_data/data/`, `freqtrade/user_data/_workers/`, `__pycache__/`
- `pyproject.toml` with the pins above
- `docker-compose.yml` — `postgres:15`, `redis:7`, healthchecks, named volumes
- `README.md` — short, points at `BRD.md`
- `.github/workflows/ci.yml` — ruff + mypy + pytest unit

**Tests:** none yet.

**DoD:** `docker compose up -d`; `psql $DATABASE_URL -c '\l'` lists `app`, `langgraph_checkpoints`, `langgraph_store`; `redis-cli ping` returns `PONG`; `git tag stage-0-complete`.

### Stage 1 — App skeleton + migrations

**Goal:** FastAPI app starts with lifespan; Postgres schema applied; LangGraph saver/store initialize.

**Files to create:**
- `orchestrator/__init__.py`
- `orchestrator/main.py` — FastAPI with `lifespan` that opens `AsyncPostgresSaver` and `AsyncPostgresStore`, calls `setup()`, stashes on `app.state`
- `alembic.ini`, `db/migrations/env.py`, `db/migrations/versions/0001_init.py` with the `app` DB schema from §5.8
- `orchestrator/state.py` — `StrategyState` TypedDict + helper dataclasses
- `tests/unit/test_postgres_lifecycle.py`

**DoD:** `uvicorn orchestrator.main:app --reload` starts; `GET /health` returns `{"ok": true}`; `psql` shows tables `strategy_registry`, `gate_audits`, `telemetry`, `kill_switch_events`, `regime_log`; checkpointer + store tables also present.

### Stage 2 — StrategyState + skeleton graph

**Goal:** minimal per-strategy graph with two real nodes (`research_stub`, `archive`) round-tripping the checkpointer.

**Files to create:**
- `orchestrator/graph.py` — `build_per_strategy_graph(saver, store)`; nodes `research_stub` (transitions stage to "archived") and `archive` (writes failure_reason)
- `tests/integration/test_graph_skeleton.py` — invokes the graph, asserts a checkpoint row exists

**DoD:** test passes; thread state persists across two `graph.ainvoke` calls.

### Stage 3 — Freqtrade integration tools

**Goal:** install Freqtrade via Docker; build subprocess driver; build REST client.

**Install commands:**
```bash
# Pull the Freqtrade images (one with FreqAI ML extras)
docker pull freqtradeorg/freqtrade:stable_freqai

# In the project root, create user_data
docker run --rm -v "$(pwd)/freqtrade/user_data:/freqtrade/user_data" \
  freqtradeorg/freqtrade:stable_freqai create-userdir --userdir /freqtrade/user_data

# Download initial OHLCV data for BTC/USDT 5m (operator's chosen exchange)
docker run --rm -v "$(pwd)/freqtrade/user_data:/freqtrade/user_data" \
  freqtradeorg/freqtrade:stable_freqai download-data \
  --exchange binance --pairs BTC/USDT --timeframes 5m 15m 1h --days 730
```

**Files to create:**
- `orchestrator/tools/freqtrade_api.py` — async httpx client with JWT auth
- `orchestrator/tools/backtest_runner.py` — `async def run_backtest(strategy_id, params, timerange, pair_group) -> BacktestResult` using `asyncio.create_subprocess_exec` with isolated `--userdir` and `--cache none`
- `orchestrator/tools/regime.py` — APScheduler-driven vol+trend bucketing, writes to `regime_log`
- `strategy_templates/mean_reversion_template.py` + `_schema.py` + `_README.md` (first template)
- `tests/integration/test_freqtrade_subprocess.py` — runs the mean reversion template on 1 week of cached data, asserts trades > 0

**DoD:** test passes; isolated userdir cleanup happens after run; regime job writes a row.

### Stage 4 — Validation subgraph

**Goal:** Send fan-out backtests + robustness; `risk_analyst` returns `Command`.

**Files to create:**
- `orchestrator/subgraphs/validation.py` — all nodes from §5.4
- `orchestrator/gates/thresholds.py` — values from §10
- `orchestrator/agents/risk_analyst.py` — Opus 4.7 ReAct
- `tests/unit/test_send_fanout.py` — 5-worker fan-out test with reducer
- `tests/integration/test_validation_subgraph.py` — known-good strategy passes; known-bad strategy archives

**DoD:** integration test passes; `backtest_results` has 5 entries after fan-out; gate threshold violations route to archive.

### Stage 5 — Research subgraph

**Goal:** researcher + structured-output generator + adversarial critic + lookahead gate.

**Files to create:**
- `orchestrator/subgraphs/research.py` — all nodes from §5.3
- `orchestrator/agents/researcher.py` (Sonnet 4.6), `critic.py` (Opus 4.7)
- `orchestrator/security/ast_validator.py` — allowlist parser
- `strategy_templates/freqai_classifier_template.py` + schema + README
- `strategy_templates/freqai_regressor_template.py` + schema + README
- `tests/unit/test_ast_validator.py`, `test_template_filling.py`, `test_critic_loop.py`

**DoD:** generated strategy passes `freqtrade lookahead-analysis`; AST validator rejects `import os`; critic loop terminates at ≤3 revisions on stub responses.

### Stage 6 — HITL + dashboard + FastAPI

**Goal:** dynamic `interrupt()` gates resumable from a real UI.

**Install commands:**
```bash
pip install "streamlit-autorefresh==1.0.*"
```

**Files to create:**
- `orchestrator/gates/hitl.py` — `interrupt()` helpers + `resume_thread(thread_id, payload)`
- `orchestrator/main.py` (additions) — `POST /threads/{tid}/approve`, `GET /threads`, `WS /events`
- `dashboard/app.py` — Streamlit page listing threads by stage with approve/reject/notes
- `tests/integration/test_hitl_resume.py` — stream graph to `paper_gate` interrupt, resume with `Command(resume={"approved": True})`, assert advance

**DoD:** approve from Streamlit; `gate_audits` row written; thread state moves to `paper`.

### Stage 7 — Paper subgraph + APScheduler

**Goal:** dry-run Freqtrade per strategy + 6-hour wake-cycle.

**Files to create:**
- `orchestrator/subgraphs/paper.py` — nodes from §5.5
- `orchestrator/agents/monitors.py` — paper monitor (Haiku 4.5)
- `orchestrator/scheduler.py` — APScheduler with `SQLAlchemyJobStore`, wake job, regime job
- `docker-compose.freqtrade.yml` — templated paper service definition
- `tests/integration/test_paper_subgraph.py` — synthetic 1-hour window before trusting 30 days

**DoD:** paper container runs; wake fires; monitor returns metrics; teardown cleans up.

### Stage 8 — Live subgraph + kill switch

**Goal:** live spawn with separate keys + multi-agent review + out-of-band kill switch.

**Files to create:**
- `orchestrator/subgraphs/live.py` — nodes from §5.6
- `orchestrator/agents/coordinator.py` (Sonnet 4.6, Opus on disagreement)
- `orchestrator/scheduler.py` (additions) — kill_switch_job (every 5 min), daily_loss_job (every 15 min)
- `orchestrator/security/secrets.py` — env-based with optional `sops`/`1password` adapters
- `tests/integration/test_kill_switch.py` — simulate 13% drawdown, assert `/stop` called within 5 min and thread routes to `live_pause`

**DoD:** synthetic-drawdown test passes; live container uses separate keys verified by exchange API key id; thread routes to `live_pause` HITL.

### Stage 9 — Supervisor

**Goal:** top-level autonomous orchestrator.

**Files to create:**
- `orchestrator/supervisor.py` — `create_agent` (Sonnet 4.6) + tools `view_portfolio`, `query_store`, `spawn_strategy`, `retire_strategy`, `get_market_regime`
- `orchestrator/scheduler.py` (additions) — supervisor cron + event subscriptions
- `tests/integration/test_supervisor_loop.py`

**DoD:** supervisor runs nightly; spawns within capacity; honors regime-change events.

### Stage 10 — Observability + DR

**Install commands:**
```bash
pip install "langsmith==0.1.*" "prometheus-client==0.21.*" "structlog==24.*"
```

**Files to create:**
- `ops/backup.sh`, `ops/restore.sh` — `pg_dump` to off-box (Backblaze B2 example)
- `ops/reconcile.py` — on-startup scan of `strategy_registry`, attach to running Freqtrade containers or mark `live_pause`
- `orchestrator/main.py` (additions) — Prometheus `/metrics` endpoint
- LangSmith env wiring: `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=ai-trading-agent`

**DoD:** quarterly DR drill plan documented in `ops/DR.md`; backup script tested with restore to a scratch DB.

### Stage 11 — Hardening

**Files to create:**
- `ops/SECURITY.md` — checklist from §15
- audit pass on every port binding (must be `127.0.0.1`)
- audit pass on every secret access (must be via `secrets.py`)
- `tests/integration/test_security_smoke.py`

**DoD:** every checklist item in §15 ticked.

---

## 14. Observability

- **LangSmith** for LLM tracing. Enable in `.env`. Tag traces with `strategy_id` and `stage`.
- **Prometheus** scrapes:
  - `/metrics` on the orchestrator (FastAPI)
  - Each Freqtrade container's `/api/v1/profit`, `/status` via a small exporter sidecar
- **Structured logging** with `structlog`: every node logs `{strategy_id, thread_id, node, event, payload}` as JSON to stdout.
- **Grafana** dashboards (optional in v1; required in Stage 10): per-strategy P&L, drawdown, open trades; per-host CPU/mem; kill-switch fire-count.

---

## 15. Security checklist

- [ ] `LANGGRAPH_STRICT_MSGPACK=true` set in production.
- [ ] All ports bound to `127.0.0.1` (Postgres, Redis, FastAPI, Streamlit, Freqtrade REST).
- [ ] Remote access via WireGuard or SSH tunnel only.
- [ ] Exchange API keys: live key on subaccount, withdrawals disabled at exchange, IP allowlist set, separate keys for paper vs live.
- [ ] Secrets loaded via `secrets.py` (env / sops / 1password); never in `config.json` committed to git.
- [ ] Postgres SCRAM-SHA-256 auth; separate DB user per logical DB.
- [ ] AST validator rejects disallowed imports in every generated strategy.
- [ ] Generator uses `with_structured_output(Schema)`; no free-form code path.
- [ ] `.env`, `secrets/`, `freqtrade/user_data/data/` in `.gitignore`.
- [ ] LangGraph checkpoint serializer set to strict mode.
- [ ] Quarterly key rotation reminder set.

---

## 16. Disaster recovery

**Backups:** nightly `pg_dump --format=custom` of all three logical DBs to off-box storage (B2/S3). WAL archiving if PITR is needed. **Test restore quarterly.**

**Reconciliation on startup:** `ops/reconcile.py` reads `strategy_registry`, pings each `freqtrade_api_url`; if reachable → keep state; if not → transition to `live_pause` and write a `kill_switch_events` row with `reason="orchestrator_restart_no_freqtrade"`.

**What survives a Postgres death:**

| State | Recoverable from | How |
|---|---|---|
| LangGraph threads | Postgres backup | restore + replay from last checkpoint |
| Freqtrade trade history | Freqtrade's own SQLite | always intact in worker volume |
| Open positions | exchange itself | re-attach via Freqtrade |
| OHLCV cache | feather files on disk | unaffected |
| Long-term Store | Postgres backup | restore |

---

## 17. Common failure modes — what to watch for

1. Skipping paper trading because the backtest looked great. The whole gauntlet exists because backtests routinely overstate live by 2–5×. Non-negotiable.
2. Letting the LLM write strategy code instead of filling templates. Subtle look-ahead bias, off-by-one in lookback, division by zero on early candles.
3. Missing reducer on a Send fan-out field. Parallel writes overwrite each other; you debug for hours.
4. Static `interrupt_before` in the compile call. Use the dynamic `interrupt()` form inside the node.
5. Using the deprecated `create_react_agent` from `langgraph.prebuilt`. Use `langchain.agents.create_agent`.
6. Shared `user_data/` across parallel Freqtrade backtests. Cache contamination, stale results.
7. Shared API keys between paper and live. When (not if) a config bug points "paper" at live keys, you find out by watching real orders fill.
8. Auto-approving HITL gates "just for testing". This sentence appears in every blow-up postmortem.
9. Running the Supervisor before the per-strategy graph is stable. Supervisor multiplies bugs.
10. Critic prompt that is too friendly. Use opinionated phrasing: "find the look-ahead bias", not "review this strategy".
11. PostgresSaver/Store context-manager lifecycle. Open them inside FastAPI `lifespan`; do not call `from_conn_string` ad-hoc per request.
12. Kill switch dependent on the graph being awake. Must be an APScheduler job that talks to Freqtrade directly.

---

## 18. Open questions — answer in Stage 0 SPEC.md

1. Which exchange for v1? (Binance / Bybit / Kraken / OKX — pick one.)
2. Initial pair list to research? (Recommend: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT for Binance.)
3. Live capital cap in USD? (Default 500.)
4. Will you self-host LangSmith alternative (Jaeger) or use LangSmith SaaS? (Default: LangSmith.)
5. Where will backups live? (Default: Backblaze B2.)
6. Risk_analyst on Opus or Sonnet by default? (Default: Opus.)
7. Should v1 include Freqtrade hyperopt as a sub-step in validation? (Default: no — defer to v2.)
8. Will every `live_pause` require human review, or may the coordinator auto-resume in some cases? (Default: every `live_pause` is HITL.)

---

## 19. Subagents for `/agents`

Create these via Claude Code's `/agents`:

- **freqtrade-strategy** — loaded with Freqtrade + FreqAI docs. Writes and audits strategy *templates*. Never wires graph code.
- **langgraph-engineer** — loaded with LangGraph v1 + LangChain v1 docs. Writes graph, nodes, subgraphs, Store usage. Never writes strategy logic.
- **test-writer** — writes tests for whatever was just built. Bias: integration tests over unit tests for graph code; unit tests for thresholds, template filling, AST validator, kill switch.
- **freqtrade-ops** — Docker Compose, REST API wiring, port allocation, container lifecycle.

For long-running tests (real Freqtrade backtests over weeks of data), dispatch via the Task tool so the main session stays responsive.

---

## 20. Where to start next session

1. If the repo is empty except for this BRD: this is Stage 0 — answer §18 questions, write `SPEC.md`, commit and push.
2. If `SPEC.md` exists but no `pyproject.toml`: run Stage 0 install commands.
3. Otherwise: `git tag --list 'stage-*-complete'` to find the highest completed stage and start the next one.

When in doubt about a LangGraph primitive, prefer the documented v1 agentic pattern (`Send`, `Command`, `interrupt()`, `create_agent`, subgraphs, Store) over sequential edges. Sequential is a fallback, not the default.

Every stage ends with: `git add -A && git commit -m "stage N: <summary>" && git tag stage-N-complete && git push --tags`.
