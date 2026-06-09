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

> **SUPERSEDED in part by §22 (Stage 13, 2026-06-09).** The "spot only / no futures, no margin, no leverage" constraint is an *operator-revisable* market-scope choice, not one of the §1.1 safety non-negotiables. §22 adds **short-selling capability** in two gated phases: **Phase 1** lets templates emit short signals and runs them through the validation gauntlet in a **backtest-only futures trading mode** (zero live/real-money exposure) to prove whether shorting has edge; **Phase 2** (only if Phase 1 finds edge) adds **live short execution on Binance futures/margin** behind a hard leverage+stop+liquidation risk model (§22.2). Until Phase 2 ships and is operator-approved, the **live and paper execution paths remain long-only spot** and every §1.1 rule still holds. See §22 for the full contract and build checklist.

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

1. **Supervisor graph** — one thread, `thread_id="supervisor"`. ReAct agent (Sonnet 4.6) with portfolio-level tools. Runs on APScheduler cron + event triggers. **v1 statelessness (Stage 9c):** `thread_id="supervisor"` is a stable invoke-config identifier only — v1 wires **no checkpointer**, so the supervisor reasons *stateless-per-run* and the durable decision record is the `telemetry` row (`source="supervisor_decision"`; Fork C), not an accumulating LLM conversation. The literal "one thread" here must not be read as implying persistent memory. Enabling cross-run reflection by wiring the saver is a deliberate v1 deferral (see `DEFERRED.md` D-8); the owner of any such future change MUST concurrently adopt bounded-context management (per-run summarization or sliding-window retention) before the persistent thread accumulates production token cost.
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
    freqtrade_process_id: Optional[str]   # str, not int — Docker Compose handle,
                                          # not an OS PID (BRD §7.4; SPEC 7c)
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
    freqtrade_pid       INT,            -- VESTIGIAL: created by migration 0001,
                                        -- never written by any code path. See note below.
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

> **Note on `strategy_registry.freqtrade_pid` (Stage 9h docs pass).** This column is **vestigial**: migration `0001_init` created it as `INT` (it predates the BRD §7.4 Docker-Compose decision), but **no code path writes or reads it**. The live/paper container handle lives in `StrategyState.freqtrade_process_id` (a `str` — the Compose project name, e.g. `paper-<strategy_id>` / `live-<strategy_id>`; BRD §5.7, SPEC 7c) and in `artifacts.live_container_id`, neither of which maps to this column. The originally-planned "§5.8 `freqtrade_pid INT → freqtrade_process_id TEXT`" rename (SPEC 2026-05-27 7c entry) is intentionally **NOT** applied to the doc here, because renaming the doc without a migration would make this schema diverge from the actual `0001`-created column — the docs would lie. The real reconciliation (a migration that renames + retypes `freqtrade_pid INT` → `freqtrade_process_id TEXT`, or drops the unused column) is deferred to **Stage 11 hardening**; until then the BRD documents the column as it actually exists.

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
# NOTE: the backtest-gate values carry SPEC-recorded operator re-tunes (this
# block tracks thresholds.py; full rationale in the SPEC §6 change-log).
MIN_TRADES_IS = 90                  # re-tuned 150→90 on 2026-06-09 (SPEC §2/§6)
MIN_TRADES_PER_FOLD = 5             # guard zero-trade folds (walk-forward outside cache)
MIN_OOS_TRADES = 30
MIN_SHARPE_IS = 0.5                 # re-tuned 1.5→0.5 on 2026-06-08 (SPEC §2/§6)
MIN_PROFIT_FACTOR_IS = 1.2          # re-tuned 1.5→1.2 on 2026-06-08 (SPEC §2/§6)
MAX_DRAWDOWN_IS = 0.20
MIN_POSITIVE_FOLDS = 4              # cross-fold consistency: ≥4/6 folds positive (added 2026-06-08, SPEC §2/§6)

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

# Supervisor capacity (Stage 9)
MAX_CONCURRENT_STRATEGIES = 4       # supervisor capacity gate (total non-archived); resource-bound on the 8GB/4-core host
MAX_CONCURRENT_LIVE_STRATEGIES = 1  # capital-bound (SPEC §1 Q3 = $500 whole-amount per live spawn)
```

**These values are operator-tunable in `SPEC.md`.** They were re-tuned after the first ~10 strategies completed a lifecycle (the trigger named below): the backtest hard gate now reads `MIN_SHARPE_IS=0.5`, `MIN_PROFIT_FACTOR_IS=1.2`, `MIN_TRADES_IS=90`, plus the new `MIN_POSITIVE_FOLDS=4` cross-fold consistency gate. `thresholds.py` is the runtime source of truth; this block tracks it and the SPEC §2 override ledger + §6 change-log carry the dated rationale. Further re-tunes follow the same path (a dated SPEC §6 entry + this block + `thresholds.py` + `test_thresholds.py` EXPECTED in lockstep).

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
| 12 | Manual validation, direct FreqAI, operator UI | see **§21** for the full spec. A hand-supplied strategy (template + explicit params + pairs/timeframe) runs the existing validation gauntlet with **zero researcher/generator/critic LLM calls**; a FreqAI template completes train → backtest → gate (or surfaces a clear FreqAI setup error); the operator can drive every action in the OPERATOR_RUNBOOK from a web UI instead of PowerShell/curl. Decouples strategy CREATION from VALIDATION. |
| 13 | Short-selling capability | see **§22** for the full spec (SPEC-only as of 2026-06-09; supersedes the §1 long-only-spot scope). **Phase 1:** templates emit short signals (`can_short=True`, `enter_short`/`exit_short`) and the validation gauntlet evaluates them in a **backtest-only futures trading mode** against the SAME BRD §10 gates — zero live/real-money exposure. **Phase 2 (only if Phase 1 proves edge):** live short execution on Binance futures/margin behind a hard leverage+stop+isolated-margin+liquidation risk model (§22.2). |

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

### Stage 12 — Manual validation, direct FreqAI, and operator UI

**Goal:** decouple strategy CREATION from VALIDATION (operator finding: the
LLM template-fill path produces only losing strategies and burns the 30k-tok/min
Anthropic tier relearning that, while the validation gauntlet is the system's
real value). Add an operator UI over the existing API. **Full spec in §21.**
This stage is SPEC-first per SPEC §4.4 rule 3 — the §21 text lands as a docs
commit before any implementation.

**DoD:** the three §21 acceptance-criteria blocks (Feature 1 manual injection,
Feature 2 direct FreqAI spawn, Feature 3 operator UI) are met.

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

---

## 21. Stage 12 — Manual validation, direct FreqAI spawn, and operator UI

> **Status: SPEC ONLY.** This section is the contract we build Stage 12 from.
> It introduces NO code. Per SPEC §4.4 rule 3 it lands as a standalone docs
> commit; implementation lands in later commits, each gated on this spec.
>
> **Numbering note:** the operator brief suggested "§19", but §19 (Subagents)
> and §20 (Where to start) already exist — this section is §21 to avoid
> renumbering and breaking cross-references.

### 21.0 Motivation (operator findings from the first real runs)

The build is complete and tagged (`stage-0`…`stage-11-complete`); running it
surfaced a clear split (recorded in `OPERATOR_RUNBOOK.md` §6):

- The **validation gauntlet works perfectly** — 6-fold walk-forward + the
  backtest/robustness gates correctly reject bad strategies. That is the
  system's real value.
- The **LLM creation path is the bottleneck**: ~6 strategies spawned, **0
  passed**; the completed `mean_reversion` runs all had badly negative Sharpe,
  and **both FreqAI runs never completed** (died on the 30k-tok/min Anthropic
  429, never reaching a backtest). Re-running the researcher→generator→critic
  chain re-spends tokens to relearn the same losing lesson.

Stage 12 therefore **decouples strategy CREATION from VALIDATION**: let the
operator feed a strategy straight into the proven gauntlet (Feature 1), force a
specific template so FreqAI can finally be exercised (Feature 2), and drive it
all from a UI instead of PowerShell/curl (Feature 3). None of this changes the
gauntlet, the gates (BRD §10), or the HITL contract (§1.1 rule 3) — only how a
strategy ENTERS the pipeline.

**What does not change (non-negotiables still hold):** §1.1 rule 1 (templates +
Pydantic-validated param slots only — manual params are validated against the
SAME schema), rule 2 (only Freqtrade trades), rule 3 (paper_gate / live_gate are
real `interrupt()` gates — manual-injected strategies stop at them like any
other), rules 4–7. Manual injection is a CREATION shortcut, not a gauntlet or
HITL bypass.

### 21.1 Shared design notes (apply to all three features)

These are reused seams the features build on; the spec calls them out once here.

1. **D-16 spawn-vocabulary guards (reuse, do not re-derive).** Template validity
   = membership in `orchestrator.agents.generator.SHIPPED_TEMPLATES`
   (`{mean_reversion_template, freqai_classifier_template,
   freqai_regressor_template}`, BRD §8.1). Pair validity = subset of
   `orchestrator.supervisor._PAIR_UNIVERSE` (the SPEC §1 Q2 universe). These are
   exactly the checks `aspawn_strategy` runs (orchestrator/supervisor.py, D-16).
   Stage 12 should **factor the two guards into one shared validator** (e.g.
   `orchestrator/gates/spawn_vocab.py`) that both `aspawn_strategy` and the new
   manual-inject endpoint call, so there is one definition, not two. Same
   no-write contract on rejection: reject the whole request, write no registry
   row (mirrors the `aspawn_strategy` refusal contract).

2. **Deterministic render path (reuse the generator MINUS the LLM).**
   `generator_node` (orchestrator/agents/generator.py) is already split: the
   only LLM call is `_default_params_extractor` (Sonnet structured output that
   FILLS the params); everything after it is deterministic —
   `render_template(source, params)` → write `_generated/<sid>.py` →
   `validate_strategy_source()` (AST allowlist). Stage 12 should **factor that
   deterministic tail into a shared helper** (e.g.
   `render_and_validate(strategy_id, template, params) -> path`) that
   `generator_node` and the manual-inject path both call. Manual injection runs
   this helper with operator-supplied params and **never constructs the
   extractor**, so it makes zero generator LLM calls.

3. **Param schema is the hard constraint either way.** The generator validates
   LLM output against `load_schema(template)` (the co-located Pydantic
   `Field(ge=, le=)` schema, BRD §8 rule 3). Manual injection validates the
   OPERATOR's params the same way: `load_schema(template)(**params)` — a missing
   slot or out-of-range value raises and the request is rejected (HTTP 422) with
   the Pydantic error. This is §1.1 rule 1 holding for hand-supplied params.

4. **Registry insert (reuse).** `aspawn_strategy` already INSERTs the
   `strategy_registry` row (identity + template + pairs + timeframe + stage).
   Factor a shared `_insert_registry_row(...)` so the manual paths reuse it; the
   only difference is the seeded `stage` (manual injection seeds `validation`,
   not `research`) and that `template` is the real chosen template, never the
   `pending` sentinel.

5. **Entering the parent graph at the validation stage (the load-bearing
   mechanism).** The parent per-strategy graph (orchestrator/graph.py) is
   `START → research_subgraph → validation_subgraph → paper_subgraph →
   live_subgraph`, with `_route_after_research(state)` returning `"validation"`
   for any non-archived state. To run a strategy through validation onward
   WITHOUT executing research, seed the checkpoint as if research had just
   produced its output and resume:

   ```
   # (spec pseudocode — NOT to implement in this commit)
   config = {"configurable": {"thread_id": f"strategy_{sid}"}}
   await graph.aupdate_state(config, seed_state, as_node="research_subgraph")
   async for _ in graph.astream(None, config=traced_config):  # backgrounded
       ...
   ```

   `seed_state` carries everything the validation subgraph reads:
   `strategy_id, name, template, params, pairs, timeframe`, `strategy_path` +
   `artifacts["generated_strategy_path"]` (so `prepare_validation_inputs` /
   `backtest_worker` find the rendered file — orchestrator/subgraphs/validation.py),
   `stage` left non-archived, `started_at`/`last_updated`, `artifacts={…}`.
   Because `_route_after_research` then routes to `validation_subgraph`, the run
   **reuses the entire downstream path** — validation → `paper_gate` (HITL) →
   paper subgraph → `live_gate` (HITL) → live — identical gates and gauntlet.
   Within the validation subgraph the first node executed is
   `prepare_validation_inputs`; the conceptual entry point is the edge
   `research_subgraph → validation_subgraph`.

   *Fallback if `as_node=` seeding proves fragile:* add an injection-only
   conditional entry (`START → validation_subgraph` when an `injected=True` flag
   is set in the initial state), guarded so the production research path is
   untouched. Recommend the `as_node` seed first (no graph-topology change).

6. **Backgrounding + return shape.** Validation runs 6 folds × backtests
   (minutes) plus robustness; the spawn must be a fire-and-forget asyncio task
   exactly like `_default_spawn_thread_fn` (orchestrator/supervisor.py). The new
   endpoints return immediately with `{strategy_id, thread_id, stage}`; the
   operator then watches `GET /threads` and approves `paper_gate` via the
   existing `POST /threads/{tid}/approve`.

7. **Auth (reuse).** Every new write endpoint is guarded by the existing
   `_require_operator_token` dependency (`X-Operator-Token`, SPEC §6 2026-05-27),
   identical to `/approve`, `/wake`, `/supervisor/run`, and takes the same
   per-thread `asyncio.Lock` for resume/race safety.

8. **Observability (reuse).** Each manual run is one graph execution — wrap it in
   `run_context(...)` + `trace_config(...)` (Stage 10c/10d) like the existing
   spawn/approve/wake paths so its structlog lines and LangSmith trace share one
   `run_id`. No new telemetry table.

---

### 21.2 Feature 1 — Manual strategy injection

**Goal.** Let the operator submit a complete strategy definition (template +
explicit params + pairs/timeframe) and run it straight through the EXISTING
validation subgraph (`prepare_validation_inputs → plan_backtests → backtest_worker
×folds → aggregate_results → gate_backtest → plan_robustness → … → risk_analyst
→ paper_gate`, then the paper subgraph on approve), **bypassing the LLM
researcher + generator + critic entirely**. Cheapest way to test a hypothesis the
operator already has, and to re-run a strategy without re-spending creation
tokens.

**Design.**

- **New endpoint:** `POST /strategies/validate` (token-gated). Body:
  ```jsonc
  {
    "template":  "mean_reversion_template",      // ∈ SHIPPED_TEMPLATES
    "params":    { "rsi_buy_threshold": 22, ... }, // validated by load_schema(template)
    "pairs":     ["BTC/USDT", "ETH/USDT"],        // ⊆ _PAIR_UNIVERSE
    "timeframe": "5m",
    "name":      "manual-bb-stretched"            // optional label
  }
  ```
- **Validation order (all before any write — §21.1.1/.3):**
  1. `template ∈ SHIPPED_TEMPLATES` (D-16) — else 422 `unknown_template`.
  2. `pairs ⊆ _PAIR_UNIVERSE` (D-16) — else 422 `pairs_outside_universe`.
  3. `load_schema(template)(**params)` — else 422 with the Pydantic validation
     error (missing slot / out-of-range value). This is §1.1 rule 1 for manual
     params.
- **Render (no LLM):** call the shared `render_and_validate(strategy_id,
  template, params)` (§21.1.2) → writes `strategy_templates/_generated/<sid>.py`
  and runs the AST allowlist. An AST failure → 422 (should be impossible for a
  shipped template + schema-valid params, but the check stays — defense in depth).
- **Registry row:** `_insert_registry_row(...)` with `stage="validation"`,
  `template=<real template>`, the request's pairs/timeframe.
- **Enter the graph at validation:** seed `as_node="research_subgraph"` and
  `astream(None)` backgrounded (§21.1.5/.6). Enters at the
  `research_subgraph → validation_subgraph` edge; first node executed is
  `prepare_validation_inputs`.
- **Reused, unchanged:** the whole validation subgraph + its BRD §10 gates, the
  `paper_gate` `interrupt()` and `POST /threads/{tid}/approve` resume, and the
  paper/live subgraphs on approval.

**Token cost (precise — do not claim zero).** Researcher (~$0.20), generator
(~$0.04), and critic (~$0.70) — the bulk of the per-strategy LLM cost and the
*repeated-relearning* cost — are all skipped. The **one** LLM call left in the
path is the single `risk_analyst` Opus verdict at the end of validation (~$0.17,
BRD §12). For a true zero-LLM run (e.g. a pure gate-tuning sweep) the operator
may optionally inject a deterministic `risk_analyst_fn` stub via the existing
`build_validation_subgraph` seam; the DEFAULT keeps the real Opus verdict so the
paper_gate card still shows a rationale (SPEC §4.1).

**Acceptance criteria.**
1. A hand-supplied param set for a shipped template runs the full 6-fold
   walk-forward backtest + `gate_backtest` + robustness + `risk_analyst` +
   `paper_gate` with **zero researcher/generator/critic LLM calls** (verify via
   the LangSmith trace / structlog: no `researcher`/`generator`/`critic` node
   events for the thread).
2. On `paper_gate` approve, the thread spawns paper exactly as an LLM-path
   strategy does (same `paper_spawn`, same 30-day clock).
3. An unknown template or an out-of-universe pair is rejected at the endpoint
   (D-16) with **no registry row written**; schema-invalid params return 422
   with the Pydantic error and no row.

---

### 21.3 Feature 2 — Direct FreqAI spawn

**Goal.** Spawn a strategy on a SPECIFIC template (`freqai_regressor_template` /
`freqai_classifier_template`) directly, **bypassing the supervisor/researcher
template choice**, so the operator can finally test whether FreqAI trains and
runs — it has never completed (always died on the rate limit or the researcher
picked `mean_reversion` instead).

**Key fact that shapes the design.** FreqAI model training happens *inside*
`freqtrade backtesting` (the `backtest_worker` subprocess trains per-pair models
when the template enables FreqAI). So FreqAI is exercised by the **validation
backtest**, regardless of whether research ran. Two paths expose it:

- **Path A — FreqAI via Feature 1 (recommended FIRST FreqAI smoke).** Submit a
  FreqAI template + a hand-supplied param set to `POST /strategies/validate`
  (Feature 1). Zero LLM, no rate-limit exposure (directly fixes the "died on the
  429" failure), and it runs the FreqAI training inside the very first backtest
  fold. This is the cheapest answer to "does FreqAI train at all."

- **Path B — forced-template research spawn.** When the operator wants the
  researcher to FILL the FreqAI params (hypothesis-grounded) but not to CHOOSE
  the template: `POST /strategies/spawn` with `{template: "freqai_regressor_template",
  force_template: true, pairs?, timeframe?}`. Mechanism: reuse
  `aspawn_strategy(template=…)` (which already D-16-validates the template and
  seeds the registry row), and add a `force_template` flag carried into the
  initial `StrategyState` that `researcher_node` honors — the researcher still
  proposes hypothesis + `suggested_param_ranges`, but its `template_name` degree
  of freedom is **pinned** to the operator's choice before `generator` runs (a
  small new seam: today `ResearchProposal.template_name` is the only template
  source). Cost: normal research LLM cost, but the template is guaranteed.

  Recommend exposing both via the UI; default the operator to Path A for the
  first FreqAI proof-of-life, Path B when they want LLM-filled FreqAI params.

**Design check — wire/verify the FreqAI backtest path.** Because no FreqAI run
has ever completed, the spec must confirm (not assume) that the validation
backtest actually runs Freqtrade with FreqAI enabled:
- the FreqAI Docker image (`freqtradeorg/freqtrade:stable_freqai`, BRD §4) is the
  one the `backtest_runner` subprocess uses;
- the rendered config carries a `freqai_config` block with the BRD §7.3 pins
  (`train_period_days`, `backtest_period_days`, `live_retrain_hours`,
  `expiration_hours`, `feature_parameters.DI_threshold`,
  outlier rejection) — these are required, no defaults;
- the walk-forward window (`prepare_validation_inputs` / `walk_forward_from_cache`)
  leaves enough history for FreqAI's `train_period_days` + `backtest_period_days`
  ahead of the first OOS fold (a FreqAI model needs a training window the pure-TA
  template does not). If the current planner's window is too short for FreqAI,
  that is a concrete sub-item to fix in Stage 12 implementation.
A genuine FreqAI setup error (missing `freqai_config`, image mismatch, training
failure) must land in the thread's `failure_reason` / logs as a CLEAR message —
not a silent hang and not a generic 429.

**Acceptance criteria.**
1. A FreqAI strategy (regressor or classifier), spawned with its template pinned
   (Path A or B), **completes the full gauntlet train → backtest → gate** and
   reports its per-fold metrics (Sharpe / profit factor / drawdown) into
   `gate_decisions["backtest"]` — whether it passes or fails the gate.
2. **OR** it surfaces a CLEAR FreqAI setup error (model training failure / missing
   `freqai_config` / image mismatch) in `failure_reason` and the structlog/
   LangSmith trace — distinguishable from a rate-limit death.
3. The template the operator named is the template that actually ran (verify the
   rendered `_generated/<sid>.py` is the FreqAI template, not `mean_reversion`).

---

### 21.4 Feature 3 — Operator frontend (UI) over the existing API

**FIRST: what the existing dashboard already does** (`dashboard/app.py`, the
Stage 6 Streamlit dashboard — inspected for this spec). **Do not re-spec UI for
anything in this list.**

- Streamlit app; polls a SINGLE endpoint `GET /threads` (which embeds each
  thread's pending interrupt payload). Token in `OPERATOR_TOKEN` env.
- **Threads LIST view** (`render_threads_list`): a table of every registry row —
  columns `strategy_id`, `stage`, `last updated`, `HITL`. Threads with a pending
  interrupt sort to the top with a **Review** button; surfaces
  `"waiting · live slot occupied"` (D-9). Autorefresh on (5 s) so new HITL events
  appear without a manual refresh.
- **HITL CARD views** (the operator's approve/reject surface, SPEC §4.1 layout —
  rationale prominent, metrics collapsible):
  - `paper_gate` card — `risk_analyst` rationale + verdict/confidence chips +
    collapsible backtest/robustness metrics.
  - `live_gate` card — `paper_monitor` rationale + paper-vs-backtest metrics.
  - `live_pause_review` card — coordinator path (coordinator rationale + the
    three reviewer-vote chips) and the kill-switch path (distinct red
    `kill_switch_card`, acknowledge-only).
  - All approve/reject go through `_render_approve_reject_form` → `POST
    /threads/{tid}/approve` with notes + `X-Operator-Token`.

So the dashboard **already covers**: the live thread table (stage + HITL state)
and the approve/reject HITL controls at all three gates. The §4.1 rationale
layout is done.

**What it does NOT cover (the F3 gap):**
- No way to **trigger a supervisor run** (no button → `POST /supervisor/run`).
- No **manual-inject** form (Feature 1).
- No **direct FreqAI spawn** form (Feature 2).
- No **metrics columns** (Sharpe / drawdown) in the thread table — metrics exist
  ONLY inside the gate cards' collapsible JSON (read from the interrupt payload),
  never as a sortable at-a-glance column, and never sourced from `/metrics` or
  `telemetry`.
- No operator-initiated **pause/stop** (v1 has no such endpoint — see below).
- No **wake** button (APScheduler drives `/wake`; that is correct — leave it).

**Recommendation: EXTEND the existing Streamlit dashboard. Do NOT build a new
frontend.** Rationale:
- It already implements the hardest part — the SPEC §4.1 HITL card rendering
  (incl. the kill-switch variant), the single-poll `/threads` data model, and the
  token-gated approve flow — all tested.
- Streamlit 1.40+ is the BRD §4-pinned UI; this is a single-operator,
  loopback/WireGuard deployment (BRD §15) that does not warrant an SPA.
- A second frontend would duplicate the HITL surface for zero operator benefit
  and double the maintenance / auth surface.

**New UI surfaces to add (all driving existing or Feature-1/2 endpoints):**

| UI element | Drives | Exists vs new |
|---|---|---|
| "Run supervisor" button (+ `dry_run` toggle) | `POST /supervisor/run` | endpoint EXISTS; UI NEW |
| "Manual inject" form: template dropdown (`SHIPPED_TEMPLATES`), params (schema-driven fields from `load_schema`, or a JSON textarea for v1), pairs multiselect (`_PAIR_UNIVERSE`), timeframe | `POST /strategies/validate` (Feature 1) | endpoint NEW (F1); UI NEW |
| "Direct FreqAI spawn" form: template dropdown limited to `freqai_*`, Path A (params) or Path B (`force_template`) | `POST /strategies/validate` (Path A) / `POST /strategies/spawn` (Path B) (Feature 2) | endpoint NEW (F2); UI NEW |
| Thread-table **Sharpe + max-drawdown** columns | enriched `GET /threads` (see below) | data source NEW; UI NEW |
| Approve / reject at every gate | `POST /threads/{tid}/approve` | EXISTS — keep |

**The one genuinely-new backend bit for F3 — metrics in the table.** Sharpe /
drawdown are not in `GET /threads` today. `list_threads` already calls
`graph.aget_state(config)` per thread; extend it to also pull the best param
set's `sharpe_is` / `profit_factor` / `max_dd` from
`snapshot.values["gate_decisions"]["backtest"]` (already in the checkpoint — no
new query) for validated/paper threads, and for live/paper containers read the
in-orchestrator Freqtrade exporter (`collect_freqtrade_metrics`, Stage 10e.2) or
the Prometheus `/metrics` it feeds. Recommend embedding the backtest summary into
the existing `/threads` payload (cheapest — one endpoint, no new query) and
sourcing live P&L/drawdown from the exporter.

**On "pause" controls (clarification, not a silent gap).** v1 has **no
operator-initiated pause**: per BRD §5.6 a `live_pause` is produced only by the
coordinator vote or the out-of-band kill switch, and per SPEC §4.3 every
`live_pause` is HITL with no auto-resume. The dashboard already surfaces those
`live_pause_review` cards. If the operator wants a manual "pause/stop this live
strategy" button, that is a NEW `POST /threads/{tid}/pause` endpoint calling
Freqtrade `/api/v1/stop` and routing the thread to `live_pause` — flag it as a
small, optional addition for F3 (not required by the acceptance criterion, which
is parity with the *current* CLI workflow, and the CLI has no pause either).

**Acceptance criteria.** From the UI alone, the operator can do everything the
OPERATOR_RUNBOOK currently does via PowerShell/curl:
1. Trigger a supervisor run (with `dry_run` toggle).
2. Manually inject a validated strategy (Feature 1) and see it appear in the
   thread table at `stage="validation"`.
3. Directly spawn a FreqAI template (Feature 2).
4. See a live thread table with stage **plus** Sharpe and max-drawdown.
5. Approve / reject at every HITL gate (already works).

No remaining operator action requires dropping to PowerShell/curl for routine
operation (DB-level dead-thread cleanup, OPERATOR_RUNBOOK §5.2, stays a CLI/admin
task — out of scope for F3).

---

## 22. Short-selling capability (Stage 13)

> **Status: SPEC ONLY.** This section is the contract Stage 13 is built from.
> It introduces **NO code** — per SPEC §4.4 rule 3 it lands as a standalone
> docs commit; implementation lands in later commits, each gated on this spec
> and on operator review of the full plan.
>
> **This section SUPERSEDES the long-only-spot market scope of §1** ("spot
> only … no futures, no margin, no leverage in v1"). That was an
> operator-revisable *market-scope* choice — NOT one of the §1.1 safety
> non-negotiables — and the operator has deliberately revised it (see SPEC §6
> 2026-06-09). The §1.1 rules (templates-only, Freqtrade-only execution, real
> HITL gates, ≥30-day paper, separate keys, Postgres-from-day-one, out-of-band
> kill switch) all still hold and are *extended*, not weakened, by §22.

### 22.0 Why (operator decision)

The validation window is a **bear market** (Nov 2025–May 2026: every SPEC §1 Q2
pair fell 12–39%, with a ~−25% crash in one fold — SPEC §6 2026-06-09). Every
long-only strategy tested has failed: the backtest-passing ones fail the
robustness gate on fee+regime fragility, and the only long-only shape that
stays positive through a bearish window is a thin, selective regime-filtered
entry. A long-only spot system structurally *cannot* profit from the dominant
move of its own validation period (price falling). The operator has decided to
add **short-selling** so the system can profit in bear regimes, and **accepts
liquidation risk** in exchange for a calculated-risk management system around
it (§22.2).

This is a deliberate, contract-level change, split into two phases so that the
**expensive, dangerous half (live margin execution) is only built if the cheap,
safe half (backtest validation) first proves shorting has edge.**

- **Phase 1 — short signals in BACKTEST / VALIDATION only.** No live risk, no
  real money, no futures *execution*. Prove or disprove edge through the SAME
  gauntlet and gates that judge long strategies.
- **Phase 2 — LIVE short execution on Binance futures/margin.** Only justified,
  and only begun, **if Phase 1 finds edge.** This is where the full risk model
  lives, because this is where real capital can be lost beyond the spot bound.

**Gating rule (non-negotiable):** Phase 2 implementation does not start until
(a) Phase 1 is shipped and (b) at least one short-capable strategy has cleared
the full Phase-1 validation gauntlet (passed the BRD §10 gates) on a real
`POST /strategies/validate` run — i.e. shorting has *demonstrated* edge on
honest walk-forward, not just been hypothesized. If Phase 1 shows no short
strategy can pass the gates, Phase 2 is abandoned and the system stays
long-only spot live.

---

### 22.1 PHASE 1 — short signals in backtest / validation only

**Goal.** Let strategy templates emit short entries/exits (`enter_short` /
`exit_short`) and have the validation gauntlet (research → 6-fold anchored
walk-forward backtest → backtest gate → robustness → `risk_analyst` →
`paper_gate`) evaluate them, **with zero live or real-money exposure and zero
change to the live execution path.** The only mechanism that changes is the
*backtest* Freqtrade trading mode; everything that touches real money stays
long-only spot.

**Design — what changes.**

1. **Template contract (§8) gains a short side.** Today every template hard-sets
   `can_short = False  # BRD §1: spot-only, long-only` and only writes
   `enter_long` / `exit_long` columns (e.g.
   `strategy_templates/mean_reversion_template.py:55`,
   `bb_regime_reversion_template.py:56`, `donchian_regime_trend_template.py:53`).
   Phase 1 adds a *new* class of short-capable template whose structural shell
   sets `can_short = True` and whose `populate_entry_trend` /
   `populate_exit_trend` may set `enter_short` / `exit_short` (Freqtrade's
   native short columns) **in addition to** the long columns. The structural
   shell stays hand-written and untouchable (§8 rule 1); the short-side
   thresholds become new `# SLOT:` lines. **Existing long-only templates are
   left exactly as they are** — `can_short = False` is still valid and still the
   default; short capability is opt-in per template, never a global flip.

2. **Co-located Pydantic schema (§8 rule 3) gains short slots.** A short-capable
   template's `*_schema.py` adds the short-side fields (e.g. an
   `rsi_short_threshold`, a short stoploss / take-profit) with the same
   `Field(ge=, le=)` discipline. `load_schema(template)` and the generator's
   structured-output extractor fill them exactly like long slots — **no new LLM
   path, no free-form code** (§1.1 rule 1 holds verbatim).

3. **AST validator (`orchestrator/security/ast_validator.py`) — UNCHANGED.**
   Verified: shorting needs no new imports (still `talib` / `freqtrade` /
   `pandas` / `numpy`) and introduces no forbidden name. `enter_short` /
   `exit_short` are DataFrame column assignments, not imports or calls, so the
   allowlist + `FORBIDDEN_NAMES` walk passes a short template unchanged. The
   allowlist intentionally stays as narrow as it is.

4. **Backtest trading mode flips to futures — IN BACKTEST ONLY.** The
   load-bearing change. `_build_backtest_config`
   (`orchestrator/tools/backtest_runner.py:374`) hard-codes
   `"trading_mode": "spot"  # BRD §1: spot-only`. Phase 1 makes this
   **per-strategy**: a short-capable strategy (template `can_short = True`)
   renders its backtest config with `"trading_mode": "futures"` +
   `"margin_mode": "isolated"` and a `leverage` of **1× in Phase 1** (no
   leverage — a 1× short on futures is the cleanest way to backtest the short
   *signal* without conflating it with leverage risk, which is a Phase-2
   concern). A long-only strategy renders byte-identical spot config as today.
   The same flip must be mirrored in the **lookahead-analysis** config
   (`orchestrator/tools/lookahead.py:248-249`, currently
   `"trading_mode": "spot"`, `"margin_mode": ""`) so the §5.3 `lookahead_gate`
   analyses the strategy in the same mode it will be backtested in — otherwise
   lookahead runs spot while the backtest runs futures and the bias check is
   meaningless for the short side.

5. **Futures OHLCV + funding/mark data must exist for the backtest.** A futures
   backtest needs futures candles (and, for honest P&L, funding-rate + mark-price
   series). The Stage 3 download (`download-data … --exchange binance`) pulls
   spot candles only. Phase 1 needs a futures data pull
   (`download-data --trading-mode futures --candle-types futures funding_rate mark`)
   for the SPEC §1 Q2 pairs/timeframes. **This is a Docker/data step the operator
   runs** (operator checkpoint — see checklist).

6. **Gates handle short metrics by being direction-agnostic — confirm, do not
   assume.** Freqtrade computes Sharpe, profit factor, drawdown, and trade counts
   from realized trade P&L regardless of trade *direction*, so `gate_backtest`,
   the OOS/walk-forward gate, and the robustness workers (`monte_carlo_worker`,
   `regime_worker`, `fee_stress_worker`) operate on the same numbers and apply
   the **same BRD §10 thresholds unchanged** — that is the whole point (prove
   short edge against the *same* bar). Two short-specific confirmations the
   implementation must make: (a) the **fee-stress** worker must include futures
   *taker* fees (futures fee schedule differs from spot) and (b) **funding cost**
   must flow into the backtest P&L (Freqtrade futures backtest accounts for
   funding when the data is present) so a short that only "wins" by ignoring
   funding bleed fails honestly. No new threshold is added in Phase 1 — if a
   short strategy can't clear the existing gates, it has no edge.

7. **`risk_analyst` (§5.4) prompt gains short awareness.** The Opus verdict node
   should note when a strategy's edge comes from the short side and reason about
   short-specific fragility (funding drag, short squeezes, the asymmetry that a
   short's loss is unbounded as price rises) — a prompt change, not a gate
   change. The verdict still routes to `paper_gate` or `archive` exactly as
   today.

**What STAYS long-only spot in Phase 1 (the containment boundary).**

- **The entire LIVE path is untouched:** `live-base.json`
  (`"trading_mode": "spot"`, `"margin_mode": ""`), `render_live_config` /
  `spawn_live_container` (`orchestrator/subgraphs/live.py`), the live stake cap
  (`capped_stake = min(stake_intent, LIVE_CAPITAL_CAP_USD)`), the kill switch
  (`orchestrator/scheduler.py`), and the live secrets (`BINANCE_LIVE_API_*`) all
  stay exactly as shipped. No real-money code path learns the word "short" in
  Phase 1.
- **The PAPER path stays long-only spot, and short-capable strategies are
  BLOCKED from promotion to paper in Phase 1.** `paper-base.json` stays
  `"trading_mode": "spot"`. A short-capable strategy that *passes* `paper_gate`
  must NOT spawn into a spot paper container (it cannot short there). Phase 1
  adds a guard at `paper_spawn` (mirroring the D-9 "block the gate" pattern):
  if `can_short = True` (or trading_mode would be futures) and Phase 2 is not
  shipped, refuse promotion and record a clear
  `gate_decisions["paper_gate"]["status"] = "short_paper_deferred_to_phase2"`
  hold — the strategy's validation verdict stands as the Phase-1 deliverable,
  but it does not paper-trade until Phase 2's paper/live margin path exists.
  This keeps Phase 1 strictly "validation proves edge; nothing executes."
  *(Open sub-decision flagged for the operator at Phase-1 build time: whether to
  instead allow a futures **dry-run** paper container — still no real money — so
  short strategies get a 30-day paper observation before Phase 2. Deferred,
  because a futures paper container is most of Phase 2's spawn plumbing and
  belongs with it; see DEFERRED D-21.)*

**Risk model (Phase 1).** Trivial by construction: **zero live exposure, zero
real money.** A backtest in `trading_mode: "futures"` at 1× leverage with
`dry_run` is a simulation over historical candles — no exchange order, no
margin account, no liquidation. The only "risk" is an *honesty* risk — that a
short backtest overstates edge by mis-accounting fees or funding — which §22.1
items 6 mitigate (futures fees + funding in P&L) and which the existing
robustness gate (fee-stress) and the §1.1-rule-4 30-day paper requirement
(deferred to Phase 2 for shorts) are the backstops for.

**Files / subsystems that change (Phase 1).**

| Subsystem | Change | Contained vs ripples |
|---|---|---|
| `strategy_templates/<new short template>.py` + `_schema.py` + `_README.md` | NEW short-capable template(s): `can_short = True`, `enter_short`/`exit_short`, short SLOTs | **Contained** — new files; existing templates untouched |
| `orchestrator/agents/generator.py` | register new template in `SHIPPED_TEMPLATES` / `_SCHEMA_CLASS_NAMES` (D-16 single-source set) | **Contained** — additive set membership; render/AST tail unchanged |
| `orchestrator/tools/backtest_runner.py` (`_build_backtest_config`) | `trading_mode`/`margin_mode`/`leverage` become per-strategy (futures+isolated+1× for short-capable; spot otherwise) | **Ripples** — touches the one config builder every backtest uses; long path must stay byte-identical (regression-pin it) |
| `orchestrator/tools/lookahead.py` | mirror the same per-strategy trading-mode in the lookahead config | **Contained** — same flip, one place |
| `orchestrator/subgraphs/validation.py` | confirm fee-stress uses futures fees + funding flows into backtest P&L; no threshold change | **Contained** — verification + small config wiring |
| `orchestrator/agents/risk_analyst.py` (prompt) | short-awareness in the rationale prompt | **Contained** — prompt only |
| `orchestrator/subgraphs/paper.py` (`paper_spawn` guard) | block short-capable promotion to spot paper in Phase 1 (D-9-style "block the gate") | **Contained** — guard + status field; reuses an existing pattern |
| `orchestrator/gates/thresholds.py` | **NO change** in Phase 1 (same gates judge short edge) | — |
| `orchestrator/security/ast_validator.py` | **NO change** (verified — no new imports/names) | — |

**Operator checkpoints (Phase 1).** Downloading futures + funding-rate + mark
OHLCV via the Freqtrade Docker image, and running the first real
`POST /strategies/validate` of a short template end-to-end against that data,
are Docker/data steps **the operator runs** (marked `[OPERATOR]` in the
checklist). CC writes the code and the templates; the operator pulls the data
and runs the gauntlet.

**Acceptance criteria (Phase 1).**

1. A short-capable strategy (template `can_short = True`, schema-valid short
   params) runs the **full validation gauntlet** — research/generator/critic OR
   `POST /strategies/validate` manual inject (BRD §21 F1), then 6-fold anchored
   walk-forward backtest in `trading_mode: "futures"` 1× → `gate_backtest` →
   robustness → `risk_analyst` → `paper_gate` — and **either passes or fails the
   SAME BRD §10 gates** as a long strategy. The backtest demonstrably executed
   short trades (verify `enter_short`/`exit_short` trades present in the parsed
   `BacktestResult`).
2. **Zero live/real-money exposure:** no live container, no paper container, no
   futures account touched. A short strategy that passes `paper_gate` halts at
   the Phase-1 `paper_spawn` guard with `status="short_paper_deferred_to_phase2"`
   (verify no Freqtrade container spawned).
3. The long-only path is provably unchanged: an existing long template's
   rendered backtest config is byte-identical to pre-Stage-13 (regression test
   on `_build_backtest_config` for a `can_short = False` strategy).
4. **The Phase-1 verdict is the Phase-2 gate:** the run produces a clear
   pass/fail record per the gating rule (§22.0). At least one short strategy
   passing the gates is the precondition to begin Phase 2.

---

### 22.2 PHASE 2 — LIVE short execution (only if Phase 1 finds edge)

> **Precondition (hard):** do not begin Phase 2 implementation until §22.0's
> gating rule is met — Phase 1 shipped AND ≥1 short strategy has passed the full
> validation gauntlet. This is the load-bearing half of §22; the risk model
> below is the reason short-selling is safe to take live at all.

**Goal.** Real short trading on **Binance USDⓈ-M futures (perpetuals),
isolated margin, conservative leverage**, live, with a risk model under which a
single position's **maximum loss is bounded by a hard leverage + stop +
liquidation policy** — restoring the bounded-loss property that spot gave for
free and that naive margin trading destroys.

**Why the current $500 model breaks under margin (the core problem).** In spot,
loss is bounded for free: you can only lose what you posted (price floors at 0,
and the stoploss exits long before that), so `LIVE_CAPITAL_CAP_USD = 500` with
`capped_stake = min(stake_intent, 500)` (`orchestrator/subgraphs/live.py`) means
the catastrophic account loss ≈ $500 and a single position's loss ≤ its stake.
**A short on margin breaks every assumption in that math:**

- A short's loss is **unbounded to the upside** — price can rise arbitrarily,
  so notional loss has no natural ceiling.
- With leverage *L*, a posted margin *M* controls notional *N = M·L*; an adverse
  move of only ≈ `(1/L − maintenanceMarginRate)` **liquidates** the position,
  losing ~the full margin *M* (isolated) — and under **cross** margin the loss
  can spill into the rest of the balance and even go **negative** (you owe the
  exchange), exceeding principal.
- So `stake_amount` no longer means "max loss." The spot cap math
  (`min(stake, 500)`) silently under-bounds risk: a $125 margin at 5× is $625
  notional with a liquidation only ~20% away.

**The risk model (the hard policy that re-bounds single-position loss).** Layered,
all of these together, so the bound holds even if one layer fails:

1. **Isolated margin, never cross.** `margin_mode: "isolated"` is hard-set in the
   live futures config and asserted at config-write (mirroring how
   `render_live_config` hard-asserts `dry_run: false`). Isolated caps a
   position's loss at *its own posted margin* — a cross-margin liquidation that
   reaches into the whole balance (and can exceed principal) is **structurally
   forbidden**, not merely avoided.
2. **Hard, conservative leverage cap.** New `MAX_LEVERAGE` in
   `orchestrator/gates/thresholds.py` (operator wants conservative — **propose
   `MAX_LEVERAGE = 2`, hard ceiling 3**; operator signs off the value, recorded
   in SPEC §2). Enforced in two places: the strategy's Freqtrade `leverage()`
   callback returns `min(requested, MAX_LEVERAGE)`, AND the live config /
   spawn path refuses to start a container whose effective leverage exceeds the
   cap. Lower *L* pushes the liquidation price further from entry, buying room
   for the stop to fire first.
3. **Stop-loss strictly inside the liquidation distance (the load-bearing
   invariant).** New `MIN_LIQUIDATION_BUFFER_PCT`: the strategy's hard
   `stoploss` distance must be ≤ a fraction (propose 50%) of the liquidation
   distance implied by its leverage, so under normal conditions the **stop exits
   before liquidation** and the realized loss is the (small) stop loss, not the
   (full-margin) liquidation. At `MAX_LEVERAGE = 2` liquidation is ~50% away; a
   −5%…−10% stop fires far earlier. A strategy whose stop is *outside* its
   liquidation buffer is rejected by a new validation/admission gate — it is not
   allowed to go live.
4. **Position sizing under margin.** `stake_amount` is reinterpreted as **posted
   margin**, and the cap is applied to BOTH margin and notional: sum of posted
   margins ≤ `LIVE_CAPITAL_CAP_USD`, and `notional = margin·leverage` is itself
   capped so total notional exposure stays within an operator-set multiple of
   the $500 (propose: total notional ≤ 1×–2× `LIVE_CAPITAL_CAP_USD`, never the
   uncapped `M·L`). The Binance **futures** minimum notional differs from spot,
   so the min-notional sanity check (today implicit in the ~$125/position spot
   math) is recomputed for futures. `MAX_CONCURRENT_LIVE_STRATEGIES = 1` and the
   stake-subdivision concern already flagged in `thresholds.py` carry straight
   into this.
5. **Funding-rate accounting.** Perpetual shorts pay or receive funding every 8h.
   Live P&L, the daily-loss limit, and the drawdown computation must include
   accrued funding (a persistent funding bleed can drain a position that the
   price chart says is flat). The kill-switch drawdown denominator must be
   *equity including unrealized + funding*, not realized-only.
6. **Kill switch + drawdown + daily-loss limits rebuilt for margin.** Today
   `_kill_reason` (`orchestrator/scheduler.py`) fires on `max_drawdown ≥ 0.12`
   or `consecutive_losses ≥ 10`, polling `/api/v1/profit` + `/trades` every 5
   min — all premised on bounded spot loss and a slow-moving account. Margin
   needs:
   - **A new liquidation-proximity trigger** — the primary new safety. Poll the
     futures position (`/api/v1/status` carries `liquidation_price`, `leverage`,
     `isolated`; or the futures `/profit`) and fire `POST /api/v1/forceexit`
     (or `/stop`) when **mark price comes within `KILL_SWITCH_MARGIN_RATIO` of
     the liquidation price** (new threshold — propose firing at margin ratio /
     liquidation-distance well before the exchange would liquidate). Because a
     5-minute poll can be too slow for a fast adverse move, **the per-trade
     stoploss-inside-liquidation invariant (item 3) is the primary bound and the
     kill switch is the secondary net** — the spec must say so explicitly so the
     poll cadence is never mistaken for the main protection.
   - **Drawdown / daily-loss computed on margin equity** (item 5), and the 12%
     global-drawdown + −3% daily-loss limits re-validated for the leverage in
     use (a 12% account move arrives `L`× faster under leverage).
   - The kill switch must still run **out-of-band of the graph** (§1.1 rule 7),
     now also surviving as the liquidation guardian.
7. **Futures account / API / KYC prerequisites (operator).** A Binance USDⓈ-M
   **futures** account is separate from spot: it needs its own enablement,
   derivatives **KYC** (note: SPEC §6 records India regional gating already
   blocked *spot* API access at KYC level 2 — **futures/derivatives access is
   typically more restricted and may be unavailable in the operator's
   jurisdiction; this is a hard operator pre-flight that could block Phase 2
   entirely**), and a **separate futures API key** with *Futures trading
   permission enabled* and **withdrawals disabled** (§1.1 rule 5 extended: live
   futures keys are distinct from both paper and live-spot keys —
   `secrets.load_live_credentials`' `SecretCollisionError` must be extended to a
   futures-key triple). The first real futures dry-run and the first tiny live
   short are operator-run exchange steps.
8. **HITL gates adapt (still real gates — §1.1 rule 3).** `paper_gate`,
   `live_gate`, and `live_pause_review` cards (SPEC §4.1) gain short/leverage
   context: leverage in use, liquidation price + distance, the stop-vs-liquidation
   buffer, accrued/expected funding, and isolated-margin confirmation, rendered
   **above** the metrics so the operator approves a live short with the
   liquidation picture front-and-center. **New explicit risk gates** (validation
   + live-admission, in `thresholds.py` + the live path):
   `MAX_LEVERAGE`, `MIN_LIQUIDATION_BUFFER_PCT`, `KILL_SWITCH_MARGIN_RATIO`,
   `MAX_TOTAL_NOTIONAL` (or a notional multiple of `LIVE_CAPITAL_CAP_USD`), and
   an `isolated`-margin assertion. A strategy failing any is refused live
   admission regardless of HITL — HITL can only *reject*, never *override* a hard
   risk gate (same asymmetry as the kill switch overriding the graph).

**Files / subsystems that change (Phase 2).**

| Subsystem | Change | Contained vs ripples |
|---|---|---|
| `freqtrade/user_data/configs/live-base.json` | `trading_mode: "futures"`, `margin_mode: "isolated"`, leverage wiring for short-capable live strategies | **Ripples** — the live contract surface; long-spot live must stay supported (per-strategy, not global) |
| `orchestrator/subgraphs/live.py` (`render_live_config`, `spawn_live_container`, stake cap) | margin-aware sizing (margin vs notional caps), leverage cap enforcement, isolated assertion, futures key wiring | **Ripples** — the money path; highest-care changes |
| `orchestrator/gates/thresholds.py` | NEW: `MAX_LEVERAGE`, `MIN_LIQUIDATION_BUFFER_PCT`, `KILL_SWITCH_MARGIN_RATIO`, `MAX_TOTAL_NOTIONAL` | **Contained** — additive constants (recorded in SPEC §2) |
| `orchestrator/scheduler.py` (`kill_switch_poll_job`, `_kill_reason`, `daily_loss_job`) | liquidation-proximity trigger; drawdown/daily-loss on margin equity incl. funding; futures endpoints | **Ripples** — out-of-band safety; must stay graph-independent |
| `orchestrator/security/secrets.py` | futures-key triple + extended `SecretCollisionError` (paper ≠ live-spot ≠ live-futures) | **Contained** — extends an existing seam |
| short-capable templates | Freqtrade `leverage()` callback returning `min(req, MAX_LEVERAGE)`; futures-aware stoploss | **Contained** — per-template method |
| `dashboard/app.py` + `GET /threads` | leverage / liquidation-distance / funding / isolated context on the HITL cards (SPEC §4.1) | **Contained** — extends existing cards |
| `orchestrator/subgraphs/paper.py` | replace the Phase-1 `short_paper_deferred_to_phase2` guard with a real futures (dry-run first, then live) paper path | **Ripples** — unblocks what Phase 1 deferred |

**Risk model — acceptance (Phase 2).** A **documented risk model** in which:
1. a single short position's **maximum loss is bounded** by the hard
   leverage + stop + isolated-margin policy (isolated margin caps loss at posted
   margin; the stop sits provably inside the liquidation distance via
   `MIN_LIQUIDATION_BUFFER_PCT`; leverage ≤ `MAX_LEVERAGE`), and a strategy that
   cannot satisfy the buffer is refused live admission;
2. the **kill switch triggers on margin / liquidation proximity** (new
   `KILL_SWITCH_MARGIN_RATIO` trigger), out-of-band of the graph, on margin
   equity including funding — and the spec states plainly that the per-trade
   stop is the primary bound and the poll the secondary net;
3. **every live short still passes HITL** at `live_gate` with the full
   leverage/liquidation/funding picture, and no hard risk gate can be overridden
   by approval;
4. exchange/KYC/API prerequisites are operator-verified before the first live
   short (separate futures key, withdrawals disabled, isolated margin, tiny
   first position).

**Operator checkpoints (Phase 2).** Futures-account enablement + derivatives
KYC + futures-API-key creation (withdrawals disabled), the first real futures
dry-run container, and the first tiny live short are all `[OPERATOR]` exchange
steps. CC builds the risk model, gates, sizing math, kill-switch logic, and
HITL surfaces; the operator runs every step that touches a real futures account.

---

### 22.3 Build checklist (executable plan — Phase 1 then Phase 2)

Each step is `[CC]` (code — Claude Code does it) or `[OPERATOR]` (Docker /
real-exchange — operator does it). **No Phase-2 step starts until §22.0's
gating rule is met.** Each `[CC]` code step follows SPEC §4.4 (SPEC/contract
edits in their own commit ahead of dependent code) and lands tested.

**PHASE 1 — backtest/validation short capability (no live risk)**

- [ ] **P1-1 `[CC]`** Add a short-capable template (start with one pure-TA
  shape, e.g. a Bollinger/RSI mean-reversion that also shorts the upper band):
  new `*_template.py` with `can_short = True`, `enter_short`/`exit_short`,
  short `# SLOT:`s; co-located `*_schema.py` with short `Field(ge=,le=)` slots;
  `*_README.md` hypothesis. Existing templates untouched.
- [ ] **P1-2 `[CC]`** Register the new template in `SHIPPED_TEMPLATES` /
  `_SCHEMA_CLASS_NAMES` (`orchestrator/agents/generator.py`) — D-16 single
  source of truth. (Auto-flows into `POST /strategies/validate` and the
  supervisor vocabulary.)
- [ ] **P1-3 `[CC]`** Make `_build_backtest_config`
  (`orchestrator/tools/backtest_runner.py`) per-strategy: `trading_mode:
  "futures"` + `margin_mode: "isolated"` + `leverage: 1` for `can_short = True`
  templates; byte-identical spot config otherwise. Regression-pin the long-only
  output.
- [ ] **P1-4 `[CC]`** Mirror the same per-strategy trading mode in the
  lookahead-analysis config (`orchestrator/tools/lookahead.py`).
- [ ] **P1-5 `[CC]`** Confirm + wire futures fees and funding into the backtest
  P&L and the `fee_stress_worker` (`orchestrator/subgraphs/validation.py`); no
  threshold change. Add a test that a short backtest's trades carry funding.
- [ ] **P1-6 `[CC]`** Add the `paper_spawn` Phase-1 guard
  (`orchestrator/subgraphs/paper.py`): block short-capable promotion to the
  spot paper container, record `status="short_paper_deferred_to_phase2"`
  (D-9-style block-the-gate). Test: short strategy passing `paper_gate` spawns
  no container.
- [ ] **P1-7 `[CC]`** Short-awareness in the `risk_analyst` prompt
  (`orchestrator/agents/risk_analyst.py`).
- [ ] **P1-8 `[OPERATOR]`** Download Binance **futures** OHLCV + funding-rate +
  mark data for the SPEC §1 Q2 pairs/timeframes via the Freqtrade Docker image
  (`download-data --trading-mode futures --candle-types futures funding_rate
  mark …`). *(Docker / data — operator runs.)*
- [ ] **P1-9 `[OPERATOR]`** Run the first short template end-to-end through
  `POST /strategies/validate` against that futures data; confirm short trades
  executed in the backtest and the run reaches `paper_gate` with a real
  pass/fail verdict. *(This run is the Phase-2 gate.)*
- [ ] **P1-10 `[CC]`** Record the Phase-1 result: if a short strategy passed the
  gates → shorting has edge → Phase 2 is justified (SPEC §6 note). If none
  passed → Phase 2 is abandoned; system stays long-only spot live.

**PHASE 2 — live short execution (ONLY if P1-9/P1-10 proved edge)**

- [ ] **P2-1 `[OPERATOR]`** Pre-flight: confirm Binance **futures/derivatives**
  access is available in the operator's jurisdiction and KYC tier (could block
  Phase 2 entirely). Enable the futures account; create a **separate futures API
  key** with Futures-trading permission and **withdrawals disabled**. *(Exchange
  — operator runs.)*
- [ ] **P2-2 `[CC]`** SPEC §2 + `thresholds.py` (own commit, ahead of code):
  add `MAX_LEVERAGE` (propose 2, ceiling 3 — operator signs off),
  `MIN_LIQUIDATION_BUFFER_PCT`, `KILL_SWITCH_MARGIN_RATIO`, `MAX_TOTAL_NOTIONAL`.
- [ ] **P2-3 `[CC]`** Extend `secrets.py` to a futures-key triple +
  `SecretCollisionError` (paper ≠ live-spot ≠ live-futures).
- [ ] **P2-4 `[CC]`** `leverage()` callback on short-capable templates returning
  `min(requested, MAX_LEVERAGE)`; futures-aware stoploss within the liquidation
  buffer.
- [ ] **P2-5 `[CC]`** `live-base.json` + `render_live_config` /
  `spawn_live_container` (`orchestrator/subgraphs/live.py`): per-strategy
  `trading_mode: "futures"` + `margin_mode: "isolated"` (asserted), margin-aware
  sizing (margin + notional caps), leverage-cap + isolated assertions. Long-spot
  live must stay supported.
- [ ] **P2-6 `[CC]`** Live-admission risk gate: refuse to admit a strategy whose
  stop is outside `MIN_LIQUIDATION_BUFFER_PCT` or whose leverage/notional
  exceeds the caps — HITL cannot override.
- [ ] **P2-7 `[CC]`** Rebuild the kill switch (`orchestrator/scheduler.py`):
  liquidation-proximity trigger (`KILL_SWITCH_MARGIN_RATIO`, futures endpoints,
  `forceexit`); drawdown + daily-loss on margin equity incl. funding; keep
  out-of-band of the graph.
- [ ] **P2-8 `[CC]`** Replace the Phase-1 `paper_spawn` deferral with the real
  futures paper path (**dry-run first**), then the live promotion path.
- [ ] **P2-9 `[CC]`** HITL cards + `GET /threads` (`dashboard/app.py`):
  leverage / liquidation-distance / stop-vs-liquidation buffer / funding /
  isolated context above the metrics (SPEC §4.1).
- [ ] **P2-10 `[OPERATOR]`** First real **futures dry-run** container against the
  futures key; verify `/api/v1/status` reports isolated margin + the expected
  leverage + a sane liquidation price. *(Docker / exchange — operator runs.)*
- [ ] **P2-11 `[OPERATOR]`** First **tiny live short** (well under
  `LIVE_CAPITAL_CAP_USD`) through `live_gate` HITL; verify the kill switch fires
  on a synthetic margin-proximity breach before liquidation. *(Real money,
  smallest possible — operator runs.)*

> **STOP for operator review.** This §22 plan (both phases + checklist) is the
> contract. Operator reviews the full plan before any Stage 13 implementation
> begins; Phase 2 is additionally gated on Phase 1 proving edge (§22.0).
