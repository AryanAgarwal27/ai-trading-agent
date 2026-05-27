# SPEC — operator-specific choices for ai-trading-agent v1

> **Companion to `BRD.md`. The BRD is the contract; this SPEC records the single-operator choices the BRD §18 left open.**
> **Operator:** Aryan Agarwal
> **Date locked:** 2026-05-26
> **Stage:** 0 (initial)

---

## 1. §18 answers

| # | Question | Decision | Rationale (short) |
|---|---|---|---|
| Q1 | Exchange for v1 | **Binance** | Deepest spot liquidity; tightest spreads; most-tested Freqtrade integration; best FreqAI data availability. |
| Q2 | Initial pair list | **BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT** | Deep order books → backtests least distorted by slippage; spans behavioural variety (anchor / BTC-beta / higher-vol L1 / exchange-correlated) for regime-based research. |
| Q3 | Live capital cap | **$500 USD** | Matches BRD §10 threshold tuning. With `max_open_trades=4` gives ~$125/position, comfortably above Binance min notional. 12% kill-switch worst case = $60 lesson, not a meaningful loss. |
| Q4 | LLM tracing | **LangSmith SaaS** | Native LangGraph/LangChain integration, auto-tags `thread_id`/`node`/tool calls/tokens without custom instrumentation; free tier sufficient for single-operator v1. |
| Q5 | Backup target | **Google Drive via `rclone`** (placeholder; setup deferred to Stage 10) | Operator preference. **Deviates from BRD §13's B2/S3 example** — see §3 below. |
| Q6 | Risk analyst model | **Opus 4.7** | Final automated checkpoint before a strategy consumes a 30-day human-monitored paper slot; deeper reasoning over the robustness table is worth the extra ~$0.12 per strategy. |
| Q7 | Hyperopt in v1 | **No — defer to v2** | Classic overfitting vector; would balloon backtest cost 10–100×. Researcher proposes parameter ranges grounded in regime + past failures (Store-backed); walk-forward judges them. |
| Q8 | `live_pause` policy | **Every `live_pause` is HITL — no auto-resume in v1** | Every pause is a learning event while operator builds intuition for routine vs serious pause reasons. **Revisit criterion:** propose auto-resume in a BRD diff only after operator has manually reviewed ≥20 `live_pause` events. |

---

## 2. Threshold overrides

**None.** All values in `BRD.md §10` (gate thresholds) stand as defaults for v1. To be re-tuned per BRD §10 after the first 10 strategies have completed a lifecycle. They will live in `orchestrator/gates/thresholds.py` (single source of truth, per BRD §10).

---

## 3. Deviations from BRD defaults

Documented here so they are not silent changes against the contract.

### 3.1 Backup target — Google Drive via `rclone` (vs BRD §13 Backblaze B2 / S3)

- **What changes:** `ops/backup.sh` (Stage 10) will use `rclone copy ... gdrive:ai-trading-agent-backups/` against an operator-configured `rclone` remote, rather than the B2/S3 example in BRD §13.
- **Stage 0 impact:** `.env.example` includes `BACKUP_TARGET=gdrive_rclone` as a placeholder. No `rclone` install, remote creation, or backup script in Stage 0.
- **Stage 10 impact:** Operator creates the `rclone` remote (`rclone config` → `gdrive`) and provides the path. Claude Code installs `rclone` in the orchestrator container, writes `ops/backup.sh` against the `gdrive:` remote, and adds quarterly restore-test plan to `ops/DR.md`.
- **Why this is OK:** off-box requirement (BRD §3) is still satisfied — Google Drive is off-host. Trade-off: rclone's gdrive backend has lower throughput than B2/S3 and has occasional rate-limit retries; backup payload is small (`pg_dump` of three logical DBs), so acceptable.

---

## 4. Operator-experience notes — implementation guidance for later stages

These are not threshold overrides; they are UX/workflow constraints surfaced from operator self-assessment. Stage authors should respect them.

### 4.1 Dashboard reasoning prominence (Stage 6)

The Streamlit dashboard (`dashboard/app.py`) must, at all three HITL surfaces, render the responsible agent's **written reasoning prominently above any metrics tables**. The operator is new to systematic trading and the rationale will be the primary input for approve/reject decisions; metrics like Sharpe / PF / DD will be secondary while operator intuition develops.

| Gate | Primary rationale to display | Secondary (collapsible) |
|---|---|---|
| `paper_gate` | `risk_analyst` rationale (Opus 4.7) | walk-forward IS/OOS Sharpe, profit factor, drawdown, robustness summary |
| `live_gate` | `paper_monitor` rationale (Haiku 4.5) | KS p-value paper-vs-backtest, paper Sharpe deviation, trade count |
| `live_pause_review` | `coordinator` rationale **and** each reviewer's vote (`risk_check`, `performance_check`, `regime_check`) | current drawdown, daily P&L, consecutive losses, regime delta |
| `live_pause_review` *(kill-switch path)* | the `kill_switch_events` row: `reason`, `metrics` snapshot, `action_taken`, `fired_at`. There is **no coordinator rationale** in this path — per BRD §5.6, the out-of-band APScheduler kill switch calls `/api/v1/stop` directly and the orchestrator routes the thread to `live_pause` on next wake without a coordinator vote. | recent trades around `fired_at`, drawdown trajectory leading up to the trigger |

Layout: agent rationale (large, full-width, markdown-rendered) → vote/confidence chips → expandable "Metrics" section below. Auto-refresh metrics; pin rationale block at the top. For the kill-switch path, the `kill_switch_events` row replaces the rationale block but takes the same top-of-card prominence; the dashboard MUST visually flag the gate as kill-switch-originated (distinct colour / icon) so the operator does not confuse it with a coordinator-driven pause.

### 4.2 Live capital ramp

`LIVE_CAPITAL_CAP_USD = 500` (BRD §10) is the v1 ceiling and is not to be raised without:
1. ≥1 strategy completed a full lifecycle (paper graduated to live → live ran ≥30 days without `live_pause`),
2. operator has manually reviewed ≥20 `live_pause` events (same criterion as §4.3 below),
3. BRD §10 thresholds re-tuned for the new cap (drawdown gates may need tightening).

### 4.3 Auto-resume on `live_pause` — explicitly out of scope for v1

Do not implement any auto-resume path in the coordinator agent or the `live_pause` node during Stages 8–11. The criterion to *propose* auto-resume in a future BRD diff is: operator has manually reviewed ≥20 `live_pause` events.

### 4.4 Session protocol for Claude Code (binding on every session)

These rules govern how Claude Code sessions operate on this repository. They are binding on every future session, not just the one that authored this SPEC.

1. **Every session starts by reading `BRD.md` and `SPEC.md` end-to-end** — in that order — before any other action (no `git status`, no globs, no clarifying questions). Confirm the current stage via `git log` and `git tag --list 'stage-*-complete'` afterwards, per BRD §0.
2. **This SPEC supersedes casual chat preferences that conflict with §1–§4.** If a passing request from the operator conflicts with a decision recorded in §1 (Q1–Q8 answers), §2 (threshold overrides), §3 (deviations from BRD), or §4 (operator-experience notes), surface the conflict and ask before acting — do not silently follow the chat.
3. **SPEC changes go in a dedicated commit before any dependent code changes.** When an operator decision causes a SPEC update (new deviation, threshold override, scope change), the SPEC edit is its own commit (`spec: <summary>`). Code that depends on the new SPEC value lands in a separate, later commit. This keeps the SPEC's git history a clean audit trail of operator intent independent of implementation churn.
4. **`BRD.md` still overrides `SPEC.md` where they conflict.** The SPEC fills in BRD-§18 blanks and records BRD-permitted deviations; it does not override BRD §1.1 non-negotiables or BRD §6 LangGraph patterns. If a SPEC edit would contradict the BRD, propose a BRD diff first (per BRD §0).

---

## 5. Stage 0 deliverables (for reference)

Per BRD §13 Stage 0 DoD:

- [x] `BRD.md` (already present, uncommitted)
- [x] `SPEC.md` (this file, awaiting operator approval)
- [ ] `.env.example` — includes `DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `LANGGRAPH_STRICT_MSGPACK=true`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=ai-trading-agent`, `BACKUP_TARGET=gdrive_rclone`, exchange API keys (commented)
- [ ] `.gitignore`
- [ ] `pyproject.toml` with pins from BRD §4
- [ ] `docker-compose.yml` — `postgres:15` (with `pgvector`), `redis:7`, healthchecks, named volumes, all ports bound to `127.0.0.1`
- [ ] `README.md` — short, points at `BRD.md`
- [ ] `LICENSE` — "All rights reserved" private-repo notice (operator: Aryan Agarwal; copyright year: 2026)
- [ ] `tests/unit/__init__.py`, `tests/integration/__init__.py`, and a smoke test (`tests/unit/test_smoke.py`) that asserts (a) the `orchestrator` package imports cleanly and (b) `pyproject.toml` exists at repo root — so `pytest` exits 0 in CI, not 5 (no-tests-collected)
- [ ] `.github/workflows/ci.yml` — ruff + mypy + pytest unit
- [ ] Initial commit (`BRD.md` + `SPEC.md` in same commit) — **do not push** (operator pushes manually)
- [ ] `docker compose up -d` brings up Postgres + Redis healthy
- [ ] `git tag stage-0-complete` — **only after operator explicitly says "tag it"** (see [Git standing rules in operator memory])

---

## 6. Change log

| Date | Change | By |
|---|---|---|
| 2026-05-26 | Initial SPEC — §18 Q1–Q8 locked; BRD §13 backup-target deviation documented (§3.1); operator-experience notes added (§4) | Aryan + Claude Code |
| 2026-05-26 | Added kill-switch row to §4.1 dashboard table; added §4.4 session protocol (binding on every session); added LICENSE + tests skeleton + smoke test to §5 deliverables; aligned §5 with git standing rules (no auto-push, no auto-tag) | Aryan + Claude Code |
| 2026-05-27 | BRD §4 + §13 corrected: dropped phantom `langgraph-store-postgres==0.3.*` pin (no such PyPI distribution); `AsyncPostgresStore` ships in `langgraph-checkpoint-postgres` per the official langgraph 1.x docs. `pyproject.toml` runtime deps updated atomically in the same commit. | Aryan + Claude Code |
| 2026-05-27 | Stage 1b: BRD §6.5 sample pattern hardened to `AsyncExitStack` in `orchestrator/main.py` for correct cleanup on second-context-manager setup failure. Sample in BRD §6.5 remains as-is (illustrative, not normative — no BRD edit). | Aryan + Claude Code |
| 2026-05-27 | Stage 3c: `backtest_runner.py` uses `asyncio.to_thread(subprocess.run)` instead of `asyncio.create_subprocess_exec` due to Windows event-loop incompatibility between psycopg async (needs `SelectorEventLoop`) and native async subprocess (needs `ProactorEventLoop`). Thread pool still permits Stage 4 Send fan-out parallelism since workers are I/O-bound on Freqtrade. | Aryan + Claude Code |
| 2026-05-27 | Convention: all Python commands inside Claude Code sessions use `.venv\Scripts\python.exe -m <module>` explicitly. The bare `python` on PATH is system 3.10 without project deps and will produce misleading errors. The activated venv handles this in interactive operator shells; explicit interpreter path handles it in automated/agent shells. | Aryan + Claude Code |
| 2026-05-27 | Stage 5c finding: the generator's Sonnet structured-output extractor partially default-hugs even with the tightened prompt (commit `6358baa`). In one smoke run, three of eight params encoded the regime thesis; five were textbook/midpoint values. Accepted for v1 — the 5d Opus critic loop is the enforcement point for thesis-aligned parameters. If the critic loop fires for "params don't match thesis" on >50% of strategies after Stage 5 ships, revisit by splitting extraction into two LLM calls (reasoning step on Opus → structured-output on Sonnet). Tracked as v2 candidate. | Aryan + Claude Code |
| 2026-05-27 | Stage 5c finding: the Sonnet researcher's template choice is sensitive to ReAct tool-call ordering — same seed data + same regime produced different template selections across two smoke runs. Acknowledged as a property of tool-use loops; the critic loop and validation subgraph collectively gate on strategy quality, not on researcher reproducibility. | Aryan + Claude Code |
| 2026-05-27 | Stage 6 design decision: FastAPI resume endpoint guarded by a shared-secret OPERATOR_TOKEN (X-Operator-Token header). Streamlit reads the same env var. Rationale: prevents accidental resumes from stale browser tabs, hot-reloads, and replay scenarios on a single-operator deployment. Not for network adversary protection — that remains BRD §15's 127.0.0.1 binding + WireGuard. Cost: ~2 lines of FastAPI + 1 line in Streamlit client. Token rotation = .env edit + restart. | Aryan + Claude Code |
