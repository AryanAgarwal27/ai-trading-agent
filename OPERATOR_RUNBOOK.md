# OPERATOR RUNBOOK — ai-trading-agent

> **Purpose:** hand this to a fresh Claude session (this is an incognito chat with no memory) so it can help me run and operate the app I built. It captures the current state, how to start/use the system, the gotchas I hit, and what to do next.
>
> **Operator:** Aryan | **Box:** Windows 11, PowerShell + Git Bash | **Repo:** `C:\dev\ai-trading-agent`

---

## 0. TL;DR for a new Claude session

- The app is an **autonomous crypto trading agent**: LangGraph orchestrates LLM agents that research → generate → backtest → paper-trade → (with my approval) live-trade strategies. Freqtrade is the execution engine.
- **The build is 100% complete.** All 11 stages are done and git-tagged (`stage-0-complete` … `stage-11-complete`). CI is green. Do **not** start new build stages.
- **Read `BRD.md` then `SPEC.md` end-to-end first** (binding session protocol, SPEC §4.4). `BRD.md` is the contract; `SPEC.md` records my operator decisions. They override casual chat.
- **Current phase = running it / strategy-quality research**, NOT engineering. The machine works; the strategies it generates don't make money yet. That's the open problem.
- **Keep answers short.** I prefer step-by-step commands, minimal explanation.

---

## 1. What this app does (plain version)

- A **supervisor** AI reads the market regime and spawns research strategies (up to 4 at once).
- Each strategy goes through a **gauntlet**: researcher (Sonnet) proposes params → generator fills a template → critic (Opus) attacks it → look-ahead gate → **6-fold walk-forward backtest** → quality gate.
- If it passes the gate (Sharpe ≥ 1.5, profit factor ≥ 1.5), it goes to **30-day paper trading** (dry-run, no real money).
- After 30 days, **I manually approve** it to **live** (real money, capped at $500, one live strategy at a time).
- Out-of-band **kill switch** (every 5 min) halts a live strategy on a 12% drawdown.

**Templates that ship:** `mean_reversion_template`, `freqai_classifier_template`, `freqai_regressor_template` (the last two are ML / FreqAI). Pairs: BTC/ETH/SOL/BNB-USDT only.

---

## 2. How to start the system

All from `C:\dev\ai-trading-agent`.

### 2.1 Bring up infra (Postgres + Redis)
```powershell
docker compose up -d
docker compose ps          # both ait-postgres + ait-redis should be healthy
```
Note: Postgres is mapped to host port **5433** (→ container 5432).

### 2.2 Start the app (Windows-specific launch — IMPORTANT)
```powershell
uv run python -m orchestrator.main
```
- **Must** use `python -m orchestrator.main`, NOT bare `uvicorn`. On Windows, uvicorn forces the ProactorEventLoop which psycopg can't use; the `__main__` entrypoint installs the SelectorEventLoop policy first. (This was a real bug we fixed — D-14.)
- Wait for `Application startup complete`. **Leave this window running** — it streams all the JSON logs. Call it "window 1".

### 2.3 Get the operator token (needed for every API call)
In a **second** PowerShell window:
```powershell
$tok = (Get-Content .env | Where-Object { $_ -match '^OPERATOR_TOKEN=' }) -replace 'OPERATOR_TOKEN=',''
$h = @{ "X-Operator-Token" = $tok }
echo $tok          # confirm not blank
```
> The token clears if you reopen the window — re-run the two lines above each new session.

---

## 3. How to run a strategy (spawn → gauntlet)

### 3.1 Fire the supervisor (it researches + spawns strategies)
```powershell
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/supervisor/run" -Headers $h -Body '{"dry_run": false}' -ContentType 'application/json' | ConvertTo-Json -Depth 6
```
- `dry_run: false` = actually spawns strategies. `dry_run: true` = supervisor reasons but spawns nothing (safe test).
- **CRITICAL — spawn ONE AT A TIME.** Fire once, WAIT for the strategy to finish (watch window 1 for `archive` or `gate_backtest`), THEN fire again. Spawning in batches blows the Anthropic **30k-token/min rate limit** and kills strategies mid-research (see §5).

### 3.2 Watch progress (window 1 logs)
Each strategy streams: `load_context → researcher → generator → critic → revise_or_proceed → lookahead_gate → plan_backtests → backtest_worker (×6 folds) → aggregate_results → gate_backtest → archive` (fail) or → paper (pass).

Look at the `generator` line for the **template**; look at `archive` for the **failure_reason**.

### 3.3 List all strategies + their stage
```powershell
Invoke-RestMethod "http://127.0.0.1:8000/threads" -Headers $h | ConvertTo-Json -Depth 4
```
Stages: `research` (in progress or stuck) / `archived` (failed gate or cleared) / `paper` / `live`.

---

## 4. Observability

- **Logs:** window 1 streams structured JSON (`strategy_id`, `thread_id`, `run_id`, `node`, `event`).
- **LangSmith traces:** smith.langchain.com → project `ai-trading-agent`. Full ReAct loop of every agent run. (Tracing on if `LANGSMITH_TRACING=true` + key in `.env`.)
- **Prometheus:** `GET http://127.0.0.1:8000/metrics` (kill-switch fires, strategies-by-stage, supervisor runs, per-container Freqtrade profit/drawdown). Grafana dashboards are committed in `ops/grafana/` (definitions only — not running; stand up at deploy time per `ops/observability.md`).

---

## 5. KNOWN GOTCHAS (things I hit — save the new session time)

### 5.1 Anthropic rate limit (the big one)
- My tier = **30,000 input tokens/minute** on `claude-sonnet-4-6`. Batch spawning (2-3 strategies at once) exceeds it → strategies die mid-research with a **429** and get stuck at "research".
- **Fix:** spawn ONE at a time, wait for each to finish. OR raise the limit: console.anthropic.com → Settings → Billing → buy credits (auto-promotes tiers). Check limit at Settings → Limits.

### 5.2 Stuck "research" threads
- A strategy that died on a 429 mid-research parks forever at "research" and eats a capacity slot (cap = 4). It is NOT recoverable / resumable — it never produced a strategy to continue.
- **Clear them** (frees capacity):
```powershell
Get-Content .env | Where-Object { $_ -match '^POSTGRES_' } | ForEach-Object { $p=$_-split'=',2; Set-Item "env:$($p[0])" $p[1] }
docker exec -e PGPASSWORD="$env:POSTGRES_PASSWORD" ait-postgres psql -h 127.0.0.1 -p 5432 -U "$env:POSTGRES_USER" -d app -c "UPDATE strategy_registry SET stage='archived' WHERE stage='research';"
```
(Blunt manual override of the registry view — fine for local dev. Run when capacity is full of dead threads.)

### 5.3 mypy lies on Windows
- Local Windows `uv run mypy` can report a FALSE 0; the gate is BLOCKING. CI (Linux) is the only authority. To check locally CI-equivalent: Docker `python:3.12-slim` + `uv sync --locked` + fresh `MYPY_CACHE_DIR` (a stale Windows `.mypy_cache` causes phantom errors). (D-17.)

### 5.4 Runner note
- Use `uv run python -m pytest`, NOT bare `uv run pytest` (the latter can resolve the wrong interpreter, missing deps).

### 5.5 LangSmith upload timeouts
- Occasional `Read timed out ... api.smith.langchain.com` = harmless network blip, traces just don't upload that run. Ignore.

### 5.6 rclone on Git Bash
- `ops/backup.sh` upload needs rclone on PATH; winget-installed rclone isn't on Git Bash's PATH by default. Add it: `export PATH="$PATH:/c/Users/<you>/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_*/rclone-*-windows-amd64"`.

---

## 6. CURRENT STATE (as of last session, 2026-06-07)

- **Build:** complete, all stages tagged, CI green, pushed.
- **Paper real-spawn smoke:** PASSED (real Freqtrade container boots against Binance paper keys). Live-keyed smoke NOT done.
- **Live Binance keys:** created, in `.env` as `BINANCE_LIVE_*`. Verified: `enableWithdrawals=False` ✓, `ipRestrict=True` ✓, spot trading on. BUT they're on my **main account**, not an isolated $500 subaccount — must move to a subaccount before any live trading.
- **Strategy runs so far:** ~6 spawned. Result: **0 passed.** The 4 that completed all FAILED the backtest gate with badly negative Sharpe (-15 to -26) — all `mean_reversion`. The 2 FreqAI ones (regressor + classifier) **never completed** — they died on the 429 rate limit. **So FreqAI is still UNTESTED.**
- **Key finding:** the system works perfectly (correctly rejects bad strategies). But mean-reversion as generated has no edge, and FreqAI hasn't been proven to run.

---

## 7. WHAT TO DO NEXT (open work — this is RESEARCH, not building)

In priority order:

1. **Test whether FreqAI works.** Spawn ONE at a time (avoid the rate limit) until the supervisor picks `freqai_regressor_template` or `freqai_classifier_template`, and let it run to completion. Watch window 1:
   - reaches `gate_backtest`/`archive` = FreqAI works (just failed the gate, fine).
   - errors on model training / FreqAI config = a real ML setup issue to fix.
   - 429 = still spawning too fast.

2. **Improve strategy quality** (why nothing passes the gate):
   - **Generator param default-hugging** (SPEC §5c): the generator fills templates with textbook/midpoint values instead of regime-fitted ones → strategies underperform. Known v1 weakness.
   - **Re-tune gate thresholds** (SPEC §2 + BRD §10): Sharpe ≥1.5 / PF ≥1.5 on 6-fold walk-forward may be too strict for what a template-filler produces. SPEC §2 explicitly allows re-tuning after the first ~10 strategies (`orchestrator/gates/thresholds.py`). My call, recorded as a SPEC change.

3. **Before any LIVE trading** (weeks away, only after a strategy completes 30-day paper):
   - Move live key to a **Binance subaccount funded with only ~$500**.
   - Run the **live-keyed smoke** (`AIT_RUN_REAL_LIVE_SPAWN_TESTS=1`) — places REAL orders, so confirm withdrawals disabled + small balance first.

4. **Raise Anthropic rate limit** (buy credits) so I can spawn faster / run FreqAI reliably.

---

## 8. Live-cap design (my decisions — don't let a new session change these)

- **1 live strategy at a time** ($500 all-in, `MAX_CONCURRENT_LIVE_STRATEGIES=1`, SPEC §1 Q3).
- **D-9 = block-the-gate:** when a 2nd strategy is approved-ready but the live slot is full, the system does NOT offer approval; it waits until I free the slot. No auto-promote, no queue.
- **v2 idea (parked, not built):** a basket of pre-proven strategies + a regime-router that rotates live capital into whichever fits current conditions. Only after a full live lifecycle + 20 reviewed pauses (SPEC §4.2). NOT v1.

---

## 9. Shut down cleanly

```powershell
# window 1 (app): Ctrl+C
docker compose stop          # or: docker compose down  (keeps the named volume)
```
Do NOT `docker compose down -v` unless you want to wipe the DB (then re-apply migrations per README).

---

## 10. Open deferred items (tracked in DEFERRED.md, none block use)

- D-3 (daily-loss calendar-day approximation), D-8 (supervisor persistence, dormant-by-design), D-17 (mypy tooling wrapper), D-18 (route remaining secrets through secrets.py — when a real sops/1Password backend lands), D-19 (per-DB Postgres roles — infra). All low-risk, landing-stated.

---

*Hand sections 0–3 to a new Claude session to get running fast. Sections 5–7 are the important context for what's actually left to do.*
