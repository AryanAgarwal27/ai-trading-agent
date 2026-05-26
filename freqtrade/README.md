# freqtrade/

Freqtrade execution layer for the trading agent. Per BRD §1.1 rule 2, only Freqtrade
touches the exchange; the orchestrator starts, stops, and queries Freqtrade
instances via the REST API. See [BRD.md](../BRD.md) §7 for the integration contract.

---

## Image and version

This project pins **`freqtradeorg/freqtrade:stable_freqai`** (BRD §4). The
running image must report `Freqtrade Version: 2026.4` for stage 3 onward:

```
docker run --rm freqtradeorg/freqtrade:stable_freqai --version
```

The `stable_freqai` tag includes the FreqAI ML extras used by the classifier and
regressor templates. Use `stable_freqairl` only if/when the optional v2 RL
template is implemented (BRD §8.1).

---

## Directory layout

```
freqtrade/
├── README.md            (this file)
└── user_data/
    ├── data/            # SHARED OHLCV cache, mounted read-only into every worker
    ├── strategies/      # symlinks to strategy_templates/_generated/<id>.py
    ├── configs/         # base config templates (config-base.json, etc.)
    └── _workers/        # ephemeral per-worker userdirs, one per Send fan-out
```

`data/` and `_workers/` are in `.gitignore` (per-host artefacts). `strategies/`
and `configs/` are tracked via `.gitkeep`; the JSON pair caches that Freqtrade
writes into `strategies/` are ignored via the `strategies/*.json` rule.

---

## Layout invariants

These are not preferences. Each maps to a real failure mode; violations cause
silent data corruption or trade losses.

### Invariant 1 — Per-worker isolated `--userdir` via Docker bind mounts (BRD §7.1)

**Every Freqtrade invocation MUST have its own `--userdir`.** Parallel backtests
sharing a userdir contaminate `backtest_results/` and the strategy-pair JSON
caches, producing stale or cross-mixed results that look plausible but are
wrong. This is BRD §7.1 verbatim and Stage 4 walk-forward fan-out depends on it.

**Implementation: Docker bind mounts, not OS symlinks.**

Each worker container is launched with the shared OHLCV cache mounted
**read-only** at the canonical Freqtrade data path:

```bash
docker run --rm \
  -v "<repo>/freqtrade/user_data/_workers/<worker_id>:/freqtrade/user_data" \
  -v "<repo>/freqtrade/user_data/data:/freqtrade/user_data/data:ro" \
  freqtradeorg/freqtrade:stable_freqai \
  backtesting --userdir /freqtrade/user_data \
              --strategy GenStrategy_<id> \
              --timerange <fold> \
              --cache none
```

The bind-mount pattern was picked over OS symlinks because:

1. **Cross-platform.** Windows requires admin rights or Developer Mode for
   `mklink`; macOS/Linux symlinks work but break inside containers when the
   target path is not also mounted. Bind mounts are a single, portable
   mechanism that behaves identically on Windows (this dev host), Linux, and
   macOS.
2. **Read-only enforcement.** `:ro` on the data mount guarantees a runaway
   worker cannot corrupt the shared OHLCV cache. An OS symlink offers no such
   guarantee — a `download-data` call inside a worker would write back through
   the symlink into shared state.
3. **No symlink dereference inside the container.** Freqtrade resolves
   `--userdir` paths via `pathlib.Path.resolve()`, which follows symlinks. A
   symlinked `data/` would resolve outside the container's mount namespace and
   fail.
4. **Atomic worker teardown.** `_workers/<worker_id>/` can be `rm -rf`-ed after
   each fan-out leg without touching the shared cache, because nothing inside
   the worker dir is a symlink to shared data.

`--cache none` is also mandatory (BRD §7.5): cache reuse with a freshly
generated strategy is a stale-result hazard.

### Invariant 2 — Do not run this project from a OneDrive- or iCloud-synced directory

This repository was migrated from `~/OneDrive/Desktop/Aryan/ai-trading-agent/`
to `C:\dev\ai-trading-agent\` precisely to escape OneDrive sync interference.

**The failure mode:** Freqtrade writes continuously into `user_data/_workers/`
during backtests (per-worker logs, pair-JSON caches, `backtest_results/` zips)
and into `user_data/data/` during `download-data` (multi-megabyte feather
files). OneDrive treats every file change as a sync event:

- Upload throttle: file-locking conflicts during writes → Freqtrade I/O errors.
- "File in use" dialogs from the OneDrive client when zips are being parsed.
- Sync of `_workers/` artefacts that are explicitly per-host ephemeral, wasting
  bandwidth and quota; the OneDrive client will also re-download them on other
  machines.
- Subtle race: OneDrive's hash-then-upload pass can hold a read lock while
  Freqtrade still writes, producing truncated `.feather` files that read
  successfully but contain partial data.

**Rule.** Clone this repo into a path outside any cloud-sync root. On Windows,
`C:\dev\` or any directory not under `%OneDriveConsumer%`, `%OneDriveCommercial%`,
`%iCloudDrive%`, Dropbox, or Google Drive. On macOS, avoid `~/Library/Mobile
Documents/`, `~/iCloud Drive/`, and `~/Dropbox/`.

If you must keep this repo on a synced drive (you don't), at minimum exclude
`freqtrade/user_data/` and `.venv/` from the sync client's selective-sync list
*before* the first `download-data` run.

---

## First-time setup

The Stage 3 commit creates the directory tree and pulls the image. To populate
the shared OHLCV cache, use the platform-appropriate command below.

**Linux / macOS / WSL2 native shell:**

```bash
docker run --rm \
  -v "$(pwd)/freqtrade/user_data:/freqtrade/user_data" \
  freqtradeorg/freqtrade:stable_freqai \
  download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --timeframes 5m 15m 1h \
  --days 730
```

**Windows Git Bash / MSYS:**

```bash
docker run --rm \
  -v "$(pwd -W)/freqtrade/user_data:/freqtrade/user_data" \
  freqtradeorg/freqtrade:stable_freqai \
  download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --timeframes 5m 15m 1h \
  --days 730
```

Pairs and timeframes match [SPEC.md](../SPEC.md) §1 Q2 (Binance v1 pair list).
The 730-day window gives the anchored 6-fold walk-forward (BRD §5.4) full data.

### Windows path-translation gotcha

On Windows under Git Bash / MSYS, plain `$(pwd)` returns the MSYS-style path
`/c/dev/ai-trading-agent`. Docker Desktop happily accepts this and the
`docker run` command **succeeds silently**, but the data lands in an
inaccessible WSL2-internal mount instead of the host path — leaving
`freqtrade/user_data/data/` empty on the host filesystem. The download bar will
even hit `4/4 100%` while writing into the void.

Use `$(pwd -W)` (Windows-style `C:/dev/ai-trading-agent`) on Git Bash / MSYS.
PowerShell users should use `${PWD}` directly, which already produces a
Windows path. `backtest_runner.py` resolves the host path via
`pathlib.Path.resolve()` and a Windows-aware helper to remove this footgun for
orchestrated runs; the manual `docker run` lines above are the only place the
caller needs to think about it.

---

## What lives where

| Path | Owned by | Purpose |
|---|---|---|
| `user_data/data/<exchange>/` | Freqtrade `download-data` | shared OHLCV feathers, read-only mount into every worker |
| `user_data/strategies/<Name>.py` | orchestrator (symlink) | points at `strategy_templates/_generated/<id>.py` |
| `user_data/configs/config-*.json` | orchestrator | base config templates (paper, live, backtest) |
| `user_data/_workers/<worker_id>/` | `backtest_runner.py` | ephemeral, one per Send fan-out leg; `rm -rf`-ed after parse |
