"""Subprocess driver for one-shot Freqtrade backtests (BRD §13 Stage 3, §7.1).

Owns the per-worker isolation contract from `freqtrade/README.md`: every
backtest gets a unique `_workers/<worker_id>/` userdir, mounted into the
Freqtrade Docker container as read-write, with the shared OHLCV data dir
mounted at `/freqtrade/user_data/data` as read-only. No OS symlinks; no
shared caches; ``--cache none`` always (BRD §7.5).

Returns a ``BacktestResult`` (``orchestrator.state``). Stage 4 will wrap this
driver in a Send fan-out (BRD §5.4 + §6.3) so the per-fold walk-forward runs
in parallel. The runner itself stays fold-agnostic: the caller passes
``fold_id`` and ``param_set_id`` so the result carries enough provenance for
the aggregator to bucket results downstream.

Workers under ``_workers/`` are not auto-cleaned. ``.gitignore`` excludes
them from version control, and keeping them on-disk lets the operator
inspect the raw zip, the Freqtrade logs, and rerun a single fold locally.
Call :func:`cleanup_worker` explicitly when you want a worker pruned.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Any

from orchestrator.state import BacktestResult

log = logging.getLogger(__name__)

# Repo root is two parents up from orchestrator/tools/.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FREQTRADE_IMAGE = "freqtradeorg/freqtrade:stable_freqai"
SHARED_DATA_DIR = REPO_ROOT / "freqtrade" / "user_data" / "data"
WORKERS_DIR = REPO_ROOT / "freqtrade" / "user_data" / "_workers"

# Default subprocess timeout for one fold. 30 minutes is generous for a
# 1-week backtest on cached data; Stage 4 walk-forward folds (4 months train
# / 1 month test) will land well inside this. Stage 4 may want to bump this
# per-fold; do so via the `timeout_s` keyword, not by editing the default —
# changing it silently here would let a hung worker eat orchestrator capacity.
DEFAULT_TIMEOUT_S = 30 * 60


class BacktestError(RuntimeError):
    """Raised when a Freqtrade subprocess fails or its output cannot be parsed.

    Carries the docker process exit code (or -1 for timeout) and a tail of
    stderr/stdout so the caller can log a useful diagnostic without having
    to dig through the worker dir.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int = 0,
        stderr_tail: str = "",
        stdout_tail: str = "",
        worker_dir: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail
        self.worker_dir = worker_dir


async def run_backtest(
    strategy_path: Path,
    *,
    pairs: list[str],
    timeframe: str,
    timerange: str,
    fold_id: str = "single",
    param_set_id: str = "default",
    stake_amount: float = 100.0,
    max_open_trades: int = 4,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> BacktestResult:
    """Run one Freqtrade backtest in an isolated Docker worker.

    Parameters
    ----------
    strategy_path
        Host path to the strategy ``.py``. Either a versioned template (Stage
        3 smoke test) or a generator-rendered file under
        ``strategy_templates/_generated/`` (Stage 5+).
    pairs
        Pair whitelist (e.g. ``["BTC/USDT"]``). Must already be present in
        the shared data dir for ``timeframe``.
    timeframe
        Candle interval, e.g. ``"5m"``.
    timerange
        Freqtrade ``--timerange`` string, e.g. ``"20240501-20240508"``.
    fold_id
        Caller-supplied tag (Stage 4 sets this per walk-forward fold).
        Defaults to ``"single"`` for Stage 3 one-shot use.
    param_set_id
        Caller-supplied tag for the param-set under test.
    stake_amount
        Per-trade stake in quote currency.
    max_open_trades
        Concurrent position cap (BRD §10).
    timeout_s
        Subprocess timeout in seconds. Raises ``BacktestError`` if exceeded.

    Returns
    -------
    BacktestResult
        Populated TypedDict from :mod:`orchestrator.state`. ``oos_sharpe`` is
        set to ``0.0`` — single-fold runs have no out-of-sample signal of
        their own. Stage 4's aggregator computes OOS Sharpe by holding out
        a fold at a higher level.
    """

    WORKERS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = uuid.uuid4().hex[:12]
    worker_dir = WORKERS_DIR / worker_id
    worker_dir.mkdir(parents=False, exist_ok=False)

    # Required subdirs for Freqtrade's --userdir layout. Pre-create them so a
    # permission error surfaces before we spend minutes on the backtest.
    (worker_dir / "strategies").mkdir()
    (worker_dir / "backtest_results").mkdir()
    (worker_dir / "logs").mkdir()

    # Copy (not symlink) the strategy file into the worker. Symlinks on the
    # host don't traverse into the container's mount namespace cleanly, and
    # the freqtrade/README.md invariant 1 commits us to bind mounts only.
    strategy_dest = worker_dir / "strategies" / strategy_path.name
    shutil.copy2(strategy_path, strategy_dest)

    strategy_class = _extract_strategy_class_name(strategy_path)

    config = _build_backtest_config(
        strategy_class=strategy_class,
        pairs=pairs,
        timeframe=timeframe,
        stake_amount=stake_amount,
        max_open_trades=max_open_trades,
    )
    config_path = worker_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    cmd = _build_docker_cmd(
        worker_dir=worker_dir,
        timerange=timerange,
        strategy_class=strategy_class,
    )

    log.info(
        "backtest_runner: worker=%s strategy=%s timerange=%s pairs=%s",
        worker_id,
        strategy_class,
        timerange,
        pairs,
    )
    stdout_bytes, stderr_bytes, returncode = await _run_subprocess(cmd, timeout_s)

    if returncode != 0:
        raise BacktestError(
            f"freqtrade backtesting exited with code {returncode}",
            returncode=returncode,
            stderr_tail=_tail(stderr_bytes),
            stdout_tail=_tail(stdout_bytes),
            worker_dir=worker_dir,
        )

    stats_path, raw_zip_path = _locate_result_artifacts(worker_dir)
    stats = _parse_backtest_stats(stats_path, strategy_class=strategy_class)

    return BacktestResult(
        param_set_id=param_set_id,
        pair=",".join(pairs),
        timeframe=timeframe,
        fold_id=fold_id,
        is_sharpe=float(stats["sharpe"]),
        oos_sharpe=0.0,
        profit_factor=float(stats["profit_factor"]),
        max_dd=float(stats["max_drawdown"]),
        trades=int(stats["trades"]),
        raw_zip_path=str(raw_zip_path) if raw_zip_path else str(stats_path),
    )


def cleanup_worker(worker_dir: Path) -> None:
    """Remove a worker dir produced by :func:`run_backtest`.

    Idempotent. The runner leaves workers in place by default so the
    operator can inspect failed runs; the caller decides when to prune.
    Stage 4's aggregator will call this for successful folds once their
    stats are persisted to Postgres.
    """
    if not worker_dir.exists():
        return
    if worker_dir.parent.resolve() != WORKERS_DIR.resolve():
        raise ValueError(f"refusing to remove {worker_dir} — not a child of {WORKERS_DIR}")
    shutil.rmtree(worker_dir)


# ────────────────────────────── internals ──────────────────────────────


def _extract_strategy_class_name(strategy_path: Path) -> str:
    """Find the first top-level ``class`` definition in the strategy file.

    Freqtrade requires the strategy class name to be passed as ``--strategy``;
    parsing the file with AST keeps this in sync with the actual code so the
    template author doesn't have to register the class name in two places.
    """
    source = strategy_path.read_text()
    tree = ast.parse(source, filename=str(strategy_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    raise BacktestError(
        f"no top-level class found in {strategy_path}; cannot derive --strategy name"
    )


def _build_backtest_config(
    *,
    strategy_class: str,
    pairs: list[str],
    timeframe: str,
    stake_amount: float,
    max_open_trades: int,
) -> dict[str, Any]:
    """Construct the minimum Freqtrade config to backtest.

    Kept inline (not loaded from a template file) so that Stage 3's runner
    has no on-disk dependency outside the strategy itself. Stage 6/7 will
    introduce config-paper.json and config-live.json under
    ``user_data/configs/`` for the long-lived spawns; this backtest config
    is per-worker and deliberately ephemeral.
    """
    return {
        "max_open_trades": max_open_trades,
        "stake_currency": "USDT",
        "stake_amount": stake_amount,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": timeframe,
        "trading_mode": "spot",  # BRD §1: spot-only
        "dry_run": True,
        "cancel_open_orders_on_exit": False,
        "unfilledtimeout": {"entry": 10, "exit": 10},
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "price_last_balance": 0.0,
            "check_depth_of_market": {"enabled": False},
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "price_last_balance": 0.0,
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "pair_whitelist": pairs,
            "pair_blacklist": [],
            "ccxt_config": {"enableRateLimit": True},
            "ccxt_async_config": {"enableRateLimit": True},
        },
        "pairlists": [{"method": "StaticPairList"}],
        "dataformat_ohlcv": "feather",
        "strategy": strategy_class,
    }


def _build_docker_cmd(
    *,
    worker_dir: Path,
    timerange: str,
    strategy_class: str,
) -> list[str]:
    """Compose the ``docker run`` argv per the bind-mount invariant.

    The worker dir is mounted **read-write** at the container's user_data
    path; the shared data dir is mounted **read-only** at the canonical
    Freqtrade data path. ``--cache none`` is unconditional (BRD §7.5).

    On Windows, native Python's ``Path.resolve()`` returns a backslash form
    (``C:\\dev\\...``) which Docker Desktop accepts. The Git Bash MSYS
    ``/c/dev/...`` form would NOT be produced here because this runner is
    invoked from the orchestrator's Python process, not from a shell.
    """
    worker_host = str(worker_dir.resolve())
    data_host = str(SHARED_DATA_DIR.resolve())
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{worker_host}:/freqtrade/user_data",
        "-v",
        f"{data_host}:/freqtrade/user_data/data:ro",
        FREQTRADE_IMAGE,
        "backtesting",
        "--userdir",
        "/freqtrade/user_data",
        "--config",
        "/freqtrade/user_data/config.json",
        "--strategy",
        strategy_class,
        "--timerange",
        timerange,
        "--cache",
        "none",
        "--export",
        "trades",
    ]


async def _run_subprocess(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
    """Run ``cmd`` in a thread, capture stdout/stderr, enforce ``timeout_s``.

    Why not ``asyncio.create_subprocess_exec``? On Windows, async subprocess
    transport requires ``ProactorEventLoop``, but psycopg's async mode
    refuses Proactor and demands ``SelectorEventLoop`` — and the orchestrator
    needs both in the same process (Postgres for state, docker for backtests).

    ``asyncio.to_thread`` runs the sync ``subprocess.run`` on the default
    thread executor; the event loop stays free to schedule other Sends.
    Stage 4 fan-out parallelism is preserved because each parallel worker
    awaits its own ``to_thread`` call and ``ThreadPoolExecutor`` schedules
    them concurrently up to its ``max_workers`` (default = ``min(32, cpu+4)``,
    enough for the BRD-§5.4 6-fold walk-forward × a few param sets).
    """
    return await asyncio.to_thread(_run_subprocess_sync, cmd, timeout_s)


def _run_subprocess_sync(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BacktestError(
            f"freqtrade backtesting exceeded timeout of {timeout_s}s",
            returncode=-1,
            stdout_tail=_tail(exc.stdout or b""),
            stderr_tail=_tail(exc.stderr or b""),
        ) from exc
    return result.stdout, result.stderr, result.returncode


def _locate_result_artifacts(worker_dir: Path) -> tuple[Path, Path | None]:
    """Find the most recent backtest result artifacts in ``worker_dir``.

    Freqtrade 2026.x writes a ``backtest-result-<ts>.json`` and a
    ``backtest-result-<ts>.zip`` under ``backtest_results/``. We prefer the
    ``.json`` for stats parsing — it has the full top-level summary without
    needing to crack open the zip. The zip is recorded in the
    ``BacktestResult.raw_zip_path`` for downstream retention.
    """
    results_dir = worker_dir / "backtest_results"
    if not results_dir.exists():
        raise BacktestError(
            f"backtest_results/ missing under worker {worker_dir.name}",
            worker_dir=worker_dir,
        )

    json_candidates = sorted(
        (
            p
            for p in results_dir.glob("backtest-result-*.json")
            if not p.name.endswith(".meta.json")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_candidates:
        # Fall back: try zip-only output.
        zip_candidates = sorted(
            results_dir.glob("backtest-result-*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not zip_candidates:
            raise BacktestError(
                f"no backtest-result-* artifacts found under {results_dir}",
                worker_dir=worker_dir,
            )
        # Extract the embedded JSON in-place.
        zip_path = zip_candidates[0]
        with zipfile.ZipFile(zip_path) as zf:
            json_name = next(
                (n for n in zf.namelist() if n.endswith(".json") and not n.endswith(".meta.json")),
                None,
            )
            if json_name is None:
                raise BacktestError(
                    f"backtest zip {zip_path.name} has no JSON entry",
                    worker_dir=worker_dir,
                )
            zf.extract(json_name, results_dir)
        return results_dir / json_name, zip_path

    stats_path = json_candidates[0]
    sibling_zip = stats_path.with_suffix(".zip")
    return stats_path, sibling_zip if sibling_zip.exists() else None


def _parse_backtest_stats(stats_path: Path, *, strategy_class: str) -> dict[str, float | int]:
    """Extract the headline stats the BacktestResult needs from the JSON.

    Freqtrade's backtest JSON has a top-level ``strategy`` map keyed by
    strategy class name, plus a ``strategy_comparison`` array. We read from
    the ``strategy[strategy_class]`` map because it has all the per-strategy
    fields we need without ambiguity.
    """
    data = json.loads(stats_path.read_text())
    strategies = data.get("strategy")
    if not isinstance(strategies, dict) or strategy_class not in strategies:
        # Single-strategy backtests sometimes flatten the structure; handle that.
        candidate = (
            next(iter(strategies.values())) if isinstance(strategies, dict) and strategies else None
        )
        if candidate is None:
            raise BacktestError(
                f"backtest JSON {stats_path.name} has no 'strategy' entry for {strategy_class}",
            )
        stats = candidate
    else:
        stats = strategies[strategy_class]

    # Freqtrade reports these under consistent keys; default to 0 when missing
    # rather than KeyError so a degenerate backtest (zero trades) still
    # produces a BacktestResult the gate logic can route to archive.
    return {
        "sharpe": float(stats.get("sharpe", 0.0) or 0.0),
        "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
        "max_drawdown": float(
            stats.get("max_relative_drawdown", stats.get("max_drawdown_account", 0.0)) or 0.0
        ),
        "trades": int(stats.get("total_trades", 0) or 0),
    }


def _tail(b: bytes, max_chars: int = 1500) -> str:
    """Last ``max_chars`` characters of decoded ``b``, for diagnostics."""
    text = b.decode("utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text
