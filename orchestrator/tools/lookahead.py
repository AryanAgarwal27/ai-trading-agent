"""Subprocess driver for ``freqtrade lookahead-analysis`` (BRD §5.3, §8 rule 5).

Runs the rendered strategy file through Freqtrade's lookahead-bias
detector before the strategy ever reaches the validation subgraph.
A failure here is a hard archive route — the LLM-generated strategy
silently uses future data in an indicator, so any backtest the
validation subgraph runs would overstate live performance by an
unknowable amount.

Same isolation pattern as :mod:`orchestrator.tools.backtest_runner`:
each invocation gets a unique ``_workers/<id>/`` userdir under the
host repo (mounted into the Freqtrade Docker container as read-write),
with the shared OHLCV data dir mounted at
``/freqtrade/user_data/data`` as read-only. Cleanup is explicit.

Why a separate module from ``backtest_runner.py``: lookahead-analysis
has a different CLI surface (no ``--export``, no zip artifact to
parse, exit code IS the signal), and bundling them would force a
``mode=`` parameter that would obscure both. Splitting also lets the
lookahead gate stub the runner separately in tests.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, TypedDict

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FREQTRADE_IMAGE = "freqtradeorg/freqtrade:stable_freqai"
SHARED_DATA_DIR = REPO_ROOT / "freqtrade" / "user_data" / "data"
WORKERS_DIR = REPO_ROOT / "freqtrade" / "user_data" / "_workers"

# Default subprocess timeout for one lookahead run. Lookahead-analysis
# is a fast sweep over a small data window; 5 minutes is generous and
# anything beyond that is a hang.
DEFAULT_TIMEOUT_S = 5 * 60


class LookaheadResult(TypedDict):
    """Structured outcome of one ``freqtrade lookahead-analysis`` run.

    ``passed`` is the binary verdict the gate routes on; ``details``
    carries the operator-readable explanation (with the stderr tail on
    failure paths so the operator can debug without digging into the
    worker dir).
    """

    passed: bool
    details: str
    returncode: int
    worker_dir: str  # absolute path; preserved for operator inspection
    stderr_tail: str
    stdout_tail: str


async def run_lookahead_analysis(
    strategy_path: Path,
    *,
    pairs: list[str],
    timeframe: str,
    timerange: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> LookaheadResult:
    """Run ``freqtrade lookahead-analysis`` on one strategy file.

    Parameters
    ----------
    strategy_path
        Host path to the rendered strategy ``.py``, typically under
        ``strategy_templates/_generated/``.
    pairs
        Pair whitelist (e.g. ``["BTC/USDT"]``). Must be present in
        the shared data dir for ``timeframe``.
    timeframe
        Candle interval, e.g. ``"5m"``.
    timerange
        Freqtrade ``--timerange`` string, e.g. ``"20240501-20240508"``.
        A 1-week recent window is enough; lookahead-analysis doesn't
        need the full walk-forward span.
    timeout_s
        Subprocess timeout in seconds. Defaults to 5 minutes.

    Returns
    -------
    LookaheadResult
        ``passed=True`` means no look-ahead bias detected (Freqtrade
        returncode 0 AND no "Found" markers in stdout). ``passed=False``
        with ``details`` carrying the Freqtrade-emitted reason.
    """
    WORKERS_DIR.mkdir(parents=True, exist_ok=True)
    worker_id = f"la_{uuid.uuid4().hex[:10]}"
    worker_dir = WORKERS_DIR / worker_id
    worker_dir.mkdir(parents=False, exist_ok=False)

    (worker_dir / "strategies").mkdir()
    (worker_dir / "logs").mkdir()
    (worker_dir / "backtest_results").mkdir()

    strategy_dest = worker_dir / "strategies" / strategy_path.name
    shutil.copy2(strategy_path, strategy_dest)

    strategy_class = _extract_strategy_class_name(strategy_path)
    config_path = worker_dir / "config.json"
    _write_minimal_config(config_path, pairs=pairs, timeframe=timeframe)

    cmd = _build_docker_cmd(
        worker_dir=worker_dir,
        strategy_class=strategy_class,
        timerange=timerange,
    )

    log.info(
        "lookahead-analysis: strategy=%s pairs=%s timerange=%s worker=%s",
        strategy_path.name,
        pairs,
        timerange,
        worker_dir,
    )

    # Same Windows-event-loop reasoning as backtest_runner: psycopg async
    # needs SelectorEventLoop; native asyncio.create_subprocess_exec
    # needs ProactorEventLoop on Windows. asyncio.to_thread bridges by
    # running the blocking subprocess.run in a thread pool, leaving the
    # main event loop selector-compatible.
    try:
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return LookaheadResult(
            passed=False,
            details=(
                f"lookahead-analysis timed out after {timeout_s}s "
                f"(strategy={strategy_path.name})"
            ),
            returncode=-1,
            worker_dir=str(worker_dir),
            stderr_tail=(exc.stderr or b"").decode("utf-8", errors="replace")[-1000:],
            stdout_tail=(exc.stdout or b"").decode("utf-8", errors="replace")[-1000:],
        )

    stderr_tail = (completed.stderr or "")[-1000:]
    stdout_tail = (completed.stdout or "")[-1000:]

    if completed.returncode != 0:
        return LookaheadResult(
            passed=False,
            details=(
                f"freqtrade lookahead-analysis exited {completed.returncode}; "
                f"likely a parse error or missing data. stderr tail: "
                f"{stderr_tail}"
            ),
            returncode=completed.returncode,
            worker_dir=str(worker_dir),
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )

    # Freqtrade's lookahead-analysis exits 0 in BOTH the no-leakage and
    # leakage-detected cases; the verdict is in stdout. Look for the
    # canonical markers. (Conservative: presence of "Found" or
    # "leakage" in the lower-cased stdout → fail.)
    stdout_lower = (completed.stdout or "").lower()
    leakage_markers = ("found a problem", "look-ahead bias", "lookahead bias")
    if any(marker in stdout_lower for marker in leakage_markers):
        return LookaheadResult(
            passed=False,
            details=(
                "lookahead-analysis flagged a problem. stdout tail: "
                f"{stdout_tail}"
            ),
            returncode=0,
            worker_dir=str(worker_dir),
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )

    return LookaheadResult(
        passed=True,
        details="no look-ahead bias detected",
        returncode=0,
        worker_dir=str(worker_dir),
        stderr_tail=stderr_tail,
        stdout_tail=stdout_tail,
    )


def _build_docker_cmd(
    *,
    worker_dir: Path,
    strategy_class: str,
    timerange: str,
) -> list[str]:
    """Build the docker run command for lookahead-analysis.

    Mirrors backtest_runner's docker invocation shape, swapping
    ``backtesting`` for ``lookahead-analysis`` and dropping
    backtest-specific flags (``--cache``, ``--export``).
    """
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{worker_dir.resolve()}:/freqtrade/user_data",
        "-v",
        f"{SHARED_DATA_DIR.resolve()}:/freqtrade/user_data/data:ro",
        FREQTRADE_IMAGE,
        "lookahead-analysis",
        "--userdir",
        "/freqtrade/user_data",
        "--config",
        "/freqtrade/user_data/config.json",
        "--strategy",
        strategy_class,
        "--timerange",
        timerange,
    ]


def _write_minimal_config(
    config_path: Path,
    *,
    pairs: list[str],
    timeframe: str,
) -> None:
    """Write a minimal Freqtrade config.json sufficient for lookahead-analysis."""
    import json

    config = {
        "max_open_trades": 4,
        "stake_currency": "USDT",
        "stake_amount": 100.0,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "cancel_open_orders_on_exit": False,
        "timeframe": timeframe,
        "trading_mode": "spot",
        "margin_mode": "",
        "unfilledtimeout": {"entry": 10, "exit": 10},
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": False,
            "order_book_top": 1,
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": pairs,
            "pair_blacklist": [],
        },
        "pairlists": [{"method": "StaticPairList"}],
        "internals": {"process_throttle_secs": 5},
        "bot_name": "lookahead_probe",
        "initial_state": "running",
        "force_entry_enable": False,
        "log_level": "INFO",
        "user_data_dir": "/freqtrade/user_data",
        "datadir": "/freqtrade/user_data/data",
    }
    config_path.write_text(json.dumps(config, indent=2))


def _extract_strategy_class_name(strategy_path: Path) -> str:
    """Find the IStrategy subclass name in the strategy file via AST.

    Same algorithm as backtest_runner; duplicated here to keep the two
    modules independent (avoid creating a cross-import that would
    couple their lifecycle).
    """
    import ast

    source = strategy_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(strategy_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "IStrategy":
                    return node.name
                if isinstance(base, ast.Attribute) and base.attr == "IStrategy":
                    return node.name
    raise ValueError(
        f"Could not find an IStrategy subclass in {strategy_path}"
    )


def cleanup_worker(worker_dir_str: str) -> None:
    """Remove a lookahead worker dir. Best-effort; missing dirs are silent."""
    p = Path(worker_dir_str)
    if not p.exists():
        return
    try:
        shutil.rmtree(p)
    except OSError as exc:
        log.warning("lookahead.cleanup_worker(%s) failed: %s", p, exc)


# Default lookahead runner used by the gate node when none is injected.
# Exposed as a module-level name (not just a function reference) so
# unit tests can monkey-patch it without contorting the gate's
# injection seam.
async def _default_lookahead_runner(
    strategy_path: Path,
    *,
    pairs: list[str],
    timeframe: str,
    timerange: str,
) -> LookaheadResult:
    """Default real-Freqtrade runner. Identity-equal to
    :func:`run_lookahead_analysis` — separate name lets the gate's
    injection seam swap it for a stub in tests without monkey-patching
    a public function."""
    return await run_lookahead_analysis(
        strategy_path,
        pairs=pairs,
        timeframe=timeframe,
        timerange=timerange,
    )


# Type alias for the gate's injection seam.
LookaheadRunner = Any  # Callable[..., Awaitable[LookaheadResult]] — kept loose
                       # to avoid the same closure-async generic noise as the
                       # validation subgraph's BacktestWorkerFn.
