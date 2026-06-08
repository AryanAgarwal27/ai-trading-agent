"""Unit tests for orchestrator.tools.backtest_runner.

No Docker / no network: the ``_run_subprocess`` seam is monkeypatched to return
a synthetic ``(stdout, stderr, returncode)`` triple, and ``WORKERS_DIR`` is
redirected to ``tmp_path`` so worker dirs land in pytest's scratch space.

Covers:
- BacktestError surfaces the stderr tail in its MESSAGE (debuggability fix —
  str(exc) is what the logged failure_reason / manual-inject _drive() shows).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.tools import backtest_runner as br

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEAN_REVERSION = REPO_ROOT / "strategy_templates" / "mean_reversion_template.py"
FREQAI_REGRESSOR = REPO_ROOT / "strategy_templates" / "freqai_regressor_template.py"


async def test_backtest_error_message_includes_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero Freqtrade exit must put the real stderr error INTO the
    BacktestError message — not just an unread attribute. Before the fix the
    message was only ``exited with code 2`` and the cause was swallowed."""
    sentinel = "freqAI is not enabled. Please enable it in your config to use this strategy."

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        return (b"backtest progress 100%", f"2026-... freqtrade - ERROR - {sentinel}".encode(), 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            MEAN_REVERSION,
            pairs=["BTC/USDT"],
            timeframe="5m",
            timerange="20250101-20250108",
        )

    exc = excinfo.value
    assert exc.returncode == 2
    # The load-bearing assertion: str(exc) now carries Freqtrade's real error,
    # so the logged failure_reason is debuggable.
    assert sentinel in str(exc), f"stderr cause missing from message: {exc}"
    assert "exited with code 2" in str(exc)
    # Attribute still populated for any structured consumer.
    assert sentinel in exc.stderr_tail


async def test_backtest_error_message_falls_back_to_stdout_when_stderr_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Freqtrade prints the error to stdout and stderr is empty, the
    message falls back to the stdout tail rather than showing nothing."""
    sentinel = "OperationalException: some failure on stdout"

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        return (sentinel.encode(), b"", 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            MEAN_REVERSION,
            pairs=["BTC/USDT"],
            timeframe="5m",
            timerange="20250101-20250108",
        )

    assert sentinel in str(excinfo.value)


# ─── FreqAI config injection (commit 2 — the exit-2 fix) ────────────────


def test_build_backtest_config_non_freqai_has_no_freqai_key() -> None:
    """Non-FreqAI path is byte-identical to the Stage 3/4 form: no freqai key."""
    cfg = br._build_backtest_config(
        strategy_class="MeanReversionTemplate",
        pairs=["BTC/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        max_open_trades=4,
    )
    assert "freqai" not in cfg


def test_build_backtest_config_injects_freqai_block_only_as_extra_key() -> None:
    """Passing a freqai block adds exactly the ``freqai`` key — nothing else
    in the config changes versus the non-FreqAI form."""
    base = br._build_backtest_config(
        strategy_class="X", pairs=["BTC/USDT"], timeframe="5m", stake_amount=100.0, max_open_trades=4
    )
    block = {"enabled": True, "identifier": "X"}
    with_freqai = br._build_backtest_config(
        strategy_class="X",
        pairs=["BTC/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        max_open_trades=4,
        freqai=block,
    )
    assert with_freqai["freqai"] is block
    assert {k: v for k, v in with_freqai.items() if k != "freqai"} == base


def test_build_docker_cmd_adds_freqaimodel() -> None:
    cmd = br._build_docker_cmd(
        worker_dir=Path("."),
        timerange="20250101-20250201",
        strategy_class="FreqaiRegressorTemplate",
        freqai_model="LightGBMRegressor",
    )
    assert "--freqaimodel" in cmd
    assert cmd[cmd.index("--freqaimodel") + 1] == "LightGBMRegressor"


def test_build_docker_cmd_omits_freqaimodel_when_none() -> None:
    cmd = br._build_docker_cmd(
        worker_dir=Path("."), timerange="20250101-20250201", strategy_class="MeanReversionTemplate"
    )
    assert "--freqaimodel" not in cmd


async def test_run_backtest_freqai_strategy_injects_config_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through run_backtest (subprocess stubbed): a FreqAI strategy
    gets an enabled freqai block written to config.json AND --freqaimodel on the
    docker cmd — so Freqtrade no longer errors 'freqAI is not enabled'."""
    captured: dict[str, list[str]] = {}

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        captured["cmd"] = cmd
        # Non-zero so run_backtest raises BEFORE artifact parsing (no real run);
        # the worker dir + config.json have already been written by this point.
        return (b"", b"stub: stop before artifact parsing", 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            FREQAI_REGRESSOR,
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="5m",
            timerange="20250101-20250201",
        )

    cmd = captured["cmd"]
    assert "--freqaimodel" in cmd
    assert cmd[cmd.index("--freqaimodel") + 1] == "LightGBMRegressor"

    worker_dir = excinfo.value.worker_dir
    assert worker_dir is not None
    cfg = json.loads((worker_dir / "config.json").read_text())
    assert cfg["freqai"]["enabled"] is True
    assert cfg["freqai"]["feature_parameters"]["include_timeframes"] == ["5m"]
    # Slot value from the template (label_period_candles=12) flows into the config.
    assert cfg["freqai"]["feature_parameters"]["label_period_candles"] == 12


async def test_run_backtest_non_freqai_strategy_has_no_freqai_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pure-TA path writes NO freqai block and adds NO --freqaimodel."""
    captured: dict[str, list[str]] = {}

    async def _fake_run(cmd: list[str], timeout_s: int) -> tuple[bytes, bytes, int]:
        captured["cmd"] = cmd
        return (b"", b"stub", 2)

    monkeypatch.setattr(br, "WORKERS_DIR", tmp_path / "_workers")
    monkeypatch.setattr(br, "_run_subprocess", _fake_run)

    with pytest.raises(br.BacktestError) as excinfo:
        await br.run_backtest(
            MEAN_REVERSION,
            pairs=["BTC/USDT"],
            timeframe="5m",
            timerange="20250101-20250108",
        )

    assert "--freqaimodel" not in captured["cmd"]
    worker_dir = excinfo.value.worker_dir
    assert worker_dir is not None
    cfg = json.loads((worker_dir / "config.json").read_text())
    assert "freqai" not in cfg
