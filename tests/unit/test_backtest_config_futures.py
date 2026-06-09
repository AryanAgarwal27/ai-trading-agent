"""Stage 13 Phase 1 (BRD §22.1): per-strategy futures backtest config.

Pins two contracts:
  1. A short-capable strategy (``can_short = True``) renders a FUTURES backtest
     config — ``trading_mode="futures"`` + ``margin_mode="isolated"`` + the pair
     whitelist in perpetual notation — in both the backtest runner and the
     lookahead config.
  2. The long-only / spot path is **byte-identical** to the pre-Stage-13 form:
     no behaviour change for any existing template. This is the containment
     guarantee — short capability must not perturb the spot path at all.

No Docker / no Freqtrade — these test the pure config builders + the can_short
detector directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.tools.backtest_runner import _build_backtest_config, _to_futures_pair
from orchestrator.tools.freqai_config import extract_class_bool, strategy_can_short
from orchestrator.tools.lookahead import _write_minimal_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "strategy_templates"


# ─── can_short detection (single source of truth) ───────────────────────────


def test_strategy_can_short_true_for_short_template() -> None:
    path = TEMPLATES_DIR / "bb_regime_short_template.py"
    assert strategy_can_short(path) is True
    assert extract_class_bool(path, "can_short") is True


def test_strategy_can_short_false_for_long_templates() -> None:
    # Every shipped long template sets can_short = False explicitly.
    for stem in (
        "mean_reversion_template",
        "bb_regime_reversion_template",
        "donchian_regime_trend_template",
    ):
        path = TEMPLATES_DIR / f"{stem}.py"
        assert strategy_can_short(path) is False, stem
        assert extract_class_bool(path, "can_short") is False, stem


def test_strategy_can_short_fails_closed_when_absent(tmp_path: Path) -> None:
    # A strategy that never declares can_short is long-only (fail closed).
    src = "class S:\n    timeframe = '5m'\n"
    p = tmp_path / "no_attr.py"
    p.write_text(src, encoding="utf-8")
    assert strategy_can_short(p) is False
    assert extract_class_bool(p, "can_short") is None


# ─── futures pair notation ──────────────────────────────────────────────────


def test_to_futures_pair_converts_usdt() -> None:
    assert _to_futures_pair("BTC/USDT") == "BTC/USDT:USDT"
    assert _to_futures_pair("ETH/USDT") == "ETH/USDT:USDT"


def test_to_futures_pair_idempotent_on_perpetual() -> None:
    # Already-suffixed pairs are returned unchanged (no double suffix).
    assert _to_futures_pair("BTC/USDT:USDT") == "BTC/USDT:USDT"


# ─── backtest config: futures injection for short-capable strategies ────────


def test_backtest_config_futures_for_short() -> None:
    cfg = _build_backtest_config(
        strategy_class="BbRegimeShortTemplate",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="15m",
        stake_amount=125.0,
        max_open_trades=4,
        can_short=True,
    )
    assert cfg["trading_mode"] == "futures"
    assert cfg["margin_mode"] == "isolated"
    assert cfg["exchange"]["pair_whitelist"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def _expected_spot_config(
    *, strategy_class: str, pairs: list[str], timeframe: str, stake_amount: float, mot: int
) -> dict[str, Any]:
    """The exact pre-Stage-13 spot config (the byte-identical reference)."""
    return {
        "max_open_trades": mot,
        "stake_currency": "USDT",
        "stake_amount": stake_amount,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": timeframe,
        "trading_mode": "spot",
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


def test_backtest_config_spot_is_byte_identical() -> None:
    """The long-only/spot path must equal the pre-Stage-13 config exactly.

    No ``margin_mode`` key, ``trading_mode="spot"``, spot pair whitelist.
    """
    kwargs = dict(
        strategy_class="MeanReversionTemplate",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        max_open_trades=4,
    )
    cfg = _build_backtest_config(**kwargs, can_short=False)  # type: ignore[arg-type]
    expected = _expected_spot_config(
        strategy_class="MeanReversionTemplate",
        pairs=["BTC/USDT", "ETH/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        mot=4,
    )
    assert cfg == expected
    assert "margin_mode" not in cfg


def test_backtest_config_default_can_short_is_spot() -> None:
    # Omitting can_short entirely must keep the spot default (defensive).
    cfg = _build_backtest_config(
        strategy_class="MeanReversionTemplate",
        pairs=["BTC/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        max_open_trades=4,
    )
    assert cfg["trading_mode"] == "spot"
    assert "margin_mode" not in cfg
    assert cfg["exchange"]["pair_whitelist"] == ["BTC/USDT"]


def test_backtest_config_freqai_still_injected_with_futures() -> None:
    # A (hypothetical) short FreqAI strategy keeps the freqai block AND futures.
    cfg = _build_backtest_config(
        strategy_class="X",
        pairs=["BTC/USDT"],
        timeframe="5m",
        stake_amount=100.0,
        max_open_trades=4,
        freqai={"enabled": True},
        can_short=True,
    )
    assert cfg["freqai"] == {"enabled": True}
    assert cfg["trading_mode"] == "futures"
    assert cfg["margin_mode"] == "isolated"


# ─── lookahead config mirrors the same trading mode ─────────────────────────


def test_lookahead_config_futures_for_short(tmp_path: Path) -> None:
    import json

    out = tmp_path / "config.json"
    _write_minimal_config(out, pairs=["BTC/USDT", "SOL/USDT"], timeframe="15m", can_short=True)
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["trading_mode"] == "futures"
    assert cfg["margin_mode"] == "isolated"
    assert cfg["exchange"]["pair_whitelist"] == ["BTC/USDT:USDT", "SOL/USDT:USDT"]


def test_lookahead_config_spot_unchanged(tmp_path: Path) -> None:
    import json

    out = tmp_path / "config.json"
    _write_minimal_config(out, pairs=["BTC/USDT"], timeframe="5m")  # can_short default False
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["trading_mode"] == "spot"
    assert cfg["margin_mode"] == ""  # lookahead's spot form carries an empty margin_mode
    assert cfg["exchange"]["pair_whitelist"] == ["BTC/USDT"]
