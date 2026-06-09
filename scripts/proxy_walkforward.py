"""Proxy walk-forward backtester — a FIRST-PASS FILTER for strategy ideas.

We often want to know whether a strategy idea is even plausible before paying for
a full Docker/Freqtrade run. This module replays the pipeline's anchored 6-fold
walk-forward (4mo train / 1mo test, last test_end anchored to the cache max date)
over the cached feathers in ``freqtrade/user_data/data/binance`` and computes the
SAME gate metrics as ``orchestrator/gates/thresholds.py``:

    MIN_TRADES_IS, MIN_TRADES_PER_FOLD, MIN_SHARPE_IS, MIN_PROFIT_FACTOR_IS,
    MAX_DRAWDOWN_IS, MIN_POSITIVE_FOLDS.

*** IMPORTANT — this is NOT Freqtrade. ***
It is a slightly-conservative approximation: long-only spot, one position per
pair, entry at the next candle's open, hard stop checked intrabar, optional
trailing stop / take-profit, fees 0.1%/side (~0.2% round trip). Freqtrade's fill
model, Sharpe definition and fee tiers differ. Use this to triage and tune; then
ALWAYS confirm a promising idea via ``POST /strategies/validate``, which runs the
authoritative gate. Do not declare victory on proxy results alone.

Run:
    python scripts/proxy_walkforward.py            # check the recommended configs
    python scripts/proxy_walkforward.py --sweep    # small bb-reversion sweep

Requires pandas + pyarrow (already in the project venv).
"""

from __future__ import annotations

import argparse
import statistics
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.feather as pf

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "freqtrade" / "user_data" / "data" / "binance"
DEFAULT_PAIRS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")
FEE = 0.001  # per side

# Gate thresholds mirrored from orchestrator/gates/thresholds.py (2026-06-09).
GATE = dict(MIN_TRADES_IS=90, MIN_TRADES_PER_FOLD=5, MIN_SHARPE_IS=0.5,
            MIN_PROFIT_FACTOR_IS=1.2, MAX_DRAWDOWN_IS=0.20, MIN_POSITIVE_FOLDS=4)


# ── walk-forward planning (mirrors orchestrator/subgraphs/validation.py) ──
def _add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    y = d.year + total // 12
    m = total % 12 + 1
    return date(y, m, min(d.day, monthrange(y, m)[1]))


def plan_folds(max_date: date, train_months=4, test_months=1, n_folds=6) -> list[dict]:
    span = train_months + (n_folds - 1) + test_months
    data_start = _add_months(max_date, -span)
    folds = []
    for i in range(n_folds):
        te0 = _add_months(data_start, train_months + i)
        folds.append({"fold_id": f"fold_{i+1}", "test_start": te0,
                      "test_end": _add_months(te0, test_months)})
    return folds


_CACHE: dict = {}


def load_pair(pair: str, timeframe: str) -> pd.DataFrame:
    key = (pair, timeframe)
    if key in _CACHE:
        return _CACHE[key]
    fn = DATA_DIR / f"{pair.replace('/', '_')}-{timeframe}.feather"
    df = pf.read_table(fn).to_pandas()[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    _CACHE[key] = df
    return df


def tf_minutes(tf: str) -> int:
    return int(tf[:-1]) * {"m": 1, "h": 60, "d": 1440}[tf[-1]]


@dataclass
class StrategySpec:
    name: str
    timeframe: str
    stoploss: float
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame]
    startup: int = 250
    roi_target: float = 0.0      # take-profit at +roi (0 disables)
    trail_pct: float = 0.0       # trailing-stop distance (0 disables)
    trail_activate: float = 0.0  # profit offset before trailing engages
    params: dict = field(default_factory=dict)


@dataclass
class _Trade:
    entry_px: float
    exit_px: float
    exit_time: object

    @property
    def net_ret(self) -> float:
        return (self.exit_px * (1 - FEE)) / (self.entry_px * (1 + FEE)) - 1.0


def _simulate_pair(df, spec, test_start, test_end):
    ts = pd.Timestamp(test_start, tz="UTC")
    te = pd.Timestamp(test_end, tz="UTC")
    si = int(df["date"].searchsorted(ts))
    lo = max(0, si - spec.startup - 5)
    sub = df.iloc[lo:int(df["date"].searchsorted(te))].copy().reset_index(drop=True)
    if len(sub) < spec.startup + 10:
        return []
    sub = spec.signal_fn(sub)
    o = sub["open"].to_numpy(float); high = sub["high"].to_numpy(float)
    low = sub["low"].to_numpy(float); c = sub["close"].to_numpy(float)
    enter = sub["enter"].fillna(False).to_numpy(bool)
    exit_sig = sub["exit"].fillna(False).to_numpy(bool)
    dates = sub["date"].to_numpy()
    win = ((sub["date"] >= ts) & (sub["date"] < te)).to_numpy(bool)
    trades = []; n = len(sub); pos = False
    entry_px = stop_px = peak = 0.0; entry_i = 0; i = 0
    while i < n - 1:
        if not pos:
            if enter[i] and win[i]:
                entry_i = i + 1; entry_px = o[entry_i]
                stop_px = entry_px * (1 + spec.stoploss); peak = entry_px
                pos = True; i = entry_i; continue
            i += 1
        else:
            if low[i] <= stop_px:
                trades.append(_Trade(entry_px, stop_px, dates[i])); pos = False; i += 1; continue
            peak = max(peak, high[i])
            if spec.trail_pct > 0 and (peak / entry_px - 1) >= spec.trail_activate:
                tstop = peak * (1 - spec.trail_pct)
                if low[i] <= tstop and tstop > stop_px:
                    trades.append(_Trade(entry_px, tstop, dates[i])); pos = False; i += 1; continue
            if spec.roi_target > 0 and high[i] >= entry_px * (1 + spec.roi_target):
                trades.append(_Trade(entry_px, entry_px * (1 + spec.roi_target), dates[i]))
                pos = False; i += 1; continue
            if exit_sig[i] and i + 1 < n:
                trades.append(_Trade(entry_px, o[i + 1], dates[i + 1])); pos = False; i += 1; continue
            i += 1
    if pos:
        trades.append(_Trade(entry_px, c[n - 1], dates[n - 1]))
    return trades


def _fold_metrics(trades, test_start, test_end) -> dict:
    if not trades:
        return {"trades": 0, "sharpe": 0.0, "profit_factor": 0.0, "max_dd": 0.0}
    rets = np.array([t.net_ret for t in trades])
    df = pd.DataFrame({"exit": [pd.Timestamp(t.exit_time, tz="UTC") if pd.Timestamp(t.exit_time).tz is None
                                else pd.Timestamp(t.exit_time) for t in trades], "ret": rets})
    ds = df.sort_values("exit"); eq = (1 + ds["ret"]).cumprod()
    max_dd = float((1 - eq / eq.cummax()).max())
    wins = ds.loc[ds["ret"] > 0, "ret"].sum(); losses = -ds.loc[ds["ret"] < 0, "ret"].sum()
    pf_ = float(wins / losses) if losses > 1e-12 else (99.0 if wins > 0 else 0.0)
    df["day"] = df["exit"].dt.floor("D")
    daily = df.groupby("day")["ret"].sum()
    idx = pd.date_range(pd.Timestamp(test_start, tz="UTC"),
                        pd.Timestamp(test_end, tz="UTC") - pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(idx, fill_value=0.0); sd = daily.std(ddof=1)
    sharpe = float(daily.mean() / sd * np.sqrt(365)) if sd > 1e-12 else 0.0
    return {"trades": len(trades), "sharpe": sharpe, "profit_factor": min(pf_, 99.0), "max_dd": max_dd}


def run_spec(spec: StrategySpec, pairs=DEFAULT_PAIRS) -> dict:
    data = {p: load_pair(p, spec.timeframe) for p in pairs}
    folds = plan_folds(min(d["date"].max().date() for d in data.values()))
    rows, total = [], 0
    for f in folds:
        ft = []
        for p in pairs:
            ft += _simulate_pair(data[p], spec, f["test_start"], f["test_end"])
        m = _fold_metrics(ft, f["test_start"], f["test_end"]); rows.append(m); total += m["trades"]
    sharpes = [r["sharpe"] for r in rows]; pfs = [r["profit_factor"] for r in rows]
    dds = [r["max_dd"] for r in rows]; tpf = [r["trades"] for r in rows]
    ms = statistics.fmean(sharpes); mp = statistics.fmean(pfs)
    positive = sum(1 for s in sharpes if s > 0)
    fails = []
    if total < GATE["MIN_TRADES_IS"]: fails.append(f"trades={total}<{GATE['MIN_TRADES_IS']}")
    if tpf and min(tpf) < GATE["MIN_TRADES_PER_FOLD"]: fails.append(f"min/fold={min(tpf)}<5")
    if ms < GATE["MIN_SHARPE_IS"]: fails.append(f"sharpe={ms:.2f}<0.5")
    if mp < GATE["MIN_PROFIT_FACTOR_IS"]: fails.append(f"pf={mp:.2f}<1.2")
    if max(dds) > GATE["MAX_DRAWDOWN_IS"]: fails.append(f"dd={max(dds):.2f}>0.20")
    if positive < GATE["MIN_POSITIVE_FOLDS"]: fails.append(f"pos={positive}<4")
    return {"name": spec.name, "timeframe": spec.timeframe, "params": spec.params,
            "total_trades": total, "mean_sharpe": round(ms, 3), "mean_pf": round(mp, 3),
            "positive_folds": positive, "max_dd": round(max(dds), 3),
            "trades_per_fold": tpf, "sharpe_per_fold": [round(s, 2) for s in sharpes],
            "passed": not fails, "failures": fails}


# ── indicator helpers + the two shipped TA archetypes ──
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def bb_regime_reversion(bb_period, bb_std, rsi_period, rsi_buy, ema_trend):
    """Mirror of bb_regime_reversion_template (BB on typical price)."""
    def f(df):
        df = df.copy()
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        ma = tp.rolling(bb_period).mean(); sd = tp.rolling(bb_period).std()
        df["enter"] = ((df["close"] <= ma - bb_std * sd)
                       & (_rsi(df["close"], rsi_period) < rsi_buy)
                       & (df["close"] > _ema(df["close"], ema_trend)))
        df["exit"] = df["close"] >= ma
        return df
    return f


def donchian_regime_trend(donchian_period, ema_fast, ema_slow):
    """Mirror of donchian_regime_trend_template."""
    def f(df):
        df = df.copy()
        ph = df["high"].rolling(donchian_period).max().shift(1)
        df["enter"] = (df["close"] > ph) & (_ema(df["close"], ema_fast) > _ema(df["close"], ema_slow))
        df["exit"] = _ema(df["close"], ema_fast) < _ema(df["close"], ema_slow)
        return df
    return f


def _print(r):
    flag = "PASS" if r["passed"] else "FAIL"
    print(f"[{flag}] {r['name']:26} {r['timeframe']:3} | sharpe {r['mean_sharpe']:>6} "
          f"pf {r['mean_pf']:>6} pos {r['positive_folds']}/6 trades {r['total_trades']:>4} "
          f"dd {r['max_dd']:>5} | per-fold {r['sharpe_per_fold']}")
    if r["failures"]:
        print(f"        gate failures: {r['failures']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="small bb-reversion sweep")
    args = ap.parse_args()
    print(f"Anchored 6-fold walk-forward over {DATA_DIR}")
    print(f"Gate: {GATE}\n")

    print("== Recommended shipped configs ==")
    _print(run_spec(StrategySpec(
        "bb_regime_reversion", "15m", -0.06,
        bb_regime_reversion(20, 2.3, 14, 35, 200), startup=250, roi_target=0.025,
        params={"bb_std": 2.3, "rsi_buy": 35, "ema_trend": 200, "roi": 0.025})))
    _print(run_spec(StrategySpec(
        "donchian_regime_trend", "1h", -0.10,
        donchian_regime_trend(20, 20, 50), startup=250, trail_pct=0.05,
        params={"donchian": 20, "ema_fast": 20, "ema_slow": 50, "trail": 0.05})))

    if args.sweep:
        print("\n== bb-reversion sweep (bb_std x rsi_buy) ==")
        for bs in (2.2, 2.3, 2.4, 2.5, 2.6):
            for rb in (33, 35, 37):
                _print(run_spec(StrategySpec(
                    f"bb_{bs}_{rb}", "15m", -0.06,
                    bb_regime_reversion(20, bs, 14, rb, 200), startup=250, roi_target=0.025,
                    params={"bb_std": bs, "rsi_buy": rb})))


if __name__ == "__main__":
    main()
