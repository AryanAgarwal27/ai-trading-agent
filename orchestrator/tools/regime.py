"""Market regime classification + regime_log writer (BRD §5.7, §5.8, §13 Stage 3).

The minimal v1 regime is a two-axis bucket: realized volatility × trend. The
researcher and validation subgraphs query the most recent regime to bias
hypothesis selection and to slice walk-forward folds (BRD §5.4 ``regime_worker``).

Why "minimal"?
  BRD §13 Stage 3 only requires "a regime job writes a row" — full HMM-based
  detection is explicitly deferred to a later iteration (BRD §9 mentions HMM
  as optional). The two-axis bucket is enough to:
  - Tag every ``regime_log`` row with a coarse label the supervisor (Stage 9)
    can correlate with strategy lifecycle outcomes,
  - Provide ``("failures", regime)`` and ``("wins", regime)`` Store partitions
    (BRD §5.9) that are stable across re-tunings.

Design constraints honored here:
  - **stdlib-only computation.** No pandas, no numpy. The classifier accepts
    a sequence of closes and returns a label + features dict. This keeps the
    runtime dep surface flat (BRD §4 doesn't pin pandas; pulling it in for
    something this small would be premature).
  - **DB write is async.** The APScheduler regime job (Stage 7) will run in
    the orchestrator's event loop; an async insert lets it cooperate with
    the rest of the graph.
  - **Caller supplies the OHLCV source.** A future Stage 4 regime_worker may
    use a Freqtrade-cached feather; the Stage 7 APScheduler job may pull
    fresh closes via the Freqtrade REST client. Decoupling lets both work
    against the same ``classify_regime`` core.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import psycopg

# Bucket labels. The composite label ``f"{vol}_{trend}"`` is what gets stored
# in regime_log.regime and used as a Store namespace partition (BRD §5.9).
VolLabel = Literal["low_vol", "mid_vol", "high_vol"]
TrendLabel = Literal["down", "flat", "up"]
RegimeLabel = str  # f"{vol}_{trend}"

# Crypto-tuned default thresholds. These are not BRD-§10 gate thresholds —
# they are descriptive cutoffs for tagging, not pass/fail logic. Stage 4 will
# re-calibrate them from the regime_log history once enough samples exist.
DEFAULT_ANNUALIZED_VOL_LOW_PCT = 0.40  # below 40% annualized realized vol → low
DEFAULT_ANNUALIZED_VOL_HIGH_PCT = 0.80  # above 80% → high
DEFAULT_TREND_FLAT_BAND_PCT = 0.03  # within ±3% of SMA → flat


@dataclass(frozen=True, slots=True)
class RegimeFeatures:
    """Numeric features that produced a regime label.

    Stored alongside the label in ``regime_log.features`` so a future
    re-classifier can recompute labels from the same inputs without re-pulling
    OHLCV.
    """

    annualized_vol: float
    sma_distance_pct: float
    closes_used: int
    timeframe_minutes: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "annualized_vol": self.annualized_vol,
            "sma_distance_pct": self.sma_distance_pct,
            "closes_used": self.closes_used,
            "timeframe_minutes": self.timeframe_minutes,
        }


def classify_regime(
    closes: Sequence[float],
    *,
    timeframe_minutes: int,
    vol_low: float = DEFAULT_ANNUALIZED_VOL_LOW_PCT,
    vol_high: float = DEFAULT_ANNUALIZED_VOL_HIGH_PCT,
    trend_flat_band: float = DEFAULT_TREND_FLAT_BAND_PCT,
) -> tuple[RegimeLabel, RegimeFeatures]:
    """Bucket the most recent closes into a ``{low,mid,high}_vol_{down,flat,up}`` label.

    Parameters
    ----------
    closes
        Recent close prices, oldest first. Must contain ≥ 30 samples — fewer
        produces a noisy SMA and an unreliable vol estimate.
    timeframe_minutes
        Candle interval in minutes (5 for 5m, 60 for 1h, …). Used to
        annualize the realized vol.
    vol_low, vol_high, trend_flat_band
        Override the crypto-tuned defaults (see module docstring).

    Returns
    -------
    (label, features)
        ``label`` is the composite string written to ``regime_log.regime``.
        ``features`` is the raw numbers backing the bucket choice.
    """
    if len(closes) < 30:
        raise ValueError(f"classify_regime needs ≥ 30 closes; got {len(closes)}")
    if timeframe_minutes <= 0:
        raise ValueError(f"timeframe_minutes must be positive; got {timeframe_minutes}")

    # Log returns over the full window. Using log returns (rather than pct)
    # gives a vol estimate that's additive across timeframes and matches the
    # convention Freqtrade and most quant libraries use internally.
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    per_candle_std = statistics.stdev(log_returns) if len(log_returns) >= 2 else 0.0

    # Annualize: 365 trading days × (minutes per day / timeframe).
    candles_per_year = 365 * (24 * 60) / timeframe_minutes
    annualized_vol = per_candle_std * math.sqrt(candles_per_year)

    # Trend: current close relative to SMA. We use the whole window as the
    # SMA period — for the BRD-default 30+ candles that's a reasonable smoother.
    sma = statistics.fmean(closes)
    sma_distance_pct = (closes[-1] - sma) / sma

    vol_label: VolLabel
    if annualized_vol < vol_low:
        vol_label = "low_vol"
    elif annualized_vol < vol_high:
        vol_label = "mid_vol"
    else:
        vol_label = "high_vol"

    trend_label: TrendLabel
    if sma_distance_pct > trend_flat_band:
        trend_label = "up"
    elif sma_distance_pct < -trend_flat_band:
        trend_label = "down"
    else:
        trend_label = "flat"

    features = RegimeFeatures(
        annualized_vol=annualized_vol,
        sma_distance_pct=sma_distance_pct,
        closes_used=len(closes),
        timeframe_minutes=timeframe_minutes,
    )
    return f"{vol_label}_{trend_label}", features


async def insert_regime_log(
    *,
    conn: psycopg.AsyncConnection,
    regime: RegimeLabel,
    features: RegimeFeatures,
    detector: str,
    at: datetime | None = None,
) -> None:
    """Persist a regime classification to the ``regime_log`` table (BRD §5.8).

    Parameters
    ----------
    conn
        An open async psycopg connection to the ``app`` database. The caller
        owns the connection lifecycle and the transaction boundary — this
        function does not commit. The APScheduler regime job (Stage 7) will
        wrap a batch of inserts in one transaction.
    regime
        Composite label from :func:`classify_regime`.
    features
        The numeric features that produced the label, persisted as JSONB.
    detector
        Free-form tag identifying the classifier version (e.g.
        ``"vol_trend_v1"``). Lets future regime classifiers coexist in the
        same table without colliding on the ``(at, detector)`` PK.
    at
        Override the timestamp (mainly for tests). Defaults to ``now()`` on
        the DB side via ``NOW()`` — passing ``None`` here means "let the DB
        decide".
    """
    if at is None:
        # We explicitly pass `now()` to the DB so the value matches the PK
        # convention used in the migration's CREATE TABLE DEFAULT clause.
        sql = (
            "INSERT INTO regime_log (at, regime, features, detector) " "VALUES (NOW(), %s, %s, %s)"
        )
        params: tuple[Any, ...] = (regime, json.dumps(features.to_jsonable()), detector)
    else:
        # Caller-supplied timestamps are normalized to UTC. PK is (at, detector)
        # so a duplicate insert at the same instant from the same detector will
        # collide — that's the intended uniqueness guarantee.
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        sql = "INSERT INTO regime_log (at, regime, features, detector) " "VALUES (%s, %s, %s, %s)"
        params = (at, regime, json.dumps(features.to_jsonable()), detector)

    async with conn.cursor() as cur:
        await cur.execute(sql, params)
