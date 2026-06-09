"""Pydantic schema co-located with ``donchian_regime_trend_template.py`` (BRD §8 rule 3).

Fields mirror the ``# SLOT:`` markers in the template byte-for-byte. The generator and
the manual-injection endpoint both validate params against this model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DonchianRegimeTrendParams(BaseModel):
    """Validated parameter set for the regime-filtered Donchian breakout template.

    Ranges:
      - ``donchian_period`` 10-100: breakout lookback. Shorter = more, noisier
        breakouts; longer = rarer, higher-conviction.
      - ``ema_fast`` 10-100 / ``ema_slow`` 30-200: the regime filter. Keep
        ema_fast < ema_slow for the uptrend test to be meaningful.
      - ``trailing_stop_positive`` 0.02-0.15: trailing-stop distance below the peak.
        Tighter protects profit but whipsaws; wider gives the trend room.
      - ``stoploss`` -0.15 to -0.03: hard backstop beneath the trailing stop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    donchian_period: int = Field(ge=10, le=100, description="Breakout lookback (prior-N high).")
    ema_fast: int = Field(ge=10, le=100, description="Fast regime EMA.")
    ema_slow: int = Field(ge=30, le=200, description="Slow regime EMA (uptrend = fast > slow).")
    trailing_stop_positive: float = Field(
        ge=0.02, le=0.15, description="Trailing-stop distance below peak."
    )
    stoploss: float = Field(
        ge=-0.15, le=-0.03, description="Hard per-trade stoploss as a negative fraction."
    )
