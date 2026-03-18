"""
Volatility indicators.

ATR 14, Bollinger Band width + percentile, ATR pullback/impulse ratio,
breakout candle body ratio.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators import ta_core as ta

# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR absolute value + 50-candle percentile.

    Columns
    ~~~~~~~
    * ``atr_14``             – ATR value
    * ``atr_pct_50``         – percentile rank over last 50 candles (0–100)
    """
    out = df.copy()
    atr = ta.atr(out["high"], out["low"], out["close"], length=period)
    out[f"atr_{period}"] = atr
    out["atr_pct_50"] = atr.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    return out


# ---------------------------------------------------------------------------
# Bollinger Band Width
# ---------------------------------------------------------------------------


def add_bb_width(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Band width + percentile.

    Columns
    ~~~~~~~
    * ``bb_width``           – (upper − lower) / middle
    * ``bb_width_pct_50``    – percentile rank over 50 candles (0–100)
    * ``bb_width_below_30``  – binary: below 30th percentile (compression)
    * ``bb_width_below_40``  – binary: below 40th percentile
    """
    out = df.copy()
    bb = ta.bbands(out["close"], length=period, std=std)
    # pandas-ta columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    upper = bb[f"BBU_{period}_{std}"]
    lower = bb[f"BBL_{period}_{std}"]
    mid = bb[f"BBM_{period}_{std}"]
    width = (upper - lower) / mid
    out["bb_width"] = width

    pct = width.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    out["bb_width_pct_50"] = pct
    out["bb_width_below_30"] = (pct < 30).astype(int)
    out["bb_width_below_40"] = (pct < 40).astype(int)
    return out


# ---------------------------------------------------------------------------
# ATR Pullback / Impulse Ratio
# ---------------------------------------------------------------------------


def compute_atr_ratio(
    df: pd.DataFrame,
    impulse_start: int,
    impulse_end: int,
    pullback_start: int,
    pullback_end: int,
    atr_col: str = "atr_14",
) -> float:
    """Compute pullback ATR / impulse ATR ratio for a specific range.

    This is a *per-signal* helper — the scanner calls it when evaluating
    a specific setup.  It is NOT applied as a rolling column.

    Returns
    -------
    float
        Ratio < 0.8 → quality pullback.  > 0.8 → too volatile.
    """
    impulse_atr = df[atr_col].iloc[impulse_start:impulse_end].mean()
    pullback_atr = df[atr_col].iloc[pullback_start:pullback_end].mean()
    if impulse_atr == 0 or np.isnan(impulse_atr):
        return np.nan
    return pullback_atr / impulse_atr


def add_rolling_atr_ratio(
    df: pd.DataFrame, impulse_len: int = 5, pullback_len: int = 5
) -> pd.DataFrame:
    """Rolling ATR ratio: mean ATR of last ``pullback_len`` candles /
    mean ATR of the ``impulse_len`` candles before that.

    This is a *rolling approximation* useful as a feature.  The scanner
    will use ``compute_atr_ratio`` with exact impulse/pullback boundaries.

    Columns: ``atr_ratio_rolling``.
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    atr = out["atr_14"]
    total = impulse_len + pullback_len
    ratios = []
    for i in range(len(out)):
        if i < total - 1:
            ratios.append(np.nan)
            continue
        impulse_mean = atr.iloc[i - total + 1 : i - pullback_len + 1].mean()
        pullback_mean = atr.iloc[i - pullback_len + 1 : i + 1].mean()
        if impulse_mean == 0 or np.isnan(impulse_mean):
            ratios.append(np.nan)
        else:
            ratios.append(pullback_mean / impulse_mean)

    out["atr_ratio_rolling"] = ratios
    return out


# ---------------------------------------------------------------------------
# Breakout Candle Body Ratio
# ---------------------------------------------------------------------------


def add_body_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Candle body as fraction of total range.

    Columns
    ~~~~~~~
    * ``body_ratio``          – |close − open| / (high − low), 0–1
    * ``body_ratio_above_60`` – binary: conviction candle
    """
    out = df.copy()
    body = (out["close"] - out["open"]).abs()
    rng = out["high"] - out["low"]
    out["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
    out["body_ratio_above_60"] = (out["body_ratio"] > 0.6).astype(int)
    return out
