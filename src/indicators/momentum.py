"""
Momentum indicators.

RSI 14, MACD histogram (12/26/9), RSI divergence detector.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators import ta_core as ta

# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI value + threshold flags + delta.

    Columns
    ~~~~~~~
    * ``rsi_14``         – RSI value
    * ``rsi_above_50``   – binary
    * ``rsi_above_70``   – binary
    * ``rsi_below_30``   – binary
    * ``rsi_delta_3``    – 3-candle change
    """
    out = df.copy()
    rsi = ta.rsi(out["close"], length=period)
    out[f"rsi_{period}"] = rsi
    out["rsi_above_50"] = (rsi > 50).astype(int)
    out["rsi_above_70"] = (rsi > 70).astype(int)
    out["rsi_below_30"] = (rsi < 30).astype(int)
    out["rsi_delta_3"] = rsi.diff(3)
    return out


# ---------------------------------------------------------------------------
# MACD Histogram
# ---------------------------------------------------------------------------


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD histogram + direction flags.

    Columns
    ~~~~~~~
    * ``macd_hist``          – histogram value
    * ``macd_hist_positive`` – binary (histogram > 0)
    * ``macd_hist_expanding``– binary (abs(hist) > abs(prev hist))
    * ``macd_hist_delta``    – 1-candle Δ of histogram
    """
    out = df.copy()
    macd_df = ta.macd(out["close"], fast=fast, slow=slow, signal=signal)
    # pandas-ta columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    hist = macd_df[hist_col]
    out["macd_hist"] = hist
    out["macd_hist_positive"] = (hist > 0).astype(int)
    out["macd_hist_expanding"] = (hist.abs() > hist.shift(1).abs()).astype(int)
    out["macd_hist_delta"] = hist.diff(1)
    return out


# ---------------------------------------------------------------------------
# RSI Divergence Detector
# ---------------------------------------------------------------------------


def add_rsi_divergence(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Detect RSI divergence at swing points.

    Bearish divergence: price makes a higher high but RSI makes a lower high.
    Bullish divergence: price makes a lower low but RSI makes a higher low.

    Requires ``swing_high_price`` and ``swing_low_price`` from ``add_swings()``,
    and ``rsi_14`` from ``add_rsi()``.

    Columns
    ~~~~~~~
    * ``rsi_div_bearish``  – 1 at swing highs with bearish divergence
    * ``rsi_div_bullish``  – 1 at swing lows with bullish divergence
    """
    out = df.copy()

    if f"rsi_{period}" not in out.columns:
        out = add_rsi(out, period)

    rsi = out[f"rsi_{period}"].values.astype(float)
    sh_price = out["swing_high_price"].values.astype(float)
    sl_price = out["swing_low_price"].values.astype(float)
    n = len(out)

    div_bear = np.zeros(n, dtype=np.int8)
    div_bull = np.zeros(n, dtype=np.int8)

    # Track previous swing high/low price and RSI
    prev_sh_price = np.nan
    prev_sh_rsi = np.nan
    prev_sl_price = np.nan
    prev_sl_rsi = np.nan

    for i in range(n):
        # Bearish divergence at swing highs
        if not np.isnan(sh_price[i]):
            if (
                not np.isnan(prev_sh_price)
                and sh_price[i] > prev_sh_price
                and rsi[i] < prev_sh_rsi
            ):
                div_bear[i] = 1
            prev_sh_price = sh_price[i]
            prev_sh_rsi = rsi[i]

        # Bullish divergence at swing lows
        if not np.isnan(sl_price[i]):
            if (
                not np.isnan(prev_sl_price)
                and sl_price[i] < prev_sl_price
                and rsi[i] > prev_sl_rsi
            ):
                div_bull[i] = 1
            prev_sl_price = sl_price[i]
            prev_sl_rsi = rsi[i]

    out["rsi_div_bearish"] = div_bear
    out["rsi_div_bullish"] = div_bull
    return out
