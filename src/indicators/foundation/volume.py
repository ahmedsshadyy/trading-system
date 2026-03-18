"""
Volume indicators.

Volume ratio, volume slope, key candle volume flags, candle delta proxy,
Volume Spread Analysis, wick rejection ratio.

All functions are pure: input DataFrame is never mutated.

NOTE: These work with tick volume from MetaAPI (CFD).  True order-flow
features require Level 2 / futures data — these are OHLC approximations
as documented in strategy_definitions.docx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Volume Ratio
# ---------------------------------------------------------------------------


def add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Current volume / N-period SMA of volume.

    Columns
    ~~~~~~~
    * ``vol_ratio``          – continuous ratio
    * ``vol_above_1_5x``     – binary: above 1.5× average
    * ``vol_below_0_8x``     – binary: below 0.8× (weak participation)
    * ``vol_slope_5``        – linear slope of volume over 5 candles
    """
    out = df.copy()
    vol = out["volume"].astype(float)
    avg = vol.rolling(period).mean()
    out["vol_ratio"] = vol / avg
    out["vol_above_1_5x"] = (out["vol_ratio"] > 1.5).astype(int)
    out["vol_below_0_8x"] = (out["vol_ratio"] < 0.8).astype(int)

    # Volume slope: linear regression slope over 5 candles, normalised by avg
    out["vol_slope_5"] = (
        vol.rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
        / avg
    )
    return out


# ---------------------------------------------------------------------------
# Key Candle Volume Flags
# ---------------------------------------------------------------------------


def add_key_volume_flags(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Binary flags for volume relative to average on the current candle.

    These flags are generic — the scanner uses them at the right candle
    (breakout, sweep, rejection) depending on strategy context.

    Columns
    ~~~~~~~
    * ``vol_above_avg``  – volume > 20-period mean
    * ``vol_below_avg``  – volume < 20-period mean
    """
    out = df.copy()
    vol = out["volume"].astype(float)
    avg = vol.rolling(period).mean()
    out["vol_above_avg"] = (vol > avg).astype(int)
    out["vol_below_avg"] = (vol < avg).astype(int)
    return out


# ---------------------------------------------------------------------------
# Candle Delta Proxy
# ---------------------------------------------------------------------------


def add_candle_delta_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC-approximated buying/selling pressure.

    ``delta_proxy = (close − low) / (high − low) × vol_ratio``

    Higher → more implied buying pressure.  Lower → selling.

    Columns: ``candle_delta_proxy``.
    """
    out = df.copy()
    rng = (out["high"] - out["low"]).astype(float)
    close_pos = (out["close"] - out["low"]).astype(float)

    if "vol_ratio" not in out.columns:
        vol = out["volume"].astype(float)
        avg = vol.rolling(20).mean()
        vr = vol / avg
    else:
        vr = out["vol_ratio"]

    out["candle_delta_proxy"] = np.where(rng > 0, (close_pos / rng) * vr, 0.0)
    return out


# ---------------------------------------------------------------------------
# Volume Spread Analysis (VSA)
# ---------------------------------------------------------------------------


def add_vsa(df: pd.DataFrame) -> pd.DataFrame:
    """Simple VSA flags.

    Columns
    ~~~~~~~
    * ``vsa_absorption``   – high volume + small range (absorption)
    * ``vsa_directional``  – high volume + large range (genuine move)
    """
    out = df.copy()
    vol = out["volume"].astype(float)
    avg_vol = vol.rolling(20).mean()
    rng = (out["high"] - out["low"]).astype(float)
    avg_rng = rng.rolling(20).mean()

    high_vol = vol > avg_vol
    small_rng = rng < (avg_rng * 0.7)
    large_rng = rng > (avg_rng * 1.3)

    out["vsa_absorption"] = (high_vol & small_rng).astype(int)
    out["vsa_directional"] = (high_vol & large_rng).astype(int)
    return out


# ---------------------------------------------------------------------------
# Wick Rejection Ratio
# ---------------------------------------------------------------------------


def add_wick_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Upper and lower wick as fraction of total range.

    Columns
    ~~~~~~~
    * ``upper_wick_ratio``  – (high − max(open,close)) / (high − low)
    * ``lower_wick_ratio``  – (min(open,close) − low) / (high − low)
    """
    out = df.copy()
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    c = out["close"].astype(float)
    rng = h - l

    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)

    out["upper_wick_ratio"] = np.where(rng > 0, (h - body_top) / rng, 0.0)
    out["lower_wick_ratio"] = np.where(rng > 0, (body_bot - l) / rng, 0.0)
    return out
