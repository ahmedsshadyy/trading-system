"""
Volume Profile calculator.

POC (Point of Control), VAH (Value Area High), VAL (Value Area Low)
computed from tick volume over a configurable lookback.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 80,
    n_bins: int = 50,
) -> dict:
    """Compute Volume Profile levels for the last ``lookback`` candles.

    Parameters
    ----------
    df : DataFrame
        Must have ``high, low, close, volume`` columns.
        Should be the *last* ``lookback`` rows of a larger frame.
    lookback : int
        Number of candles to include (default 80 ≈ 20 sessions of H4).
    n_bins : int
        Number of price bins.

    Returns
    -------
    dict with keys: ``poc``, ``vah``, ``val``, ``profile`` (np.array of volumes per bin),
    ``bin_edges`` (np.array).
    """
    segment = df.tail(lookback).copy()
    h = segment["high"].values.astype(float)
    l = segment["low"].values.astype(float)
    c = segment["close"].values.astype(float)
    v = segment["volume"].values.astype(float)
    tp = (h + l + c) / 3.0

    price_min = l.min()
    price_max = h.max()
    if price_max == price_min:
        return {
            "poc": price_min,
            "vah": price_max,
            "val": price_min,
            "profile": np.zeros(n_bins),
            "bin_edges": np.array([]),
        }

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    profile = np.zeros(n_bins)

    # Assign each candle's volume to the bin containing its typical price
    bin_indices = np.digitize(tp, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    for j in range(len(tp)):
        profile[bin_indices[j]] += v[j]

    # POC: bin with highest volume
    poc_bin = np.argmax(profile)
    poc = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

    # Value Area: 70% of total volume, expanding from POC
    total_vol = profile.sum()
    target = total_vol * 0.7

    va_low_bin = poc_bin
    va_high_bin = poc_bin
    va_vol = profile[poc_bin]

    while va_vol < target and (va_low_bin > 0 or va_high_bin < n_bins - 1):
        vol_below = profile[va_low_bin - 1] if va_low_bin > 0 else 0
        vol_above = profile[va_high_bin + 1] if va_high_bin < n_bins - 1 else 0

        if vol_below >= vol_above and va_low_bin > 0:
            va_low_bin -= 1
            va_vol += profile[va_low_bin]
        elif va_high_bin < n_bins - 1:
            va_high_bin += 1
            va_vol += profile[va_high_bin]
        else:
            va_low_bin -= 1
            va_vol += profile[va_low_bin]

    vah = bin_edges[va_high_bin + 1]
    val = bin_edges[va_low_bin]

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "profile": profile,
        "bin_edges": bin_edges,
    }


def add_volume_profile(
    df: pd.DataFrame,
    lookback: int = 80,
    n_bins: int = 50,
) -> pd.DataFrame:
    """Add rolling Volume Profile levels as columns.

    Recomputes VP every ``lookback`` candles (not every row — too expensive).
    Forward-fills between recomputes.

    Columns
    ~~~~~~~
    * ``vp_poc``               – Point of Control price
    * ``vp_vah``               – Value Area High
    * ``vp_val``               – Value Area Low
    * ``vp_poc_distance_atr``  – |close − POC| / ATR
    """
    out = df.copy()
    n = len(out)

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    # Recompute at every lookback interval
    step = max(lookback // 4, 1)  # recompute every quarter-lookback for some overlap
    for i in range(lookback, n, step):
        vp = compute_volume_profile(out.iloc[:i], lookback=lookback, n_bins=n_bins)
        # Apply to the next chunk
        end_idx = min(i + step, n)
        poc[i:end_idx] = vp["poc"]
        vah[i:end_idx] = vp["vah"]
        val[i:end_idx] = vp["val"]

    out["vp_poc"] = pd.Series(poc).ffill().values
    out["vp_vah"] = pd.Series(vah).ffill().values
    out["vp_val"] = pd.Series(val).ffill().values

    atr = out["atr_14"].values.astype(float)
    close = out["close"].values.astype(float)
    out["vp_poc_distance_atr"] = np.where(
        atr > 0,
        np.abs(close - out["vp_poc"].values.astype(float)) / atr,
        np.nan,
    )
    return out
