"""
structure/swings.py

Canonical swing high/low detector — causal only.

This is the single source of truth for structural swing detection across:
- research
- training
- backtesting
- live scanner
- live inference

A swing high at bar i means high[i] exceeds the highs of the previous
``window - 1`` bars. A swing low at bar i means low[i] is below the lows
of the previous ``window - 1`` bars.

This detector uses only past and current data, never future bars.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc


def add_swings(
    df: pd.DataFrame,
    window: int = 6,
    *,
    atr_length: int = 14,
    min_prominence_atr: float = 0.0,
    min_separation: int = 1,
    require_rejection: bool = False,
    min_wick_frac: float = 0.0,
    max_body_frac: float = 1.0,
) -> pd.DataFrame:
    """Canonical causal swing detector — no look-ahead.

    Parameters
    ----------
    window : int
        Past-only lookback window. Current bar is compared against the prior
        ``window - 1`` bars.
    atr_length : int
        ATR length used for prominence normalization.
    min_prominence_atr : float
        Minimum swing prominence divided by ATR.
    min_separation : int
        Minimum bars between consecutive same-side swings.
    require_rejection : bool
        If True, require wick/body geometry consistent with rejection.
    min_wick_frac : float
        Minimum same-side wick fraction when ``require_rejection=True``.
        - swing high -> upper wick fraction
        - swing low  -> lower wick fraction
    max_body_frac : float
        Maximum candle body fraction when ``require_rejection=True``.

    Returns
    -------
    DataFrame
        Original data plus canonical causal swing columns.

    Canonical columns
    -----------------
    * ``swing_high / swing_low``               – 1 on detection bar
    * ``swing_high_price / swing_low_price``   – price on detection bar
    * ``swing_high_idx / swing_low_idx``       – index of detected swing
    * ``swing_high_detect_flag / swing_low_detect_flag``
    * ``swing_high_detect_idx / swing_low_detect_idx``
    * ``last_swing_high / last_swing_low``     – latest known swing level
    * ``last_swing_high_idx / last_swing_low_idx``
    * ``swing_high_age / swing_low_age``       – bars since latest swing
    * ``swing_high_prominence / swing_low_prominence``
    * ``swing_high_prominence_atr / swing_low_prominence_atr``
    * ``swing_high_strength / swing_low_strength``

    Backward compatibility
    ----------------------
    ``add_swings_causal`` is retained as an alias to this function.
    """
    out = df.copy()
    require_ohlc(out)

    n = len(out)
    if n == 0:
        return out

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)

    rng = h - lo
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo

    with np.errstate(invalid="ignore", divide="ignore"):
        upper_wick_frac = np.where(rng > 0, upper_wick / rng, 0.0)
        lower_wick_frac = np.where(rng > 0, lower_wick / rng, 0.0)
        body_frac = np.where(rng > 0, body / rng, 0.0)

    sh = np.zeros(n, dtype=np.int8)
    sl = np.zeros(n, dtype=np.int8)

    sh_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)

    sh_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)

    sh_detect_flag = np.zeros(n, dtype=np.int8)
    sl_detect_flag = np.zeros(n, dtype=np.int8)

    sh_detect_idx = np.full(n, -1, dtype=int)
    sl_detect_idx = np.full(n, -1, dtype=int)

    sh_prom = np.full(n, np.nan)
    sl_prom = np.full(n, np.nan)

    sh_prom_atr = np.full(n, np.nan)
    sl_prom_atr = np.full(n, np.nan)

    sh_strength = np.full(n, np.nan)
    sl_strength = np.full(n, np.nan)

    last_sh_bar = -(10**9)
    last_sl_bar = -(10**9)

    for i in range(1, n):
        start = max(0, i - window + 1)
        if i - start < 1:
            continue

        hist_high = h[start:i]
        hist_low = lo[start:i]

        prev_max = np.max(hist_high)
        prev_min = np.min(hist_low)

        atr_i = atr[i]
        atr_ok = np.isfinite(atr_i) and atr_i > 0

        # Swing high candidate
        if h[i] >= prev_max:
            prom = h[i] - prev_max
            prom_atr = prom / atr_i if atr_ok else np.nan
            prom_ok = (min_prominence_atr <= 0) or (
                np.isfinite(prom_atr) and prom_atr >= min_prominence_atr
            )
            sep_ok = (i - last_sh_bar) > min_separation
            geom_ok = (not require_rejection) or (
                upper_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )

            if prom_ok and sep_ok and geom_ok:
                sh[i] = 1
                sh_price[i] = h[i]
                sh_idx[i] = i
                sh_detect_flag[i] = 1
                sh_detect_idx[i] = i
                sh_prom[i] = prom
                sh_prom_atr[i] = prom_atr
                sh_strength[i] = prom_atr
                last_sh_bar = i

        # Swing low candidate
        if lo[i] <= prev_min:
            prom = prev_min - lo[i]
            prom_atr = prom / atr_i if atr_ok else np.nan
            prom_ok = (min_prominence_atr <= 0) or (
                np.isfinite(prom_atr) and prom_atr >= min_prominence_atr
            )
            sep_ok = (i - last_sl_bar) > min_separation
            geom_ok = (not require_rejection) or (
                lower_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )

            if prom_ok and sep_ok and geom_ok:
                sl[i] = 1
                sl_price[i] = lo[i]
                sl_idx[i] = i
                sl_detect_flag[i] = 1
                sl_detect_idx[i] = i
                sl_prom[i] = prom
                sl_prom_atr[i] = prom_atr
                sl_strength[i] = prom_atr
                last_sl_bar = i

    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_swing_high_idx = np.full(n, np.nan)
    last_swing_low_idx = np.full(n, np.nan)

    cur_h = np.nan
    cur_l = np.nan
    cur_h_idx = np.nan
    cur_l_idx = np.nan

    for i in range(n):
        if sh[i] == 1:
            cur_h = sh_price[i]
            cur_h_idx = float(i)
        if sl[i] == 1:
            cur_l = sl_price[i]
            cur_l_idx = float(i)

        last_swing_high[i] = cur_h
        last_swing_low[i] = cur_l
        last_swing_high_idx[i] = cur_h_idx
        last_swing_low_idx[i] = cur_l_idx

    idx = np.arange(n, dtype=float)
    swing_high_age = idx - last_swing_high_idx
    swing_low_age = idx - last_swing_low_idx

    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = sh_price
    out["swing_low_price"] = sl_price
    out["swing_high_idx"] = sh_idx
    out["swing_low_idx"] = sl_idx

    out["swing_high_detect_flag"] = sh_detect_flag
    out["swing_low_detect_flag"] = sl_detect_flag
    out["swing_high_detect_idx"] = sh_detect_idx
    out["swing_low_detect_idx"] = sl_detect_idx

    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low
    out["last_swing_high_idx"] = last_swing_high_idx
    out["last_swing_low_idx"] = last_swing_low_idx
    out["swing_high_age"] = swing_high_age
    out["swing_low_age"] = swing_low_age

    out["swing_high_prominence"] = sh_prom
    out["swing_low_prominence"] = sl_prom
    out["swing_high_prominence_atr"] = sh_prom_atr
    out["swing_low_prominence_atr"] = sl_prom_atr
    out["swing_high_strength"] = sh_strength
    out["swing_low_strength"] = sl_strength

    return out


def add_swings_causal(
    df: pd.DataFrame,
    window: int = 6,
    *,
    atr_length: int = 14,
    min_prominence_atr: float = 0.0,
    min_separation: int = 1,
    require_rejection: bool = False,
    min_wick_frac: float = 0.0,
    max_body_frac: float = 1.0,
) -> pd.DataFrame:
    """Backward-compatible alias for the canonical causal swing detector."""
    return add_swings(
        df,
        window=window,
        atr_length=atr_length,
        min_prominence_atr=min_prominence_atr,
        min_separation=min_separation,
        require_rejection=require_rejection,
        min_wick_frac=min_wick_frac,
        max_body_frac=max_body_frac,
    )
