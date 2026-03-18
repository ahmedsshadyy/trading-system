"""
structure/swings.py

Swing high/low detectors: symmetric and causal variants.

These are the structural backbone that BOS, CHoCH, trend state,
and all SMC detectors depend on.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array

# ---------------------------------------------------------------------------
# Swing High / Low Detector — Symmetric
# ---------------------------------------------------------------------------


def add_swings(
    df: pd.DataFrame,
    window: int = 3,
    *,
    causal: bool = True,
    atr_length: int = 14,
    min_prominence_atr: float = 0.0,
    min_separation: int = 0,
    tie_mode: str = "strict",
) -> pd.DataFrame:
    """Enhanced swing high/low detector using a symmetric rolling window."""
    out = df.copy()
    n = len(out)

    req = {"high", "low", "close"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"add_swings: missing required columns: {sorted(missing)}")

    highs = out["high"].to_numpy(dtype=float)
    lows = out["low"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)

    sh = np.zeros(n, dtype=np.int8)
    sl = np.zeros(n, dtype=np.int8)
    sh_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)
    sh_confirmed = np.zeros(n, dtype=np.int8)
    sl_confirmed = np.zeros(n, dtype=np.int8)
    sh_confirm_idx = np.full(n, -1, dtype=int)
    sl_confirm_idx = np.full(n, -1, dtype=int)
    sh_idx_arr = np.full(n, -1, dtype=int)
    sl_idx_arr = np.full(n, -1, dtype=int)
    sh_prom = np.full(n, np.nan)
    sl_prom = np.full(n, np.nan)
    sh_prom_atr = np.full(n, np.nan)
    sl_prom_atr = np.full(n, np.nan)
    sh_strength = np.full(n, np.nan)
    sl_strength = np.full(n, np.nan)

    last_sh_i = -(10**9)
    last_sl_i = -(10**9)

    for i in range(window, n - window):
        left_highs = highs[i - window : i]
        right_highs = highs[i + 1 : i + window + 1]
        left_lows = lows[i - window : i]
        right_lows = lows[i + 1 : i + window + 1]

        left_h_max = left_highs.max()
        right_h_max = right_highs.max()
        left_l_min = left_lows.min()
        right_l_min = right_lows.min()

        high_is_local_max = highs[i] >= left_h_max and highs[i] >= right_h_max
        if tie_mode == "strict":
            high_tie_ok = highs[i] > left_h_max or highs[i] > right_h_max
        else:
            high_tie_ok = True

        if high_is_local_max and high_tie_ok:
            prominence = highs[i] - max(left_h_max, right_h_max)
            prominence_atr = (
                prominence / atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
            )
            sep_ok = (i - last_sh_i) > min_separation
            prom_ok = (min_prominence_atr <= 0) or (
                np.isfinite(prominence_atr) and prominence_atr >= min_prominence_atr
            )

            if sep_ok and prom_ok:
                sh[i] = 1
                sh_price[i] = highs[i]
                sh_idx_arr[i] = i
                sh_prom[i] = prominence
                sh_prom_atr[i] = prominence_atr
                sh_strength[i] = prominence_atr
                if i + window < n:
                    sh_confirm_idx[i] = i + window
                    sh_confirmed[i + window] = 1
                last_sh_i = i

        low_is_local_min = lows[i] <= left_l_min and lows[i] <= right_l_min
        if tie_mode == "strict":
            low_tie_ok = lows[i] < left_l_min or lows[i] < right_l_min
        else:
            low_tie_ok = True

        if low_is_local_min and low_tie_ok:
            prominence = min(left_l_min, right_l_min) - lows[i]
            prominence_atr = (
                prominence / atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
            )
            sep_ok = (i - last_sl_i) > min_separation
            prom_ok = (min_prominence_atr <= 0) or (
                np.isfinite(prominence_atr) and prominence_atr >= min_prominence_atr
            )

            if sep_ok and prom_ok:
                sl[i] = 1
                sl_price[i] = lows[i]
                sl_idx_arr[i] = i
                sl_prom[i] = prominence
                sl_prom_atr[i] = prominence_atr
                sl_strength[i] = prominence_atr
                if i + window < n:
                    sl_confirm_idx[i] = i + window
                    sl_confirmed[i + window] = 1
                last_sl_i = i

    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = sh_price
    out["swing_low_price"] = sl_price
    out["swing_high_confirmed"] = sh_confirmed
    out["swing_low_confirmed"] = sl_confirmed
    out["swing_high_confirm_idx"] = sh_confirm_idx
    out["swing_low_confirm_idx"] = sl_confirm_idx
    out["swing_high_idx"] = sh_idx_arr
    out["swing_low_idx"] = sl_idx_arr
    out["swing_high_prominence"] = sh_prom
    out["swing_low_prominence"] = sl_prom
    out["swing_high_prominence_atr"] = sh_prom_atr
    out["swing_low_prominence_atr"] = sl_prom_atr
    out["swing_high_strength"] = sh_strength
    out["swing_low_strength"] = sl_strength

    # Build last confirmed swing levels causally
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_swing_high_idx = np.full(n, np.nan)
    last_swing_low_idx = np.full(n, np.nan)
    cur_h = np.nan
    cur_l = np.nan
    cur_h_idx = np.nan
    cur_l_idx = np.nan

    for i in range(n):
        if causal:
            confirmed_h_sources = np.where(sh_confirm_idx == i)[0]
            confirmed_l_sources = np.where(sl_confirm_idx == i)[0]

            if len(confirmed_h_sources) > 0:
                src = confirmed_h_sources[-1]
                cur_h = sh_price[src]
                cur_h_idx = float(src)

            if len(confirmed_l_sources) > 0:
                src = confirmed_l_sources[-1]
                cur_l = sl_price[src]
                cur_l_idx = float(src)
        else:
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

    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low
    out["last_swing_high_idx"] = last_swing_high_idx
    out["last_swing_low_idx"] = last_swing_low_idx

    idx = np.arange(n, dtype=float)
    out["swing_high_age"] = idx - last_swing_high_idx
    out["swing_low_age"] = idx - last_swing_low_idx

    return out


# ---------------------------------------------------------------------------
# Swing High / Low Detector — Causal (no look-ahead)
# ---------------------------------------------------------------------------


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
    """Causal swing detector — uses only past data, no look-ahead."""
    out = df.copy()
    n = len(out)
    o = out["open"].values.astype(float)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

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

        if h[i] >= prev_max:
            prom = h[i] - prev_max
            prom_atr = prom / atr_i if atr_ok else 0.0
            prom_ok = min_prominence_atr <= 0 or prom_atr >= min_prominence_atr
            sep_ok = (i - last_sh_bar) > min_separation
            geom_ok = (not require_rejection) or (
                upper_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )
            if prom_ok and sep_ok and geom_ok:
                sh[i] = 1
                last_sh_bar = i

        if lo[i] <= prev_min:
            prom = prev_min - lo[i]
            prom_atr = prom / atr_i if atr_ok else 0.0
            prom_ok = min_prominence_atr <= 0 or prom_atr >= min_prominence_atr
            sep_ok = (i - last_sl_bar) > min_separation
            geom_ok = (not require_rejection) or (
                lower_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )
            if prom_ok and sep_ok and geom_ok:
                sl[i] = 1
                last_sl_bar = i

    highs = h
    lows = lo
    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = np.where(sh == 1, highs, np.nan)
    out["swing_low_price"] = np.where(sl == 1, lows, np.nan)

    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    cur_h = np.nan
    cur_l = np.nan

    for i in range(n):
        if sh[i] == 1:
            cur_h = highs[i]
        if sl[i] == 1:
            cur_l = lows[i]
        last_swing_high[i] = cur_h
        last_swing_low[i] = cur_l

    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low

    idx = np.arange(n, dtype=float)
    sh_idx_arr = np.where(sh == 1, idx, np.nan)
    sl_idx_arr = np.where(sl == 1, idx, np.nan)
    out["swing_high_age"] = idx - pd.Series(sh_idx_arr).ffill().values
    out["swing_low_age"] = idx - pd.Series(sl_idx_arr).ffill().values

    return out
