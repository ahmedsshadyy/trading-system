"""
smc/ob.py

Enhanced Order Block (OB) detector.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.pivots import last_confirmed_swing_levels


def _ob_zone_from_candle(
    open_: float,
    high: float,
    low: float,
    close: float,
    side: str,
    mode: str = "openwick",
):
    """Compute OB zone boundaries.

    mode='openwick' (default): bull OB=[low, max(open,close)], bear OB=[min(open,close), high]
    mode='full': full wick range
    mode='body': body only
    """
    if mode == "body":
        return max(open_, close), min(open_, close)
    if mode == "openwick":
        if side == "bull":
            return max(open_, close), low
        return high, min(open_, close)
    return high, low


def add_ob(
    df: pd.DataFrame,
    impulse_candles: int = 4,
    impulse_atr_mult: float = 1.5,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    searchback: int = 10,
    min_same_dir_frac: float = 0.67,
    require_bos: bool = True,
    zone_mode: str = "openwick",
    max_ob_width_atr: float = 3.0,
) -> pd.DataFrame:
    """Enhanced Order Block detector.

    Columns: ``ob_bull``, ``ob_bear``, ``ob_bull_high/low``,
    ``ob_bear_high/low``, ``ob_width_atr``.
    """
    out = df.copy()
    n = len(out)
    o = out["open"].values.astype(float)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    atr = get_atr_array(out, atr_length)
    last_ph, last_pl = last_confirmed_swing_levels(h, lo, swing_left, swing_right)

    ob_bull = np.zeros(n, dtype=np.int8)
    ob_bear = np.zeros(n, dtype=np.int8)
    ob_bull_h = np.full(n, np.nan)
    ob_bull_l = np.full(n, np.nan)
    ob_bear_h = np.full(n, np.nan)
    ob_bear_l = np.full(n, np.nan)
    ob_w = np.full(n, np.nan)

    for i in range(max(impulse_candles - 1, 1), n):
        start = i - impulse_candles + 1
        if start < 0:
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        o_win = o[start : i + 1]
        c_win = c[start : i + 1]
        h_win = h[start : i + 1]
        l_win = lo[start : i + 1]

        bull_frac = np.sum(c_win > o_win) / impulse_candles
        bear_frac = np.sum(c_win < o_win) / impulse_candles

        bullish_impulse = (
            bull_frac >= min_same_dir_frac
            and c[i] - np.min(l_win) >= impulse_atr_mult * atr_i
            and np.sum(np.maximum(c_win - o_win, 0.0)) >= 0.8 * atr_i
            and c[i] > c[start]
        )
        bearish_impulse = (
            bear_frac >= min_same_dir_frac
            and np.max(h_win) - c[i] >= impulse_atr_mult * atr_i
            and np.sum(np.maximum(o_win - c_win, 0.0)) >= 0.8 * atr_i
            and c[i] < c[start]
        )

        if require_bos:
            bull_bos = (
                np.isfinite(last_ph[i]) and h[i] > last_ph[i] and c[i] > last_ph[i]
            )
            bear_bos = (
                np.isfinite(last_pl[i]) and lo[i] < last_pl[i] and c[i] < last_pl[i]
            )
        else:
            bull_bos = bullish_impulse
            bear_bos = bearish_impulse

        if bullish_impulse and bull_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] < o[k]:
                    z_hi, z_lo = _ob_zone_from_candle(
                        o[k], h[k], lo[k], c[k], "bull", zone_mode
                    )
                    w = (z_hi - z_lo) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(w) and w <= max_ob_width_atr:
                        ob_bull[k] = 1
                        ob_bull_h[k] = z_hi
                        ob_bull_l[k] = z_lo
                        if np.isnan(ob_w[k]):
                            ob_w[k] = w
                        break

        if bearish_impulse and bear_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] > o[k]:
                    z_hi, z_lo = _ob_zone_from_candle(
                        o[k], h[k], lo[k], c[k], "bear", zone_mode
                    )
                    w = (z_hi - z_lo) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(w) and w <= max_ob_width_atr:
                        ob_bear[k] = 1
                        ob_bear_h[k] = z_hi
                        ob_bear_l[k] = z_lo
                        if np.isnan(ob_w[k]):
                            ob_w[k] = w
                        break

    out["ob_bull"] = ob_bull
    out["ob_bear"] = ob_bear
    out["ob_bull_high"] = ob_bull_h
    out["ob_bull_low"] = ob_bull_l
    out["ob_bear_high"] = ob_bear_h
    out["ob_bear_low"] = ob_bear_l
    out["ob_width_atr"] = ob_w
    return out
