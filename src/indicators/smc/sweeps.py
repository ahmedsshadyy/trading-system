"""
smc/sweeps.py

Enhanced liquidity sweep detector.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.pivots import pivot_high, pivot_low


def add_liquidity_sweep(
    df: pd.DataFrame,
    atr_threshold: float = 0.2,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    level_tolerance_atr: float = 0.0,
    min_reclaim_atr: float = 0.0,
    min_wick_frac: float = 0.25,
    max_body_frac: float = 0.8,
    max_level_age: int | None = 80,
    require_next_bar_confirmation: bool = False,
    confirmation_mode: str = "close",
    use_provided_swings: bool = True,
) -> pd.DataFrame:
    """Enhanced liquidity sweep detector.

    A sweep is defined as:
    - price trades beyond a reference liquidity level (swing high/low)
    - then closes back inside the level on the same candle
    - optionally, the next bar confirms reversal

    Backward-compatible columns: sweep_high, sweep_low, sweep_magnitude
    Added columns: sweep_high_magnitude, sweep_low_magnitude, etc.
    """
    out = df.copy()
    n = len(out)

    req = {"open", "high", "low", "close"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(
            f"add_liquidity_sweep: missing required columns: {sorted(missing)}"
        )

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)

    # Reference swing levels
    if (
        use_provided_swings
        and "last_swing_high" in out.columns
        and "last_swing_low" in out.columns
    ):
        last_sh = out["last_swing_high"].to_numpy(dtype=float)
        last_sl = out["last_swing_low"].to_numpy(dtype=float)

        if "swing_high_age" in out.columns:
            sh_age = out["swing_high_age"].to_numpy(dtype=float)
        else:
            sh_age = np.full(n, np.nan)

        if "swing_low_age" in out.columns:
            sl_age = out["swing_low_age"].to_numpy(dtype=float)
        else:
            sl_age = np.full(n, np.nan)

    else:
        ph = pivot_high(h, left=swing_left, right=swing_right)
        pl = pivot_low(lo, left=swing_left, right=swing_right)

        last_sh = np.full(n, np.nan)
        last_sl = np.full(n, np.nan)
        sh_age = np.full(n, np.nan)
        sl_age = np.full(n, np.nan)

        cur_sh = np.nan
        cur_sl = np.nan
        cur_sh_idx = -1
        cur_sl_idx = -1

        for i in range(n):
            j = i - swing_right
            if j >= 0:
                if ph[j] == 1:
                    cur_sh = h[j]
                    cur_sh_idx = j
                if pl[j] == 1:
                    cur_sl = lo[j]
                    cur_sl_idx = j

            last_sh[i] = cur_sh
            last_sl[i] = cur_sl
            sh_age[i] = (i - cur_sh_idx) if cur_sh_idx >= 0 else np.nan
            sl_age[i] = (i - cur_sl_idx) if cur_sl_idx >= 0 else np.nan

    # Candle geometry
    rng = h - lo
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo

    with np.errstate(invalid="ignore", divide="ignore"):
        upper_wick_frac = np.where(rng > 0, upper_wick / rng, 0.0)
        lower_wick_frac = np.where(rng > 0, lower_wick / rng, 0.0)
        body_frac = np.where(rng > 0, body / rng, 0.0)

    # Output arrays
    sw_h = np.zeros(n, dtype=np.int8)
    sw_l = np.zeros(n, dtype=np.int8)
    sw_mag = np.full(n, np.nan)
    sw_h_mag = np.full(n, np.nan)
    sw_l_mag = np.full(n, np.nan)
    sw_h_reclaim = np.full(n, np.nan)
    sw_l_reclaim = np.full(n, np.nan)
    sw_h_wick_frac = np.full(n, np.nan)
    sw_l_wick_frac = np.full(n, np.nan)
    sw_h_body_frac = np.full(n, np.nan)
    sw_l_body_frac = np.full(n, np.nan)
    sw_level_high = np.full(n, np.nan)
    sw_level_low = np.full(n, np.nan)
    sw_level_high_age = np.full(n, np.nan)
    sw_level_low_age = np.full(n, np.nan)
    sw_conf_h = np.zeros(n, dtype=np.int8)
    sw_conf_l = np.zeros(n, dtype=np.int8)
    sw_side = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        tol = level_tolerance_atr * atr_i

        # High sweep
        if np.isfinite(last_sh[i]):
            level = last_sh[i]
            age_ok = (
                True
                if max_level_age is None
                else (np.isfinite(sh_age[i]) and sh_age[i] <= max_level_age)
            )

            wick_through = h[i] > (level + tol)
            close_back_inside = c[i] < (level - tol)
            excursion = h[i] - level
            excursion_atr = excursion / atr_i
            reclaim_atr = (level - c[i]) / atr_i
            wick_ok = upper_wick_frac[i] >= min_wick_frac
            body_ok = body_frac[i] <= max_body_frac

            high_sweep_now = (
                age_ok
                and wick_through
                and close_back_inside
                and excursion_atr >= atr_threshold
                and reclaim_atr >= min_reclaim_atr
                and wick_ok
                and body_ok
            )

            if high_sweep_now:
                confirmed = True
                if require_next_bar_confirmation and i + 1 < n:
                    if confirmation_mode == "mid":
                        confirmed = c[i + 1] < (o[i] + c[i]) / 2.0
                    elif confirmation_mode == "low":
                        confirmed = c[i + 1] < lo[i]
                    else:
                        confirmed = c[i + 1] < c[i]
                elif require_next_bar_confirmation and i + 1 >= n:
                    confirmed = False

                if confirmed:
                    sw_h[i] = 1
                    sw_h_mag[i] = excursion_atr
                    sw_h_reclaim[i] = reclaim_atr
                    sw_h_wick_frac[i] = upper_wick_frac[i]
                    sw_h_body_frac[i] = body_frac[i]
                    sw_level_high[i] = level
                    sw_level_high_age[i] = sh_age[i]
                    sw_conf_h[i] = 1
                    sw_side[i] = 1

        # Low sweep
        if np.isfinite(last_sl[i]):
            level = last_sl[i]
            age_ok = (
                True
                if max_level_age is None
                else (np.isfinite(sl_age[i]) and sl_age[i] <= max_level_age)
            )

            wick_through = lo[i] < (level - tol)
            close_back_inside = c[i] > (level + tol)
            excursion = level - lo[i]
            excursion_atr = excursion / atr_i
            reclaim_atr = (c[i] - level) / atr_i
            wick_ok = lower_wick_frac[i] >= min_wick_frac
            body_ok = body_frac[i] <= max_body_frac

            low_sweep_now = (
                age_ok
                and wick_through
                and close_back_inside
                and excursion_atr >= atr_threshold
                and reclaim_atr >= min_reclaim_atr
                and wick_ok
                and body_ok
            )

            if low_sweep_now:
                confirmed = True
                if require_next_bar_confirmation and i + 1 < n:
                    if confirmation_mode == "mid":
                        confirmed = c[i + 1] > (o[i] + c[i]) / 2.0
                    elif confirmation_mode == "high":
                        confirmed = c[i + 1] > h[i]
                    else:
                        confirmed = c[i + 1] > c[i]
                elif require_next_bar_confirmation and i + 1 >= n:
                    confirmed = False

                if confirmed:
                    sw_l[i] = 1
                    sw_l_mag[i] = excursion_atr
                    sw_l_reclaim[i] = reclaim_atr
                    sw_l_wick_frac[i] = lower_wick_frac[i]
                    sw_l_body_frac[i] = body_frac[i]
                    sw_level_low[i] = level
                    sw_level_low_age[i] = sl_age[i]
                    sw_conf_l[i] = 1
                    sw_side[i] = -1 if sw_side[i] == 0 else 0

        mags = []
        if np.isfinite(sw_h_mag[i]):
            mags.append(sw_h_mag[i])
        if np.isfinite(sw_l_mag[i]):
            mags.append(sw_l_mag[i])
        if mags:
            sw_mag[i] = np.nanmax(mags)

    out["sweep_high"] = sw_h
    out["sweep_low"] = sw_l
    out["sweep_magnitude"] = sw_mag
    out["sweep_high_magnitude"] = sw_h_mag
    out["sweep_low_magnitude"] = sw_l_mag
    out["sweep_high_reclaim_atr"] = sw_h_reclaim
    out["sweep_low_reclaim_atr"] = sw_l_reclaim
    out["sweep_high_wick_frac"] = sw_h_wick_frac
    out["sweep_low_wick_frac"] = sw_l_wick_frac
    out["sweep_high_body_frac"] = sw_h_body_frac
    out["sweep_low_body_frac"] = sw_l_body_frac
    out["sweep_level_high"] = sw_level_high
    out["sweep_level_low"] = sw_level_low
    out["sweep_level_high_age"] = sw_level_high_age
    out["sweep_level_low_age"] = sw_level_low_age
    out["sweep_confirmed_high"] = sw_conf_h
    out["sweep_confirmed_low"] = sw_conf_l
    out["sweep_side"] = sw_side

    return out
