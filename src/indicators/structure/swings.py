"""
structure/swings.py

Canonical swing high/low detector — causal retrace-confirmed.

This is the single source of truth for structural swing detection across:
- research
- training
- backtesting
- live scanner
- live inference

Detection logic
---------------
A swing is detected in two stages:

1. Candidate formation:
   A new high exceeding the lookback window becomes a swing-high candidate.
   A new low below the lookback window becomes a swing-low candidate.

   Only one active candidate exists per side at a time.
   A higher high replaces the current unconfirmed swing-high candidate.
   A lower low replaces the current unconfirmed swing-low candidate.

2. Retrace confirmation:
   The candidate is promoted to a confirmed swing when price retraces at least
   ``min_retrace_atr × ATR`` from the candidate level.

Timing semantics
----------------
- Origin bar:
  where the extreme price actually occurred
- Confirm bar:
  where enough retrace occurred to make the swing knowable

Important downstream rule
-------------------------
Downstream consumers should use:
- confirmation-bar columns
- or running confirmed state (`last_swing_*`)

They must not treat origin-bar flags as if the swing were known immediately.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc


def add_swings(
    df: pd.DataFrame,
    window: int = 4,
    *,
    atr_length: int = 14,
    min_retrace_atr: float = 0.7,
    min_prominence_atr: float = 0.0,
    min_separation: int = 1,
    min_confirm_bars: int = 2,
) -> pd.DataFrame:
    """Canonical causal retrace-confirmed swing detector."""
    out = df.copy()
    require_ohlc(out)

    n = len(out)
    if n == 0:
        return out

    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    atr = get_atr_array(out, atr_length)

    # ------------------------------------------------------------------
    # Origin-bar annotation columns
    # ------------------------------------------------------------------
    sh = np.zeros(n, dtype=np.int8)
    sl = np.zeros(n, dtype=np.int8)

    sh_price = np.full(n, np.nan)
    sl_price = np.full(n, np.nan)

    sh_idx = np.full(n, -1, dtype=int)
    sl_idx = np.full(n, -1, dtype=int)

    sh_prom = np.full(n, np.nan)
    sl_prom = np.full(n, np.nan)

    sh_prom_atr = np.full(n, np.nan)
    sl_prom_atr = np.full(n, np.nan)

    sh_strength = np.full(n, np.nan)
    sl_strength = np.full(n, np.nan)

    sh_retrace_atr = np.full(n, np.nan)
    sl_retrace_atr = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Confirm-bar event columns (live-safe interface)
    # ------------------------------------------------------------------
    sh_confirm_flag = np.zeros(n, dtype=np.int8)
    sl_confirm_flag = np.zeros(n, dtype=np.int8)

    sh_confirm_origin_idx = np.full(n, np.nan)
    sl_confirm_origin_idx = np.full(n, np.nan)

    sh_confirm_price = np.full(n, np.nan)
    sl_confirm_price = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Confirmed running state
    # ------------------------------------------------------------------
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    last_swing_high_idx = np.full(n, np.nan)
    last_swing_low_idx = np.full(n, np.nan)

    # Detect bar of latest confirmed swing, used for age
    last_swing_high_detect_bar = np.full(n, np.nan)
    last_swing_low_detect_bar = np.full(n, np.nan)

    # ------------------------------------------------------------------
    # Active candidate tracking
    # ------------------------------------------------------------------
    cand_sh_bar = -1
    cand_sh_price = np.nan
    cand_sh_prom = np.nan
    cand_sh_prom_atr = np.nan
    cand_sh_atr_at_origin = np.nan

    cand_sl_bar = -1
    cand_sl_price = np.nan
    cand_sl_prom = np.nan
    cand_sl_prom_atr = np.nan
    cand_sl_atr_at_origin = np.nan

    last_confirmed_sh_bar = -(10**9)
    last_confirmed_sl_bar = -(10**9)

    cur_last_sh = np.nan
    cur_last_sl = np.nan
    cur_last_sh_idx = np.nan
    cur_last_sl_idx = np.nan
    cur_last_sh_detect = np.nan
    cur_last_sl_detect = np.nan

    for i in range(n):
        atr_i = atr[i]
        atr_ok = np.isfinite(atr_i) and atr_i > 0

        # ==============================================================
        # 1) Confirm active candidates if retrace is sufficient
        # ==============================================================

        # Swing high confirmation
        if cand_sh_bar >= 0 and (i - cand_sh_bar) >= min_confirm_bars:
            retrace = cand_sh_price - lo[i]
            threshold_atr = cand_sh_atr_at_origin
            if not (np.isfinite(threshold_atr) and threshold_atr > 0):
                threshold_atr = atr_i

            threshold = (
                min_retrace_atr * threshold_atr
                if np.isfinite(threshold_atr) and threshold_atr > 0
                else 0.0
            )

            if retrace >= threshold:
                sep_ok = (cand_sh_bar - last_confirmed_sh_bar) > min_separation

                if sep_ok:
                    origin = cand_sh_bar
                    actual_retrace_atr = (
                        retrace / threshold_atr
                        if np.isfinite(threshold_atr) and threshold_atr > 0
                        else np.nan
                    )

                    # Origin-bar annotations
                    sh[origin] = 1
                    sh_price[origin] = cand_sh_price
                    sh_idx[origin] = origin
                    sh_prom[origin] = cand_sh_prom
                    sh_prom_atr[origin] = cand_sh_prom_atr
                    sh_strength[origin] = cand_sh_prom_atr
                    sh_retrace_atr[origin] = actual_retrace_atr

                    # Confirm-bar event
                    sh_confirm_flag[i] = 1
                    sh_confirm_origin_idx[i] = float(origin)
                    sh_confirm_price[i] = cand_sh_price

                    # Running confirmed state
                    cur_last_sh = cand_sh_price
                    cur_last_sh_idx = float(origin)
                    cur_last_sh_detect = float(i)
                    last_confirmed_sh_bar = origin

                # Resolve candidate either way once retrace condition is met
                cand_sh_bar = -1
                cand_sh_price = np.nan
                cand_sh_prom = np.nan
                cand_sh_prom_atr = np.nan
                cand_sh_atr_at_origin = np.nan

        # Swing low confirmation
        if cand_sl_bar >= 0 and (i - cand_sl_bar) >= min_confirm_bars:
            retrace = h[i] - cand_sl_price
            threshold_atr = cand_sl_atr_at_origin
            if not (np.isfinite(threshold_atr) and threshold_atr > 0):
                threshold_atr = atr_i

            threshold = (
                min_retrace_atr * threshold_atr
                if np.isfinite(threshold_atr) and threshold_atr > 0
                else 0.0
            )

            if retrace >= threshold:
                sep_ok = (cand_sl_bar - last_confirmed_sl_bar) > min_separation

                if sep_ok:
                    origin = cand_sl_bar
                    actual_retrace_atr = (
                        retrace / threshold_atr
                        if np.isfinite(threshold_atr) and threshold_atr > 0
                        else np.nan
                    )

                    # Origin-bar annotations
                    sl[origin] = 1
                    sl_price[origin] = cand_sl_price
                    sl_idx[origin] = origin
                    sl_prom[origin] = cand_sl_prom
                    sl_prom_atr[origin] = cand_sl_prom_atr
                    sl_strength[origin] = cand_sl_prom_atr
                    sl_retrace_atr[origin] = actual_retrace_atr

                    # Confirm-bar event
                    sl_confirm_flag[i] = 1
                    sl_confirm_origin_idx[i] = float(origin)
                    sl_confirm_price[i] = cand_sl_price

                    # Running confirmed state
                    cur_last_sl = cand_sl_price
                    cur_last_sl_idx = float(origin)
                    cur_last_sl_detect = float(i)
                    last_confirmed_sl_bar = origin

                # Resolve candidate either way once retrace condition is met
                cand_sl_bar = -1
                cand_sl_price = np.nan
                cand_sl_prom = np.nan
                cand_sl_prom_atr = np.nan
                cand_sl_atr_at_origin = np.nan

        # ==============================================================
        # 2) Form/update candidates from current bar
        # ==============================================================

        if i >= 1:
            start = max(0, i - window + 1)
            if i - start >= 1:
                hist_high = h[start:i]
                hist_low = lo[start:i]

                prev_max = np.max(hist_high)
                prev_min = np.min(hist_low)

                # High candidate
                if h[i] >= prev_max:
                    prom = h[i] - prev_max
                    prom_atr = prom / atr_i if atr_ok else np.nan
                    prom_ok = (min_prominence_atr <= 0) or (
                        np.isfinite(prom_atr) and prom_atr >= min_prominence_atr
                    )

                    if prom_ok and (cand_sh_bar < 0 or h[i] >= cand_sh_price):
                        cand_sh_bar = i
                        cand_sh_price = h[i]
                        cand_sh_prom = prom
                        cand_sh_prom_atr = prom_atr
                        cand_sh_atr_at_origin = atr_i

                # Low candidate
                if lo[i] <= prev_min:
                    prom = prev_min - lo[i]
                    prom_atr = prom / atr_i if atr_ok else np.nan
                    prom_ok = (min_prominence_atr <= 0) or (
                        np.isfinite(prom_atr) and prom_atr >= min_prominence_atr
                    )

                    if prom_ok and (cand_sl_bar < 0 or lo[i] <= cand_sl_price):
                        cand_sl_bar = i
                        cand_sl_price = lo[i]
                        cand_sl_prom = prom
                        cand_sl_prom_atr = prom_atr
                        cand_sl_atr_at_origin = atr_i

        # ==============================================================
        # 3) Stamp confirmed running state on every bar
        # ==============================================================

        last_swing_high[i] = cur_last_sh
        last_swing_low[i] = cur_last_sl
        last_swing_high_idx[i] = cur_last_sh_idx
        last_swing_low_idx[i] = cur_last_sl_idx
        last_swing_high_detect_bar[i] = cur_last_sh_detect
        last_swing_low_detect_bar[i] = cur_last_sl_detect

    # ------------------------------------------------------------------
    # Build origin-bar detect_idx from confirm rows
    # ------------------------------------------------------------------
    sh_detect_idx = np.full(n, np.nan)
    sl_detect_idx = np.full(n, np.nan)

    for i in range(n):
        if sh_confirm_flag[i] == 1 and np.isfinite(sh_confirm_origin_idx[i]):
            origin = int(sh_confirm_origin_idx[i])
            if 0 <= origin < n:
                sh_detect_idx[origin] = float(i)

        if sl_confirm_flag[i] == 1 and np.isfinite(sl_confirm_origin_idx[i]):
            origin = int(sl_confirm_origin_idx[i])
            if 0 <= origin < n:
                sl_detect_idx[origin] = float(i)

    # ------------------------------------------------------------------
    # Age is based on confirmation bar, not origin bar
    # ------------------------------------------------------------------
    idx_arr = np.arange(n, dtype=float)
    swing_high_age = idx_arr - last_swing_high_detect_bar
    swing_low_age = idx_arr - last_swing_low_detect_bar

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    # Origin-bar annotations
    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = sh_price
    out["swing_low_price"] = sl_price
    out["swing_high_idx"] = sh_idx
    out["swing_low_idx"] = sl_idx

    # Backward-compatible reference to confirm index, stamped on origin bar
    out["swing_high_detect_idx"] = sh_detect_idx
    out["swing_low_detect_idx"] = sl_detect_idx

    # Confirm-bar event interface
    out["swing_high_confirm_flag"] = sh_confirm_flag
    out["swing_low_confirm_flag"] = sl_confirm_flag
    out["swing_high_confirm_origin_idx"] = sh_confirm_origin_idx
    out["swing_low_confirm_origin_idx"] = sl_confirm_origin_idx
    out["swing_high_confirm_price"] = sh_confirm_price
    out["swing_low_confirm_price"] = sl_confirm_price

    # Backward-compatible aliases
    out["swing_high_detect_flag"] = out["swing_high_confirm_flag"]
    out["swing_low_detect_flag"] = out["swing_low_confirm_flag"]

    # Confirmed running state
    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low
    out["last_swing_high_idx"] = last_swing_high_idx
    out["last_swing_low_idx"] = last_swing_low_idx
    out["swing_high_age"] = swing_high_age
    out["swing_low_age"] = swing_low_age

    # Quality on origin bar
    out["swing_high_prominence"] = sh_prom
    out["swing_low_prominence"] = sl_prom
    out["swing_high_prominence_atr"] = sh_prom_atr
    out["swing_low_prominence_atr"] = sl_prom_atr
    out["swing_high_strength"] = sh_strength
    out["swing_low_strength"] = sl_strength
    out["swing_high_retrace_atr"] = sh_retrace_atr
    out["swing_low_retrace_atr"] = sl_retrace_atr

    return out


def add_swings_causal(
    df: pd.DataFrame,
    window: int = 4,
    *,
    atr_length: int = 14,
    min_retrace_atr: float = 0.7,
    min_prominence_atr: float = 0.0,
    min_separation: int = 1,
    min_confirm_bars: int = 2,
) -> pd.DataFrame:
    return add_swings(
        df,
        window=window,
        atr_length=atr_length,
        min_retrace_atr=min_retrace_atr,
        min_prominence_atr=min_prominence_atr,
        min_separation=min_separation,
        min_confirm_bars=min_confirm_bars,
    )
