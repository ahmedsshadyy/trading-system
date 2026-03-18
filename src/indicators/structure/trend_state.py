"""
structure/trend_state.py

Trend state machine: tracks HH/HL (bullish) and LH/LL (bearish) sequences.

Requires swing detection (add_swings) to have run first.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_trend_state(df: pd.DataFrame) -> pd.DataFrame:
    """Track consecutive HH/HL (bullish) and LH/LL (bearish).

    Requires ``add_swings()`` first.
    Columns: ``trend_state`` (1 bull / −1 bear / 0 undefined), ``hh_count``, ``ll_count``.
    """
    out = df.copy()
    n = len(out)

    sh_prices = out["swing_high_price"].values.astype(float)
    sl_prices = out["swing_low_price"].values.astype(float)

    trend = np.zeros(n, dtype=np.int8)
    hh_count = np.zeros(n, dtype=np.int16)
    ll_count = np.zeros(n, dtype=np.int16)

    prev_sh = np.nan
    prev_sl = np.nan
    curr_trend = 0
    curr_hh = 0
    curr_ll = 0

    for i in range(n):
        if not np.isnan(sh_prices[i]):
            if not np.isnan(prev_sh):
                if sh_prices[i] > prev_sh:
                    curr_hh += 1
                    curr_ll = 0
                elif sh_prices[i] < prev_sh:
                    curr_ll += 1
                    curr_hh = 0
            prev_sh = sh_prices[i]

        if not np.isnan(sl_prices[i]):
            if not np.isnan(prev_sl):
                if sl_prices[i] > prev_sl:
                    curr_hh = max(curr_hh, 1)
                elif sl_prices[i] < prev_sl:
                    curr_ll = max(curr_ll, 1)
            prev_sl = sl_prices[i]

        if curr_hh >= 2:
            curr_trend = 1
        elif curr_ll >= 2:
            curr_trend = -1

        trend[i] = curr_trend
        hh_count[i] = curr_hh
        ll_count[i] = curr_ll

    out["trend_state"] = trend
    out["hh_count"] = hh_count
    out["ll_count"] = ll_count
    return out
