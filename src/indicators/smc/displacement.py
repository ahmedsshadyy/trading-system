"""
smc/displacement.py

Enhanced displacement candle detector.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array


def add_displacement_candle(
    df: pd.DataFrame,
    body_atr_mult: float = 1.5,
    *,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
) -> pd.DataFrame:
    """Enhanced displacement candle detector."""
    out = df.copy()
    atr = get_atr_array(out)
    atr_s = pd.Series(atr, index=out.index)

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    lo = out["low"].astype(float)
    c = out["close"].astype(float)

    body = (c - o).abs()
    rng = h - lo

    direction = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)
    bull = direction == 1
    bear = direction == -1

    with np.errstate(invalid="ignore", divide="ignore"):
        body_atr = np.where(atr_s > 0, body / atr_s, 0.0)
        body_frac_arr = np.where(rng > 0, body / rng, 0.0)

    dist_to_extreme = np.where(bull, h - c, np.where(bear, c - lo, np.nan)).astype(
        float
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        close_ext_frac = np.where(rng > 0, dist_to_extreme / rng, np.nan)
    close_extreme_ok = np.where(
        np.isfinite(close_ext_frac), close_ext_frac <= close_extreme_frac, False
    )

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo
    opp_wick = np.where(bull, lower_wick, np.where(bear, upper_wick, 0.0)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        opp_wick_frac = np.where(rng > 0, opp_wick / rng, 0.0)
    opp_wick_ok = opp_wick_frac <= max_opposite_wick_frac

    body_dom_ok = body_frac_arr >= min_body_frac

    valid_dir = direction != 0
    final = (
        valid_dir
        & (body_atr >= body_atr_mult)
        & close_extreme_ok
        & body_dom_ok
        & opp_wick_ok
    )

    out["displacement_candle"] = final.astype(int)
    out["displacement_body_atr"] = body_atr
    out["displacement_close_extreme"] = close_extreme_ok.astype(int)
    out["displacement_direction"] = direction
    out["displacement_bull_candle"] = (final & bull).astype(int)
    out["displacement_bear_candle"] = (final & bear).astype(int)

    return out
