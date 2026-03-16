"""
Market Regime Classifier.

Classifies 4H market state as TRENDING / RANGING / TRANSITIONAL
based on ADX, ATR, and Bollinger Band width — used as a gate layer
before any strategy logic executes.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify market regime on each candle.

    Requires columns from ``add_adx()``, ``add_bb_width()``, ``add_swings()``,
    and ``add_trend_state()``.

    Rules (from strategy_definitions.docx):
    * TRENDING (2):      ADX > 25 AND EMA 20 slope meaningful AND
                         consistent HH/HL or LH/LL (hh_count≥2 or ll_count≥2)
    * RANGING (0):       ADX < 20 AND bb_width_below_40 == 1
    * TRANSITIONAL (1):  ADX 20–25 OR BB width compressing to multi-candle low

    Columns
    ~~~~~~~
    * ``regime``       – ordinal: 0 = ranging, 1 = transitional, 2 = trending
    * ``regime_label`` – string label for debugging
    """
    out = df.copy()
    n = len(out)

    adx = out.get("adx_14", pd.Series(np.full(n, np.nan))).values.astype(float)
    bb_below_40 = out.get("bb_width_below_40", pd.Series(np.zeros(n))).values.astype(
        int
    )
    hh = out.get("hh_count", pd.Series(np.zeros(n))).values.astype(int)
    ll = out.get("ll_count", pd.Series(np.zeros(n))).values.astype(int)
    ema_slope = out.get("ema_20_slope", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )

    regime = np.ones(n, dtype=np.int8)  # default: transitional
    labels = np.full(n, "TRANSITIONAL", dtype=object)

    for i in range(n):
        if np.isnan(adx[i]):
            continue

        # Ranging
        if adx[i] < 20 and bb_below_40[i] == 1:
            regime[i] = 0
            labels[i] = "RANGING"
        # Trending
        elif (
            adx[i] > 25
            and (hh[i] >= 2 or ll[i] >= 2)
            and not np.isnan(ema_slope[i])
            and abs(ema_slope[i]) > 0.1
        ):
            regime[i] = 2
            labels[i] = "TRENDING"
        # else stays transitional

    out["regime"] = regime
    out["regime_label"] = labels
    return out
