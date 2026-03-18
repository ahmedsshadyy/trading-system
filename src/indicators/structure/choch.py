"""
structure/choch.py

Change of Character (CHoCH) detector.

Requires swing detection, trend state, and BOS to have run first.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators import ta_core as ta


def add_choch(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Change of Character — first BOS against the prevailing trend.

    Requires ``add_swings()``, ``add_trend_state()``, ``add_bos()``.
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    trend = out["trend_state"].values
    bos_bull = out["bos_bull"].values
    bos_bear = out["bos_bear"].values
    atr = out["atr_14"].values.astype(float)
    body = np.abs(out["close"].values.astype(float) - out["open"].values.astype(float))
    n = len(out)

    choch_bull = np.zeros(n, dtype=np.int8)
    choch_bear = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        if trend[i - 1] == -1 and bos_bull[i] == 1:
            choch_bull[i] = 1
        if trend[i - 1] == 1 and bos_bear[i] == 1:
            choch_bear[i] = 1

    out["choch_bull"] = choch_bull
    out["choch_bear"] = choch_bear
    out["choch_candle_body_atr"] = np.where(
        (choch_bull == 1) | (choch_bear == 1),
        np.where(atr > 0, body / atr, np.nan),
        np.nan,
    )
    return out
