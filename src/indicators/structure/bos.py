"""
structure/bos.py

Break of Structure (BOS) detector.

Requires swing detection (add_swings) to have run first.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators import ta_core as ta


def add_bos(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Break of Structure (full candle close beyond swing).

    Requires ``add_swings()`` first.
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    close = out["close"].values.astype(float)
    opn = out["open"].values.astype(float)
    last_sh = out["last_swing_high"].values.astype(float)
    last_sl = out["last_swing_low"].values.astype(float)
    sh_age = out["swing_high_age"].values.astype(float)
    sl_age = out["swing_low_age"].values.astype(float)
    atr = out["atr_14"].values.astype(float)
    body = np.abs(close - opn)

    n = len(out)
    bos_bull = np.zeros(n, dtype=np.int8)
    bos_bear = np.zeros(n, dtype=np.int8)

    prev_sh_level = np.nan
    prev_sl_level = np.nan

    for i in range(1, n):
        if (
            not np.isnan(last_sh[i])
            and close[i] > last_sh[i]
            and last_sh[i] != prev_sh_level
        ):
            bos_bull[i] = 1
            prev_sh_level = last_sh[i]

        if (
            not np.isnan(last_sl[i])
            and close[i] < last_sl[i]
            and last_sl[i] != prev_sl_level
        ):
            bos_bear[i] = 1
            prev_sl_level = last_sl[i]

    out["bos_bull"] = bos_bull
    out["bos_bear"] = bos_bear

    direction = np.where(bos_bull == 1, 1, np.where(bos_bear == 1, -1, np.nan))
    out["bos_direction"] = pd.Series(direction).ffill().fillna(0).astype(int).values

    bos_mask = (bos_bull == 1) | (bos_bear == 1)
    out["bos_candle_body_atr"] = np.where(
        bos_mask, np.where(atr > 0, body / atr, np.nan), np.nan
    )
    out["bos_swing_age"] = np.where(
        bos_bull == 1, sh_age, np.where(bos_bear == 1, sl_age, np.nan)
    )

    return out
