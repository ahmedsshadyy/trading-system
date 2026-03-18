"""
foundation/adx.py

ADX indicator: add_adx.

Pure trailing indicator with no structural dependencies.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import pandas as pd
from src.indicators import ta_core as ta


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX value, threshold flags, and 3-candle delta.

    Columns: ``adx_14``, ``adx_above_20/25/40``, ``adx_delta_3``.
    """
    out = df.copy()
    adx_df = ta.adx(out["high"], out["low"], out["close"], length=period)
    out[f"adx_{period}"] = adx_df[f"ADX_{period}"]
    out["adx_above_20"] = (out[f"adx_{period}"] > 20).astype(int)
    out["adx_above_25"] = (out[f"adx_{period}"] > 25).astype(int)
    out["adx_above_40"] = (out[f"adx_{period}"] > 40).astype(int)
    out["adx_delta_3"] = out[f"adx_{period}"].diff(3)
    return out
