"""
foundation/ema.py

EMA indicators: compute_ema, add_emas.

Pure trailing indicators with no structural dependencies.
All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import pandas as pd
from src.indicators import ta_core as ta


def compute_ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Return EMA Series named ``ema_{period}``."""
    return ta.ema(df[col], length=period).rename(f"ema_{period}")


def add_emas(
    df: pd.DataFrame, periods: tuple[int, ...] = (20, 50, 200)
) -> pd.DataFrame:
    """Add EMA columns + slope and position flags.

    Per period P
    ~~~~~~~~~~~~
    * ``ema_{P}``              – EMA value
    * ``ema_{P}_slope``        – 3-candle Δ normalised by ATR-14
    * ``price_above_ema_{P}``  – binary

    Cross flags
    ~~~~~~~~~~~
    * ``ema_20_above_50``
    * ``ema_50_above_200``
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    for p in periods:
        ema = ta.ema(out["close"], length=p)
        out[f"ema_{p}"] = ema
        out[f"ema_{p}_slope"] = ema.diff(3) / out["atr_14"]
        out[f"price_above_ema_{p}"] = (out["close"] > ema).astype(int)

    if 20 in periods and 50 in periods:
        out["ema_20_above_50"] = (out["ema_20"] > out["ema_50"]).astype(int)
    if 50 in periods and 200 in periods:
        out["ema_50_above_200"] = (out["ema_50"] > out["ema_200"]).astype(int)

    return out
