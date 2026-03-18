"""
_helpers/arrays.py

Numpy-level array helpers: true range, ATR computation, ATR accessor.

These are low-level building blocks used by multiple detector modules.
No pandas dependency — pure numpy operations only.

The ``get_atr_array`` function replaces the old ``ensure_atr`` which
violated the purity contract by mutating the input DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range_np(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range on numpy arrays."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )


def atr_rma_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    """Wilder-style ATR on numpy arrays (self-reliant, no pandas)."""
    tr = true_range_np(high, low, close).astype(float)
    out = np.full(len(tr), np.nan, dtype=float)
    if len(tr) < length:
        csum = np.cumsum(tr)
        out[:] = csum / np.arange(1, len(tr) + 1)
        return out
    out[length - 1] = np.nanmean(tr[:length])
    alpha = 1.0 / length
    for i in range(length, len(tr)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    if length > 1:
        csum = np.cumsum(tr[: length - 1])
        out[: length - 1] = csum / np.arange(1, length)
    return out


def get_atr_array(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Return ATR array from DataFrame, computing if column not present.

    **Pure function** — never writes to the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have ``high``, ``low``, ``close`` columns.
    length : int
        ATR period. If a column named ``atr_{length}`` exists, returns it
        directly. Otherwise computes from OHLC.

    Returns
    -------
    np.ndarray
        ATR values, same length as df.
    """
    col = f"atr_{length}"
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    return atr_rma_np(
        df["high"].to_numpy(dtype=float),
        df["low"].to_numpy(dtype=float),
        df["close"].to_numpy(dtype=float),
        length=length,
    )
