"""
_helpers/validators.py

Input validation helpers for indicator functions.

These raise clear error messages when required columns are missing,
preventing confusing KeyError exceptions deep in detector logic.
"""

from __future__ import annotations

import pandas as pd


def require_ohlc(df: pd.DataFrame, caller: str = "") -> None:
    """Verify DataFrame has open, high, low, close columns."""
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        prefix = f"{caller}: " if caller else ""
        raise ValueError(f"{prefix}missing required OHLC columns: {sorted(missing)}")


def require_ohlcv(df: pd.DataFrame, caller: str = "") -> None:
    """Verify DataFrame has open, high, low, close, volume columns."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        prefix = f"{caller}: " if caller else ""
        raise ValueError(f"{prefix}missing required OHLCV columns: {sorted(missing)}")


def require_columns(df: pd.DataFrame, columns: set[str], caller: str = "") -> None:
    """Verify DataFrame has all specified columns."""
    missing = columns - set(df.columns)
    if missing:
        prefix = f"{caller}: " if caller else ""
        raise ValueError(f"{prefix}missing required columns: {sorted(missing)}")
