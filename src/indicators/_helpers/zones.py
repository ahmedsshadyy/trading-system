"""
_helpers/zones.py

Zone-related utilities: overlap detection, boundary merging, body helpers.

These are used by FVG, OB, and other zone-type detectors.
"""

from __future__ import annotations

import numpy as np


def zones_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """Check if two price zones overlap."""
    return not (a_hi < b_lo or b_hi < a_lo)


def merge_zone_bounds(
    a_lo: float, a_hi: float, b_lo: float, b_hi: float
) -> tuple[float, float]:
    """Merge two overlapping zones into their union bounds."""
    return min(a_lo, b_lo), max(a_hi, b_hi)


def body_high(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Return the higher of open and close (candle body top)."""
    return np.maximum(open_, close)


def body_low(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Return the lower of open and close (candle body bottom)."""
    return np.minimum(open_, close)
