"""
_helpers/pivots.py

Pivot detection helpers: pivot high, pivot low, last confirmed swing levels.

These are internal swing-finding utilities used by OB detector and FVG
detector for their own BOS checks. They are independent of the main
``add_swings()`` function in ``structure/swings.py`` (or currently ``trend.py``).
"""

from __future__ import annotations

import numpy as np


def pivot_high(high: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    """Detect pivot highs using a symmetric window.

    Returns an int8 array: 1 at pivot high bars, 0 elsewhere.
    """
    n = len(high)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(high[i] > high[i - left : i]) and np.all(
            high[i] >= high[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def pivot_low(low: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    """Detect pivot lows using a symmetric window.

    Returns an int8 array: 1 at pivot low bars, 0 elsewhere.
    """
    n = len(low)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(low[i] < low[i - left : i]) and np.all(
            low[i] <= low[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def last_confirmed_swing_levels(
    high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """For each bar i, return the most recent confirmed pivot levels (knowable by bar i).

    A pivot at bar j is confirmed at bar j + right. So at bar i, the most
    recent knowable pivot is the latest one where j + right <= i.

    Returns
    -------
    last_ph : np.ndarray
        Last confirmed pivot high price for each bar.
    last_pl : np.ndarray
        Last confirmed pivot low price for each bar.
    """
    n = len(high)
    ph = pivot_high(high, left, right)
    pl = pivot_low(low, left, right)
    last_ph = np.full(n, np.nan)
    last_pl = np.full(n, np.nan)
    cur_ph = np.nan
    cur_pl = np.nan
    for i in range(n):
        j = i - right
        if j >= 0:
            if ph[j] == 1:
                cur_ph = high[j]
            if pl[j] == 1:
                cur_pl = low[j]
        last_ph[i] = cur_ph
        last_pl[i] = cur_pl
    return last_ph, last_pl
