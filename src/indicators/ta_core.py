"""
Standard technical indicators — pure numpy/pandas implementations.

Zero external dependencies beyond numpy + pandas.
API-compatible with pandas-ta call signatures (accepts ``length=`` kwarg).

Uses proper Wilder's RMA (SMA-seeded recursive smoothing) for RSI, ATR,
and ADX to match TradingView exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_period(period: int | None, length: int | None, default: int) -> int:
    """Accept either ``period=`` or ``length=`` (pandas-ta compat)."""
    if length is not None:
        return length
    if period is not None:
        return period
    return default


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA) — matches TradingView exactly.

    Seed: SMA of first ``period`` non-NaN values.
    Recursion: rma[i] = (rma[i-1] * (period - 1) + series[i]) / period

    This is what TradingView calls ``ta.rma()`` in Pine Script.
    """
    vals = series.values.astype(float)
    n = len(vals)
    result = np.full(n, np.nan)
    alpha = 1.0 / period

    # Find first valid window for SMA seed
    count = 0
    seed_end = -1
    for i in range(n):
        if not np.isnan(vals[i]):
            count += 1
            if count == period:
                seed_end = i
                break

    if seed_end < 0:
        return pd.Series(result, index=series.index)

    # SMA seed
    seed_start = seed_end - period + 1
    result[seed_end] = np.nanmean(vals[seed_start : seed_end + 1])

    # Recursive smoothing
    for i in range(seed_end + 1, n):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = result[i - 1] * (1 - alpha) + vals[i] * alpha

    return pd.Series(result, index=series.index)


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------


def ema(
    series: pd.Series, period: int | None = None, *, length: int | None = None
) -> pd.Series:
    """Exponential Moving Average (matches TradingView ``ta.ema()``)."""
    p = _resolve_period(period, length, 20)
    return series.ewm(span=p, adjust=False).mean()


def sma(
    series: pd.Series, period: int | None = None, *, length: int | None = None
) -> pd.Series:
    """Simple Moving Average."""
    p = _resolve_period(period, length, 20)
    return series.rolling(p).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(
    close: pd.Series, period: int | None = None, *, length: int | None = None
) -> pd.Series:
    """Relative Strength Index using Wilder's RMA (matches TradingView)."""
    p = _resolve_period(period, length, 14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _rma(gain, p)
    avg_loss = _rma(loss, p)
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename(f"rsi_{p}")


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int | None = None,
    *,
    length: int | None = None,
) -> pd.Series:
    """Average True Range using Wilder's RMA (matches TradingView)."""
    p = _resolve_period(period, length, 14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _rma(tr, p).rename(f"atr_{p}")


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def _wilder_sum(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running sum — used for smoothed TR, +DM, -DM in ADX.

    Seed: simple sum of first ``period`` non-NaN values.
    Recursion: ws[i] = ws[i-1] - (ws[i-1] / period) + series[i]

    This is NOT the same as RMA (which divides the seed by period).
    TradingView's Pine ``ta.dmi()`` uses this for the DI calculation.
    """
    vals = series.values.astype(float)
    n = len(vals)
    result = np.full(n, np.nan)

    # Find first valid window for sum seed
    count = 0
    seed_end = -1
    for i in range(n):
        if not np.isnan(vals[i]):
            count += 1
            if count == period:
                seed_end = i
                break

    if seed_end < 0:
        return pd.Series(result, index=series.index)

    # Sum seed (not average)
    seed_start = seed_end - period + 1
    result[seed_end] = np.nansum(vals[seed_start : seed_end + 1])

    # Wilder's running sum
    for i in range(seed_end + 1, n):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = result[i - 1] - (result[i - 1] / period) + vals[i]

    return pd.Series(result, index=series.index)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int | None = None,
    *,
    length: int | None = None,
) -> pd.DataFrame:
    """Average Directional Index — matches TradingView's ``ta.dmi()``.

    Uses Wilder's running sum for smoothed TR/+DM/-DM (steps 1-3),
    then RMA for ADX smoothing (step 5).

    Returns DataFrame: ``ADX_{p}``, ``DMP_{p}``, ``DMN_{p}``.
    """
    p = _resolve_period(period, length, 14)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index
    )

    # Wilder's running sum (NOT RMA) for TR, +DM, -DM
    smooth_tr = _wilder_sum(tr, p)
    smooth_plus = _wilder_sum(plus_dm, p)
    smooth_minus = _wilder_sum(minus_dm, p)

    # +DI / -DI
    plus_di = pd.Series(
        np.where(smooth_tr > 0, 100 * smooth_plus / smooth_tr, 0.0),
        index=high.index,
    )
    minus_di = pd.Series(
        np.where(smooth_tr > 0, 100 * smooth_minus / smooth_tr, 0.0),
        index=high.index,
    )

    # DX → ADX (RMA smoothing)
    di_sum = plus_di + minus_di
    dx = pd.Series(
        np.where(di_sum > 0, (plus_di - minus_di).abs() / di_sum * 100, 0.0),
        index=high.index,
    )
    adx_val = _rma(dx, p)

    return pd.DataFrame(
        {f"ADX_{p}": adx_val, f"DMP_{p}": plus_di, f"DMN_{p}": minus_di},
        index=high.index,
    )


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD Line, Signal, Histogram.

    Returns DataFrame: ``MACD_``, ``MACDs_``, ``MACDh_`` columns.
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    tag = f"{fast}_{slow}_{signal}"
    return pd.DataFrame(
        {
            f"MACD_{tag}": macd_line,
            f"MACDs_{tag}": signal_line,
            f"MACDh_{tag}": histogram,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bbands(
    close: pd.Series,
    period: int | None = None,
    std: float = 2.0,
    *,
    length: int | None = None,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns DataFrame: ``BBL_``, ``BBM_``, ``BBU_`` columns.
    """
    p = _resolve_period(period, length, 20)
    mid = sma(close, p)
    rolling_std = close.rolling(p).std()
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std
    return pd.DataFrame(
        {f"BBL_{p}_{std}": lower, f"BBM_{p}_{std}": mid, f"BBU_{p}_{std}": upper},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Shared numpy-level helpers (used by trend.py and smc.py)
# ---------------------------------------------------------------------------


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


def ensure_atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Return ATR array from DataFrame, computing if column not present."""
    if "atr_14" in df.columns:
        return df["atr_14"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr_vals = atr_rma_np(h, lo, c, length=length)
    df["atr_14"] = atr_vals
    return atr_vals
