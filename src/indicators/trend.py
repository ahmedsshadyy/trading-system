"""
Trend & Structure indicators.

EMA 20/50/200, ADX 14, Swing H/L detector, BOS detector, CHoCH detector,
trend state machine (HH/HL/LH/LL).

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators import ta_core as ta

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Swing High / Low Detector
# ---------------------------------------------------------------------------


def add_swings(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect swing highs and lows using a symmetric rolling window.

    A swing high at *i* means ``high[i]`` is the max in ``[i−window, i+window]``.
    Uses look-ahead — suitable for historical backtesting only.

    Columns
    ~~~~~~~
    * ``swing_high / swing_low``       – binary flags
    * ``swing_high_price / swing_low_price`` – price (NaN elsewhere)
    * ``last_swing_high / last_swing_low``   – forward-filled
    * ``swing_high_age / swing_low_age``     – candles since last swing
    """
    out = df.copy()
    n = len(out)
    highs = out["high"].values.astype(float)
    lows = out["low"].values.astype(float)

    sh = np.zeros(n, dtype=np.int8)
    sl = np.zeros(n, dtype=np.int8)

    for i in range(window, n - window):
        if (
            highs[i] >= highs[i - window : i].max()
            and highs[i] >= highs[i + 1 : i + window + 1].max()
        ):
            # Strict: must be strictly greater than at least one side
            if (
                highs[i] > highs[i - window : i].max()
                or highs[i] > highs[i + 1 : i + window + 1].max()
            ):
                sh[i] = 1

        if (
            lows[i] <= lows[i - window : i].min()
            and lows[i] <= lows[i + 1 : i + window + 1].min()
        ):
            if (
                lows[i] < lows[i - window : i].min()
                or lows[i] < lows[i + 1 : i + window + 1].min()
            ):
                sl[i] = 1

    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = np.where(sh == 1, highs, np.nan)
    out["swing_low_price"] = np.where(sl == 1, lows, np.nan)
    out["last_swing_high"] = pd.Series(out["swing_high_price"].values).ffill().values
    out["last_swing_low"] = pd.Series(out["swing_low_price"].values).ffill().values

    # Age: candles since last swing
    idx = np.arange(n, dtype=float)
    sh_idx = np.where(sh == 1, idx, np.nan)
    sl_idx = np.where(sl == 1, idx, np.nan)
    out["swing_high_age"] = idx - pd.Series(sh_idx).ffill().values
    out["swing_low_age"] = idx - pd.Series(sl_idx).ffill().values

    return out


# ---------------------------------------------------------------------------
# Swing High / Low Detector — Causal (no look-ahead)
# ---------------------------------------------------------------------------


def add_swings_causal(
    df: pd.DataFrame,
    window: int = 6,
    *,
    atr_length: int = 14,
    min_prominence_atr: float = 0.0,
    min_separation: int = 1,
    require_rejection: bool = False,
    min_wick_frac: float = 0.0,
    max_body_frac: float = 1.0,
) -> pd.DataFrame:
    """Causal swing detector — uses only past data, no look-ahead.

    A swing high at bar i means ``high[i]`` is the max of the last
    ``window`` bars. Suitable for both backtesting and live deployment.
    Produces the same output columns as ``add_swings()`` so downstream
    code (BOS, CHoCH, OB, sweeps) works identically.

    Parameters
    ----------
    window : int
        Lookback window (past bars only, no look-ahead).
    min_prominence_atr : float
        Minimum prominence above recent window, normalised by ATR.
    min_separation : int
        Minimum bars between consecutive same-side swings.
    require_rejection : bool
        If True, require wick/body geometry consistent with rejection.
    min_wick_frac : float
        Minimum wick fraction when require_rejection is True.
    max_body_frac : float
        Maximum body fraction when require_rejection is True.

    Columns (identical to add_swings for downstream compatibility)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``swing_high / swing_low``
    * ``swing_high_price / swing_low_price``
    * ``last_swing_high / last_swing_low``
    * ``swing_high_age / swing_low_age``
    """
    from src.indicators.ta_core import ensure_atr

    out = df.copy()
    n = len(out)
    o = out["open"].values.astype(float)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    atr = ensure_atr(out, atr_length)

    rng = h - lo
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        upper_wick_frac = np.where(rng > 0, upper_wick / rng, 0.0)
        lower_wick_frac = np.where(rng > 0, lower_wick / rng, 0.0)
        body_frac = np.where(rng > 0, body / rng, 0.0)

    sh = np.zeros(n, dtype=np.int8)
    sl = np.zeros(n, dtype=np.int8)

    last_sh_bar = -(10**9)
    last_sl_bar = -(10**9)

    for i in range(1, n):
        start = max(0, i - window + 1)
        if i - start < 1:
            continue

        hist_high = h[start:i]  # past only, excluding current
        hist_low = lo[start:i]
        prev_max = np.max(hist_high)
        prev_min = np.min(hist_low)
        atr_i = atr[i]
        atr_ok = np.isfinite(atr_i) and atr_i > 0

        # Swing high candidate
        if h[i] >= prev_max:
            prom = h[i] - prev_max
            prom_atr = prom / atr_i if atr_ok else 0.0
            prom_ok = min_prominence_atr <= 0 or prom_atr >= min_prominence_atr
            sep_ok = (i - last_sh_bar) > min_separation
            geom_ok = (not require_rejection) or (
                upper_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )
            if prom_ok and sep_ok and geom_ok:
                sh[i] = 1
                last_sh_bar = i

        # Swing low candidate
        if lo[i] <= prev_min:
            prom = prev_min - lo[i]
            prom_atr = prom / atr_i if atr_ok else 0.0
            prom_ok = min_prominence_atr <= 0 or prom_atr >= min_prominence_atr
            sep_ok = (i - last_sl_bar) > min_separation
            geom_ok = (not require_rejection) or (
                lower_wick_frac[i] >= min_wick_frac and body_frac[i] <= max_body_frac
            )
            if prom_ok and sep_ok and geom_ok:
                sl[i] = 1
                last_sl_bar = i

    # Build output columns — same names as add_swings()
    highs = h
    lows = lo
    out["swing_high"] = sh
    out["swing_low"] = sl
    out["swing_high_price"] = np.where(sh == 1, highs, np.nan)
    out["swing_low_price"] = np.where(sl == 1, lows, np.nan)

    # Causal forward-fill of last swing levels
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    cur_h = np.nan
    cur_l = np.nan

    for i in range(n):
        if sh[i] == 1:
            cur_h = highs[i]
        if sl[i] == 1:
            cur_l = lows[i]
        last_swing_high[i] = cur_h
        last_swing_low[i] = cur_l

    out["last_swing_high"] = last_swing_high
    out["last_swing_low"] = last_swing_low

    idx = np.arange(n, dtype=float)
    sh_idx_arr = np.where(sh == 1, idx, np.nan)
    sl_idx_arr = np.where(sl == 1, idx, np.nan)
    out["swing_high_age"] = idx - pd.Series(sh_idx_arr).ffill().values
    out["swing_low_age"] = idx - pd.Series(sl_idx_arr).ffill().values

    return out


# ---------------------------------------------------------------------------
# Trend State Machine  (HH / HL / LH / LL)
# ---------------------------------------------------------------------------


def add_trend_state(df: pd.DataFrame) -> pd.DataFrame:
    """Track consecutive HH/HL (bullish) and LH/LL (bearish).

    Requires ``add_swings()`` first.

    Columns: ``trend_state`` (1 bull / −1 bear / 0 undefined),
    ``hh_count``, ``ll_count``.
    """
    out = df.copy()
    n = len(out)

    sh_prices = out["swing_high_price"].values.astype(float)
    sl_prices = out["swing_low_price"].values.astype(float)

    trend = np.zeros(n, dtype=np.int8)
    hh_count = np.zeros(n, dtype=np.int16)
    ll_count = np.zeros(n, dtype=np.int16)

    prev_sh = np.nan
    prev_sl = np.nan
    curr_trend = 0
    curr_hh = 0
    curr_ll = 0

    for i in range(n):
        if not np.isnan(sh_prices[i]):
            if not np.isnan(prev_sh):
                if sh_prices[i] > prev_sh:
                    curr_hh += 1
                    curr_ll = 0
                elif sh_prices[i] < prev_sh:
                    curr_ll += 1
                    curr_hh = 0
            prev_sh = sh_prices[i]

        if not np.isnan(sl_prices[i]):
            if not np.isnan(prev_sl):
                if sl_prices[i] > prev_sl:
                    curr_hh = max(curr_hh, 1)
                elif sl_prices[i] < prev_sl:
                    curr_ll = max(curr_ll, 1)
            prev_sl = sl_prices[i]

        if curr_hh >= 2:
            curr_trend = 1
        elif curr_ll >= 2:
            curr_trend = -1

        trend[i] = curr_trend
        hh_count[i] = curr_hh
        ll_count[i] = curr_ll

    out["trend_state"] = trend
    out["hh_count"] = hh_count
    out["ll_count"] = ll_count
    return out


# ---------------------------------------------------------------------------
# BOS Detector
# ---------------------------------------------------------------------------


def add_bos(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Break of Structure (full candle close beyond swing).

    Requires ``add_swings()`` first.

    Columns
    ~~~~~~~
    * ``bos_bull / bos_bear``        – binary (fires once per broken level)
    * ``bos_direction``              – forward-filled: 1 / −1 / 0
    * ``bos_candle_body_atr``        – body / ATR on BOS candle (NaN elsewhere)
    * ``bos_swing_age``              – age of the broken swing (NaN elsewhere)
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


# ---------------------------------------------------------------------------
# CHoCH Detector
# ---------------------------------------------------------------------------


def add_choch(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Change of Character — first BOS against the prevailing trend.

    Requires ``add_swings()``, ``add_trend_state()``, ``add_bos()``.

    Columns: ``choch_bull``, ``choch_bear``, ``choch_candle_body_atr``.
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
