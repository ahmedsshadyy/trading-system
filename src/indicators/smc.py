"""
SMC (Smart Money Concepts) structure detectors.

FVG detector, IFVG classifier, Order Block detector, liquidity sweep
detector, equal highs/lows detector, displacement candle detector,
AMD phase classifier.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FVG Detector
# ---------------------------------------------------------------------------


def add_fvg(df: pd.DataFrame, min_atr_mult: float = 0.3) -> pd.DataFrame:
    """Detect Fair Value Gaps (3-candle imbalance pattern).

    Bullish FVG: candle[i-1].high < candle[i+1].low
    Bearish FVG: candle[i-1].low  > candle[i+1].high

    Minimum gap size: ``min_atr_mult × ATR_14``.

    Uses 1-bar look-ahead — suitable for historical backtesting.

    Columns
    ~~~~~~~
    * ``fvg_bull``          – 1 on middle candle of bullish FVG
    * ``fvg_bear``          – 1 on middle candle of bearish FVG
    * ``fvg_bull_low``      – FVG zone low (candle[i−1] high)
    * ``fvg_bull_high``     – FVG zone high (candle[i+1] low)
    * ``fvg_bear_high``     – FVG zone high (candle[i−1] low)
    * ``fvg_bear_low``      – FVG zone low (candle[i+1] high)
    * ``fvg_size_atr``      – FVG size normalised by ATR
    """
    out = df.copy()
    n = len(out)
    h = out["high"].values.astype(float)
    l = out["low"].values.astype(float)

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    atr = out["atr_14"].values.astype(float)

    fb = np.zeros(n, dtype=np.int8)
    fr = np.zeros(n, dtype=np.int8)
    fb_lo = np.full(n, np.nan)
    fb_hi = np.full(n, np.nan)
    fr_hi = np.full(n, np.nan)
    fr_lo = np.full(n, np.nan)
    fvg_size = np.full(n, np.nan)

    for i in range(1, n - 1):
        # Bullish FVG
        gap_bull = l[i + 1] - h[i - 1]
        if gap_bull > 0 and gap_bull >= min_atr_mult * atr[i]:
            fb[i] = 1
            fb_lo[i] = h[i - 1]
            fb_hi[i] = l[i + 1]
            fvg_size[i] = gap_bull / atr[i] if atr[i] > 0 else np.nan

        # Bearish FVG
        gap_bear = l[i - 1] - h[i + 1]
        if gap_bear > 0 and gap_bear >= min_atr_mult * atr[i]:
            fr[i] = 1
            fr_hi[i] = l[i - 1]
            fr_lo[i] = h[i + 1]
            if np.isnan(fvg_size[i]):
                fvg_size[i] = gap_bear / atr[i] if atr[i] > 0 else np.nan

    out["fvg_bull"] = fb
    out["fvg_bear"] = fr
    out["fvg_bull_low"] = fb_lo
    out["fvg_bull_high"] = fb_hi
    out["fvg_bear_high"] = fr_hi
    out["fvg_bear_low"] = fr_lo
    out["fvg_size_atr"] = fvg_size

    return out


# ---------------------------------------------------------------------------
# FVG Fill Tracker
# ---------------------------------------------------------------------------


def add_fvg_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Track how much of the most recent active FVG has been filled.

    Iterates forward from each FVG and computes fill percentage as
    subsequent candles penetrate the zone.

    Columns
    ~~~~~~~
    * ``fvg_active_bull``   – 1 if a bullish FVG is still active (not fully filled)
    * ``fvg_active_bear``   – 1 if a bearish FVG is still active
    * ``fvg_fill_pct``      – fill percentage of the active FVG (0–1)
    * ``fvg_age``           – candles since the active FVG was created
    """
    out = df.copy()
    n = len(out)
    l = out["low"].values.astype(float)
    h = out["high"].values.astype(float)

    active_bull = np.zeros(n, dtype=np.int8)
    active_bear = np.zeros(n, dtype=np.int8)
    fill_pct = np.full(n, np.nan)
    age = np.full(n, np.nan)

    # Track most recent active FVG
    bull_zone = None  # (low, high, creation_idx)
    bear_zone = None

    fb = out.get("fvg_bull", pd.Series(np.zeros(n))).values
    fr = out.get("fvg_bear", pd.Series(np.zeros(n))).values
    fb_lo = out.get("fvg_bull_low", pd.Series(np.full(n, np.nan))).values
    fb_hi = out.get("fvg_bull_high", pd.Series(np.full(n, np.nan))).values
    fr_hi = out.get("fvg_bear_high", pd.Series(np.full(n, np.nan))).values
    fr_lo = out.get("fvg_bear_low", pd.Series(np.full(n, np.nan))).values

    for i in range(n):
        # Register new FVGs
        if fb[i] == 1 and not np.isnan(fb_lo[i]):
            bull_zone = (fb_lo[i], fb_hi[i], i)
        if fr[i] == 1 and not np.isnan(fr_hi[i]):
            bear_zone = (fr_lo[i], fr_hi[i], i)

        # Track bullish FVG fill
        if bull_zone is not None:
            zone_lo, zone_hi, ci = bull_zone
            zone_size = zone_hi - zone_lo
            if zone_size > 0:
                penetration = max(0, zone_hi - l[i])
                fp = min(penetration / zone_size, 1.0)
                if fp >= 1.0:
                    bull_zone = None  # fully filled
                else:
                    active_bull[i] = 1
                    fill_pct[i] = fp
                    age[i] = i - ci

        # Track bearish FVG fill
        if bear_zone is not None:
            zone_lo, zone_hi, ci = bear_zone
            zone_size = zone_hi - zone_lo
            if zone_size > 0:
                penetration = max(0, h[i] - zone_lo)
                fp = min(penetration / zone_size, 1.0)
                if fp >= 1.0:
                    bear_zone = None
                else:
                    active_bear[i] = 1
                    if np.isnan(fill_pct[i]):
                        fill_pct[i] = fp
                        age[i] = i - ci

    out["fvg_active_bull"] = active_bull
    out["fvg_active_bear"] = active_bear
    out["fvg_fill_pct"] = fill_pct
    out["fvg_age"] = age

    return out


# ---------------------------------------------------------------------------
# IFVG Classifier
# ---------------------------------------------------------------------------


def add_ifvg(df: pd.DataFrame) -> pd.DataFrame:
    """Classify FVGs that have been crossed from the other side → IFVG.

    A bullish FVG that price later breaks down through and then
    re-enters from below becomes a bearish IFVG (and vice versa).

    Columns
    ~~~~~~~
    * ``ifvg_bull``       – 1 when a bearish FVG has been inverted to bullish
    * ``ifvg_bear``       – 1 when a bullish FVG has been inverted to bearish
    * ``ifvg_width_atr``  – IFVG width / ATR
    """
    out = df.copy()
    n = len(out)
    c = out["close"].values.astype(float)

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    atr = out["atr_14"].values.astype(float)

    ifvg_bull = np.zeros(n, dtype=np.int8)
    ifvg_bear = np.zeros(n, dtype=np.int8)
    ifvg_width = np.full(n, np.nan)

    # Collect all FVG zones
    fb = out.get("fvg_bull", pd.Series(np.zeros(n))).values
    fr = out.get("fvg_bear", pd.Series(np.zeros(n))).values
    fb_lo = out.get("fvg_bull_low", pd.Series(np.full(n, np.nan))).values
    fb_hi = out.get("fvg_bull_high", pd.Series(np.full(n, np.nan))).values
    fr_hi = out.get("fvg_bear_high", pd.Series(np.full(n, np.nan))).values
    fr_lo = out.get("fvg_bear_low", pd.Series(np.full(n, np.nan))).values

    # Track active bull FVGs that could invert
    active_bull_fvgs = []  # (low, high, idx)
    active_bear_fvgs = []

    for i in range(n):
        if fb[i] == 1 and not np.isnan(fb_lo[i]):
            active_bull_fvgs.append((fb_lo[i], fb_hi[i], i))
        if fr[i] == 1 and not np.isnan(fr_lo[i]):
            active_bear_fvgs.append((fr_lo[i], fr_hi[i], i))

        # Check if close broke through a bullish FVG → it becomes bearish IFVG
        remaining = []
        for zone_lo, zone_hi, ci in active_bull_fvgs:
            if c[i] < zone_lo:
                # Broken below — now if price returns above zone_lo it's IFVG
                ifvg_bear[i] = 1
                width = zone_hi - zone_lo
                ifvg_width[i] = width / atr[i] if atr[i] > 0 else np.nan
            else:
                remaining.append((zone_lo, zone_hi, ci))
        active_bull_fvgs = remaining

        # Check if close broke through a bearish FVG → bullish IFVG
        remaining = []
        for zone_lo, zone_hi, ci in active_bear_fvgs:
            if c[i] > zone_hi:
                ifvg_bull[i] = 1
                width = zone_hi - zone_lo
                ifvg_width[i] = width / atr[i] if atr[i] > 0 else np.nan
            else:
                remaining.append((zone_lo, zone_hi, ci))
        active_bear_fvgs = remaining

    out["ifvg_bull"] = ifvg_bull
    out["ifvg_bear"] = ifvg_bear
    out["ifvg_width_atr"] = ifvg_width
    return out


# ---------------------------------------------------------------------------
# OB Internal Helpers
# ---------------------------------------------------------------------------


def _ob_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )


def _ob_atr_rma(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    """Wilder-style ATR for OB detector (self-reliant)."""
    tr = _ob_true_range(high, low, close).astype(float)
    out = np.full(len(tr), np.nan, dtype=float)
    if len(tr) < length:
        csum = np.cumsum(tr)
        out[:] = csum / np.arange(1, len(tr) + 1)
        return out
    out[length - 1] = np.nanmean(tr[:length])
    alpha = 1.0 / length
    for i in range(length, len(tr)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    csum = np.cumsum(tr[: length - 1])
    if length > 1:
        out[: length - 1] = csum / np.arange(1, length)
    return out


def _ob_pivot_high(high: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    n = len(high)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(high[i] > high[i - left : i]) and np.all(
            high[i] >= high[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def _ob_pivot_low(low: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(low[i] < low[i - left : i]) and np.all(
            low[i] <= low[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def _ob_last_confirmed_swings(
    high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2
):
    """For each bar i, the most recent confirmed pivot (knowable by bar i)."""
    n = len(high)
    ph = _ob_pivot_high(high, left, right)
    pl = _ob_pivot_low(low, left, right)
    last_ph = np.full(n, np.nan)
    last_pl = np.full(n, np.nan)
    cur_ph = np.nan
    cur_pl = np.nan
    for i in range(n):
        idx = i - right
        if idx >= 0:
            if ph[idx] == 1:
                cur_ph = high[idx]
            if pl[idx] == 1:
                cur_pl = low[idx]
        last_ph[i] = cur_ph
        last_pl[i] = cur_pl
    return last_ph, last_pl


def _ob_zone_from_candle(
    open_: float,
    high: float,
    low: float,
    close: float,
    side: str,
    mode: str = "openwick",
):
    """Compute OB zone boundaries.

    mode='openwick' (default, SMC style):
        bull OB (bearish candle): zone = [low, max(open, close)]
        bear OB (bullish candle): zone = [min(open, close), high]
    mode='full': full wick range
    mode='body': body only
    """
    body_low = min(open_, close)
    body_high = max(open_, close)
    if mode == "body":
        return body_high, body_low
    if mode == "openwick":
        if side == "bull":
            return max(open_, close), low
        return high, min(open_, close)
    # full
    return high, low


# ---------------------------------------------------------------------------
# Order Block Detector (Enhanced)
# ---------------------------------------------------------------------------


def add_ob(
    df: pd.DataFrame,
    impulse_candles: int = 4,
    impulse_atr_mult: float = 1.5,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    searchback: int = 10,
    min_same_dir_frac: float = 0.67,
    require_bos: bool = True,
    zone_mode: str = "openwick",
    max_ob_width_atr: float = 3.0,
) -> pd.DataFrame:
    """Detect Order Blocks — last opposite candle before a strong impulse.

    Enhanced version with flexible impulse detection, optional BOS
    requirement, configurable zone definition, and width filtering.

    Columns
    ~~~~~~~
    * ``ob_bull``          – 1 at the OB candle
    * ``ob_bear``          – 1 at the OB candle
    * ``ob_bull_high/low`` – OB zone boundaries
    * ``ob_bear_high/low`` – OB zone boundaries
    * ``ob_width_atr``     – OB width / ATR
    """
    out = df.copy()
    n = len(out)
    o = out["open"].values.astype(float)
    h = out["high"].values.astype(float)
    l = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    if "atr_14" in out.columns:
        atr = out["atr_14"].values.astype(float)
    else:
        atr = _ob_atr_rma(h, l, c, length=atr_length)
        out["atr_14"] = atr

    last_ph, last_pl = _ob_last_confirmed_swings(h, l, swing_left, swing_right)

    ob_bull = np.zeros(n, dtype=np.int8)
    ob_bear = np.zeros(n, dtype=np.int8)
    ob_bull_h = np.full(n, np.nan)
    ob_bull_l = np.full(n, np.nan)
    ob_bear_h = np.full(n, np.nan)
    ob_bear_l = np.full(n, np.nan)
    ob_w = np.full(n, np.nan)

    for i in range(max(impulse_candles - 1, 1), n):
        start = i - impulse_candles + 1
        if start < 0:
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        o_win = o[start : i + 1]
        c_win = c[start : i + 1]
        h_win = h[start : i + 1]
        l_win = l[start : i + 1]

        bull_count = np.sum(c_win > o_win)
        bear_count = np.sum(c_win < o_win)
        bull_frac = bull_count / impulse_candles
        bear_frac = bear_count / impulse_candles

        bull_displacement = c[i] - np.min(l_win)
        bear_displacement = np.max(h_win) - c[i]
        bull_body_sum = np.sum(np.maximum(c_win - o_win, 0.0))
        bear_body_sum = np.sum(np.maximum(o_win - c_win, 0.0))

        bullish_impulse = (
            bull_frac >= min_same_dir_frac
            and bull_displacement >= impulse_atr_mult * atr_i
            and bull_body_sum >= 0.8 * atr_i
            and c[i] > c[start]
        )
        bearish_impulse = (
            bear_frac >= min_same_dir_frac
            and bear_displacement >= impulse_atr_mult * atr_i
            and bear_body_sum >= 0.8 * atr_i
            and c[i] < c[start]
        )

        if require_bos:
            bull_bos = (
                np.isfinite(last_ph[i]) and h[i] > last_ph[i] and c[i] > last_ph[i]
            )
            bear_bos = (
                np.isfinite(last_pl[i]) and l[i] < last_pl[i] and c[i] < last_pl[i]
            )
        else:
            bull_bos = bullish_impulse
            bear_bos = bearish_impulse

        # Bullish OB
        if bullish_impulse and bull_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] < o[k]:
                    z_high, z_low = _ob_zone_from_candle(
                        o[k], h[k], l[k], c[k], "bull", zone_mode
                    )
                    width_atr = (z_high - z_low) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(width_atr) and width_atr <= max_ob_width_atr:
                        ob_bull[k] = 1
                        ob_bull_h[k] = z_high
                        ob_bull_l[k] = z_low
                        if np.isnan(ob_w[k]):
                            ob_w[k] = width_atr
                        break

        # Bearish OB
        if bearish_impulse and bear_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] > o[k]:
                    z_high, z_low = _ob_zone_from_candle(
                        o[k], h[k], l[k], c[k], "bear", zone_mode
                    )
                    width_atr = (z_high - z_low) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(width_atr) and width_atr <= max_ob_width_atr:
                        ob_bear[k] = 1
                        ob_bear_h[k] = z_high
                        ob_bear_l[k] = z_low
                        if np.isnan(ob_w[k]):
                            ob_w[k] = width_atr
                        break

    out["ob_bull"] = ob_bull
    out["ob_bear"] = ob_bear
    out["ob_bull_high"] = ob_bull_h
    out["ob_bull_low"] = ob_bull_l
    out["ob_bear_high"] = ob_bear_h
    out["ob_bear_low"] = ob_bear_l
    out["ob_width_atr"] = ob_w
    return out


# ---------------------------------------------------------------------------
# OB Mitigation Tracker (Enhanced — tracks multiple OBs)
# ---------------------------------------------------------------------------


def add_ob_mitigation(df: pd.DataFrame, *, keep_last_n: int = 3) -> pd.DataFrame:
    """Track whether recent OBs are still unmitigated.

    Tracks up to ``keep_last_n`` active OBs per side simultaneously.
    Avoids same-bar retest on OB creation candle.

    Columns
    ~~~~~~~
    * ``ob_unmitigated_bull`` – 1 if any bullish OB is unmitigated
    * ``ob_unmitigated_bear`` – 1 if any bearish OB is unmitigated
    * ``ob_first_retest``     – 1 on the candle of first retest
    """
    out = df.copy()
    n = len(out)
    h = out["high"].values.astype(float)
    l = out["low"].values.astype(float)

    ob_b = out.get("ob_bull", pd.Series(np.zeros(n))).values.astype(np.int8)
    ob_r = out.get("ob_bear", pd.Series(np.zeros(n))).values.astype(np.int8)
    ob_bh = out.get("ob_bull_high", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_bl = out.get("ob_bull_low", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_rh = out.get("ob_bear_high", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_rl = out.get("ob_bear_low", pd.Series(np.full(n, np.nan))).values.astype(float)

    unmit_bull = np.zeros(n, dtype=np.int8)
    unmit_bear = np.zeros(n, dtype=np.int8)
    first_retest = np.zeros(n, dtype=np.int8)

    # Each zone: {"idx": int, "high": float, "low": float, "retested": bool}
    bull_zones = []
    bear_zones = []

    for i in range(n):
        if ob_b[i] == 1 and np.isfinite(ob_bh[i]):
            bull_zones.append(
                {"idx": i, "high": ob_bh[i], "low": ob_bl[i], "retested": False}
            )
            bull_zones = bull_zones[-keep_last_n:]

        if ob_r[i] == 1 and np.isfinite(ob_rh[i]):
            bear_zones.append(
                {"idx": i, "high": ob_rh[i], "low": ob_rl[i], "retested": False}
            )
            bear_zones = bear_zones[-keep_last_n:]

        any_bull = False
        for z in bull_zones:
            if i <= z["idx"]:
                any_bull = True
                continue
            if l[i] <= z["high"]:  # price touched zone
                if not z["retested"]:
                    first_retest[i] = 1
                    z["retested"] = True
                if l[i] < z["low"]:  # broken through
                    z["idx"] = -999  # mark dead
                else:
                    any_bull = True
            else:
                any_bull = True

        any_bear = False
        for z in bear_zones:
            if i <= z["idx"]:
                any_bear = True
                continue
            if h[i] >= z["low"]:
                if not z["retested"]:
                    first_retest[i] = 1
                    z["retested"] = True
                if h[i] > z["high"]:
                    z["idx"] = -999
                else:
                    any_bear = True
            else:
                any_bear = True

        # Clean dead zones
        bull_zones = [z for z in bull_zones if z["idx"] != -999]
        bear_zones = [z for z in bear_zones if z["idx"] != -999]

        unmit_bull[i] = 1 if any_bull else 0
        unmit_bear[i] = 1 if any_bear else 0

    out["ob_unmitigated_bull"] = unmit_bull
    out["ob_unmitigated_bear"] = unmit_bear
    out["ob_first_retest"] = first_retest
    return out


# ---------------------------------------------------------------------------
# Liquidity Sweep Detector
# ---------------------------------------------------------------------------


def add_liquidity_sweep(df: pd.DataFrame, atr_threshold: float = 0.2) -> pd.DataFrame:
    """Detect liquidity sweeps: wick beyond a level + close back inside.

    Checks against ``last_swing_high`` and ``last_swing_low`` from
    ``add_swings()``.

    Columns
    ~~~~~~~
    * ``sweep_high``        – 1 when wick broke above last swing high but closed below
    * ``sweep_low``         – 1 when wick broke below last swing low but closed above
    * ``sweep_magnitude``   – distance of wick beyond level / ATR
    """
    out = df.copy()
    n = len(out)
    h = out["high"].values.astype(float)
    l = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    atr = out["atr_14"].values.astype(float)

    last_sh = out.get("last_swing_high", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )
    last_sl = out.get("last_swing_low", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )

    sw_h = np.zeros(n, dtype=np.int8)
    sw_l = np.zeros(n, dtype=np.int8)
    sw_mag = np.full(n, np.nan)

    for i in range(1, n):
        # Sweep of highs: wick above level, close below
        if not np.isnan(last_sh[i]) and h[i] > last_sh[i] and c[i] < last_sh[i]:
            mag = h[i] - last_sh[i]
            if atr[i] > 0 and mag >= atr_threshold * atr[i]:
                sw_h[i] = 1
                sw_mag[i] = mag / atr[i]

        # Sweep of lows
        if not np.isnan(last_sl[i]) and l[i] < last_sl[i] and c[i] > last_sl[i]:
            mag = last_sl[i] - l[i]
            if atr[i] > 0 and mag >= atr_threshold * atr[i]:
                sw_l[i] = 1
                if np.isnan(sw_mag[i]):
                    sw_mag[i] = mag / atr[i]

    out["sweep_high"] = sw_h
    out["sweep_low"] = sw_l
    out["sweep_magnitude"] = sw_mag
    return out


# ---------------------------------------------------------------------------
# Equal Highs / Lows Detector
# ---------------------------------------------------------------------------


def add_equal_hl(df: pd.DataFrame, atr_tolerance: float = 0.1) -> pd.DataFrame:
    """Detect clusters of equal highs or equal lows (liquidity pools).

    Two or more swing highs within ``atr_tolerance × ATR`` of each other.

    Columns
    ~~~~~~~
    * ``equal_highs``       – 1 at swing high that clusters with a prior one
    * ``equal_lows``        – 1 at swing low that clusters
    * ``equal_highs_count`` – count of clustered swing highs at this level
    * ``equal_lows_count``  – count of clustered swing lows at this level
    """
    out = df.copy()
    n = len(out)
    sh_price = out.get("swing_high_price", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )
    sl_price = out.get("swing_low_price", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    atr = out["atr_14"].values.astype(float)

    eq_h = np.zeros(n, dtype=np.int8)
    eq_l = np.zeros(n, dtype=np.int8)
    eq_h_cnt = np.zeros(n, dtype=np.int16)
    eq_l_cnt = np.zeros(n, dtype=np.int16)

    recent_sh = []  # list of (price, idx) for recent swing highs
    recent_sl = []

    for i in range(n):
        tol = atr_tolerance * atr[i] if not np.isnan(atr[i]) else 0

        if not np.isnan(sh_price[i]):
            cnt = sum(1 for p, _ in recent_sh if abs(p - sh_price[i]) <= tol)
            if cnt > 0:
                eq_h[i] = 1
                eq_h_cnt[i] = cnt + 1
            recent_sh.append((sh_price[i], i))
            # Keep only last 20 swings
            if len(recent_sh) > 20:
                recent_sh.pop(0)

        if not np.isnan(sl_price[i]):
            cnt = sum(1 for p, _ in recent_sl if abs(p - sl_price[i]) <= tol)
            if cnt > 0:
                eq_l[i] = 1
                eq_l_cnt[i] = cnt + 1
            recent_sl.append((sl_price[i], i))
            if len(recent_sl) > 20:
                recent_sl.pop(0)

    out["equal_highs"] = eq_h
    out["equal_lows"] = eq_l
    out["equal_highs_count"] = eq_h_cnt
    out["equal_lows_count"] = eq_l_cnt
    return out


# ---------------------------------------------------------------------------
# Displacement Candle Detector
# ---------------------------------------------------------------------------


def add_displacement_candle(
    df: pd.DataFrame, body_atr_mult: float = 1.5
) -> pd.DataFrame:
    """Flag candles with body ≥ ``body_atr_mult × ATR`` (strong conviction).

    Columns
    ~~~~~~~
    * ``displacement_candle``       – 1 if body ≥ threshold
    * ``displacement_body_atr``     – body / ATR
    * ``displacement_close_extreme``– 1 if close within 20% of range from extreme
    """
    out = df.copy()
    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    body = (out["close"] - out["open"]).abs().astype(float)
    atr = out["atr_14"].astype(float)
    rng = (out["high"] - out["low"]).astype(float)

    ratio = np.where(atr > 0, body / atr, 0.0)
    out["displacement_candle"] = (ratio >= body_atr_mult).astype(int)
    out["displacement_body_atr"] = ratio

    # Close near extreme: within 20% of range from the candle's directional extreme
    bull = out["close"].values >= out["open"].values
    dist_from_extreme = np.where(
        bull,
        out["high"].values - out["close"].values,
        out["close"].values - out["low"].values,
    ).astype(float)
    out["displacement_close_extreme"] = np.where(
        rng > 0, (dist_from_extreme / rng < 0.2).astype(int), 0
    )
    return out


# ---------------------------------------------------------------------------
# AMD Phase Classifier
# ---------------------------------------------------------------------------


def add_amd_phase(df: pd.DataFrame, accumulation_candles: int = 10) -> pd.DataFrame:
    """Classify current market phase: Accumulation / Manipulation / Distribution.

    Simplified rolling classification:
    * Accumulation: ATR below median for ``accumulation_candles``+ candles
    * Manipulation: displacement candle breaks out of accumulation range
    * Distribution: trending phase after manipulation

    Columns: ``amd_phase`` — ordinal: 0 = accumulation, 1 = manipulation, 2 = distribution.
    """
    out = df.copy()
    n = len(out)

    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    if "atr_pct_50" not in out.columns:
        atr = out["atr_14"]
        out["atr_pct_50"] = atr.rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

    atr_pct = out["atr_pct_50"].values.astype(float)
    disp = out.get("displacement_candle", pd.Series(np.zeros(n))).values

    phase = np.zeros(n, dtype=np.int8)
    low_atr_streak = 0

    for i in range(n):
        if np.isnan(atr_pct[i]):
            continue

        if atr_pct[i] < 50:
            low_atr_streak += 1
        else:
            low_atr_streak = 0

        if low_atr_streak >= accumulation_candles:
            phase[i] = 0  # accumulation
        elif disp[i] == 1 and low_atr_streak > 0:
            phase[i] = 1  # manipulation
            low_atr_streak = 0
        elif atr_pct[i] >= 50:
            phase[i] = 2  # distribution
        else:
            phase[i] = 0

    out["amd_phase"] = phase
    return out
