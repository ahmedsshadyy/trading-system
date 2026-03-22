# src/indicators/structure/bos.py

"""
Canonical causal BOS detector.

BOS is an event layer, not a regime/state model.

Rules
-----
- Uses only already-confirmed causal swings from add_swings().
- Canonical bullish BOS = close breaks above prior confirmed swing high.
- Canonical bearish BOS = close breaks below prior confirmed swing low.
- Wick breaks are diagnostics only, never canonical BOS by themselves.
- One canonical BOS per source swing level (deduplicated by source index).
- Quality filters are applied causally on the event bar.

This detector is "as live as swings":
it only relies on levels that are already available in the current row,
which themselves only become available after swing confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns


def add_bos(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_source_age_bars: int = 0,
    max_source_age_bars: int | None = None,
    min_break_distance_atr: float = 0.0,
    min_body_atr: float = 0.0,
    min_source_strength: float | None = None,
    require_trend_alignment: bool = False,
    allow_neutral_trend_breaks: bool = True,
) -> pd.DataFrame:
    """
    Detect sparse canonical BOS events from confirmed swing levels.

    Parameters
    ----------
    df:
        OHLC dataframe with causal swing outputs already present.
    atr_length:
        ATR length used for normalization.
    min_source_age_bars:
        Minimum age of the broken swing level.
    max_source_age_bars:
        Optional maximum age of the broken swing level.
    min_break_distance_atr:
        Minimum ATR-normalized close-through distance beyond source level.
    min_body_atr:
        Minimum ATR-normalized candle body on event bar.
    min_source_strength:
        Optional minimum required source swing strength if available.
    require_trend_alignment:
        If True, bullish BOS requires non-bear trend_state and bearish BOS
        requires non-bull trend_state.
    allow_neutral_trend_breaks:
        Only relevant if require_trend_alignment=True.
        If False, trend_state must already agree with BOS direction exactly.

    Returns
    -------
    DataFrame
        Original df plus BOS event and diagnostics columns.
    """
    out = df.copy()

    require_columns(
        out,
        {
            "open",
            "high",
            "low",
            "close",
            "last_swing_high",
            "last_swing_low",
            "last_swing_high_idx",
            "last_swing_low_idx",
            "swing_high_age",
            "swing_low_age",
        },
    )

    n = len(out)
    if n == 0:
        return out

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    last_sh = out["last_swing_high"].to_numpy(dtype=float)
    last_sl = out["last_swing_low"].to_numpy(dtype=float)
    last_sh_idx = out["last_swing_high_idx"].to_numpy(dtype=float)
    last_sl_idx = out["last_swing_low_idx"].to_numpy(dtype=float)
    sh_age = out["swing_high_age"].to_numpy(dtype=float)
    sl_age = out["swing_low_age"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)

    trend_state = (
        out["trend_state"].to_numpy(dtype=float)
        if "trend_state" in out.columns
        else np.full(n, np.nan)
    )

    swing_high_strength = (
        out["swing_high_strength"].to_numpy(dtype=float)
        if "swing_high_strength" in out.columns
        else np.full(n, np.nan)
    )
    swing_low_strength = (
        out["swing_low_strength"].to_numpy(dtype=float)
        if "swing_low_strength" in out.columns
        else np.full(n, np.nan)
    )
    swing_high_prominence_atr = (
        out["swing_high_prominence_atr"].to_numpy(dtype=float)
        if "swing_high_prominence_atr" in out.columns
        else np.full(n, np.nan)
    )
    swing_low_prominence_atr = (
        out["swing_low_prominence_atr"].to_numpy(dtype=float)
        if "swing_low_prominence_atr" in out.columns
        else np.full(n, np.nan)
    )

    bos_bull = np.zeros(n, dtype=np.int8)
    bos_bear = np.zeros(n, dtype=np.int8)
    bos_direction = np.zeros(n, dtype=np.int8)
    bos_event_id = np.zeros(n, dtype=np.int32)

    bos_source_side = np.zeros(
        n, dtype=np.int8
    )  # +1 broken swing high, -1 broken swing low
    bos_source_idx = np.full(n, np.nan)
    bos_source_price = np.full(n, np.nan)
    bos_level = np.full(n, np.nan)

    bos_close_break_bull = np.zeros(n, dtype=np.int8)
    bos_close_break_bear = np.zeros(n, dtype=np.int8)
    bos_wick_break_bull = np.zeros(n, dtype=np.int8)
    bos_wick_break_bear = np.zeros(n, dtype=np.int8)

    bos_raw_candidate_bull = np.zeros(n, dtype=np.int8)
    bos_raw_candidate_bear = np.zeros(n, dtype=np.int8)

    bos_pass_source_age_bull = np.zeros(n, dtype=np.int8)
    bos_pass_source_age_bear = np.zeros(n, dtype=np.int8)
    bos_pass_break_distance_bull = np.zeros(n, dtype=np.int8)
    bos_pass_break_distance_bear = np.zeros(n, dtype=np.int8)
    bos_pass_body_bull = np.zeros(n, dtype=np.int8)
    bos_pass_body_bear = np.zeros(n, dtype=np.int8)
    bos_pass_source_strength_bull = np.zeros(n, dtype=np.int8)
    bos_pass_source_strength_bear = np.zeros(n, dtype=np.int8)
    bos_pass_trend_bull = np.zeros(n, dtype=np.int8)
    bos_pass_trend_bear = np.zeros(n, dtype=np.int8)

    bos_break_distance = np.full(n, np.nan)
    bos_break_distance_atr = np.full(n, np.nan)
    bos_candle_body_atr = np.full(n, np.nan)
    bos_source_age = np.full(n, np.nan)
    bos_source_strength = np.full(n, np.nan)

    broken_bull_sources: set[int] = set()
    broken_bear_sources: set[int] = set()
    event_counter = 0

    def _valid_source(level: float, idx_val: float, i: int) -> bool:
        return np.isfinite(level) and np.isfinite(idx_val) and int(idx_val) < i

    def _age_pass(age_val: float) -> bool:
        if not np.isfinite(age_val):
            return False
        if age_val < float(min_source_age_bars):
            return False
        if max_source_age_bars is not None and age_val > float(max_source_age_bars):
            return False
        return True

    def _body_atr(i: int) -> float:
        if np.isfinite(atr[i]) and atr[i] > 0:
            return abs(c[i] - o[i]) / atr[i]
        return np.nan

    def _break_dist_atr(level: float, i: int, bull: bool) -> tuple[float, float]:
        if bull:
            dist = c[i] - level
        else:
            dist = level - c[i]
        if np.isfinite(dist) and np.isfinite(atr[i]) and atr[i] > 0:
            return float(dist), float(dist / atr[i])
        return float(dist), np.nan

    def _source_strength(src_idx: int, bull: bool) -> float:
        if src_idx < 0 or src_idx >= n:
            return np.nan
        if bull:
            val = swing_high_strength[src_idx]
            if not np.isfinite(val):
                val = swing_high_prominence_atr[src_idx]
        else:
            val = swing_low_strength[src_idx]
            if not np.isfinite(val):
                val = swing_low_prominence_atr[src_idx]
        return float(val) if np.isfinite(val) else np.nan

    def _trend_pass(i: int, bull: bool) -> bool:
        if not require_trend_alignment:
            return True
        ts = trend_state[i]
        if not np.isfinite(ts):
            return allow_neutral_trend_breaks
        ts_i = int(ts)
        if bull:
            return ts_i == 1 or (allow_neutral_trend_breaks and ts_i == 0)
        return ts_i == -1 or (allow_neutral_trend_breaks and ts_i == 0)

    for i in range(n):
        bull_source_ok = _valid_source(last_sh[i], last_sh_idx[i], i)
        bear_source_ok = _valid_source(last_sl[i], last_sl_idx[i], i)

        wick_bull = bool(bull_source_ok and h[i] > last_sh[i])
        close_bull = bool(bull_source_ok and c[i] > last_sh[i])
        wick_bear = bool(bear_source_ok and l[i] < last_sl[i])
        close_bear = bool(bear_source_ok and c[i] < last_sl[i])

        bos_wick_break_bull[i] = int(wick_bull)
        bos_close_break_bull[i] = int(close_bull)
        bos_wick_break_bear[i] = int(wick_bear)
        bos_close_break_bear[i] = int(close_bear)

        bull_raw = close_bull
        bear_raw = close_bear

        # Ambiguous pathological case guard
        if bull_raw and bear_raw:
            bull_raw = False
            bear_raw = False

        bos_raw_candidate_bull[i] = int(bull_raw)
        bos_raw_candidate_bear[i] = int(bear_raw)

        if bull_raw:
            src_idx = int(last_sh_idx[i])
            level = float(last_sh[i])
            age = float(sh_age[i])
            strength = _source_strength(src_idx, bull=True)
            body_atr = _body_atr(i)
            break_dist, break_dist_atr = _break_dist_atr(level, i, bull=True)

            pass_age = _age_pass(age)
            pass_dist = bool(
                np.isfinite(break_dist_atr) and break_dist_atr >= min_break_distance_atr
            )
            pass_body = bool(np.isfinite(body_atr) and body_atr >= min_body_atr)
            pass_strength = (
                True
                if min_source_strength is None
                else bool(np.isfinite(strength) and strength >= min_source_strength)
            )
            pass_trend = _trend_pass(i, bull=True)

            bos_pass_source_age_bull[i] = int(pass_age)
            bos_pass_break_distance_bull[i] = int(pass_dist)
            bos_pass_body_bull[i] = int(pass_body)
            bos_pass_source_strength_bull[i] = int(pass_strength)
            bos_pass_trend_bull[i] = int(pass_trend)

            if (
                src_idx not in broken_bull_sources
                and pass_age
                and pass_dist
                and pass_body
                and pass_strength
                and pass_trend
            ):
                broken_bull_sources.add(src_idx)
                event_counter += 1

                bos_bull[i] = 1
                bos_direction[i] = 1
                bos_event_id[i] = event_counter

                bos_source_side[i] = 1
                bos_source_idx[i] = float(src_idx)
                bos_source_price[i] = level
                bos_level[i] = level

                bos_break_distance[i] = break_dist
                bos_break_distance_atr[i] = break_dist_atr
                bos_candle_body_atr[i] = body_atr
                bos_source_age[i] = age
                bos_source_strength[i] = strength

        if bear_raw:
            src_idx = int(last_sl_idx[i])
            level = float(last_sl[i])
            age = float(sl_age[i])
            strength = _source_strength(src_idx, bull=False)
            body_atr = _body_atr(i)
            break_dist, break_dist_atr = _break_dist_atr(level, i, bull=False)

            pass_age = _age_pass(age)
            pass_dist = bool(
                np.isfinite(break_dist_atr) and break_dist_atr >= min_break_distance_atr
            )
            pass_body = bool(np.isfinite(body_atr) and body_atr >= min_body_atr)
            pass_strength = (
                True
                if min_source_strength is None
                else bool(np.isfinite(strength) and strength >= min_source_strength)
            )
            pass_trend = _trend_pass(i, bull=False)

            bos_pass_source_age_bear[i] = int(pass_age)
            bos_pass_break_distance_bear[i] = int(pass_dist)
            bos_pass_body_bear[i] = int(pass_body)
            bos_pass_source_strength_bear[i] = int(pass_strength)
            bos_pass_trend_bear[i] = int(pass_trend)

            if (
                src_idx not in broken_bear_sources
                and pass_age
                and pass_dist
                and pass_body
                and pass_strength
                and pass_trend
            ):
                broken_bear_sources.add(src_idx)
                event_counter += 1

                bos_bear[i] = 1
                bos_direction[i] = -1
                bos_event_id[i] = event_counter

                bos_source_side[i] = -1
                bos_source_idx[i] = float(src_idx)
                bos_source_price[i] = level
                bos_level[i] = level

                bos_break_distance[i] = break_dist
                bos_break_distance_atr[i] = break_dist_atr
                bos_candle_body_atr[i] = body_atr
                bos_source_age[i] = age
                bos_source_strength[i] = strength

    out["bos_bull"] = bos_bull
    out["bos_bear"] = bos_bear
    out["bos_direction"] = bos_direction
    out["bos_event_id"] = bos_event_id

    out["bos_source_side"] = bos_source_side
    out["bos_source_idx"] = bos_source_idx
    out["bos_source_price"] = bos_source_price
    out["bos_level"] = bos_level

    out["bos_close_break_bull"] = bos_close_break_bull
    out["bos_close_break_bear"] = bos_close_break_bear
    out["bos_wick_break_bull"] = bos_wick_break_bull
    out["bos_wick_break_bear"] = bos_wick_break_bear

    out["bos_raw_candidate_bull"] = bos_raw_candidate_bull
    out["bos_raw_candidate_bear"] = bos_raw_candidate_bear

    out["bos_pass_source_age_bull"] = bos_pass_source_age_bull
    out["bos_pass_source_age_bear"] = bos_pass_source_age_bear
    out["bos_pass_break_distance_bull"] = bos_pass_break_distance_bull
    out["bos_pass_break_distance_bear"] = bos_pass_break_distance_bear
    out["bos_pass_body_bull"] = bos_pass_body_bull
    out["bos_pass_body_bear"] = bos_pass_body_bear
    out["bos_pass_source_strength_bull"] = bos_pass_source_strength_bull
    out["bos_pass_source_strength_bear"] = bos_pass_source_strength_bear
    out["bos_pass_trend_bull"] = bos_pass_trend_bull
    out["bos_pass_trend_bear"] = bos_pass_trend_bear

    out["bos_break_distance"] = bos_break_distance
    out["bos_break_distance_atr"] = bos_break_distance_atr
    out["bos_candle_body_atr"] = bos_candle_body_atr
    out["bos_source_age"] = bos_source_age
    out["bos_source_strength"] = bos_source_strength

    return out
