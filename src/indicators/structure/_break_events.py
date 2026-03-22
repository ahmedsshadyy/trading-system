"""
Private shared structural break-event engine.

This module centralizes the causal break mechanics shared by:
- BOS
- CHoCH

The raw break logic is identical. The only intended difference between
detectors is the trend filter applied at event acceptance time.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns

BreakTrendFilter = Literal["none", "same", "against"]


def _compute_structural_breaks(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_source_age_bars: int = 0,
    max_source_age_bars: int | None = None,
    min_break_distance_atr: float = 0.0,
    min_body_atr: float = 0.0,
    min_source_strength: float | None = None,
    trend_filter: BreakTrendFilter = "none",
    allow_neutral_trend_breaks: bool = True,
) -> dict[str, np.ndarray]:
    """Compute canonical causal structural break events.

    Returns suffix-keyed arrays that wrappers can prefix as BOS/CHoCH.
    """
    require_columns(
        df,
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

    n = len(df)
    if n == 0:
        return {}

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    last_sh = df["last_swing_high"].to_numpy(dtype=float)
    last_sl = df["last_swing_low"].to_numpy(dtype=float)
    last_sh_idx = df["last_swing_high_idx"].to_numpy(dtype=float)
    last_sl_idx = df["last_swing_low_idx"].to_numpy(dtype=float)
    sh_age = df["swing_high_age"].to_numpy(dtype=float)
    sl_age = df["swing_low_age"].to_numpy(dtype=float)

    atr = get_atr_array(df, atr_length)

    trend_state = (
        df["trend_state"].to_numpy(dtype=float)
        if "trend_state" in df.columns
        else np.full(n, np.nan)
    )

    swing_high_strength = (
        df["swing_high_strength"].to_numpy(dtype=float)
        if "swing_high_strength" in df.columns
        else np.full(n, np.nan)
    )
    swing_low_strength = (
        df["swing_low_strength"].to_numpy(dtype=float)
        if "swing_low_strength" in df.columns
        else np.full(n, np.nan)
    )
    swing_high_prominence_atr = (
        df["swing_high_prominence_atr"].to_numpy(dtype=float)
        if "swing_high_prominence_atr" in df.columns
        else np.full(n, np.nan)
    )
    swing_low_prominence_atr = (
        df["swing_low_prominence_atr"].to_numpy(dtype=float)
        if "swing_low_prominence_atr" in df.columns
        else np.full(n, np.nan)
    )

    result: dict[str, np.ndarray] = {
        "bull": np.zeros(n, dtype=np.int8),
        "bear": np.zeros(n, dtype=np.int8),
        "direction": np.zeros(n, dtype=np.int8),
        "event_id": np.zeros(n, dtype=np.int32),
        "source_side": np.zeros(n, dtype=np.int8),
        "source_idx": np.full(n, np.nan),
        "source_price": np.full(n, np.nan),
        "level": np.full(n, np.nan),
        "close_break_bull": np.zeros(n, dtype=np.int8),
        "close_break_bear": np.zeros(n, dtype=np.int8),
        "wick_break_bull": np.zeros(n, dtype=np.int8),
        "wick_break_bear": np.zeros(n, dtype=np.int8),
        "raw_candidate_bull": np.zeros(n, dtype=np.int8),
        "raw_candidate_bear": np.zeros(n, dtype=np.int8),
        "pass_source_age_bull": np.zeros(n, dtype=np.int8),
        "pass_source_age_bear": np.zeros(n, dtype=np.int8),
        "pass_break_distance_bull": np.zeros(n, dtype=np.int8),
        "pass_break_distance_bear": np.zeros(n, dtype=np.int8),
        "pass_body_bull": np.zeros(n, dtype=np.int8),
        "pass_body_bear": np.zeros(n, dtype=np.int8),
        "pass_source_strength_bull": np.zeros(n, dtype=np.int8),
        "pass_source_strength_bear": np.zeros(n, dtype=np.int8),
        "pass_trend_bull": np.zeros(n, dtype=np.int8),
        "pass_trend_bear": np.zeros(n, dtype=np.int8),
        "break_distance": np.full(n, np.nan),
        "break_distance_atr": np.full(n, np.nan),
        "candle_body_atr": np.full(n, np.nan),
        "candle_range_atr": np.full(n, np.nan),
        "upper_wick_atr": np.full(n, np.nan),
        "lower_wick_atr": np.full(n, np.nan),
        "body_to_range": np.full(n, np.nan),
        "close_location": np.full(n, np.nan),
        "gap_from_level_atr": np.full(n, np.nan),
        "displacement_score": np.full(n, np.nan),
        "source_age": np.full(n, np.nan),
        "source_strength": np.full(n, np.nan),
        "source_prominence_atr": np.full(n, np.nan),
        "source_fresh": np.zeros(n, dtype=np.int8),
        "source_stale": np.zeros(n, dtype=np.int8),
        "source_rank": np.full(n, np.nan),
        "source_in_trend_direction": np.zeros(n, dtype=np.int8),
    }

    broken_bull_sources: set[int] = set()
    broken_bear_sources: set[int] = set()
    confirmed_high_sources: list[int] = []
    confirmed_low_sources: list[int] = []
    prev_running_sh_idx = np.nan
    prev_running_sl_idx = np.nan
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
            return float(abs(c[i] - o[i]) / atr[i])
        return np.nan

    def _range_atr(i: int) -> float:
        candle_range = h[i] - l[i]
        if np.isfinite(candle_range) and np.isfinite(atr[i]) and atr[i] > 0:
            return float(candle_range / atr[i])
        return np.nan

    def _upper_wick_atr(i: int) -> float:
        wick = h[i] - max(o[i], c[i])
        if np.isfinite(wick) and np.isfinite(atr[i]) and atr[i] > 0:
            return float(max(wick, 0.0) / atr[i])
        return np.nan

    def _lower_wick_atr(i: int) -> float:
        wick = min(o[i], c[i]) - l[i]
        if np.isfinite(wick) and np.isfinite(atr[i]) and atr[i] > 0:
            return float(max(wick, 0.0) / atr[i])
        return np.nan

    def _body_to_range(i: int) -> float:
        candle_range = h[i] - l[i]
        if np.isfinite(candle_range) and candle_range > 0:
            return float(abs(c[i] - o[i]) / candle_range)
        return np.nan

    def _close_location(i: int) -> float:
        candle_range = h[i] - l[i]
        if np.isfinite(candle_range) and candle_range > 0:
            return float((c[i] - l[i]) / candle_range)
        return np.nan

    def _break_dist_atr(level: float, i: int, bull: bool) -> tuple[float, float]:
        dist = c[i] - level if bull else level - c[i]
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

    def _source_prominence(src_idx: int, bull: bool) -> float:
        if src_idx < 0 or src_idx >= n:
            return np.nan
        val = (
            swing_high_prominence_atr[src_idx]
            if bull
            else swing_low_prominence_atr[src_idx]
        )
        return float(val) if np.isfinite(val) else np.nan

    def _source_rank(src_idx: int, bull: bool) -> float:
        history = confirmed_high_sources if bull else confirmed_low_sources
        prominence_arr = swing_high_prominence_atr if bull else swing_low_prominence_atr
        strength_arr = swing_high_strength if bull else swing_low_strength

        if src_idx < 0 or src_idx >= n:
            return np.nan

        recent = history[-10:]
        if src_idx not in recent:
            recent = [*recent, src_idx][-10:]
        if not recent:
            return np.nan

        scored: list[tuple[int, float]] = []
        for idx_val in recent:
            score = prominence_arr[idx_val]
            if not np.isfinite(score):
                score = strength_arr[idx_val]
            if not np.isfinite(score):
                score = float("-inf")
            scored.append((idx_val, float(score)))

        ranked = sorted(scored, key=lambda item: (-item[1], recent.index(item[0])))
        for rank, (idx_val, _) in enumerate(ranked, start=1):
            if idx_val == src_idx:
                return float(rank)
        return np.nan

    def _trend_pass(i: int, bull: bool) -> bool:
        if trend_filter == "none":
            return True

        ts = trend_state[i]
        if not np.isfinite(ts):
            return allow_neutral_trend_breaks

        ts_i = int(ts)
        if ts_i == 0:
            return allow_neutral_trend_breaks

        if trend_filter == "same":
            return ts_i == (1 if bull else -1)
        return ts_i == (-1 if bull else 1)

    for i in range(n):
        if np.isfinite(last_sh_idx[i]):
            current_sh_idx = int(last_sh_idx[i])
            if not np.isfinite(prev_running_sh_idx) or current_sh_idx != int(
                prev_running_sh_idx
            ):
                confirmed_high_sources.append(current_sh_idx)
            prev_running_sh_idx = float(current_sh_idx)

        if np.isfinite(last_sl_idx[i]):
            current_sl_idx = int(last_sl_idx[i])
            if not np.isfinite(prev_running_sl_idx) or current_sl_idx != int(
                prev_running_sl_idx
            ):
                confirmed_low_sources.append(current_sl_idx)
            prev_running_sl_idx = float(current_sl_idx)

        bull_source_ok = _valid_source(last_sh[i], last_sh_idx[i], i)
        bear_source_ok = _valid_source(last_sl[i], last_sl_idx[i], i)

        wick_bull = bool(bull_source_ok and h[i] > last_sh[i])
        close_bull = bool(bull_source_ok and c[i] > last_sh[i])
        wick_bear = bool(bear_source_ok and l[i] < last_sl[i])
        close_bear = bool(bear_source_ok and c[i] < last_sl[i])

        result["wick_break_bull"][i] = int(wick_bull)
        result["close_break_bull"][i] = int(close_bull)
        result["wick_break_bear"][i] = int(wick_bear)
        result["close_break_bear"][i] = int(close_bear)

        bull_raw = close_bull
        bear_raw = close_bear
        if bull_raw and bear_raw:
            bull_raw = False
            bear_raw = False

        result["raw_candidate_bull"][i] = int(bull_raw)
        result["raw_candidate_bear"][i] = int(bear_raw)

        if bull_raw:
            src_idx = int(last_sh_idx[i])
            level = float(last_sh[i])
            age = float(sh_age[i])
            strength = _source_strength(src_idx, bull=True)
            prominence = _source_prominence(src_idx, bull=True)
            body_atr = _body_atr(i)
            range_atr = _range_atr(i)
            upper_wick_atr = _upper_wick_atr(i)
            lower_wick_atr = _lower_wick_atr(i)
            body_to_range = _body_to_range(i)
            close_location = _close_location(i)
            break_dist, break_dist_atr = _break_dist_atr(level, i, bull=True)
            source_rank = _source_rank(src_idx, bull=True)
            source_in_trend_direction = int(
                np.isfinite(trend_state[i]) and int(trend_state[i]) == 1
            )
            displacement_score = np.nan
            if (
                np.isfinite(body_atr)
                and np.isfinite(body_to_range)
                and np.isfinite(close_location)
            ):
                displacement_score = float(body_atr * body_to_range * close_location)

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

            result["pass_source_age_bull"][i] = int(pass_age)
            result["pass_break_distance_bull"][i] = int(pass_dist)
            result["pass_body_bull"][i] = int(pass_body)
            result["pass_source_strength_bull"][i] = int(pass_strength)
            result["pass_trend_bull"][i] = int(pass_trend)

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

                result["bull"][i] = 1
                result["direction"][i] = 1
                result["event_id"][i] = event_counter
                result["source_side"][i] = 1
                result["source_idx"][i] = float(src_idx)
                result["source_price"][i] = level
                result["level"][i] = level
                result["break_distance"][i] = break_dist
                result["break_distance_atr"][i] = break_dist_atr
                result["candle_body_atr"][i] = body_atr
                result["candle_range_atr"][i] = range_atr
                result["upper_wick_atr"][i] = upper_wick_atr
                result["lower_wick_atr"][i] = lower_wick_atr
                result["body_to_range"][i] = body_to_range
                result["close_location"][i] = close_location
                result["gap_from_level_atr"][i] = break_dist_atr
                result["displacement_score"][i] = displacement_score
                result["source_age"][i] = age
                result["source_strength"][i] = strength
                result["source_prominence_atr"][i] = prominence
                result["source_fresh"][i] = int(np.isfinite(age) and age <= 10.0)
                result["source_stale"][i] = int(np.isfinite(age) and age > 50.0)
                result["source_rank"][i] = source_rank
                result["source_in_trend_direction"][i] = source_in_trend_direction

        if bear_raw:
            src_idx = int(last_sl_idx[i])
            level = float(last_sl[i])
            age = float(sl_age[i])
            strength = _source_strength(src_idx, bull=False)
            prominence = _source_prominence(src_idx, bull=False)
            body_atr = _body_atr(i)
            range_atr = _range_atr(i)
            upper_wick_atr = _upper_wick_atr(i)
            lower_wick_atr = _lower_wick_atr(i)
            body_to_range = _body_to_range(i)
            close_location = _close_location(i)
            break_dist, break_dist_atr = _break_dist_atr(level, i, bull=False)
            source_rank = _source_rank(src_idx, bull=False)
            source_in_trend_direction = int(
                np.isfinite(trend_state[i]) and int(trend_state[i]) == -1
            )
            displacement_score = np.nan
            if (
                np.isfinite(body_atr)
                and np.isfinite(body_to_range)
                and np.isfinite(close_location)
            ):
                displacement_score = float(
                    body_atr * body_to_range * (1.0 - close_location)
                )

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

            result["pass_source_age_bear"][i] = int(pass_age)
            result["pass_break_distance_bear"][i] = int(pass_dist)
            result["pass_body_bear"][i] = int(pass_body)
            result["pass_source_strength_bear"][i] = int(pass_strength)
            result["pass_trend_bear"][i] = int(pass_trend)

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

                result["bear"][i] = 1
                result["direction"][i] = -1
                result["event_id"][i] = event_counter
                result["source_side"][i] = -1
                result["source_idx"][i] = float(src_idx)
                result["source_price"][i] = level
                result["level"][i] = level
                result["break_distance"][i] = break_dist
                result["break_distance_atr"][i] = break_dist_atr
                result["candle_body_atr"][i] = body_atr
                result["candle_range_atr"][i] = range_atr
                result["upper_wick_atr"][i] = upper_wick_atr
                result["lower_wick_atr"][i] = lower_wick_atr
                result["body_to_range"][i] = body_to_range
                result["close_location"][i] = close_location
                result["gap_from_level_atr"][i] = break_dist_atr
                result["displacement_score"][i] = displacement_score
                result["source_age"][i] = age
                result["source_strength"][i] = strength
                result["source_prominence_atr"][i] = prominence
                result["source_fresh"][i] = int(np.isfinite(age) and age <= 10.0)
                result["source_stale"][i] = int(np.isfinite(age) and age > 50.0)
                result["source_rank"][i] = source_rank
                result["source_in_trend_direction"][i] = source_in_trend_direction

    return result
