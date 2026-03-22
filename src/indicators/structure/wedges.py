# src/indicators/structure/wedges.py

"""
Causal wedge detector built on confirmed swings.

This version treats a wedge as an explicit geometric object:
- confirmed swing highs / lows only
- wedge edges defined by straight lines through recent swing anchors
- active wedge state persists until breakout or invalidation
- breakout target = breakout price +/- wedge base height

Outputs include:
- wedge_rising / wedge_falling / wedge_active / wedge_kind
- wedge_upper_bound / wedge_lower_bound
- wedge_upper_slope / wedge_lower_slope
- wedge_width / wedge_width_atr / wedge_compression_ratio
- wedge_quality / wedge_confirm_count / wedge_age
- wedge_apex_idx / wedge_bars_to_apex
- wedge_start_idx / wedge_end_idx
- wedge_upper_x1 / wedge_upper_y1 / wedge_upper_x2 / wedge_upper_y2
- wedge_lower_x1 / wedge_lower_y1 / wedge_lower_x2 / wedge_lower_y2
- wedge_breakout_up / wedge_breakout_down / wedge_breakout_dir
- wedge_breakout_price / wedge_breakout_idx
- wedge_target_price / wedge_target_distance / wedge_target_distance_atr
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns


@dataclass
class _SwingPoint:
    side: str
    origin_idx: int
    confirm_idx: int
    price: float


@dataclass
class _Line:
    slope: float
    intercept: float
    r2: float
    x1: int
    y1: float
    x2: int
    y2: float


def _fit_line(points: list[_SwingPoint]) -> _Line:
    x = np.array([p.origin_idx for p in points], dtype=float)
    y = np.array([p.price for p in points], dtype=float)

    if len(x) < 2:
        return _Line(np.nan, np.nan, np.nan, -1, np.nan, -1, np.nan)

    x_mean = x.mean()
    y_mean = y.mean()
    ssx = float(((x - x_mean) ** 2).sum())
    if ssx <= 0:
        slope = 0.0
        intercept = float(y_mean)
    else:
        slope = float(((x - x_mean) * (y - y_mean)).sum() / ssx)
        intercept = float(y_mean - slope * x_mean)

    y_hat = slope * x + intercept
    sst = float(((y - y_mean) ** 2).sum())
    sse = float(((y - y_hat) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    r2 = float(max(min(r2, 1.0), -1.0))

    x1 = int(x.min())
    x2 = int(x.max())
    y1 = float(slope * x1 + intercept)
    y2 = float(slope * x2 + intercept)

    return _Line(slope, intercept, r2, x1, y1, x2, y2)


def _line_y(line: _Line, x: int) -> float:
    if not np.isfinite(line.slope) or not np.isfinite(line.intercept):
        return np.nan
    return float(line.slope * x + line.intercept)


def _alternating_enough(events: list[_SwingPoint], min_total_events: int) -> bool:
    if len(events) < min_total_events:
        return False
    changes = 0
    for i in range(1, len(events)):
        if events[i].side != events[i - 1].side:
            changes += 1
    return changes >= min_total_events - 2


def add_wedges(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_high_points: int = 3,
    min_low_points: int = 3,
    max_lookback_bars: int = 140,
    max_pattern_span_bars: int = 140,
    min_line_r2: float = 0.20,
    min_width_atr: float = 0.30,
    max_width_atr: float = 10.0,
    min_compression_ratio: float = 0.18,
    breakout_atr_buffer: float = 0.08,
    apex_max_bars_forward: int = 100,
    min_total_events: int = 6,
) -> pd.DataFrame:
    out = df.copy()

    require_columns(
        out,
        {
            "open",
            "high",
            "low",
            "close",
            "swing_high_confirm_flag",
            "swing_low_confirm_flag",
            "swing_high_confirm_origin_idx",
            "swing_low_confirm_origin_idx",
            "swing_high_confirm_price",
            "swing_low_confirm_price",
        },
    )

    n = len(out)
    atr = get_atr_array(out, atr_length)

    wedge_rising = np.zeros(n, dtype=np.int8)
    wedge_falling = np.zeros(n, dtype=np.int8)
    wedge_active = np.zeros(n, dtype=np.int8)
    wedge_kind = np.zeros(n, dtype=np.int8)

    wedge_upper_bound = np.full(n, np.nan)
    wedge_lower_bound = np.full(n, np.nan)
    wedge_apex_idx = np.full(n, np.nan)
    wedge_bars_to_apex = np.full(n, np.nan)

    wedge_width = np.full(n, np.nan)
    wedge_width_atr = np.full(n, np.nan)
    wedge_compression_ratio = np.full(n, np.nan)

    wedge_upper_slope = np.full(n, np.nan)
    wedge_lower_slope = np.full(n, np.nan)
    wedge_upper_r2 = np.full(n, np.nan)
    wedge_lower_r2 = np.full(n, np.nan)
    wedge_quality = np.full(n, np.nan)

    wedge_confirm_count = np.zeros(n, dtype=np.int16)
    wedge_age = np.zeros(n, dtype=np.int32)

    wedge_start_idx = np.full(n, np.nan)
    wedge_end_idx = np.full(n, np.nan)

    wedge_upper_x1 = np.full(n, np.nan)
    wedge_upper_y1 = np.full(n, np.nan)
    wedge_upper_x2 = np.full(n, np.nan)
    wedge_upper_y2 = np.full(n, np.nan)

    wedge_lower_x1 = np.full(n, np.nan)
    wedge_lower_y1 = np.full(n, np.nan)
    wedge_lower_x2 = np.full(n, np.nan)
    wedge_lower_y2 = np.full(n, np.nan)

    wedge_breakout_up = np.zeros(n, dtype=np.int8)
    wedge_breakout_down = np.zeros(n, dtype=np.int8)
    wedge_breakout_dir = np.zeros(n, dtype=np.int8)
    wedge_breakout_price = np.full(n, np.nan)
    wedge_breakout_idx = np.full(n, np.nan)
    wedge_breakout_distance_atr = np.full(n, np.nan)

    wedge_target_price = np.full(n, np.nan)
    wedge_target_distance = np.full(n, np.nan)
    wedge_target_distance_atr = np.full(n, np.nan)

    wedge_target_50_price = np.full(n, np.nan)
    wedge_target_75_price = np.full(n, np.nan)
    wedge_target_100_price = np.full(n, np.nan)

    wedge_target_50_distance = np.full(n, np.nan)
    wedge_target_75_distance = np.full(n, np.nan)
    wedge_target_100_distance = np.full(n, np.nan)

    wedge_target_50_distance_atr = np.full(n, np.nan)
    wedge_target_75_distance_atr = np.full(n, np.nan)
    wedge_target_100_distance_atr = np.full(n, np.nan)

    events: list[_SwingPoint] = []

    active_kind = 0
    active_since = -1
    active_upper: _Line | None = None
    active_lower: _Line | None = None
    active_apex = np.nan
    active_start_idx = np.nan
    active_end_idx = np.nan
    active_width0 = np.nan
    active_quality = np.nan
    active_confirm_count = 0

    for i in range(n):
        if int(out["swing_high_confirm_flag"].iloc[i]) == 1:
            oi = out["swing_high_confirm_origin_idx"].iloc[i]
            px = out["swing_high_confirm_price"].iloc[i]
            if np.isfinite(oi) and np.isfinite(px):
                events.append(_SwingPoint("high", int(oi), i, float(px)))

        if int(out["swing_low_confirm_flag"].iloc[i]) == 1:
            oi = out["swing_low_confirm_origin_idx"].iloc[i]
            px = out["swing_low_confirm_price"].iloc[i]
            if np.isfinite(oi) and np.isfinite(px):
                events.append(_SwingPoint("low", int(oi), i, float(px)))

        recent = [
            e
            for e in events
            if e.confirm_idx <= i and i - e.confirm_idx <= max_lookback_bars
        ]
        highs = [e for e in recent if e.side == "high"][-min_high_points:]
        lows = [e for e in recent if e.side == "low"][-min_low_points:]
        merged = sorted(highs + lows, key=lambda e: (e.confirm_idx, e.origin_idx))

        found_kind = 0
        found_upper = None
        found_lower = None
        found_apex = np.nan
        found_width0 = np.nan
        found_quality = np.nan
        found_confirm_count = 0
        found_start_idx = np.nan
        found_end_idx = np.nan

        if (
            len(highs) >= min_high_points
            and len(lows) >= min_low_points
            and _alternating_enough(merged, min_total_events)
        ):
            upper = _fit_line(highs)
            lower = _fit_line(lows)

            start_idx = min(upper.x1, lower.x1)
            end_idx = max(upper.x2, lower.x2)

            if end_idx - start_idx <= max_pattern_span_bars:
                width_start = _line_y(upper, start_idx) - _line_y(lower, start_idx)
                width_end = _line_y(upper, end_idx) - _line_y(lower, end_idx)
                width_now = _line_y(upper, i) - _line_y(lower, i)

                width_now_atr = (
                    width_now / atr[i]
                    if np.isfinite(width_now) and np.isfinite(atr[i]) and atr[i] > 0
                    else np.nan
                )

                compression_ratio = (
                    (width_start - width_end) / width_start
                    if np.isfinite(width_start)
                    and width_start > 0
                    and np.isfinite(width_end)
                    else np.nan
                )

                denom = upper.slope - lower.slope
                apex_idx = np.nan
                if np.isfinite(denom) and abs(denom) > 1e-12:
                    apex_idx = (lower.intercept - upper.intercept) / denom

                common_ok = (
                    np.isfinite(width_start)
                    and np.isfinite(width_end)
                    and np.isfinite(width_now)
                    and width_start > 0
                    and width_end > 0
                    and width_now > 0
                    and np.isfinite(width_now_atr)
                    and min_width_atr <= width_now_atr <= max_width_atr
                    and np.isfinite(compression_ratio)
                    and compression_ratio >= min_compression_ratio
                    and upper.r2 >= min_line_r2
                    and lower.r2 >= min_line_r2
                    and np.isfinite(apex_idx)
                    and apex_idx >= end_idx
                    and apex_idx <= i + apex_max_bars_forward
                )

                rising_ok = (
                    common_ok
                    and upper.slope > 0
                    and lower.slope > 0
                    and lower.slope > upper.slope
                )
                falling_ok = (
                    common_ok
                    and upper.slope < 0
                    and lower.slope < 0
                    and upper.slope < lower.slope
                )

                if rising_ok or falling_ok:
                    found_kind = 1 if rising_ok else -1
                    found_upper = upper
                    found_lower = lower
                    found_apex = float(apex_idx)
                    found_width0 = float(width_start)
                    found_confirm_count = len(merged)
                    found_start_idx = float(start_idx)
                    found_end_idx = float(end_idx)
                    found_quality = float(
                        np.nanmean(
                            [
                                upper.r2,
                                lower.r2,
                                max(0.0, min(1.0, compression_ratio)),
                                1.0 - min(width_now_atr / max_width_atr, 1.0),
                            ]
                        )
                    )

        if found_kind != 0:
            active_kind = found_kind
            active_since = (
                i if (i == 0 or active_kind != wedge_kind[i - 1]) else active_since
            )
            active_upper = found_upper
            active_lower = found_lower
            active_apex = found_apex
            active_start_idx = found_start_idx
            active_end_idx = found_end_idx
            active_width0 = found_width0
            active_quality = found_quality
            active_confirm_count = found_confirm_count
        elif active_kind != 0 and active_upper is not None and active_lower is not None:
            upper_now = _line_y(active_upper, i)
            lower_now = _line_y(active_lower, i)
            buf = (
                breakout_atr_buffer * atr[i]
                if np.isfinite(atr[i]) and atr[i] > 0
                else 0.0
            )

            broke_up = (
                np.isfinite(upper_now) and float(out["close"].iloc[i]) > upper_now + buf
            )
            broke_down = (
                np.isfinite(lower_now) and float(out["close"].iloc[i]) < lower_now - buf
            )

            if broke_up or broke_down:
                direction = 1 if broke_up else -1
                base_width = (
                    active_width0
                    if np.isfinite(active_width0) and active_width0 > 0
                    else np.nan
                )
                breakout_price = float(out["close"].iloc[i])

                target_50 = np.nan
                target_75 = np.nan
                target_100 = np.nan

                if np.isfinite(base_width):
                    if broke_up:
                        target_50 = breakout_price + 0.50 * base_width
                        target_75 = breakout_price + 0.75 * base_width
                        target_100 = breakout_price + 1.00 * base_width
                    else:
                        target_50 = breakout_price - 0.50 * base_width
                        target_75 = breakout_price - 0.75 * base_width
                        target_100 = breakout_price - 1.00 * base_width

                if broke_up:
                    wedge_breakout_up[i] = 1
                if broke_down:
                    wedge_breakout_down[i] = 1

                wedge_breakout_dir[i] = direction
                wedge_breakout_price[i] = breakout_price
                wedge_breakout_idx[i] = float(i)

                ref_line = upper_now if broke_up else lower_now
                wedge_breakout_distance_atr[i] = (
                    abs(breakout_price - ref_line) / atr[i]
                    if np.isfinite(ref_line) and np.isfinite(atr[i]) and atr[i] > 0
                    else np.nan
                )

                # backward-compatible default target = full classical target
                wedge_target_price[i] = target_100
                wedge_target_distance[i] = (
                    abs(target_100 - breakout_price)
                    if np.isfinite(target_100)
                    else np.nan
                )
                wedge_target_distance_atr[i] = (
                    wedge_target_distance[i] / atr[i]
                    if np.isfinite(wedge_target_distance[i])
                    and np.isfinite(atr[i])
                    and atr[i] > 0
                    else np.nan
                )

                wedge_target_50_price[i] = target_50
                wedge_target_75_price[i] = target_75
                wedge_target_100_price[i] = target_100

                wedge_target_50_distance[i] = (
                    abs(target_50 - breakout_price)
                    if np.isfinite(target_50)
                    else np.nan
                )
                wedge_target_75_distance[i] = (
                    abs(target_75 - breakout_price)
                    if np.isfinite(target_75)
                    else np.nan
                )
                wedge_target_100_distance[i] = (
                    abs(target_100 - breakout_price)
                    if np.isfinite(target_100)
                    else np.nan
                )

                wedge_target_50_distance_atr[i] = (
                    wedge_target_50_distance[i] / atr[i]
                    if np.isfinite(wedge_target_50_distance[i])
                    and np.isfinite(atr[i])
                    and atr[i] > 0
                    else np.nan
                )
                wedge_target_75_distance_atr[i] = (
                    wedge_target_75_distance[i] / atr[i]
                    if np.isfinite(wedge_target_75_distance[i])
                    and np.isfinite(atr[i])
                    and atr[i] > 0
                    else np.nan
                )
                wedge_target_100_distance_atr[i] = (
                    wedge_target_100_distance[i] / atr[i]
                    if np.isfinite(wedge_target_100_distance[i])
                    and np.isfinite(atr[i])
                    and atr[i] > 0
                    else np.nan
                )

                active_kind = 0
                active_since = -1
                active_upper = None
                active_lower = None
                active_apex = np.nan
                active_start_idx = np.nan
                active_end_idx = np.nan
                active_width0 = np.nan
                active_quality = np.nan
                active_confirm_count = 0

        if active_kind != 0 and active_upper is not None and active_lower is not None:
            upper_now = _line_y(active_upper, i)
            lower_now = _line_y(active_lower, i)
            width_now = (
                upper_now - lower_now
                if np.isfinite(upper_now) and np.isfinite(lower_now)
                else np.nan
            )

            wedge_active[i] = 1
            wedge_kind[i] = active_kind
            wedge_rising[i] = 1 if active_kind == 1 else 0
            wedge_falling[i] = 1 if active_kind == -1 else 0

            wedge_upper_bound[i] = upper_now
            wedge_lower_bound[i] = lower_now
            wedge_apex_idx[i] = active_apex
            wedge_bars_to_apex[i] = (
                active_apex - i if np.isfinite(active_apex) else np.nan
            )

            wedge_width[i] = width_now
            wedge_width_atr[i] = (
                width_now / atr[i]
                if np.isfinite(width_now) and np.isfinite(atr[i]) and atr[i] > 0
                else np.nan
            )
            wedge_compression_ratio[i] = (
                (active_width0 - width_now) / active_width0
                if np.isfinite(active_width0)
                and active_width0 > 0
                and np.isfinite(width_now)
                else np.nan
            )

            wedge_upper_slope[i] = active_upper.slope
            wedge_lower_slope[i] = active_lower.slope
            wedge_upper_r2[i] = active_upper.r2
            wedge_lower_r2[i] = active_lower.r2
            wedge_quality[i] = active_quality
            wedge_confirm_count[i] = active_confirm_count
            wedge_age[i] = i - active_since + 1 if active_since >= 0 else 0

            wedge_start_idx[i] = active_start_idx
            wedge_end_idx[i] = active_end_idx

            wedge_upper_x1[i] = active_upper.x1
            wedge_upper_y1[i] = active_upper.y1
            wedge_upper_x2[i] = active_upper.x2
            wedge_upper_y2[i] = active_upper.y2

            wedge_lower_x1[i] = active_lower.x1
            wedge_lower_y1[i] = active_lower.y1
            wedge_lower_x2[i] = active_lower.x2
            wedge_lower_y2[i] = active_lower.y2

    out["wedge_rising"] = wedge_rising
    out["wedge_falling"] = wedge_falling
    out["wedge_active"] = wedge_active
    out["wedge_kind"] = wedge_kind

    out["wedge_upper_bound"] = wedge_upper_bound
    out["wedge_lower_bound"] = wedge_lower_bound
    out["wedge_apex_idx"] = wedge_apex_idx
    out["wedge_bars_to_apex"] = wedge_bars_to_apex

    out["wedge_width"] = wedge_width
    out["wedge_width_atr"] = wedge_width_atr
    out["wedge_compression_ratio"] = wedge_compression_ratio

    out["wedge_upper_slope"] = wedge_upper_slope
    out["wedge_lower_slope"] = wedge_lower_slope
    out["wedge_upper_r2"] = wedge_upper_r2
    out["wedge_lower_r2"] = wedge_lower_r2
    out["wedge_quality"] = wedge_quality
    out["wedge_confirm_count"] = wedge_confirm_count
    out["wedge_age"] = wedge_age

    out["wedge_start_idx"] = wedge_start_idx
    out["wedge_end_idx"] = wedge_end_idx

    out["wedge_upper_x1"] = wedge_upper_x1
    out["wedge_upper_y1"] = wedge_upper_y1
    out["wedge_upper_x2"] = wedge_upper_x2
    out["wedge_upper_y2"] = wedge_upper_y2

    out["wedge_lower_x1"] = wedge_lower_x1
    out["wedge_lower_y1"] = wedge_lower_y1
    out["wedge_lower_x2"] = wedge_lower_x2
    out["wedge_lower_y2"] = wedge_lower_y2

    out["wedge_breakout_up"] = wedge_breakout_up
    out["wedge_breakout_down"] = wedge_breakout_down
    out["wedge_breakout_dir"] = wedge_breakout_dir
    out["wedge_breakout_price"] = wedge_breakout_price
    out["wedge_breakout_idx"] = wedge_breakout_idx
    out["wedge_breakout_distance_atr"] = wedge_breakout_distance_atr

    out["wedge_target_price"] = wedge_target_price
    out["wedge_target_distance"] = wedge_target_distance
    out["wedge_target_distance_atr"] = wedge_target_distance_atr

    out["wedge_target_50_price"] = wedge_target_50_price
    out["wedge_target_75_price"] = wedge_target_75_price
    out["wedge_target_100_price"] = wedge_target_100_price

    out["wedge_target_50_distance"] = wedge_target_50_distance
    out["wedge_target_75_distance"] = wedge_target_75_distance
    out["wedge_target_100_distance"] = wedge_target_100_distance

    out["wedge_target_50_distance_atr"] = wedge_target_50_distance_atr
    out["wedge_target_75_distance_atr"] = wedge_target_75_distance_atr
    out["wedge_target_100_distance_atr"] = wedge_target_100_distance_atr

    return out
