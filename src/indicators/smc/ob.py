"""
Canonical causal Order Block detector.

Doctrine
--------
- Canonical production family is BOS-derived.
- Source = last opposing candle before the displacement leg that directly leads
  into the confirmed parent BOS.
- Activation = parent BOS confirmation bar.
- Geometry = full source candle wick-to-wick range.
- Displacement is persisted as metadata; it is not a hard existence gate.
- Downstream ranking/research may become sophisticated, but canonical source
  selection remains frozen and simple.

Strategic stance
----------------
- BOS and CHoCH are the superior structural signals in this codebase.
- OB is retained only as a traced execution/research layer on top of structure.
- Production strategies should not require OB for signal generation when BOS or
  CHoCH already expresses the structural edge more directly and more robustly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc

OB_STATE_INACTIVE_PRE_ACTIVATION = 0
OB_STATE_ACTIVE_FRESH = 1
OB_STATE_ACTIVE_TOUCHED = 2
OB_STATE_MITIGATED_PARTIAL = 3
OB_STATE_MITIGATED_FULL = 4
OB_STATE_INVALIDATED = 5
OB_STATE_RETIRED = 6

OB_IDENTITY_COLUMNS = [
    "ob_id",
    "ob_family",
    "ob_side",
    "ob_parent_event_type",
    "ob_parent_bos_idx",
    "ob_parent_bos_ts",
    "ob_parent_bos_timestamp",
    "ob_parent_displacement_idx",
    "ob_source_idx",
    "ob_source_ts",
    "ob_source_timestamp",
    "ob_source_is_opposing_candle_bool",
    "ob_traceback_start_idx",
    "ob_traceback_end_idx",
    "ob_source_selection_reason",
    "ob_activate_idx",
    "ob_activate_ts",
    "ob_activate_timestamp",
]

OB_GEOMETRY_COLUMNS = [
    "ob_zone_high",
    "ob_zone_low",
    "ob_zone_mid",
    "ob_zone_height_abs",
    "ob_zone_height_atr",
    "ob_body_low",
    "ob_body_high",
    "ob_body_mid",
    "ob_mid",
    "ob_height",
    "ob_height_atr",
    "ob_body_fraction_of_full_range",
    "ob_upper_wick_fraction",
    "ob_lower_wick_fraction",
    "ob_giant_candle_flag",
]

OB_SOURCE_COLUMNS = [
    "ob_source_open",
    "ob_source_high",
    "ob_source_low",
    "ob_source_close",
    "ob_source_body_abs",
    "ob_source_body_frac",
    "ob_source_wick_upper_abs",
    "ob_source_wick_lower_abs",
]

OB_PARENT_COLUMNS = [
    "ob_parent_bos_side",
    "ob_parent_displacement_score",
    "ob_parent_move_away_atr",
    "ob_parent_bos_quality",
    "ob_impulse_range_atr",
    "ob_impulse_close_efficiency",
    "ob_number_of_bars_in_impulse_leg",
    "ob_impulse_fvg_overlap_flag",
]

OB_QUALITY_COLUMNS = [
    "ob_strength_raw",
    "ob_strength",
    "ob_quality_tier",
]

OB_LIFECYCLE_COLUMNS = [
    "ob_state",
    "ob_is_active",
    "ob_is_fresh",
    "ob_is_invalidated",
    "ob_is_retired",
    "ob_age_bars",
    "ob_age_since_activation_bars",
]

OB_COMPAT_COLUMNS = [
    "ob_bull",
    "ob_bear",
    "ob_bull_low",
    "ob_bull_high",
    "ob_bear_low",
    "ob_bear_high",
    "ob_width_atr",
]

OB_CORE_COLUMNS = (
    OB_IDENTITY_COLUMNS
    + OB_GEOMETRY_COLUMNS
    + OB_SOURCE_COLUMNS
    + OB_PARENT_COLUMNS
    + OB_QUALITY_COLUMNS
    + OB_LIFECYCLE_COLUMNS
)

OB_LIVE_SAFE_COLUMNS = OB_CORE_COLUMNS + OB_COMPAT_COLUMNS

_REQUIRED_BOS_COLUMNS = {
    "bos_bull",
    "bos_bear",
    "bos_source_idx",
}

_SOURCE_SELECTION_REASON = "last_opposing_before_displacement_leg"


def _empty_timestamp_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")


def _empty_string_series(index: pd.Index) -> pd.Series:
    return pd.Series("", index=index, dtype="object")


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _clip_unit(value: float | None, *, default: float = np.nan) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(np.clip(value, 0.0, 1.0))


def _weighted_unit_score(components: list[tuple[float, float]]) -> float:
    total = 0.0
    used_weight = 0.0
    for value, weight in components:
        if np.isfinite(value) and weight > 0:
            total += value * weight
            used_weight += weight
    if used_weight <= 0:
        return np.nan
    return float(np.clip(total / used_weight, 0.0, 1.0))


def _series_to_numpy(
    df: pd.DataFrame,
    column: str,
    *,
    default: float | int = np.nan,
    dtype: type[np.floating[Any]] | type[np.integer[Any]] | type[float] = float,
) -> np.ndarray:
    series = df[column] if column in df.columns else pd.Series(default, index=df.index)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=dtype)


def _candle_direction(
    *,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    idx: int,
) -> int:
    if close_arr[idx] > open_arr[idx]:
        return 1
    if close_arr[idx] < open_arr[idx]:
        return -1
    return 0


def _find_traceback_source(
    *,
    side: int,
    event_idx: int,
    search_floor: int,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
) -> tuple[int | None, int | None, int]:
    """Return source idx and traceback start for the canonical leg.

    The displacement leg is defined as the terminal suffix of candles leading
    into the break that are not opposing the break direction. The source is the
    last opposing candle immediately before that suffix.
    """

    if event_idx <= max(search_floor, 0):
        return None, None, event_idx

    source_idx: int | None = None
    for idx in range(event_idx - 1, max(search_floor, 0) - 1, -1):
        direction = _candle_direction(open_arr=open_arr, close_arr=close_arr, idx=idx)
        if direction == -side:
            source_idx = idx
            break
        if direction == side or direction == 0:
            continue

    if source_idx is None:
        return None, None, event_idx
    return source_idx, source_idx + 1, event_idx


def _source_efficiency(
    *,
    source_open: float,
    source_high: float,
    source_low: float,
    source_close: float,
) -> tuple[float, float, float, float]:
    body_abs = float(abs(source_close - source_open))
    range_abs = float(source_high - source_low)
    body_frac = _safe_fraction(body_abs, range_abs)
    wick_upper_abs = float(source_high - max(source_open, source_close))
    wick_lower_abs = float(min(source_open, source_close) - source_low)
    return body_abs, body_frac, wick_upper_abs, wick_lower_abs


def _parent_bos_quality(
    *,
    side: int,
    bos_displacement_score: float,
    bos_break_distance_atr: float,
    bos_candle_body_atr: float,
    bos_source_strength: float,
    bos_close_location: float,
) -> float:
    close_quality = np.nan
    if np.isfinite(bos_close_location):
        close_quality = (
            _clip_unit(bos_close_location, default=np.nan)
            if side == 1
            else _clip_unit(1.0 - bos_close_location, default=np.nan)
        )
    return _weighted_unit_score(
        [
            (_clip_unit(bos_displacement_score, default=np.nan), 0.35),
            (
                _clip_unit(_safe_fraction(bos_break_distance_atr, 1.0), default=np.nan),
                0.25,
            ),
            (
                _clip_unit(_safe_fraction(bos_candle_body_atr, 1.5), default=np.nan),
                0.20,
            ),
            (_clip_unit(bos_source_strength, default=np.nan), 0.10),
            (close_quality, 0.10),
        ]
    )


def _quality_tier(strength: float) -> int:
    if not np.isfinite(strength):
        return 0
    if strength >= 0.75:
        return 3
    if strength >= 0.50:
        return 2
    return 1


def _initialize_empty_contract(out: pd.DataFrame) -> pd.DataFrame:
    n = len(out)
    out["ob_id"] = np.zeros(n, dtype=np.int64)
    out["ob_family"] = _empty_string_series(out.index)
    out["ob_side"] = np.zeros(n, dtype=np.int8)
    out["ob_parent_event_type"] = _empty_string_series(out.index)
    out["ob_parent_bos_idx"] = np.full(n, np.nan)
    out["ob_parent_bos_ts"] = _empty_timestamp_series(out.index)
    out["ob_parent_bos_timestamp"] = _empty_timestamp_series(out.index)
    out["ob_parent_displacement_idx"] = np.full(n, np.nan)
    out["ob_source_idx"] = np.full(n, np.nan)
    out["ob_source_ts"] = _empty_timestamp_series(out.index)
    out["ob_source_timestamp"] = _empty_timestamp_series(out.index)
    out["ob_source_is_opposing_candle_bool"] = np.zeros(n, dtype=np.int8)
    out["ob_traceback_start_idx"] = np.full(n, np.nan)
    out["ob_traceback_end_idx"] = np.full(n, np.nan)
    out["ob_source_selection_reason"] = _empty_string_series(out.index)
    out["ob_activate_idx"] = np.full(n, np.nan)
    out["ob_activate_ts"] = _empty_timestamp_series(out.index)
    out["ob_activate_timestamp"] = _empty_timestamp_series(out.index)

    for col in (
        OB_GEOMETRY_COLUMNS + OB_SOURCE_COLUMNS + OB_PARENT_COLUMNS + OB_QUALITY_COLUMNS
    ):
        out[col] = np.full(n, np.nan)

    out["ob_state"] = np.zeros(n, dtype=np.int8)
    out["ob_is_active"] = np.zeros(n, dtype=np.int8)
    out["ob_is_fresh"] = np.zeros(n, dtype=np.int8)
    out["ob_is_invalidated"] = np.zeros(n, dtype=np.int8)
    out["ob_is_retired"] = np.zeros(n, dtype=np.int8)
    out["ob_age_bars"] = np.full(n, np.nan)
    out["ob_age_since_activation_bars"] = np.full(n, np.nan)

    out["ob_bull"] = np.zeros(n, dtype=np.int8)
    out["ob_bear"] = np.zeros(n, dtype=np.int8)
    out["ob_bull_low"] = np.full(n, np.nan)
    out["ob_bull_high"] = np.full(n, np.nan)
    out["ob_bear_low"] = np.full(n, np.nan)
    out["ob_bear_high"] = np.full(n, np.nan)
    out["ob_width_atr"] = np.full(n, np.nan)
    return out


def add_ob(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_parent_move_away_atr: float = 1e-9,
    bos_source_floor: bool = True,
) -> pd.DataFrame:
    """Append canonical BOS-derived OB core columns."""
    require_ohlc(df, caller="add_ob")
    out = _initialize_empty_contract(df.copy())

    missing_bos = _REQUIRED_BOS_COLUMNS - set(out.columns)
    if missing_bos:
        return out

    n = len(out)
    ts = (
        pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if "timestamp" in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )
    open_arr = out["open"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    atr_arr = get_atr_array(out, length=atr_length).astype(float, copy=False)

    bos_bull = (
        pd.to_numeric(out["bos_bull"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
    )
    bos_bear = (
        pd.to_numeric(out["bos_bear"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
    )
    bos_source_idx = pd.to_numeric(out["bos_source_idx"], errors="coerce").to_numpy(
        dtype=float
    )
    bos_displacement_score = _series_to_numpy(
        out, "bos_displacement_score", dtype=float
    )
    bos_break_distance_atr = _series_to_numpy(
        out, "bos_break_distance_atr", dtype=float
    )
    bos_candle_body_atr = _series_to_numpy(out, "bos_candle_body_atr", dtype=float)
    bos_source_strength = _series_to_numpy(out, "bos_source_strength", dtype=float)
    bos_close_location = _series_to_numpy(out, "bos_close_location", dtype=float)

    fvg_bull = (
        pd.to_numeric(out["fvg_bull"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
        if "fvg_bull" in out.columns
        else np.zeros(n, dtype=np.int8)
    )
    fvg_bear = (
        pd.to_numeric(out["fvg_bear"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
        if "fvg_bear" in out.columns
        else np.zeros(n, dtype=np.int8)
    )

    next_ob_id = 1
    for bos_idx in range(n):
        side = 1 if bos_bull[bos_idx] == 1 else -1 if bos_bear[bos_idx] == 1 else 0
        if side == 0:
            continue

        activate_idx = bos_idx
        parent_source_idx = bos_source_idx[bos_idx]
        search_floor = (
            int(parent_source_idx)
            if bos_source_floor and np.isfinite(parent_source_idx)
            else 0
        )
        source_idx, traceback_start_idx, traceback_end_idx = _find_traceback_source(
            side=side,
            event_idx=bos_idx,
            search_floor=search_floor,
            open_arr=open_arr,
            close_arr=close_arr,
        )
        if source_idx is None or traceback_start_idx is None:
            continue

        source_open = float(open_arr[source_idx])
        source_high = float(high_arr[source_idx])
        source_low = float(low_arr[source_idx])
        source_close = float(close_arr[source_idx])
        source_direction = _candle_direction(
            open_arr=open_arr, close_arr=close_arr, idx=source_idx
        )
        source_is_opposing = int(source_direction == -side)
        if source_is_opposing != 1:
            continue

        zone_low = float(source_low)
        zone_high = float(source_high)
        zone_height_abs = float(zone_high - zone_low)
        atr_value = atr_arr[activate_idx]
        zone_height_atr = _safe_fraction(zone_height_abs, atr_value)
        if not np.isfinite(zone_height_abs) or zone_height_abs <= 0:
            continue
        if not np.isfinite(zone_height_atr) or zone_height_atr <= 0:
            continue

        impulse_high = float(
            np.nanmax(high_arr[traceback_start_idx : traceback_end_idx + 1])
        )
        impulse_low = float(
            np.nanmin(low_arr[traceback_start_idx : traceback_end_idx + 1])
        )
        if side == 1:
            parent_move_away_abs = float(max(impulse_high - zone_high, 0.0))
            impulse_close_move_abs = float(max(close_arr[bos_idx] - zone_high, 0.0))
        else:
            parent_move_away_abs = float(max(zone_low - impulse_low, 0.0))
            impulse_close_move_abs = float(max(zone_low - close_arr[bos_idx], 0.0))
        parent_move_away_atr = _safe_fraction(parent_move_away_abs, atr_value)
        if np.isfinite(min_parent_move_away_atr) and (
            not np.isfinite(parent_move_away_atr)
            or parent_move_away_atr <= min_parent_move_away_atr
        ):
            continue

        impulse_range_atr = _safe_fraction(impulse_high - impulse_low, atr_value)
        impulse_close_efficiency = _safe_fraction(
            impulse_close_move_abs, max(impulse_high - impulse_low, 1e-12)
        )
        impulse_bars = int(traceback_end_idx - traceback_start_idx + 1)
        impulse_fvg_overlap_flag = int(
            np.any(fvg_bull[traceback_start_idx : traceback_end_idx + 1] == 1)
            if side == 1
            else np.any(fvg_bear[traceback_start_idx : traceback_end_idx + 1] == 1)
        )

        source_body_abs, source_body_frac, wick_upper_abs, wick_lower_abs = (
            _source_efficiency(
                source_open=source_open,
                source_high=source_high,
                source_low=source_low,
                source_close=source_close,
            )
        )
        body_low = float(min(source_open, source_close))
        body_high = float(max(source_open, source_close))
        body_mid = float((body_low + body_high) / 2.0)
        ob_mid = float((zone_low + zone_high) / 2.0)
        upper_wick_frac = _safe_fraction(wick_upper_abs, zone_height_abs)
        lower_wick_frac = _safe_fraction(wick_lower_abs, zone_height_abs)
        giant_candle_flag = int(np.isfinite(zone_height_atr) and zone_height_atr >= 1.5)

        parent_disp_score = _clip_unit(bos_displacement_score[bos_idx], default=np.nan)
        parent_quality = _parent_bos_quality(
            side=side,
            bos_displacement_score=bos_displacement_score[bos_idx],
            bos_break_distance_atr=bos_break_distance_atr[bos_idx],
            bos_candle_body_atr=bos_candle_body_atr[bos_idx],
            bos_source_strength=bos_source_strength[bos_idx],
            bos_close_location=bos_close_location[bos_idx],
        )
        compactness_score = _clip_unit(
            1.0 - _safe_fraction(zone_height_atr, 4.0), default=np.nan
        )
        move_away_score = _clip_unit(
            _safe_fraction(parent_move_away_atr, 2.0), default=np.nan
        )
        strength_raw = _weighted_unit_score(
            [
                (parent_quality, 0.35),
                (parent_disp_score, 0.25),
                (_clip_unit(source_body_frac, default=np.nan), 0.15),
                (compactness_score, 0.10),
                (move_away_score, 0.15),
            ]
        )
        strength = _clip_unit(strength_raw, default=np.nan)

        out.at[activate_idx, "ob_id"] = next_ob_id
        out.at[activate_idx, "ob_family"] = "bos"
        out.at[activate_idx, "ob_side"] = side
        out.at[activate_idx, "ob_parent_event_type"] = "bos"
        out.at[activate_idx, "ob_parent_bos_idx"] = float(bos_idx)
        out.at[activate_idx, "ob_parent_bos_ts"] = ts.iloc[bos_idx]
        out.at[activate_idx, "ob_parent_bos_timestamp"] = ts.iloc[bos_idx]
        out.at[activate_idx, "ob_parent_displacement_idx"] = float(traceback_end_idx)
        out.at[activate_idx, "ob_source_idx"] = float(source_idx)
        out.at[activate_idx, "ob_source_ts"] = ts.iloc[source_idx]
        out.at[activate_idx, "ob_source_timestamp"] = ts.iloc[source_idx]
        out.at[activate_idx, "ob_source_is_opposing_candle_bool"] = source_is_opposing
        out.at[activate_idx, "ob_traceback_start_idx"] = float(traceback_start_idx)
        out.at[activate_idx, "ob_traceback_end_idx"] = float(traceback_end_idx)
        out.at[activate_idx, "ob_source_selection_reason"] = _SOURCE_SELECTION_REASON
        out.at[activate_idx, "ob_activate_idx"] = float(activate_idx)
        out.at[activate_idx, "ob_activate_ts"] = ts.iloc[activate_idx]
        out.at[activate_idx, "ob_activate_timestamp"] = ts.iloc[activate_idx]

        out.at[activate_idx, "ob_zone_high"] = zone_high
        out.at[activate_idx, "ob_zone_low"] = zone_low
        out.at[activate_idx, "ob_zone_mid"] = ob_mid
        out.at[activate_idx, "ob_zone_height_abs"] = zone_height_abs
        out.at[activate_idx, "ob_zone_height_atr"] = zone_height_atr
        out.at[activate_idx, "ob_body_low"] = body_low
        out.at[activate_idx, "ob_body_high"] = body_high
        out.at[activate_idx, "ob_body_mid"] = body_mid
        out.at[activate_idx, "ob_mid"] = ob_mid
        out.at[activate_idx, "ob_height"] = zone_height_abs
        out.at[activate_idx, "ob_height_atr"] = zone_height_atr
        out.at[activate_idx, "ob_body_fraction_of_full_range"] = source_body_frac
        out.at[activate_idx, "ob_upper_wick_fraction"] = upper_wick_frac
        out.at[activate_idx, "ob_lower_wick_fraction"] = lower_wick_frac
        out.at[activate_idx, "ob_giant_candle_flag"] = giant_candle_flag

        out.at[activate_idx, "ob_source_open"] = source_open
        out.at[activate_idx, "ob_source_high"] = source_high
        out.at[activate_idx, "ob_source_low"] = source_low
        out.at[activate_idx, "ob_source_close"] = source_close
        out.at[activate_idx, "ob_source_body_abs"] = source_body_abs
        out.at[activate_idx, "ob_source_body_frac"] = source_body_frac
        out.at[activate_idx, "ob_source_wick_upper_abs"] = wick_upper_abs
        out.at[activate_idx, "ob_source_wick_lower_abs"] = wick_lower_abs

        out.at[activate_idx, "ob_parent_bos_side"] = side
        out.at[activate_idx, "ob_parent_displacement_score"] = parent_disp_score
        out.at[activate_idx, "ob_parent_move_away_atr"] = parent_move_away_atr
        out.at[activate_idx, "ob_parent_bos_quality"] = parent_quality
        out.at[activate_idx, "ob_impulse_range_atr"] = impulse_range_atr
        out.at[activate_idx, "ob_impulse_close_efficiency"] = impulse_close_efficiency
        out.at[activate_idx, "ob_number_of_bars_in_impulse_leg"] = float(impulse_bars)
        out.at[activate_idx, "ob_impulse_fvg_overlap_flag"] = float(
            impulse_fvg_overlap_flag
        )

        out.at[activate_idx, "ob_strength_raw"] = strength_raw
        out.at[activate_idx, "ob_strength"] = strength
        out.at[activate_idx, "ob_quality_tier"] = float(_quality_tier(strength))

        out.at[activate_idx, "ob_state"] = OB_STATE_ACTIVE_FRESH
        out.at[activate_idx, "ob_is_active"] = 1
        out.at[activate_idx, "ob_is_fresh"] = 1
        out.at[activate_idx, "ob_is_invalidated"] = 0
        out.at[activate_idx, "ob_is_retired"] = 0
        out.at[activate_idx, "ob_age_bars"] = float(activate_idx - source_idx)
        out.at[activate_idx, "ob_age_since_activation_bars"] = 0.0

        if side == 1:
            out.at[activate_idx, "ob_bull"] = 1
            out.at[activate_idx, "ob_bull_low"] = zone_low
            out.at[activate_idx, "ob_bull_high"] = zone_high
        else:
            out.at[activate_idx, "ob_bear"] = 1
            out.at[activate_idx, "ob_bear_low"] = zone_low
            out.at[activate_idx, "ob_bear_high"] = zone_high
        out.at[activate_idx, "ob_width_atr"] = zone_height_atr
        next_ob_id += 1

    int8_cols = [
        "ob_side",
        "ob_source_is_opposing_candle_bool",
        "ob_state",
        "ob_is_active",
        "ob_is_fresh",
        "ob_is_invalidated",
        "ob_is_retired",
        "ob_bull",
        "ob_bear",
        "ob_giant_candle_flag",
    ]
    for col in int8_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int8)
    out["ob_id"] = (
        pd.to_numeric(out["ob_id"], errors="coerce").fillna(0).astype(np.int64)
    )
    return out


__all__ = [
    "OB_STATE_INACTIVE_PRE_ACTIVATION",
    "OB_STATE_ACTIVE_FRESH",
    "OB_STATE_ACTIVE_TOUCHED",
    "OB_STATE_MITIGATED_PARTIAL",
    "OB_STATE_MITIGATED_FULL",
    "OB_STATE_INVALIDATED",
    "OB_STATE_RETIRED",
    "OB_CORE_COLUMNS",
    "OB_COMPAT_COLUMNS",
    "OB_LIVE_SAFE_COLUMNS",
    "add_ob",
]
