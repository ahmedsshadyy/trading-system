"""
Canonical UTC-only session/time context features.

This module is a reference/context layer, not a pattern detector.
All live-safe outputs are causal by default and may depend only on:

* the current row timestamp,
* OHLC values up to and including the current row,
* previously completed sessions.

Research-only outputs are gated behind ``include_research_only=True`` and
are prefixed with ``r_`` so research/live parity stays explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array

__all__ = ["add_time_features", "add_session_features"]

SESSION_CODE_TO_NAME = {
    0: "Asia",
    1: "London",
    2: "Overlap",
    3: "NY",
    4: "Dead",
}
SESSION_START_MINUTES = np.array([0, 8 * 60, 13 * 60, 17 * 60, 22 * 60], dtype=int)
SESSION_END_MINUTES = np.array([8 * 60, 13 * 60, 17 * 60, 22 * 60, 24 * 60], dtype=int)
SESSION_TOTAL_MINUTES = SESSION_END_MINUTES - SESSION_START_MINUTES
SESSION_FAMILY_PREFIX = {
    0: "asia",
    1: "london",
    3: "ny",
}
SESSION_COMBO_LABELS = (
    ("r_asia_london_direction_combo_final", 0, 1),
    ("r_london_ny_direction_combo_final", 1, 3),
    ("r_asia_ny_direction_combo_final", 0, 3),
)
DAY_MINUTES = 24 * 60
DIRECTION_LABELS = {
    -1: "down",
    0: "flat",
    1: "up",
}


def _require_columns(
    df: pd.DataFrame, required: tuple[str, ...], func_name: str
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{func_name}: missing required columns {missing}")


def _normalize_utc_timestamps(df: pd.DataFrame, func_name: str) -> pd.Series:
    _require_columns(df, ("timestamp",), func_name)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(np.flatnonzero(ts.isna().to_numpy())[0])
        raise ValueError(f"{func_name}: invalid timestamp at row {bad}")

    diff_ns = ts.astype("int64").to_numpy()
    if len(diff_ns) > 1 and np.any(np.diff(diff_ns) <= 0):
        if pd.Index(ts).has_duplicates:
            raise ValueError(f"{func_name}: duplicate timestamps are not allowed")
        raise ValueError(f"{func_name}: timestamps must be strictly increasing")

    return pd.Series(ts, index=df.index, copy=False)


def _infer_bar_interval_minutes(ts: pd.Series) -> int:
    diffs = ts.diff().dt.total_seconds().div(60.0)
    valid = diffs[(diffs > 0) & diffs.notna()]
    if len(valid) < 3:
        raise ValueError(
            "add_session_features: unable to infer bar interval; "
            "need at least 3 positive timestamp gaps"
        )

    rounded = valid.round().astype(int)
    counts = rounded.value_counts()
    interval = int(counts[counts == counts.max()].index.min())
    if interval <= 0:
        raise ValueError("add_session_features: inferred non-positive bar interval")
    return interval


def _compute_session_codes(ts: pd.Series) -> np.ndarray:
    seconds = (
        ts.dt.hour.to_numpy(dtype=int) * 3600
        + ts.dt.minute.to_numpy(dtype=int) * 60
        + ts.dt.second.to_numpy(dtype=int)
    )
    session_code = np.full(len(ts), 4, dtype=np.int8)
    session_code[(seconds >= 0) & (seconds < 8 * 3600)] = 0
    session_code[(seconds >= 8 * 3600) & (seconds < 13 * 3600)] = 1
    session_code[(seconds >= 13 * 3600) & (seconds < 17 * 3600)] = 2
    session_code[(seconds >= 17 * 3600) & (seconds < 22 * 3600)] = 3
    return session_code


def _window_overlap_flags(
    start_minutes: np.ndarray,
    *,
    interval_minutes: int,
    window_start: int,
    window_end: int,
) -> np.ndarray:
    """Return 1 where the bar interval overlaps the UTC window.

    Session identity stays bar-start based. Open/active windows use interval
    overlap semantics so coarse bars do not alias away real participation.
    """
    end_minutes = start_minutes + interval_minutes
    overlap_primary = (start_minutes < window_end) & (
        np.minimum(end_minutes, DAY_MINUTES) > window_start
    )

    wrap_end = end_minutes - DAY_MINUTES
    overlap_wrapped = (end_minutes > DAY_MINUTES) & (wrap_end > window_start)
    return (overlap_primary | overlap_wrapped).astype(np.int8)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator != 0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _build_session_group_ids(
    session_code: np.ndarray,
    *,
    ts: pd.Series,
    interval_minutes: int,
) -> np.ndarray:
    if len(session_code) == 0:
        return np.array([], dtype=int)
    day_keys = ts.dt.floor("D").astype("int64").to_numpy()
    starts = np.ones(len(session_code), dtype=bool)
    starts[1:] = (session_code[1:] != session_code[:-1]) | (
        day_keys[1:] != day_keys[:-1]
    )
    return np.cumsum(starts) - 1


def _session_direction(change_atr: float) -> int:
    if np.isfinite(change_atr):
        if change_atr > 0.05:
            return 1
        if change_atr < -0.05:
            return -1
    return 0


def _direction_triple_label(first: int, second: int, third: int) -> str:
    return (
        f"{DIRECTION_LABELS[first]}_"
        f"{DIRECTION_LABELS[second]}_"
        f"{DIRECTION_LABELS[third]}"
    )


def _block_direction(
    *,
    open_price: float,
    close_price: float,
    atr_value: float,
) -> int:
    if np.isfinite(atr_value) and atr_value != 0:
        return _session_direction((close_price - open_price) / atr_value)
    return 0


def _add_research_only_columns(
    out: pd.DataFrame,
    *,
    open_values: np.ndarray,
    session_group_ids: np.ndarray,
    session_open_price: np.ndarray,
    atr_values: np.ndarray,
    session_high_so_far: np.ndarray,
    session_low_so_far: np.ndarray,
    close_values: np.ndarray,
    session_code: np.ndarray,
) -> pd.DataFrame:
    group_series = pd.Series(session_group_ids, index=out.index)
    final_high = (
        pd.Series(session_high_so_far, index=out.index)
        .groupby(group_series)
        .transform("max")
    )
    final_low = (
        pd.Series(session_low_so_far, index=out.index)
        .groupby(group_series)
        .transform("min")
    )
    final_close = (
        pd.Series(close_values, index=out.index).groupby(group_series).transform("last")
    )
    final_atr = (
        pd.Series(atr_values, index=out.index).groupby(group_series).transform("last")
    )

    out["r_session_final_high"] = final_high.astype(float)
    out["r_session_final_low"] = final_low.astype(float)
    out["r_session_final_mid"] = ((final_high + final_low) / 2.0).astype(float)
    out["r_session_final_range"] = (final_high - final_low).astype(float)
    out["r_session_return_final"] = _safe_divide(
        (final_close - session_open_price).to_numpy(dtype=float),
        session_open_price.to_numpy(dtype=float),
    )

    final_change_atr = _safe_divide(
        (final_close - session_open_price).to_numpy(dtype=float),
        final_atr.to_numpy(dtype=float),
    )
    out["r_session_direction_final"] = np.array(
        [_session_direction(val) for val in final_change_atr],
        dtype=np.int8,
    )

    completed_direction: dict[int, dict[str, float]] = {0: {}, 1: {}, 3: {}}
    combo_columns = {
        name: np.full(len(out), np.nan, dtype=object)
        for name, _, _ in SESSION_COMBO_LABELS
    }
    triad_final = np.full(len(out), np.nan, dtype=object)
    triad_label = np.full(len(out), np.nan, dtype=object)
    alt_triad_final = np.full(len(out), np.nan, dtype=object)
    alt_triad_label = np.full(len(out), np.nan, dtype=object)
    day_keys = pd.to_datetime(out["timestamp"], utc=True).dt.floor("D")
    completed_by_day: dict[pd.Timestamp, dict[int, int]] = {}
    active_day_directions: dict[pd.Timestamp, dict[str, int]] = {}

    unique_days = pd.Index(day_keys.unique())
    for day in unique_days:
        mask_day = day_keys == day
        positions = np.flatnonzero(mask_day.to_numpy())
        if len(positions) == 0:
            continue

        asia_positions = positions[session_code[positions] == 0]
        london_active_positions = positions[np.isin(session_code[positions], [1, 2])]
        ny_positions = positions[session_code[positions] == 3]

        day_map: dict[str, int] = {}
        if len(asia_positions) > 0:
            last_atr = float(atr_values[asia_positions[-1]])
            day_map["asia"] = _block_direction(
                open_price=float(open_values[asia_positions[0]]),
                close_price=float(close_values[asia_positions[-1]]),
                atr_value=last_atr,
            )
        if len(london_active_positions) > 0:
            last_atr = float(atr_values[london_active_positions[-1]])
            day_map["london_active"] = _block_direction(
                open_price=float(open_values[london_active_positions[0]]),
                close_price=float(close_values[london_active_positions[-1]]),
                atr_value=last_atr,
            )
        if len(ny_positions) > 0:
            last_atr = float(atr_values[ny_positions[-1]])
            day_map["ny"] = _block_direction(
                open_price=float(open_values[ny_positions[0]]),
                close_price=float(close_values[ny_positions[-1]]),
                atr_value=last_atr,
            )
        active_day_directions[pd.Timestamp(day)] = day_map

    for pos, idx in enumerate(out.index):
        if pos > 0 and session_group_ids[pos] != session_group_ids[pos - 1]:
            prev_gid = int(session_group_ids[pos - 1])
            prev_code = int(session_code[pos - 1])
            prev_day = pd.Timestamp(day_keys.iloc[pos - 1])
            if prev_code in completed_direction:
                prev_dir = int(
                    out.at[
                        out.index[session_group_ids == prev_gid][-1],
                        "r_session_direction_final",
                    ]
                )
                last_row = out.index[session_group_ids == prev_gid][-1]
                completed_direction[prev_code] = {"direction": float(prev_dir)}
                completed_by_day.setdefault(prev_day, {})[prev_code] = prev_dir

        for col_name, left_code, right_code in SESSION_COMBO_LABELS:
            left = completed_direction.get(left_code)
            right = completed_direction.get(right_code)
            if left and right:
                combo_columns[col_name][
                    pos
                ] = f"{int(left['direction'])}_{int(right['direction'])}"

        if pos > 0 and session_group_ids[pos] != session_group_ids[pos - 1]:
            prev_code = int(session_code[pos - 1])
            prev_day = pd.Timestamp(day_keys.iloc[pos - 1])
            if prev_code == 3:
                completed = completed_by_day.get(prev_day, {})
                if all(code in completed for code in (0, 1, 3)):
                    triple = f"{completed[0]}_{completed[1]}_{completed[3]}"
                    triad_final[pos] = triple
                    triad_label[pos] = _direction_triple_label(
                        completed[0], completed[1], completed[3]
                    )
                active_day = active_day_directions.get(prev_day, {})
                if all(key in active_day for key in ("asia", "london_active", "ny")):
                    alt_triple = (
                        f"{active_day['asia']}_"
                        f"{active_day['london_active']}_"
                        f"{active_day['ny']}"
                    )
                    alt_triad_final[pos] = alt_triple
                    alt_triad_label[pos] = _direction_triple_label(
                        active_day["asia"],
                        active_day["london_active"],
                        active_day["ny"],
                    )

    for col_name, values in combo_columns.items():
        out[col_name] = values
    out["r_asia_london_ny_direction_triple_final"] = triad_final
    out["r_asia_london_ny_direction_triple_label"] = triad_label
    out["r_asia_london_active_ny_direction_triple_final"] = alt_triad_final
    out["r_asia_london_active_ny_direction_triple_label"] = alt_triad_label

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pure UTC calendar features.

    Added columns
    -------------
    ``hour_utc``, ``minute_utc``, ``day_of_week``, ``is_friday``,
    ``is_week_open_day``, ``is_week_close_day``, ``session_day_id``
    """
    ts = _normalize_utc_timestamps(df, "add_time_features")
    out = df.copy()

    day_floor = ts.dt.floor("D")
    out["hour_utc"] = ts.dt.hour.astype(np.int8)
    out["minute_utc"] = ts.dt.minute.astype(np.int8)
    out["day_of_week"] = ts.dt.dayofweek.astype(np.int8)
    out["is_friday"] = (out["day_of_week"] == 4).astype(np.int8)
    out["is_week_open_day"] = (out["day_of_week"] == 0).astype(np.int8)
    out["is_week_close_day"] = (out["day_of_week"] == 4).astype(np.int8)
    out["session_day_id"] = day_floor.dt.strftime("%Y-%m-%d")

    return out


def add_session_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add the canonical causal session context layer.

    Required input columns
    ----------------------
    ``timestamp``, ``open``, ``high``, ``low``, ``close``

    Notes
    -----
    * UTC only.
    * Fully causal for all non-``r_`` columns.
    * Research-only columns are added only when requested.
    """
    _require_columns(
        df, ("timestamp", "open", "high", "low", "close"), "add_session_features"
    )
    ts = _normalize_utc_timestamps(df, "add_session_features")
    interval_minutes = _infer_bar_interval_minutes(ts)

    out = add_time_features(df)
    open_values = pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close_values = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)

    if atr_col in out.columns:
        atr_values = pd.to_numeric(out[atr_col], errors="coerce").to_numpy(dtype=float)
    elif atr_col.startswith("atr_") and atr_col[4:].isdigit():
        atr_values = get_atr_array(out, length=int(atr_col[4:])).astype(
            float, copy=False
        )
    else:
        atr_values = get_atr_array(out, length=14).astype(float, copy=False)

    session_code = _compute_session_codes(ts)
    session_names = np.array(
        [SESSION_CODE_TO_NAME[int(code)] for code in session_code], dtype=object
    )

    out["session_code"] = session_code.astype(np.int8)
    out["session_name"] = session_names
    out["session_asia_flag"] = (session_code == 0).astype(np.int8)
    out["session_london_flag"] = (session_code == 1).astype(np.int8)
    out["session_overlap_flag"] = (session_code == 2).astype(np.int8)
    out["session_ny_flag"] = (session_code == 3).astype(np.int8)
    out["session_dead_flag"] = (session_code == 4).astype(np.int8)

    exclusivity = (
        out[
            [
                "session_asia_flag",
                "session_london_flag",
                "session_overlap_flag",
                "session_ny_flag",
                "session_dead_flag",
            ]
        ]
        .sum(axis=1)
        .to_numpy(dtype=int)
    )
    if not np.all(exclusivity == 1):
        raise ValueError("add_session_features: invalid session-flag exclusivity")

    seconds = (
        ts.dt.hour.to_numpy(dtype=int) * 3600
        + ts.dt.minute.to_numpy(dtype=int) * 60
        + ts.dt.second.to_numpy(dtype=int)
    )
    start_minutes = ts.dt.hour.to_numpy(dtype=int) * 60 + ts.dt.minute.to_numpy(
        dtype=int
    )
    out["is_london_open_window"] = _window_overlap_flags(
        start_minutes,
        interval_minutes=interval_minutes,
        window_start=8 * 60,
        window_end=10 * 60,
    )
    out["is_ny_open_window"] = _window_overlap_flags(
        start_minutes,
        interval_minutes=interval_minutes,
        window_start=13 * 60,
        window_end=15 * 60,
    )
    out["is_london_active_window"] = _window_overlap_flags(
        start_minutes,
        interval_minutes=interval_minutes,
        window_start=8 * 60,
        window_end=17 * 60,
    )
    out["is_ny_active_window"] = _window_overlap_flags(
        start_minutes,
        interval_minutes=interval_minutes,
        window_start=13 * 60,
        window_end=22 * 60,
    )
    out["is_dead_zone"] = (session_code == 4).astype(np.int8)

    day_floor = ts.dt.floor("D")
    session_start = day_floor + pd.to_timedelta(
        SESSION_START_MINUTES[session_code], unit="m"
    )
    session_end = day_floor + pd.to_timedelta(
        SESSION_END_MINUTES[session_code], unit="m"
    )

    elapsed_minutes = (
        (ts - session_start).dt.total_seconds().div(60.0).astype(float).to_numpy()
    )
    minutes_since_session_open = np.floor(elapsed_minutes).astype(np.int64)
    session_total_minutes = SESSION_TOTAL_MINUTES[session_code].astype(np.int64)
    session_group_ids = _build_session_group_ids(
        session_code,
        ts=ts,
        interval_minutes=interval_minutes,
    )
    session_group = pd.Series(session_group_ids, index=out.index)
    bars_since_session_open = (
        pd.Series(session_group_ids, index=out.index)
        .groupby(session_group)
        .cumcount()
        .to_numpy(dtype=np.int64)
    )

    remaining_minutes = (
        (session_end - ts).dt.total_seconds().div(60.0).astype(float).to_numpy()
    )
    bars_remaining = np.maximum(
        np.ceil(np.maximum(remaining_minutes, 0.0) / interval_minutes).astype(np.int64)
        - 1,
        0,
    )

    out["bars_since_session_open"] = bars_since_session_open
    out["minutes_since_session_open"] = minutes_since_session_open
    out["session_progress_frac"] = (
        minutes_since_session_open / session_total_minutes.astype(float)
    )
    out["bars_remaining_in_session"] = bars_remaining.astype(np.int64)
    out["is_first_bar_of_session"] = (bars_since_session_open == 0).astype(np.int8)
    out["is_last_bar_of_session"] = (bars_remaining == 0).astype(np.int8)

    session_open_price = (
        pd.Series(open_values, index=out.index)
        .groupby(session_group)
        .transform("first")
    )
    session_high_so_far = (
        pd.Series(high_values, index=out.index).groupby(session_group).cummax()
    )
    session_low_so_far = (
        pd.Series(low_values, index=out.index).groupby(session_group).cummin()
    )
    session_range_so_far = session_high_so_far - session_low_so_far
    session_close_change_from_open = (
        pd.Series(close_values, index=out.index) - session_open_price
    )
    session_close_change_from_open_atr = pd.Series(
        _safe_divide(
            session_close_change_from_open.to_numpy(dtype=float),
            atr_values,
        ),
        index=out.index,
    )
    session_return_so_far = pd.Series(
        _safe_divide(
            session_close_change_from_open.to_numpy(dtype=float),
            session_open_price.to_numpy(dtype=float),
        ),
        index=out.index,
    )
    session_direction_so_far = np.array(
        [
            _session_direction(val)
            for val in session_close_change_from_open_atr.to_numpy(dtype=float)
        ],
        dtype=np.int8,
    )
    session_position_in_range = _safe_divide(
        (pd.Series(close_values, index=out.index) - session_low_so_far).to_numpy(
            dtype=float
        ),
        session_range_so_far.to_numpy(dtype=float),
    )

    out["session_open_price"] = session_open_price.astype(float)
    out["session_high_so_far"] = session_high_so_far.astype(float)
    out["session_low_so_far"] = session_low_so_far.astype(float)
    out["session_range_so_far"] = session_range_so_far.astype(float)
    out["session_range_so_far_atr"] = _safe_divide(
        session_range_so_far.to_numpy(dtype=float), atr_values
    )
    out["session_close_change_from_open"] = session_close_change_from_open.astype(float)
    out["session_close_change_from_open_atr"] = (
        session_close_change_from_open_atr.astype(float)
    )
    out["session_return_so_far"] = session_return_so_far.astype(float)
    out["session_direction_so_far"] = session_direction_so_far
    out["session_position_in_range_so_far"] = session_position_in_range
    session_high_so_far_values = out["session_high_so_far"].to_numpy(dtype=float)
    session_low_so_far_values = out["session_low_so_far"].to_numpy(dtype=float)

    group_meta: dict[int, dict[str, float]] = {}
    for group_id in np.unique(session_group_ids):
        positions = np.flatnonzero(session_group_ids == group_id)
        code = int(session_code[positions[0]])
        group_high = float(np.nanmax(high_values[positions]))
        group_low = float(np.nanmin(low_values[positions]))
        group_open = float(open_values[positions[0]])
        group_close = float(close_values[positions[-1]])
        group_mid = (group_high + group_low) / 2.0
        group_range = group_high - group_low
        group_atr = float(atr_values[positions[-1]]) if len(positions) else np.nan
        group_change_atr = (
            (group_close - group_open) / group_atr
            if np.isfinite(group_atr) and group_atr != 0
            else np.nan
        )
        group_meta[int(group_id)] = {
            "code": code,
            "high": group_high,
            "low": group_low,
            "mid": group_mid,
            "range": group_range,
            "open": group_open,
            "close": group_close,
            "direction": float(_session_direction(group_change_atr)),
        }

    n_rows = len(out)
    prev_cols = [
        "prev_asia_high",
        "prev_asia_low",
        "prev_asia_mid",
        "prev_asia_range",
        "prev_asia_range_atr",
        "dist_to_prev_asia_high_atr",
        "dist_to_prev_asia_low_atr",
        "dist_to_prev_asia_mid_atr",
        "prev_london_high",
        "prev_london_low",
        "prev_london_mid",
        "prev_london_range",
        "prev_london_range_atr",
        "dist_to_prev_london_high_atr",
        "dist_to_prev_london_low_atr",
        "dist_to_prev_london_mid_atr",
        "prev_ny_high",
        "prev_ny_low",
        "prev_ny_mid",
        "prev_ny_range",
        "prev_ny_range_atr",
        "dist_to_prev_ny_high_atr",
        "dist_to_prev_ny_low_atr",
        "dist_to_prev_ny_mid_atr",
    ]
    prev_arrays = {col: np.full(n_rows, np.nan, dtype=float) for col in prev_cols}

    asia_cols = [
        "asia_range_high_so_far",
        "asia_range_low_so_far",
        "asia_range_mid_so_far",
        "asia_range_high_final",
        "asia_range_low_final",
        "asia_range_mid_final",
        "asia_range_width_final",
        "asia_range_width_final_atr",
        "dist_to_asia_high_atr",
        "dist_to_asia_low_atr",
        "dist_to_asia_mid_atr",
    ]
    asia_arrays = {col: np.full(n_rows, np.nan, dtype=float) for col in asia_cols}

    asia_active_flag = np.zeros(n_rows, dtype=np.int8)
    asia_complete_flag = np.zeros(n_rows, dtype=np.int8)
    inside_asia_flag = np.zeros(n_rows, dtype=np.int8)
    above_asia_flag = np.zeros(n_rows, dtype=np.int8)
    below_asia_flag = np.zeros(n_rows, dtype=np.int8)
    asia_high_swept_flag = np.zeros(n_rows, dtype=np.int8)
    asia_low_swept_flag = np.zeros(n_rows, dtype=np.int8)

    last_completed: dict[int, dict[str, float] | None] = {0: None, 1: None, 3: None}

    for pos in range(n_rows):
        if pos > 0 and session_group_ids[pos] != session_group_ids[pos - 1]:
            prev_group_id = int(session_group_ids[pos - 1])
            prev_meta = group_meta[prev_group_id]
            prev_code = int(prev_meta["code"])
            if prev_code in last_completed:
                last_completed[prev_code] = prev_meta

        for code, prefix in SESSION_FAMILY_PREFIX.items():
            state = last_completed[code]
            if state is None:
                continue

            high = float(state["high"])
            low = float(state["low"])
            mid = float(state["mid"])
            width = float(state["range"])
            close = float(close_values[pos])
            atr = float(atr_values[pos])

            prev_arrays[f"prev_{prefix}_high"][pos] = high
            prev_arrays[f"prev_{prefix}_low"][pos] = low
            prev_arrays[f"prev_{prefix}_mid"][pos] = mid
            prev_arrays[f"prev_{prefix}_range"][pos] = width

            if np.isfinite(atr) and atr != 0:
                prev_arrays[f"prev_{prefix}_range_atr"][pos] = width / atr
                prev_arrays[f"dist_to_prev_{prefix}_high_atr"][pos] = (
                    close - high
                ) / atr
                prev_arrays[f"dist_to_prev_{prefix}_low_atr"][pos] = (close - low) / atr
                prev_arrays[f"dist_to_prev_{prefix}_mid_atr"][pos] = (close - mid) / atr

        if session_code[pos] == 0:
            asia_active_flag[pos] = 1
            asia_complete_flag[pos] = 0
            asia_arrays["asia_range_high_so_far"][pos] = session_high_so_far_values[pos]
            asia_arrays["asia_range_low_so_far"][pos] = session_low_so_far_values[pos]
            asia_arrays["asia_range_mid_so_far"][pos] = (
                asia_arrays["asia_range_high_so_far"][pos]
                + asia_arrays["asia_range_low_so_far"][pos]
            ) / 2.0
            continue

        last_asia = last_completed[0]
        if last_asia is None:
            continue

        asia_complete_flag[pos] = 1
        asia_arrays["asia_range_high_final"][pos] = float(last_asia["high"])
        asia_arrays["asia_range_low_final"][pos] = float(last_asia["low"])
        asia_arrays["asia_range_mid_final"][pos] = float(last_asia["mid"])
        asia_arrays["asia_range_width_final"][pos] = float(last_asia["range"])
        if np.isfinite(atr_values[pos]) and atr_values[pos] != 0:
            asia_arrays["asia_range_width_final_atr"][pos] = (
                float(last_asia["range"]) / atr_values[pos]
            )
            asia_arrays["dist_to_asia_high_atr"][pos] = (
                close_values[pos] - float(last_asia["high"])
            ) / atr_values[pos]
            asia_arrays["dist_to_asia_low_atr"][pos] = (
                close_values[pos] - float(last_asia["low"])
            ) / atr_values[pos]
            asia_arrays["dist_to_asia_mid_atr"][pos] = (
                close_values[pos] - float(last_asia["mid"])
            ) / atr_values[pos]

        inside_asia_flag[pos] = np.int8(
            float(last_asia["low"]) <= close_values[pos] <= float(last_asia["high"])
        )
        above_asia_flag[pos] = np.int8(close_values[pos] > float(last_asia["high"]))
        below_asia_flag[pos] = np.int8(close_values[pos] < float(last_asia["low"]))
        asia_high_swept_flag[pos] = np.int8(high_values[pos] > float(last_asia["high"]))
        asia_low_swept_flag[pos] = np.int8(low_values[pos] < float(last_asia["low"]))

    for col, values in prev_arrays.items():
        out[col] = values
    for col, values in asia_arrays.items():
        out[col] = values
    out["asia_range_active_flag"] = asia_active_flag
    out["asia_range_complete_flag"] = asia_complete_flag
    out["inside_asia_range_flag"] = inside_asia_flag
    out["above_asia_high_flag"] = above_asia_flag
    out["below_asia_low_flag"] = below_asia_flag
    out["asia_high_swept_flag"] = asia_high_swept_flag
    out["asia_low_swept_flag"] = asia_low_swept_flag

    if include_research_only:
        out = _add_research_only_columns(
            out,
            open_values=open_values,
            session_group_ids=session_group_ids,
            session_open_price=session_open_price,
            atr_values=atr_values,
            session_high_so_far=out["session_high_so_far"].to_numpy(dtype=float),
            session_low_so_far=out["session_low_so_far"].to_numpy(dtype=float),
            close_values=close_values,
            session_code=session_code,
        )

    return out


def add_session_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the pre-freeze session API."""
    out = add_session_features(df, include_research_only=False)
    legacy_code_map = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    out["session"] = out["session_code"].map(legacy_code_map).astype(np.int8)
    out["london_open"] = out["is_london_open_window"].astype(np.int8)
    out["ny_open"] = out["is_ny_open_window"].astype(np.int8)
    out["hours_since_session"] = out["minutes_since_session_open"].astype(float) / 60.0
    return out
