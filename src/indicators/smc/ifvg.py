"""
smc/ifvg.py

Canonical IFVG consumer built from FVG Core debug truth.

IFVG is a second-order event: a raw invalidated FVG becomes an inverse FVG with
opposite polarity. Source identity is always the raw FVG event ID, never the
currently selected structural/fill-facing FVG export.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns
from src.indicators.smc.fvg import STATE_INVALIDATED

DIR_BULL = 1
DIR_BEAR = -1

IFVG_STATE_NONE = 0
IFVG_STATE_ACTIVE_UNTESTED = 1
IFVG_STATE_ACTIVE_TESTED = 2
IFVG_STATE_ACTIVE_HOLDING = 3
IFVG_STATE_ACTIVE_FAILING = 4
IFVG_STATE_FULLY_RECLAIMED = 5
IFVG_STATE_EXPIRED = 6

EPS = 1e-12
DEFAULT_MIN_SOURCE_WIDTH_ATR = 0.1
DEFAULT_MIN_SOURCE_AGE_BARS = 2
DEFAULT_MIN_INVERSION_BODY_ATR = 0.0
DEFAULT_MIN_CROSS_DISTANCE_ATR = 0.0
MAX_IFVG_ACTIVE_AGE_BARS = 48
HALF_LIFE_BARS = 12.0

HOLDING_INTERACTION_MODIFIER = 0.9
FAILING_INTERACTION_MODIFIER = 0.7

DETECT_COLUMNS = [
    "ifvg_bull_detect_flag",
    "ifvg_bull_detect_idx",
    "ifvg_bull_detect_ts",
    "ifvg_bull_event_id",
    "ifvg_bull_direction",
    "ifvg_bull_source_fvg_event_id",
    "ifvg_bull_source_fvg_detect_idx",
    "ifvg_bull_source_fvg_direction",
    "ifvg_bear_detect_flag",
    "ifvg_bear_detect_idx",
    "ifvg_bear_detect_ts",
    "ifvg_bear_event_id",
    "ifvg_bear_direction",
    "ifvg_bear_source_fvg_event_id",
    "ifvg_bear_source_fvg_detect_idx",
    "ifvg_bear_source_fvg_direction",
]

GEOMETRY_COLUMNS = [
    "ifvg_bull_low",
    "ifvg_bull_high",
    "ifvg_bull_mid",
    "ifvg_bull_width",
    "ifvg_bull_width_atr",
    "ifvg_bear_low",
    "ifvg_bear_high",
    "ifvg_bear_mid",
    "ifvg_bear_width",
    "ifvg_bear_width_atr",
]

INVERSION_QUALITY_COLUMNS = [
    "ifvg_bull_source_age_bars",
    "ifvg_bull_source_width_atr",
    "ifvg_bull_source_fill_pct_at_inversion",
    "ifvg_bull_inversion_body_atr",
    "ifvg_bull_inversion_range_atr",
    "ifvg_bull_inversion_body_frac",
    "ifvg_bull_inversion_close_location",
    "ifvg_bull_inversion_conviction_location",
    "ifvg_bull_cross_distance_atr",
    "ifvg_bull_inversion_score",
    "ifvg_bear_source_age_bars",
    "ifvg_bear_source_width_atr",
    "ifvg_bear_source_fill_pct_at_inversion",
    "ifvg_bear_inversion_body_atr",
    "ifvg_bear_inversion_range_atr",
    "ifvg_bear_inversion_body_frac",
    "ifvg_bear_inversion_close_location",
    "ifvg_bear_inversion_conviction_location",
    "ifvg_bear_cross_distance_atr",
    "ifvg_bear_inversion_score",
]

ACTIVE_STATE_COLUMNS = [
    "ifvg_bull_active",
    "ifvg_bull_active_id",
    "ifvg_bull_active_low",
    "ifvg_bull_active_high",
    "ifvg_bull_active_mid",
    "ifvg_bull_active_width",
    "ifvg_bull_active_width_atr",
    "ifvg_bull_active_age",
    "ifvg_bull_active_age_decay",
    "ifvg_bull_active_effective_significance",
    "ifvg_bull_active_state",
    "ifvg_bull_active_since_idx",
    "ifvg_bull_active_test_count",
    "ifvg_bull_active_hold_count",
    "ifvg_bull_first_retest_flag",
    "ifvg_bull_retest_depth_frac",
    "ifvg_bull_distance_to_mid_atr",
    "ifvg_bull_distance_to_near_edge_atr",
    "ifvg_bull_distance_to_far_edge_atr",
    "ifvg_bull_active_count",
    "ifvg_bull_zone_fully_reclaimed",
    "ifvg_bear_active",
    "ifvg_bear_active_id",
    "ifvg_bear_active_low",
    "ifvg_bear_active_high",
    "ifvg_bear_active_mid",
    "ifvg_bear_active_width",
    "ifvg_bear_active_width_atr",
    "ifvg_bear_active_age",
    "ifvg_bear_active_age_decay",
    "ifvg_bear_active_effective_significance",
    "ifvg_bear_active_state",
    "ifvg_bear_active_since_idx",
    "ifvg_bear_active_test_count",
    "ifvg_bear_active_hold_count",
    "ifvg_bear_first_retest_flag",
    "ifvg_bear_retest_depth_frac",
    "ifvg_bear_distance_to_mid_atr",
    "ifvg_bear_distance_to_near_edge_atr",
    "ifvg_bear_distance_to_far_edge_atr",
    "ifvg_bear_active_count",
    "ifvg_bear_zone_fully_reclaimed",
]

LIFECYCLE_COLUMNS = [
    "ifvg_bull_first_retest_idx",
    "ifvg_bull_fully_reclaimed_idx",
    "ifvg_bull_expiry_idx",
    "ifvg_bull_terminal_state",
    "ifvg_bear_first_retest_idx",
    "ifvg_bear_fully_reclaimed_idx",
    "ifvg_bear_expiry_idx",
    "ifvg_bear_terminal_state",
]

IFVG_COMPAT_COLUMNS = [
    "ifvg_bull",
    "ifvg_bear",
    "ifvg_width_atr",
    "ifvg_bull_break",
    "ifvg_bear_break",
    "ifvg_bull_candidate",
    "ifvg_bear_candidate",
    "ifvg_bull_retest",
    "ifvg_bear_retest",
    "ifvg_state_bull",
    "ifvg_state_bear",
    "ifvg_origin_idx_bull",
    "ifvg_origin_idx_bear",
]

ALL_IFVG_COLUMNS = (
    DETECT_COLUMNS
    + GEOMETRY_COLUMNS
    + INVERSION_QUALITY_COLUMNS
    + ACTIVE_STATE_COLUMNS
    + LIFECYCLE_COLUMNS
)

EVENT_TABLE_REQUIRED_COLUMNS = {
    "event_id",
    "side",
    "detect_idx",
    "low",
    "high",
    "mid",
    "width",
    "width_atr",
    "invalidation_idx",
    "terminal_state",
}

ACTIVE_MEMBER_REQUIRED_COLUMNS = {
    "row_idx",
    "side",
    "source_count",
    "source_event_id",
    "fill_pct",
}


@dataclass
class _IfvgEvent:
    event_id: int
    side: str
    direction: int
    detect_idx: int
    detect_ts: pd.Timestamp
    low: float
    high: float
    mid: float
    width: float
    width_atr: float
    source_fvg_event_id: int
    source_fvg_detect_idx: int
    source_fvg_direction: int
    source_age_bars: int
    source_width_atr: float
    source_fill_pct_at_inversion: float
    inversion_body_atr: float
    inversion_range_atr: float
    inversion_body_frac: float
    inversion_close_location: float
    inversion_conviction_location: float
    cross_distance_atr: float
    inversion_score: float
    active_since_idx: int
    current_state: int = IFVG_STATE_ACTIVE_UNTESTED
    active_test_count: int = 0
    active_hold_count: int = 0
    first_retest_idx: int | None = None
    fully_reclaimed_idx: int | None = None
    expiry_idx: int | None = None
    terminal_state: int = IFVG_STATE_NONE
    prev_testing: bool = False
    episode_hold_recorded: bool = False


def _side_to_direction(side: str) -> int:
    return DIR_BULL if side == "bull" else DIR_BEAR


def _opposite_side(side: str) -> str:
    return "bear" if side == "bull" else "bull"


def _distance_triplet(
    close_value: float, low_value: float, high_value: float
) -> tuple[float, float, float]:
    mid = (low_value + high_value) / 2.0
    dist_mid = abs(close_value - mid)
    if close_value < low_value:
        return dist_mid, low_value - close_value, high_value - close_value
    if close_value > high_value:
        return dist_mid, close_value - high_value, close_value - low_value
    return (
        dist_mid,
        min(close_value - low_value, high_value - close_value),
        max(close_value - low_value, high_value - close_value),
    )


def _protected_side_close(
    side: str, close_value: float, low_value: float, high_value: float
) -> bool:
    if side == "bull":
        return close_value > high_value
    return close_value < low_value


def _testing_condition(
    side: str,
    detect_idx: int,
    row_idx: int,
    low_value: float,
    high_value: float,
    bar_low: float,
    bar_high: float,
) -> bool:
    if row_idx <= detect_idx:
        return False
    if side == "bull":
        return bar_low <= high_value
    return bar_high >= low_value


def _retest_depth_frac(
    side: str, low_value: float, high_value: float, bar_low: float, bar_high: float
) -> float:
    width = max(high_value - low_value, EPS)
    if side == "bull":
        penetration = max(0.0, high_value - bar_low)
    else:
        penetration = max(0.0, bar_high - low_value)
    return float(np.clip(penetration / width, 0.0, 1.0))


def _fully_reclaimed(
    side: str, close_value: float, low_value: float, high_value: float
) -> bool:
    if side == "bull":
        return close_value < low_value
    return close_value > high_value


def _inside_zone(close_value: float, low_value: float, high_value: float) -> bool:
    return low_value <= close_value <= high_value


def _compute_inversion_score(
    *,
    inversion_body_atr: float,
    cross_distance_atr: float,
    inversion_body_frac: float,
    source_age_bars: int,
    source_width_atr: float,
    source_fill_pct_at_inversion: float,
) -> float:
    score = (
        (np.clip(inversion_body_atr, 0.0, 2.0) / 2.0) * 0.30
        + np.clip(cross_distance_atr, 0.0, 1.0) * 0.20
        + inversion_body_frac * 0.15
        + np.clip(source_age_bars / 20.0, 0.0, 1.0) * 0.15
        + np.clip(source_width_atr, 0.0, 1.0) * 0.10
        + (1.0 - source_fill_pct_at_inversion) * 0.10
    )
    return float(np.clip(score, 0.0, 1.0))


def _build_source_fill_lookup(
    active_members: pd.DataFrame,
) -> dict[tuple[int, str, int], float]:
    scoped = active_members[active_members["source_count"] == 1].copy()
    if scoped.empty:
        return {}
    fill_col = (
        "historical_fill_pct" if "historical_fill_pct" in scoped.columns else "fill_pct"
    )
    grouped = scoped.groupby(["row_idx", "side", "source_event_id"], sort=False)
    lookup: dict[tuple[int, str, int], float] = {}
    for key, rows in grouped:
        if len(rows) != 1:
            raise ValueError(
                "IFVG source fill lookup expected one single-source member per "
                f"(row_idx, side, source_event_id), got {len(rows)} for {key}."
            )
        lookup[(int(key[0]), str(key[1]), int(key[2]))] = float(rows[fill_col].iloc[0])
    return lookup


def _resolve_source_fill_pct_at_inversion(
    *,
    fill_lookup: dict[tuple[int, str, int], float],
    source_event_id: int,
    source_side: str,
    detect_idx: int,
) -> float:
    lookup_key = (detect_idx - 1, source_side, source_event_id)
    if lookup_key not in fill_lookup:
        raise ValueError(
            "IFVG source_fill_pct_at_inversion missing for "
            f"source_event_id={source_event_id}, side={source_side}, row={detect_idx - 1}."
        )
    return float(fill_lookup[lookup_key])


def _accept_ifvg_candidate(candidate: dict[str, object]) -> tuple[bool, str | None]:
    ordered_checks = (
        ("atr_unavailable", not bool(candidate["atr_available"])),
        ("invalid_width", not bool(candidate["width_valid"])),
        ("source_too_small", not bool(candidate["source_width_ok"])),
        ("source_too_young", not bool(candidate["source_age_ok"])),
        ("inversion_body_too_small", not bool(candidate["inversion_body_ok"])),
        ("cross_distance_too_small", not bool(candidate["cross_distance_ok"])),
    )
    for reason, failed in ordered_checks:
        if failed:
            return False, reason
    return True, None


def _candidate_sort_key(row: pd.Series) -> tuple[float, int, float, int]:
    return (
        -float(row["inversion_score"]),
        -int(row["source_fvg_detect_idx"]),
        -float(row["width_atr"]),
        int(row["source_fvg_event_id"]),
    )


def _age_decay(age_bars: int) -> float:
    return float(2.0 ** (-max(age_bars, 0) / HALF_LIFE_BARS))


def _interaction_modifier(state_for_row: int) -> float:
    if state_for_row == IFVG_STATE_ACTIVE_HOLDING:
        return HOLDING_INTERACTION_MODIFIER
    if state_for_row == IFVG_STATE_ACTIVE_FAILING:
        return FAILING_INTERACTION_MODIFIER
    return 1.0


def _distance_modifier(
    *,
    close_value: float,
    atr_value: float,
    low_value: float,
    high_value: float,
) -> float:
    if not np.isfinite(atr_value) or atr_value <= 0:
        return 0.0
    _dist_mid, dist_near, _dist_far = _distance_triplet(
        close_value, low_value, high_value
    )
    distance_atr = dist_near / atr_value
    return float(1.0 / (1.0 + max(distance_atr, 0.0)))


def _effective_significance(
    *,
    event: _IfvgEvent,
    row_metrics: dict[str, object],
    current_idx: int,
    close_value: float,
    atr_value: float,
) -> float:
    age_bars = current_idx - event.detect_idx
    return float(
        np.clip(
            event.inversion_score
            * _age_decay(age_bars)
            * _interaction_modifier(int(row_metrics["state_for_row"]))
            * _distance_modifier(
                close_value=close_value,
                atr_value=atr_value,
                low_value=event.low,
                high_value=event.high,
            ),
            0.0,
            1.0,
        )
    )


def _select_active_ifvg(
    *,
    side: str,
    close_value: float,
    atr_value: float,
    current_idx: int,
    active_rows: list[tuple[_IfvgEvent, dict[str, object]]],
    selector: str,
) -> tuple[_IfvgEvent, dict[str, object]] | None:
    if not active_rows:
        return None
    if selector not in {"nearest", "significance"}:
        raise ValueError("IFVG selector must be one of {'nearest', 'significance'}")

    def _key(
        item: tuple[_IfvgEvent, dict[str, object]],
    ) -> tuple[float, float, int, float, int]:
        event, _row_metrics = item
        dist_mid, dist_near, _dist_far = _distance_triplet(
            close_value, event.low, event.high
        )
        atr_safe = atr_value if np.isfinite(atr_value) and atr_value > 0 else 1.0
        return (
            dist_near / atr_safe,
            dist_mid / atr_safe,
            -event.detect_idx,
            -event.width_atr,
            event.event_id,
        )

    if selector == "nearest":
        return min(active_rows, key=_key)

    return max(
        active_rows,
        key=lambda item: (
            _effective_significance(
                event=item[0],
                row_metrics=item[1],
                current_idx=current_idx,
                close_value=close_value,
                atr_value=atr_value,
            ),
            -_key(item)[0],
            item[0].detect_idx,
            item[0].width_atr,
            -item[0].event_id,
        ),
    )


def _build_invalidated_source_candidates(
    df: pd.DataFrame,
    *,
    debug_tables: dict[str, pd.DataFrame],
    atr_length: int = 14,
    min_source_width_atr: float = DEFAULT_MIN_SOURCE_WIDTH_ATR,
    min_source_age_bars: int = DEFAULT_MIN_SOURCE_AGE_BARS,
    min_inversion_body_atr: float = DEFAULT_MIN_INVERSION_BODY_ATR,
    min_cross_distance_atr: float = DEFAULT_MIN_CROSS_DISTANCE_ATR,
) -> pd.DataFrame:
    require_columns(
        df, {"timestamp", "open", "high", "low", "close"}, caller="add_ifvg"
    )
    event_table = debug_tables.get("event_table")
    active_members = debug_tables.get("active_member_table")
    if event_table is None or active_members is None:
        raise ValueError(
            "add_ifvg requires debug_tables with 'event_table' and 'active_member_table'"
        )

    if event_table.empty:
        return pd.DataFrame()

    require_columns(event_table, EVENT_TABLE_REQUIRED_COLUMNS, caller="add_ifvg")
    if not active_members.empty:
        require_columns(
            active_members, ACTIVE_MEMBER_REQUIRED_COLUMNS, caller="add_ifvg"
        )

    atr = get_atr_array(df, atr_length)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    fill_lookup = _build_source_fill_lookup(active_members)

    # NOTE: IFVG gap — merged-zone invalidations are invisible here.
    # When two FVG zones merge in the core, all constituent raw events are stamped
    # STATE_MERGED (not STATE_INVALIDATED). The merged representation has
    # source_count > 1, so a subsequent invalidation of the merged zone does NOT
    # stamp any raw event as STATE_INVALIDATED. This filter therefore never sees
    # merged-zone invalidations. Only single-source raw FVGs can produce IFVGs.
    invalidated = event_table[event_table["terminal_state"] == STATE_INVALIDATED].copy()
    if invalidated.empty:
        return pd.DataFrame()

    invalidated = invalidated.sort_values(
        ["detect_idx", "side", "event_id"]
    ).reset_index(drop=True)
    candidate_records: list[dict[str, object]] = []
    next_ifvg_event_id = 1

    for source in invalidated.itertuples(index=False):
        detect_idx = int(float(source.invalidation_idx))
        source_side = str(source.side)
        ifvg_side = _opposite_side(source_side)
        direction = _side_to_direction(ifvg_side)
        source_direction = _side_to_direction(source_side)
        width = float(source.width)
        atr_value = float(atr[detect_idx]) if 0 <= detect_idx < len(df) else np.nan
        atr_available = bool(np.isfinite(atr_value) and atr_value > 0.0)
        source_fill_pct_at_inversion = _resolve_source_fill_pct_at_inversion(
            fill_lookup=fill_lookup,
            source_event_id=int(source.event_id),
            source_side=source_side,
            detect_idx=detect_idx,
        )

        range_value = float(high[detect_idx] - low[detect_idx])
        body_value = float(abs(close[detect_idx] - open_[detect_idx]))
        source_age_bars = int(detect_idx - int(source.detect_idx))
        if atr_available:
            inversion_body_atr = body_value / atr_value
            inversion_range_atr = range_value / atr_value
            cross_distance_atr = (
                float((close[detect_idx] - float(source.high)) / atr_value)
                if ifvg_side == "bull"
                else float((float(source.low) - close[detect_idx]) / atr_value)
            )
        else:
            inversion_body_atr = np.nan
            inversion_range_atr = np.nan
            cross_distance_atr = np.nan

        if range_value > 0.0:
            inversion_body_frac = float(body_value / range_value)
            inversion_close_location = float(
                (close[detect_idx] - low[detect_idx]) / range_value
            )
            inversion_conviction_location = (
                inversion_close_location
                if ifvg_side == "bull"
                else float((high[detect_idx] - close[detect_idx]) / range_value)
            )
        else:
            inversion_body_frac = 0.0
            inversion_close_location = 0.5
            inversion_conviction_location = 0.5

        candidate = {
            "ifvg_side": ifvg_side,
            "direction": int(direction),
            "detect_idx": detect_idx,
            "detect_ts": timestamps.iloc[detect_idx],
            "source_fvg_event_id": int(source.event_id),
            "source_fvg_detect_idx": int(source.detect_idx),
            "source_fvg_direction": int(source_direction),
            "low": float(source.low),
            "high": float(source.high),
            "mid": float(source.mid),
            "width": width,
            "width_atr": float(width / atr_value) if atr_available else np.nan,
            "source_age_bars": source_age_bars,
            "source_width_atr": float(source.width_atr),
            "source_fill_pct_at_inversion": float(source_fill_pct_at_inversion),
            "inversion_body_atr": float(inversion_body_atr),
            "inversion_range_atr": float(inversion_range_atr),
            "inversion_body_frac": float(inversion_body_frac),
            "inversion_close_location": float(inversion_close_location),
            "inversion_conviction_location": float(inversion_conviction_location),
            "cross_distance_atr": float(cross_distance_atr),
            "atr_available": atr_available,
            "width_valid": bool(width > 0.0),
            "source_width_ok": bool(float(source.width_atr) >= min_source_width_atr),
            "source_age_ok": bool(source_age_bars >= min_source_age_bars),
            "inversion_body_ok": bool(
                np.isfinite(inversion_body_atr)
                and inversion_body_atr >= min_inversion_body_atr
            ),
            "cross_distance_ok": bool(
                np.isfinite(cross_distance_atr)
                and cross_distance_atr >= min_cross_distance_atr
            ),
        }
        accepted, rejection_reason = _accept_ifvg_candidate(candidate)
        candidate["accepted"] = bool(accepted)
        candidate["rejection_reason"] = rejection_reason
        candidate["inversion_score"] = (
            _compute_inversion_score(
                inversion_body_atr=float(inversion_body_atr),
                cross_distance_atr=(
                    float(cross_distance_atr)
                    if np.isfinite(cross_distance_atr)
                    else 0.0
                ),
                inversion_body_frac=float(inversion_body_frac),
                source_age_bars=source_age_bars,
                source_width_atr=float(source.width_atr),
                source_fill_pct_at_inversion=float(source_fill_pct_at_inversion),
            )
            if accepted
            else np.nan
        )
        candidate["ifvg_event_id"] = int(next_ifvg_event_id) if accepted else 0
        if accepted:
            next_ifvg_event_id += 1
        candidate_records.append(candidate)

    candidates = pd.DataFrame(candidate_records)
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["detect_idx", "ifvg_side", "source_fvg_event_id"]
    ).reset_index(drop=True)


def _init_output(n: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in (
        DETECT_COLUMNS
        + GEOMETRY_COLUMNS
        + INVERSION_QUALITY_COLUMNS
        + LIFECYCLE_COLUMNS
    ):
        if column.endswith("_flag"):
            arrays[column] = np.zeros(n, dtype=np.int8)
        elif column.endswith("_event_id") or column.endswith("_direction"):
            arrays[column] = np.zeros(n, dtype=np.int64)
        elif column.endswith("_ts"):
            arrays[column] = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        elif column.endswith("_terminal_state"):
            arrays[column] = np.zeros(n, dtype=np.int8)
        else:
            arrays[column] = np.full(n, np.nan, dtype=float)

    for column in ACTIVE_STATE_COLUMNS:
        if (
            column.endswith("_active")
            or column.endswith("_first_retest_flag")
            or column.endswith("_zone_fully_reclaimed")
        ):
            arrays[column] = np.zeros(n, dtype=np.int8)
        elif column.endswith("_active_id"):
            arrays[column] = np.zeros(n, dtype=np.int64)
        elif column.endswith("_active_state") or column.endswith("_active_count"):
            arrays[column] = np.zeros(n, dtype=np.int32)
        elif column.endswith("_active_test_count") or column.endswith(
            "_active_hold_count"
        ):
            arrays[column] = np.zeros(n, dtype=np.int32)
        else:
            arrays[column] = np.full(n, np.nan, dtype=float)

    arrays["ifvg_bull"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bear"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_width_atr"] = np.full(n, np.nan, dtype=float)
    arrays["ifvg_bull_break"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bear_break"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bull_candidate"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bear_candidate"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bull_retest"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_bear_retest"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_state_bull"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_state_bear"] = np.zeros(n, dtype=np.int8)
    arrays["ifvg_origin_idx_bull"] = np.full(n, -1, dtype=np.int64)
    arrays["ifvg_origin_idx_bear"] = np.full(n, -1, dtype=np.int64)
    return arrays


def _advance_ifvg_event(
    event: _IfvgEvent,
    *,
    row_idx: int,
    close_value: float,
    high_value: float,
    low_value: float,
    max_ifvg_age_bars: int | None,
) -> dict[str, object]:
    row_metrics = {
        "active_after_row": False,
        "state_for_row": IFVG_STATE_NONE,
        "first_retest_flag": 0,
        "retest_depth_frac": 0.0,
        "zone_fully_reclaimed": 0,
    }
    if row_idx < event.detect_idx or event.terminal_state != IFVG_STATE_NONE:
        return row_metrics

    if row_idx == event.detect_idx:
        event.current_state = IFVG_STATE_ACTIVE_UNTESTED
        event.prev_testing = False
        row_metrics["active_after_row"] = True
        row_metrics["state_for_row"] = event.current_state
        return row_metrics

    testing = _testing_condition(
        event.side,
        event.detect_idx,
        row_idx,
        event.low,
        event.high,
        low_value,
        high_value,
    )
    if testing:
        row_metrics["retest_depth_frac"] = _retest_depth_frac(
            event.side,
            event.low,
            event.high,
            low_value,
            high_value,
        )
        if not event.prev_testing:
            event.active_test_count += 1
            event.episode_hold_recorded = False
            if event.first_retest_idx is None:
                event.first_retest_idx = row_idx
                row_metrics["first_retest_flag"] = 1

    reclaimed = _fully_reclaimed(event.side, close_value, event.low, event.high)
    expired = max_ifvg_age_bars is not None and (row_idx - event.detect_idx) >= int(
        max_ifvg_age_bars
    )
    if reclaimed:
        event.fully_reclaimed_idx = row_idx
        event.terminal_state = IFVG_STATE_FULLY_RECLAIMED
        event.current_state = IFVG_STATE_FULLY_RECLAIMED
        event.prev_testing = False
        row_metrics["zone_fully_reclaimed"] = 1
        return row_metrics
    if expired:
        event.expiry_idx = row_idx
        event.terminal_state = IFVG_STATE_EXPIRED
        event.current_state = IFVG_STATE_EXPIRED
        event.prev_testing = False
        return row_metrics

    if testing:
        if _protected_side_close(event.side, close_value, event.low, event.high):
            event.current_state = IFVG_STATE_ACTIVE_HOLDING
            if not event.episode_hold_recorded:
                event.active_hold_count += 1
                event.episode_hold_recorded = True
        elif _inside_zone(close_value, event.low, event.high):
            event.current_state = IFVG_STATE_ACTIVE_FAILING
        # NOTE: IFVG_STATE_ACTIVE_TESTED (state 2) is intentionally never assigned
        # here. After excluding _fully_reclaimed (handled above), every possible
        # close value lands in either _protected_side_close or _inside_zone, so
        # this else branch is unreachable for both bull and bear sides.
    else:
        # Price is no longer in contact with the zone. Allow a one-bar escape from
        # FAILING → HOLDING (price was stuck inside, then closed on protected side
        # without re-entering). All other cases reset to UNTESTED so the state
        # reflects the current bar, not stale contact from prior bars.
        if event.current_state == IFVG_STATE_ACTIVE_FAILING and _protected_side_close(
            event.side, close_value, event.low, event.high
        ):
            event.current_state = IFVG_STATE_ACTIVE_HOLDING
        else:
            event.current_state = IFVG_STATE_ACTIVE_UNTESTED

    event.prev_testing = testing
    row_metrics["active_after_row"] = True
    row_metrics["state_for_row"] = event.current_state
    return row_metrics


def add_ifvg(
    df: pd.DataFrame,
    *,
    debug_tables: dict[str, pd.DataFrame],
    atr_length: int = 14,
    min_source_width_atr: float = DEFAULT_MIN_SOURCE_WIDTH_ATR,
    min_source_age_bars: int = DEFAULT_MIN_SOURCE_AGE_BARS,
    min_inversion_body_atr: float = DEFAULT_MIN_INVERSION_BODY_ATR,
    min_cross_distance_atr: float = DEFAULT_MIN_CROSS_DISTANCE_ATR,
    max_ifvg_age_bars: int | None = MAX_IFVG_ACTIVE_AGE_BARS,
    selector: str = "significance",
) -> pd.DataFrame:
    if selector not in {"nearest", "significance"}:
        raise ValueError("IFVG selector must be one of {'nearest', 'significance'}")
    if max_ifvg_age_bars is None or max_ifvg_age_bars <= 0:
        raise ValueError("max_ifvg_age_bars must be a positive integer")

    out = df.copy()
    n = len(out)
    arrays = _init_output(n)
    if n == 0:
        for column in ALL_IFVG_COLUMNS + IFVG_COMPAT_COLUMNS:
            out[column] = arrays[column]
        return out

    candidates = _build_invalidated_source_candidates(
        out,
        debug_tables=debug_tables,
        atr_length=atr_length,
        min_source_width_atr=min_source_width_atr,
        min_source_age_bars=min_source_age_bars,
        min_inversion_body_atr=min_inversion_body_atr,
        min_cross_distance_atr=min_cross_distance_atr,
    )
    if candidates.empty:
        for column in ALL_IFVG_COLUMNS + IFVG_COMPAT_COLUMNS:
            out[column] = arrays[column]
        return out

    canonical = candidates[candidates["accepted"]].copy()
    canonical = canonical.sort_values(
        by=["detect_idx", "ifvg_side", "ifvg_event_id"],
        key=lambda s: s.map({"bull": 0, "bear": 1}).fillna(s),
    ).reset_index(drop=True)

    events: list[_IfvgEvent] = []
    for row in canonical.itertuples(index=False):
        events.append(
            _IfvgEvent(
                event_id=int(row.ifvg_event_id),
                side=str(row.ifvg_side),
                direction=int(row.direction),
                detect_idx=int(row.detect_idx),
                detect_ts=pd.to_datetime(row.detect_ts, utc=True),
                low=float(row.low),
                high=float(row.high),
                mid=float(row.mid),
                width=float(row.width),
                width_atr=float(row.width_atr),
                source_fvg_event_id=int(row.source_fvg_event_id),
                source_fvg_detect_idx=int(row.source_fvg_detect_idx),
                source_fvg_direction=int(row.source_fvg_direction),
                source_age_bars=int(row.source_age_bars),
                source_width_atr=float(row.source_width_atr),
                source_fill_pct_at_inversion=float(row.source_fill_pct_at_inversion),
                inversion_body_atr=float(row.inversion_body_atr),
                inversion_range_atr=float(row.inversion_range_atr),
                inversion_body_frac=float(row.inversion_body_frac),
                inversion_close_location=float(row.inversion_close_location),
                inversion_conviction_location=float(row.inversion_conviction_location),
                cross_distance_atr=float(row.cross_distance_atr),
                inversion_score=float(row.inversion_score),
                active_since_idx=int(row.detect_idx),
            )
        )

    atr = get_atr_array(out, atr_length)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)

    compat_detect_by_side_row: dict[tuple[str, int], pd.Series] = {}
    for (_detect_idx, side), group in canonical.groupby(
        ["detect_idx", "ifvg_side"], sort=False
    ):
        chosen = sorted(
            [row for _idx, row in group.iterrows()],
            key=_candidate_sort_key,
        )[0]
        compat_detect_by_side_row[(str(side), int(chosen["detect_idx"]))] = chosen

    for row_idx in range(n):
        active_by_side: dict[str, list[tuple[_IfvgEvent, dict[str, object]]]] = {
            "bull": [],
            "bear": [],
        }
        any_first_retest = {"bull": False, "bear": False}
        any_reclaimed = {"bull": False, "bear": False}

        for event in events:
            row_metrics = _advance_ifvg_event(
                event,
                row_idx=row_idx,
                close_value=float(close[row_idx]),
                high_value=float(high[row_idx]),
                low_value=float(low[row_idx]),
                max_ifvg_age_bars=max_ifvg_age_bars,
            )
            any_first_retest[event.side] = any_first_retest[event.side] or bool(
                row_metrics["first_retest_flag"]
            )
            any_reclaimed[event.side] = any_reclaimed[event.side] or bool(
                row_metrics["zone_fully_reclaimed"]
            )
            if bool(row_metrics["active_after_row"]):
                active_by_side[event.side].append((event, row_metrics))

        for side in ("bull", "bear"):
            side_prefix = f"ifvg_{side}"
            arrays[f"{side_prefix}_active_count"][row_idx] = len(active_by_side[side])
            arrays[f"{side_prefix}_zone_fully_reclaimed"][row_idx] = int(
                any_reclaimed[side]
            )

            selected = _select_active_ifvg(
                side=side,
                close_value=float(close[row_idx]),
                atr_value=float(atr[row_idx]),
                current_idx=row_idx,
                active_rows=active_by_side[side],
                selector=selector,
            )
            if selected is None:
                continue

            event, row_metrics = selected
            dist_mid, dist_near, dist_far = _distance_triplet(
                float(close[row_idx]), event.low, event.high
            )
            atr_safe = (
                float(atr[row_idx])
                if np.isfinite(atr[row_idx]) and atr[row_idx] > 0
                else np.nan
            )
            arrays[f"{side_prefix}_active"][row_idx] = 1
            arrays[f"{side_prefix}_active_id"][row_idx] = int(event.event_id)
            arrays[f"{side_prefix}_active_low"][row_idx] = float(event.low)
            arrays[f"{side_prefix}_active_high"][row_idx] = float(event.high)
            arrays[f"{side_prefix}_active_mid"][row_idx] = float(event.mid)
            arrays[f"{side_prefix}_active_width"][row_idx] = float(event.width)
            arrays[f"{side_prefix}_active_width_atr"][row_idx] = float(event.width_atr)
            active_age = int(row_idx - event.detect_idx)
            arrays[f"{side_prefix}_active_age"][row_idx] = float(active_age)
            arrays[f"{side_prefix}_active_age_decay"][row_idx] = _age_decay(active_age)
            arrays[f"{side_prefix}_active_effective_significance"][row_idx] = (
                _effective_significance(
                    event=event,
                    row_metrics=row_metrics,
                    current_idx=row_idx,
                    close_value=float(close[row_idx]),
                    atr_value=float(atr[row_idx]),
                )
            )
            arrays[f"{side_prefix}_active_state"][row_idx] = int(
                row_metrics["state_for_row"]
            )
            arrays[f"{side_prefix}_active_since_idx"][row_idx] = float(
                event.active_since_idx
            )
            arrays[f"{side_prefix}_active_test_count"][row_idx] = int(
                event.active_test_count
            )
            arrays[f"{side_prefix}_active_hold_count"][row_idx] = int(
                event.active_hold_count
            )
            arrays[f"{side_prefix}_first_retest_flag"][row_idx] = int(
                row_metrics["first_retest_flag"]
            )
            arrays[f"{side_prefix}_retest_depth_frac"][row_idx] = float(
                row_metrics["retest_depth_frac"]
            )
            arrays[f"{side_prefix}_distance_to_mid_atr"][row_idx] = (
                dist_mid / atr_safe if np.isfinite(atr_safe) else np.nan
            )
            arrays[f"{side_prefix}_distance_to_near_edge_atr"][row_idx] = (
                dist_near / atr_safe if np.isfinite(atr_safe) else np.nan
            )
            arrays[f"{side_prefix}_distance_to_far_edge_atr"][row_idx] = (
                dist_far / atr_safe if np.isfinite(atr_safe) else np.nan
            )

            arrays[f"ifvg_state_{side}"][row_idx] = int(row_metrics["state_for_row"])
            arrays[f"ifvg_origin_idx_{side}"][row_idx] = int(
                event.source_fvg_detect_idx
            )

        arrays["ifvg_bull_retest"][row_idx] = int(any_first_retest["bull"])
        arrays["ifvg_bear_retest"][row_idx] = int(any_first_retest["bear"])

    for event in events:
        detect_idx = int(event.detect_idx)
        side = event.side
        prefix = f"ifvg_{side}"

        chosen_sparse = compat_detect_by_side_row.get((side, detect_idx))
        if (
            chosen_sparse is not None
            and int(chosen_sparse["ifvg_event_id"]) == event.event_id
        ):
            arrays[f"{prefix}_detect_flag"][detect_idx] = 1
            arrays[f"{prefix}_detect_idx"][detect_idx] = float(event.detect_idx)
            arrays[f"{prefix}_detect_ts"][detect_idx] = event.detect_ts.tz_localize(
                None
            ).to_datetime64()
            arrays[f"{prefix}_event_id"][detect_idx] = int(event.event_id)
            arrays[f"{prefix}_direction"][detect_idx] = int(event.direction)
            arrays[f"{prefix}_source_fvg_event_id"][detect_idx] = int(
                event.source_fvg_event_id
            )
            arrays[f"{prefix}_source_fvg_detect_idx"][detect_idx] = float(
                event.source_fvg_detect_idx
            )
            arrays[f"{prefix}_source_fvg_direction"][detect_idx] = int(
                event.source_fvg_direction
            )
            arrays[f"{prefix}_low"][detect_idx] = float(event.low)
            arrays[f"{prefix}_high"][detect_idx] = float(event.high)
            arrays[f"{prefix}_mid"][detect_idx] = float(event.mid)
            arrays[f"{prefix}_width"][detect_idx] = float(event.width)
            arrays[f"{prefix}_width_atr"][detect_idx] = float(event.width_atr)
            arrays[f"{prefix}_source_age_bars"][detect_idx] = float(
                event.source_age_bars
            )
            arrays[f"{prefix}_source_width_atr"][detect_idx] = float(
                event.source_width_atr
            )
            arrays[f"{prefix}_source_fill_pct_at_inversion"][detect_idx] = float(
                event.source_fill_pct_at_inversion
            )
            arrays[f"{prefix}_inversion_body_atr"][detect_idx] = float(
                event.inversion_body_atr
            )
            arrays[f"{prefix}_inversion_range_atr"][detect_idx] = float(
                event.inversion_range_atr
            )
            arrays[f"{prefix}_inversion_body_frac"][detect_idx] = float(
                event.inversion_body_frac
            )
            arrays[f"{prefix}_inversion_close_location"][detect_idx] = float(
                event.inversion_close_location
            )
            arrays[f"{prefix}_inversion_conviction_location"][detect_idx] = float(
                event.inversion_conviction_location
            )
            arrays[f"{prefix}_cross_distance_atr"][detect_idx] = float(
                event.cross_distance_atr
            )
            arrays[f"{prefix}_inversion_score"][detect_idx] = float(
                event.inversion_score
            )
            arrays[f"{prefix}_first_retest_idx"][detect_idx] = (
                float(event.first_retest_idx)
                if event.first_retest_idx is not None
                else np.nan
            )
            arrays[f"{prefix}_fully_reclaimed_idx"][detect_idx] = (
                float(event.fully_reclaimed_idx)
                if event.fully_reclaimed_idx is not None
                else np.nan
            )
            arrays[f"{prefix}_expiry_idx"][detect_idx] = (
                float(event.expiry_idx) if event.expiry_idx is not None else np.nan
            )
            arrays[f"{prefix}_terminal_state"][detect_idx] = int(event.terminal_state)

            arrays[prefix] = arrays[f"{prefix}_detect_flag"]
            arrays[f"{prefix}_break"][detect_idx] = 1
            arrays[f"{prefix}_candidate"][detect_idx] = 1

    arrays["ifvg_width_atr"] = np.where(
        arrays["ifvg_bull_detect_flag"] == 1,
        arrays["ifvg_bull_width_atr"],
        np.where(
            arrays["ifvg_bear_detect_flag"] == 1,
            arrays["ifvg_bear_width_atr"],
            np.nan,
        ),
    )

    ifvg_df = pd.DataFrame(arrays, index=out.index)
    existing = [
        col for col in ALL_IFVG_COLUMNS + IFVG_COMPAT_COLUMNS if col in out.columns
    ]
    if existing:
        out = out.drop(columns=existing)
    return pd.concat([out, ifvg_df[ALL_IFVG_COLUMNS + IFVG_COMPAT_COLUMNS]], axis=1)
