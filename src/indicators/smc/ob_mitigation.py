"""
Canonical OB mitigation tracker.

This layer consumes the sparse activation-row OB contract from ``add_ob`` and
adds:
- mitigation milestones stamped on activation rows for auditability
- dense active-side exports for live-safe downstream consumption
- legacy compatibility aliases for older pipeline consumers
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc
from src.indicators.smc.ob import (
    OB_STATE_ACTIVE_FRESH,
    OB_STATE_ACTIVE_TOUCHED,
    OB_STATE_INVALIDATED,
    OB_STATE_MITIGATED_FULL,
    OB_STATE_MITIGATED_PARTIAL,
    OB_STATE_RETIRED,
)

OB_MITIGATION_STATE_COLUMNS = [
    "ob_has_been_touched",
    "ob_has_partial_mitigation",
    "ob_has_full_mitigation",
]

OB_MITIGATION_INDEX_COLUMNS = [
    "ob_first_touch_idx",
    "ob_first_touch_ts",
    "ob_first_partial_idx",
    "ob_first_partial_ts",
    "ob_first_partial_mitigation_idx",
    "ob_first_partial_mitigation_ts",
    "ob_first_full_mitigation_idx",
    "ob_first_full_mitigation_ts",
    "ob_invalidation_idx",
    "ob_invalidation_ts",
    "ob_retire_idx",
    "ob_retire_ts",
]

OB_MITIGATION_METRIC_COLUMNS = [
    "ob_mitigation_penetration_abs",
    "ob_mitigation_penetration_frac",
    "ob_mitigation_penetration_atr",
]

OB_MITIGATION_COUNT_COLUMNS = [
    "ob_touch_count",
    "ob_mitigation_count",
    "ob_bars_since_first_touch",
    "ob_bars_since_last_touch",
]

OB_MITIGATION_SUPPORT_COLUMNS = [
    "ob_midpoint_touch_flag",
    "ob_midpoint_touch_idx",
]

OB_ACTIVE_COLUMNS = [
    "ob_bull_active",
    "ob_bull_active_id",
    "ob_bull_active_low",
    "ob_bull_active_high",
    "ob_bull_active_mid",
    "ob_bull_active_width_atr",
    "ob_bull_active_state",
    "ob_bull_active_age_bars",
    "ob_bull_active_touch_count",
    "ob_bull_active_penetration_frac",
    "ob_bull_active_effective_strength",
    "ob_bull_active_count",
    "ob_bull_first_touch_flag",
    "ob_bear_active",
    "ob_bear_active_id",
    "ob_bear_active_low",
    "ob_bear_active_high",
    "ob_bear_active_mid",
    "ob_bear_active_width_atr",
    "ob_bear_active_state",
    "ob_bear_active_age_bars",
    "ob_bear_active_touch_count",
    "ob_bear_active_penetration_frac",
    "ob_bear_active_effective_strength",
    "ob_bear_active_count",
    "ob_bear_first_touch_flag",
]

OB_MITIGATION_COMPAT_COLUMNS = [
    "ob_unmitigated_bull",
    "ob_unmitigated_bear",
    "ob_first_retest",
]

OB_MITIGATION_COLUMNS = (
    OB_MITIGATION_STATE_COLUMNS
    + OB_MITIGATION_INDEX_COLUMNS
    + OB_MITIGATION_METRIC_COLUMNS
    + OB_MITIGATION_COUNT_COLUMNS
    + OB_MITIGATION_SUPPORT_COLUMNS
    + OB_ACTIVE_COLUMNS
    + OB_MITIGATION_COMPAT_COLUMNS
)


@dataclass
class _ObEvent:
    ob_id: int
    side: int
    activate_idx: int
    source_idx: int
    zone_low: float
    zone_high: float
    zone_mid: float
    zone_height_abs: float
    zone_height_atr: float
    strength: float
    state: int = OB_STATE_ACTIVE_FRESH
    touch_count: int = 0
    mitigation_count: int = 0
    first_touch_idx: int | None = None
    first_partial_idx: int | None = None
    first_full_idx: int | None = None
    invalidation_idx: int | None = None
    midpoint_touch_idx: int | None = None
    last_touch_idx: int | None = None
    retired_idx: int | None = None
    terminal_idx: int | None = None
    max_penetration_abs: float = 0.0
    max_penetration_frac: float = 0.0
    max_penetration_atr: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.state in {
            OB_STATE_ACTIVE_FRESH,
            OB_STATE_ACTIVE_TOUCHED,
            OB_STATE_MITIGATED_PARTIAL,
        }


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _clip_unit(value: float | None, *, default: float = np.nan) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(np.clip(value, 0.0, 1.0))


def _distance_triplet(
    close_value: float, low_value: float, high_value: float
) -> tuple[float, float]:
    mid_value = (low_value + high_value) / 2.0
    dist_mid = abs(close_value - mid_value)
    if close_value < low_value:
        return dist_mid, low_value - close_value
    if close_value > high_value:
        return dist_mid, close_value - high_value
    return dist_mid, min(close_value - low_value, high_value - close_value)


def _interaction_modifier(event: _ObEvent) -> float:
    if event.state == OB_STATE_ACTIVE_FRESH:
        return 1.0
    if event.state == OB_STATE_ACTIVE_TOUCHED:
        return 0.85
    return float(max(0.35, 1.0 - 0.5 * event.max_penetration_frac))


def _age_decay(age_bars: int) -> float:
    return float(2.0 ** (-max(age_bars, 0) / 12.0))


def _effective_strength(
    *,
    event: _ObEvent,
    close_value: float,
    atr_value: float,
    row_idx: int,
) -> float:
    dist_mid, dist_near = _distance_triplet(
        close_value, event.zone_low, event.zone_high
    )
    if np.isfinite(atr_value) and atr_value > 0:
        distance_modifier = 1.0 / (1.0 + (dist_near / atr_value))
    else:
        distance_modifier = 1.0
    age_bars = max(row_idx - event.activate_idx, 0)
    return float(
        _clip_unit(event.strength, default=0.0)
        * _age_decay(age_bars)
        * _interaction_modifier(event)
        * distance_modifier
    )


def _select_active_event(
    events: list[_ObEvent],
    *,
    close_value: float,
    atr_value: float,
    row_idx: int,
) -> _ObEvent | None:
    best: _ObEvent | None = None
    best_key: tuple[float, float, float, int] | None = None
    for event in events:
        if not event.is_active or row_idx < event.activate_idx:
            continue
        dist_mid, dist_near = _distance_triplet(
            close_value, event.zone_low, event.zone_high
        )
        if np.isfinite(atr_value) and atr_value > 0:
            distance_penalty = dist_near / atr_value
        else:
            distance_penalty = np.inf
        age_bars = row_idx - event.activate_idx
        score = (
            _clip_unit(event.strength, default=0.0)
            * _age_decay(age_bars)
            * _interaction_modifier(event)
            * (1.0 / (1.0 + max(distance_penalty, 0.0)))
        )
        key = (score, -distance_penalty, -age_bars, -event.ob_id)
        if best_key is None or key > best_key:
            best = event
            best_key = key
    return best


def _ensure_columns(out: pd.DataFrame) -> pd.DataFrame:
    n = len(out)
    for col in OB_MITIGATION_STATE_COLUMNS:
        out[col] = np.zeros(n, dtype=np.int8)
    timestamp_cols = {
        "ob_first_touch_ts",
        "ob_first_partial_ts",
        "ob_first_partial_mitigation_ts",
        "ob_first_full_mitigation_ts",
        "ob_invalidation_ts",
        "ob_retire_ts",
    }
    for col in (
        OB_MITIGATION_INDEX_COLUMNS
        + OB_MITIGATION_METRIC_COLUMNS
        + OB_MITIGATION_COUNT_COLUMNS
    ):
        if col in timestamp_cols:
            out[col] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        else:
            out[col] = np.full(n, np.nan)
    out["ob_midpoint_touch_flag"] = np.zeros(n, dtype=np.int8)
    out["ob_midpoint_touch_idx"] = np.full(n, np.nan)

    out["ob_bull_active"] = np.zeros(n, dtype=np.int8)
    out["ob_bull_active_id"] = np.zeros(n, dtype=np.int64)
    out["ob_bull_active_low"] = np.full(n, np.nan)
    out["ob_bull_active_high"] = np.full(n, np.nan)
    out["ob_bull_active_mid"] = np.full(n, np.nan)
    out["ob_bull_active_width_atr"] = np.full(n, np.nan)
    out["ob_bull_active_state"] = np.zeros(n, dtype=np.int8)
    out["ob_bull_active_age_bars"] = np.full(n, np.nan)
    out["ob_bull_active_touch_count"] = np.zeros(n, dtype=np.int32)
    out["ob_bull_active_penetration_frac"] = np.full(n, np.nan)
    out["ob_bull_active_effective_strength"] = np.full(n, np.nan)
    out["ob_bull_active_count"] = np.zeros(n, dtype=np.int32)
    out["ob_bull_first_touch_flag"] = np.zeros(n, dtype=np.int8)

    out["ob_bear_active"] = np.zeros(n, dtype=np.int8)
    out["ob_bear_active_id"] = np.zeros(n, dtype=np.int64)
    out["ob_bear_active_low"] = np.full(n, np.nan)
    out["ob_bear_active_high"] = np.full(n, np.nan)
    out["ob_bear_active_mid"] = np.full(n, np.nan)
    out["ob_bear_active_width_atr"] = np.full(n, np.nan)
    out["ob_bear_active_state"] = np.zeros(n, dtype=np.int8)
    out["ob_bear_active_age_bars"] = np.full(n, np.nan)
    out["ob_bear_active_touch_count"] = np.zeros(n, dtype=np.int32)
    out["ob_bear_active_penetration_frac"] = np.full(n, np.nan)
    out["ob_bear_active_effective_strength"] = np.full(n, np.nan)
    out["ob_bear_active_count"] = np.zeros(n, dtype=np.int32)
    out["ob_bear_first_touch_flag"] = np.zeros(n, dtype=np.int8)

    out["ob_unmitigated_bull"] = np.zeros(n, dtype=np.int8)
    out["ob_unmitigated_bear"] = np.zeros(n, dtype=np.int8)
    out["ob_first_retest"] = np.zeros(n, dtype=np.int8)
    return out


def add_ob_mitigation(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    invalidation_buffer_atr: float = 0.0,
    max_active_age_bars: int = 96,
) -> pd.DataFrame:
    """Append mitigation, lifecycle, and dense active-side OB columns."""
    require_ohlc(df, caller="add_ob_mitigation")
    out = _ensure_columns(df.copy())

    if "ob_id" not in out.columns:
        return out

    event_rows = out[pd.to_numeric(out["ob_id"], errors="coerce").fillna(0) > 0].copy()
    if event_rows.empty:
        return out

    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    atr_arr = get_atr_array(out, length=atr_length).astype(float, copy=False)
    ts = (
        pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if "timestamp" in out.columns
        else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    )

    events_by_activate: dict[int, list[_ObEvent]] = {}
    ordered_events: list[_ObEvent] = []
    for row_idx, row in event_rows.iterrows():
        activate_idx = int(pd.to_numeric(row.get("ob_activate_idx"), errors="coerce"))
        source_idx = int(pd.to_numeric(row.get("ob_source_idx"), errors="coerce"))
        event = _ObEvent(
            ob_id=int(row["ob_id"]),
            side=int(pd.to_numeric(row["ob_side"], errors="coerce")),
            activate_idx=activate_idx,
            source_idx=source_idx,
            zone_low=float(pd.to_numeric(row["ob_zone_low"], errors="coerce")),
            zone_high=float(pd.to_numeric(row["ob_zone_high"], errors="coerce")),
            zone_mid=float(pd.to_numeric(row["ob_zone_mid"], errors="coerce")),
            zone_height_abs=float(
                pd.to_numeric(row["ob_zone_height_abs"], errors="coerce")
            ),
            zone_height_atr=float(
                pd.to_numeric(row["ob_zone_height_atr"], errors="coerce")
            ),
            strength=float(
                pd.to_numeric(row.get("ob_strength", np.nan), errors="coerce")
            ),
        )
        events_by_activate.setdefault(activate_idx, []).append(event)
        ordered_events.append(event)

    active_events: list[_ObEvent] = []
    side_map = {1: "bull", -1: "bear"}

    for row_idx in range(len(out)):
        row_first_touch = {1: 0, -1: 0}
        if row_idx in events_by_activate:
            active_events.extend(events_by_activate[row_idx])

        next_active: list[_ObEvent] = []
        for event in active_events:
            if not event.is_active:
                continue

            if row_idx <= event.activate_idx:
                next_active.append(event)
                continue

            bar_high = high_arr[row_idx]
            bar_low = low_arr[row_idx]
            bar_close = close_arr[row_idx]
            atr_value = atr_arr[row_idx]
            invalidation_level = (
                event.zone_low - invalidation_buffer_atr * atr_value
                if event.side == 1
                else event.zone_high + invalidation_buffer_atr * atr_value
            )

            touches = bool(bar_high >= event.zone_low and bar_low <= event.zone_high)
            penetration_abs = 0.0
            penetration_frac = 0.0
            midpoint_touched = False

            if touches:
                event.touch_count += 1
                if event.first_touch_idx is None:
                    event.first_touch_idx = row_idx
                    row_first_touch[event.side] = 1
                event.last_touch_idx = row_idx
                midpoint_touched = bool(
                    bar_high >= event.zone_mid and bar_low <= event.zone_mid
                )
                if midpoint_touched and event.midpoint_touch_idx is None:
                    event.midpoint_touch_idx = row_idx

                if event.side == 1:
                    low_inside_zone = max(bar_low, event.zone_low)
                    penetration_abs = float(max(event.zone_high - low_inside_zone, 0.0))
                else:
                    high_inside_zone = min(bar_high, event.zone_high)
                    penetration_abs = float(max(high_inside_zone - event.zone_low, 0.0))

                penetration_frac = float(
                    np.clip(
                        _safe_fraction(penetration_abs, event.zone_height_abs),
                        0.0,
                        1.0,
                    )
                )
                penetration_atr = _safe_fraction(penetration_abs, atr_value)

                event.max_penetration_abs = max(
                    event.max_penetration_abs, penetration_abs
                )
                event.max_penetration_frac = max(
                    event.max_penetration_frac, penetration_frac
                )
                if np.isfinite(penetration_atr):
                    event.max_penetration_atr = max(
                        event.max_penetration_atr, penetration_atr
                    )

                if penetration_frac > 0.0:
                    event.mitigation_count += 1
                    if event.first_partial_idx is None and penetration_frac < 1.0:
                        event.first_partial_idx = row_idx
                if penetration_frac >= 1.0 and event.first_full_idx is None:
                    event.first_full_idx = row_idx

            invalidated = (
                bool(np.isfinite(bar_close) and bar_close < invalidation_level)
                if event.side == 1
                else bool(np.isfinite(bar_close) and bar_close > invalidation_level)
            )

            if invalidated:
                event.invalidation_idx = (
                    row_idx
                    if event.invalidation_idx is None
                    else event.invalidation_idx
                )
                event.state = OB_STATE_INVALIDATED
                event.terminal_idx = row_idx
                continue

            if event.first_full_idx is not None:
                event.state = OB_STATE_MITIGATED_FULL
                event.terminal_idx = row_idx
                continue

            age_since_activation = row_idx - event.activate_idx
            if age_since_activation >= max_active_age_bars:
                event.state = OB_STATE_RETIRED
                event.retired_idx = row_idx
                event.terminal_idx = row_idx
                continue

            if penetration_frac > 0.0:
                event.state = (
                    OB_STATE_MITIGATED_PARTIAL
                    if penetration_frac < 1.0
                    else OB_STATE_MITIGATED_FULL
                )
            elif touches and event.state == OB_STATE_ACTIVE_FRESH:
                event.state = OB_STATE_ACTIVE_TOUCHED

            next_active.append(event)

        active_events = next_active

        bull_events = [
            event for event in active_events if event.side == 1 and event.is_active
        ]
        bear_events = [
            event for event in active_events if event.side == -1 and event.is_active
        ]
        out.at[row_idx, "ob_bull_active_count"] = len(bull_events)
        out.at[row_idx, "ob_bear_active_count"] = len(bear_events)
        out.at[row_idx, "ob_bull_first_touch_flag"] = row_first_touch[1]
        out.at[row_idx, "ob_bear_first_touch_flag"] = row_first_touch[-1]
        out.at[row_idx, "ob_first_retest"] = 1 if any(row_first_touch.values()) else 0
        out.at[row_idx, "ob_unmitigated_bull"] = int(
            any(event.state == OB_STATE_ACTIVE_FRESH for event in bull_events)
        )
        out.at[row_idx, "ob_unmitigated_bear"] = int(
            any(event.state == OB_STATE_ACTIVE_FRESH for event in bear_events)
        )

        close_value = close_arr[row_idx]
        atr_value = atr_arr[row_idx]
        for side, prefix in side_map.items():
            side_events = bull_events if side == 1 else bear_events
            selected = _select_active_event(
                side_events,
                close_value=close_value,
                atr_value=atr_value,
                row_idx=row_idx,
            )
            if selected is None:
                continue
            age_bars = row_idx - selected.activate_idx
            out.at[row_idx, f"ob_{prefix}_active"] = 1
            out.at[row_idx, f"ob_{prefix}_active_id"] = selected.ob_id
            out.at[row_idx, f"ob_{prefix}_active_low"] = selected.zone_low
            out.at[row_idx, f"ob_{prefix}_active_high"] = selected.zone_high
            out.at[row_idx, f"ob_{prefix}_active_mid"] = selected.zone_mid
            out.at[row_idx, f"ob_{prefix}_active_width_atr"] = selected.zone_height_atr
            out.at[row_idx, f"ob_{prefix}_active_state"] = selected.state
            out.at[row_idx, f"ob_{prefix}_active_age_bars"] = float(age_bars)
            out.at[row_idx, f"ob_{prefix}_active_touch_count"] = selected.touch_count
            out.at[row_idx, f"ob_{prefix}_active_penetration_frac"] = (
                selected.max_penetration_frac
            )
            out.at[row_idx, f"ob_{prefix}_active_effective_strength"] = (
                _effective_strength(
                    event=selected,
                    close_value=close_value,
                    atr_value=atr_value,
                    row_idx=row_idx,
                )
            )

    for event in ordered_events:
        row_idx = event.activate_idx
        final_idx = (
            event.terminal_idx if event.terminal_idx is not None else len(out) - 1
        )
        out.at[row_idx, "ob_state"] = event.state
        out.at[row_idx, "ob_is_active"] = int(event.is_active)
        out.at[row_idx, "ob_is_fresh"] = int(event.state == OB_STATE_ACTIVE_FRESH)
        out.at[row_idx, "ob_is_invalidated"] = int(event.state == OB_STATE_INVALIDATED)
        out.at[row_idx, "ob_is_retired"] = int(event.state == OB_STATE_RETIRED)
        out.at[row_idx, "ob_age_bars"] = float(final_idx - event.source_idx)
        out.at[row_idx, "ob_age_since_activation_bars"] = float(
            final_idx - event.activate_idx
        )

        out.at[row_idx, "ob_has_been_touched"] = int(event.first_touch_idx is not None)
        out.at[row_idx, "ob_has_partial_mitigation"] = int(
            event.first_partial_idx is not None
        )
        out.at[row_idx, "ob_has_full_mitigation"] = int(
            event.first_full_idx is not None
        )

        out.at[row_idx, "ob_first_touch_idx"] = (
            float(event.first_touch_idx)
            if event.first_touch_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_first_touch_ts"] = (
            ts.iloc[event.first_touch_idx]
            if event.first_touch_idx is not None
            else pd.NaT
        )
        out.at[row_idx, "ob_first_partial_idx"] = (
            float(event.first_partial_idx)
            if event.first_partial_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_first_partial_ts"] = (
            ts.iloc[event.first_partial_idx]
            if event.first_partial_idx is not None
            else pd.NaT
        )
        out.at[row_idx, "ob_first_partial_mitigation_idx"] = (
            float(event.first_partial_idx)
            if event.first_partial_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_first_partial_mitigation_ts"] = (
            ts.iloc[event.first_partial_idx]
            if event.first_partial_idx is not None
            else pd.NaT
        )
        out.at[row_idx, "ob_first_full_mitigation_idx"] = (
            float(event.first_full_idx) if event.first_full_idx is not None else np.nan
        )
        out.at[row_idx, "ob_first_full_mitigation_ts"] = (
            ts.iloc[event.first_full_idx]
            if event.first_full_idx is not None
            else pd.NaT
        )
        out.at[row_idx, "ob_invalidation_idx"] = (
            float(event.invalidation_idx)
            if event.invalidation_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_invalidation_ts"] = (
            ts.iloc[event.invalidation_idx]
            if event.invalidation_idx is not None
            else pd.NaT
        )
        out.at[row_idx, "ob_retire_idx"] = (
            float(event.retired_idx) if event.retired_idx is not None else np.nan
        )
        out.at[row_idx, "ob_retire_ts"] = (
            ts.iloc[event.retired_idx] if event.retired_idx is not None else pd.NaT
        )

        out.at[row_idx, "ob_mitigation_penetration_abs"] = event.max_penetration_abs
        out.at[row_idx, "ob_mitigation_penetration_frac"] = event.max_penetration_frac
        out.at[row_idx, "ob_mitigation_penetration_atr"] = event.max_penetration_atr
        out.at[row_idx, "ob_touch_count"] = event.touch_count
        out.at[row_idx, "ob_mitigation_count"] = event.mitigation_count
        out.at[row_idx, "ob_bars_since_first_touch"] = (
            float(final_idx - event.first_touch_idx)
            if event.first_touch_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_bars_since_last_touch"] = (
            float(final_idx - event.last_touch_idx)
            if event.last_touch_idx is not None
            else np.nan
        )
        out.at[row_idx, "ob_midpoint_touch_flag"] = int(
            event.midpoint_touch_idx is not None
        )
        out.at[row_idx, "ob_midpoint_touch_idx"] = (
            float(event.midpoint_touch_idx)
            if event.midpoint_touch_idx is not None
            else np.nan
        )

    int8_cols = [
        "ob_has_been_touched",
        "ob_has_partial_mitigation",
        "ob_has_full_mitigation",
        "ob_midpoint_touch_flag",
        "ob_bull_active",
        "ob_bull_active_state",
        "ob_bull_first_touch_flag",
        "ob_bear_active",
        "ob_bear_active_state",
        "ob_bear_first_touch_flag",
        "ob_unmitigated_bull",
        "ob_unmitigated_bear",
        "ob_first_retest",
    ]
    for col in int8_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int8)

    for col in ("ob_bull_active_id", "ob_bear_active_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)

    for col in (
        "ob_bull_active_count",
        "ob_bear_active_count",
        "ob_bull_active_touch_count",
        "ob_bear_active_touch_count",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int32)

    return out


__all__ = [
    "OB_MITIGATION_COLUMNS",
    "OB_ACTIVE_COLUMNS",
    "OB_MITIGATION_COMPAT_COLUMNS",
    "add_ob_mitigation",
]
