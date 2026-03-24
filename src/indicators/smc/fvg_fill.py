"""
smc/fvg_fill.py

Canonical FVG Fill consumer built on top of FVG Core.

Doctrine
--------
- Pure function: never mutates the input DataFrame.
- Selected-zone features follow the fill-active representation per side.
- Structural FVG Core active outputs remain intact and are used as fallback
  context when no touched/filling representation exists.
- Multi-zone summaries reconstruct the same-side active representation pool
  from detect rows and causal lifecycle semantics.
- Annotation/origin columns are never read.
- Dense fill columns are overwritten deterministically if already present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns
from src.indicators.smc.fvg import (
    HALF_LIFE_BARS,
    MAX_ACTIVE_AGE_BARS,
    PARTIAL_FILL_GTE_50_MODIFIER,
    PARTIAL_FILL_LT_25_MODIFIER,
    PARTIAL_FILL_LT_50_MODIFIER,
    STATE_ACTIVE_PARTIAL_FILL,
    STATE_ACTIVE_TOUCHED,
    STATE_ACTIVE_UNTOUCHED,
    TOUCHED_ONLY_MODIFIER,
)

EPS = 1e-12
FILL_TYPE_NONE = 0
FILL_TYPE_WICK_ONLY = 1
FILL_TYPE_ORDERLY = 2
FILL_TYPE_AGGRESSIVE = 3
FILL_VELOCITY_LOOKBACKS = (1, 3, 5)

SELECTED_ACTIVE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "fvg_bull_active",
    "fvg_bull_active_id",
    "fvg_bull_active_low",
    "fvg_bull_active_high",
    "fvg_bull_active_mid",
    "fvg_bull_active_width",
    "fvg_bull_active_width_atr",
    "fvg_bull_active_fill_pct",
    "fvg_bull_active_touch_count",
    "fvg_bull_active_state",
    "fvg_bull_active_count",
    "fvg_bull_active_age_bars",
    "fvg_bull_first_retest_flag",
    "fvg_bear_active",
    "fvg_bear_active_id",
    "fvg_bear_active_low",
    "fvg_bear_active_high",
    "fvg_bear_active_mid",
    "fvg_bear_active_width",
    "fvg_bear_active_width_atr",
    "fvg_bear_active_fill_pct",
    "fvg_bear_active_touch_count",
    "fvg_bear_active_state",
    "fvg_bear_active_count",
    "fvg_bear_active_age_bars",
    "fvg_bear_first_retest_flag",
}

POOL_RECONSTRUCTION_COLUMNS = {
    "fvg_bull_detect_flag",
    "fvg_bull_event_id",
    "fvg_bull_low",
    "fvg_bull_high",
    "fvg_bull_width",
    "fvg_bull_first_touch_idx",
    "fvg_bull_full_fill_idx",
    "fvg_bull_invalidation_idx",
    "fvg_bull_expiry_idx",
    "fvg_bull_terminal_state",
    "fvg_bear_detect_flag",
    "fvg_bear_event_id",
    "fvg_bear_low",
    "fvg_bear_high",
    "fvg_bear_width",
    "fvg_bear_first_touch_idx",
    "fvg_bear_full_fill_idx",
    "fvg_bear_invalidation_idx",
    "fvg_bear_expiry_idx",
    "fvg_bear_terminal_state",
}

FVG_FILL_COLUMNS = [
    "fvg_fill_bull_remaining_low",
    "fvg_fill_bull_remaining_high",
    "fvg_fill_bull_remaining_mid",
    "fvg_fill_bull_remaining_width",
    "fvg_fill_bull_remaining_width_atr",
    "fvg_fill_bull_remaining_frac",
    "fvg_fill_bull_fill_pct_delta_1",
    "fvg_fill_bull_fill_pct_delta_3",
    "fvg_fill_bull_fill_pct_delta_5",
    "fvg_fill_bull_fill_type",
    "fvg_fill_bull_bounce_strength_atr",
    "fvg_fill_bull_bars_since_touch",
    "fvg_fill_bull_bars_since_first_touch",
    "fvg_fill_bull_touch_rejection_flag",
    "fvg_fill_bull_dist_to_remaining_mid_atr",
    "fvg_fill_bull_dist_to_remaining_near_atr",
    "fvg_fill_bull_dist_to_remaining_far_atr",
    "fvg_fill_bull_mean_fill_pct",
    "fvg_fill_bull_max_fill_pct",
    "fvg_fill_bull_pool_mean_fill_pct",
    "fvg_fill_bull_pool_max_fill_pct",
    "fvg_fill_bull_untouched_count",
    "fvg_fill_bull_rep_id",
    "fvg_fill_bull_rep_detect_idx",
    "fvg_fill_bull_rep_touch_count",
    "fvg_fill_bull_rep_fill_pct",
    "fvg_fill_bull_rep_state",
    "fvg_fill_bull_rep_matches_structural_selected",
    "fvg_fill_bull_rep_selection_mode",
    "fvg_fill_bear_remaining_low",
    "fvg_fill_bear_remaining_high",
    "fvg_fill_bear_remaining_mid",
    "fvg_fill_bear_remaining_width",
    "fvg_fill_bear_remaining_width_atr",
    "fvg_fill_bear_remaining_frac",
    "fvg_fill_bear_fill_pct_delta_1",
    "fvg_fill_bear_fill_pct_delta_3",
    "fvg_fill_bear_fill_pct_delta_5",
    "fvg_fill_bear_fill_type",
    "fvg_fill_bear_bounce_strength_atr",
    "fvg_fill_bear_bars_since_touch",
    "fvg_fill_bear_bars_since_first_touch",
    "fvg_fill_bear_touch_rejection_flag",
    "fvg_fill_bear_dist_to_remaining_mid_atr",
    "fvg_fill_bear_dist_to_remaining_near_atr",
    "fvg_fill_bear_dist_to_remaining_far_atr",
    "fvg_fill_bear_mean_fill_pct",
    "fvg_fill_bear_max_fill_pct",
    "fvg_fill_bear_pool_mean_fill_pct",
    "fvg_fill_bear_pool_max_fill_pct",
    "fvg_fill_bear_untouched_count",
    "fvg_fill_bear_rep_id",
    "fvg_fill_bear_rep_detect_idx",
    "fvg_fill_bear_rep_touch_count",
    "fvg_fill_bear_rep_fill_pct",
    "fvg_fill_bear_rep_state",
    "fvg_fill_bear_rep_matches_structural_selected",
    "fvg_fill_bear_rep_selection_mode",
]


@dataclass
class _ActiveFillRep:
    active_id: int
    side: str
    detect_idx: int
    age_anchor_idx: int
    low: float
    high: float
    base_quality: float
    source_ids: list[int] = field(default_factory=list)
    source_detect_indices: list[int] = field(default_factory=list)
    active_since_idx: int = 0
    state: int = STATE_ACTIVE_UNTOUCHED
    touch_count: int = 0
    max_fill_pct: float = 0.0
    first_touch_idx: int | None = None
    last_touch_idx: int | None = None
    first_partial_fill_idx: int | None = None
    max_single_bar_fill_jump: float = 0.0
    all_touch_bars_closed_outside: bool = True
    last_touch_rejection_flag: bool | None = None
    bounce_raw_max: float = 0.0
    max_fill_type_seen: int = FILL_TYPE_NONE

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def min_source_event_id(self) -> int:
        return min(self.source_ids) if self.source_ids else self.active_id


@dataclass
class _SelectedFillState:
    active_id: int
    side: str
    low: float
    high: float
    first_touch_idx: int | None = None
    last_touch_idx: int | None = None
    first_partial_fill_idx: int | None = None
    max_fill_pct: float = 0.0
    touch_count: int = 0
    max_single_bar_fill_jump: float = 0.0
    all_touch_bars_closed_outside: bool = True
    last_touch_rejection_flag: bool | None = None
    bounce_raw_max: float = 0.0
    max_fill_type_seen: int = FILL_TYPE_NONE

    @property
    def width(self) -> float:
        return self.high - self.low


def _overlaps_or_touches(
    low_1: float, high_1: float, low_2: float, high_2: float
) -> bool:
    return max(low_1, low_2) <= min(high_1, high_2)


def _distance_to_near_edge(close_value: float, low: float, high: float) -> float:
    if close_value < low:
        return low - close_value
    if close_value > high:
        return close_value - high
    return min(close_value - low, high - close_value)


def _distance_to_far_edge(close_value: float, low: float, high: float) -> float:
    if close_value < low:
        return high - close_value
    if close_value > high:
        return close_value - low
    return max(close_value - low, high - close_value)


def _close_outside_zone(side: str, close_value: float, low: float, high: float) -> bool:
    if side == "bull":
        return close_value >= high
    return close_value <= low


def _current_fill_type(rep: _ActiveFillRep, current_idx: int) -> int:
    if rep.touch_count == 0:
        return FILL_TYPE_NONE

    if rep.last_touch_idx is None or rep.first_touch_idx is None:
        return FILL_TYPE_NONE

    bars_since_first_touch = current_idx - rep.first_touch_idx
    mean_fill_velocity = rep.max_fill_pct / max(1, bars_since_first_touch)
    bars_to_full_fill = (
        current_idx - rep.first_touch_idx if rep.max_fill_pct >= 1.0 else np.nan
    )

    if rep.max_single_bar_fill_jump > 0.5 or (
        np.isfinite(bars_to_full_fill) and bars_to_full_fill <= 2
    ):
        return FILL_TYPE_AGGRESSIVE
    if rep.max_fill_pct > 0.0 and mean_fill_velocity < 0.2:
        return FILL_TYPE_ORDERLY
    if rep.all_touch_bars_closed_outside and rep.max_fill_pct <= 0.0:
        return FILL_TYPE_WICK_ONLY
    return FILL_TYPE_ORDERLY


def _new_rep(
    active_id: int,
    side: str,
    detect_idx: int,
    event_id: int,
    low: float,
    high: float,
    base_quality: float,
) -> _ActiveFillRep:
    return _ActiveFillRep(
        active_id=active_id,
        side=side,
        detect_idx=detect_idx,
        age_anchor_idx=detect_idx,
        low=float(low),
        high=float(high),
        base_quality=float(base_quality),
        source_ids=[int(event_id)],
        source_detect_indices=[int(detect_idx)],
        active_since_idx=detect_idx,
    )


def _recompute_base_quality(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    quality_map: dict[str, dict[int, float]] = {"bull": {}, "bear": {}}
    for side in ("bull", "bear"):
        detect = pd.to_numeric(df[f"fvg_{side}_detect_flag"], errors="coerce").fillna(0)
        scoped = df[detect == 1].copy()
        if scoped.empty:
            continue
        required = {
            f"fvg_{side}_width_atr",
            f"fvg_{side}_mid_body_atr",
            f"fvg_{side}_mid_body_frac",
            f"fvg_{side}_gap_cleanliness",
            f"fvg_{side}_mid_close_location",
        }
        if not required.issubset(scoped.columns):
            quality_map[side] = {
                int(event_id): 0.0
                for event_id in pd.to_numeric(
                    scoped[f"fvg_{side}_event_id"], errors="coerce"
                )
                .fillna(0)
                .astype(int)
                .tolist()
                if int(event_id) > 0
            }
            continue
        width_atr = pd.to_numeric(scoped[f"fvg_{side}_width_atr"], errors="coerce")
        mid_body_atr = pd.to_numeric(
            scoped[f"fvg_{side}_mid_body_atr"], errors="coerce"
        )
        mid_body_frac = pd.to_numeric(
            scoped[f"fvg_{side}_mid_body_frac"], errors="coerce"
        )
        gap_cleanliness = pd.to_numeric(
            scoped[f"fvg_{side}_gap_cleanliness"], errors="coerce"
        )
        mid_close_location = pd.to_numeric(
            scoped[f"fvg_{side}_mid_close_location"], errors="coerce"
        )
        directional_close = (
            mid_close_location if side == "bull" else (1.0 - mid_close_location)
        )
        score = np.clip(
            0.25 * np.clip(width_atr / 2.0, 0.0, 1.0)
            + 0.20 * np.clip(mid_body_atr / 2.0, 0.0, 1.0)
            + 0.15 * np.clip(mid_body_frac, 0.0, 1.0)
            + 0.20 * np.clip(gap_cleanliness / 2.0, 0.0, 1.0)
            + 0.20 * np.clip(directional_close, 0.0, 1.0),
            0.0,
            1.0,
        )
        event_ids = pd.to_numeric(
            scoped[f"fvg_{side}_event_id"], errors="coerce"
        ).fillna(0)
        quality_map[side] = {
            int(event_id): float(base_quality)
            for event_id, base_quality in zip(
                event_ids.to_list(), score.to_list(), strict=False
            )
            if int(event_id) > 0
        }
    return quality_map


def _interaction_modifier(rep: _ActiveFillRep) -> float:
    if rep.max_fill_pct >= 0.50:
        return PARTIAL_FILL_GTE_50_MODIFIER
    if rep.max_fill_pct >= 0.25:
        return PARTIAL_FILL_LT_50_MODIFIER
    if rep.max_fill_pct > 0.0:
        return PARTIAL_FILL_LT_25_MODIFIER
    if rep.touch_count > 0:
        return TOUCHED_ONLY_MODIFIER
    return 1.0


def _age_decay(age_bars: int) -> float:
    return float(2.0 ** (-max(age_bars, 0) / HALF_LIFE_BARS))


def _rep_effective_significance(
    rep: _ActiveFillRep,
    *,
    current_idx: int,
    close_value: float,
    atr_value: float,
) -> float:
    if not np.isfinite(atr_value) or atr_value <= 0:
        return 0.0
    distance_atr = _distance_to_near_edge(close_value, rep.low, rep.high) / atr_value
    distance_modifier = float(1.0 / (1.0 + max(distance_atr, 0.0)))
    age_bars = current_idx - rep.age_anchor_idx
    return float(
        np.clip(
            rep.base_quality
            * _age_decay(age_bars)
            * _interaction_modifier(rep)
            * distance_modifier,
            0.0,
            1.0,
        )
    )


def _is_true_untouched(rep: _ActiveFillRep) -> bool:
    return rep.touch_count == 0 and np.isclose(rep.max_fill_pct, 0.0)


def _is_touched_candidate(rep: _ActiveFillRep) -> bool:
    return rep.touch_count > 0 or rep.max_fill_pct > 0.0


def _fill_severity_rank(rep: _ActiveFillRep) -> int:
    if rep.max_fill_pct > 0.0:
        return 2
    if rep.touch_count > 0:
        return 1
    return 0


def _pick_fill_active_rep(
    *,
    current_idx: int,
    close_value: float,
    atr_value: float,
    structural_selected_id: int,
    representations: list[_ActiveFillRep],
) -> tuple[_ActiveFillRep | None, str]:
    if not representations:
        return None, "none"

    touched_candidates = [rep for rep in representations if _is_touched_candidate(rep)]
    if not touched_candidates:
        structural = next(
            (rep for rep in representations if rep.active_id == structural_selected_id),
            None,
        )
        if structural is not None:
            return structural, "structural_fallback"
        return (
            max(
                representations,
                key=lambda rep: (
                    rep.detect_idx,
                    rep.base_quality,
                    -rep.min_source_event_id,
                ),
            ),
            "structural_fallback",
        )

    chosen = max(
        touched_candidates,
        key=lambda rep: (
            rep.last_touch_idx if rep.last_touch_idx is not None else -1,
            _fill_severity_rank(rep),
            rep.max_fill_pct,
            -_distance_to_near_edge(close_value, rep.low, rep.high),
            rep.detect_idx,
            _rep_effective_significance(
                rep,
                current_idx=current_idx,
                close_value=close_value,
                atr_value=atr_value,
            ),
            -rep.active_id,
        ),
    )
    return chosen, "touched_pool"


def _current_selected_fill_type(state: _SelectedFillState, current_idx: int) -> int:
    if state.touch_count == 0 or state.first_touch_idx is None:
        return FILL_TYPE_NONE

    bars_since_first_touch = current_idx - state.first_touch_idx
    mean_fill_velocity = state.max_fill_pct / max(1, bars_since_first_touch)
    bars_to_full_fill = (
        current_idx - state.first_touch_idx if state.max_fill_pct >= 1.0 else np.nan
    )

    if state.max_single_bar_fill_jump > 0.5 or (
        np.isfinite(bars_to_full_fill) and bars_to_full_fill <= 2
    ):
        return FILL_TYPE_AGGRESSIVE
    if state.max_fill_pct > 0.0 and mean_fill_velocity < 0.2:
        return FILL_TYPE_ORDERLY
    if state.all_touch_bars_closed_outside and state.max_fill_pct <= 0.0:
        return FILL_TYPE_WICK_ONLY
    return FILL_TYPE_ORDERLY


def _update_rep_on_row(
    rep: _ActiveFillRep,
    *,
    i: int,
    low_value: float,
    high_value: float,
    close_value: float,
) -> tuple[str | None, int]:
    if rep.side == "bull":
        invalidated = close_value < rep.low
        full_fill = low_value <= rep.low
        touch = low_value <= rep.high
        penetration = max(0.0, rep.high - low_value)
        favorable_move = max(0.0, high_value - rep.high)
    else:
        invalidated = close_value > rep.high
        full_fill = high_value >= rep.high
        touch = high_value >= rep.low
        penetration = max(0.0, high_value - rep.low)
        favorable_move = max(0.0, rep.low - low_value)

    fill_pct = float(np.clip(penetration / max(rep.width, EPS), 0.0, 1.0))
    if fill_pct > rep.max_fill_pct:
        rep.max_single_bar_fill_jump = max(
            rep.max_single_bar_fill_jump, fill_pct - rep.max_fill_pct
        )
        rep.max_fill_pct = fill_pct

    if touch:
        if rep.touch_count == 0:
            rep.first_touch_idx = i
        rep.touch_count += 1
        rep.last_touch_idx = i
        if rep.max_fill_pct > 0.0 and rep.first_partial_fill_idx is None:
            rep.first_partial_fill_idx = i
        close_outside = _close_outside_zone(rep.side, close_value, rep.low, rep.high)
        rep.all_touch_bars_closed_outside = (
            rep.all_touch_bars_closed_outside and close_outside
        )
        rep.last_touch_rejection_flag = close_outside
        rep.bounce_raw_max = favorable_move
    elif rep.last_touch_idx is not None:
        rep.bounce_raw_max = max(rep.bounce_raw_max, favorable_move)

    if invalidated:
        terminal_type = _current_fill_type(rep, i)
        rep.max_fill_type_seen = max(rep.max_fill_type_seen, terminal_type)
        return "invalidated", rep.max_fill_type_seen

    if full_fill:
        terminal_type = _current_fill_type(rep, i)
        rep.max_fill_type_seen = max(rep.max_fill_type_seen, terminal_type)
        return "full_fill", rep.max_fill_type_seen

    if 0.0 < rep.max_fill_pct < 1.0:
        rep.state = STATE_ACTIVE_PARTIAL_FILL
    elif touch:
        rep.state = STATE_ACTIVE_TOUCHED
    else:
        rep.state = STATE_ACTIVE_UNTOUCHED

    rep.max_fill_type_seen = max(rep.max_fill_type_seen, _current_fill_type(rep, i))
    return None, rep.max_fill_type_seen


def _update_selected_state(
    state: _SelectedFillState,
    *,
    current_idx: int,
    low_value: float,
    high_value: float,
    close_value: float,
    fill_pct: float,
    touch_count: int,
    row_touch_override: bool | None = None,
) -> None:
    if fill_pct > state.max_fill_pct:
        state.max_single_bar_fill_jump = max(
            state.max_single_bar_fill_jump, fill_pct - state.max_fill_pct
        )
        state.max_fill_pct = fill_pct

    if state.side == "bull":
        row_touched = low_value <= state.high
        favorable_move = max(0.0, high_value - state.high)
    else:
        row_touched = high_value >= state.low
        favorable_move = max(0.0, state.low - low_value)

    if row_touch_override is not None:
        row_touched = row_touch_override

    if row_touched and touch_count > 0:
        if state.first_touch_idx is None:
            state.first_touch_idx = current_idx
        state.last_touch_idx = current_idx
        if fill_pct > 0.0 and state.first_partial_fill_idx is None:
            state.first_partial_fill_idx = current_idx
        close_outside = _close_outside_zone(
            state.side, close_value, state.low, state.high
        )
        state.all_touch_bars_closed_outside = (
            state.all_touch_bars_closed_outside and close_outside
        )
        state.last_touch_rejection_flag = close_outside
        state.bounce_raw_max = favorable_move
    elif state.last_touch_idx is not None:
        state.bounce_raw_max = max(state.bounce_raw_max, favorable_move)

    state.touch_count = max(state.touch_count, int(touch_count))
    state.max_fill_type_seen = max(
        state.max_fill_type_seen,
        _current_selected_fill_type(state, current_idx),
    )


def _reconstruct_active_reps(
    df: pd.DataFrame,
    *,
    atr_length: int,
) -> tuple[
    list[dict[int, _ActiveFillRep]], list[dict[int, int | None]], dict[str, np.ndarray]
]:
    n = len(df)
    base_quality_map = _recompute_base_quality(df)
    detect_flag = {
        "bull": df["fvg_bull_detect_flag"].to_numpy(dtype=np.int8),
        "bear": df["fvg_bear_detect_flag"].to_numpy(dtype=np.int8),
    }
    event_id = {
        "bull": df["fvg_bull_event_id"].to_numpy(dtype=np.int64),
        "bear": df["fvg_bear_event_id"].to_numpy(dtype=np.int64),
    }
    zone_low = {
        "bull": df["fvg_bull_low"].to_numpy(dtype=float),
        "bear": df["fvg_bear_low"].to_numpy(dtype=float),
    }
    zone_high = {
        "bull": df["fvg_bull_high"].to_numpy(dtype=float),
        "bear": df["fvg_bear_high"].to_numpy(dtype=float),
    }
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    active_reps: dict[str, list[_ActiveFillRep]] = {"bull": [], "bear": []}
    rep_snapshots: list[dict[int, _ActiveFillRep]] = []
    terminal_fill_type: dict[str, np.ndarray] = {
        "bull": np.zeros(n, dtype=np.int8),
        "bear": np.zeros(n, dtype=np.int8),
    }
    terminal_active_id: list[dict[str, int | None]] = []
    next_active_id = 1

    for i in range(n):
        row_terminal_active = {"bull": None, "bear": None}

        for side in ("bull", "bear"):
            survivors: list[_ActiveFillRep] = []
            for rep in active_reps[side]:
                terminal_reason, terminal_type = _update_rep_on_row(
                    rep,
                    i=i,
                    low_value=low[i],
                    high_value=high[i],
                    close_value=close[i],
                )
                if terminal_reason is not None:
                    row_terminal_active[side] = rep.active_id
                    terminal_fill_type[side][i] = terminal_type
                    continue
                survivors.append(rep)
            active_reps[side] = survivors

        new_events: list[tuple[str, _ActiveFillRep]] = []
        for side in ("bull", "bear"):
            if (
                detect_flag[side][i] == 1
                and np.isfinite(zone_low[side][i])
                and np.isfinite(zone_high[side][i])
            ):
                new_events.append(
                    (
                        side,
                        _new_rep(
                            next_active_id,
                            side,
                            i,
                            int(event_id[side][i]),
                            float(zone_low[side][i]),
                            float(zone_high[side][i]),
                            float(
                                base_quality_map[side].get(int(event_id[side][i]), 0.0)
                            ),
                        ),
                    )
                )
                next_active_id += 1

        for side, event_rep in new_events:
            side_reps = active_reps[side]
            overlapping = [
                rep
                for rep in side_reps
                if _overlaps_or_touches(
                    rep.low, rep.high, event_rep.low, event_rep.high
                )
            ]
            if not overlapping:
                side_reps.append(event_rep)
                continue

            merged_low = event_rep.low
            merged_high = event_rep.high
            merged_age_anchor = event_rep.age_anchor_idx
            merged_source_ids = list(event_rep.source_ids)
            merged_source_detects = list(event_rep.source_detect_indices)
            merged_base_quality = event_rep.base_quality
            for rep in overlapping:
                merged_low = min(merged_low, rep.low)
                merged_high = max(merged_high, rep.high)
                merged_age_anchor = min(merged_age_anchor, rep.age_anchor_idx)
                merged_source_ids.extend(rep.source_ids)
                merged_source_detects.extend(rep.source_detect_indices)
                merged_base_quality = max(merged_base_quality, rep.base_quality)

            # Merged reps start with no fill history. Constituent fill_pct values
            # are relative to each constituent's width and cannot be inherited by
            # the merged zone (different width = different denominator).
            active_reps[side] = [rep for rep in side_reps if rep not in overlapping]
            merged_rep = _ActiveFillRep(
                active_id=next_active_id,
                side=side,
                detect_idx=i,
                age_anchor_idx=merged_age_anchor,
                low=float(merged_low),
                high=float(merged_high),
                base_quality=float(merged_base_quality),
                source_ids=sorted(set(merged_source_ids)),
                source_detect_indices=sorted(set(merged_source_detects)),
                active_since_idx=i,
                state=STATE_ACTIVE_UNTOUCHED,
                touch_count=0,
                max_fill_pct=0.0,
                first_touch_idx=None,
                last_touch_idx=None,
                first_partial_fill_idx=None,
                max_single_bar_fill_jump=0.0,
                all_touch_bars_closed_outside=True,
                last_touch_rejection_flag=None,
                bounce_raw_max=0.0,
                max_fill_type_seen=FILL_TYPE_NONE,
            )
            active_reps[side].append(merged_rep)
            next_active_id += 1

        for side in ("bull", "bear"):
            survivors = []
            for rep in active_reps[side]:
                if (i - rep.age_anchor_idx) >= int(MAX_ACTIVE_AGE_BARS):
                    row_terminal_active[side] = rep.active_id
                    terminal_fill_type[side][i] = max(
                        terminal_fill_type[side][i],
                        rep.max_fill_type_seen,
                    )
                    continue
                survivors.append(rep)
            active_reps[side] = survivors

        rep_snapshots.append(
            {
                rep.active_id: _ActiveFillRep(
                    active_id=rep.active_id,
                    side=rep.side,
                    detect_idx=rep.detect_idx,
                    age_anchor_idx=rep.age_anchor_idx,
                    low=rep.low,
                    high=rep.high,
                    base_quality=rep.base_quality,
                    source_ids=list(rep.source_ids),
                    source_detect_indices=list(rep.source_detect_indices),
                    active_since_idx=rep.active_since_idx,
                    state=rep.state,
                    touch_count=rep.touch_count,
                    max_fill_pct=rep.max_fill_pct,
                    first_touch_idx=rep.first_touch_idx,
                    last_touch_idx=rep.last_touch_idx,
                    first_partial_fill_idx=rep.first_partial_fill_idx,
                    max_single_bar_fill_jump=rep.max_single_bar_fill_jump,
                    all_touch_bars_closed_outside=rep.all_touch_bars_closed_outside,
                    last_touch_rejection_flag=rep.last_touch_rejection_flag,
                    bounce_raw_max=rep.bounce_raw_max,
                    max_fill_type_seen=rep.max_fill_type_seen,
                )
                for reps in active_reps.values()
                for rep in reps
            }
        )
        terminal_active_id.append(row_terminal_active)

    return rep_snapshots, terminal_active_id, terminal_fill_type


def _init_output(n: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    float_cols = [
        col
        for col in FVG_FILL_COLUMNS
        if not col.endswith(
            ("fill_type", "untouched_count", "rep_matches_structural_selected")
        )
        and not col.endswith("_selection_mode")
    ]
    int_cols = [
        "fvg_fill_bull_fill_type",
        "fvg_fill_bear_fill_type",
        "fvg_fill_bull_untouched_count",
        "fvg_fill_bear_untouched_count",
        "fvg_fill_bull_rep_matches_structural_selected",
        "fvg_fill_bear_rep_matches_structural_selected",
        "fvg_active_bull",
        "fvg_active_bear",
    ]
    object_cols = [
        "fvg_fill_bull_rep_selection_mode",
        "fvg_fill_bear_rep_selection_mode",
    ]
    for col in float_cols:
        arrays[col] = np.full(n, np.nan, dtype=float)
    for col in int_cols:
        arrays[col] = np.zeros(n, dtype=np.int32)
    for col in object_cols:
        arrays[col] = np.full(n, "none", dtype=object)
    arrays["fvg_fill_pct"] = np.full(n, np.nan, dtype=float)
    arrays["fvg_age"] = np.full(n, np.nan, dtype=float)
    return arrays


CORE_MEMBER_REQUIRED_COLUMNS = {
    "row_idx",
    "side",
    "active_id",
    "detect_idx",
    "age_anchor_idx",
    "active_since_idx",
    "low",
    "high",
    "base_quality",
    "touch_count",
    "fill_pct",
    "state",
    "selected",
    "first_touch_idx",
    "last_touch_idx",
    "first_partial_fill_idx",
    "max_single_bar_fill_jump",
    "all_touch_bars_closed_outside",
    "last_touch_rejection_flag",
    "bounce_raw_max",
    "component_id",
    "component_owned_this_row",
    "touch_delta",
    "fill_delta",
    "entered_this_row",
}


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(int(value))


def _build_member_snapshots_from_core(
    debug_tables: dict[str, pd.DataFrame],
    *,
    n_rows: int,
) -> list[dict[int, _ActiveFillRep]]:
    active_members = debug_tables.get("active_member_table")
    if active_members is None:
        raise ValueError("debug_tables must contain 'active_member_table'")
    if active_members.empty:
        return [{} for _ in range(n_rows)]
    require_columns(
        active_members,
        CORE_MEMBER_REQUIRED_COLUMNS,
        caller="add_fvg_fill",
    )

    grouped: dict[int, pd.DataFrame] = {
        int(row_idx): scoped.copy()
        for row_idx, scoped in active_members.groupby("row_idx", sort=False)
    }
    snapshots: list[dict[int, _ActiveFillRep]] = []
    for row_idx in range(n_rows):
        scoped = grouped.get(row_idx)
        if scoped is None:
            snapshots.append({})
            continue
        snapshot: dict[int, _ActiveFillRep] = {}
        for row in scoped.itertuples():
            active_id = int(row.active_id)
            rep = _ActiveFillRep(
                active_id=active_id,
                side=str(row.side),
                detect_idx=int(row.detect_idx),
                age_anchor_idx=int(row.age_anchor_idx),
                low=float(row.low),
                high=float(row.high),
                base_quality=float(row.base_quality),
                source_ids=[active_id],
                source_detect_indices=[int(row.detect_idx)],
                active_since_idx=int(row.active_since_idx),
                state=int(row.state),
                touch_count=int(row.touch_count),
                max_fill_pct=float(row.fill_pct),
                first_touch_idx=_optional_int(row.first_touch_idx),
                last_touch_idx=_optional_int(row.last_touch_idx),
                first_partial_fill_idx=_optional_int(row.first_partial_fill_idx),
                max_single_bar_fill_jump=float(row.max_single_bar_fill_jump),
                all_touch_bars_closed_outside=bool(
                    int(row.all_touch_bars_closed_outside)
                ),
                last_touch_rejection_flag=_optional_bool(row.last_touch_rejection_flag),
                bounce_raw_max=float(row.bounce_raw_max),
            )
            snapshot[active_id] = rep
        snapshots.append(snapshot)
    return snapshots


def add_fvg_fill(
    df: pd.DataFrame,
    *,
    debug_tables: dict[str, pd.DataFrame],
    atr_length: int = 14,
) -> pd.DataFrame:
    """
    Add FVG Fill consumer features on top of canonical FVG Core outputs.

    The implementation is causal and pure. Selected-zone features follow the
    fill-active representation per side, while FVG Core structural active
    outputs remain unchanged. Multi-zone summaries are reconstructed causally
    across the full active pool.
    """
    out = df.copy()
    require_columns(
        out,
        SELECTED_ACTIVE_COLUMNS,
        caller="add_fvg_fill",
    )

    n = len(out)
    if n == 0:
        return out

    atr = get_atr_array(out, atr_length)
    active_flag = {
        "bull": out["fvg_bull_active"].to_numpy(dtype=np.int8),
        "bear": out["fvg_bear_active"].to_numpy(dtype=np.int8),
    }
    active_id = {
        "bull": out["fvg_bull_active_id"].to_numpy(dtype=np.int64),
        "bear": out["fvg_bear_active_id"].to_numpy(dtype=np.int64),
    }
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)

    rep_snapshots = _build_member_snapshots_from_core(debug_tables, n_rows=n)

    arrays = _init_output(n)
    fill_rep_id_history = {
        "bull": np.zeros(n, dtype=np.int64),
        "bear": np.zeros(n, dtype=np.int64),
    }
    fill_rep_fill_history = {
        "bull": np.full(n, np.nan, dtype=float),
        "bear": np.full(n, np.nan, dtype=float),
    }
    previous_fill_rep: dict[str, _ActiveFillRep | None] = {
        "bull": None,
        "bear": None,
    }

    for i in range(n):
        snapshot = rep_snapshots[i]
        for side in ("bull", "bear"):
            rep_list = [rep for rep in snapshot.values() if rep.side == side]
            structural_selected_id = (
                int(active_id[side][i]) if active_flag[side][i] == 1 else 0
            )
            fill_rep, selection_mode = _pick_fill_active_rep(
                current_idx=i,
                close_value=float(close[i]),
                atr_value=float(atr[i]),
                structural_selected_id=structural_selected_id,
                representations=rep_list,
            )
            fill_rep_id = int(fill_rep.active_id) if fill_rep is not None else 0
            fill_rep_id_history[side][i] = fill_rep_id
            arrays[f"fvg_fill_{side}_rep_selection_mode"][i] = selection_mode
            arrays[f"fvg_fill_{side}_rep_id"][i] = (
                float(fill_rep_id) if fill_rep_id else 0.0
            )
            arrays[f"fvg_fill_{side}_rep_detect_idx"][i] = (
                float(fill_rep.detect_idx) if fill_rep is not None else np.nan
            )
            arrays[f"fvg_fill_{side}_rep_touch_count"][i] = (
                float(fill_rep.touch_count) if fill_rep is not None else np.nan
            )
            arrays[f"fvg_fill_{side}_rep_fill_pct"][i] = (
                float(fill_rep.max_fill_pct) if fill_rep is not None else np.nan
            )
            arrays[f"fvg_fill_{side}_rep_state"][i] = (
                float(fill_rep.state) if fill_rep is not None else np.nan
            )
            matches_structural = False
            if fill_rep is not None and structural_selected_id != 0:
                matches_structural = fill_rep_id == structural_selected_id
            arrays[f"fvg_fill_{side}_rep_matches_structural_selected"][i] = int(
                matches_structural
            )
            # Pool-wide metrics across ALL active reps (including price-disjoint zones).
            arrays[f"fvg_fill_{side}_pool_mean_fill_pct"][i] = (
                float(np.mean([rep.max_fill_pct for rep in rep_list]))
                if rep_list
                else np.nan
            )
            arrays[f"fvg_fill_{side}_pool_max_fill_pct"][i] = (
                float(np.max([rep.max_fill_pct for rep in rep_list]))
                if rep_list
                else np.nan
            )
            # Touched-only metrics: restrict to zones that have actually been
            # interacted with. This prevents price-disjoint untouched zones from
            # diluting the signal and making unrelated zones appear to fill together.
            _touched_reps = [rep for rep in rep_list if not _is_true_untouched(rep)]
            arrays[f"fvg_fill_{side}_mean_fill_pct"][i] = (
                float(np.mean([rep.max_fill_pct for rep in _touched_reps]))
                if _touched_reps
                else np.nan
            )
            arrays[f"fvg_fill_{side}_max_fill_pct"][i] = (
                float(np.max([rep.max_fill_pct for rep in _touched_reps]))
                if _touched_reps
                else np.nan
            )
            arrays[f"fvg_fill_{side}_untouched_count"][i] = int(
                sum(_is_true_untouched(rep) for rep in rep_list)
            )

            arrays[f"fvg_active_{side}"][i] = int(active_flag[side][i] == 1)

            if fill_rep_id == 0:
                if previous_fill_rep[side] is not None:
                    terminal_rep = _ActiveFillRep(
                        active_id=previous_fill_rep[side].active_id,
                        side=previous_fill_rep[side].side,
                        detect_idx=previous_fill_rep[side].detect_idx,
                        age_anchor_idx=previous_fill_rep[side].age_anchor_idx,
                        low=previous_fill_rep[side].low,
                        high=previous_fill_rep[side].high,
                        base_quality=previous_fill_rep[side].base_quality,
                        source_ids=list(previous_fill_rep[side].source_ids),
                        source_detect_indices=list(
                            previous_fill_rep[side].source_detect_indices
                        ),
                        active_since_idx=previous_fill_rep[side].active_since_idx,
                        state=previous_fill_rep[side].state,
                        touch_count=previous_fill_rep[side].touch_count,
                        max_fill_pct=previous_fill_rep[side].max_fill_pct,
                        first_touch_idx=previous_fill_rep[side].first_touch_idx,
                        last_touch_idx=previous_fill_rep[side].last_touch_idx,
                        first_partial_fill_idx=previous_fill_rep[
                            side
                        ].first_partial_fill_idx,
                        max_single_bar_fill_jump=previous_fill_rep[
                            side
                        ].max_single_bar_fill_jump,
                        all_touch_bars_closed_outside=previous_fill_rep[
                            side
                        ].all_touch_bars_closed_outside,
                        last_touch_rejection_flag=previous_fill_rep[
                            side
                        ].last_touch_rejection_flag,
                        bounce_raw_max=previous_fill_rep[side].bounce_raw_max,
                    )
                    terminal_reason, terminal_type = _update_rep_on_row(
                        terminal_rep,
                        i=i,
                        low_value=low[i],
                        high_value=high[i],
                        close_value=close[i],
                    )
                    arrays[f"fvg_fill_{side}_fill_type"][i] = int(
                        terminal_type
                        if terminal_reason is not None
                        or terminal_rep.touch_count
                        > previous_fill_rep[side].touch_count
                        or terminal_rep.max_fill_pct
                        > previous_fill_rep[side].max_fill_pct
                        else _current_fill_type(terminal_rep, i)
                    )
                previous_fill_rep[side] = None
                continue

            zone_low = float(fill_rep.low)
            zone_high = float(fill_rep.high)
            zone_width = zone_high - zone_low
            fill_pct = float(fill_rep.max_fill_pct)
            fill_rep_fill_history[side][i] = fill_pct
            previous_fill_rep[side] = fill_rep

            remaining_width = max(zone_width * (1.0 - fill_pct), 0.0)
            if remaining_width > 0.0 and np.isfinite(atr[i]) and atr[i] > 0:
                if side == "bull":
                    remaining_low = zone_low
                    remaining_high = zone_high - (fill_pct * zone_width)
                else:
                    remaining_low = zone_low + (fill_pct * zone_width)
                    remaining_high = zone_high
                remaining_mid = (remaining_low + remaining_high) / 2.0
                arrays[f"fvg_fill_{side}_remaining_low"][i] = remaining_low
                arrays[f"fvg_fill_{side}_remaining_high"][i] = remaining_high
                arrays[f"fvg_fill_{side}_remaining_mid"][i] = remaining_mid
                arrays[f"fvg_fill_{side}_remaining_width"][i] = remaining_width
                arrays[f"fvg_fill_{side}_remaining_width_atr"][i] = (
                    remaining_width / atr[i]
                )
                arrays[f"fvg_fill_{side}_remaining_frac"][i] = remaining_width / max(
                    zone_width, EPS
                )
                arrays[f"fvg_fill_{side}_dist_to_remaining_mid_atr"][i] = (
                    abs(close[i] - remaining_mid) / atr[i]
                )
                arrays[f"fvg_fill_{side}_dist_to_remaining_near_atr"][i] = (
                    _distance_to_near_edge(close[i], remaining_low, remaining_high)
                    / atr[i]
                )
                arrays[f"fvg_fill_{side}_dist_to_remaining_far_atr"][i] = (
                    _distance_to_far_edge(close[i], remaining_low, remaining_high)
                    / atr[i]
                )

            arrays[f"fvg_fill_{side}_fill_type"][i] = int(
                _current_fill_type(fill_rep, i)
            )
            arrays[f"fvg_fill_{side}_bounce_strength_atr"][i] = (
                fill_rep.bounce_raw_max / atr[i]
                if fill_rep.last_touch_idx is not None
                and np.isfinite(atr[i])
                and atr[i] > 0
                else np.nan
            )
            arrays[f"fvg_fill_{side}_bars_since_touch"][i] = (
                float(i - fill_rep.last_touch_idx)
                if fill_rep.last_touch_idx is not None
                else np.nan
            )
            arrays[f"fvg_fill_{side}_bars_since_first_touch"][i] = (
                float(i - fill_rep.first_touch_idx)
                if fill_rep.first_touch_idx is not None
                else np.nan
            )
            arrays[f"fvg_fill_{side}_touch_rejection_flag"][i] = (
                float(int(fill_rep.last_touch_rejection_flag))
                if fill_rep.last_touch_rejection_flag is not None
                else np.nan
            )

    for side in ("bull", "bear"):
        ids = fill_rep_id_history[side]
        fills = fill_rep_fill_history[side]
        for lookback in FILL_VELOCITY_LOOKBACKS:
            col = f"fvg_fill_{side}_fill_pct_delta_{lookback}"
            for i in range(n):
                current_id = ids[i]
                if current_id == 0 or i < lookback:
                    continue
                if np.any(ids[i - lookback : i + 1] != current_id):
                    continue
                if not np.isfinite(fills[i]) or not np.isfinite(fills[i - lookback]):
                    continue
                delta = fills[i] - fills[i - lookback]
                arrays[col][i] = max(delta, 0.0)

    bull_selected = np.where(
        fill_rep_id_history["bull"] > 0, fill_rep_fill_history["bull"], np.nan
    )
    bear_selected = np.where(
        fill_rep_id_history["bear"] > 0, fill_rep_fill_history["bear"], np.nan
    )
    arrays["fvg_fill_pct"] = np.where(
        np.isnan(bull_selected),
        bear_selected,
        np.where(
            np.isnan(bear_selected),
            bull_selected,
            np.maximum(bull_selected, bear_selected),
        ),
    )

    bull_age_alias = np.where(
        np.isfinite(arrays["fvg_fill_bull_bars_since_first_touch"]),
        arrays["fvg_fill_bull_bars_since_first_touch"],
        np.where(
            fill_rep_id_history["bull"] > 0,
            arrays["fvg_fill_bull_rep_detect_idx"] * 0.0
            + np.arange(n)
            - arrays["fvg_fill_bull_rep_detect_idx"],
            np.nan,
        ),
    )
    bear_age_alias = np.where(
        np.isfinite(arrays["fvg_fill_bear_bars_since_first_touch"]),
        arrays["fvg_fill_bear_bars_since_first_touch"],
        np.where(
            fill_rep_id_history["bear"] > 0,
            arrays["fvg_fill_bear_rep_detect_idx"] * 0.0
            + np.arange(n)
            - arrays["fvg_fill_bear_rep_detect_idx"],
            np.nan,
        ),
    )
    arrays["fvg_age"] = np.where(
        np.isnan(bull_age_alias),
        bear_age_alias,
        np.where(
            np.isnan(bear_age_alias),
            bull_age_alias,
            np.maximum(bull_age_alias, bear_age_alias),
        ),
    )

    fill_df = pd.DataFrame(arrays, index=out.index)
    for col in FVG_FILL_COLUMNS + [
        "fvg_fill_pct",
        "fvg_age",
        "fvg_active_bull",
        "fvg_active_bear",
    ]:
        out[col] = fill_df[col]
    return out
