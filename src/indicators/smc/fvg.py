"""
Canonical causal FVG core detector.

Doctrine
--------
- Causality: no live-safe column is populated before the gap is knowable at the
  close of the right bar.
- Separation: emits four families only: annotation, detect-event, dense
  active-state, and lifecycle.
- No future contamination: no forward-looking diagnostics or labels.
- Pure function: never mutates the input DataFrame.
- Zero-width tolerance: zone_high - zone_low must be strictly positive.
- Float stability: eps protects denominators only and must never rescue an
  invalid gap.

Formal pattern definition
-------------------------
Let ``t`` be the middle candle, ``left = t - 1``, ``right = t + 1`` and
``detect_idx = right``.

Body helpers:
- ``body_low[i] = min(open[i], close[i])``
- ``body_high[i] = max(open[i], close[i])``

Wick mode (canonical default)
- Bullish condition: ``low[right] > high[left]``
  - ``zone_low = high[left]``
  - ``zone_high = low[right]``
- Bearish condition: ``low[left] > high[right]``
  - ``zone_low = high[right]``
  - ``zone_high = low[left]``

Body mode
- Bullish condition: ``body_low[right] > body_high[left]``
  - ``zone_low = body_high[left]``
  - ``zone_high = body_low[right]``
- Bearish condition: ``body_low[left] > body_high[right]``
  - ``zone_low = body_high[right]``
  - ``zone_high = body_low[left]``

Hybrid mode
- Bullish condition: ``low[right] > body_high[left]``
  - ``zone_low = body_high[left]``
  - ``zone_high = low[right]``
- Bearish condition: ``body_low[left] > high[right]``
  - ``zone_low = high[right]``
  - ``zone_high = body_low[left]``

Validity for all modes
- ``zone_high - zone_low > 0``
- ``zone_width >= min_width_atr * ATR[detect_idx]``
- ``ATR[detect_idx]`` must be finite and strictly positive

Timing law
----------
- ``origin_idx = t``
- ``detect_idx = t + 1``
- ``origin_idx < detect_idx``
- ``active_start_idx = detect_idx``
- ``detect_delay = 1`` for vanilla three-candle FVGs
- Detect-time usability begins at the close of ``detect_idx``
- Same-bar detect + touch is prohibited: lifecycle evaluation starts from the
  NEXT bar only, therefore ``first_touch_idx > detect_idx`` whenever it exists
  and ``fill_pct[detect_idx] = 0`` by definition.

Lifecycle state machine
-----------------------
Dense active state enum:
- ``0 = none``
- ``1 = active_untouched``
- ``2 = active_touched``
- ``3 = active_partial_fill``
- ``4 = fully_filled`` (terminal)
- ``5 = invalidated`` (terminal)
- ``6 = expired`` (terminal)
- ``7 = merged`` (terminal)

Allowed transitions:
- ``none -> active_untouched`` on detect row
- ``active_untouched -> active_touched``
- ``active_untouched -> active_partial_fill``
- ``active_touched -> active_partial_fill``
- ``active_touched -> fully_filled``
- ``active_partial_fill -> fully_filled``
- ``any active_* -> invalidated / expired / merged``

Terminal precedence on the same bar:
- invalidated
- fully_filled
- merged
- expired
- active_partial_fill
- active_touched
- active_untouched

Fill and invalidation semantics
-------------------------------
Wick-based lifecycle interaction for a zone ``[L, H]``:

Bullish:
- touch: ``low[i] <= H``
- partial fill penetration: ``max(0, H - low[i])``
- full fill: ``low[i] <= L``
- invalidation: ``close[i] < L``

Bearish:
- touch: ``high[i] >= L``
- partial fill penetration: ``max(0, high[i] - L)``
- full fill: ``high[i] >= H``
- invalidation: ``close[i] > H``

``fill_pct = clip(penetration / zone_width, 0, 1)`` and is tracked as a
monotone maximum while the zone remains active.

Merge law
---------
- Raw events are always detected first.
- Same-side active representations merge iff their closed intervals overlap or
  touch: ``max(L1, L2) <= min(H1, H2)``.
- Merged geometry is the interval union.
- ``merged_detect_idx = max(component detect_idx)``
- Active merged representations use a namespace separate from raw event IDs.
- Raw source events receive ``terminal_state = merged`` on their detect rows and
  are never counted as active after the merge.

Selector law
------------
When multiple active representations exist on a side, export one dense current
representation using maximum effective significance:

``effective_significance = base_quality * age_decay * interaction_modifier
* distance_modifier``

Where:
- ``base_quality`` is fixed at detect/merge time
- ``age_decay = 2 ** (-age_bars / HALF_LIFE_BARS)``
- ``interaction_modifier`` penalizes touched / partially-filled zones
- ``distance_modifier = 1 / (1 + distance_to_zone_atr)``

Tie-breaks are:
- minimum distance to zone
- fresher detect index
- wider ``width_atr``
- lower raw source event ID

Distances use current close as the price basis and are ATR-normalized by the
current row ATR:
- ``distance_to_mid = abs(C - mid)``
- ``distance_to_near_edge`` =
  - ``L - C`` if ``C < L``
  - ``min(C - L, H - C)`` if inside zone
  - ``C - H`` if ``C > H``
- ``distance_to_far_edge`` =
  - ``H - C`` if ``C < L``
  - ``max(C - L, H - C)`` if inside zone
  - ``C - L`` if ``C > H``

Sparse lifecycle stamping law
-----------------------------
Lifecycle milestone columns live on the raw detect row only and are never
forward-filled:
- first touch
- first partial fill
- full fill
- invalidation
- expiry
- terminal state

Compatibility surface
---------------------
The detector also overwrites the legacy compatibility columns used elsewhere in
the repository:
- ``fvg_bull / fvg_bear`` become detect-time flags
- ``fvg_bull_low/high`` and ``fvg_bear_low/high`` mirror detect-time geometry
- ``fvg_size_atr`` mirrors detect-time width_atr
- ``fvg_bull_confirm_idx`` / ``fvg_bear_confirm_idx`` equal detect row index
- ``fvg_confirm_delay`` equals 1 on detect rows
- ``fvg_mid_body_atr`` mirrors detect-time middle-body ATR
- ``fvg_bos_bull`` / ``fvg_bos_bear`` are deterministic zero placeholders in
  core because BOS coupling is outside the canonical FVG core contract
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns

EPS = 1e-12
MAX_ACTIVE_AGE_BARS = 48
HALF_LIFE_BARS = 12.0

BASE_QUALITY_WIDTH_ATR_CAP = 2.0
BASE_QUALITY_MID_BODY_ATR_CAP = 2.0
BASE_QUALITY_GAP_CLEANLINESS_CAP = 2.0
MID_RANGE_FLOOR_ATR_FRACTION = 0.02
TOUCHED_ONLY_MODIFIER = 0.85
PARTIAL_FILL_LT_25_MODIFIER = 0.75
PARTIAL_FILL_LT_50_MODIFIER = 0.55
PARTIAL_FILL_GTE_50_MODIFIER = 0.30
LIVE_FILL_PERSISTENCE_BARS = 2

PATTERN_MODE_WICK = 0
PATTERN_MODE_BODY = 1
PATTERN_MODE_HYBRID = 2

STATE_NONE = 0
STATE_ACTIVE_UNTOUCHED = 1
STATE_ACTIVE_TOUCHED = 2
STATE_ACTIVE_PARTIAL_FILL = 3
STATE_FULLY_FILLED = 4
STATE_INVALIDATED = 5
STATE_EXPIRED = 6
STATE_MERGED = 7

ANNOTATION_COLUMNS = [
    "fvg_bull_origin_flag",
    "fvg_bull_origin_idx",
    "fvg_bull_origin_ts",
    "fvg_bull_left_idx",
    "fvg_bull_mid_idx",
    "fvg_bull_right_idx",
    "fvg_bear_origin_flag",
    "fvg_bear_origin_idx",
    "fvg_bear_origin_ts",
    "fvg_bear_left_idx",
    "fvg_bear_mid_idx",
    "fvg_bear_right_idx",
]

DETECT_COLUMNS = [
    "fvg_bull_detect_flag",
    "fvg_bull_detect_idx",
    "fvg_bull_detect_ts",
    "fvg_bull_event_id",
    "fvg_bull_detect_delay",
    "fvg_bear_detect_flag",
    "fvg_bear_detect_idx",
    "fvg_bear_detect_ts",
    "fvg_bear_event_id",
    "fvg_bear_detect_delay",
]

GEOMETRY_COLUMNS = [
    "fvg_bull_low",
    "fvg_bull_high",
    "fvg_bull_mid",
    "fvg_bull_width",
    "fvg_bull_width_atr",
    "fvg_bear_low",
    "fvg_bear_high",
    "fvg_bear_mid",
    "fvg_bear_width",
    "fvg_bear_width_atr",
]

PATTERN_QUALITY_COLUMNS = [
    "fvg_bull_mid_body",
    "fvg_bull_mid_range",
    "fvg_bull_mid_body_frac",
    "fvg_bull_mid_body_atr",
    "fvg_bull_mid_range_atr",
    "fvg_bull_gap_cleanliness",
    "fvg_bull_mid_close_location",
    "fvg_bull_left_overlap_frac",
    "fvg_bull_right_overlap_frac",
    "fvg_bull_signed_impulse_body_atr",
    "fvg_bull_pattern_mode",
    "fvg_bear_mid_body",
    "fvg_bear_mid_range",
    "fvg_bear_mid_body_frac",
    "fvg_bear_mid_body_atr",
    "fvg_bear_mid_range_atr",
    "fvg_bear_gap_cleanliness",
    "fvg_bear_mid_close_location",
    "fvg_bear_left_overlap_frac",
    "fvg_bear_right_overlap_frac",
    "fvg_bear_signed_impulse_body_atr",
    "fvg_bear_pattern_mode",
]

ACTIVE_COLUMNS = [
    "fvg_bull_active",
    "fvg_bull_active_id",
    "fvg_bull_active_low",
    "fvg_bull_active_high",
    "fvg_bull_active_mid",
    "fvg_bull_active_width",
    "fvg_bull_active_width_atr",
    "fvg_bull_active_age",
    "fvg_bull_active_age_bars",
    "fvg_bull_active_age_decay",
    "fvg_bull_active_base_quality",
    "fvg_bull_active_effective_significance",
    "fvg_bull_active_fill_pct",
    "fvg_bull_active_touch_count",
    "fvg_bull_active_state",
    "fvg_bull_active_since_idx",
    "fvg_bull_first_retest_flag",
    "fvg_bull_zone_invalidated",
    "fvg_bull_distance_to_mid_atr",
    "fvg_bull_distance_to_near_edge_atr",
    "fvg_bull_distance_to_far_edge_atr",
    "fvg_bull_active_count",
    "fvg_bull_active_weighted_count",
    "fvg_bear_active",
    "fvg_bear_active_id",
    "fvg_bear_active_low",
    "fvg_bear_active_high",
    "fvg_bear_active_mid",
    "fvg_bear_active_width",
    "fvg_bear_active_width_atr",
    "fvg_bear_active_age",
    "fvg_bear_active_age_bars",
    "fvg_bear_active_age_decay",
    "fvg_bear_active_base_quality",
    "fvg_bear_active_effective_significance",
    "fvg_bear_active_fill_pct",
    "fvg_bear_active_touch_count",
    "fvg_bear_active_state",
    "fvg_bear_active_since_idx",
    "fvg_bear_first_retest_flag",
    "fvg_bear_zone_invalidated",
    "fvg_bear_distance_to_mid_atr",
    "fvg_bear_distance_to_near_edge_atr",
    "fvg_bear_distance_to_far_edge_atr",
    "fvg_bear_active_count",
    "fvg_bear_active_weighted_count",
]

LIFECYCLE_COLUMNS = [
    "fvg_bull_first_touch_idx",
    "fvg_bull_first_partial_fill_idx",
    "fvg_bull_full_fill_idx",
    "fvg_bull_invalidation_idx",
    "fvg_bull_expiry_idx",
    "fvg_bull_terminal_state",
    "fvg_bear_first_touch_idx",
    "fvg_bear_first_partial_fill_idx",
    "fvg_bear_full_fill_idx",
    "fvg_bear_invalidation_idx",
    "fvg_bear_expiry_idx",
    "fvg_bear_terminal_state",
]

ALL_FVG_CORE_COLUMNS = (
    ANNOTATION_COLUMNS
    + DETECT_COLUMNS
    + GEOMETRY_COLUMNS
    + PATTERN_QUALITY_COLUMNS
    + ACTIVE_COLUMNS
    + LIFECYCLE_COLUMNS
)


@dataclass
class _RawEvent:
    event_id: int
    side: str
    origin_idx: int
    detect_idx: int
    detect_ts: object
    low: float
    high: float
    mid: float
    width: float
    width_atr: float
    pattern_mode: int
    base_quality: float
    first_touch_idx: float = np.nan
    first_partial_fill_idx: float = np.nan
    full_fill_idx: float = np.nan
    invalidation_idx: float = np.nan
    expiry_idx: float = np.nan
    merge_idx: float = np.nan
    terminal_state: int = STATE_NONE

    @property
    def lifecycle_prefix(self) -> str:
        return f"fvg_{self.side}"


@dataclass
class _ActiveRepresentation:
    active_id: int
    side: str
    detect_idx: int
    age_anchor_idx: int
    low: float
    high: float
    width_atr: float
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
    first_retest_emitted: bool = False
    historical_touch_count: int = 0
    historical_fill_pct: float = 0.0
    historical_first_touch_idx: int | None = None
    historical_last_touch_idx: int | None = None
    historical_first_partial_fill_idx: int | None = None
    historical_max_single_bar_fill_jump: float = 0.0
    historical_all_touch_bars_closed_outside: bool = True
    historical_last_touch_rejection_flag: bool | None = None
    historical_bounce_raw_max: float = 0.0

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def source_count(self) -> int:
        return len(self.source_ids)

    @property
    def min_source_event_id(self) -> int:
        return min(self.source_ids)


@dataclass
class _RepComponent:
    component_id: int
    side: str
    members: list[_ActiveRepresentation]
    low: float
    high: float

    @property
    def member_count(self) -> int:
        return len(self.members)


def _pattern_mode_code(pattern_mode: str) -> int:
    mapping = {
        "wick": PATTERN_MODE_WICK,
        "body": PATTERN_MODE_BODY,
        "hybrid": PATTERN_MODE_HYBRID,
    }
    if pattern_mode not in mapping:
        raise ValueError("pattern_mode must be one of {'wick', 'body', 'hybrid'}")
    return mapping[pattern_mode]


def _empty_ts_series(index: pd.Index, ts_dtype: object) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype=ts_dtype)


def _zone_distance(close_value: float, low: float, high: float) -> float:
    if close_value < low:
        return low - close_value
    if close_value > high:
        return close_value - high
    return 0.0


def _distance_to_near_edge(close_value: float, low: float, high: float) -> float:
    if close_value < low:
        return low - close_value
    if close_value > high:
        return close_value - high
    return min(close_value - low, high - close_value)


def _fill_state_from_metrics(fill_pct: float, touch_count: int) -> int:
    if 0.0 < fill_pct < 1.0:
        return STATE_ACTIVE_PARTIAL_FILL
    if touch_count > 0:
        return STATE_ACTIVE_TOUCHED
    return STATE_ACTIVE_UNTOUCHED


def _reset_live_fill_state(rep: _ActiveRepresentation) -> None:
    rep.state = STATE_ACTIVE_UNTOUCHED
    rep.touch_count = 0
    rep.max_fill_pct = 0.0
    rep.first_touch_idx = None
    rep.last_touch_idx = None
    rep.first_partial_fill_idx = None
    rep.max_single_bar_fill_jump = 0.0
    rep.all_touch_bars_closed_outside = True
    rep.last_touch_rejection_flag = None
    rep.bounce_raw_max = 0.0


def _penetrated_price_from_fill_pct(
    *,
    side: str,
    low: float,
    high: float,
    fill_pct: float,
) -> float | None:
    width = max(high - low, EPS)
    clipped_fill = float(np.clip(fill_pct, 0.0, 1.0))
    if clipped_fill <= 0.0:
        return None
    if side == "bull":
        return float(high - (clipped_fill * width))
    return float(low + (clipped_fill * width))


def _renormalize_fill_pct_from_penetrated_prices(
    *,
    side: str,
    low: float,
    high: float,
    penetrated_prices: list[float | None],
) -> float:
    valid_prices = [float(price) for price in penetrated_prices if price is not None]
    if not valid_prices:
        return 0.0
    width = max(high - low, EPS)
    if side == "bull":
        deepest_price = min(valid_prices)
        penetration = max(0.0, high - deepest_price)
    else:
        deepest_price = max(valid_prices)
        penetration = max(0.0, deepest_price - low)
    return float(np.clip(penetration / width, 0.0, 1.0))


def _distance_to_far_edge(close_value: float, low: float, high: float) -> float:
    if close_value < low:
        return high - close_value
    if close_value > high:
        return close_value - low
    return max(close_value - low, high - close_value)


def _overlaps_or_touches(
    low_1: float, high_1: float, low_2: float, high_2: float
) -> bool:
    return max(low_1, low_2) <= min(high_1, high_2)


def _close_outside_zone(
    side: str,
    close_value: float,
    low: float,
    high: float,
) -> bool:
    if side == "bull":
        return close_value >= high
    return close_value <= low


def _rep_entered_this_row(
    side: str,
    rep: _ActiveRepresentation,
    low_value: float,
    high_value: float,
) -> bool:
    if side == "bull":
        return low_value <= rep.high
    return high_value >= rep.low


def _component_entered_this_row(
    side: str,
    component: _RepComponent,
    low_value: float,
    high_value: float,
) -> bool:
    if side == "bull":
        return low_value <= component.high
    return high_value >= component.low


def _build_side_components(
    reps: list[_ActiveRepresentation],
) -> list[_RepComponent]:
    if not reps:
        return []

    ordered = sorted(
        reps,
        key=lambda rep: (
            rep.low,
            rep.high,
            rep.detect_idx,
            rep.active_id,
        ),
    )
    components: list[_RepComponent] = []
    current_members: list[_ActiveRepresentation] = [ordered[0]]
    current_low = ordered[0].low
    current_high = ordered[0].high

    for rep in ordered[1:]:
        if rep.low <= current_high:
            current_members.append(rep)
            current_high = max(current_high, rep.high)
            current_low = min(current_low, rep.low)
            continue
        components.append(
            _RepComponent(
                component_id=len(components) + 1,
                side=current_members[0].side,
                members=list(current_members),
                low=float(current_low),
                high=float(current_high),
            )
        )
        current_members = [rep]
        current_low = rep.low
        current_high = rep.high

    components.append(
        _RepComponent(
            component_id=len(components) + 1,
            side=current_members[0].side,
            members=list(current_members),
            low=float(current_low),
            high=float(current_high),
        )
    )
    return components


def _pick_owned_component(
    side: str,
    components: list[_RepComponent],
    low_value: float,
    high_value: float,
) -> _RepComponent | None:
    entered = [
        component
        for component in components
        if _component_entered_this_row(side, component, low_value, high_value)
    ]
    if not entered:
        return None
    if side == "bull":
        return max(
            entered,
            key=lambda component: (
                component.high,
                component.low,
                -component.component_id,
            ),
        )
    return min(
        entered,
        key=lambda component: (
            component.low,
            component.high,
            component.component_id,
        ),
    )


def _compute_zone(
    pattern_mode: str,
    side: str,
    left_high: float,
    left_low: float,
    left_body_high: float,
    left_body_low: float,
    right_high: float,
    right_low: float,
    right_body_high: float,
    right_body_low: float,
) -> tuple[float, float] | None:
    if pattern_mode == "wick":
        if side == "bull":
            if right_low > left_high:
                return float(left_high), float(right_low)
        else:
            if left_low > right_high:
                return float(right_high), float(left_low)
        return None

    if pattern_mode == "body":
        if side == "bull":
            if right_body_low > left_body_high:
                return float(left_body_high), float(right_body_low)
        else:
            if left_body_low > right_body_high:
                return float(right_body_high), float(left_body_low)
        return None

    if side == "bull":
        if right_low > left_body_high:
            return float(left_body_high), float(right_low)
    else:
        if left_body_low > right_high:
            return float(right_high), float(left_body_low)
    return None


def _compute_pattern_quality(
    *,
    side: str,
    zone_low: float,
    zone_high: float,
    open_mid: float,
    high_mid: float,
    low_mid: float,
    close_mid: float,
    atr_detect: float,
    pattern_mode_code: int,
) -> dict[str, float]:
    mid_body = abs(close_mid - open_mid)
    mid_range = high_mid - low_mid
    denom = max(mid_range, EPS)
    zone_width = zone_high - zone_low
    range_floor = max(EPS, MID_RANGE_FLOOR_ATR_FRACTION * atr_detect)
    gap_cleanliness = np.nan
    if mid_range > range_floor:
        gap_cleanliness = float(zone_width / mid_range)

    if side == "bull":
        left_overlap_frac = max(0.0, zone_low - low_mid) / denom
        right_overlap_frac = max(0.0, high_mid - zone_high) / denom
        signed_impulse_body_atr = (close_mid - open_mid) / atr_detect
    else:
        left_overlap_frac = max(0.0, high_mid - zone_high) / denom
        right_overlap_frac = max(0.0, zone_low - low_mid) / denom
        signed_impulse_body_atr = (open_mid - close_mid) / atr_detect

    return {
        "mid_body": float(mid_body),
        "mid_range": float(mid_range),
        "mid_body_frac": float(mid_body / denom),
        "mid_body_atr": float(mid_body / atr_detect),
        "mid_range_atr": float(mid_range / atr_detect),
        "gap_cleanliness": gap_cleanliness,
        "mid_close_location": float((close_mid - low_mid) / denom),
        "left_overlap_frac": float(left_overlap_frac),
        "right_overlap_frac": float(right_overlap_frac),
        "signed_impulse_body_atr": float(signed_impulse_body_atr),
        "pattern_mode": float(pattern_mode_code),
    }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _base_quality_score(
    side: str, quality: dict[str, float], width_atr: float
) -> float:
    close_location = quality["mid_close_location"]
    directional_close = close_location if side == "bull" else (1.0 - close_location)
    return float(
        np.clip(
            (
                0.25 * _clip01(width_atr / BASE_QUALITY_WIDTH_ATR_CAP)
                + 0.20
                * _clip01(quality["mid_body_atr"] / BASE_QUALITY_MID_BODY_ATR_CAP)
                + 0.15 * _clip01(quality["mid_body_frac"])
                + 0.20
                * _clip01(quality["gap_cleanliness"] / BASE_QUALITY_GAP_CLEANLINESS_CAP)
                + 0.20 * _clip01(directional_close)
            ),
            0.0,
            1.0,
        )
    )


def _age_decay(age_bars: int) -> float:
    return float(2.0 ** (-max(age_bars, 0) / HALF_LIFE_BARS))


def _interaction_modifier(rep: _ActiveRepresentation) -> float:
    if rep.max_fill_pct >= 0.50:
        return PARTIAL_FILL_GTE_50_MODIFIER
    if rep.max_fill_pct >= 0.25:
        return PARTIAL_FILL_LT_50_MODIFIER
    if rep.max_fill_pct > 0.0:
        return PARTIAL_FILL_LT_25_MODIFIER
    if rep.touch_count > 0:
        return TOUCHED_ONLY_MODIFIER
    return 1.0


def _distance_modifier(
    close_value: float,
    atr_value: float,
    rep: _ActiveRepresentation,
) -> float:
    if not np.isfinite(atr_value) or atr_value <= 0:
        return 0.0
    distance_atr = _zone_distance(close_value, rep.low, rep.high) / atr_value
    return float(1.0 / (1.0 + max(distance_atr, 0.0)))


def _effective_significance(
    *,
    close_value: float,
    atr_value: float,
    current_idx: int,
    rep: _ActiveRepresentation,
) -> float:
    age_bars = current_idx - rep.age_anchor_idx
    return float(
        np.clip(
            rep.base_quality
            * _age_decay(age_bars)
            * _interaction_modifier(rep)
            * _distance_modifier(close_value, atr_value, rep),
            0.0,
            1.0,
        )
    )


def _pick_selected_representation(
    close_value: float,
    atr_value: float,
    current_idx: int,
    representations: list[_ActiveRepresentation],
) -> _ActiveRepresentation | None:
    if not representations:
        return None

    return max(
        representations,
        key=lambda rep: (
            _effective_significance(
                close_value=close_value,
                atr_value=atr_value,
                current_idx=current_idx,
                rep=rep,
            ),
            -_zone_distance(close_value, rep.low, rep.high),
            rep.detect_idx,
            rep.width_atr,
            -rep.min_source_event_id,
        ),
    )


def _run_fvg(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_width_atr: float = 0.1,
    pattern_mode: str = "wick",
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
    selector: str = "significance",
    collect_debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Detect canonical causal FVG events and dense active state."""
    if selector not in {"nearest", "significance"}:
        raise ValueError("selector must be one of {'nearest', 'significance'}")
    if max_active_age_bars is None or max_active_age_bars <= 0:
        raise ValueError("max_active_age_bars must be a positive integer")

    out = df.copy()
    require_columns(out, {"timestamp", "open", "high", "low", "close"})

    n = len(out)
    if n == 0:
        return out

    pattern_mode_code = _pattern_mode_code(pattern_mode)

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    ts = pd.to_datetime(out["timestamp"], utc=True)
    atr = get_atr_array(out, atr_length)

    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)

    arrays: dict[str, np.ndarray | pd.Series] = {
        "fvg_bull_origin_flag": np.zeros(n, dtype=np.int8),
        "fvg_bull_origin_idx": np.full(n, np.nan),
        "fvg_bull_origin_ts": _empty_ts_series(out.index, ts.dtype),
        "fvg_bull_left_idx": np.full(n, np.nan),
        "fvg_bull_mid_idx": np.full(n, np.nan),
        "fvg_bull_right_idx": np.full(n, np.nan),
        "fvg_bear_origin_flag": np.zeros(n, dtype=np.int8),
        "fvg_bear_origin_idx": np.full(n, np.nan),
        "fvg_bear_origin_ts": _empty_ts_series(out.index, ts.dtype),
        "fvg_bear_left_idx": np.full(n, np.nan),
        "fvg_bear_mid_idx": np.full(n, np.nan),
        "fvg_bear_right_idx": np.full(n, np.nan),
        "fvg_bull_detect_flag": np.zeros(n, dtype=np.int8),
        "fvg_bull_detect_idx": np.full(n, np.nan),
        "fvg_bull_detect_ts": _empty_ts_series(out.index, ts.dtype),
        "fvg_bull_event_id": np.zeros(n, dtype=np.int64),
        "fvg_bull_detect_delay": np.zeros(n, dtype=np.int8),
        "fvg_bear_detect_flag": np.zeros(n, dtype=np.int8),
        "fvg_bear_detect_idx": np.full(n, np.nan),
        "fvg_bear_detect_ts": _empty_ts_series(out.index, ts.dtype),
        "fvg_bear_event_id": np.zeros(n, dtype=np.int64),
        "fvg_bear_detect_delay": np.zeros(n, dtype=np.int8),
        "fvg_bull_low": np.full(n, np.nan),
        "fvg_bull_high": np.full(n, np.nan),
        "fvg_bull_mid": np.full(n, np.nan),
        "fvg_bull_width": np.full(n, np.nan),
        "fvg_bull_width_atr": np.full(n, np.nan),
        "fvg_bear_low": np.full(n, np.nan),
        "fvg_bear_high": np.full(n, np.nan),
        "fvg_bear_mid": np.full(n, np.nan),
        "fvg_bear_width": np.full(n, np.nan),
        "fvg_bear_width_atr": np.full(n, np.nan),
        "fvg_bull_mid_body": np.full(n, np.nan),
        "fvg_bull_mid_range": np.full(n, np.nan),
        "fvg_bull_mid_body_frac": np.full(n, np.nan),
        "fvg_bull_mid_body_atr": np.full(n, np.nan),
        "fvg_bull_mid_range_atr": np.full(n, np.nan),
        "fvg_bull_gap_cleanliness": np.full(n, np.nan),
        "fvg_bull_mid_close_location": np.full(n, np.nan),
        "fvg_bull_left_overlap_frac": np.full(n, np.nan),
        "fvg_bull_right_overlap_frac": np.full(n, np.nan),
        "fvg_bull_signed_impulse_body_atr": np.full(n, np.nan),
        "fvg_bull_pattern_mode": np.zeros(n, dtype=np.int8),
        "fvg_bear_mid_body": np.full(n, np.nan),
        "fvg_bear_mid_range": np.full(n, np.nan),
        "fvg_bear_mid_body_frac": np.full(n, np.nan),
        "fvg_bear_mid_body_atr": np.full(n, np.nan),
        "fvg_bear_mid_range_atr": np.full(n, np.nan),
        "fvg_bear_gap_cleanliness": np.full(n, np.nan),
        "fvg_bear_mid_close_location": np.full(n, np.nan),
        "fvg_bear_left_overlap_frac": np.full(n, np.nan),
        "fvg_bear_right_overlap_frac": np.full(n, np.nan),
        "fvg_bear_signed_impulse_body_atr": np.full(n, np.nan),
        "fvg_bear_pattern_mode": np.zeros(n, dtype=np.int8),
        "fvg_bull_active": np.zeros(n, dtype=np.int8),
        "fvg_bull_active_id": np.zeros(n, dtype=np.int64),
        "fvg_bull_active_low": np.full(n, np.nan),
        "fvg_bull_active_high": np.full(n, np.nan),
        "fvg_bull_active_mid": np.full(n, np.nan),
        "fvg_bull_active_width": np.full(n, np.nan),
        "fvg_bull_active_width_atr": np.full(n, np.nan),
        "fvg_bull_active_age": np.full(n, np.nan),
        "fvg_bull_active_age_bars": np.full(n, np.nan),
        "fvg_bull_active_age_decay": np.full(n, np.nan),
        "fvg_bull_active_base_quality": np.full(n, np.nan),
        "fvg_bull_active_effective_significance": np.full(n, np.nan),
        "fvg_bull_active_fill_pct": np.full(n, np.nan),
        "fvg_bull_active_touch_count": np.zeros(n, dtype=np.int32),
        "fvg_bull_active_state": np.zeros(n, dtype=np.int8),
        "fvg_bull_active_since_idx": np.full(n, np.nan),
        "fvg_bull_first_retest_flag": np.zeros(n, dtype=np.int8),
        "fvg_bull_zone_invalidated": np.zeros(n, dtype=np.int8),
        "fvg_bull_distance_to_mid_atr": np.full(n, np.nan),
        "fvg_bull_distance_to_near_edge_atr": np.full(n, np.nan),
        "fvg_bull_distance_to_far_edge_atr": np.full(n, np.nan),
        "fvg_bull_active_count": np.zeros(n, dtype=np.int32),
        "fvg_bull_active_weighted_count": np.full(n, np.nan),
        "fvg_bear_active": np.zeros(n, dtype=np.int8),
        "fvg_bear_active_id": np.zeros(n, dtype=np.int64),
        "fvg_bear_active_low": np.full(n, np.nan),
        "fvg_bear_active_high": np.full(n, np.nan),
        "fvg_bear_active_mid": np.full(n, np.nan),
        "fvg_bear_active_width": np.full(n, np.nan),
        "fvg_bear_active_width_atr": np.full(n, np.nan),
        "fvg_bear_active_age": np.full(n, np.nan),
        "fvg_bear_active_age_bars": np.full(n, np.nan),
        "fvg_bear_active_age_decay": np.full(n, np.nan),
        "fvg_bear_active_base_quality": np.full(n, np.nan),
        "fvg_bear_active_effective_significance": np.full(n, np.nan),
        "fvg_bear_active_fill_pct": np.full(n, np.nan),
        "fvg_bear_active_touch_count": np.zeros(n, dtype=np.int32),
        "fvg_bear_active_state": np.zeros(n, dtype=np.int8),
        "fvg_bear_active_since_idx": np.full(n, np.nan),
        "fvg_bear_first_retest_flag": np.zeros(n, dtype=np.int8),
        "fvg_bear_zone_invalidated": np.zeros(n, dtype=np.int8),
        "fvg_bear_distance_to_mid_atr": np.full(n, np.nan),
        "fvg_bear_distance_to_near_edge_atr": np.full(n, np.nan),
        "fvg_bear_distance_to_far_edge_atr": np.full(n, np.nan),
        "fvg_bear_active_count": np.zeros(n, dtype=np.int32),
        "fvg_bear_active_weighted_count": np.full(n, np.nan),
        "fvg_bull_first_touch_idx": np.full(n, np.nan),
        "fvg_bull_first_partial_fill_idx": np.full(n, np.nan),
        "fvg_bull_full_fill_idx": np.full(n, np.nan),
        "fvg_bull_invalidation_idx": np.full(n, np.nan),
        "fvg_bull_expiry_idx": np.full(n, np.nan),
        "fvg_bull_terminal_state": np.zeros(n, dtype=np.int8),
        "fvg_bear_first_touch_idx": np.full(n, np.nan),
        "fvg_bear_first_partial_fill_idx": np.full(n, np.nan),
        "fvg_bear_full_fill_idx": np.full(n, np.nan),
        "fvg_bear_invalidation_idx": np.full(n, np.nan),
        "fvg_bear_expiry_idx": np.full(n, np.nan),
        "fvg_bear_terminal_state": np.zeros(n, dtype=np.int8),
    }

    raw_events: dict[int, _RawEvent] = {}
    active_reps: dict[str, list[_ActiveRepresentation]] = {"bull": [], "bear": []}
    debug_rep_records: dict[int, dict[str, object]] = {}
    debug_active_members: list[dict[str, object]] = []

    next_event_id = 1
    next_active_id = 1

    def _stamp_raw_event(event: _RawEvent, quality: dict[str, float]) -> None:
        side = event.side
        origin_prefix = f"fvg_{side}_origin"
        detect_prefix = f"fvg_{side}"
        arrays[f"{origin_prefix}_flag"][event.origin_idx] = 1
        arrays[f"{origin_prefix}_idx"][event.origin_idx] = float(event.origin_idx)
        arrays[f"{origin_prefix}_ts"].iloc[event.origin_idx] = ts.iloc[event.origin_idx]
        arrays[f"fvg_{side}_left_idx"][event.origin_idx] = float(event.origin_idx - 1)
        arrays[f"fvg_{side}_mid_idx"][event.origin_idx] = float(event.origin_idx)
        arrays[f"fvg_{side}_right_idx"][event.origin_idx] = float(event.detect_idx)

        arrays[f"{detect_prefix}_detect_flag"][event.detect_idx] = 1
        arrays[f"{detect_prefix}_detect_idx"][event.detect_idx] = float(
            event.detect_idx
        )
        arrays[f"{detect_prefix}_detect_ts"].iloc[event.detect_idx] = event.detect_ts
        arrays[f"{detect_prefix}_event_id"][event.detect_idx] = event.event_id
        arrays[f"{detect_prefix}_detect_delay"][event.detect_idx] = 1

        arrays[f"{detect_prefix}_low"][event.detect_idx] = event.low
        arrays[f"{detect_prefix}_high"][event.detect_idx] = event.high
        arrays[f"{detect_prefix}_mid"][event.detect_idx] = event.mid
        arrays[f"{detect_prefix}_width"][event.detect_idx] = event.width
        arrays[f"{detect_prefix}_width_atr"][event.detect_idx] = event.width_atr

        arrays[f"{detect_prefix}_mid_body"][event.detect_idx] = quality["mid_body"]
        arrays[f"{detect_prefix}_mid_range"][event.detect_idx] = quality["mid_range"]
        arrays[f"{detect_prefix}_mid_body_frac"][event.detect_idx] = quality[
            "mid_body_frac"
        ]
        arrays[f"{detect_prefix}_mid_body_atr"][event.detect_idx] = quality[
            "mid_body_atr"
        ]
        arrays[f"{detect_prefix}_mid_range_atr"][event.detect_idx] = quality[
            "mid_range_atr"
        ]
        arrays[f"{detect_prefix}_gap_cleanliness"][event.detect_idx] = quality[
            "gap_cleanliness"
        ]
        arrays[f"{detect_prefix}_mid_close_location"][event.detect_idx] = quality[
            "mid_close_location"
        ]
        arrays[f"{detect_prefix}_left_overlap_frac"][event.detect_idx] = quality[
            "left_overlap_frac"
        ]
        arrays[f"{detect_prefix}_right_overlap_frac"][event.detect_idx] = quality[
            "right_overlap_frac"
        ]
        arrays[f"{detect_prefix}_signed_impulse_body_atr"][event.detect_idx] = quality[
            "signed_impulse_body_atr"
        ]
        arrays[f"{detect_prefix}_pattern_mode"][event.detect_idx] = int(
            quality["pattern_mode"]
        )

    def _sync_raw_lifecycle(event_id: int) -> None:
        event = raw_events[event_id]
        prefix = event.lifecycle_prefix
        detect_idx = event.detect_idx
        arrays[f"{prefix}_first_touch_idx"][detect_idx] = event.first_touch_idx
        arrays[f"{prefix}_first_partial_fill_idx"][
            detect_idx
        ] = event.first_partial_fill_idx
        arrays[f"{prefix}_full_fill_idx"][detect_idx] = event.full_fill_idx
        arrays[f"{prefix}_invalidation_idx"][detect_idx] = event.invalidation_idx
        arrays[f"{prefix}_expiry_idx"][detect_idx] = event.expiry_idx
        arrays[f"{prefix}_terminal_state"][detect_idx] = event.terminal_state

    def _mark_raw_merged(event_id: int, merge_idx: int) -> None:
        event = raw_events[event_id]
        event.merge_idx = float(merge_idx)
        event.terminal_state = STATE_MERGED
        _sync_raw_lifecycle(event_id)

    def _start_rep_debug(rep: _ActiveRepresentation) -> None:
        if not collect_debug:
            return
        debug_rep_records[rep.active_id] = {
            "active_id": rep.active_id,
            "side": rep.side,
            "start_idx": int(rep.active_since_idx),
            "detect_idx": int(rep.detect_idx),
            "age_anchor_idx": int(rep.age_anchor_idx),
            "end_idx": np.nan,
            "terminal_reason": "unresolved",
            "low": float(rep.low),
            "high": float(rep.high),
            "width": float(rep.width),
            "width_atr": float(rep.width_atr),
            "base_quality": float(rep.base_quality),
            "source_count": int(rep.source_count),
            "source_ids": ",".join(str(source_id) for source_id in rep.source_ids),
            "source_detect_indices": ",".join(
                str(source_idx) for source_idx in rep.source_detect_indices
            ),
        }

    def _finish_rep_debug(
        rep: _ActiveRepresentation, end_idx: int, reason: str
    ) -> None:
        if not collect_debug:
            return
        record = debug_rep_records.get(rep.active_id)
        if record is None or np.isfinite(record["end_idx"]):
            return
        record["end_idx"] = float(end_idx)
        record["terminal_reason"] = reason

    for i in range(n):
        new_events: list[_RawEvent] = []
        if i >= 2 and np.isfinite(atr[i]) and atr[i] > 0:
            origin_idx = i - 1
            left_idx = i - 2

            candidates: list[tuple[str, float, float]] = []
            bull_zone = _compute_zone(
                pattern_mode,
                "bull",
                h[left_idx],
                l[left_idx],
                body_hi[left_idx],
                body_lo[left_idx],
                h[i],
                l[i],
                body_hi[i],
                body_lo[i],
            )
            if bull_zone is not None:
                candidates.append(("bull", bull_zone[0], bull_zone[1]))

            bear_zone = _compute_zone(
                pattern_mode,
                "bear",
                h[left_idx],
                l[left_idx],
                body_hi[left_idx],
                body_lo[left_idx],
                h[i],
                l[i],
                body_hi[i],
                body_lo[i],
            )
            if bear_zone is not None:
                candidates.append(("bear", bear_zone[0], bear_zone[1]))

            candidates.sort(
                key=lambda item: (
                    0 if item[0] == "bull" else 1,
                    item[1],
                    item[2],
                    origin_idx,
                )
            )

            for side, zone_low, zone_high in candidates:
                zone_width = zone_high - zone_low
                if not (zone_width > 0):
                    continue
                if zone_width < min_width_atr * atr[i]:
                    continue

                event = _RawEvent(
                    event_id=next_event_id,
                    side=side,
                    origin_idx=origin_idx,
                    detect_idx=i,
                    detect_ts=ts.iloc[i],
                    low=float(zone_low),
                    high=float(zone_high),
                    mid=float((zone_low + zone_high) / 2.0),
                    width=float(zone_width),
                    width_atr=float(zone_width / atr[i]),
                    pattern_mode=pattern_mode_code,
                    base_quality=np.nan,
                )
                quality = _compute_pattern_quality(
                    side=side,
                    zone_low=event.low,
                    zone_high=event.high,
                    open_mid=o[origin_idx],
                    high_mid=h[origin_idx],
                    low_mid=l[origin_idx],
                    close_mid=c[origin_idx],
                    atr_detect=atr[i],
                    pattern_mode_code=pattern_mode_code,
                )
                event.base_quality = _base_quality_score(
                    side=side,
                    quality=quality,
                    width_atr=event.width_atr,
                )
                raw_events[event.event_id] = event
                _stamp_raw_event(event, quality)
                _sync_raw_lifecycle(event.event_id)
                new_events.append(event)
                next_event_id += 1

        invalidated_on_row: dict[str, bool] = {"bull": False, "bear": False}
        row_ownership_meta: dict[str, dict[int, dict[str, object]]] = {
            "bull": {},
            "bear": {},
        }

        for side in ("bull", "bear"):
            components = _build_side_components(active_reps[side])
            owned_component = _pick_owned_component(side, components, l[i], h[i])
            component_by_rep_id = {
                rep.active_id: component
                for component in components
                for rep in component.members
            }
            updated_reps: list[_ActiveRepresentation] = []
            for rep in active_reps[side]:
                component = component_by_rep_id[rep.active_id]
                rep_entered = _rep_entered_this_row(side, rep, l[i], h[i])
                component_entered = _component_entered_this_row(
                    side, component, l[i], h[i]
                )
                component_owned = (
                    owned_component is not None
                    and component.component_id == owned_component.component_id
                )
                row_ownership_meta[side][rep.active_id] = {
                    "component_id": int(component.component_id),
                    "component_low": float(component.low),
                    "component_high": float(component.high),
                    "component_member_count": int(component.member_count),
                    "component_entered_this_row": int(component_entered),
                    "component_owned_this_row": int(component_owned),
                    "entered_this_row": int(rep_entered),
                    "ownership_blocked_by_higher_component": int(
                        side == "bull" and component_entered and not component_owned
                    ),
                    "ownership_blocked_by_lower_component": int(
                        side == "bear" and component_entered and not component_owned
                    ),
                    "touch_delta": 0,
                    "fill_delta": 0.0,
                }

                prev_touch_count = rep.touch_count
                prev_fill_pct = rep.max_fill_pct
                if side == "bull":
                    invalidated = c[i] < rep.low
                    full_fill = l[i] <= rep.low
                    touch = l[i] <= rep.high
                    penetration = max(0.0, rep.high - l[i])
                    favorable_move = max(0.0, h[i] - rep.high)
                else:
                    invalidated = c[i] > rep.high
                    full_fill = h[i] >= rep.high
                    touch = h[i] >= rep.low
                    penetration = max(0.0, h[i] - rep.low)
                    favorable_move = max(0.0, rep.low - l[i])

                fill_pct = float(np.clip(penetration / max(rep.width, EPS), 0.0, 1.0))
                if touch:
                    if rep.historical_touch_count == 0 and rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        raw_event.first_touch_idx = float(i)
                        _sync_raw_lifecycle(raw_event.event_id)
                    if rep.historical_first_touch_idx is None:
                        rep.historical_first_touch_idx = i
                    rep.historical_touch_count += 1
                    rep.historical_last_touch_idx = i
                    if fill_pct > rep.historical_fill_pct:
                        rep.historical_max_single_bar_fill_jump = max(
                            rep.historical_max_single_bar_fill_jump,
                            fill_pct - rep.historical_fill_pct,
                        )
                    if fill_pct > 0.0 and rep.historical_first_partial_fill_idx is None:
                        rep.historical_first_partial_fill_idx = i
                        if rep.source_count == 1:
                            raw_event = raw_events[rep.source_ids[0]]
                            if not np.isfinite(raw_event.first_partial_fill_idx):
                                raw_event.first_partial_fill_idx = float(i)
                                _sync_raw_lifecycle(raw_event.event_id)
                    close_outside = _close_outside_zone(side, c[i], rep.low, rep.high)
                    rep.historical_all_touch_bars_closed_outside = (
                        rep.historical_all_touch_bars_closed_outside and close_outside
                    )
                    rep.historical_last_touch_rejection_flag = close_outside
                    rep.historical_bounce_raw_max = favorable_move
                elif rep.historical_last_touch_idx is not None:
                    rep.historical_bounce_raw_max = max(
                        rep.historical_bounce_raw_max, favorable_move
                    )

                if fill_pct > rep.historical_fill_pct:
                    rep.historical_max_single_bar_fill_jump = max(
                        rep.historical_max_single_bar_fill_jump,
                        fill_pct - rep.historical_fill_pct,
                    )
                    rep.historical_fill_pct = fill_pct

                stale_disjoint_live_fill = (
                    len(components) > 1
                    and rep.last_touch_idx is not None
                    and not rep_entered
                    and not component_owned
                    and (i - rep.last_touch_idx) > LIVE_FILL_PERSISTENCE_BARS
                )
                if stale_disjoint_live_fill:
                    _reset_live_fill_state(rep)

                if owned_component is not None and not component_owned:
                    if invalidated or full_fill:
                        if rep.source_count == 1:
                            raw_event = raw_events[rep.source_ids[0]]
                            if not np.isfinite(raw_event.first_touch_idx):
                                raw_event.first_touch_idx = float(i)
                            if fill_pct > 0.0 and not np.isfinite(
                                raw_event.first_partial_fill_idx
                            ):
                                raw_event.first_partial_fill_idx = float(i)
                            if invalidated:
                                raw_event.invalidation_idx = float(i)
                                raw_event.terminal_state = STATE_INVALIDATED
                            else:
                                raw_event.full_fill_idx = float(i)
                                raw_event.terminal_state = STATE_FULLY_FILLED
                            _sync_raw_lifecycle(raw_event.event_id)
                        if invalidated:
                            invalidated_on_row[side] = True
                            _finish_rep_debug(rep, i, "invalidated")
                        else:
                            _finish_rep_debug(rep, i, "fully_filled")
                        continue
                    _reset_live_fill_state(rep)
                    updated_reps.append(rep)
                    continue

                if touch:
                    if rep.touch_count == 0 and rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        raw_event.first_touch_idx = float(i)
                        _sync_raw_lifecycle(raw_event.event_id)
                    if rep.first_touch_idx is None:
                        rep.first_touch_idx = i
                    rep.touch_count += 1
                    rep.last_touch_idx = i
                    if fill_pct > rep.max_fill_pct:
                        rep.max_single_bar_fill_jump = max(
                            rep.max_single_bar_fill_jump, fill_pct - rep.max_fill_pct
                        )
                    if fill_pct > 0.0 and rep.first_partial_fill_idx is None:
                        rep.first_partial_fill_idx = i
                    close_outside = _close_outside_zone(side, c[i], rep.low, rep.high)
                    rep.all_touch_bars_closed_outside = (
                        rep.all_touch_bars_closed_outside and close_outside
                    )
                    rep.last_touch_rejection_flag = close_outside
                    rep.bounce_raw_max = favorable_move
                elif rep.last_touch_idx is not None:
                    rep.bounce_raw_max = max(rep.bounce_raw_max, favorable_move)

                if fill_pct > rep.max_fill_pct:
                    rep.max_single_bar_fill_jump = max(
                        rep.max_single_bar_fill_jump, fill_pct - rep.max_fill_pct
                    )
                    rep.max_fill_pct = fill_pct

                row_ownership_meta[side][rep.active_id]["touch_delta"] = int(
                    rep.touch_count - prev_touch_count
                )
                row_ownership_meta[side][rep.active_id]["fill_delta"] = float(
                    rep.max_fill_pct - prev_fill_pct
                )

                if invalidated:
                    if rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        raw_event.invalidation_idx = float(i)
                        raw_event.terminal_state = STATE_INVALIDATED
                        _sync_raw_lifecycle(raw_event.event_id)
                    invalidated_on_row[side] = True
                    _finish_rep_debug(rep, i, "invalidated")
                    continue

                if full_fill:
                    if rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        raw_event.full_fill_idx = float(i)
                        raw_event.terminal_state = STATE_FULLY_FILLED
                        _sync_raw_lifecycle(raw_event.event_id)
                    _finish_rep_debug(rep, i, "fully_filled")
                    continue

                if 0.0 < rep.max_fill_pct < 1.0:
                    rep.state = STATE_ACTIVE_PARTIAL_FILL
                    if rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        if not np.isfinite(raw_event.first_partial_fill_idx):
                            raw_event.first_partial_fill_idx = float(i)
                            _sync_raw_lifecycle(raw_event.event_id)
                elif touch:
                    rep.state = STATE_ACTIVE_TOUCHED
                else:
                    rep.state = (
                        STATE_ACTIVE_UNTOUCHED
                        if rep.touch_count == 0
                        else STATE_ACTIVE_TOUCHED
                    )

                updated_reps.append(rep)

            active_reps[side] = updated_reps

        for event in new_events:
            side_reps = active_reps[event.side]
            overlapping = [
                rep
                for rep in side_reps
                if _overlaps_or_touches(rep.low, rep.high, event.low, event.high)
            ]

            if not overlapping:
                new_rep = _ActiveRepresentation(
                    active_id=next_active_id,
                    side=event.side,
                    detect_idx=event.detect_idx,
                    age_anchor_idx=event.detect_idx,
                    low=event.low,
                    high=event.high,
                    width_atr=event.width_atr,
                    base_quality=event.base_quality,
                    source_ids=[event.event_id],
                    source_detect_indices=[event.detect_idx],
                    active_since_idx=event.detect_idx,
                )
                side_reps.append(new_rep)
                _start_rep_debug(new_rep)
                next_active_id += 1
                continue

            source_ids = [event.event_id]
            source_detect_indices = [event.detect_idx]
            merged_low = event.low
            merged_high = event.high
            merged_width_atr = event.width_atr
            merged_base_quality = event.base_quality
            merged_age_anchor_idx = event.detect_idx
            live_touch_count = 0
            historical_touch_count = 0
            live_first_touch_idx: int | None = None
            live_last_touch_idx: int | None = None
            live_first_partial_fill_idx: int | None = None
            historical_first_touch_idx: int | None = None
            historical_last_touch_idx: int | None = None
            historical_first_partial_fill_idx: int | None = None
            live_max_single_bar_fill_jump = 0.0
            historical_max_single_bar_fill_jump = 0.0
            live_all_touch_bars_closed_outside = True
            historical_all_touch_bars_closed_outside = True
            live_last_touch_rejection_flag: bool | None = None
            historical_last_touch_rejection_flag: bool | None = None
            live_bounce_raw_max = 0.0
            historical_bounce_raw_max = 0.0
            live_penetrated_prices: list[float | None] = []
            historical_penetrated_prices: list[float | None] = []

            for rep in overlapping:
                merged_low = min(merged_low, rep.low)
                merged_high = max(merged_high, rep.high)
                merged_width_atr = max(merged_width_atr, rep.width_atr)
                merged_base_quality = max(merged_base_quality, rep.base_quality)
                merged_age_anchor_idx = min(merged_age_anchor_idx, rep.age_anchor_idx)
                source_ids.extend(rep.source_ids)
                source_detect_indices.extend(rep.source_detect_indices)
                live_touch_count += rep.touch_count
                historical_touch_count += rep.historical_touch_count
                live_first_touch_idx = (
                    rep.first_touch_idx
                    if live_first_touch_idx is None
                    else (
                        min(live_first_touch_idx, rep.first_touch_idx)
                        if rep.first_touch_idx is not None
                        else live_first_touch_idx
                    )
                )
                live_last_touch_idx = (
                    rep.last_touch_idx
                    if live_last_touch_idx is None
                    else (
                        max(live_last_touch_idx, rep.last_touch_idx)
                        if rep.last_touch_idx is not None
                        else live_last_touch_idx
                    )
                )
                live_first_partial_fill_idx = (
                    rep.first_partial_fill_idx
                    if live_first_partial_fill_idx is None
                    else (
                        min(live_first_partial_fill_idx, rep.first_partial_fill_idx)
                        if rep.first_partial_fill_idx is not None
                        else live_first_partial_fill_idx
                    )
                )
                historical_first_touch_idx = (
                    rep.historical_first_touch_idx
                    if historical_first_touch_idx is None
                    else (
                        min(
                            historical_first_touch_idx,
                            rep.historical_first_touch_idx,
                        )
                        if rep.historical_first_touch_idx is not None
                        else historical_first_touch_idx
                    )
                )
                historical_last_touch_idx = (
                    rep.historical_last_touch_idx
                    if historical_last_touch_idx is None
                    else (
                        max(
                            historical_last_touch_idx,
                            rep.historical_last_touch_idx,
                        )
                        if rep.historical_last_touch_idx is not None
                        else historical_last_touch_idx
                    )
                )
                historical_first_partial_fill_idx = (
                    rep.historical_first_partial_fill_idx
                    if historical_first_partial_fill_idx is None
                    else (
                        min(
                            historical_first_partial_fill_idx,
                            rep.historical_first_partial_fill_idx,
                        )
                        if rep.historical_first_partial_fill_idx is not None
                        else historical_first_partial_fill_idx
                    )
                )
                live_max_single_bar_fill_jump = max(
                    live_max_single_bar_fill_jump, rep.max_single_bar_fill_jump
                )
                historical_max_single_bar_fill_jump = max(
                    historical_max_single_bar_fill_jump,
                    rep.historical_max_single_bar_fill_jump,
                )
                live_all_touch_bars_closed_outside = (
                    live_all_touch_bars_closed_outside
                    and rep.all_touch_bars_closed_outside
                )
                historical_all_touch_bars_closed_outside = (
                    historical_all_touch_bars_closed_outside
                    and rep.historical_all_touch_bars_closed_outside
                )
                if rep.last_touch_idx is not None and (
                    live_last_touch_rejection_flag is None
                    or live_last_touch_idx == rep.last_touch_idx
                ):
                    live_last_touch_rejection_flag = rep.last_touch_rejection_flag
                if rep.historical_last_touch_idx is not None and (
                    historical_last_touch_rejection_flag is None
                    or historical_last_touch_idx == rep.historical_last_touch_idx
                ):
                    historical_last_touch_rejection_flag = (
                        rep.historical_last_touch_rejection_flag
                    )
                live_bounce_raw_max = max(live_bounce_raw_max, rep.bounce_raw_max)
                historical_bounce_raw_max = max(
                    historical_bounce_raw_max, rep.historical_bounce_raw_max
                )
                live_penetrated_prices.append(
                    _penetrated_price_from_fill_pct(
                        side=rep.side,
                        low=rep.low,
                        high=rep.high,
                        fill_pct=rep.max_fill_pct,
                    )
                )
                historical_penetrated_prices.append(
                    _penetrated_price_from_fill_pct(
                        side=rep.side,
                        low=rep.low,
                        high=rep.high,
                        fill_pct=rep.historical_fill_pct,
                    )
                )

            merged_live_fill_pct = _renormalize_fill_pct_from_penetrated_prices(
                side=event.side,
                low=float(merged_low),
                high=float(merged_high),
                penetrated_prices=live_penetrated_prices,
            )
            merged_historical_fill_pct = _renormalize_fill_pct_from_penetrated_prices(
                side=event.side,
                low=float(merged_low),
                high=float(merged_high),
                penetrated_prices=historical_penetrated_prices,
            )

            for source_id in source_ids:
                _mark_raw_merged(source_id, i)

            for rep in overlapping:
                _finish_rep_debug(rep, i, "merged")

            active_reps[event.side] = [
                rep for rep in side_reps if rep not in overlapping
            ]
            new_rep = _ActiveRepresentation(
                active_id=next_active_id,
                side=event.side,
                detect_idx=i,
                age_anchor_idx=merged_age_anchor_idx,
                low=float(merged_low),
                high=float(merged_high),
                width_atr=float(merged_width_atr),
                base_quality=float(merged_base_quality),
                source_ids=sorted(set(source_ids)),
                source_detect_indices=sorted(set(source_detect_indices)),
                active_since_idx=i,
                state=_fill_state_from_metrics(
                    merged_live_fill_pct,
                    live_touch_count,
                ),
                touch_count=live_touch_count,
                max_fill_pct=merged_live_fill_pct,
                first_touch_idx=live_first_touch_idx,
                last_touch_idx=live_last_touch_idx,
                first_partial_fill_idx=live_first_partial_fill_idx,
                max_single_bar_fill_jump=live_max_single_bar_fill_jump,
                all_touch_bars_closed_outside=live_all_touch_bars_closed_outside,
                last_touch_rejection_flag=live_last_touch_rejection_flag,
                bounce_raw_max=live_bounce_raw_max,
                first_retest_emitted=False,
                historical_touch_count=historical_touch_count,
                historical_fill_pct=merged_historical_fill_pct,
                historical_first_touch_idx=historical_first_touch_idx,
                historical_last_touch_idx=historical_last_touch_idx,
                historical_first_partial_fill_idx=historical_first_partial_fill_idx,
                historical_max_single_bar_fill_jump=historical_max_single_bar_fill_jump,
                historical_all_touch_bars_closed_outside=historical_all_touch_bars_closed_outside,
                historical_last_touch_rejection_flag=historical_last_touch_rejection_flag,
                historical_bounce_raw_max=historical_bounce_raw_max,
            )
            # NOTE: IFVG gap — merged-zone invalidations never generate IFVGs.
            # All constituent source_ids (including the newly detected event) are
            # stamped STATE_MERGED above. The merged rep has source_count > 1, so
            # the invalidation guard below (`if rep.source_count == 1`) means none
            # of its constituents are ever stamped STATE_INVALIDATED. The IFVG
            # consumer filters exclusively on STATE_INVALIDATED, so it will never
            # see these zones. Fixing this would require either iterating all
            # constituent source_ids on invalidation or introducing a synthetic raw
            # event for merged zones — a larger architectural change deferred for now.
            active_reps[event.side].append(new_rep)
            _start_rep_debug(new_rep)
            next_active_id += 1

        for side in ("bull", "bear"):
            survivors: list[_ActiveRepresentation] = []
            for rep in active_reps[side]:
                if (i - rep.age_anchor_idx) >= int(max_active_age_bars):
                    if rep.source_count == 1:
                        raw_event = raw_events[rep.source_ids[0]]
                        raw_event.expiry_idx = float(i)
                        raw_event.terminal_state = STATE_EXPIRED
                        _sync_raw_lifecycle(raw_event.event_id)
                    _finish_rep_debug(rep, i, "expired")
                    continue
                survivors.append(rep)
            active_reps[side] = survivors

        final_component_meta: dict[str, dict[int, dict[str, object]]] = {
            "bull": {},
            "bear": {},
        }
        for side in ("bull", "bear"):
            for component in _build_side_components(active_reps[side]):
                for rep in component.members:
                    final_component_meta[side][rep.active_id] = {
                        "component_id": int(component.component_id),
                        "component_low": float(component.low),
                        "component_high": float(component.high),
                        "component_member_count": int(component.member_count),
                    }

        for side in ("bull", "bear"):
            rep_list = active_reps[side]
            arrays[f"fvg_{side}_active_count"][i] = len(rep_list)
            if rep_list:
                arrays[f"fvg_{side}_active_weighted_count"][i] = float(
                    sum(
                        _effective_significance(
                            close_value=c[i],
                            atr_value=atr[i],
                            current_idx=i,
                            rep=rep,
                        )
                        for rep in rep_list
                    )
                )

            if selector == "nearest":
                selected = (
                    min(
                        rep_list,
                        key=lambda rep: (
                            _zone_distance(c[i], rep.low, rep.high),
                            abs(c[i] - rep.mid),
                            -rep.detect_idx,
                            -rep.width_atr,
                            rep.min_source_event_id,
                        ),
                    )
                    if rep_list
                    else None
                )
            else:
                selected = _pick_selected_representation(c[i], atr[i], i, rep_list)
            selected_id = selected.active_id if selected is not None else 0
            if collect_debug:
                for rep in rep_list:
                    ownership_meta = row_ownership_meta[side].get(rep.active_id, {})
                    component_meta = final_component_meta[side].get(rep.active_id, {})
                    rep_age_bars = i - rep.age_anchor_idx
                    debug_active_members.append(
                        {
                            "row_idx": int(i),
                            "timestamp": ts.iloc[i],
                            "side": side,
                            "active_id": int(rep.active_id),
                            "detect_idx": int(rep.detect_idx),
                            "age_anchor_idx": int(rep.age_anchor_idx),
                            "active_since_idx": int(rep.active_since_idx),
                            "low": float(rep.low),
                            "high": float(rep.high),
                            "width": float(rep.width),
                            "width_atr": float(rep.width_atr),
                            "base_quality": float(rep.base_quality),
                            "age_bars": int(rep_age_bars),
                            "age_decay": float(_age_decay(rep_age_bars)),
                            "effective_significance": float(
                                _effective_significance(
                                    close_value=c[i],
                                    atr_value=atr[i],
                                    current_idx=i,
                                    rep=rep,
                                )
                            ),
                            "source_count": int(rep.source_count),
                            "source_event_id": int(
                                rep.source_ids[0] if rep.source_count == 1 else 0
                            ),
                            "touch_count": int(rep.touch_count),
                            "fill_pct": float(rep.max_fill_pct),
                            "state": int(rep.state),
                            "historical_touch_count": int(rep.historical_touch_count),
                            "historical_fill_pct": float(rep.historical_fill_pct),
                            "historical_state": int(
                                _fill_state_from_metrics(
                                    rep.historical_fill_pct,
                                    rep.historical_touch_count,
                                )
                            ),
                            "first_touch_idx": rep.first_touch_idx,
                            "last_touch_idx": rep.last_touch_idx,
                            "first_partial_fill_idx": rep.first_partial_fill_idx,
                            "historical_first_touch_idx": rep.historical_first_touch_idx,
                            "historical_last_touch_idx": rep.historical_last_touch_idx,
                            "historical_first_partial_fill_idx": rep.historical_first_partial_fill_idx,
                            "max_single_bar_fill_jump": float(
                                rep.max_single_bar_fill_jump
                            ),
                            "historical_max_single_bar_fill_jump": float(
                                rep.historical_max_single_bar_fill_jump
                            ),
                            "all_touch_bars_closed_outside": int(
                                rep.all_touch_bars_closed_outside
                            ),
                            "historical_all_touch_bars_closed_outside": int(
                                rep.historical_all_touch_bars_closed_outside
                            ),
                            "last_touch_rejection_flag": (
                                float(rep.last_touch_rejection_flag)
                                if rep.last_touch_rejection_flag is not None
                                else np.nan
                            ),
                            "historical_last_touch_rejection_flag": (
                                float(rep.historical_last_touch_rejection_flag)
                                if rep.historical_last_touch_rejection_flag is not None
                                else np.nan
                            ),
                            "bounce_raw_max": float(rep.bounce_raw_max),
                            "historical_bounce_raw_max": float(
                                rep.historical_bounce_raw_max
                            ),
                            "component_id": int(
                                component_meta.get(
                                    "component_id",
                                    ownership_meta.get("component_id", 0),
                                )
                            ),
                            "component_low": float(
                                component_meta.get(
                                    "component_low",
                                    ownership_meta.get("component_low", np.nan),
                                )
                            ),
                            "component_high": float(
                                component_meta.get(
                                    "component_high",
                                    ownership_meta.get("component_high", np.nan),
                                )
                            ),
                            "component_member_count": int(
                                component_meta.get(
                                    "component_member_count",
                                    ownership_meta.get("component_member_count", 1),
                                )
                            ),
                            "component_entered_this_row": int(
                                ownership_meta.get("component_entered_this_row", 0)
                            ),
                            "component_owned_this_row": int(
                                ownership_meta.get("component_owned_this_row", 0)
                            ),
                            "touch_delta": int(ownership_meta.get("touch_delta", 0)),
                            "fill_delta": float(ownership_meta.get("fill_delta", 0.0)),
                            "entered_this_row": int(
                                ownership_meta.get("entered_this_row", 0)
                            ),
                            "ownership_blocked_by_higher_component": int(
                                ownership_meta.get(
                                    "ownership_blocked_by_higher_component", 0
                                )
                            ),
                            "ownership_blocked_by_lower_component": int(
                                ownership_meta.get(
                                    "ownership_blocked_by_lower_component", 0
                                )
                            ),
                            "selected": int(rep.active_id == selected_id),
                        }
                    )
            if selected is None:
                arrays[f"fvg_{side}_zone_invalidated"][i] = int(
                    invalidated_on_row[side]
                )
                continue

            arrays[f"fvg_{side}_active"][i] = 1
            arrays[f"fvg_{side}_active_id"][i] = selected.active_id
            arrays[f"fvg_{side}_active_low"][i] = selected.low
            arrays[f"fvg_{side}_active_high"][i] = selected.high
            arrays[f"fvg_{side}_active_mid"][i] = selected.mid
            arrays[f"fvg_{side}_active_width"][i] = selected.width
            arrays[f"fvg_{side}_active_width_atr"][i] = selected.width_atr
            selected_age_bars = i - selected.age_anchor_idx
            arrays[f"fvg_{side}_active_age"][i] = float(selected_age_bars)
            arrays[f"fvg_{side}_active_age_bars"][i] = float(selected_age_bars)
            arrays[f"fvg_{side}_active_age_decay"][i] = _age_decay(selected_age_bars)
            arrays[f"fvg_{side}_active_base_quality"][i] = selected.base_quality
            arrays[f"fvg_{side}_active_effective_significance"][i] = (
                _effective_significance(
                    close_value=c[i],
                    atr_value=atr[i],
                    current_idx=i,
                    rep=selected,
                )
            )
            arrays[f"fvg_{side}_active_fill_pct"][i] = selected.max_fill_pct
            arrays[f"fvg_{side}_active_touch_count"][i] = selected.touch_count
            arrays[f"fvg_{side}_active_state"][i] = selected.state
            arrays[f"fvg_{side}_active_since_idx"][i] = float(selected.active_since_idx)

            first_retest_flag = int(
                selected.state == STATE_ACTIVE_TOUCHED
                and selected.touch_count == 1
                and selected.detect_idx < i
                and not selected.first_retest_emitted
            )
            arrays[f"fvg_{side}_first_retest_flag"][i] = first_retest_flag
            if first_retest_flag == 1:
                selected.first_retest_emitted = True

            arrays[f"fvg_{side}_zone_invalidated"][i] = int(invalidated_on_row[side])

            if np.isfinite(atr[i]) and atr[i] > 0:
                arrays[f"fvg_{side}_distance_to_mid_atr"][i] = (
                    abs(c[i] - selected.mid) / atr[i]
                )
                arrays[f"fvg_{side}_distance_to_near_edge_atr"][i] = (
                    _distance_to_near_edge(c[i], selected.low, selected.high) / atr[i]
                )
                arrays[f"fvg_{side}_distance_to_far_edge_atr"][i] = (
                    _distance_to_far_edge(c[i], selected.low, selected.high) / atr[i]
                )

    core_df = pd.DataFrame(arrays, index=out.index)

    compat_size_atr = np.where(
        np.isfinite(arrays["fvg_bull_width_atr"]),
        arrays["fvg_bull_width_atr"],
        arrays["fvg_bear_width_atr"],
    )
    both_mask = np.isfinite(arrays["fvg_bull_width_atr"]) & np.isfinite(
        arrays["fvg_bear_width_atr"]
    )
    compat_size_atr = np.asarray(compat_size_atr, dtype=float)
    compat_size_atr[both_mask] = np.maximum(
        np.asarray(arrays["fvg_bull_width_atr"], dtype=float)[both_mask],
        np.asarray(arrays["fvg_bear_width_atr"], dtype=float)[both_mask],
    )

    compat_df = pd.DataFrame(
        {
            "fvg_bull": arrays["fvg_bull_detect_flag"],
            "fvg_bear": arrays["fvg_bear_detect_flag"],
            "fvg_size_atr": compat_size_atr,
            "fvg_bull_confirm_idx": np.where(
                arrays["fvg_bull_detect_flag"] == 1,
                arrays["fvg_bull_detect_idx"],
                -1,
            ).astype(int),
            "fvg_bear_confirm_idx": np.where(
                arrays["fvg_bear_detect_flag"] == 1,
                arrays["fvg_bear_detect_idx"],
                -1,
            ).astype(int),
            "fvg_confirm_delay": np.where(
                (arrays["fvg_bull_detect_flag"] == 1)
                | (arrays["fvg_bear_detect_flag"] == 1),
                1,
                np.nan,
            ),
            "fvg_mid_body_atr": np.where(
                np.isfinite(arrays["fvg_bull_mid_body_atr"]),
                arrays["fvg_bull_mid_body_atr"],
                arrays["fvg_bear_mid_body_atr"],
            ),
            "fvg_bos_bull": np.zeros(n, dtype=np.int8),
            "fvg_bos_bear": np.zeros(n, dtype=np.int8),
        },
        index=out.index,
    )

    result = pd.concat([out, core_df, compat_df], axis=1)

    if not collect_debug:
        return result

    for rep_list in active_reps.values():
        for rep in rep_list:
            _finish_rep_debug(rep, n - 1, "unresolved")

    event_records: list[dict[str, object]] = []
    for event in raw_events.values():
        if event.terminal_state == STATE_FULLY_FILLED:
            terminal_idx = event.full_fill_idx
        elif event.terminal_state == STATE_INVALIDATED:
            terminal_idx = event.invalidation_idx
        elif event.terminal_state == STATE_EXPIRED:
            terminal_idx = event.expiry_idx
        elif event.terminal_state == STATE_MERGED:
            terminal_idx = event.merge_idx
        else:
            terminal_idx = np.nan

        event_records.append(
            {
                "event_id": int(event.event_id),
                "side": event.side,
                "origin_idx": int(event.origin_idx),
                "detect_idx": int(event.detect_idx),
                "low": float(event.low),
                "high": float(event.high),
                "mid": float(event.mid),
                "width": float(event.width),
                "width_atr": float(event.width_atr),
                "pattern_mode": int(event.pattern_mode),
                "base_quality": float(event.base_quality),
                "first_touch_idx": event.first_touch_idx,
                "first_partial_fill_idx": event.first_partial_fill_idx,
                "full_fill_idx": event.full_fill_idx,
                "invalidation_idx": event.invalidation_idx,
                "expiry_idx": event.expiry_idx,
                "merge_idx": event.merge_idx,
                "terminal_idx": terminal_idx,
                "terminal_state": int(event.terminal_state),
            }
        )

    event_table = pd.DataFrame(event_records)
    if not event_table.empty:
        event_table = event_table.sort_values(["side", "detect_idx", "event_id"])

    active_rep_table = pd.DataFrame(debug_rep_records.values())
    if not active_rep_table.empty:
        active_rep_table = active_rep_table.sort_values(
            ["side", "start_idx", "active_id"]
        )

    active_member_table = pd.DataFrame(debug_active_members)
    if not active_member_table.empty:
        active_member_table = active_member_table.sort_values(
            ["row_idx", "side", "active_id"]
        )

    debug = {
        "event_table": event_table,
        "active_rep_table": active_rep_table,
        "active_member_table": active_member_table,
    }
    return result, debug


def add_fvg(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_width_atr: float = 0.1,
    pattern_mode: str = "wick",
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
    selector: str = "significance",
) -> pd.DataFrame:
    """Detect canonical causal FVG events and dense active state."""
    return _run_fvg(
        df,
        atr_length=atr_length,
        min_width_atr=min_width_atr,
        pattern_mode=pattern_mode,
        max_active_age_bars=max_active_age_bars,
        selector=selector,
        collect_debug=False,
    )


def collect_fvg_debug_tables(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_width_atr: float = 0.1,
    pattern_mode: str = "wick",
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
    selector: str = "significance",
) -> dict[str, pd.DataFrame]:
    result, debug = _run_fvg(
        df,
        atr_length=atr_length,
        min_width_atr=min_width_atr,
        pattern_mode=pattern_mode,
        max_active_age_bars=max_active_age_bars,
        selector=selector,
        collect_debug=True,
    )
    debug["frame"] = result
    return debug
