"""
smc/equal_hl.py

Canonical EQH/EQL liquidity-cluster engine.

Contract
--------
- Equal highs/lows are active-state liquidity clusters, not one-bar markers.
- Only confirmed causal swings may create or update clusters.
- A cluster becomes live only when it first reaches ``min_touches``.
- Active state begins on the detect row and never appears before it.
- Swept clusters stay exported on the sweep row, then deactivate from the next row.
- Multiple internal clusters may coexist, but the dense export selects one current
  best cluster per side on every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.event_ids import SequentialEventIdAllocator

EQHL_STATE_NONE = 0
EQHL_STATE_FORMING = 1
EQHL_STATE_ACTIVE = 2
EQHL_STATE_SWEPT = 3
EQHL_STATE_INVALIDATED = 4
EQHL_STATE_EXPIRED = 5
EQHL_STATE_MERGED = 6

EQHL_SIDE_HIGH = -1
EQHL_SIDE_LOW = 1

EQHL_REQUIRED_COLUMNS = {
    "high",
    "low",
    "close",
    "swing_high_confirm_flag",
    "swing_low_confirm_flag",
    "swing_high_confirm_price",
    "swing_low_confirm_price",
    "swing_high_confirm_origin_idx",
    "swing_low_confirm_origin_idx",
    "last_swing_high",
    "last_swing_low",
    "last_swing_high_idx",
    "last_swing_low_idx",
}

EQHL_DETECT_COLUMNS = [
    "eqh_detect_flag",
    "eqh_detect_idx",
    "eqh_event_id",
    "eqh_cluster_id_on_detect",
    "eqh_origin_idx",
    "eqh_first_member_detect_idx",
    "eqh_last_member_detect_idx",
    "eqh_last_member_origin_idx",
    "eqh_member_count_on_detect",
    "eqh_level_on_detect",
    "eqh_low_on_detect",
    "eqh_high_on_detect",
    "eqh_width_on_detect",
    "eqh_width_atr_on_detect",
    "eqh_width_component_on_detect",
    "eqh_formation_delay_component_on_detect",
    "eqh_structural_score_on_detect",
    "eqh_tradeable_live_score_on_detect",
    "eqh_score_on_detect",
    "eqh_formation_delay_on_detect",
    "eql_detect_flag",
    "eql_detect_idx",
    "eql_event_id",
    "eql_cluster_id_on_detect",
    "eql_origin_idx",
    "eql_first_member_detect_idx",
    "eql_last_member_detect_idx",
    "eql_last_member_origin_idx",
    "eql_member_count_on_detect",
    "eql_level_on_detect",
    "eql_low_on_detect",
    "eql_high_on_detect",
    "eql_width_on_detect",
    "eql_width_atr_on_detect",
    "eql_width_component_on_detect",
    "eql_formation_delay_component_on_detect",
    "eql_structural_score_on_detect",
    "eql_tradeable_live_score_on_detect",
    "eql_score_on_detect",
    "eql_formation_delay_on_detect",
]

EQHL_ACTIVE_COLUMNS = [
    "eqh_active",
    "eqh_active_id",
    "eqh_active_level",
    "eqh_active_low",
    "eqh_active_high",
    "eqh_active_width",
    "eqh_active_width_atr",
    "eqh_active_touch_count",
    "eqh_active_width_component",
    "eqh_active_formation_delay_component",
    "eqh_active_structural_score",
    "eqh_active_tradeable_live_score",
    "eqh_active_score",
    "eqh_active_age",
    "eqh_active_formation_delay",
    "eqh_active_distance_atr",
    "eqh_active_since_idx",
    "eqh_active_swept",
    "eqh_active_state",
    "eqh_active_count",
    "eql_active",
    "eql_active_id",
    "eql_active_level",
    "eql_active_low",
    "eql_active_high",
    "eql_active_width",
    "eql_active_width_atr",
    "eql_active_touch_count",
    "eql_active_width_component",
    "eql_active_formation_delay_component",
    "eql_active_structural_score",
    "eql_active_tradeable_live_score",
    "eql_active_score",
    "eql_active_age",
    "eql_active_formation_delay",
    "eql_active_distance_atr",
    "eql_active_since_idx",
    "eql_active_swept",
    "eql_active_state",
    "eql_active_count",
    "eqh_active_bars_since_touch",
    "eqh_active_touches_since_detect",
    "eqh_active_close_tests",
    "eqh_active_max_pen_atr",
    "eqh_active_far_edge_atr",
    "eqh_active_inside_zone",
    "eqh_active_signed_dist_atr",
    "eqh_active_same_side_count_nearby",
    "eqh_active_nearest_same_side_atr",
    "eql_active_bars_since_touch",
    "eql_active_touches_since_detect",
    "eql_active_close_tests",
    "eql_active_max_pen_atr",
    "eql_active_far_edge_atr",
    "eql_active_inside_zone",
    "eql_active_signed_dist_atr",
    "eql_active_same_side_count_nearby",
    "eql_active_nearest_same_side_atr",
]

EQHL_RANKED_ACTIVE_COLUMNS = [
    *(
        f"{prefix}_rank{rank}_active_{field}"
        for prefix in ("eqh", "eql")
        for rank in (1, 2)
        for field in (
            "id",
            "level",
            "touch_count",
            "width_atr",
            "structural_score",
            "tradeable_live_score",
            "distance_atr",
        )
    )
]

EQHL_LIFECYCLE_COLUMNS = [
    "eqh_swept_flag",
    "eqh_swept_idx",
    "eqh_swept_cluster_id",
    "eqh_swept_level",
    "eqh_inactive_idx",
    "eql_swept_flag",
    "eql_swept_idx",
    "eql_swept_cluster_id",
    "eql_swept_level",
    "eql_inactive_idx",
]

EQHL_DIAGNOSTIC_COLUMNS = [
    "eqh_pass_source_ready",
    "eqh_pass_tolerance",
    "eqh_pass_width_cap",
    "eqh_pass_span_cap",
    "eqh_cluster_merge_flag",
    "eql_pass_source_ready",
    "eql_pass_tolerance",
    "eql_pass_width_cap",
    "eql_pass_span_cap",
    "eql_cluster_merge_flag",
]

EQHL_LEGACY_COLUMNS = [
    "equal_highs",
    "equal_lows",
    "equal_highs_count",
    "equal_lows_count",
    "equal_highs_level",
    "equal_lows_level",
    "equal_highs_width",
    "equal_lows_width",
    "equal_highs_width_atr",
    "equal_lows_width_atr",
    "equal_highs_age",
    "equal_lows_age",
    "equal_highs_active",
    "equal_lows_active",
    "equal_highs_score",
    "equal_lows_score",
    "equal_highs_structural_score",
    "equal_lows_structural_score",
    "equal_highs_tradeable_live_score",
    "equal_lows_tradeable_live_score",
    "equal_highs_cluster_id",
    "equal_lows_cluster_id",
]

# Combined canonical live-safe column set — used by tests and parity checks
EQHL_LIVE_COLUMNS = (
    EQHL_DETECT_COLUMNS
    + EQHL_ACTIVE_COLUMNS
    + EQHL_RANKED_ACTIVE_COLUMNS
    + EQHL_LIFECYCLE_COLUMNS
    + EQHL_DIAGNOSTIC_COLUMNS
)

SOURCE_STRENGTH_REF = 1.0
MAX_WIDTH_ATR_REF = 1.0
SPAN_CAP_DEFAULT = 200
AGE_CAP_DEFAULT = 200
TRADEABLE_WIDTH_REF = 0.25
# H4-calibrated: penalises slow formation across real distribution (was 25, too tight)
TRADEABLE_FORM_REF = 60.0
# EQH = bearish liquidity pool (side = -1); EQL = bullish (side = +1) — SMC direction convention
EQHL_SELECTOR_MODE_CALIBRATION = "structural_first"
EPS = 1e-12

# ── Score weights — calibrated on XAU/USD H4 via search_equal_hl_tradeable.
# Update these after re-running calibration on new data. Signs are baked in:
#   formation_delay and width use their raw ATR-normalised value (not 1−x),
#   because the search found sign=inverted on the 1−x component form.
EQHL_V2_W_WICK_RATIO = 0.389  # rejection wick at last formation swing
EQHL_V2_W_FORMATION_DELAY = 0.333  # longer formation = more liquidity build-up
EQHL_V2_W_WIDTH = 0.111  # wider zone = more resting orders
EQHL_V2_W_DISTANCE = 0.111  # price further from zone = cleaner approach
EQHL_V2_W_ATR_PCT = 0.056  # lower ATR percentile = quieter regime
EQHL_V2_DISTANCE_CAP_ATR = 5.0  # distance cap for normalisation
ATR_PCT_ROLLING_WINDOW = 50  # rolling window for ATR percentile


@dataclass
class _Cluster:
    cluster_id: int
    side: int
    member_prices: list[float] = field(default_factory=list)
    member_origin_indices: list[int] = field(default_factory=list)
    member_detect_indices: list[int] = field(default_factory=list)
    member_strengths: list[float] = field(default_factory=list)
    first_member_origin_idx: int = -1
    first_member_detect_idx: int = -1
    last_member_origin_idx: int = -1
    last_member_detect_idx: int = -1
    detect_idx: int | None = None
    event_id: int | None = None
    state: int = EQHL_STATE_FORMING
    swept_flag: int = 0
    swept_idx: int | None = None
    inactive_idx: int | None = None
    merged_idx: int | None = None
    merged_into_id: int | None = None
    level: float = np.nan
    width: float = np.nan
    low_bound: float = np.nan
    high_bound: float = np.nan
    touch_count: int = 0
    structural_score: float = np.nan
    tradeable_live_score: float = np.nan
    score: float = np.nan
    span: int = 0
    wick_ratio_at_detect: float = np.nan
    age: int = 0
    formation_delay: int = 0
    selected_active_flag: int = 0
    # Pressure tracking — updated each bar while ACTIVE
    bars_since_last_touch: int = 0
    touches_since_detect: int = 0
    close_tests_since_detect: int = 0
    max_penetration_since_detect_atr: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            EQHL_STATE_SWEPT,
            EQHL_STATE_INVALIDATED,
            EQHL_STATE_EXPIRED,
            EQHL_STATE_MERGED,
        }

    @property
    def is_active_state(self) -> bool:
        return self.state == EQHL_STATE_ACTIVE


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _compute_wick_ratio(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    origin_idx: int,
    side: int,
) -> float:
    """Rejection-wick / total-range at the formation swing bar (origin_idx).

    EQH (side=-1): upper wick = high - max(open, close)
    EQL (side=+1): lower wick = min(open, close) - low
    Returns NaN when origin_idx is invalid or range is zero.
    """
    if origin_idx < 0 or origin_idx >= len(high):
        return np.nan
    h = float(high[origin_idx])
    l = float(low[origin_idx])
    o = float(open_[origin_idx])
    c = float(close[origin_idx])
    rng = h - l
    if rng < EPS:
        return np.nan
    if side == EQHL_SIDE_HIGH:
        wick = h - max(o, c)
    else:
        wick = min(o, c) - l
    return float(np.clip(wick / rng, 0.0, 1.0))


def _v2_score(
    *,
    width_atr: float,
    formation_delay: float,
    wick_ratio: float,
    atr_pct: float,
    distance_atr: float,
) -> float:
    """Calibrated v2 confidence score using XAU/USD H4 weights.

    All inputs are normalised to [0, 1] before weighting. Missing/NaN inputs
    fall back to neutral (0.5) so the score degrades gracefully.
    """
    s_width = _clip01(_safe_div(width_atr, TRADEABLE_WIDTH_REF))
    s_form = _clip01(_safe_div(formation_delay, TRADEABLE_FORM_REF))
    s_wick = _clip01(wick_ratio) if np.isfinite(wick_ratio) else 0.5
    s_atr_quiet = _clip01(1.0 - atr_pct) if np.isfinite(atr_pct) else 0.5
    s_dist = _clip01(_safe_div(distance_atr, EQHL_V2_DISTANCE_CAP_ATR))
    return float(
        EQHL_V2_W_WIDTH * s_width
        + EQHL_V2_W_FORMATION_DELAY * s_form
        + EQHL_V2_W_WICK_RATIO * s_wick
        + EQHL_V2_W_ATR_PCT * s_atr_quiet
        + EQHL_V2_W_DISTANCE * s_dist
    )


def _require_inputs(out: pd.DataFrame) -> None:
    missing = EQHL_REQUIRED_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(f"add_equal_hl: missing required columns: {sorted(missing)}")


def _strength_for_origin(
    out: pd.DataFrame,
    *,
    side: int,
    origin_idx: int,
) -> float:
    if side == EQHL_SIDE_HIGH:
        for col in ("swing_high_strength", "swing_high_prominence_atr"):
            if col in out.columns and 0 <= origin_idx < len(out):
                value = float(out.iloc[origin_idx][col])
                if np.isfinite(value):
                    return value
    else:
        for col in ("swing_low_strength", "swing_low_prominence_atr"):
            if col in out.columns and 0 <= origin_idx < len(out):
                value = float(out.iloc[origin_idx][col])
                if np.isfinite(value):
                    return value
    return np.nan


def _recompute_cluster(
    cluster: _Cluster,
    *,
    current_idx: int,
    atr_value: float,
    min_touches: int,
    max_cluster_width_atr: float | None,
    max_cluster_span: int | None,
    max_active_age: int | None,
) -> None:
    prices = np.asarray(cluster.member_prices, dtype=float)
    strengths = np.asarray(cluster.member_strengths, dtype=float)

    cluster.touch_count = int(len(prices))
    cluster.level = float(np.median(prices)) if len(prices) else np.nan
    cluster.low_bound = float(np.min(prices)) if len(prices) else np.nan
    cluster.high_bound = float(np.max(prices)) if len(prices) else np.nan
    cluster.width = (
        float(cluster.high_bound - cluster.low_bound) if len(prices) else np.nan
    )
    cluster.span = max(
        int(cluster.last_member_detect_idx - cluster.first_member_detect_idx),
        0,
    )
    cluster.age = (
        int(current_idx - cluster.detect_idx)
        if cluster.detect_idx is not None
        else max(int(current_idx - cluster.last_member_detect_idx), 0)
    )

    width_atr = _safe_div(cluster.width, atr_value)
    s_touches = _clip01((cluster.touch_count - min_touches + 1) / 4.0)
    width_ref = (
        float(max_cluster_width_atr)
        if max_cluster_width_atr is not None
        else MAX_WIDTH_ATR_REF
    )
    s_compact = _clip01(1.0 - _safe_div(width_atr, width_ref))
    span_cap = (
        float(max_cluster_span)
        if max_cluster_span is not None
        else float(SPAN_CAP_DEFAULT)
    )
    s_span = _clip01(1.0 - _safe_div(float(cluster.span), span_cap))

    finite_strengths = strengths[np.isfinite(strengths)]
    if len(finite_strengths):
        avg_strength = float(np.mean(finite_strengths))
        s_source = _clip01(avg_strength / SOURCE_STRENGTH_REF)
        cluster.structural_score = float(
            0.35 * s_touches + 0.30 * s_compact + 0.20 * s_span + 0.15 * s_source
        )
    else:
        cluster.structural_score = float(
            (0.35 * s_touches + 0.30 * s_compact + 0.20 * s_span) / 0.85
        )
    formation_delay = (
        max(int(cluster.detect_idx - cluster.first_member_detect_idx), 0)
        if cluster.detect_idx is not None
        else max(int(current_idx - cluster.first_member_detect_idx), 0)
    )
    cluster.formation_delay = formation_delay
    # tradeable_live_score is set externally after _recompute_cluster using _v2_score,
    # which requires wick_ratio_at_detect, atr_pct, and distance_atr not available here.


def _simulate_updated_metrics(
    cluster: _Cluster,
    *,
    price: float,
    detect_idx: int,
    atr_value: float,
) -> tuple[float, float, int]:
    prices = np.asarray([*cluster.member_prices, price], dtype=float)
    low_bound = float(np.min(prices))
    high_bound = float(np.max(prices))
    width = float(high_bound - low_bound)
    width_atr = _safe_div(width, atr_value)
    span = int(detect_idx - cluster.first_member_detect_idx)
    return width_atr, width, span


def _candidate_sort_key(
    cluster: _Cluster,
    *,
    price: float,
    atr_value: float,
) -> tuple[float, int, int, float, float, int]:
    width_atr = _safe_div(cluster.width, atr_value)
    return (
        abs(cluster.level - price) if np.isfinite(cluster.level) else np.inf,
        -cluster.touch_count,
        -(cluster.last_member_detect_idx),
        width_atr if np.isfinite(width_atr) else np.inf,
        -(
            cluster.structural_score
            if np.isfinite(cluster.structural_score)
            else -np.inf
        ),
        cluster.cluster_id,
    )


def _distance_to_source_atr(
    *,
    close_value: float,
    source_low: float,
    source_high: float,
    source_level: float,
    atr_value: float,
) -> float:
    if (
        not np.isfinite(close_value)
        or not np.isfinite(source_level)
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return np.inf
    if np.isfinite(source_low) and np.isfinite(source_high):
        if source_low <= close_value <= source_high:
            return 0.0
        if close_value < source_low:
            return float((source_low - close_value) / atr_value)
        if close_value > source_high:
            return float((close_value - source_high) / atr_value)
    return float(abs(close_value - source_level) / atr_value)


def _distance_to_far_edge_atr(
    *,
    close_value: float,
    source_low: float,
    source_high: float,
    atr_value: float,
) -> float:
    if (
        not np.isfinite(close_value)
        or not np.isfinite(source_low)
        or not np.isfinite(source_high)
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return np.nan
    if close_value < source_low:
        return float((source_high - close_value) / atr_value)
    if close_value > source_high:
        return float((close_value - source_low) / atr_value)
    dist_low = float((close_value - source_low) / atr_value)
    dist_high = float((source_high - close_value) / atr_value)
    return float(max(dist_low, dist_high))


def _signed_distance_side_aware_atr(
    *,
    close_value: float,
    cluster_level: float,
    cluster_side: int,
    atr_value: float,
) -> float:
    if (
        not np.isfinite(close_value)
        or not np.isfinite(cluster_level)
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return np.nan
    raw = float((cluster_level - close_value) / atr_value)
    return raw if cluster_side == EQHL_SIDE_HIGH else -raw


def _update_cluster_pressure(
    cluster: _Cluster,
    *,
    high_i: float,
    low_i: float,
    close_i: float,
    atr_i: float,
) -> None:
    if not np.isfinite(cluster.low_bound) or not np.isfinite(cluster.high_bound):
        return
    if cluster.side == EQHL_SIDE_HIGH:
        wick = high_i
    else:
        wick = low_i
    in_band = cluster.low_bound <= wick <= cluster.high_bound
    close_in_band = cluster.low_bound <= close_i <= cluster.high_bound
    if in_band and atr_i > 0:
        cluster.bars_since_last_touch = 0
        cluster.touches_since_detect += 1
        if cluster.side == EQHL_SIDE_HIGH:
            penetration = max(0.0, wick - cluster.low_bound)
        else:
            penetration = max(0.0, cluster.high_bound - wick)
        pen_atr = float(penetration / atr_i)
        if pen_atr > cluster.max_penetration_since_detect_atr:
            cluster.max_penetration_since_detect_atr = pen_atr
    else:
        cluster.bars_since_last_touch += 1
    if close_in_band:
        cluster.close_tests_since_detect += 1


def _current_best_key(
    cluster: _Cluster,
    *,
    close_value: float,
    atr_value: float,
) -> tuple[float, float, float, int, float, int, int]:
    width_atr = _safe_div(cluster.width, atr_value)
    distance_atr = _distance_to_source_atr(
        close_value=close_value,
        source_low=cluster.low_bound,
        source_high=cluster.high_bound,
        source_level=cluster.level,
        atr_value=atr_value,
    )
    return (
        -(
            cluster.structural_score
            if np.isfinite(cluster.structural_score)
            else -np.inf
        ),
        distance_atr,
        -cluster.touch_count,
        width_atr if np.isfinite(width_atr) else np.inf,
        -(cluster.detect_idx if cluster.detect_idx is not None else -1),
        -(
            cluster.tradeable_live_score
            if np.isfinite(cluster.tradeable_live_score)
            else -np.inf
        ),
        cluster.cluster_id,
    )


def _cluster_is_exportable(cluster: _Cluster, idx: int) -> bool:
    if cluster.state == EQHL_STATE_ACTIVE:
        return cluster.detect_idx is not None and idx >= cluster.detect_idx
    return cluster.state == EQHL_STATE_SWEPT and cluster.swept_idx == idx


def _append_member(
    cluster: _Cluster,
    *,
    price: float,
    origin_idx: int,
    detect_idx: int,
    source_strength: float,
    atr_value: float,
    min_touches: int,
    max_cluster_width_atr: float | None,
    max_cluster_span: int | None,
    max_active_age: int | None,
) -> None:
    cluster.member_prices.append(float(price))
    cluster.member_origin_indices.append(int(origin_idx))
    cluster.member_detect_indices.append(int(detect_idx))
    cluster.member_strengths.append(float(source_strength))
    cluster.last_member_origin_idx = int(origin_idx)
    cluster.last_member_detect_idx = int(detect_idx)
    if cluster.first_member_origin_idx < 0:
        cluster.first_member_origin_idx = int(origin_idx)
        cluster.first_member_detect_idx = int(detect_idx)
    _recompute_cluster(
        cluster,
        current_idx=detect_idx,
        atr_value=atr_value,
        min_touches=min_touches,
        max_cluster_width_atr=max_cluster_width_atr,
        max_cluster_span=max_cluster_span,
        max_active_age=max_active_age,
    )


def _merge_clusters(
    primary: _Cluster,
    secondaries: list[_Cluster],
    *,
    current_idx: int,
    atr_value: float,
    min_touches: int,
    max_cluster_width_atr: float | None,
    max_cluster_span: int | None,
    max_active_age: int | None,
) -> None:
    for secondary in secondaries:
        primary.member_prices.extend(secondary.member_prices)
        primary.member_origin_indices.extend(secondary.member_origin_indices)
        primary.member_detect_indices.extend(secondary.member_detect_indices)
        primary.member_strengths.extend(secondary.member_strengths)
        primary.first_member_origin_idx = min(
            primary.first_member_origin_idx, secondary.first_member_origin_idx
        )
        primary.first_member_detect_idx = min(
            primary.first_member_detect_idx, secondary.first_member_detect_idx
        )
        primary.last_member_origin_idx = max(
            primary.last_member_origin_idx, secondary.last_member_origin_idx
        )
        primary.last_member_detect_idx = max(
            primary.last_member_detect_idx, secondary.last_member_detect_idx
        )
        secondary.state = EQHL_STATE_MERGED
        secondary.inactive_idx = current_idx
        secondary.merged_idx = current_idx
        secondary.merged_into_id = primary.cluster_id
    _recompute_cluster(
        primary,
        current_idx=current_idx,
        atr_value=atr_value,
        min_touches=min_touches,
        max_cluster_width_atr=max_cluster_width_atr,
        max_cluster_span=max_cluster_span,
        max_active_age=max_active_age,
    )


def add_equal_hl(
    df: pd.DataFrame,
    atr_tolerance: float = 0.12,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    use_provided_swings: bool = True,
    lookback_swings: int = 50,
    min_touches: int = 2,
    level_mode: str = "median",
    max_cluster_width_atr: float | None = 0.5,
    max_cluster_span: int | None = 120,
    invalidate_on_sweep: bool = False,
    sweep_tolerance_atr: float = 0.0,
    keep_last_n_clusters: int = 20,
    max_active_age: int | None = 200,
) -> pd.DataFrame:
    """Append canonical EQH/EQL cluster state and legacy aliases."""
    del swing_left, swing_right, invalidate_on_sweep, sweep_tolerance_atr

    if level_mode != "median":
        raise ValueError("level_mode is frozen to 'median'")
    if not use_provided_swings:
        raise ValueError(
            "EQH/EQL requires confirmed swing inputs; pivot fallback is forbidden"
        )
    if min_touches < 2:
        raise ValueError("min_touches must be >= 2")
    if atr_tolerance < 0:
        raise ValueError("atr_tolerance must be >= 0")
    if max_cluster_width_atr is not None and max_cluster_width_atr < 0:
        raise ValueError("max_cluster_width_atr must be >= 0 or None")
    if max_cluster_span is not None and max_cluster_span < 0:
        raise ValueError("max_cluster_span must be >= 0 or None")
    if max_active_age is not None and max_active_age < 0:
        raise ValueError("max_active_age must be >= 0 or None")

    out = df.copy()
    if len(out) == 0:
        return out

    _require_inputs(out)

    n = len(out)
    atr = get_atr_array(out, atr_length).astype(float, copy=False)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float) if "open" in out.columns else close.copy()
    # Rolling ATR percentile rank (50-bar window) — pre-computed for use in v2 score
    atr_pct = (
        pd.Series(atr)
        .rolling(ATR_PCT_ROLLING_WINDOW, min_periods=10)
        .rank(pct=True)
        .to_numpy(dtype=float)
    )

    sh_confirm_flag = out["swing_high_confirm_flag"].to_numpy(dtype=np.int8)
    sl_confirm_flag = out["swing_low_confirm_flag"].to_numpy(dtype=np.int8)
    sh_confirm_price = out["swing_high_confirm_price"].to_numpy(dtype=float)
    sl_confirm_price = out["swing_low_confirm_price"].to_numpy(dtype=float)
    sh_confirm_origin = out["swing_high_confirm_origin_idx"].to_numpy(dtype=float)
    sl_confirm_origin = out["swing_low_confirm_origin_idx"].to_numpy(dtype=float)

    eqh_detect_flag = np.zeros(n, dtype=np.int8)
    eqh_detect_idx = np.full(n, np.nan, dtype=float)
    eqh_event_id = np.full(n, np.nan, dtype=float)
    eqh_cluster_id_on_detect = np.full(n, np.nan, dtype=float)
    eqh_origin_idx = np.full(n, np.nan, dtype=float)
    eqh_first_member_detect_idx = np.full(n, np.nan, dtype=float)
    eqh_last_member_detect_idx = np.full(n, np.nan, dtype=float)
    eqh_last_member_origin_idx = np.full(n, np.nan, dtype=float)
    eqh_member_count_on_detect = np.full(n, np.nan, dtype=float)
    eqh_level_on_detect = np.full(n, np.nan, dtype=float)
    eqh_low_on_detect = np.full(n, np.nan, dtype=float)
    eqh_high_on_detect = np.full(n, np.nan, dtype=float)
    eqh_width_on_detect = np.full(n, np.nan, dtype=float)
    eqh_width_atr_on_detect = np.full(n, np.nan, dtype=float)
    eqh_width_component_on_detect = np.full(n, np.nan, dtype=float)
    eqh_formation_delay_component_on_detect = np.full(n, np.nan, dtype=float)
    eqh_structural_score_on_detect = np.full(n, np.nan, dtype=float)
    eqh_tradeable_live_score_on_detect = np.full(n, np.nan, dtype=float)
    eqh_score_on_detect = np.full(n, np.nan, dtype=float)
    eqh_formation_delay_on_detect = np.full(n, np.nan, dtype=float)

    eql_detect_flag = np.zeros(n, dtype=np.int8)
    eql_detect_idx = np.full(n, np.nan, dtype=float)
    eql_event_id = np.full(n, np.nan, dtype=float)
    eql_cluster_id_on_detect = np.full(n, np.nan, dtype=float)
    eql_origin_idx = np.full(n, np.nan, dtype=float)
    eql_first_member_detect_idx = np.full(n, np.nan, dtype=float)
    eql_last_member_detect_idx = np.full(n, np.nan, dtype=float)
    eql_last_member_origin_idx = np.full(n, np.nan, dtype=float)
    eql_member_count_on_detect = np.full(n, np.nan, dtype=float)
    eql_level_on_detect = np.full(n, np.nan, dtype=float)
    eql_low_on_detect = np.full(n, np.nan, dtype=float)
    eql_high_on_detect = np.full(n, np.nan, dtype=float)
    eql_width_on_detect = np.full(n, np.nan, dtype=float)
    eql_width_atr_on_detect = np.full(n, np.nan, dtype=float)
    eql_width_component_on_detect = np.full(n, np.nan, dtype=float)
    eql_formation_delay_component_on_detect = np.full(n, np.nan, dtype=float)
    eql_structural_score_on_detect = np.full(n, np.nan, dtype=float)
    eql_tradeable_live_score_on_detect = np.full(n, np.nan, dtype=float)
    eql_score_on_detect = np.full(n, np.nan, dtype=float)
    eql_formation_delay_on_detect = np.full(n, np.nan, dtype=float)

    eqh_active = np.zeros(n, dtype=np.int8)
    eqh_active_id = np.full(n, np.nan, dtype=float)
    eqh_active_level = np.full(n, np.nan, dtype=float)
    eqh_active_low = np.full(n, np.nan, dtype=float)
    eqh_active_high = np.full(n, np.nan, dtype=float)
    eqh_active_width = np.full(n, np.nan, dtype=float)
    eqh_active_width_atr = np.full(n, np.nan, dtype=float)
    eqh_active_touch_count = np.full(n, np.nan, dtype=float)
    eqh_active_width_component = np.full(n, np.nan, dtype=float)
    eqh_active_formation_delay_component = np.full(n, np.nan, dtype=float)
    eqh_active_structural_score = np.full(n, np.nan, dtype=float)
    eqh_active_tradeable_live_score = np.full(n, np.nan, dtype=float)
    eqh_active_score = np.full(n, np.nan, dtype=float)
    eqh_active_age = np.full(n, np.nan, dtype=float)
    eqh_active_formation_delay = np.full(n, np.nan, dtype=float)
    eqh_active_distance_atr = np.full(n, np.nan, dtype=float)
    eqh_active_since_idx = np.full(n, np.nan, dtype=float)
    eqh_active_swept = np.zeros(n, dtype=np.int8)
    eqh_active_state = np.zeros(n, dtype=np.int8)
    eqh_active_count = np.zeros(n, dtype=np.int16)

    eql_active = np.zeros(n, dtype=np.int8)
    eql_active_id = np.full(n, np.nan, dtype=float)
    eql_active_level = np.full(n, np.nan, dtype=float)
    eql_active_low = np.full(n, np.nan, dtype=float)
    eql_active_high = np.full(n, np.nan, dtype=float)
    eql_active_width = np.full(n, np.nan, dtype=float)
    eql_active_width_atr = np.full(n, np.nan, dtype=float)
    eql_active_touch_count = np.full(n, np.nan, dtype=float)
    eql_active_width_component = np.full(n, np.nan, dtype=float)
    eql_active_formation_delay_component = np.full(n, np.nan, dtype=float)
    eql_active_structural_score = np.full(n, np.nan, dtype=float)
    eql_active_tradeable_live_score = np.full(n, np.nan, dtype=float)
    eql_active_score = np.full(n, np.nan, dtype=float)
    eql_active_age = np.full(n, np.nan, dtype=float)
    eql_active_formation_delay = np.full(n, np.nan, dtype=float)
    eql_active_distance_atr = np.full(n, np.nan, dtype=float)
    eql_active_since_idx = np.full(n, np.nan, dtype=float)
    eql_active_swept = np.zeros(n, dtype=np.int8)
    eql_active_state = np.zeros(n, dtype=np.int8)
    eql_active_count = np.zeros(n, dtype=np.int16)

    eqh_swept_flag = np.zeros(n, dtype=np.int8)
    eqh_swept_idx = np.full(n, np.nan, dtype=float)
    eqh_swept_cluster_id = np.full(n, np.nan, dtype=float)
    eqh_swept_level = np.full(n, np.nan, dtype=float)
    eqh_inactive_idx = np.full(n, np.nan, dtype=float)
    eql_swept_flag = np.zeros(n, dtype=np.int8)
    eql_swept_idx = np.full(n, np.nan, dtype=float)
    eql_swept_cluster_id = np.full(n, np.nan, dtype=float)
    eql_swept_level = np.full(n, np.nan, dtype=float)
    eql_inactive_idx = np.full(n, np.nan, dtype=float)

    eqh_pass_source_ready = np.zeros(n, dtype=np.int8)
    eqh_pass_tolerance = np.zeros(n, dtype=np.int8)
    eqh_pass_width_cap = np.zeros(n, dtype=np.int8)
    eqh_pass_span_cap = np.zeros(n, dtype=np.int8)
    eqh_cluster_merge_flag = np.zeros(n, dtype=np.int8)
    eql_pass_source_ready = np.zeros(n, dtype=np.int8)
    eql_pass_tolerance = np.zeros(n, dtype=np.int8)
    eql_pass_width_cap = np.zeros(n, dtype=np.int8)
    eql_pass_span_cap = np.zeros(n, dtype=np.int8)
    eql_cluster_merge_flag = np.zeros(n, dtype=np.int8)

    ranked_active_arrays: dict[str, np.ndarray] = {
        f"{prefix}_rank{rank}_active_{field}": np.full(n, np.nan, dtype=float)
        for prefix in ("eqh", "eql")
        for rank in (1, 2)
        for field in (
            "id",
            "level",
            "touch_count",
            "width_atr",
            "structural_score",
            "tradeable_live_score",
            "distance_atr",
        )
    }

    pressure_active_arrays: dict[str, np.ndarray] = {
        **{
            f"{prefix}_active_{field}": np.full(n, np.nan, dtype=float)
            for prefix in ("eqh", "eql")
            for field in (
                "bars_since_touch",
                "touches_since_detect",
                "close_tests",
                "max_pen_atr",
                "far_edge_atr",
                "signed_dist_atr",
                "nearest_same_side_atr",
            )
        },
        **{
            f"{prefix}_active_inside_zone": np.zeros(n, dtype=np.int8)
            for prefix in ("eqh", "eql")
        },
        **{
            f"{prefix}_active_same_side_count_nearby": np.zeros(n, dtype=np.int16)
            for prefix in ("eqh", "eql")
        },
    }

    cluster_id_allocator = SequentialEventIdAllocator()
    event_id_allocator = SequentialEventIdAllocator()
    high_clusters: list[_Cluster] = []
    low_clusters: list[_Cluster] = []

    def _new_cluster(
        side: int,
        price: float,
        origin_idx: int,
        detect_idx: int,
        source_strength: float,
        atr_value: float,
    ) -> _Cluster:
        cluster = _Cluster(cluster_id=cluster_id_allocator.allocate(), side=side)
        _append_member(
            cluster,
            price=price,
            origin_idx=origin_idx,
            detect_idx=detect_idx,
            source_strength=source_strength,
            atr_value=atr_value,
            min_touches=min_touches,
            max_cluster_width_atr=max_cluster_width_atr,
            max_cluster_span=max_cluster_span,
            max_active_age=max_active_age,
        )
        return cluster

    def _eligible_for_matching(cluster: _Cluster, idx: int) -> bool:
        if cluster.is_terminal:
            return False
        return idx - cluster.last_member_detect_idx <= lookback_swings

    for i in range(n):
        atr_i = float(atr[i])

        for cluster in [*high_clusters, *low_clusters]:
            if cluster.state != EQHL_STATE_ACTIVE or cluster.detect_idx is None:
                continue
            if max_active_age is not None and (i - cluster.detect_idx) > max_active_age:
                cluster.state = EQHL_STATE_EXPIRED
                cluster.inactive_idx = i
                if cluster.side == EQHL_SIDE_HIGH:
                    eqh_inactive_idx[i] = float(i)
                else:
                    eql_inactive_idx[i] = float(i)
                continue
            if cluster.side == EQHL_SIDE_HIGH and high[i] > cluster.high_bound:
                cluster.state = EQHL_STATE_SWEPT
                cluster.swept_flag = 1
                cluster.swept_idx = i
                cluster.inactive_idx = i + 1
                if cluster.side == EQHL_SIDE_HIGH:
                    eqh_swept_flag[i] = 1
                    eqh_swept_idx[i] = float(i)
                    eqh_swept_cluster_id[i] = float(cluster.cluster_id)
                    eqh_swept_level[i] = cluster.level
                    eqh_inactive_idx[i] = float(i + 1)
                continue
            if cluster.side == EQHL_SIDE_LOW and low[i] < cluster.low_bound:
                cluster.state = EQHL_STATE_SWEPT
                cluster.swept_flag = 1
                cluster.swept_idx = i
                cluster.inactive_idx = i + 1
                eql_swept_flag[i] = 1
                eql_swept_idx[i] = float(i)
                eql_swept_cluster_id[i] = float(cluster.cluster_id)
                eql_swept_level[i] = cluster.level
                eql_inactive_idx[i] = float(i + 1)
                continue
            if cluster.state == EQHL_STATE_ACTIVE:
                _update_cluster_pressure(
                    cluster,
                    high_i=float(high[i]),
                    low_i=float(low[i]),
                    close_i=float(close[i]),
                    atr_i=atr_i,
                )

        for (
            side,
            confirm_flag,
            confirm_price,
            confirm_origin,
            clusters,
            pass_source_ready,
            pass_tolerance,
            pass_width_cap,
            pass_span_cap,
            merge_flag,
            detect_flag_arr,
            detect_idx_arr,
            event_id_arr,
            cluster_id_on_detect_arr,
            origin_idx_arr,
            first_detect_arr,
            last_detect_arr,
            last_origin_arr,
            member_count_arr,
            level_arr,
            low_arr,
            high_arr,
            width_arr,
            width_atr_arr,
            width_component_arr,
            formation_delay_component_arr,
            structural_score_arr,
            tradeable_score_arr,
            score_arr,
            formation_delay_arr,
        ) in (
            (
                EQHL_SIDE_HIGH,
                sh_confirm_flag,
                sh_confirm_price,
                sh_confirm_origin,
                high_clusters,
                eqh_pass_source_ready,
                eqh_pass_tolerance,
                eqh_pass_width_cap,
                eqh_pass_span_cap,
                eqh_cluster_merge_flag,
                eqh_detect_flag,
                eqh_detect_idx,
                eqh_event_id,
                eqh_cluster_id_on_detect,
                eqh_origin_idx,
                eqh_first_member_detect_idx,
                eqh_last_member_detect_idx,
                eqh_last_member_origin_idx,
                eqh_member_count_on_detect,
                eqh_level_on_detect,
                eqh_low_on_detect,
                eqh_high_on_detect,
                eqh_width_on_detect,
                eqh_width_atr_on_detect,
                eqh_width_component_on_detect,
                eqh_formation_delay_component_on_detect,
                eqh_structural_score_on_detect,
                eqh_tradeable_live_score_on_detect,
                eqh_score_on_detect,
                eqh_formation_delay_on_detect,
            ),
            (
                EQHL_SIDE_LOW,
                sl_confirm_flag,
                sl_confirm_price,
                sl_confirm_origin,
                low_clusters,
                eql_pass_source_ready,
                eql_pass_tolerance,
                eql_pass_width_cap,
                eql_pass_span_cap,
                eql_cluster_merge_flag,
                eql_detect_flag,
                eql_detect_idx,
                eql_event_id,
                eql_cluster_id_on_detect,
                eql_origin_idx,
                eql_first_member_detect_idx,
                eql_last_member_detect_idx,
                eql_last_member_origin_idx,
                eql_member_count_on_detect,
                eql_level_on_detect,
                eql_low_on_detect,
                eql_high_on_detect,
                eql_width_on_detect,
                eql_width_atr_on_detect,
                eql_width_component_on_detect,
                eql_formation_delay_component_on_detect,
                eql_structural_score_on_detect,
                eql_tradeable_live_score_on_detect,
                eql_score_on_detect,
                eql_formation_delay_on_detect,
            ),
        ):
            if int(confirm_flag[i]) != 1:
                continue
            if (
                not np.isfinite(confirm_price[i])
                or not np.isfinite(confirm_origin[i])
                or not np.isfinite(atr_i)
                or atr_i <= 0
            ):
                continue

            price = float(confirm_price[i])
            origin_idx = int(confirm_origin[i])
            source_strength = _strength_for_origin(
                out, side=side, origin_idx=origin_idx
            )
            pass_source_ready[i] = 1

            matching_candidates: list[_Cluster] = []
            width_ok_count = 0
            span_ok_count = 0
            for cluster in clusters:
                if not _eligible_for_matching(cluster, i):
                    continue
                tolerance = atr_tolerance * atr_i
                if (
                    not np.isfinite(cluster.level)
                    or abs(price - cluster.level) > tolerance
                ):
                    continue
                pass_tolerance[i] = 1
                width_atr_candidate, _, span_candidate = _simulate_updated_metrics(
                    cluster,
                    price=price,
                    detect_idx=i,
                    atr_value=atr_i,
                )
                width_ok = (
                    max_cluster_width_atr is None
                    or not np.isfinite(width_atr_candidate)
                    or width_atr_candidate <= max_cluster_width_atr
                )
                span_ok = max_cluster_span is None or span_candidate <= max_cluster_span
                if width_ok:
                    width_ok_count += 1
                if span_ok:
                    span_ok_count += 1
                if width_ok and span_ok:
                    matching_candidates.append(cluster)

            if width_ok_count > 0:
                pass_width_cap[i] = 1
            if span_ok_count > 0:
                pass_span_cap[i] = 1

            if not matching_candidates:
                cluster = _new_cluster(
                    side=side,
                    price=price,
                    origin_idx=origin_idx,
                    detect_idx=i,
                    source_strength=source_strength,
                    atr_value=atr_i,
                )
                clusters.append(cluster)
            else:
                matching_candidates.sort(
                    key=lambda cl: _candidate_sort_key(
                        cl,
                        price=price,
                        atr_value=atr_i,
                    )
                )
                cluster = matching_candidates[0]
                if len(matching_candidates) > 1:
                    _merge_clusters(
                        cluster,
                        matching_candidates[1:],
                        current_idx=i,
                        atr_value=atr_i,
                        min_touches=min_touches,
                        max_cluster_width_atr=max_cluster_width_atr,
                        max_cluster_span=max_cluster_span,
                        max_active_age=max_active_age,
                    )
                    merge_flag[i] = 1

                _append_member(
                    cluster,
                    price=price,
                    origin_idx=origin_idx,
                    detect_idx=i,
                    source_strength=source_strength,
                    atr_value=atr_i,
                    min_touches=min_touches,
                    max_cluster_width_atr=max_cluster_width_atr,
                    max_cluster_span=max_cluster_span,
                    max_active_age=max_active_age,
                )

            if (
                cluster.state == EQHL_STATE_FORMING
                and cluster.touch_count >= min_touches
            ):
                cluster.state = EQHL_STATE_ACTIVE
                cluster.detect_idx = i
                cluster.event_id = event_id_allocator.allocate()
                _recompute_cluster(
                    cluster,
                    current_idx=i,
                    atr_value=atr_i,
                    min_touches=min_touches,
                    max_cluster_width_atr=max_cluster_width_atr,
                    max_cluster_span=max_cluster_span,
                    max_active_age=max_active_age,
                )

                detect_flag_arr[i] = 1
                detect_idx_arr[i] = float(i)
                event_id_arr[i] = float(cluster.event_id)
                cluster_id_on_detect_arr[i] = float(cluster.cluster_id)
                origin_idx_arr[i] = float(cluster.first_member_origin_idx)
                first_detect_arr[i] = float(cluster.first_member_detect_idx)
                last_detect_arr[i] = float(cluster.last_member_detect_idx)
                last_origin_arr[i] = float(cluster.last_member_origin_idx)
                member_count_arr[i] = float(cluster.touch_count)
                level_arr[i] = cluster.level
                low_arr[i] = cluster.low_bound
                high_arr[i] = cluster.high_bound
                width_arr[i] = cluster.width
                _detect_width_atr = _safe_div(cluster.width, atr_i)
                _detect_wick = _compute_wick_ratio(
                    high, low, open_, close, cluster.last_member_origin_idx, side
                )
                cluster.wick_ratio_at_detect = _detect_wick
                _detect_atr_pct = float(atr_pct[i]) if np.isfinite(atr_pct[i]) else 0.5
                _detect_v2 = _v2_score(
                    width_atr=_detect_width_atr,
                    formation_delay=float(cluster.formation_delay),
                    wick_ratio=_detect_wick,
                    atr_pct=_detect_atr_pct,
                    distance_atr=0.0,
                )
                cluster.tradeable_live_score = _detect_v2
                cluster.score = _detect_v2
                width_atr_arr[i] = _detect_width_atr
                width_component_arr[i] = _clip01(
                    _safe_div(_detect_width_atr, TRADEABLE_WIDTH_REF)
                )
                formation_delay_component_arr[i] = _clip01(
                    _safe_div(float(cluster.formation_delay), TRADEABLE_FORM_REF)
                )
                structural_score_arr[i] = cluster.structural_score
                tradeable_score_arr[i] = _detect_v2
                score_arr[i] = _detect_v2
                formation_delay_arr[i] = float(cluster.formation_delay)

        for (
            prefix,
            clusters,
            active_arr,
            active_id_arr,
            active_level_arr,
            active_low_arr,
            active_high_arr,
            active_width_arr,
            active_width_atr_arr,
            active_touch_count_arr,
            active_width_component_arr,
            active_formation_delay_component_arr,
            active_structural_arr,
            active_tradeable_arr,
            active_score_arr,
            active_age_arr,
            active_formation_delay_arr,
            active_distance_arr,
            active_since_arr,
            active_swept_arr,
            active_state_arr,
            active_count_arr,
        ) in (
            (
                "eqh",
                high_clusters,
                eqh_active,
                eqh_active_id,
                eqh_active_level,
                eqh_active_low,
                eqh_active_high,
                eqh_active_width,
                eqh_active_width_atr,
                eqh_active_touch_count,
                eqh_active_width_component,
                eqh_active_formation_delay_component,
                eqh_active_structural_score,
                eqh_active_tradeable_live_score,
                eqh_active_score,
                eqh_active_age,
                eqh_active_formation_delay,
                eqh_active_distance_atr,
                eqh_active_since_idx,
                eqh_active_swept,
                eqh_active_state,
                eqh_active_count,
            ),
            (
                "eql",
                low_clusters,
                eql_active,
                eql_active_id,
                eql_active_level,
                eql_active_low,
                eql_active_high,
                eql_active_width,
                eql_active_width_atr,
                eql_active_touch_count,
                eql_active_width_component,
                eql_active_formation_delay_component,
                eql_active_structural_score,
                eql_active_tradeable_live_score,
                eql_active_score,
                eql_active_age,
                eql_active_formation_delay,
                eql_active_distance_atr,
                eql_active_since_idx,
                eql_active_swept,
                eql_active_state,
                eql_active_count,
            ),
        ):
            candidates = [
                cluster for cluster in clusters if _cluster_is_exportable(cluster, i)
            ]
            active_count_arr[i] = len(candidates)
            if not candidates:
                continue
            for cluster in candidates:
                _recompute_cluster(
                    cluster,
                    current_idx=i,
                    atr_value=atr_i,
                    min_touches=min_touches,
                    max_cluster_width_atr=max_cluster_width_atr,
                    max_cluster_span=max_cluster_span,
                    max_active_age=max_active_age,
                )
            _active_atr_pct = float(atr_pct[i]) if np.isfinite(atr_pct[i]) else 0.5
            for cluster in candidates:
                _cand_width_atr = _safe_div(cluster.width, atr_i)
                _cand_dist_atr = _distance_to_source_atr(
                    close_value=float(close[i]),
                    source_low=cluster.low_bound,
                    source_high=cluster.high_bound,
                    source_level=cluster.level,
                    atr_value=atr_i,
                )
                _cand_v2 = _v2_score(
                    width_atr=_cand_width_atr,
                    formation_delay=float(cluster.formation_delay),
                    wick_ratio=cluster.wick_ratio_at_detect,
                    atr_pct=_active_atr_pct,
                    distance_atr=_cand_dist_atr,
                )
                cluster.tradeable_live_score = _cand_v2
                cluster.score = _cand_v2
            for cluster in candidates:
                cluster.selected_active_flag = 0
            candidates.sort(
                key=lambda cl: _current_best_key(
                    cl,
                    close_value=float(close[i]),
                    atr_value=atr_i,
                )
            )
            for rank, candidate in enumerate(candidates[:2], start=1):
                prefix_rank = f"{prefix}_rank{rank}_active_"
                ranked_active_arrays[f"{prefix_rank}id"][i] = float(
                    candidate.cluster_id
                )
                ranked_active_arrays[f"{prefix_rank}level"][i] = candidate.level
                ranked_active_arrays[f"{prefix_rank}touch_count"][i] = float(
                    candidate.touch_count
                )
                ranked_active_arrays[f"{prefix_rank}width_atr"][i] = _safe_div(
                    candidate.width, atr_i
                )
                ranked_active_arrays[f"{prefix_rank}structural_score"][
                    i
                ] = candidate.structural_score
                ranked_active_arrays[f"{prefix_rank}tradeable_live_score"][
                    i
                ] = candidate.tradeable_live_score
                ranked_active_arrays[f"{prefix_rank}distance_atr"][i] = (
                    _distance_to_source_atr(
                        close_value=float(close[i]),
                        source_low=candidate.low_bound,
                        source_high=candidate.high_bound,
                        source_level=candidate.level,
                        atr_value=atr_i,
                    )
                )
            selected = candidates[0]
            selected.selected_active_flag = 1
            _sel_width_atr = _safe_div(selected.width, atr_i)
            _sel_dist_atr = _distance_to_source_atr(
                close_value=float(close[i]),
                source_low=selected.low_bound,
                source_high=selected.high_bound,
                source_level=selected.level,
                atr_value=atr_i,
            )
            active_arr[i] = 1
            active_id_arr[i] = float(selected.cluster_id)
            active_level_arr[i] = selected.level
            active_low_arr[i] = selected.low_bound
            active_high_arr[i] = selected.high_bound
            active_width_arr[i] = selected.width
            active_width_atr_arr[i] = _sel_width_atr
            active_touch_count_arr[i] = float(selected.touch_count)
            active_width_component_arr[i] = _clip01(
                _safe_div(_sel_width_atr, TRADEABLE_WIDTH_REF)
            )
            active_formation_delay_component_arr[i] = _clip01(
                _safe_div(float(selected.formation_delay), TRADEABLE_FORM_REF)
            )
            active_structural_arr[i] = selected.structural_score
            active_tradeable_arr[i] = selected.tradeable_live_score
            active_score_arr[i] = selected.score
            active_age_arr[i] = float(
                i - selected.detect_idx if selected.detect_idx is not None else np.nan
            )
            active_formation_delay_arr[i] = float(selected.formation_delay)
            active_distance_arr[i] = _sel_dist_atr
            active_since_arr[i] = float(
                selected.detect_idx if selected.detect_idx is not None else np.nan
            )
            active_swept_arr[i] = 1 if selected.state == EQHL_STATE_SWEPT else 0
            active_state_arr[i] = selected.state
            close_i = float(close[i])
            pressure_active_arrays[f"{prefix}_active_bars_since_touch"][i] = float(
                selected.bars_since_last_touch
            )
            pressure_active_arrays[f"{prefix}_active_touches_since_detect"][i] = float(
                selected.touches_since_detect
            )
            pressure_active_arrays[f"{prefix}_active_close_tests"][i] = float(
                selected.close_tests_since_detect
            )
            pressure_active_arrays[f"{prefix}_active_max_pen_atr"][i] = float(
                selected.max_penetration_since_detect_atr
            )
            pressure_active_arrays[f"{prefix}_active_far_edge_atr"][i] = (
                _distance_to_far_edge_atr(
                    close_value=close_i,
                    source_low=selected.low_bound,
                    source_high=selected.high_bound,
                    atr_value=atr_i,
                )
            )
            pressure_active_arrays[f"{prefix}_active_inside_zone"][i] = int(
                np.isfinite(selected.low_bound)
                and np.isfinite(selected.high_bound)
                and selected.low_bound <= close_i <= selected.high_bound
            )
            pressure_active_arrays[f"{prefix}_active_signed_dist_atr"][i] = (
                _signed_distance_side_aware_atr(
                    close_value=close_i,
                    cluster_level=selected.level,
                    cluster_side=selected.side,
                    atr_value=atr_i,
                )
            )
            _nearby_count = 0
            _nearest_dist = np.inf
            for _other in candidates:
                if _other.cluster_id == selected.cluster_id:
                    continue
                if np.isfinite(_other.level) and atr_i > 0:
                    _d = abs(_other.level - selected.level) / atr_i
                    if _d <= 3.0:
                        _nearby_count += 1
                    if _d < _nearest_dist:
                        _nearest_dist = _d
            pressure_active_arrays[f"{prefix}_active_same_side_count_nearby"][
                i
            ] = _nearby_count
            pressure_active_arrays[f"{prefix}_active_nearest_same_side_atr"][i] = (
                float(_nearest_dist) if np.isfinite(_nearest_dist) else np.nan
            )

        if keep_last_n_clusters > 0:
            if len(high_clusters) > keep_last_n_clusters:
                high_clusters = sorted(
                    high_clusters,
                    key=lambda cl: (
                        cl.state != EQHL_STATE_ACTIVE,
                        -(cl.last_member_detect_idx),
                    ),
                )[:keep_last_n_clusters]
            if len(low_clusters) > keep_last_n_clusters:
                low_clusters = sorted(
                    low_clusters,
                    key=lambda cl: (
                        cl.state != EQHL_STATE_ACTIVE,
                        -(cl.last_member_detect_idx),
                    ),
                )[:keep_last_n_clusters]

    arrays: dict[str, np.ndarray] = {
        "eqh_detect_flag": eqh_detect_flag,
        "eqh_detect_idx": eqh_detect_idx,
        "eqh_event_id": eqh_event_id,
        "eqh_cluster_id_on_detect": eqh_cluster_id_on_detect,
        "eqh_origin_idx": eqh_origin_idx,
        "eqh_first_member_detect_idx": eqh_first_member_detect_idx,
        "eqh_last_member_detect_idx": eqh_last_member_detect_idx,
        "eqh_last_member_origin_idx": eqh_last_member_origin_idx,
        "eqh_member_count_on_detect": eqh_member_count_on_detect,
        "eqh_level_on_detect": eqh_level_on_detect,
        "eqh_low_on_detect": eqh_low_on_detect,
        "eqh_high_on_detect": eqh_high_on_detect,
        "eqh_width_on_detect": eqh_width_on_detect,
        "eqh_width_atr_on_detect": eqh_width_atr_on_detect,
        "eqh_width_component_on_detect": eqh_width_component_on_detect,
        "eqh_formation_delay_component_on_detect": eqh_formation_delay_component_on_detect,
        "eqh_structural_score_on_detect": eqh_structural_score_on_detect,
        "eqh_tradeable_live_score_on_detect": eqh_tradeable_live_score_on_detect,
        "eqh_score_on_detect": eqh_score_on_detect,
        "eqh_formation_delay_on_detect": eqh_formation_delay_on_detect,
        "eql_detect_flag": eql_detect_flag,
        "eql_detect_idx": eql_detect_idx,
        "eql_event_id": eql_event_id,
        "eql_cluster_id_on_detect": eql_cluster_id_on_detect,
        "eql_origin_idx": eql_origin_idx,
        "eql_first_member_detect_idx": eql_first_member_detect_idx,
        "eql_last_member_detect_idx": eql_last_member_detect_idx,
        "eql_last_member_origin_idx": eql_last_member_origin_idx,
        "eql_member_count_on_detect": eql_member_count_on_detect,
        "eql_level_on_detect": eql_level_on_detect,
        "eql_low_on_detect": eql_low_on_detect,
        "eql_high_on_detect": eql_high_on_detect,
        "eql_width_on_detect": eql_width_on_detect,
        "eql_width_atr_on_detect": eql_width_atr_on_detect,
        "eql_width_component_on_detect": eql_width_component_on_detect,
        "eql_formation_delay_component_on_detect": eql_formation_delay_component_on_detect,
        "eql_structural_score_on_detect": eql_structural_score_on_detect,
        "eql_tradeable_live_score_on_detect": eql_tradeable_live_score_on_detect,
        "eql_score_on_detect": eql_score_on_detect,
        "eql_formation_delay_on_detect": eql_formation_delay_on_detect,
        "eqh_active": eqh_active,
        "eqh_active_id": eqh_active_id,
        "eqh_active_level": eqh_active_level,
        "eqh_active_low": eqh_active_low,
        "eqh_active_high": eqh_active_high,
        "eqh_active_width": eqh_active_width,
        "eqh_active_width_atr": eqh_active_width_atr,
        "eqh_active_touch_count": eqh_active_touch_count,
        "eqh_active_width_component": eqh_active_width_component,
        "eqh_active_formation_delay_component": eqh_active_formation_delay_component,
        "eqh_active_structural_score": eqh_active_structural_score,
        "eqh_active_tradeable_live_score": eqh_active_tradeable_live_score,
        "eqh_active_score": eqh_active_score,
        "eqh_active_age": eqh_active_age,
        "eqh_active_formation_delay": eqh_active_formation_delay,
        "eqh_active_distance_atr": eqh_active_distance_atr,
        "eqh_active_since_idx": eqh_active_since_idx,
        "eqh_active_swept": eqh_active_swept,
        "eqh_active_state": eqh_active_state,
        "eqh_active_count": eqh_active_count,
        "eql_active": eql_active,
        "eql_active_id": eql_active_id,
        "eql_active_level": eql_active_level,
        "eql_active_low": eql_active_low,
        "eql_active_high": eql_active_high,
        "eql_active_width": eql_active_width,
        "eql_active_width_atr": eql_active_width_atr,
        "eql_active_touch_count": eql_active_touch_count,
        "eql_active_width_component": eql_active_width_component,
        "eql_active_formation_delay_component": eql_active_formation_delay_component,
        "eql_active_structural_score": eql_active_structural_score,
        "eql_active_tradeable_live_score": eql_active_tradeable_live_score,
        "eql_active_score": eql_active_score,
        "eql_active_age": eql_active_age,
        "eql_active_formation_delay": eql_active_formation_delay,
        "eql_active_distance_atr": eql_active_distance_atr,
        "eql_active_since_idx": eql_active_since_idx,
        "eql_active_swept": eql_active_swept,
        "eql_active_state": eql_active_state,
        "eql_active_count": eql_active_count,
        "eqh_swept_flag": eqh_swept_flag,
        "eqh_swept_idx": eqh_swept_idx,
        "eqh_swept_cluster_id": eqh_swept_cluster_id,
        "eqh_swept_level": eqh_swept_level,
        "eqh_inactive_idx": eqh_inactive_idx,
        "eql_swept_flag": eql_swept_flag,
        "eql_swept_idx": eql_swept_idx,
        "eql_swept_cluster_id": eql_swept_cluster_id,
        "eql_swept_level": eql_swept_level,
        "eql_inactive_idx": eql_inactive_idx,
        "eqh_pass_source_ready": eqh_pass_source_ready,
        "eqh_pass_tolerance": eqh_pass_tolerance,
        "eqh_pass_width_cap": eqh_pass_width_cap,
        "eqh_pass_span_cap": eqh_pass_span_cap,
        "eqh_cluster_merge_flag": eqh_cluster_merge_flag,
        "eql_pass_source_ready": eql_pass_source_ready,
        "eql_pass_tolerance": eql_pass_tolerance,
        "eql_pass_width_cap": eql_pass_width_cap,
        "eql_pass_span_cap": eql_pass_span_cap,
        "eql_cluster_merge_flag": eql_cluster_merge_flag,
        "equal_highs": eqh_detect_flag,
        "equal_lows": eql_detect_flag,
        "equal_highs_count": np.nan_to_num(eqh_active_touch_count, nan=0.0),
        "equal_lows_count": np.nan_to_num(eql_active_touch_count, nan=0.0),
        "equal_highs_level": eqh_active_level,
        "equal_lows_level": eql_active_level,
        "equal_highs_width": eqh_active_width,
        "equal_lows_width": eql_active_width,
        "equal_highs_width_atr": eqh_active_width_atr,
        "equal_lows_width_atr": eql_active_width_atr,
        "equal_highs_age": eqh_active_age,
        "equal_lows_age": eql_active_age,
        "equal_highs_active": eqh_active,
        "equal_lows_active": eql_active,
        "equal_highs_score": eqh_active_score,
        "equal_lows_score": eql_active_score,
        "equal_highs_structural_score": eqh_active_structural_score,
        "equal_lows_structural_score": eql_active_structural_score,
        "equal_highs_tradeable_live_score": eqh_active_tradeable_live_score,
        "equal_lows_tradeable_live_score": eql_active_tradeable_live_score,
        "equal_highs_cluster_id": eqh_active_id,
        "equal_lows_cluster_id": eql_active_id,
    }
    arrays.update(ranked_active_arrays)
    arrays.update(pressure_active_arrays)

    return pd.concat([out, pd.DataFrame(arrays, index=out.index)], axis=1)
