"""
foundation/sr_levels.py

Canonical support/resistance zone engine.

The public API remains additive and backward-compatible with the legacy
``nearest_*`` projections, but the internal model is now:

raw causal source events -> absorbed / emitted zones -> zone lifecycle ->
context projections + research-only calibration columns.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.event_ids import SequentialEventIdAllocator

__all__ = [
    "add_sr_levels",
    "extract_sr_source_events",
    "build_sr_level_registry",
    "build_sr_touch_audit_table",
    "update_sr_lifecycle",
    "project_sr_context",
    "add_sr_research_columns",
    "serialize_sr_registry",
    "deserialize_sr_registry",
    "SR_STATE_INACTIVE_PRE_LIVE",
    "SR_STATE_ACTIVE",
    "SR_STATE_ACTIVE_WEAKENED",
    "SR_STATE_BREAK_PENDING",
    "SR_STATE_INVALIDATED",
    "SR_STATE_RETIRED",
    "SR_TERMINAL_NONE",
    "SR_TERMINAL_INVALIDATED",
    "SR_TERMINAL_RETIRED_POST_INVALIDATION",
    "SR_TERMINAL_EXPIRED_HARD",
    "SR_TERMINAL_RETIRED_WEAKENED",
    "SR_TERMINAL_RETIRED_ABSORBED",
    "SR_SIDE_SUPPORT",
    "SR_SIDE_RESISTANCE",
    "SR_FAMILY_SWING",
    "SR_FAMILY_EQHL",
    "SR_FAMILY_SESSION",
    "SR_FAMILY_DAY",
    "SR_FAMILY_WEEK",
    "SR_FAMILY_VP",
    "SRLevel",
]

SR_STATE_INACTIVE_PRE_LIVE = 0
SR_STATE_ACTIVE = 1
SR_STATE_ACTIVE_WEAKENED = 2
SR_STATE_BREAK_PENDING = 3
SR_STATE_INVALIDATED = 4
SR_STATE_RETIRED = 5

SR_TERMINAL_NONE = 0
SR_TERMINAL_INVALIDATED = 1
SR_TERMINAL_RETIRED_POST_INVALIDATION = 2
SR_TERMINAL_EXPIRED_HARD = 3
SR_TERMINAL_RETIRED_WEAKENED = 4
SR_TERMINAL_RETIRED_ABSORBED = 5

SR_TERMINAL_REASON_INVALIDATED = "invalidated"
SR_TERMINAL_REASON_RETIRED_POST_INVALIDATION = "retired_post_invalidation"
SR_TERMINAL_REASON_EXPIRED_HARD = "expired_hard"
SR_TERMINAL_REASON_RETIRED_WEAKENED = "retired_weakened"
SR_TERMINAL_REASON_RETIRED_ABSORBED = "retired_absorbed"

SR_SIDE_SUPPORT = -1
SR_SIDE_RESISTANCE = 1

SR_FAMILY_SWING = "swing"
SR_FAMILY_EQHL = "eqhl"
SR_FAMILY_SESSION = "session"
SR_FAMILY_DAY = "day"
SR_FAMILY_WEEK = "week"
SR_FAMILY_VP = "vp"

FAMILY_PRIOR: dict[str, float] = {
    SR_FAMILY_SWING: 0.60,
    SR_FAMILY_EQHL: 0.55,
    SR_FAMILY_SESSION: 0.50,
    SR_FAMILY_DAY: 0.65,
    SR_FAMILY_WEEK: 0.80,
    SR_FAMILY_VP: 0.50,
}

TOUCH_TOL_ATR: float = 0.15
INVALIDATION_BUFFER_ATR: float = 0.10
MERGE_TOL_ATR: float = 0.20

ABSORB_TOL_ATR: float = 0.25
ABSORB_CROSS_FAMILY_TOL_ATR: float = 0.14
ZONE_MIN_HALF_WIDTH_ATR: float = 0.08
ZONE_MAX_HALF_WIDTH_ATR: float = 0.80
VP_DWELL_BARS: int = 3
VP_SHIFT_MIN_ATR: float = 0.35
VP_STABILITY_TOL_ATR: float = 0.05
BREAK_OVERSHOOT_ATR: float = 0.35
BREAK_CONFIRM_CLOSES: int = 2
BREAK_RECLAIM_WINDOW_BARS: int = 2
PRIMARY_ZONE_RADIUS_ATR: float = 3.0
FRESHNESS_TAU: float = 120.0
PRIMARY_SELECTION_SCORE_MARGIN: float = 0.12
WIDTH_GOOD_MIN_ATR: float = 0.18
WIDTH_GOOD_MAX_ATR: float = 0.55

MAX_AGE_BARS: int = 500
INVALIDATED_RETIRE_BARS: int = 50
MAX_WEAKEN_COUNT: int = 4
SR_LADDER_DEPTH: int = 3

FAMILY_MAX_AGE: dict[str, int] = {
    SR_FAMILY_SESSION: 30,
    SR_FAMILY_DAY: 60,
    SR_FAMILY_WEEK: 120,
    SR_FAMILY_VP: 200,
    SR_FAMILY_SWING: 500,
    SR_FAMILY_EQHL: 500,
}

_PRIOR_PERIOD_SOURCES: list[tuple[str, int, str, str]] = [
    ("prev_day_high", SR_SIDE_RESISTANCE, SR_FAMILY_DAY, "prior_day_high"),
    ("prev_day_low", SR_SIDE_SUPPORT, SR_FAMILY_DAY, "prior_day_low"),
    ("prev_week_high", SR_SIDE_RESISTANCE, SR_FAMILY_WEEK, "prior_week_high"),
    ("prev_week_low", SR_SIDE_SUPPORT, SR_FAMILY_WEEK, "prior_week_low"),
    ("prev_asia_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_asia_high"),
    ("prev_asia_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_asia_low"),
    ("prev_london_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_london_high"),
    ("prev_london_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_london_low"),
    ("prev_ny_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_ny_high"),
    ("prev_ny_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_ny_low"),
]

_VP_SOURCES: list[tuple[str, int, str, str]] = [
    ("vp_vah", SR_SIDE_RESISTANCE, SR_FAMILY_VP, "vp_vah"),
    ("vp_val", SR_SIDE_SUPPORT, SR_FAMILY_VP, "vp_val"),
]

_RESEARCH_HORIZONS = (4, 8, 12)

# Outcome metric thresholds
#  - HELD: zone is "held" if the close never breaches the zone's far edge
#    (zone_low for support, zone_high for resistance) over (t, t+h].
#  - SIGNED: asymmetric reaction in the zone's implied direction. Support
#    expects price to move UP from base_close; resistance expects DOWN. We
#    require the signed move to exceed _OUTCOME_SIGNED_K * base_atr.
_OUTCOME_SIGNED_K_ATR: float = 0.30

FAMILY_INFORMATION_WEIGHT: dict[str, float] = {
    SR_FAMILY_EQHL: 1.00,
    SR_FAMILY_SWING: 0.90,
    SR_FAMILY_WEEK: 0.82,
    SR_FAMILY_DAY: 0.72,
    SR_FAMILY_VP: 0.55,
    SR_FAMILY_SESSION: 0.42,
}

FAMILY_CALIBRATED_PRIOR: dict[str, float] = {
    SR_FAMILY_EQHL: 0.95,
    SR_FAMILY_SWING: 0.76,
    SR_FAMILY_VP: 0.56,
    SR_FAMILY_SESSION: 0.48,
    SR_FAMILY_DAY: 0.38,
    SR_FAMILY_WEEK: 0.28,
}

FAMILY_SCORE_MULTIPLIER: dict[str, float] = {
    SR_FAMILY_EQHL: 1.12,
    SR_FAMILY_SWING: 1.04,
    SR_FAMILY_VP: 1.00,
    SR_FAMILY_SESSION: 0.98,
    SR_FAMILY_DAY: 0.88,
    SR_FAMILY_WEEK: 0.80,
}


@dataclass
class SRLevel:
    """One causal S/R source or emitted zone."""

    level_id: int
    side: int
    level_price: float
    source_family: str
    source_sub: str = ""
    source_origin_idx: int = -1
    source_confirm_idx: int = -1
    source_live_from_idx: int = -1
    source_origin_ts: Any = None
    source_confirm_ts: Any = None
    source_live_from_ts: Any = None
    source_strength_initial: float = 0.50
    source_metadata_norm: dict = field(default_factory=dict)

    state: int = field(default=SR_STATE_INACTIVE_PRE_LIVE)
    age_bars: int = field(default=0)
    touch_count: int = field(default=0)
    refresh_count: int = field(default=0)
    weaken_count: int = field(default=0)
    last_touch_idx: int = field(default=-1)
    invalidation_idx: int = field(default=-1)
    invalidation_ts: Any = field(default=None)
    expiry_idx: int = field(default=-1)
    expiry_ts: Any = field(default=None)
    bars_since_invalidation: int = field(default=0)
    superseded_by: Optional[int] = field(default=None)
    retirement_idx: int = field(default=-1)
    terminal_state: int = field(default=SR_TERMINAL_NONE)
    terminal_reason: str = field(default="")
    level_strength: float = field(default=0.0)

    zone_low: float = field(default=np.nan)
    zone_high: float = field(default=np.nan)
    zone_half_width_abs: float = field(default=np.nan)
    zone_half_width_atr: float = field(default=ZONE_MIN_HALF_WIDTH_ATR)
    zone_width_atr: float = field(default=ZONE_MIN_HALF_WIDTH_ATR * 2.0)
    mean_anchor_atr: float = field(default=1.0)
    anchor_count: int = field(default=1)
    source_count: int = field(default=1)
    family_count: int = field(default=1)
    family_mix_counts: dict[str, int] = field(default_factory=dict)
    family_strength_max: dict[str, float] = field(default_factory=dict)
    dominant_family: str = field(default="")
    best_source_family: str = field(default="")
    best_source_strength: float = field(default=0.0)
    last_anchor_idx: int = field(default=-1)
    emitted_zone_flag: bool = field(default=False)
    absorbed_by: Optional[int] = field(default=None)
    absorbed_into_family: str = field(default="")
    absorbed_into_best_family: str = field(default="")

    pending_break_idx: int = field(default=-1)
    pending_break_consecutive: int = field(default=0)
    reclaim_count: int = field(default=0)
    failed_break_count: int = field(default=0)
    accepted_break_count: int = field(default=0)
    clean_touch_count: int = field(default=0)
    weak_touch_count: int = field(default=0)

    source_quality_score: float = field(default=0.0)
    confluence_score: float = field(default=0.0)
    reaction_quality_score: float = field(default=0.0)
    freshness_score: float = field(default=0.0)
    family_prior_score: float = field(default=0.0)
    width_quality_score: float = field(default=0.0)
    score_penalty_value: float = field(default=0.0)

    anchor_prices: list[float] = field(default_factory=list)
    anchor_weights: list[float] = field(default_factory=list)
    anchor_atrs: list[float] = field(default_factory=list)
    anchor_source_strengths: list[float] = field(default_factory=list)
    interaction_prices: list[float] = field(default_factory=list)
    touch_rows: list[int] = field(default_factory=list)
    touch_scores: list[float] = field(default_factory=list)
    touch_records: list[dict[str, object]] = field(default_factory=list)


def _get_timestamps(df: pd.DataFrame) -> list:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return ts.tolist()
    return [None] * len(df)


def _weighted_mean(values: list[float], weights: list[float], fallback: float) -> float:
    if not values or not weights:
        return fallback
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    denom = float(np.nansum(w))
    if not np.isfinite(denom) or denom <= 0:
        return fallback
    return float(np.nansum(v * w) / denom)


def _best_family(level: SRLevel) -> str:
    return level.best_source_family or level.source_family


def _bars_to_hard_expiry(level: SRLevel) -> int:
    max_age = FAMILY_MAX_AGE.get(_best_family(level), MAX_AGE_BARS)
    return int(max(max_age - level.age_bars, 0))


def _first_terminal_idx(level: SRLevel) -> int:
    candidates = [
        idx
        for idx in (level.invalidation_idx, level.expiry_idx, level.retirement_idx)
        if isinstance(idx, int) and idx >= 0
    ]
    return min(candidates) if candidates else -1


def _mark_terminal(
    level: SRLevel,
    *,
    state: int,
    reason: str,
) -> None:
    level.terminal_state = int(state)
    level.terminal_reason = str(reason)


def _piecewise_score(
    value: float,
    thresholds: tuple[float, ...],
    scores: tuple[float, ...],
) -> float:
    if not np.isfinite(value):
        return float(scores[0])
    for threshold, score in zip(thresholds, scores, strict=False):
        if value <= threshold:
            return float(score)
    return float(scores[-1])


def _anchor_count_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    return "5+"


def _family_count_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _width_bucket(value: float) -> str:
    if not np.isfinite(value) or value <= 0.30:
        return "<=0.30"
    if value <= 0.45:
        return "0.30-0.45"
    return ">0.45"


def _age_bucket(value: int) -> str:
    if value <= 10:
        return "0-10"
    if value <= 30:
        return "11-30"
    if value <= 60:
        return "31-60"
    return "61+"


def _family_information_score(family: str) -> float:
    return float(FAMILY_INFORMATION_WEIGHT.get(family, 0.50))


def _family_calibrated_prior(family: str) -> float:
    return float(FAMILY_CALIBRATED_PRIOR.get(family, 0.45))


def _family_score_multiplier(family: str) -> float:
    return float(FAMILY_SCORE_MULTIPLIER.get(family, 1.0))


def _dominant_family(level: SRLevel) -> str:
    if not level.family_mix_counts:
        return level.source_family
    return max(
        level.family_mix_counts.items(),
        key=lambda item: (item[1], FAMILY_PRIOR.get(item[0], 0.50)),
    )[0]


def _best_source_family(level: SRLevel) -> str:
    if not level.family_strength_max:
        return level.source_family
    return max(
        level.family_strength_max.items(),
        key=lambda item: (
            item[1],
            _family_information_score(item[0]),
            FAMILY_PRIOR.get(item[0], 0.50),
        ),
    )[0]


def _width_bucket_prior(width: float) -> float:
    # Monotonic decreasing prior: narrower zones are more precise levels and
    # historically have higher reaction quality. The previous (0.72, 0.28, 0.60)
    # tuple was non-monotonic; the middle bucket got the lowest weight, which
    # was an artifact of an earlier tuning pass and not justified empirically.
    bucket = _width_bucket(width)
    if bucket == "<=0.30":
        return 0.65
    if bucket == "0.30-0.45":
        return 0.45
    return 0.30


def _normalized_age(level: SRLevel) -> float:
    max_age = FAMILY_MAX_AGE.get(
        level.best_source_family or level.source_family, MAX_AGE_BARS
    )
    return float(level.age_bars / max(max_age, 1))


def _record_touch_event(
    level: SRLevel,
    *,
    row: int,
    price: float,
    touch_type: str,
) -> None:
    price_value = float(price) if np.isfinite(price) else float(level.level_price)
    level.touch_rows.append(int(row))
    level.touch_records.append(
        {
            "row": int(row),
            "touch_type": touch_type,
            "price": price_value,
            "zone_id": int(level.level_id),
            "side": int(level.side),
            "score_snapshot": float(level.level_strength),
            "source_family": str(level.source_family),
            "best_source_family": str(level.best_source_family or level.source_family),
            "dominant_family": str(level.dominant_family or level.source_family),
            "anchor_count": int(level.anchor_count),
            "family_count": int(level.family_count),
            "width_atr": float(level.zone_width_atr),
            "age_bars": int(level.age_bars),
            "source_quality_score": float(level.source_quality_score),
            "confluence_score": float(level.confluence_score),
            "reaction_quality_score": float(level.reaction_quality_score),
            "freshness_score": float(level.freshness_score),
            "family_prior_score": float(level.family_prior_score),
            "width_quality_score": float(level.width_quality_score),
            "score_penalty_value": float(level.score_penalty_value),
        }
    )


def _initialize_zone_geometry(level: SRLevel, anchor_atr: float) -> SRLevel:
    atr = float(anchor_atr) if np.isfinite(anchor_atr) and anchor_atr > 0 else 1.0
    weight = max(level.source_strength_initial, 0.05)
    level.anchor_prices = [float(level.level_price)]
    level.anchor_weights = [float(weight)]
    level.anchor_atrs = [atr]
    level.anchor_source_strengths = [float(level.source_strength_initial)]
    level.family_mix_counts = {level.source_family: 1}
    level.family_strength_max = {
        level.source_family: float(level.source_strength_initial)
    }
    level.anchor_count = 1
    level.source_count = 1
    level.family_count = 1
    level.dominant_family = level.source_family
    level.best_source_family = level.source_family
    level.best_source_strength = float(level.source_strength_initial)
    level.last_anchor_idx = level.source_live_from_idx
    level.mean_anchor_atr = atr
    level.zone_half_width_atr = ZONE_MIN_HALF_WIDTH_ATR
    level.zone_half_width_abs = ZONE_MIN_HALF_WIDTH_ATR * atr
    level.zone_low = level.level_price - level.zone_half_width_abs
    level.zone_high = level.level_price + level.zone_half_width_abs
    level.zone_width_atr = level.zone_half_width_atr * 2.0
    return level


def _refresh_zone_geometry(level: SRLevel) -> None:
    fallback_mid = float(level.level_price)
    mid = _weighted_mean(level.anchor_prices, level.anchor_weights, fallback_mid)
    mean_atr = _weighted_mean(level.anchor_atrs, level.anchor_weights, 1.0)
    if not np.isfinite(mean_atr) or mean_atr <= 0:
        mean_atr = 1.0
    prices = np.asarray(level.anchor_prices, dtype=float)
    weights = np.asarray(level.anchor_weights, dtype=float)
    denom = float(np.nansum(weights))
    if not np.isfinite(denom) or denom <= 0:
        anchor_dispersion_atr = 0.0
    else:
        anchor_dispersion_atr = float(
            np.sqrt(np.nansum(weights * np.square(prices - mid)) / denom) / mean_atr
        )
    anchor_span_atr = (
        float((np.nanmax(prices) - np.nanmin(prices)) / mean_atr)
        if prices.size > 1
        else 0.0
    )
    if level.interaction_prices:
        touch_prices = np.asarray(level.interaction_prices, dtype=float)
        touch_dispersion_atr = float(np.nanstd(touch_prices) / mean_atr)
    else:
        touch_dispersion_atr = 0.0
    half_width_atr = float(
        np.clip(
            0.06
            + 0.18 * anchor_dispersion_atr
            + 0.12 * anchor_span_atr
            + 0.10 * touch_dispersion_atr
            + 0.03 * min(max(level.anchor_count - 1, 0), 4),
            ZONE_MIN_HALF_WIDTH_ATR,
            ZONE_MAX_HALF_WIDTH_ATR,
        )
    )
    level.level_price = mid
    level.mean_anchor_atr = mean_atr
    level.zone_half_width_atr = half_width_atr
    level.zone_half_width_abs = half_width_atr * mean_atr
    level.zone_low = mid - level.zone_half_width_abs
    level.zone_high = mid + level.zone_half_width_abs
    level.zone_width_atr = half_width_atr * 2.0
    level.family_count = len(level.family_mix_counts)
    level.dominant_family = _dominant_family(level)
    level.best_source_family = _best_source_family(level)
    level.best_source_strength = float(
        level.family_strength_max.get(
            level.best_source_family, level.source_strength_initial
        )
    )
    # NOTE: source_family is intentionally NOT overwritten here. It records the
    # ORIGINAL creator of the zone. Use best_source_family / dominant_family for
    # current classification.


def _compute_strength(level: SRLevel) -> float:
    weighted_strength = float(
        np.clip(
            _weighted_mean(
                level.anchor_source_strengths,
                level.anchor_weights,
                level.source_strength_initial,
            ),
            0.0,
            1.0,
        )
    )
    best_family = level.best_source_family or level.source_family
    best_family_info = _family_information_score(best_family)
    best_family_prior = _family_calibrated_prior(best_family)
    source_quality = float(
        np.clip(
            0.72 * weighted_strength
            + 0.28 * float(np.clip(level.best_source_strength, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )

    anchor_term = _piecewise_score(
        float(level.anchor_count),
        (1.0, 2.0, 4.0, 8.0),
        (0.32, 0.68, 0.74, 0.76, 0.78),
    )
    family_info_values = [
        _family_information_score(family) for family in level.family_mix_counts
    ] or [_family_information_score(level.source_family)]
    mean_family_info = float(np.mean(family_info_values))
    diversity_base = _piecewise_score(
        float(level.family_count),
        (1.0, 2.0, 3.0),
        (0.52, 0.54, 0.50, 0.44),
    )
    low_info_mix_share = 0.0
    total_family_weight = float(sum(level.family_mix_counts.values()))
    if total_family_weight > 0:
        low_info_mix_share = float(
            sum(
                count
                for family, count in level.family_mix_counts.items()
                if _family_information_score(family) < 0.5
            )
            / total_family_weight
        )
    diversity_adjust = 0.06 * max(mean_family_info - 0.55, 0.0) - 0.10 * max(
        low_info_mix_share - 0.55, 0.0
    )
    confluence = float(
        np.clip(
            0.82 * anchor_term + 0.18 * (diversity_base + diversity_adjust),
            0.0,
            1.0,
        )
    )

    clean_ratio = (
        float(level.clean_touch_count) / max(level.touch_count, 1)
        if level.touch_count > 0
        else 0.0
    )
    weak_ratio = (
        float(level.weak_touch_count) / max(level.touch_count, 1)
        if level.touch_count > 0
        else 0.0
    )
    touch_usefulness = _piecewise_score(
        float(level.touch_count),
        (0.0, 1.0, 3.0, 6.0),
        (0.20, 0.58, 0.62, 0.54, 0.40),
    )
    reclaim_bonus = _piecewise_score(
        float(level.reclaim_count),
        (0.0, 1.0, 2.0),
        (0.0, 0.12, 0.18, 0.20),
    )
    excessive_touch_penalty = (
        0.12 * max(level.touch_count - 3, 0) / max(level.touch_count, 1)
    )
    reaction_quality = float(
        np.clip(
            0.28 * touch_usefulness
            + 0.34 * clean_ratio
            + 0.18 * reclaim_bonus
            + 0.20 * (1.0 - min(weak_ratio, 1.0))
            - excessive_touch_penalty,
            0.0,
            1.0,
        )
    )

    freshness = float(np.clip(math.exp(-level.age_bars / FRESHNESS_TAU), 0.0, 1.0))
    family_prior = float(
        np.clip(
            0.85 * best_family_prior + 0.15 * best_family_info,
            0.0,
            1.0,
        )
    )

    width = float(level.zone_width_atr)
    width_bucket_prior = _width_bucket_prior(width)
    if not np.isfinite(width) or width <= 0:
        width_quality = 0.20
    elif width < WIDTH_GOOD_MIN_ATR:
        precision_score = (
            float(np.clip(width / max(WIDTH_GOOD_MIN_ATR, 1e-9), 0.0, 1.0)) * 0.7
        )
        width_quality = 0.60 * width_bucket_prior + 0.40 * precision_score
    elif width <= WIDTH_GOOD_MAX_ATR:
        precision_score = 1.0
        width_quality = 0.60 * width_bucket_prior + 0.40 * precision_score
    else:
        over = (width - WIDTH_GOOD_MAX_ATR) / max(WIDTH_GOOD_MAX_ATR, 1e-9)
        precision_score = float(np.clip(1.0 - 0.55 * over, 0.15, 1.0))
        width_quality = 0.60 * width_bucket_prior + 0.40 * precision_score

    normalized_age = _normalized_age(level)
    age_penalty = 0.10 * min(max(normalized_age - 1.0, 0.0), 1.5)
    repeated_weak_penalty = 0.05 * min(level.weaken_count, 4)
    pending_penalty = 0.10 if level.state == SR_STATE_BREAK_PENDING else 0.0
    touch_churn_penalty = (
        0.06
        * max(level.weak_touch_count - level.clean_touch_count, 0)
        / max(level.touch_count, 1)
    )
    width_penalty = 0.08 * max(1.0 - width_quality, 0.0)
    family_penalty = 0.05 * max(low_info_mix_share - 0.65, 0.0) + 0.08 * max(
        0.60 - best_family_prior, 0.0
    )
    penalty = (
        age_penalty
        + repeated_weak_penalty
        + pending_penalty
        + touch_churn_penalty
        + width_penalty
        + family_penalty
    )

    raw = (
        0.34 * source_quality
        + 0.12 * confluence
        + 0.14 * reaction_quality
        + 0.18 * freshness
        + 0.12 * family_prior
        + 0.10 * width_quality
        - penalty
    )
    raw *= _family_score_multiplier(best_family)
    level.source_quality_score = source_quality
    level.confluence_score = confluence
    level.reaction_quality_score = reaction_quality
    level.freshness_score = freshness
    level.family_prior_score = family_prior
    level.width_quality_score = width_quality
    level.score_penalty_value = penalty
    level.level_strength = float(np.clip(raw, 0.0, 1.0))
    return level.level_strength


def _make_source_level(
    *,
    allocator: SequentialEventIdAllocator,
    side: int,
    price: float,
    family: str,
    sub: str,
    origin_idx: int,
    confirm_idx: int,
    live_from_idx: int,
    timestamps: list,
    src_strength: float,
    atr_i: float,
    metadata: dict[str, object] | None = None,
) -> SRLevel:
    origin_idx = max(0, origin_idx)
    confirm_idx = max(0, confirm_idx)
    live_from_idx = max(0, live_from_idx)
    level = SRLevel(
        level_id=allocator.allocate(),
        side=side,
        level_price=float(price),
        source_family=family,
        source_sub=sub,
        source_origin_idx=origin_idx,
        source_confirm_idx=confirm_idx,
        source_live_from_idx=live_from_idx,
        source_origin_ts=(
            timestamps[min(origin_idx, len(timestamps) - 1)] if timestamps else None
        ),
        source_confirm_ts=(
            timestamps[min(confirm_idx, len(timestamps) - 1)] if timestamps else None
        ),
        source_live_from_ts=(
            timestamps[min(live_from_idx, len(timestamps) - 1)] if timestamps else None
        ),
        source_strength_initial=float(np.clip(src_strength, 0.0, 1.0)),
        source_metadata_norm=metadata or {},
    )
    _initialize_zone_geometry(level, atr_i)
    _compute_strength(level)
    return level


def _extract_swing_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    required = {
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
    }
    if not required.issubset(df.columns):
        return []
    n = len(df)
    sh_flag = df["swing_high_confirm_flag"].to_numpy()
    sl_flag = df["swing_low_confirm_flag"].to_numpy()
    sh_price = df["swing_high_confirm_price"].to_numpy(dtype=float)
    sl_price = df["swing_low_confirm_price"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    sh_origin = (
        df["swing_high_confirm_origin_idx"].to_numpy(dtype=float)
        if "swing_high_confirm_origin_idx" in df.columns
        else np.full(n, np.nan)
    )
    sl_origin = (
        df["swing_low_confirm_origin_idx"].to_numpy(dtype=float)
        if "swing_low_confirm_origin_idx" in df.columns
        else np.full(n, np.nan)
    )

    levels: list[SRLevel] = []
    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        if sh_flag[i] and np.isfinite(sh_price[i]):
            origin_idx = int(sh_origin[i]) if np.isfinite(sh_origin[i]) else i
            origin_idx = max(0, min(origin_idx, n - 1))
            mag_atr = (
                abs(sh_price[i] - close[origin_idx]) / atr_i
                if origin_idx != i and np.isfinite(close[origin_idx])
                else 1.5
            )
            levels.append(
                _make_source_level(
                    allocator=allocator,
                    side=SR_SIDE_RESISTANCE,
                    price=float(sh_price[i]),
                    family=SR_FAMILY_SWING,
                    sub="swing_high",
                    origin_idx=origin_idx,
                    confirm_idx=i,
                    live_from_idx=i,
                    timestamps=timestamps,
                    src_strength=float(np.clip(mag_atr / 3.0, 0.0, 1.0)),
                    atr_i=atr_i,
                    metadata={"magnitude_atr": round(float(mag_atr), 4)},
                )
            )
        if sl_flag[i] and np.isfinite(sl_price[i]):
            origin_idx = int(sl_origin[i]) if np.isfinite(sl_origin[i]) else i
            origin_idx = max(0, min(origin_idx, n - 1))
            mag_atr = (
                abs(sl_price[i] - close[origin_idx]) / atr_i
                if origin_idx != i and np.isfinite(close[origin_idx])
                else 1.5
            )
            levels.append(
                _make_source_level(
                    allocator=allocator,
                    side=SR_SIDE_SUPPORT,
                    price=float(sl_price[i]),
                    family=SR_FAMILY_SWING,
                    sub="swing_low",
                    origin_idx=origin_idx,
                    confirm_idx=i,
                    live_from_idx=i,
                    timestamps=timestamps,
                    src_strength=float(np.clip(mag_atr / 3.0, 0.0, 1.0)),
                    atr_i=atr_i,
                    metadata={"magnitude_atr": round(float(mag_atr), 4)},
                )
            )
    return levels


def _extract_eqhl_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    eqh_ok = {"eqh_detect_flag", "eqh_level_on_detect", "eqh_origin_idx"}.issubset(
        df.columns
    )
    eql_ok = {"eql_detect_flag", "eql_level_on_detect", "eql_origin_idx"}.issubset(
        df.columns
    )
    if not eqh_ok and not eql_ok:
        return []
    n = len(df)
    levels: list[SRLevel] = []

    if eqh_ok:
        eqh_flag = df["eqh_detect_flag"].to_numpy()
        eqh_price = df["eqh_level_on_detect"].to_numpy(dtype=float)
        eqh_orig = df["eqh_origin_idx"].to_numpy(dtype=float)
        eqh_score = (
            df["eqh_score_on_detect"].to_numpy(dtype=float)
            if "eqh_score_on_detect" in df.columns
            else np.full(n, 0.5)
        )
        eqh_count = (
            df["eqh_member_count_on_detect"].to_numpy(dtype=float)
            if "eqh_member_count_on_detect" in df.columns
            else np.full(n, 2.0)
        )
        for i in range(n):
            if not (eqh_flag[i] and np.isfinite(eqh_price[i])):
                continue
            atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
            origin_idx = int(eqh_orig[i]) if np.isfinite(eqh_orig[i]) else i
            origin_idx = max(0, min(origin_idx, n - 1))
            levels.append(
                _make_source_level(
                    allocator=allocator,
                    side=SR_SIDE_RESISTANCE,
                    price=float(eqh_price[i]),
                    family=SR_FAMILY_EQHL,
                    sub="eqh",
                    origin_idx=origin_idx,
                    confirm_idx=i,
                    live_from_idx=i,
                    timestamps=timestamps,
                    src_strength=float(np.clip(eqh_score[i], 0.0, 1.0)),
                    atr_i=atr_i,
                    metadata={
                        "score": round(float(eqh_score[i]), 4),
                        "member_count": int(eqh_count[i]),
                    },
                )
            )

    if eql_ok:
        eql_flag = df["eql_detect_flag"].to_numpy()
        eql_price = df["eql_level_on_detect"].to_numpy(dtype=float)
        eql_orig = df["eql_origin_idx"].to_numpy(dtype=float)
        eql_score = (
            df["eql_score_on_detect"].to_numpy(dtype=float)
            if "eql_score_on_detect" in df.columns
            else np.full(n, 0.5)
        )
        eql_count = (
            df["eql_member_count_on_detect"].to_numpy(dtype=float)
            if "eql_member_count_on_detect" in df.columns
            else np.full(n, 2.0)
        )
        for i in range(n):
            if not (eql_flag[i] and np.isfinite(eql_price[i])):
                continue
            atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
            origin_idx = int(eql_orig[i]) if np.isfinite(eql_orig[i]) else i
            origin_idx = max(0, min(origin_idx, n - 1))
            levels.append(
                _make_source_level(
                    allocator=allocator,
                    side=SR_SIDE_SUPPORT,
                    price=float(eql_price[i]),
                    family=SR_FAMILY_EQHL,
                    sub="eql",
                    origin_idx=origin_idx,
                    confirm_idx=i,
                    live_from_idx=i,
                    timestamps=timestamps,
                    src_strength=float(np.clip(eql_score[i], 0.0, 1.0)),
                    atr_i=atr_i,
                    metadata={
                        "score": round(float(eql_score[i]), 4),
                        "member_count": int(eql_count[i]),
                    },
                )
            )
    return levels


def _extract_prior_period_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    n = len(df)
    levels: list[SRLevel] = []
    for col, side, family, sub in _PRIOR_PERIOD_SOURCES:
        if col not in df.columns:
            continue
        series = df[col].to_numpy(dtype=float)
        prev_val = np.nan
        for i in range(n):
            val = series[i]
            if not np.isfinite(val):
                continue
            if np.isnan(prev_val) or val != prev_val:
                atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
                levels.append(
                    _make_source_level(
                        allocator=allocator,
                        side=side,
                        price=float(val),
                        family=family,
                        sub=sub,
                        origin_idx=i,
                        confirm_idx=i,
                        live_from_idx=i,
                        timestamps=timestamps,
                        src_strength=FAMILY_PRIOR.get(family, 0.50),
                        atr_i=atr_i,
                        metadata={"sub": sub},
                    )
                )
            prev_val = val
    return levels


def _extract_vp_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    n = len(df)
    levels: list[SRLevel] = []
    for col, side, family, sub in _VP_SOURCES:
        if col not in df.columns:
            continue
        series = df[col].to_numpy(dtype=float)
        last_emitted = np.nan
        for i in range(VP_DWELL_BARS - 1, n):
            window = series[i - VP_DWELL_BARS + 1 : i + 1]
            if not np.isfinite(window).all():
                continue
            atr_window = atr[i - VP_DWELL_BARS + 1 : i + 1]
            mean_atr = float(np.nanmean(atr_window))
            if not np.isfinite(mean_atr) or mean_atr <= 0:
                mean_atr = 1.0
            if (
                float(np.nanmax(window) - np.nanmin(window))
                > VP_STABILITY_TOL_ATR * mean_atr
            ):
                continue
            val = float(window[-1])
            if (
                np.isfinite(last_emitted)
                and abs(val - last_emitted) < VP_SHIFT_MIN_ATR * mean_atr
            ):
                continue
            levels.append(
                _make_source_level(
                    allocator=allocator,
                    side=side,
                    price=val,
                    family=family,
                    sub=sub,
                    origin_idx=i - VP_DWELL_BARS + 1,
                    confirm_idx=i,
                    live_from_idx=i,
                    timestamps=timestamps,
                    src_strength=FAMILY_PRIOR.get(family, 0.50),
                    atr_i=mean_atr,
                    metadata={"sub": sub, "dwell_bars": VP_DWELL_BARS},
                )
            )
            last_emitted = val
    return levels


def extract_sr_source_events(df: pd.DataFrame) -> list[SRLevel]:
    if len(df) == 0:
        return []
    atr = get_atr_array(df)
    timestamps = _get_timestamps(df)
    allocator = SequentialEventIdAllocator()
    out: list[SRLevel] = []
    out.extend(_extract_swing_levels(df, atr, timestamps, allocator))
    out.extend(_extract_eqhl_levels(df, atr, timestamps, allocator))
    out.extend(_extract_prior_period_levels(df, atr, timestamps, allocator))
    out.extend(_extract_vp_levels(df, atr, timestamps, allocator))
    out.sort(key=lambda lev: (lev.source_live_from_idx, lev.level_id))
    return out


def build_sr_level_registry(df: pd.DataFrame) -> dict[int, SRLevel]:
    return {lev.level_id: lev for lev in extract_sr_source_events(df)}


_REGISTRY_PERSIST_DROP_FIELDS = frozenset(
    {
        "anchor_prices",
        "anchor_weights",
        "anchor_atrs",
        "anchor_source_strengths",
        "interaction_prices",
        "touch_rows",
        "touch_scores",
        "touch_records",
    }
)
_REGISTRY_TIMESTAMP_FIELDS = frozenset(
    {
        "source_origin_ts",
        "source_confirm_ts",
        "source_live_from_ts",
        "invalidation_ts",
        "expiry_ts",
    }
)


def _registry_field_payload(level: SRLevel) -> dict[str, Any]:
    raw = asdict(level)
    for key in list(raw.keys()):
        if key in _REGISTRY_PERSIST_DROP_FIELDS:
            raw.pop(key, None)
            continue
        if key in _REGISTRY_TIMESTAMP_FIELDS:
            value = raw[key]
            if value is None:
                continue
            ts = pd.Timestamp(value)
            raw[key] = ts.isoformat() if not pd.isna(ts) else None
    return raw


def serialize_sr_registry(registry: dict[int, SRLevel]) -> list[dict[str, Any]]:
    """Serialize the registry to a JSON-friendly list, dropping bulky workspace lists.

    The audit table replaces ``touch_records`` for downstream consumers; the per-anchor
    lists are workspace state from the lifecycle pass and are not needed after.
    """
    return [_registry_field_payload(level) for level in registry.values()]


def deserialize_sr_registry(payload: list[dict[str, Any]]) -> dict[int, SRLevel]:
    """Reconstruct the registry from ``serialize_sr_registry`` output."""
    field_names = {f.name for f in dataclass_fields(SRLevel)}
    registry: dict[int, SRLevel] = {}
    for raw in payload:
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in field_names:
                continue
            if key in _REGISTRY_TIMESTAMP_FIELDS and value is not None:
                kwargs[key] = pd.Timestamp(value)
            else:
                kwargs[key] = value
        level = SRLevel(**kwargs)
        registry[int(level.level_id)] = level
    return registry


def _is_live_zone(level: SRLevel) -> bool:
    return level.emitted_zone_flag and level.state in {
        SR_STATE_ACTIVE,
        SR_STATE_ACTIVE_WEAKENED,
        SR_STATE_BREAK_PENDING,
    }


def _find_absorb_target(
    registry: dict[int, SRLevel],
    active_zone_ids: set[int],
    new_level: SRLevel,
    atr_i: float,
) -> SRLevel | None:
    tol = ABSORB_TOL_ATR * atr_i
    candidates: list[SRLevel] = []
    for lid in active_zone_ids:
        other = registry[lid]
        if other.side != new_level.side or not _is_live_zone(other):
            continue
        distance_abs = abs(other.level_price - new_level.level_price)
        family_match = (
            other.best_source_family == new_level.source_family
            or other.source_family == new_level.source_family
            or new_level.source_family in other.family_mix_counts
        )
        candidate_tol = tol if family_match else ABSORB_CROSS_FAMILY_TOL_ATR * atr_i
        if not family_match and _family_information_score(
            new_level.source_family
        ) > _family_information_score(other.best_source_family or other.source_family):
            candidate_tol = min(candidate_tol, MERGE_TOL_ATR * atr_i * 0.4)
        if distance_abs <= candidate_tol:
            candidates.append(other)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda lev: (
            abs(lev.level_price - new_level.level_price),
            -(
                0.60
                * (
                    1.0
                    if new_level.source_family in lev.family_mix_counts
                    or lev.best_source_family == new_level.source_family
                    else 0.0
                )
                + 0.25
                * _family_information_score(lev.best_source_family or lev.source_family)
                + 0.15 * lev.level_strength
            ),
        ),
    )


def _absorb_source_into_zone(
    target: SRLevel, source: SRLevel, activation_row: int
) -> None:
    weight = max(source.source_strength_initial, 0.05)
    target.anchor_prices.append(float(source.level_price))
    target.anchor_weights.append(float(weight))
    target.anchor_atrs.append(float(source.mean_anchor_atr))
    target.anchor_source_strengths.append(float(source.source_strength_initial))
    target.family_mix_counts[source.source_family] = (
        target.family_mix_counts.get(source.source_family, 0) + 1
    )
    target.family_strength_max[source.source_family] = max(
        float(source.source_strength_initial),
        float(target.family_strength_max.get(source.source_family, 0.0)),
    )
    target.anchor_count += 1
    target.source_count += 1
    target.last_anchor_idx = max(target.last_anchor_idx, source.source_live_from_idx)
    target.source_strength_initial = _weighted_mean(
        target.anchor_source_strengths,
        target.anchor_weights,
        target.source_strength_initial,
    )
    _refresh_zone_geometry(target)
    _compute_strength(target)

    source.emitted_zone_flag = False
    source.state = SR_STATE_RETIRED
    source.absorbed_by = target.level_id
    source.absorbed_into_family = target.dominant_family or target.source_family
    source.absorbed_into_best_family = target.best_source_family or target.source_family
    source.superseded_by = target.level_id
    source.retirement_idx = activation_row
    _mark_terminal(
        source,
        state=SR_TERMINAL_RETIRED_ABSORBED,
        reason=SR_TERMINAL_REASON_RETIRED_ABSORBED,
    )


def _set_active_state(level: SRLevel) -> None:
    if level.weaken_count > 0:
        level.state = SR_STATE_ACTIVE_WEAKENED
    else:
        level.state = SR_STATE_ACTIVE


def _rank_ladder_zones(zones: list[SRLevel], close: float) -> list[SRLevel]:
    return sorted(
        zones,
        key=lambda lev: (
            abs(lev.level_price - close),
            -lev.level_strength,
            -lev.source_live_from_idx,
            lev.level_id,
        ),
    )


def _populate_ladder(
    ctx: dict[str, np.ndarray],
    *,
    row: int,
    side_prefix: str,
    close: float,
    zones: list[SRLevel],
) -> None:
    ranked = _rank_ladder_zones(zones, close)
    for slot, level in enumerate(ranked[:SR_LADDER_DEPTH], start=1):
        prefix = f"sr_{side_prefix}_l{slot}"
        ctx[f"{prefix}_id"][row] = float(level.level_id)
        ctx[f"{prefix}_low"][row] = float(level.zone_low)
        ctx[f"{prefix}_high"][row] = float(level.zone_high)
        ctx[f"{prefix}_mid"][row] = float(level.level_price)
        ctx[f"{prefix}_score"][row] = float(level.level_strength)
        ctx[f"{prefix}_family"][row] = _best_family(level)
        ctx[f"{prefix}_state"][row] = float(level.state)
        ctx[f"{prefix}_age_bars"][row] = float(level.age_bars)
        ctx[f"{prefix}_expiry_bars_remaining"][row] = float(_bars_to_hard_expiry(level))


def _classify_touch(
    level: SRLevel, hi: float, lo: float, cl: float
) -> tuple[str | None, float]:
    if level.side == SR_SIDE_SUPPORT:
        if (
            np.isfinite(lo)
            and lo < level.zone_low
            and np.isfinite(cl)
            and cl >= level.zone_low
        ):
            return "weak pierce", float(lo)
        if (
            np.isfinite(lo)
            and lo <= level.zone_high
            and np.isfinite(cl)
            and cl >= level.zone_low
        ):
            return "clean touch", float(np.clip(lo, level.zone_low, level.zone_high))
    else:
        if (
            np.isfinite(hi)
            and hi > level.zone_high
            and np.isfinite(cl)
            and cl <= level.zone_high
        ):
            return "weak pierce", float(hi)
        if (
            np.isfinite(hi)
            and hi >= level.zone_low
            and np.isfinite(cl)
            and cl <= level.zone_high
        ):
            return "clean touch", float(np.clip(hi, level.zone_low, level.zone_high))
    return None, np.nan


def _close_breach(level: SRLevel, cl: float, atr_i: float) -> tuple[bool, float]:
    if level.side == SR_SIDE_SUPPORT:
        overshoot = max(level.zone_low - cl, 0.0)
    else:
        overshoot = max(cl - level.zone_high, 0.0)
    return overshoot > 0.0, overshoot / max(atr_i, 1e-9)


def _choose_primary_zone(
    zones: list[SRLevel],
    close: float,
    atr_i: float,
) -> SRLevel | None:
    if not zones or not np.isfinite(close) or not np.isfinite(atr_i) or atr_i <= 0:
        return None
    nearest = min(
        zones,
        key=lambda lev: abs(lev.level_price - close),
    )
    candidates = [
        lev
        for lev in zones
        if abs(lev.level_price - close) / atr_i <= PRIMARY_ZONE_RADIUS_ATR
    ]
    if not candidates:
        return nearest
    score_pick = max(
        candidates,
        key=lambda lev: (
            lev.level_strength,
            -(abs(lev.level_price - close) / atr_i),
        ),
    )
    if score_pick.level_id == nearest.level_id:
        return score_pick
    score_edge = score_pick.level_strength - nearest.level_strength
    nearest_family_prior = _family_calibrated_prior(
        nearest.best_source_family or nearest.source_family
    )
    score_pick_family_prior = _family_calibrated_prior(
        score_pick.best_source_family or score_pick.source_family
    )
    nearest_is_tighter = nearest.zone_width_atr <= score_pick.zone_width_atr * 1.05
    if score_edge < PRIMARY_SELECTION_SCORE_MARGIN or (
        score_edge < (PRIMARY_SELECTION_SCORE_MARGIN * 1.75)
        and nearest_family_prior >= score_pick_family_prior
        and nearest_is_tighter
    ):
        return nearest
    return score_pick


def update_sr_lifecycle(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
) -> dict[str, np.ndarray]:
    n = len(df)
    atr = get_atr_array(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    activation_map: dict[int, list[int]] = {}
    for lid, lev in registry.items():
        activation_map.setdefault(lev.source_live_from_idx, []).append(lid)
    # Within a single bar, process activations in descending family-information
    # order so higher-info families (eqhl, swing) anchor and lower-info families
    # absorb into them. This makes the absorption order independent of the
    # extraction order in extract_sr_source_events.
    for _row, ids in activation_map.items():
        ids.sort(
            key=lambda lid: (
                -_family_information_score(registry[lid].source_family),
                -float(registry[lid].source_strength_initial),
                int(lid),
            )
        )

    ctx: dict[str, np.ndarray] = {
        "nearest_support_price": np.full(n, np.nan),
        "nearest_support_distance": np.full(n, np.nan),
        "nearest_support_distance_atr": np.full(n, np.nan),
        "nearest_support_age_bars": np.full(n, np.nan),
        "nearest_support_strength": np.full(n, np.nan),
        "nearest_support_source_family": np.full(n, None, dtype=object),
        "nearest_support_source_idx": np.full(n, np.nan),
        "nearest_support_zone_id": np.full(n, np.nan),
        "nearest_support_touch_count": np.full(n, np.nan),
        "nearest_support_refresh_count": np.full(n, np.nan),
        "nearest_support_weaken_count": np.full(n, np.nan),
        "nearest_support_active": np.zeros(n, dtype=np.int8),
        "nearest_resistance_price": np.full(n, np.nan),
        "nearest_resistance_distance": np.full(n, np.nan),
        "nearest_resistance_distance_atr": np.full(n, np.nan),
        "nearest_resistance_age_bars": np.full(n, np.nan),
        "nearest_resistance_strength": np.full(n, np.nan),
        "nearest_resistance_source_family": np.full(n, None, dtype=object),
        "nearest_resistance_source_idx": np.full(n, np.nan),
        "nearest_resistance_zone_id": np.full(n, np.nan),
        "nearest_resistance_touch_count": np.full(n, np.nan),
        "nearest_resistance_refresh_count": np.full(n, np.nan),
        "nearest_resistance_weaken_count": np.full(n, np.nan),
        "nearest_resistance_active": np.zeros(n, dtype=np.int8),
        "inside_sr_band_flag": np.zeros(n, dtype=np.int8),
        "between_nearest_sr_flag": np.zeros(n, dtype=np.int8),
        "above_nearest_resistance_flag": np.zeros(n, dtype=np.int8),
        "below_nearest_support_flag": np.zeros(n, dtype=np.int8),
        "support_broken_this_bar": np.zeros(n, dtype=np.int8),
        "resistance_broken_this_bar": np.zeros(n, dtype=np.int8),
        "active_support_count": np.zeros(n, dtype=np.int32),
        "active_resistance_count": np.zeros(n, dtype=np.int32),
        "support_cluster_density_atr": np.full(n, np.nan),
        "resistance_cluster_density_atr": np.full(n, np.nan),
        "primary_support_zone_low": np.full(n, np.nan),
        "primary_support_zone_high": np.full(n, np.nan),
        "primary_support_zone_mid": np.full(n, np.nan),
        "primary_support_zone_score": np.full(n, np.nan),
        "primary_support_zone_anchor_count": np.full(n, np.nan),
        "primary_support_zone_family_count": np.full(n, np.nan),
        "primary_support_zone_width_atr": np.full(n, np.nan),
        "primary_support_zone_id": np.full(n, np.nan),
        "primary_resistance_zone_low": np.full(n, np.nan),
        "primary_resistance_zone_high": np.full(n, np.nan),
        "primary_resistance_zone_mid": np.full(n, np.nan),
        "primary_resistance_zone_score": np.full(n, np.nan),
        "primary_resistance_zone_anchor_count": np.full(n, np.nan),
        "primary_resistance_zone_family_count": np.full(n, np.nan),
        "primary_resistance_zone_width_atr": np.full(n, np.nan),
        "primary_resistance_zone_id": np.full(n, np.nan),
        "inside_primary_support_zone_flag": np.zeros(n, dtype=np.int8),
        "inside_primary_resistance_zone_flag": np.zeros(n, dtype=np.int8),
        "sr_break_pending_flag": np.zeros(n, dtype=np.int8),
        "sr_reclaim_this_bar_flag": np.zeros(n, dtype=np.int8),
    }
    for side_prefix in ("support", "resistance"):
        for slot in range(1, SR_LADDER_DEPTH + 1):
            prefix = f"sr_{side_prefix}_l{slot}"
            ctx[f"{prefix}_id"] = np.full(n, np.nan)
            ctx[f"{prefix}_low"] = np.full(n, np.nan)
            ctx[f"{prefix}_high"] = np.full(n, np.nan)
            ctx[f"{prefix}_mid"] = np.full(n, np.nan)
            ctx[f"{prefix}_score"] = np.full(n, np.nan)
            ctx[f"{prefix}_family"] = np.full(n, None, dtype=object)
            ctx[f"{prefix}_state"] = np.full(n, np.nan)
            ctx[f"{prefix}_age_bars"] = np.full(n, np.nan)
            ctx[f"{prefix}_expiry_bars_remaining"] = np.full(n, np.nan)

    active_zone_ids: set[int] = set()
    invalidated_zone_ids: set[int] = set()
    ts_col = "timestamp" in df.columns

    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        hi = float(high[i]) if np.isfinite(high[i]) else np.nan
        lo = float(low[i]) if np.isfinite(low[i]) else np.nan
        cl = float(close[i]) if np.isfinite(close[i]) else np.nan
        reclaimed_this_bar = False

        for lid in activation_map.get(i, []):
            lev = registry[lid]
            lev.state = SR_STATE_ACTIVE
            _compute_strength(lev)
            target = _find_absorb_target(registry, active_zone_ids, lev, atr_i)
            if target is not None:
                _absorb_source_into_zone(target, lev, i)
                continue
            lev.emitted_zone_flag = True
            _set_active_state(lev)
            active_zone_ids.add(lid)

        if not np.isfinite(cl):
            continue

        newly_invalidated: list[int] = []
        for lid in list(active_zone_ids):
            lev = registry[lid]
            if not _is_live_zone(lev):
                active_zone_ids.discard(lid)
                continue
            lev.age_bars += 1

            breached, overshoot_atr = _close_breach(lev, cl, atr_i)
            if breached:
                if lev.state != SR_STATE_BREAK_PENDING:
                    lev.state = SR_STATE_BREAK_PENDING
                    lev.pending_break_idx = i
                    lev.pending_break_consecutive = 1
                else:
                    lev.pending_break_consecutive += 1
                if (
                    lev.pending_break_consecutive >= BREAK_CONFIRM_CLOSES
                    or overshoot_atr >= BREAK_OVERSHOOT_ATR
                ):
                    lev.state = SR_STATE_INVALIDATED
                    lev.invalidation_idx = i
                    if ts_col:
                        lev.invalidation_ts = df["timestamp"].iloc[i]
                    _mark_terminal(
                        lev,
                        state=SR_TERMINAL_INVALIDATED,
                        reason=SR_TERMINAL_REASON_INVALIDATED,
                    )
                    lev.accepted_break_count += 1
                    if lev.side == SR_SIDE_SUPPORT:
                        ctx["support_broken_this_bar"][i] = 1
                    else:
                        ctx["resistance_broken_this_bar"][i] = 1
                    newly_invalidated.append(lid)
                    _compute_strength(lev)
                    continue
            elif lev.state == SR_STATE_BREAK_PENDING:
                if (
                    lev.pending_break_idx >= 0
                    and i - lev.pending_break_idx <= BREAK_RECLAIM_WINDOW_BARS
                ):
                    lev.failed_break_count += 1
                    lev.reclaim_count += 1
                    lev.touch_count += 1
                    lev.refresh_count += 1
                    lev.last_touch_idx = i
                    level_price = (
                        lev.zone_low if lev.side == SR_SIDE_SUPPORT else lev.zone_high
                    )
                    _record_touch_event(
                        lev,
                        row=i,
                        price=level_price,
                        touch_type="reclaim-after-break-pending",
                    )
                    reclaimed_this_bar = True
                lev.pending_break_idx = -1
                lev.pending_break_consecutive = 0
                _set_active_state(lev)

            touch_type, touch_price = _classify_touch(lev, hi, lo, cl)
            if touch_type is not None:
                lev.touch_count += 1
                lev.refresh_count += 1
                lev.last_touch_idx = i
                lev.interaction_prices.append(float(touch_price))
                if touch_type == "clean touch":
                    lev.clean_touch_count += 1
                else:
                    lev.weak_touch_count += 1
                _record_touch_event(
                    lev,
                    row=i,
                    price=touch_price,
                    touch_type=touch_type,
                )
            if touch_type == "weak pierce" and lev.state != SR_STATE_BREAK_PENDING:
                lev.weaken_count += 1
                lev.state = SR_STATE_ACTIVE_WEAKENED

            _compute_strength(lev)
            if touch_type is not None:
                lev.touch_scores.append(lev.level_strength)
            if lev.state != SR_STATE_BREAK_PENDING:
                _set_active_state(lev)

        for lid in newly_invalidated:
            active_zone_ids.discard(lid)
            invalidated_zone_ids.add(lid)

        for lid in list(invalidated_zone_ids):
            lev = registry[lid]
            lev.bars_since_invalidation += 1
            if lev.bars_since_invalidation > INVALIDATED_RETIRE_BARS:
                lev.state = SR_STATE_RETIRED
                lev.retirement_idx = i
                _mark_terminal(
                    lev,
                    state=SR_TERMINAL_RETIRED_POST_INVALIDATION,
                    reason=SR_TERMINAL_REASON_RETIRED_POST_INVALIDATION,
                )
                invalidated_zone_ids.discard(lid)

        for lid in list(active_zone_ids):
            lev = registry[lid]
            max_age = FAMILY_MAX_AGE.get(_best_family(lev), MAX_AGE_BARS)
            if lev.age_bars > max_age:
                lev.state = SR_STATE_RETIRED
                lev.expiry_idx = i
                lev.expiry_ts = df["timestamp"].iloc[i] if ts_col else None
                lev.retirement_idx = i
                _mark_terminal(
                    lev,
                    state=SR_TERMINAL_EXPIRED_HARD,
                    reason=SR_TERMINAL_REASON_EXPIRED_HARD,
                )
                active_zone_ids.discard(lid)
                continue
            if lev.weaken_count >= MAX_WEAKEN_COUNT:
                lev.state = SR_STATE_RETIRED
                lev.retirement_idx = i
                _mark_terminal(
                    lev,
                    state=SR_TERMINAL_RETIRED_WEAKENED,
                    reason=SR_TERMINAL_REASON_RETIRED_WEAKENED,
                )
                active_zone_ids.discard(lid)

        act_sups = [
            registry[lid]
            for lid in active_zone_ids
            if _is_live_zone(registry[lid]) and registry[lid].side == SR_SIDE_SUPPORT
        ]
        act_res = [
            registry[lid]
            for lid in active_zone_ids
            if _is_live_zone(registry[lid]) and registry[lid].side == SR_SIDE_RESISTANCE
        ]
        ctx["active_support_count"][i] = len(act_sups)
        ctx["active_resistance_count"][i] = len(act_res)
        if any(lev.state == SR_STATE_BREAK_PENDING for lev in act_sups + act_res):
            ctx["sr_break_pending_flag"][i] = 1
        if reclaimed_this_bar:
            ctx["sr_reclaim_this_bar_flag"][i] = 1

        valid_sups = [lev for lev in act_sups if lev.level_price <= cl]
        valid_res = [lev for lev in act_res if lev.level_price >= cl]
        _populate_ladder(
            ctx,
            row=i,
            side_prefix="support",
            close=cl,
            zones=valid_sups,
        )
        _populate_ladder(
            ctx,
            row=i,
            side_prefix="resistance",
            close=cl,
            zones=valid_res,
        )

        if valid_sups:
            ns = max(valid_sups, key=lambda lev: lev.level_price)
            dist = cl - ns.level_price
            ctx["nearest_support_price"][i] = ns.level_price
            ctx["nearest_support_distance"][i] = dist
            ctx["nearest_support_distance_atr"][i] = dist / atr_i
            ctx["nearest_support_age_bars"][i] = float(ns.age_bars)
            ctx["nearest_support_strength"][i] = ns.level_strength
            ctx["nearest_support_source_family"][i] = ns.source_family
            ctx["nearest_support_source_idx"][i] = float(ns.source_confirm_idx)
            ctx["nearest_support_zone_id"][i] = float(ns.level_id)
            ctx["nearest_support_touch_count"][i] = float(ns.touch_count)
            ctx["nearest_support_refresh_count"][i] = float(ns.refresh_count)
            ctx["nearest_support_weaken_count"][i] = float(ns.weaken_count)
            ctx["nearest_support_active"][i] = 1
        if valid_res:
            nr = min(valid_res, key=lambda lev: lev.level_price)
            dist = nr.level_price - cl
            ctx["nearest_resistance_price"][i] = nr.level_price
            ctx["nearest_resistance_distance"][i] = dist
            ctx["nearest_resistance_distance_atr"][i] = dist / atr_i
            ctx["nearest_resistance_age_bars"][i] = float(nr.age_bars)
            ctx["nearest_resistance_strength"][i] = nr.level_strength
            ctx["nearest_resistance_source_family"][i] = nr.source_family
            ctx["nearest_resistance_source_idx"][i] = float(nr.source_confirm_idx)
            ctx["nearest_resistance_zone_id"][i] = float(nr.level_id)
            ctx["nearest_resistance_touch_count"][i] = float(nr.touch_count)
            ctx["nearest_resistance_refresh_count"][i] = float(nr.refresh_count)
            ctx["nearest_resistance_weaken_count"][i] = float(nr.weaken_count)
            ctx["nearest_resistance_active"][i] = 1

        if ctx["nearest_support_active"][i] and ctx["nearest_resistance_active"][i]:
            ctx["between_nearest_sr_flag"][i] = 1
        if not ctx["nearest_resistance_active"][i]:
            ctx["above_nearest_resistance_flag"][i] = 1
        if not ctx["nearest_support_active"][i]:
            ctx["below_nearest_support_flag"][i] = 1

        ns_price = ctx["nearest_support_price"][i]
        nr_price = ctx["nearest_resistance_price"][i]
        if np.isfinite(ns_price) and abs(cl - ns_price) <= TOUCH_TOL_ATR * atr_i:
            ctx["inside_sr_band_flag"][i] = 1
        if np.isfinite(nr_price) and abs(cl - nr_price) <= TOUCH_TOL_ATR * atr_i:
            ctx["inside_sr_band_flag"][i] = 1

        primary_sup = _choose_primary_zone(valid_sups, cl, atr_i)
        primary_res = _choose_primary_zone(valid_res, cl, atr_i)
        for prefix, level in (
            ("primary_support_zone", primary_sup),
            ("primary_resistance_zone", primary_res),
        ):
            if level is None:
                continue
            ctx[f"{prefix}_low"][i] = level.zone_low
            ctx[f"{prefix}_high"][i] = level.zone_high
            ctx[f"{prefix}_mid"][i] = level.level_price
            ctx[f"{prefix}_score"][i] = level.level_strength
            ctx[f"{prefix}_anchor_count"][i] = float(level.anchor_count)
            ctx[f"{prefix}_family_count"][i] = float(level.family_count)
            ctx[f"{prefix}_width_atr"][i] = float(level.zone_width_atr)
            ctx[f"{prefix}_id"][i] = float(level.level_id)

        if (
            np.isfinite(ctx["primary_support_zone_low"][i])
            and ctx["primary_support_zone_low"][i]
            <= cl
            <= ctx["primary_support_zone_high"][i]
        ):
            ctx["inside_primary_support_zone_flag"][i] = 1
        if (
            np.isfinite(ctx["primary_resistance_zone_low"][i])
            and ctx["primary_resistance_zone_low"][i]
            <= cl
            <= ctx["primary_resistance_zone_high"][i]
        ):
            ctx["inside_primary_resistance_zone_flag"][i] = 1

        if len(act_sups) >= 2:
            ctx["support_cluster_density_atr"][i] = (
                float(np.std([lev.level_price for lev in act_sups])) / atr_i
            )
        if len(act_res) >= 2:
            ctx["resistance_cluster_density_atr"][i] = (
                float(np.std([lev.level_price for lev in act_res])) / atr_i
            )

    return ctx


def project_sr_context(df: pd.DataFrame, registry: dict[int, SRLevel]) -> pd.DataFrame:
    ctx = update_sr_lifecycle(df, registry)
    out = df.copy()
    return pd.concat([out, pd.DataFrame(ctx, index=out.index)], axis=1)


def build_sr_touch_audit_table(
    df: pd.DataFrame, registry: dict[int, SRLevel]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(df)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr = get_atr_array(df)
    ts = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce").tolist()
        if "timestamp" in df.columns
        else [None] * n
    )
    event_id = 1
    for lev in registry.values():
        if not lev.emitted_zone_flag:
            continue
        for idx, record_meta in enumerate(lev.touch_records):
            row = int(record_meta.get("row", -1))
            if row < 0 or row >= n:
                continue
            score = float(record_meta.get("score_snapshot", lev.level_strength))
            record: dict[str, object] = {
                "event_id": event_id,
                "row": int(row),
                "timestamp": ts[row],
                "zone_id": int(lev.level_id),
                "side": int(lev.side),
                "score": float(score),
                "touch_type": str(record_meta.get("touch_type", "")),
                "touch_price": float(record_meta.get("price", lev.level_price)),
                "source_family": str(
                    record_meta.get("source_family", lev.source_family)
                ),
                "best_source_family": str(
                    record_meta.get(
                        "best_source_family",
                        lev.best_source_family or lev.source_family,
                    )
                ),
                "dominant_family": str(
                    record_meta.get(
                        "dominant_family", lev.dominant_family or lev.source_family
                    )
                ),
                "anchor_count": int(record_meta.get("anchor_count", lev.anchor_count)),
                "anchor_count_bucket": _anchor_count_bucket(
                    int(record_meta.get("anchor_count", lev.anchor_count))
                ),
                "family_count": int(record_meta.get("family_count", lev.family_count)),
                "family_count_bucket": _family_count_bucket(
                    int(record_meta.get("family_count", lev.family_count))
                ),
                "width_atr": float(record_meta.get("width_atr", lev.zone_width_atr)),
                "width_bucket": _width_bucket(
                    float(record_meta.get("width_atr", lev.zone_width_atr))
                ),
                "age_bars": int(record_meta.get("age_bars", lev.age_bars)),
                "age_bucket": _age_bucket(
                    int(record_meta.get("age_bars", lev.age_bars))
                ),
                "source_quality_score": float(
                    record_meta.get("source_quality_score", lev.source_quality_score)
                ),
                "confluence_score": float(
                    record_meta.get("confluence_score", lev.confluence_score)
                ),
                "reaction_quality_score": float(
                    record_meta.get(
                        "reaction_quality_score", lev.reaction_quality_score
                    )
                ),
                "freshness_score": float(
                    record_meta.get("freshness_score", lev.freshness_score)
                ),
                "family_prior_score": float(
                    record_meta.get("family_prior_score", lev.family_prior_score)
                ),
                "width_quality_score": float(
                    record_meta.get("width_quality_score", lev.width_quality_score)
                ),
                "score_penalty_value": float(
                    record_meta.get("score_penalty_value", lev.score_penalty_value)
                ),
                "zone_score_at_touch": float(record_meta.get("score_snapshot", score)),
                "nearest_zone_id": np.nan,
                "nearest_zone_score": np.nan,
                "primary_zone_id": np.nan,
                "primary_zone_score": np.nan,
                "nearest_matches_touch": False,
                "primary_matches_touch": False,
            }
            if lev.side == SR_SIDE_SUPPORT:
                if "nearest_support_zone_id" in df.columns:
                    record["nearest_zone_id"] = float(
                        pd.to_numeric(
                            df["nearest_support_zone_id"], errors="coerce"
                        ).iloc[row]
                    )
                if "nearest_support_strength" in df.columns:
                    record["nearest_zone_score"] = float(
                        pd.to_numeric(
                            df["nearest_support_strength"], errors="coerce"
                        ).iloc[row]
                    )
                if "primary_support_zone_id" in df.columns:
                    record["primary_zone_id"] = float(
                        pd.to_numeric(
                            df["primary_support_zone_id"], errors="coerce"
                        ).iloc[row]
                    )
                if "primary_support_zone_score" in df.columns:
                    record["primary_zone_score"] = float(
                        pd.to_numeric(
                            df["primary_support_zone_score"], errors="coerce"
                        ).iloc[row]
                    )
            else:
                if "nearest_resistance_zone_id" in df.columns:
                    record["nearest_zone_id"] = float(
                        pd.to_numeric(
                            df["nearest_resistance_zone_id"], errors="coerce"
                        ).iloc[row]
                    )
                if "nearest_resistance_strength" in df.columns:
                    record["nearest_zone_score"] = float(
                        pd.to_numeric(
                            df["nearest_resistance_strength"], errors="coerce"
                        ).iloc[row]
                    )
                if "primary_resistance_zone_id" in df.columns:
                    record["primary_zone_id"] = float(
                        pd.to_numeric(
                            df["primary_resistance_zone_id"], errors="coerce"
                        ).iloc[row]
                    )
                if "primary_resistance_zone_score" in df.columns:
                    record["primary_zone_score"] = float(
                        pd.to_numeric(
                            df["primary_resistance_zone_score"], errors="coerce"
                        ).iloc[row]
                    )
            if np.isfinite(float(record["nearest_zone_id"])):
                record["nearest_matches_touch"] = int(
                    float(record["nearest_zone_id"])
                ) == int(lev.level_id)
            if np.isfinite(float(record["primary_zone_id"])):
                record["primary_matches_touch"] = int(
                    float(record["primary_zone_id"])
                ) == int(lev.level_id)
            base_close = close[row]
            base_atr = (
                float(atr[row]) if np.isfinite(atr[row]) and atr[row] > 0 else np.nan
            )
            zone_low = float(lev.zone_low) if np.isfinite(lev.zone_low) else np.nan
            zone_high = float(lev.zone_high) if np.isfinite(lev.zone_high) else np.nan
            for horizon in _RESEARCH_HORIZONS:
                end = min(row + horizon, n - 1)
                if (
                    row >= end
                    or not np.isfinite(base_close)
                    or not np.isfinite(base_atr)
                ):
                    record[f"outcome_{horizon}"] = np.nan
                    record[f"outcome_held_{horizon}"] = np.nan
                    record[f"outcome_signed_{horizon}"] = np.nan
                    record[f"mfe_atr_{horizon}"] = np.nan
                    record[f"mae_atr_{horizon}"] = np.nan
                    continue
                fut_high = float(np.nanmax(high[row + 1 : end + 1]))
                fut_low = float(np.nanmin(low[row + 1 : end + 1]))
                fut_close = float(close[end])
                fut_close_window = close[row + 1 : end + 1]
                if lev.side == SR_SIDE_SUPPORT:
                    outcome = 1.0 if fut_close > base_close else 0.0
                    mfe = (fut_high - base_close) / base_atr
                    mae = (base_close - fut_low) / base_atr
                    # Held: no close ever broke below the support's lower edge.
                    if np.isfinite(zone_low):
                        held = (
                            1.0
                            if not bool((fut_close_window < zone_low).any())
                            else 0.0
                        )
                    else:
                        held = np.nan
                    # Signed: price moved UP (away from support) by k*ATR.
                    signed = (
                        1.0
                        if (fut_close - base_close) > _OUTCOME_SIGNED_K_ATR * base_atr
                        else 0.0
                    )
                else:
                    outcome = 1.0 if fut_close < base_close else 0.0
                    mfe = (base_close - fut_low) / base_atr
                    mae = (fut_high - base_close) / base_atr
                    # Held: no close ever broke above the resistance's upper edge.
                    if np.isfinite(zone_high):
                        held = (
                            1.0
                            if not bool((fut_close_window > zone_high).any())
                            else 0.0
                        )
                    else:
                        held = np.nan
                    signed = (
                        1.0
                        if (base_close - fut_close) > _OUTCOME_SIGNED_K_ATR * base_atr
                        else 0.0
                    )
                record[f"outcome_{horizon}"] = float(outcome)
                record[f"outcome_held_{horizon}"] = (
                    float(held) if np.isfinite(held) else np.nan
                )
                record[f"outcome_signed_{horizon}"] = float(signed)
                record[f"mfe_atr_{horizon}"] = float(mfe)
                record[f"mae_atr_{horizon}"] = float(mae)
            rows.append(record)
            event_id += 1
    return pd.DataFrame(rows)


def _research_touch_rows(
    df: pd.DataFrame, registry: dict[int, SRLevel]
) -> pd.DataFrame:
    return build_sr_touch_audit_table(df, registry)


_OUTCOME_METRIC_COLUMNS: tuple[str, ...] = ("drift", "held", "signed")
_OUTCOME_METRIC_TO_COLUMN: dict[str, str] = {
    "drift": "outcome_{h}",
    "held": "outcome_held_{h}",
    "signed": "outcome_signed_{h}",
}


def _score_quintile_summary(touches: pd.DataFrame) -> dict[str, object]:
    empty_metric_dict = {
        metric: {str(h): None for h in _RESEARCH_HORIZONS}
        for metric in _OUTCOME_METRIC_COLUMNS
    }
    if touches.empty or touches["score"].dropna().nunique() < 2:
        return {
            "touch_count": int(len(touches)),
            "quintiles": {},
            "monotonicity": {str(h): None for h in _RESEARCH_HORIZONS},
            "top_vs_bottom_delta": {str(h): None for h in _RESEARCH_HORIZONS},
            "monotonicity_by_metric": {
                k: dict(v) for k, v in empty_metric_dict.items()
            },
            "top_vs_bottom_delta_by_metric": {
                k: dict(v) for k, v in empty_metric_dict.items()
            },
        }
    ranked = touches.copy()
    ranked["quintile"] = pd.qcut(
        ranked["score"],
        q=5,
        labels=False,
        duplicates="drop",
    )
    quintiles: dict[str, dict[str, object]] = {}
    monotonicity: dict[str, bool | None] = {}
    deltas: dict[str, float | None] = {}
    monotonicity_by_metric: dict[str, dict[str, bool | None]] = {
        metric: {} for metric in _OUTCOME_METRIC_COLUMNS
    }
    deltas_by_metric: dict[str, dict[str, float | None]] = {
        metric: {} for metric in _OUTCOME_METRIC_COLUMNS
    }
    for quintile, group in ranked.groupby("quintile", dropna=True):
        label = str(int(quintile) + 1)
        quintiles[label] = {
            "count": int(len(group)),
            "score_mean": float(group["score"].mean()),
        }
        for horizon in _RESEARCH_HORIZONS:
            outcome_col = f"outcome_{horizon}"
            mfe_col = f"mfe_atr_{horizon}"
            mae_col = f"mae_atr_{horizon}"
            quintiles[label][f"outcome_rate_{horizon}"] = float(
                pd.to_numeric(group[outcome_col], errors="coerce").mean()
            )
            quintiles[label][f"mfe_atr_mean_{horizon}"] = float(
                pd.to_numeric(group[mfe_col], errors="coerce").mean()
            )
            quintiles[label][f"mae_atr_mean_{horizon}"] = float(
                pd.to_numeric(group[mae_col], errors="coerce").mean()
            )
            for metric in _OUTCOME_METRIC_COLUMNS:
                col = _OUTCOME_METRIC_TO_COLUMN[metric].format(h=horizon)
                if col not in group.columns:
                    quintiles[label][f"outcome_rate_{metric}_{horizon}"] = None
                    continue
                quintiles[label][f"outcome_rate_{metric}_{horizon}"] = float(
                    pd.to_numeric(group[col], errors="coerce").mean()
                )
    for horizon in _RESEARCH_HORIZONS:
        for metric in _OUTCOME_METRIC_COLUMNS:
            metric_rates = [
                quintiles[key].get(f"outcome_rate_{metric}_{horizon}")
                for key in sorted(quintiles, key=int)
            ]
            metric_clean = [
                float(v) for v in metric_rates if v is not None and np.isfinite(v)
            ]
            monotonicity_by_metric[metric][str(horizon)] = (
                bool(all(a <= b for a, b in zip(metric_clean, metric_clean[1:])))
                if len(metric_clean) >= 2
                else None
            )
            if metric_clean:
                deltas_by_metric[metric][str(horizon)] = float(
                    metric_clean[-1] - metric_clean[0]
                )
            else:
                deltas_by_metric[metric][str(horizon)] = None
        rates = [
            quintiles[key].get(f"outcome_rate_{horizon}")
            for key in sorted(quintiles, key=int)
        ]
        clean_rates = [float(v) for v in rates if v is not None and np.isfinite(v)]
        monotonicity[str(horizon)] = (
            bool(all(a <= b for a, b in zip(clean_rates, clean_rates[1:])))
            if len(clean_rates) >= 2
            else None
        )
        if "1" in quintiles and str(len(quintiles)) in quintiles:
            top = quintiles[str(len(quintiles))].get(f"outcome_rate_{horizon}")
            bottom = quintiles["1"].get(f"outcome_rate_{horizon}")
            deltas[str(horizon)] = (
                float(top - bottom) if top is not None and bottom is not None else None
            )
        else:
            deltas[str(horizon)] = None
    return {
        "touch_count": int(len(ranked)),
        "quintiles": quintiles,
        "monotonicity": monotonicity,
        "top_vs_bottom_delta": deltas,
        "monotonicity_by_metric": monotonicity_by_metric,
        "top_vs_bottom_delta_by_metric": deltas_by_metric,
    }


def add_sr_research_columns(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
    *,
    touch_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if touch_rows is None:
        touch_rows = _research_touch_rows(out, registry)
    calibration_rows = touch_rows.copy()
    if not calibration_rows.empty:
        calibration_rows = (
            calibration_rows.sort_values(
                ["row", "score", "event_id"], ascending=[True, False, True]
            )
            .groupby("row", dropna=True)
            .head(1)
            .reset_index(drop=True)
        )
    calibration = _score_quintile_summary(calibration_rows)

    out["r_all_active_zone_ids"] = str(
        sorted(lev.level_id for lev in registry.values() if _is_live_zone(lev))
    )
    out["r_level_registry_snapshot_count"] = int(len(registry))
    out["r_zone_touch_count"] = 0
    out["r_sr_touch_event_id"] = np.nan
    out["r_sr_touch_zone_id"] = np.nan
    out["r_sr_touch_side"] = np.nan
    out["r_sr_touch_score"] = np.nan
    out["r_sr_touch_zone_ids"] = ""
    for horizon in _RESEARCH_HORIZONS:
        out[f"r_sr_touch_outcome_{horizon}"] = np.nan
        out[f"r_sr_touch_mfe_atr_{horizon}"] = np.nan
        out[f"r_sr_touch_mae_atr_{horizon}"] = np.nan
    calibration_json = json.dumps(calibration, sort_keys=True)
    out["r_sr_score_quintile_calibration_json"] = calibration_json

    if touch_rows.empty:
        return out

    grouped = touch_rows.groupby("row", dropna=True)
    for row, group in grouped:
        row_idx = int(row)
        if row_idx < 0 or row_idx >= n:
            continue
        ranked = group.sort_values(["score", "event_id"], ascending=[False, True])
        best = ranked.iloc[0]
        out.at[row_idx, "r_zone_touch_count"] = int(len(group))
        out.at[row_idx, "r_sr_touch_event_id"] = float(best["event_id"])
        out.at[row_idx, "r_sr_touch_zone_id"] = float(best["zone_id"])
        out.at[row_idx, "r_sr_touch_side"] = float(best["side"])
        out.at[row_idx, "r_sr_touch_score"] = float(best["score"])
        out.at[row_idx, "r_sr_touch_zone_ids"] = ",".join(
            str(int(v)) for v in ranked["zone_id"].tolist()
        )
        for horizon in _RESEARCH_HORIZONS:
            out.at[row_idx, f"r_sr_touch_outcome_{horizon}"] = best[
                f"outcome_{horizon}"
            ]
            out.at[row_idx, f"r_sr_touch_mfe_atr_{horizon}"] = best[
                f"mfe_atr_{horizon}"
            ]
            out.at[row_idx, f"r_sr_touch_mae_atr_{horizon}"] = best[
                f"mae_atr_{horizon}"
            ]
    return out


def add_sr_levels(
    df: pd.DataFrame,
    *,
    include_research_only: bool = False,
) -> pd.DataFrame:
    if df is None:
        raise ValueError("add_sr_levels: df must not be None")
    if len(df) == 0:
        return df.copy()
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"add_sr_levels: missing required column '{col}'")
    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)
    if include_research_only:
        out = add_sr_research_columns(out, registry)
    return out
