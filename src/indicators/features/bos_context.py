from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns

AFTER_SWEEP_LOOKBACK_BARS = 4
AFTER_DISPLACEMENT_LOOKBACK_BARS = 3
AFTER_FVG_LOOKBACK_BARS = 3
NEAR_WEDGE_LOOKBACK_BARS = 3

OB_NEAR_THRESHOLD_ATR = 0.20
FVG_NEAR_THRESHOLD_ATR = 0.20
EQHL_NEAR_THRESHOLD_ATR = 0.25
LIQUIDITY_NEAR_THRESHOLD_ATR = 0.30

BREAK_DISTANCE_UPPER_BOUND = 2.0
CANDLE_BODY_UPPER_BOUND = 2.0
DISPLACEMENT_SCORE_UPPER_BOUND = 1.0
SOURCE_STRENGTH_UPPER_BOUND = 3.0
SOURCE_PROMINENCE_UPPER_BOUND = 3.0
SOURCE_AGE_FRESHNESS_UPPER_BOUND = 50.0

QUALITY_BREAK_DISTANCE_WEIGHT = 0.25
QUALITY_CANDLE_BODY_WEIGHT = 0.20
QUALITY_BODY_TO_RANGE_WEIGHT = 0.10
QUALITY_DISPLACEMENT_WEIGHT = 0.15
QUALITY_SOURCE_STRENGTH_WEIGHT = 0.15
QUALITY_SOURCE_PROMINENCE_WEIGHT = 0.10
QUALITY_SOURCE_AGE_WEIGHT = 0.05

TRADEABLE_TREND_ALIGNMENT_BONUS = 0.15
TRADEABLE_AFTER_SWEEP_BONUS = 0.10
TRADEABLE_AFTER_DISPLACEMENT_BONUS = 0.10
TRADEABLE_NEAR_WEDGE_BONUS = 0.05
TRADEABLE_INTO_ZONE_BONUS = 0.05

TRADEABLE_AGAINST_TREND_PENALTY = 0.20
TRADEABLE_NEUTRAL_TREND_PENALTY = 0.05
TRADEABLE_NEAR_LIQUIDITY_PENALTY = 0.10
TRADEABLE_NEAR_EQHL_PENALTY = 0.05

HOLD_FAIL_RETEST_HORIZONS = (1, 2, 3, 5)
EXCURSION_HORIZONS = (3, 5, 10)

TREND_CONTEXT_COLUMNS = [
    "bos_trend_alignment",
    "bos_against_prev_trend",
    "bos_in_neutral_trend",
    "bos_trend_state_on_event",
    "bos_trend_bias_state_on_event",
]
STRUCTURAL_CONTEXT_COLUMNS = [
    "bos_near_wedge",
    "bos_wedge_kind",
    "bos_after_sweep",
    "bos_after_displacement",
    "bos_after_fvg",
    "bos_into_ob",
    "bos_into_fvg",
    "bos_near_eqhl",
    "bos_near_liquidity",
]
FOLLOW_THROUGH_COLUMNS = [
    "bos_hold_1",
    "bos_hold_2",
    "bos_hold_3",
    "bos_hold_5",
    "bos_failed_1",
    "bos_failed_2",
    "bos_failed_3",
    "bos_failed_5",
    "bos_retest_1",
    "bos_retest_3",
    "bos_retest_5",
]
EXCURSION_COLUMNS = [
    "bos_mfe_3_atr",
    "bos_mae_3_atr",
    "bos_mfe_5_atr",
    "bos_mae_5_atr",
    "bos_mfe_10_atr",
    "bos_mae_10_atr",
]
SCORE_COLUMNS = [
    "bos_quality_score",
    "bos_tradeable_score",
]

LIVE_BOS_CONTEXT_COLUMNS = (
    TREND_CONTEXT_COLUMNS + STRUCTURAL_CONTEXT_COLUMNS + SCORE_COLUMNS
)
RESEARCH_BOS_CONTEXT_COLUMNS = (
    LIVE_BOS_CONTEXT_COLUMNS + FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS
)


def _series_or_default(
    df: pd.DataFrame,
    *names: str,
    default: float | int | None = np.nan,
    dtype: type = float,
) -> np.ndarray:
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=dtype)
    return np.full(len(df), default, dtype=dtype)


def _clip_to_unit(value: float, *, upper_bound: float) -> float:
    if not np.isfinite(value) or upper_bound <= 0:
        return np.nan
    return float(np.clip(value / upper_bound, 0.0, 1.0))


def _freshness_from_age(age: float) -> float:
    if not np.isfinite(age):
        return np.nan
    return float(
        np.clip(1.0 - (max(age, 0.0) / SOURCE_AGE_FRESHNESS_UPPER_BOUND), 0.0, 1.0)
    )


def _weighted_unit_score(components: list[tuple[float, float]]) -> float:
    total = 0.0
    used_weight = 0.0
    for value, weight in components:
        if np.isfinite(value):
            total += value * weight
            used_weight += weight
    if used_weight <= 0:
        return np.nan
    return float(np.clip(total / used_weight, 0.0, 1.0))


def _inside_or_near_zone(
    close_value: float,
    atr_value: float,
    zones: list[tuple[float, float]],
    threshold_atr: float,
) -> bool:
    if not np.isfinite(close_value) or not np.isfinite(atr_value) or atr_value <= 0:
        return False

    for lower, upper in zones:
        lo = min(lower, upper)
        hi = max(lower, upper)
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if lo <= close_value <= hi:
            return True
        dist = min(abs(close_value - lo), abs(close_value - hi))
        if dist / atr_value <= threshold_atr:
            return True
    return False


def _recent_index_match(indices: list[int], current_idx: int, lookback: int) -> bool:
    cutoff = current_idx - lookback + 1
    return any(idx >= cutoff for idx in indices)


def _same_direction_ob_zones(
    direction: int,
    bull_zones: list[tuple[float, float]],
    bear_zones: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return bull_zones if direction == 1 else bear_zones


def _same_direction_fvg_zones(
    direction: int,
    bull_zones: list[tuple[float, float]],
    bear_zones: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return bull_zones if direction == 1 else bear_zones


def _eqhl_levels_in_path(
    direction: int,
    close_value: float,
    levels: list[float],
) -> list[float]:
    if direction == 1:
        return [
            level for level in levels if np.isfinite(level) and level >= close_value
        ]
    return [level for level in levels if np.isfinite(level) and level <= close_value]


def _liquidity_levels_in_path(
    direction: int,
    close_value: float,
    row_candidates: list[float],
    active_eq_high_levels: list[float],
    active_eq_low_levels: list[float],
) -> list[float]:
    levels = []
    if direction == 1:
        levels.extend(
            [
                level
                for level in active_eq_high_levels
                if np.isfinite(level) and level >= close_value
            ]
        )
        levels.extend(
            [
                level
                for level in row_candidates
                if np.isfinite(level) and level >= close_value
            ]
        )
    else:
        levels.extend(
            [
                level
                for level in active_eq_low_levels
                if np.isfinite(level) and level <= close_value
            ]
        )
        levels.extend(
            [
                level
                for level in row_candidates
                if np.isfinite(level) and level <= close_value
            ]
        )
    return levels


def _nearest_level_distance_atr(
    close_value: float,
    atr_value: float,
    levels: list[float],
) -> float:
    if not np.isfinite(close_value) or not np.isfinite(atr_value) or atr_value <= 0:
        return np.nan
    valid = [abs(close_value - level) for level in levels if np.isfinite(level)]
    if not valid:
        return np.nan
    return float(min(valid) / atr_value)


def _forward_hold_failed_retest(
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    event_idx: int,
    level: float,
    direction: int,
    horizon: int,
) -> tuple[float, float, float]:
    end_idx = event_idx + horizon
    if end_idx >= len(close_arr):
        return np.nan, np.nan, np.nan

    hold = 1.0
    failed = 0.0
    retest = 0.0

    for j in range(event_idx + 1, end_idx + 1):
        if direction == 1:
            if close_arr[j] < level:
                hold = 0.0
                failed = 1.0
                return hold, failed, retest
            if low_arr[j] <= level and close_arr[j] >= level:
                retest = 1.0
        else:
            if close_arr[j] > level:
                hold = 0.0
                failed = 1.0
                return hold, failed, retest
            if high_arr[j] >= level and close_arr[j] <= level:
                retest = 1.0

    return hold, failed, retest


def _forward_excursions_atr(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    event_idx: int,
    level: float,
    direction: int,
    atr_value: float,
    horizon: int,
) -> tuple[float, float]:
    end_idx = event_idx + horizon
    if end_idx >= len(high_arr) or not np.isfinite(atr_value) or atr_value <= 0:
        return np.nan, np.nan

    future_high = np.nanmax(high_arr[event_idx + 1 : end_idx + 1])
    future_low = np.nanmin(low_arr[event_idx + 1 : end_idx + 1])

    if direction == 1:
        mfe = (future_high - level) / atr_value
        mae = (level - future_low) / atr_value
    else:
        mfe = (level - future_low) / atr_value
        mae = (future_high - level) / atr_value

    return float(max(mfe, 0.0)), float(max(mae, 0.0))


def add_bos_context(
    df: pd.DataFrame,
    *,
    include_forward_diagnostics: bool = True,
) -> pd.DataFrame:
    """Add BOS event context and optional forward diagnostics.

    The function is causal for context columns and deliberately non-causal for
    forward diagnostics. When ``include_forward_diagnostics`` is False, all
    forward-looking columns are omitted entirely so the output remains live-safe.
    """
    out = df.copy()

    require_columns(out, {"close", "high", "low", "bos_bull", "bos_bear"})

    n = len(out)
    if n == 0:
        return out

    close_arr = out["close"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    atr = get_atr_array(out, 14)

    bos_bull = out["bos_bull"].to_numpy(dtype=np.int8)
    bos_bear = out["bos_bear"].to_numpy(dtype=np.int8)
    bos_direction = _series_or_default(out, "bos_direction", dtype=float)
    if not np.isfinite(bos_direction).any():
        bos_direction = bos_bull.astype(float) - bos_bear.astype(float)
    bos_level = _series_or_default(out, "bos_level", "bos_source_price", dtype=float)

    trend_state = _series_or_default(out, "trend_state", dtype=float)
    trend_bias_state = _series_or_default(
        out, "trend_bias_state", "trend_bias", dtype=float
    )

    wedge_active = _series_or_default(out, "wedge_active", default=0, dtype=float)
    wedge_kind_src = _series_or_default(out, "wedge_kind", dtype=float)
    wedge_breakout_idx = _series_or_default(out, "wedge_breakout_idx", dtype=float)

    sweep_high = _series_or_default(out, "sweep_high", default=0, dtype=float)
    sweep_low = _series_or_default(out, "sweep_low", default=0, dtype=float)

    displacement_candle = _series_or_default(
        out, "displacement_candle", default=0, dtype=float
    )
    displacement_direction = _series_or_default(
        out, "displacement_direction", dtype=float
    )

    fvg_bull = _series_or_default(out, "fvg_bull", default=0, dtype=float)
    fvg_bear = _series_or_default(out, "fvg_bear", default=0, dtype=float)
    fvg_bull_low = _series_or_default(out, "fvg_bull_low", dtype=float)
    fvg_bull_high = _series_or_default(out, "fvg_bull_high", dtype=float)
    fvg_bear_low = _series_or_default(out, "fvg_bear_low", dtype=float)
    fvg_bear_high = _series_or_default(out, "fvg_bear_high", dtype=float)

    ob_bull = _series_or_default(out, "ob_bull", default=0, dtype=float)
    ob_bear = _series_or_default(out, "ob_bear", default=0, dtype=float)
    ob_bull_low = _series_or_default(out, "ob_bull_low", dtype=float)
    ob_bull_high = _series_or_default(out, "ob_bull_high", dtype=float)
    ob_bear_low = _series_or_default(out, "ob_bear_low", dtype=float)
    ob_bear_high = _series_or_default(out, "ob_bear_high", dtype=float)

    equal_highs_level = _series_or_default(out, "equal_highs_level", dtype=float)
    equal_highs_active = _series_or_default(out, "equal_highs_active", dtype=float)
    equal_highs_cluster_id = _series_or_default(
        out, "equal_highs_cluster_id", default=np.nan, dtype=float
    )
    equal_lows_level = _series_or_default(out, "equal_lows_level", dtype=float)
    equal_lows_active = _series_or_default(out, "equal_lows_active", dtype=float)
    equal_lows_cluster_id = _series_or_default(
        out, "equal_lows_cluster_id", default=np.nan, dtype=float
    )

    last_swing_high = _series_or_default(out, "last_swing_high", dtype=float)
    last_swing_low = _series_or_default(out, "last_swing_low", dtype=float)
    asian_high = _series_or_default(out, "asian_high", dtype=float)
    asian_low = _series_or_default(out, "asian_low", dtype=float)
    prev_day_high = _series_or_default(out, "prev_day_high", dtype=float)
    prev_day_low = _series_or_default(out, "prev_day_low", dtype=float)
    prev_week_high = _series_or_default(out, "prev_week_high", dtype=float)
    prev_week_low = _series_or_default(out, "prev_week_low", dtype=float)
    vp_vah = _series_or_default(out, "vp_vah", dtype=float)
    vp_val = _series_or_default(out, "vp_val", dtype=float)

    bos_break_distance_atr = _series_or_default(
        out, "bos_break_distance_atr", dtype=float
    )
    bos_candle_body_atr = _series_or_default(out, "bos_candle_body_atr", dtype=float)
    bos_body_to_range = _series_or_default(out, "bos_body_to_range", dtype=float)
    bos_displacement_score = _series_or_default(
        out, "bos_displacement_score", dtype=float
    )
    bos_source_strength = _series_or_default(out, "bos_source_strength", dtype=float)
    bos_source_prominence_atr = _series_or_default(
        out, "bos_source_prominence_atr", dtype=float
    )
    bos_source_age = _series_or_default(out, "bos_source_age", dtype=float)

    bos_event_mask = (bos_bull == 1) | (bos_bear == 1) | (bos_direction != 0)

    result_arrays: dict[str, np.ndarray] = {
        name: np.full(n, np.nan) for name in RESEARCH_BOS_CONTEXT_COLUMNS
    }

    bull_ob_zones: list[tuple[float, float]] = []
    bear_ob_zones: list[tuple[float, float]] = []
    bull_fvg_zones: list[tuple[float, float]] = []
    bear_fvg_zones: list[tuple[float, float]] = []
    bull_sweep_indices: list[int] = []
    bear_sweep_indices: list[int] = []
    bull_displacement_indices: list[int] = []
    bear_displacement_indices: list[int] = []
    bull_fvg_indices: list[int] = []
    bear_fvg_indices: list[int] = []
    active_eq_high_clusters: dict[int, float] = {}
    active_eq_low_clusters: dict[int, float] = {}

    for i in range(n):
        if (
            int(ob_bull[i]) == 1
            and np.isfinite(ob_bull_low[i])
            and np.isfinite(ob_bull_high[i])
        ):
            bull_ob_zones.append((float(ob_bull_low[i]), float(ob_bull_high[i])))
        if (
            int(ob_bear[i]) == 1
            and np.isfinite(ob_bear_low[i])
            and np.isfinite(ob_bear_high[i])
        ):
            bear_ob_zones.append((float(ob_bear_low[i]), float(ob_bear_high[i])))

        if (
            int(fvg_bull[i]) == 1
            and np.isfinite(fvg_bull_low[i])
            and np.isfinite(fvg_bull_high[i])
        ):
            bull_fvg_zones.append((float(fvg_bull_low[i]), float(fvg_bull_high[i])))
            bull_fvg_indices.append(i)
        if (
            int(fvg_bear[i]) == 1
            and np.isfinite(fvg_bear_low[i])
            and np.isfinite(fvg_bear_high[i])
        ):
            bear_fvg_zones.append((float(fvg_bear_low[i]), float(fvg_bear_high[i])))
            bear_fvg_indices.append(i)

        if int(sweep_low[i]) == 1:
            bull_sweep_indices.append(i)
        if int(sweep_high[i]) == 1:
            bear_sweep_indices.append(i)

        if int(displacement_candle[i]) == 1:
            if int(displacement_direction[i]) == 1:
                bull_displacement_indices.append(i)
            elif int(displacement_direction[i]) == -1:
                bear_displacement_indices.append(i)

        if np.isfinite(equal_highs_cluster_id[i]) and np.isfinite(equal_highs_level[i]):
            cluster_id = int(equal_highs_cluster_id[i])
            if int(equal_highs_active[i]) == 1:
                active_eq_high_clusters[cluster_id] = float(equal_highs_level[i])
            elif cluster_id in active_eq_high_clusters:
                active_eq_high_clusters.pop(cluster_id)
        elif int(equal_highs_active[i]) == 1 and np.isfinite(equal_highs_level[i]):
            active_eq_high_clusters[i] = float(equal_highs_level[i])

        if np.isfinite(equal_lows_cluster_id[i]) and np.isfinite(equal_lows_level[i]):
            cluster_id = int(equal_lows_cluster_id[i])
            if int(equal_lows_active[i]) == 1:
                active_eq_low_clusters[cluster_id] = float(equal_lows_level[i])
            elif cluster_id in active_eq_low_clusters:
                active_eq_low_clusters.pop(cluster_id)
        elif int(equal_lows_active[i]) == 1 and np.isfinite(equal_lows_level[i]):
            active_eq_low_clusters[i] = float(equal_lows_level[i])

        if not bos_event_mask[i]:
            continue

        direction = int(bos_direction[i])
        if direction == 0:
            direction = (
                1 if int(bos_bull[i]) == 1 else -1 if int(bos_bear[i]) == 1 else 0
            )
        if direction == 0:
            continue

        atr_i = atr[i]
        level = bos_level[i]
        close_i = close_arr[i]

        trend_i = trend_state[i]
        if np.isfinite(trend_i):
            trend_alignment = (
                1.0 if int(trend_i) == direction else 0.0 if int(trend_i) == 0 else -1.0
            )
            against_prev_trend = 1.0 if int(trend_i) == -direction else 0.0
            in_neutral_trend = 1.0 if int(trend_i) == 0 else 0.0
        else:
            trend_alignment = np.nan
            against_prev_trend = 0.0
            in_neutral_trend = 0.0

        recent_wedge_break = (
            np.isfinite(wedge_breakout_idx[i])
            and int(wedge_breakout_idx[i]) >= i - NEAR_WEDGE_LOOKBACK_BARS
        )
        near_wedge = 1.0 if int(wedge_active[i]) == 1 or recent_wedge_break else 0.0

        wedge_kind = (
            wedge_kind_src[i]
            if np.isfinite(wedge_kind_src[i]) and wedge_kind_src[i] != 0
            else np.nan
        )
        if not np.isfinite(wedge_kind):
            start = max(0, i - NEAR_WEDGE_LOOKBACK_BARS)
            recent_kinds = wedge_kind_src[start : i + 1]
            non_zero = recent_kinds[np.isfinite(recent_kinds) & (recent_kinds != 0)]
            if len(non_zero) > 0:
                wedge_kind = float(non_zero[-1])

        after_sweep = (
            1.0
            if _recent_index_match(
                bull_sweep_indices if direction == 1 else bear_sweep_indices,
                i,
                AFTER_SWEEP_LOOKBACK_BARS,
            )
            else 0.0
        )
        after_displacement = (
            1.0
            if _recent_index_match(
                (
                    bull_displacement_indices
                    if direction == 1
                    else bear_displacement_indices
                ),
                i,
                AFTER_DISPLACEMENT_LOOKBACK_BARS,
            )
            else 0.0
        )
        after_fvg = (
            1.0
            if _recent_index_match(
                bull_fvg_indices if direction == 1 else bear_fvg_indices,
                i,
                AFTER_FVG_LOOKBACK_BARS,
            )
            else 0.0
        )

        into_ob = (
            1.0
            if _inside_or_near_zone(
                close_i,
                atr_i,
                _same_direction_ob_zones(direction, bull_ob_zones, bear_ob_zones),
                OB_NEAR_THRESHOLD_ATR,
            )
            else 0.0
        )
        into_fvg = (
            1.0
            if _inside_or_near_zone(
                close_i,
                atr_i,
                _same_direction_fvg_zones(direction, bull_fvg_zones, bear_fvg_zones),
                FVG_NEAR_THRESHOLD_ATR,
            )
            else 0.0
        )

        eqhl_levels = _eqhl_levels_in_path(
            direction,
            close_i,
            (
                list(active_eq_high_clusters.values())
                if direction == 1
                else list(active_eq_low_clusters.values())
            ),
        )
        eqhl_distance_atr = _nearest_level_distance_atr(close_i, atr_i, eqhl_levels)
        near_eqhl = (
            1.0
            if np.isfinite(eqhl_distance_atr)
            and eqhl_distance_atr <= EQHL_NEAR_THRESHOLD_ATR
            else 0.0
        )

        liquidity_candidates = (
            [
                last_swing_high[i],
                asian_high[i],
                prev_day_high[i],
                prev_week_high[i],
                vp_vah[i],
            ]
            if direction == 1
            else [
                last_swing_low[i],
                asian_low[i],
                prev_day_low[i],
                prev_week_low[i],
                vp_val[i],
            ]
        )
        liquidity_levels = _liquidity_levels_in_path(
            direction,
            close_i,
            liquidity_candidates,
            list(active_eq_high_clusters.values()),
            list(active_eq_low_clusters.values()),
        )
        liquidity_distance_atr = _nearest_level_distance_atr(
            close_i,
            atr_i,
            liquidity_levels,
        )
        near_liquidity = (
            1.0
            if np.isfinite(liquidity_distance_atr)
            and liquidity_distance_atr <= LIQUIDITY_NEAR_THRESHOLD_ATR
            else 0.0
        )

        quality_score = _weighted_unit_score(
            [
                (
                    _clip_to_unit(
                        bos_break_distance_atr[i],
                        upper_bound=BREAK_DISTANCE_UPPER_BOUND,
                    ),
                    QUALITY_BREAK_DISTANCE_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        bos_candle_body_atr[i],
                        upper_bound=CANDLE_BODY_UPPER_BOUND,
                    ),
                    QUALITY_CANDLE_BODY_WEIGHT,
                ),
                (
                    (
                        np.clip(bos_body_to_range[i], 0.0, 1.0)
                        if np.isfinite(bos_body_to_range[i])
                        else np.nan
                    ),
                    QUALITY_BODY_TO_RANGE_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        bos_displacement_score[i],
                        upper_bound=DISPLACEMENT_SCORE_UPPER_BOUND,
                    ),
                    QUALITY_DISPLACEMENT_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        bos_source_strength[i],
                        upper_bound=SOURCE_STRENGTH_UPPER_BOUND,
                    ),
                    QUALITY_SOURCE_STRENGTH_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        bos_source_prominence_atr[i],
                        upper_bound=SOURCE_PROMINENCE_UPPER_BOUND,
                    ),
                    QUALITY_SOURCE_PROMINENCE_WEIGHT,
                ),
                (
                    _freshness_from_age(bos_source_age[i]),
                    QUALITY_SOURCE_AGE_WEIGHT,
                ),
            ]
        )

        tradeable_score = quality_score
        if np.isfinite(tradeable_score):
            if trend_alignment == 1:
                tradeable_score += TRADEABLE_TREND_ALIGNMENT_BONUS
            if after_sweep == 1:
                tradeable_score += TRADEABLE_AFTER_SWEEP_BONUS
            if after_displacement == 1:
                tradeable_score += TRADEABLE_AFTER_DISPLACEMENT_BONUS
            if near_wedge == 1:
                tradeable_score += TRADEABLE_NEAR_WEDGE_BONUS
            if into_fvg == 1 or into_ob == 1:
                tradeable_score += TRADEABLE_INTO_ZONE_BONUS

            if against_prev_trend == 1:
                tradeable_score -= TRADEABLE_AGAINST_TREND_PENALTY
            if in_neutral_trend == 1:
                tradeable_score -= TRADEABLE_NEUTRAL_TREND_PENALTY
            if near_liquidity == 1:
                tradeable_score -= TRADEABLE_NEAR_LIQUIDITY_PENALTY
            if near_eqhl == 1:
                tradeable_score -= TRADEABLE_NEAR_EQHL_PENALTY

            tradeable_score = float(np.clip(tradeable_score, 0.0, 1.0))

        result_arrays["bos_trend_alignment"][i] = trend_alignment
        result_arrays["bos_against_prev_trend"][i] = against_prev_trend
        result_arrays["bos_in_neutral_trend"][i] = in_neutral_trend
        result_arrays["bos_trend_state_on_event"][i] = trend_i
        result_arrays["bos_trend_bias_state_on_event"][i] = trend_bias_state[i]

        result_arrays["bos_near_wedge"][i] = near_wedge
        result_arrays["bos_wedge_kind"][i] = wedge_kind
        result_arrays["bos_after_sweep"][i] = after_sweep
        result_arrays["bos_after_displacement"][i] = after_displacement
        result_arrays["bos_after_fvg"][i] = after_fvg
        result_arrays["bos_into_ob"][i] = into_ob
        result_arrays["bos_into_fvg"][i] = into_fvg
        result_arrays["bos_near_eqhl"][i] = near_eqhl
        result_arrays["bos_near_liquidity"][i] = near_liquidity
        result_arrays["bos_quality_score"][i] = quality_score
        result_arrays["bos_tradeable_score"][i] = tradeable_score

        if not include_forward_diagnostics:
            continue

        if not np.isfinite(level):
            continue

        for horizon in HOLD_FAIL_RETEST_HORIZONS:
            hold, failed, retest = _forward_hold_failed_retest(
                close_arr,
                high_arr,
                low_arr,
                i,
                float(level),
                direction,
                horizon,
            )
            result_arrays[f"bos_hold_{horizon}"][i] = hold
            result_arrays[f"bos_failed_{horizon}"][i] = failed
            if horizon in (1, 3, 5):
                result_arrays[f"bos_retest_{horizon}"][i] = retest

        for horizon in EXCURSION_HORIZONS:
            mfe, mae = _forward_excursions_atr(
                high_arr,
                low_arr,
                i,
                float(level),
                direction,
                atr_i,
                horizon,
            )
            result_arrays[f"bos_mfe_{horizon}_atr"][i] = mfe
            result_arrays[f"bos_mae_{horizon}_atr"][i] = mae

    for name in LIVE_BOS_CONTEXT_COLUMNS:
        out[name] = result_arrays[name]

    if include_forward_diagnostics:
        for name in FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS:
            out[name] = result_arrays[name]

    return out


__all__ = [
    "LIVE_BOS_CONTEXT_COLUMNS",
    "RESEARCH_BOS_CONTEXT_COLUMNS",
    "add_bos_context",
]
