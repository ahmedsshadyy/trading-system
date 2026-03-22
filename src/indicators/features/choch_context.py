from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns
from src.indicators.features.bos_context import (
    AFTER_DISPLACEMENT_LOOKBACK_BARS,
    AFTER_SWEEP_LOOKBACK_BARS,
    BREAK_DISTANCE_UPPER_BOUND,
    CANDLE_BODY_UPPER_BOUND,
    DISPLACEMENT_SCORE_UPPER_BOUND,
    FVG_NEAR_THRESHOLD_ATR,
    OB_NEAR_THRESHOLD_ATR,
    _clip_to_unit,
    _forward_excursions_atr,
    _forward_hold_failed_retest,
    _inside_or_near_zone,
    _recent_index_match,
    _series_or_default,
    _weighted_unit_score,
)

AFTER_WEDGE_LOOKBACK_BARS = 3

QUALITY_BREAK_DISTANCE_WEIGHT = 0.25
QUALITY_CANDLE_BODY_WEIGHT = 0.20
QUALITY_BODY_TO_RANGE_WEIGHT = 0.10
QUALITY_DISPLACEMENT_WEIGHT = 0.15
QUALITY_AGAINST_PREV_TREND_WEIGHT = 0.15
QUALITY_AFTER_STRUCTURE_LOSS_WEIGHT = 0.15

TRADEABLE_REVERSAL_ALIGNMENT_BONUS = 0.15
TRADEABLE_AFTER_SWEEP_BONUS = 0.15
TRADEABLE_AFTER_WEDGE_BONUS = 0.10
TRADEABLE_AFTER_DISPLACEMENT_BONUS = 0.10
TRADEABLE_AFTER_STRUCTURE_LOSS_BONUS = 0.10
TRADEABLE_INTO_ZONE_BONUS = 0.05

TRADEABLE_NEUTRAL_REVERSAL_ALIGNMENT_PENALTY = 0.05
TRADEABLE_BAD_REVERSAL_ALIGNMENT_PENALTY = 0.20

HOLD_FAIL_RETEST_HORIZONS = (1, 2, 3, 5)
EXCURSION_HORIZONS = (3, 5, 10)

REVERSAL_CONTEXT_COLUMNS = [
    "choch_reversal_alignment",
    "choch_after_sweep",
    "choch_after_wedge",
    "choch_after_displacement",
    "choch_into_fvg",
    "choch_into_ob",
]
FOLLOW_THROUGH_COLUMNS = [
    "choch_hold_1",
    "choch_hold_2",
    "choch_hold_3",
    "choch_hold_5",
    "choch_failed_1",
    "choch_failed_2",
    "choch_failed_3",
    "choch_failed_5",
    "choch_retest_1",
    "choch_retest_3",
    "choch_retest_5",
]
EXCURSION_COLUMNS = [
    "choch_mfe_3_atr",
    "choch_mae_3_atr",
    "choch_mfe_5_atr",
    "choch_mae_5_atr",
    "choch_mfe_10_atr",
    "choch_mae_10_atr",
]
SCORE_COLUMNS = [
    "choch_quality_score",
    "choch_tradeable_score",
]

LIVE_CHOCH_CONTEXT_COLUMNS = REVERSAL_CONTEXT_COLUMNS + SCORE_COLUMNS
RESEARCH_CHOCH_CONTEXT_COLUMNS = (
    LIVE_CHOCH_CONTEXT_COLUMNS + FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS
)


def _zones_for_direction(
    direction: int,
    bull_zones: list[tuple[float, float]],
    bear_zones: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return bull_zones if direction == 1 else bear_zones


def _reversal_alignment_value(
    direction: int,
    trend_from: float,
    bias_from: float,
) -> float:
    trend_ok = np.isfinite(trend_from) and int(trend_from) == -direction
    if not trend_ok:
        return -1.0

    if not np.isfinite(bias_from) or int(bias_from) == 0:
        return 0.0

    return 1.0 if int(bias_from) == -direction else -1.0


def _recent_wedge_alignment(
    direction: int,
    current_idx: int,
    wedge_active: np.ndarray,
    wedge_kind: np.ndarray,
    wedge_breakout_dir: np.ndarray,
) -> float:
    lo = max(0, current_idx - AFTER_WEDGE_LOOKBACK_BARS + 1)

    breakout_slice = wedge_breakout_dir[lo : current_idx + 1]
    finite_breakouts = breakout_slice[np.isfinite(breakout_slice)]
    if len(finite_breakouts) > 0 and np.any(
        finite_breakouts.astype(int, copy=False) == direction
    ):
        return 1.0

    active_slice = wedge_active[lo : current_idx + 1]
    kind_slice = wedge_kind[lo : current_idx + 1]
    aligned_kind = -direction
    for active_i, kind_i in zip(active_slice, kind_slice, strict=False):
        if int(active_i) == 1 and np.isfinite(kind_i) and int(kind_i) == aligned_kind:
            return 1.0

    return 0.0


def add_choch_context(
    df: pd.DataFrame,
    *,
    include_forward_diagnostics: bool = True,
) -> pd.DataFrame:
    """Add CHoCH event context and optional research-only forward diagnostics."""
    out = df.copy()

    require_columns(out, {"close", "high", "low", "choch_bull", "choch_bear"})

    n = len(out)
    if n == 0:
        return out

    close_arr = out["close"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    atr = get_atr_array(out, 14)

    choch_bull = out["choch_bull"].to_numpy(dtype=np.int8)
    choch_bear = out["choch_bear"].to_numpy(dtype=np.int8)
    choch_direction = _series_or_default(out, "choch_direction", dtype=float)
    if not np.isfinite(choch_direction).any():
        choch_direction = choch_bull.astype(float) - choch_bear.astype(float)
    choch_level = _series_or_default(
        out, "choch_level", "choch_source_price", dtype=float
    )

    trend_state = _series_or_default(out, "trend_state", dtype=float)
    trend_bias_state = _series_or_default(
        out, "trend_bias_state", "trend_bias", dtype=float
    )

    choch_trend_state_from = _series_or_default(
        out, "choch_trend_state_from", dtype=float
    )
    choch_bias_state_from = _series_or_default(
        out, "choch_bias_state_from", dtype=float
    )
    choch_against_prev_trend = _series_or_default(
        out, "choch_against_prev_trend", default=0, dtype=float
    )
    choch_after_structure_loss = _series_or_default(
        out, "choch_after_structure_loss", default=0, dtype=float
    )

    wedge_active = _series_or_default(out, "wedge_active", default=0, dtype=float)
    wedge_kind = _series_or_default(out, "wedge_kind", dtype=float)
    wedge_breakout_dir = _series_or_default(out, "wedge_breakout_dir", dtype=float)

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

    choch_break_distance_atr = _series_or_default(
        out, "choch_break_distance_atr", dtype=float
    )
    choch_candle_body_atr = _series_or_default(
        out, "choch_candle_body_atr", dtype=float
    )
    choch_body_to_range = _series_or_default(out, "choch_body_to_range", dtype=float)
    choch_displacement_score = _series_or_default(
        out, "choch_displacement_score", dtype=float
    )

    choch_event_mask = (choch_bull == 1) | (choch_bear == 1) | (choch_direction != 0)

    result_arrays: dict[str, np.ndarray] = {
        name: np.full(n, np.nan) for name in RESEARCH_CHOCH_CONTEXT_COLUMNS
    }

    bull_ob_zones: list[tuple[float, float]] = []
    bear_ob_zones: list[tuple[float, float]] = []
    bull_fvg_zones: list[tuple[float, float]] = []
    bear_fvg_zones: list[tuple[float, float]] = []
    bull_sweep_indices: list[int] = []
    bear_sweep_indices: list[int] = []
    bull_displacement_indices: list[int] = []
    bear_displacement_indices: list[int] = []

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
        if (
            int(fvg_bear[i]) == 1
            and np.isfinite(fvg_bear_low[i])
            and np.isfinite(fvg_bear_high[i])
        ):
            bear_fvg_zones.append((float(fvg_bear_low[i]), float(fvg_bear_high[i])))

        if int(sweep_low[i]) == 1:
            bull_sweep_indices.append(i)
        if int(sweep_high[i]) == 1:
            bear_sweep_indices.append(i)

        if int(displacement_candle[i]) == 1:
            if int(displacement_direction[i]) == 1:
                bull_displacement_indices.append(i)
            elif int(displacement_direction[i]) == -1:
                bear_displacement_indices.append(i)

        if not choch_event_mask[i]:
            continue

        direction = int(choch_direction[i])
        if direction == 0:
            direction = (
                1 if int(choch_bull[i]) == 1 else -1 if int(choch_bear[i]) == 1 else 0
            )
        if direction == 0:
            continue

        atr_i = atr[i]
        level = choch_level[i]
        close_i = close_arr[i]

        trend_from = (
            choch_trend_state_from[i]
            if np.isfinite(choch_trend_state_from[i])
            else trend_state[i]
        )
        bias_from = (
            choch_bias_state_from[i]
            if np.isfinite(choch_bias_state_from[i])
            else trend_bias_state[i]
        )
        reversal_alignment = _reversal_alignment_value(direction, trend_from, bias_from)
        after_sweep = (
            1.0
            if _recent_index_match(
                bull_sweep_indices if direction == 1 else bear_sweep_indices,
                i,
                AFTER_SWEEP_LOOKBACK_BARS,
            )
            else 0.0
        )
        after_wedge = _recent_wedge_alignment(
            direction,
            i,
            wedge_active,
            wedge_kind,
            wedge_breakout_dir,
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
        into_fvg = (
            1.0
            if _inside_or_near_zone(
                close_i,
                atr_i,
                _zones_for_direction(direction, bull_fvg_zones, bear_fvg_zones),
                FVG_NEAR_THRESHOLD_ATR,
            )
            else 0.0
        )
        into_ob = (
            1.0
            if _inside_or_near_zone(
                close_i,
                atr_i,
                _zones_for_direction(direction, bull_ob_zones, bear_ob_zones),
                OB_NEAR_THRESHOLD_ATR,
            )
            else 0.0
        )

        quality_score = _weighted_unit_score(
            [
                (
                    _clip_to_unit(
                        choch_break_distance_atr[i],
                        upper_bound=BREAK_DISTANCE_UPPER_BOUND,
                    ),
                    QUALITY_BREAK_DISTANCE_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        choch_candle_body_atr[i],
                        upper_bound=CANDLE_BODY_UPPER_BOUND,
                    ),
                    QUALITY_CANDLE_BODY_WEIGHT,
                ),
                (
                    (
                        np.clip(choch_body_to_range[i], 0.0, 1.0)
                        if np.isfinite(choch_body_to_range[i])
                        else np.nan
                    ),
                    QUALITY_BODY_TO_RANGE_WEIGHT,
                ),
                (
                    _clip_to_unit(
                        choch_displacement_score[i],
                        upper_bound=DISPLACEMENT_SCORE_UPPER_BOUND,
                    ),
                    QUALITY_DISPLACEMENT_WEIGHT,
                ),
                (
                    (
                        np.clip(choch_against_prev_trend[i], 0.0, 1.0)
                        if np.isfinite(choch_against_prev_trend[i])
                        else np.nan
                    ),
                    QUALITY_AGAINST_PREV_TREND_WEIGHT,
                ),
                (
                    (
                        np.clip(choch_after_structure_loss[i], 0.0, 1.0)
                        if np.isfinite(choch_after_structure_loss[i])
                        else np.nan
                    ),
                    QUALITY_AFTER_STRUCTURE_LOSS_WEIGHT,
                ),
            ]
        )

        tradeable_score = quality_score
        if np.isfinite(tradeable_score):
            if reversal_alignment == 1:
                tradeable_score += TRADEABLE_REVERSAL_ALIGNMENT_BONUS
            elif reversal_alignment == 0:
                tradeable_score -= TRADEABLE_NEUTRAL_REVERSAL_ALIGNMENT_PENALTY
            else:
                tradeable_score -= TRADEABLE_BAD_REVERSAL_ALIGNMENT_PENALTY

            if after_sweep == 1:
                tradeable_score += TRADEABLE_AFTER_SWEEP_BONUS
            if after_wedge == 1:
                tradeable_score += TRADEABLE_AFTER_WEDGE_BONUS
            if after_displacement == 1:
                tradeable_score += TRADEABLE_AFTER_DISPLACEMENT_BONUS
            if int(choch_after_structure_loss[i]) == 1:
                tradeable_score += TRADEABLE_AFTER_STRUCTURE_LOSS_BONUS
            if into_fvg == 1 or into_ob == 1:
                tradeable_score += TRADEABLE_INTO_ZONE_BONUS

            tradeable_score = float(np.clip(tradeable_score, 0.0, 1.0))

        result_arrays["choch_reversal_alignment"][i] = reversal_alignment
        result_arrays["choch_after_sweep"][i] = after_sweep
        result_arrays["choch_after_wedge"][i] = after_wedge
        result_arrays["choch_after_displacement"][i] = after_displacement
        result_arrays["choch_into_fvg"][i] = into_fvg
        result_arrays["choch_into_ob"][i] = into_ob
        result_arrays["choch_quality_score"][i] = quality_score
        result_arrays["choch_tradeable_score"][i] = tradeable_score

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
            result_arrays[f"choch_hold_{horizon}"][i] = hold
            result_arrays[f"choch_failed_{horizon}"][i] = failed
            if horizon in (1, 3, 5):
                result_arrays[f"choch_retest_{horizon}"][i] = retest

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
            result_arrays[f"choch_mfe_{horizon}_atr"][i] = mfe
            result_arrays[f"choch_mae_{horizon}_atr"][i] = mae

    for name in LIVE_CHOCH_CONTEXT_COLUMNS:
        out[name] = result_arrays[name]

    if include_forward_diagnostics:
        for name in FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS:
            out[name] = result_arrays[name]

    return out


__all__ = [
    "LIVE_CHOCH_CONTEXT_COLUMNS",
    "RESEARCH_CHOCH_CONTEXT_COLUMNS",
    "add_choch_context",
]
