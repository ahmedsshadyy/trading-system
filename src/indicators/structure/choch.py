"""
Canonical causal CHoCH detector.

CHoCH uses the same causal break mechanics as BOS, but filters for breaks
against the current strict trend_state.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.validators import require_columns
from src.indicators.structure._break_events import _compute_structural_breaks

CHOCH_BREAK_SUFFIXES = [
    "bull",
    "bear",
    "direction",
    "event_id",
    "source_side",
    "source_idx",
    "source_price",
    "level",
    "close_break_bull",
    "close_break_bear",
    "wick_break_bull",
    "wick_break_bear",
    "raw_candidate_bull",
    "raw_candidate_bear",
    "pass_source_age_bull",
    "pass_source_age_bear",
    "pass_break_distance_bull",
    "pass_break_distance_bear",
    "pass_body_bull",
    "pass_body_bear",
    "pass_source_strength_bull",
    "pass_source_strength_bear",
    "pass_trend_bull",
    "pass_trend_bear",
    "break_distance",
    "break_distance_atr",
    "candle_body_atr",
    "candle_range_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "body_to_range",
    "close_location",
    "gap_from_level_atr",
    "displacement_score",
]


def add_choch(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_source_age_bars: int = 0,
    max_source_age_bars: int | None = None,
    min_break_distance_atr: float = 0.0,
    min_body_atr: float = 0.0,
    min_source_strength: float | None = None,
    structure_loss_lookback_bars: int = 3,
) -> pd.DataFrame:
    """Detect sparse canonical CHoCH events from confirmed swing levels."""
    out = df.copy()

    require_columns(
        out,
        {
            "trend_state",
            "trend_bias_state",
            "trend_structure_loss_bull",
            "trend_structure_loss_bear",
        },
    )

    break_result = _compute_structural_breaks(
        out,
        atr_length=atr_length,
        min_source_age_bars=min_source_age_bars,
        max_source_age_bars=max_source_age_bars,
        min_break_distance_atr=min_break_distance_atr,
        min_body_atr=min_body_atr,
        min_source_strength=min_source_strength,
        trend_filter="against",
        allow_neutral_trend_breaks=False,
    )

    if not break_result:
        return out

    for suffix in CHOCH_BREAK_SUFFIXES:
        out[f"choch_{suffix}"] = break_result[suffix]

    n = len(out)
    direction = break_result["direction"]
    event_mask = (break_result["bull"] == 1) | (break_result["bear"] == 1)

    trend_state = out["trend_state"].to_numpy(dtype=float)
    trend_bias_state = out["trend_bias_state"].to_numpy(dtype=float)
    structure_loss_bull = out["trend_structure_loss_bull"].to_numpy(dtype=np.int8)
    structure_loss_bear = out["trend_structure_loss_bear"].to_numpy(dtype=np.int8)

    choch_trend_state_from = np.zeros(n, dtype=np.int8)
    choch_trend_state_to = np.zeros(n, dtype=np.int8)
    choch_bias_state_from = np.zeros(n, dtype=np.int8)
    choch_bias_state_to = np.zeros(n, dtype=np.int8)
    choch_against_prev_trend = np.zeros(n, dtype=np.int8)
    choch_after_structure_loss = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if not event_mask[i]:
            continue

        dir_i = int(direction[i])
        trend_from = int(trend_state[i]) if np.isfinite(trend_state[i]) else 0
        bias_from = int(trend_bias_state[i]) if np.isfinite(trend_bias_state[i]) else 0

        choch_trend_state_from[i] = trend_from
        choch_trend_state_to[i] = dir_i
        choch_bias_state_from[i] = bias_from
        choch_bias_state_to[i] = dir_i
        choch_against_prev_trend[i] = int(trend_from == -dir_i and trend_from != 0)

        lo = max(0, i - structure_loss_lookback_bars)
        if dir_i == 1:
            choch_after_structure_loss[i] = int(
                structure_loss_bear[lo : i + 1].max(initial=0)
            )
        else:
            choch_after_structure_loss[i] = int(
                structure_loss_bull[lo : i + 1].max(initial=0)
            )

    out["choch_trend_state_from"] = choch_trend_state_from
    out["choch_trend_state_to"] = choch_trend_state_to
    out["choch_bias_state_from"] = choch_bias_state_from
    out["choch_bias_state_to"] = choch_bias_state_to
    out["choch_against_prev_trend"] = choch_against_prev_trend
    out["choch_after_structure_loss"] = choch_after_structure_loss

    return out
