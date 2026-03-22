# src/indicators/structure/bos.py

"""
Canonical causal BOS detector.

BOS is an event layer, not a regime/state model.

Rules
-----
- Uses only already-confirmed causal swings from add_swings().
- Canonical bullish BOS = close breaks above prior confirmed swing high.
- Canonical bearish BOS = close breaks below prior confirmed swing low.
- Wick breaks are diagnostics only, never canonical BOS by themselves.
- One canonical BOS per source swing level (deduplicated by source index).
- Quality filters are applied causally on the event bar.

This detector is "as live as swings":
it only relies on levels that are already available in the current row,
which themselves only become available after swing confirmation.
"""

from __future__ import annotations

import pandas as pd

from src.indicators.structure._break_events import _compute_structural_breaks

BOS_COLUMN_SUFFIXES = [
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
    "source_age",
    "source_strength",
    "source_prominence_atr",
    "source_fresh",
    "source_stale",
    "source_rank",
    "source_in_trend_direction",
]


def add_bos(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    min_source_age_bars: int = 0,
    max_source_age_bars: int | None = None,
    min_break_distance_atr: float = 0.0,
    min_body_atr: float = 0.0,
    min_source_strength: float | None = None,
    require_trend_alignment: bool = True,
    allow_neutral_trend_breaks: bool = False,
) -> pd.DataFrame:
    """
    Detect sparse canonical BOS events from confirmed swing levels.

    Parameters
    ----------
    df:
        OHLC dataframe with causal swing outputs already present.
    atr_length:
        ATR length used for normalization.
    min_source_age_bars:
        Minimum age of the broken swing level.
    max_source_age_bars:
        Optional maximum age of the broken swing level.
    min_break_distance_atr:
        Minimum ATR-normalized close-through distance beyond source level.
    min_body_atr:
        Minimum ATR-normalized candle body on event bar.
    min_source_strength:
        Optional minimum required source swing strength if available.
    require_trend_alignment:
        Canonical BOS defaults to strict continuation-only semantics.
        If True, bullish BOS requires trend_state == 1 and bearish BOS
        requires trend_state == -1 unless neutral breaks are explicitly allowed.
    allow_neutral_trend_breaks:
        Only relevant if require_trend_alignment=True.
        If False, trend_state must already agree with BOS direction exactly.

    Returns
    -------
    DataFrame
        Original df plus BOS event and diagnostics columns.
    """
    out = df.copy()
    break_result = _compute_structural_breaks(
        out,
        atr_length=atr_length,
        min_source_age_bars=min_source_age_bars,
        max_source_age_bars=max_source_age_bars,
        min_break_distance_atr=min_break_distance_atr,
        min_body_atr=min_body_atr,
        min_source_strength=min_source_strength,
        trend_filter="same" if require_trend_alignment else "none",
        allow_neutral_trend_breaks=allow_neutral_trend_breaks,
    )

    if not break_result:
        return out

    for suffix in BOS_COLUMN_SUFFIXES:
        out[f"bos_{suffix}"] = break_result[suffix]

    return out
