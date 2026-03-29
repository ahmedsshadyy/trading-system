from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.sweeps import add_liquidity_sweep
from src.validation.indicators.sweeps import summarize_sweeps


def _make_sweep_df(
    rows: list[tuple[float, float, float, float]],
    *,
    high_levels: list[float | None] | None = None,
    low_levels: list[float | None] | None = None,
    high_ages: list[float | None] | None = None,
    low_ages: list[float | None] | None = None,
    atr_values: list[float] | None = None,
) -> pd.DataFrame:
    n = len(rows)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["atr_14"] = np.array(
        atr_values if atr_values is not None else [1.0] * n,
        dtype=float,
    )
    df["last_swing_high"] = np.array(
        [np.nan if v is None else float(v) for v in (high_levels or [None] * n)],
        dtype=float,
    )
    df["last_swing_low"] = np.array(
        [np.nan if v is None else float(v) for v in (low_levels or [None] * n)],
        dtype=float,
    )
    df["swing_high_age"] = np.array(
        [np.nan if v is None else float(v) for v in (high_ages or [None] * n)],
        dtype=float,
    )
    df["swing_low_age"] = np.array(
        [np.nan if v is None else float(v) for v in (low_ages or [None] * n)],
        dtype=float,
    )
    return df


def test_high_sweep_raw_detect_and_confirm_are_separate() -> None:
    df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.0, 102.0, 98.0, 99.5),
            (99.4, 99.6, 98.6, 99.0),
        ],
        high_levels=[None, 100.0, 100.0],
        high_ages=[None, 1.0, 2.0],
    )

    result = add_liquidity_sweep(df, require_next_bar_confirmation=True)

    detect = result.iloc[1]
    confirm = result.iloc[2]

    assert detect["sweep_high_detect_flag"] == 1
    assert detect["sweep_detect_flag"] == 1
    assert detect["sweep_confirm_flag"] == 0
    assert detect["sweep_event_id"] == 1.0
    assert detect["sweep_direction"] == 1
    assert detect["sweep_detect_idx"] == 1.0
    assert detect["sweep_high_source_idx"] == 0.0
    assert detect["sweep_source_idx"] == 0.0
    assert detect["sweep_high_level_type"] == 1
    assert detect["sweep_score"] >= 0.0
    assert detect["sweep_score"] <= 1.0

    assert confirm["sweep_high_confirm_flag"] == 1
    assert confirm["sweep_confirm_flag"] == 1
    assert confirm["sweep_high_confirm_idx"] == 2.0
    assert confirm["sweep_confirm_idx"] == 2.0
    assert confirm["sweep_event_id"] == 1.0
    assert confirm["sweep_detect_idx"] == 1.0


def test_raw_detect_survives_failed_confirmation() -> None:
    df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.0, 102.0, 98.0, 99.5),
            (99.5, 100.5, 99.0, 100.0),
        ],
        high_levels=[None, 100.0, 100.0],
        high_ages=[None, 1.0, 2.0],
    )

    result = add_liquidity_sweep(df, require_next_bar_confirmation=True)

    assert result.loc[1, "sweep_high_detect_flag"] == 1
    assert result.loc[1, "sweep_event_id"] == 1.0
    assert result["sweep_high_confirm_flag"].sum() == 0
    assert result["sweep_confirm_flag"].sum() == 0


def test_low_sweep_passes_and_legacy_aliases_match_detect_family() -> None:
    df = _make_sweep_df(
        [
            (101.0, 101.5, 100.5, 101.0),
            (101.0, 102.0, 98.0, 100.5),
        ],
        low_levels=[None, 99.0],
        low_ages=[None, 1.0],
    )

    result = add_liquidity_sweep(df)
    row = result.iloc[1]

    assert row["sweep_low_detect_flag"] == 1
    assert row["sweep_direction"] == -1
    assert row["sweep_low"] == row["sweep_low_detect_flag"]
    assert np.isclose(row["sweep_low_magnitude"], row["sweep_low_extension_atr"])
    assert np.isclose(row["sweep_magnitude"], row["sweep_low_extension_atr"])
    assert row["sweep_side"] == -1


def test_close_back_inside_is_strict_for_high_sweep() -> None:
    df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.0, 102.0, 98.0, 100.0),
        ],
        high_levels=[None, 100.0],
        high_ages=[None, 1.0],
    )

    result = add_liquidity_sweep(df)
    row = result.iloc[1]

    assert row["sweep_high_pass_wick_through"] == 1
    assert row["sweep_high_pass_close_back_inside"] == 0
    assert row["sweep_high_detect_flag"] == 0
    assert row["sweep_detect_flag"] == 0


def test_source_age_is_measured_from_live_safe_source_idx() -> None:
    df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.2, 100.2, 98.7, 99.4),
            (99.0, 102.0, 98.0, 99.5),
        ],
        high_levels=[None, None, 100.0],
        high_ages=[None, None, 2.0],
    )

    result = add_liquidity_sweep(df)
    row = result.iloc[2]

    assert row["sweep_high_source_idx"] == 0.0
    assert row["sweep_high_level_age"] == 2.0
    assert row["sweep_high_detect_idx"] == 2.0
    assert row["sweep_high_source_idx"] <= row["sweep_high_detect_idx"]


def test_stale_source_invalid_atr_and_zero_range_suppress_events() -> None:
    stale_df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.0, 102.0, 98.0, 99.5),
        ],
        high_levels=[None, 100.0],
        high_ages=[None, 10.0],
    )
    stale = add_liquidity_sweep(stale_df, max_level_age=5)
    assert stale.loc[1, "sweep_high_pass_source_age"] == 0
    assert stale.loc[1, "sweep_high_detect_flag"] == 0

    invalid_atr_df = _make_sweep_df(
        [(99.0, 102.0, 98.0, 99.5)],
        high_levels=[100.0],
        high_ages=[0.0],
        atr_values=[np.nan],
    )
    invalid_atr = add_liquidity_sweep(invalid_atr_df)
    assert invalid_atr.loc[0, "sweep_high_pass_valid_inputs"] == 0
    assert invalid_atr.loc[0, "sweep_detect_flag"] == 0

    zero_range_df = _make_sweep_df(
        [(100.0, 100.0, 100.0, 99.5)],
        high_levels=[99.0],
        high_ages=[0.0],
    )
    zero_range = add_liquidity_sweep(zero_range_df)
    assert zero_range.loc[0, "sweep_high_pass_valid_inputs"] == 0
    assert zero_range.loc[0, "sweep_detect_flag"] == 0


def test_validation_timing_invariants_use_confirm_lineage_on_collision_rows() -> None:
    df = _make_sweep_df(
        [
            (99.0, 100.5, 98.5, 99.5),
            (99.0, 102.0, 98.0, 99.5),
            (99.5, 100.0, 97.0, 99.0),
        ],
        high_levels=[None, 100.0, 100.0],
        low_levels=[None, None, 98.0],
        high_ages=[None, 1.0, 2.0],
        low_ages=[None, None, 1.0],
    )

    result = add_liquidity_sweep(df, require_next_bar_confirmation=True)
    summary = summarize_sweeps(result)
    timing = summary["timing_invariants"]

    assert result.loc[2, "sweep_high_confirm_flag"] == 1
    assert result.loc[2, "sweep_low_detect_flag"] == 1
    assert result.loc[2, "sweep_detect_flag"] == 1
    assert result.loc[2, "sweep_confirm_flag"] == 1
    assert timing["detect_idx_lt_confirm_idx"] is True
    assert timing["confirm_rows_have_detect_ancestor"] is True
    assert timing["high_confirm_rows_have_prev_detect"] is True
    assert timing["low_confirm_rows_have_prev_detect"] is True
    assert timing["shared_detect_confirm_collision_count"] == 1
