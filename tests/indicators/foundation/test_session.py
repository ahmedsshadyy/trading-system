from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.indicators.foundation.session import (
    _compute_session_codes,
    add_session_features,
    add_time_features,
)
from src.validation.indicators.session import summarize_session_features


def _make_regular_intraday_df(
    *,
    start: str = "2026-03-26 00:00:00+00:00",
    periods: int = 48,
    freq: str = "1h",
    flat: bool = False,
) -> pd.DataFrame:
    timestamp = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    base = np.arange(periods, dtype=float)

    if flat:
        open_ = np.full(periods, 100.0, dtype=float)
        high = np.full(periods, 100.0, dtype=float)
        low = np.full(periods, 100.0, dtype=float)
        close = np.full(periods, 100.0, dtype=float)
    else:
        open_ = 100.0 + base
        close = open_ + np.where((base.astype(int) % 2) == 0, 0.25, -0.25)
        high = np.maximum(open_, close) + 0.75
        low = np.minimum(open_, close) - 0.50

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr_14": np.full(periods, 10.0, dtype=float),
        }
    )
    return df


def _make_h4_df() -> pd.DataFrame:
    return _make_regular_intraday_df(periods=18, freq="4h")


def _make_h4_overlap_df() -> pd.DataFrame:
    return _make_regular_intraday_df(
        start="2026-03-26 06:00:00+00:00",
        periods=5,
        freq="4h",
    )


def test_add_time_features_is_pure() -> None:
    df = _make_regular_intraday_df(periods=8)
    original = df.copy(deep=True)

    result = add_time_features(df)

    pdt.assert_frame_equal(df, original)
    assert {"hour_utc", "minute_utc", "day_of_week", "session_day_id"} <= set(
        result.columns
    )


def test_time_features_classify_monday_and_friday() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-23 00:00:00+00:00",
                "2026-03-27 00:00:00+00:00",
            ]
        }
    )

    result = add_time_features(df)

    assert result["day_of_week"].tolist() == [0, 4]
    assert result["is_week_open_day"].tolist() == [1, 0]
    assert result["is_friday"].tolist() == [0, 1]
    assert result["is_week_close_day"].tolist() == [0, 1]
    assert result["session_day_id"].tolist() == ["2026-03-23", "2026-03-27"]


def test_session_classifier_exact_endpoint_boundaries() -> None:
    ts = pd.Series(
        pd.to_datetime(
            [
                "2026-03-27 00:00:00+00:00",
                "2026-03-27 07:59:59+00:00",
                "2026-03-27 08:00:00+00:00",
                "2026-03-27 12:59:59+00:00",
                "2026-03-27 13:00:00+00:00",
                "2026-03-27 16:59:59+00:00",
                "2026-03-27 17:00:00+00:00",
                "2026-03-27 21:59:59+00:00",
                "2026-03-27 22:00:00+00:00",
                "2026-03-27 23:59:59+00:00",
            ],
            utc=True,
        )
    )

    assert _compute_session_codes(ts).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_session_exclusivity_and_window_flags_on_h1_grid() -> None:
    result = add_session_features(_make_regular_intraday_df(periods=24))

    flags = result[
        [
            "session_asia_flag",
            "session_london_flag",
            "session_overlap_flag",
            "session_ny_flag",
            "session_dead_flag",
        ]
    ].sum(axis=1)
    assert (flags == 1).all()
    assert (result["is_dead_zone"] == result["session_dead_flag"]).all()

    london_open_rows = result[result["hour_utc"].isin([8, 9])]
    assert (london_open_rows["is_london_open_window"] == 1).all()
    assert result.loc[result["hour_utc"] == 10, "is_london_open_window"].eq(0).all()

    ny_open_rows = result[result["hour_utc"].isin([13, 14])]
    assert (ny_open_rows["is_ny_open_window"] == 1).all()
    assert result.loc[result["hour_utc"] == 15, "is_ny_open_window"].eq(0).all()

    assert result.loc[result["hour_utc"] == 8, "is_london_active_window"].item() == 1
    assert result.loc[result["hour_utc"] == 16, "is_london_active_window"].item() == 1
    assert result.loc[result["hour_utc"] == 17, "is_london_active_window"].item() == 0
    assert result.loc[result["hour_utc"] == 13, "is_ny_active_window"].item() == 1
    assert result.loc[result["hour_utc"] == 21, "is_ny_active_window"].item() == 1
    assert result.loc[result["hour_utc"] == 22, "is_ny_active_window"].item() == 0


def test_session_progress_resets_and_last_bar_logic_on_h1() -> None:
    result = add_session_features(_make_regular_intraday_df(periods=24))

    row_00 = result.loc[result["hour_utc"] == 0].iloc[0]
    row_07 = result.loc[result["hour_utc"] == 7].iloc[0]
    row_08 = result.loc[result["hour_utc"] == 8].iloc[0]
    row_12 = result.loc[result["hour_utc"] == 12].iloc[0]
    row_13 = result.loc[result["hour_utc"] == 13].iloc[0]
    row_16 = result.loc[result["hour_utc"] == 16].iloc[0]
    row_22 = result.loc[result["hour_utc"] == 22].iloc[0]
    row_23 = result.loc[result["hour_utc"] == 23].iloc[0]

    assert row_00["bars_since_session_open"] == 0
    assert row_07["bars_since_session_open"] == 7
    assert row_07["bars_remaining_in_session"] == 0
    assert row_07["is_last_bar_of_session"] == 1

    assert row_08["bars_since_session_open"] == 0
    assert row_12["bars_since_session_open"] == 4
    assert row_12["bars_remaining_in_session"] == 0
    assert row_13["bars_since_session_open"] == 0
    assert row_16["bars_remaining_in_session"] == 0
    assert row_22["bars_since_session_open"] == 0
    assert row_23["bars_remaining_in_session"] == 0

    assert result["session_progress_frac"].between(0.0, 1.0, inclusive="left").all()


def test_running_state_monotonicity_and_direction_deadband() -> None:
    df = _make_regular_intraday_df(periods=24)
    result = add_session_features(df)

    asia = result[result["session_code"] == 0]
    assert asia["session_high_so_far"].is_monotonic_increasing
    assert asia["session_low_so_far"].is_monotonic_decreasing

    first = asia.iloc[0]
    assert first["session_open_price"] == df.iloc[0]["open"]
    assert (
        first["session_range_so_far"]
        == first["session_high_so_far"] - first["session_low_so_far"]
    )

    deadband_df = _make_regular_intraday_df(periods=8, flat=True)
    deadband_df["close"] = 100.04
    deadband_df["high"] = 100.54
    deadband_df["low"] = 99.50
    deadband_result = add_session_features(deadband_df)
    assert deadband_result["session_direction_so_far"].eq(0).all()


def test_previous_session_summaries_become_available_only_after_close() -> None:
    result = add_session_features(_make_regular_intraday_df(periods=24))

    row_07 = result.loc[result["hour_utc"] == 7].iloc[0]
    row_08 = result.loc[result["hour_utc"] == 8].iloc[0]
    row_13 = result.loc[result["hour_utc"] == 13].iloc[0]
    row_22 = result.loc[result["hour_utc"] == 22].iloc[0]

    assert np.isnan(row_07["prev_asia_high"])
    assert row_08["prev_asia_high"] == pytest.approx(result.loc[:7, "high"].max())
    assert row_13["prev_london_high"] == pytest.approx(result.loc[8:12, "high"].max())
    assert row_22["prev_ny_high"] == pytest.approx(result.loc[17:21, "high"].max())

    expected_dist = (row_08["close"] - row_08["prev_asia_high"]) / 10.0
    assert row_08["dist_to_prev_asia_high_atr"] == pytest.approx(expected_dist)


def test_asia_package_non_leakage_and_post_close_carry_forward() -> None:
    result = add_session_features(_make_regular_intraday_df(periods=32))

    asia_active = result[result["hour_utc"].isin(range(0, 8))]
    assert asia_active["asia_range_active_flag"].eq(1).all()
    assert asia_active["asia_range_complete_flag"].eq(0).all()
    assert asia_active["asia_range_high_final"].isna().all()

    row_08 = result.loc[result["hour_utc"] == 8].iloc[0]
    row_12 = result.loc[result["hour_utc"] == 12].iloc[0]
    assert row_08["asia_range_complete_flag"] == 1
    assert row_08["asia_range_high_final"] == pytest.approx(
        result.loc[:7, "high"].max()
    )
    assert row_12["asia_range_width_final_atr"] == pytest.approx(
        (result.loc[:7, "high"].max() - result.loc[:7, "low"].min()) / 10.0
    )

    next_day_00 = result.loc[
        result["timestamp"] == pd.Timestamp("2026-03-27 00:00:00+00:00")
    ].iloc[0]
    assert next_day_00["asia_range_active_flag"] == 1
    assert np.isnan(next_day_00["asia_range_high_final"])


def test_research_live_parity_for_canonical_columns() -> None:
    df = _make_regular_intraday_df(periods=32)

    live = add_session_features(df, include_research_only=False)
    research = add_session_features(df, include_research_only=True)

    assert not any(col.startswith("r_") for col in live.columns)
    assert {"r_session_final_high", "r_session_direction_final"} <= set(
        research.columns
    )

    live_cols = [col for col in research.columns if not col.startswith("r_")]
    pdt.assert_frame_equal(live[live_cols], research[live_cols], check_dtype=False)


def test_h4_snapshot_values() -> None:
    result = add_session_features(_make_h4_df())

    row_00 = result.iloc[0]
    row_04 = result.iloc[1]
    row_08 = result.iloc[2]
    row_12 = result.iloc[3]
    row_16 = result.iloc[4]
    row_20 = result.iloc[5]

    assert row_00["session_code"] == 0
    assert row_04["session_code"] == 0
    assert row_08["session_code"] == 1
    assert row_12["is_last_bar_of_session"] == 1
    assert row_16["session_code"] == 2
    assert row_20["session_code"] == 3
    assert row_04["bars_since_session_open"] == 1
    assert row_08["prev_asia_high"] == pytest.approx(result.iloc[:2]["high"].max())


def test_h4_open_and_active_windows_use_bar_overlap_semantics() -> None:
    result = add_session_features(_make_h4_overlap_df())

    row_06 = result.loc[result["hour_utc"] == 6].iloc[0]
    row_10 = result.loc[result["hour_utc"] == 10].iloc[0]
    row_14 = result.loc[result["hour_utc"] == 14].iloc[0]

    assert row_06["session_code"] == 0
    assert row_06["is_london_open_window"] == 1
    assert row_06["is_london_active_window"] == 1

    assert row_10["is_london_open_window"] == 0
    assert row_10["is_ny_open_window"] == 1
    assert row_10["is_ny_active_window"] == 1

    assert row_14["session_code"] == 2
    assert row_14["is_ny_open_window"] == 1


def test_validation_summary_surfaces_hard_session_checks() -> None:
    live = add_session_features(
        _make_regular_intraday_df(periods=48), include_research_only=False
    )
    research = add_session_features(
        _make_regular_intraday_df(periods=48), include_research_only=True
    )
    summary = summarize_session_features(research, live_df=live, parity_ok=True)

    assert summary["checks"]["bars_since_session_open_reset_ok"] is True
    assert summary["checks"]["session_high_so_far_monotonic"] is True
    assert summary["checks"]["session_low_so_far_monotonic"] is True
    assert summary["checks"]["no_asia_final_leakage"] is True
    assert summary["checks"]["no_research_cols_in_live"] is True
    assert summary["previous_session_boundary_checks"]["asia"]["all_passed"] is True
    assert summary["window_semantics_audit"]["open_window_semantics"] == "bar_overlap"
    assert summary["audit_classification"]["annotation_safe"] is True
    assert summary["audit_classification"]["detect_safe"] is True
    assert summary["audit_classification"]["confirm_safe"] is True
    assert summary["audit_classification"]["active_safe"] is True
    assert summary["audit_classification"]["model_safe"] is True
    assert summary["audit_classification"]["research_only_model_safe"] is False


def test_validation_summary_research_combo_timing_and_codebook_surface() -> None:
    live = add_session_features(
        _make_regular_intraday_df(periods=72), include_research_only=False
    )
    research = add_session_features(
        _make_regular_intraday_df(periods=72), include_research_only=True
    )
    summary = summarize_session_features(research, live_df=live, parity_ok=True)

    research_summary = summary["research_direction_summary"]
    assert research_summary is not None
    assert (
        research_summary["r_asia_london_direction_combo_final"]["no_premature_values"]
        is True
    )
    assert (
        research_summary["r_london_ny_direction_combo_final"]["no_premature_values"]
        is True
    )
    assert (
        research_summary["r_asia_ny_direction_combo_final"]["no_premature_values"]
        is True
    )
    assert (
        research_summary["direction_triple_distribution"]["no_premature_values"] is True
    )
    assert (
        research_summary["direction_triple_distribution"][
            "all_values_in_valid_27_state_space"
        ]
        is True
    )
    assert (
        research_summary["direction_triple_active_london_distribution"][
            "no_premature_values"
        ]
        is True
    )
    assert (
        research_summary["direction_triple_active_london_distribution"][
            "all_values_in_valid_27_state_space"
        ]
        is True
    )

    direction_keys = set(research_summary["session_direction_final_counts"].keys())
    assert direction_keys <= {"-1", "0", "1", "NaN"}


def test_research_triple_first_appears_only_after_ny_completion() -> None:
    research = add_session_features(
        _make_regular_intraday_df(periods=24), include_research_only=True
    )

    assert (
        research.loc[
            research["hour_utc"] <= 21, "r_asia_london_ny_direction_triple_final"
        ]
        .isna()
        .all()
    )
    dead_zone_rows = research.loc[research["hour_utc"].isin([22, 23])]
    assert dead_zone_rows["r_asia_london_ny_direction_triple_final"].notna().sum() == 1
    assert dead_zone_rows["r_asia_london_ny_direction_triple_label"].notna().sum() == 1
    assert (
        dead_zone_rows["r_asia_london_active_ny_direction_triple_final"].notna().sum()
        == 1
    )
    assert (
        dead_zone_rows["r_asia_london_active_ny_direction_triple_label"].notna().sum()
        == 1
    )


def test_active_london_triple_uses_08_to_17_block_as_separate_family() -> None:
    df = _make_regular_intraday_df(periods=24, flat=True)
    df["atr_14"] = 10.0

    df.loc[0:7, ["open", "close", "high", "low"]] = [100.0, 101.0, 101.5, 99.5]
    df.loc[8:12, ["open", "close", "high", "low"]] = [102.0, 100.0, 102.5, 99.5]
    df.loc[13:16, ["open", "close", "high", "low"]] = [100.0, 104.0, 104.5, 99.5]
    df.loc[17:21, ["open", "close", "high", "low"]] = [103.0, 101.0, 103.5, 100.5]

    result = add_session_features(df, include_research_only=True)
    stamped = result.loc[result["hour_utc"] == 22].iloc[0]

    assert stamped["r_asia_london_ny_direction_triple_final"] == "1_-1_-1"
    assert stamped["r_asia_london_active_ny_direction_triple_final"] == "1_1_-1"


def test_tiny_duplicate_non_monotonic_and_flat_frames_raise_or_stay_safe() -> None:
    tiny = _make_regular_intraday_df(periods=3)
    with pytest.raises(ValueError, match="infer bar interval"):
        add_session_features(tiny)

    duplicate = _make_regular_intraday_df(periods=6)
    duplicate.loc[2, "timestamp"] = duplicate.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        add_session_features(duplicate)

    non_monotonic = _make_regular_intraday_df(periods=6)
    non_monotonic.loc[3, "timestamp"] = non_monotonic.loc[
        2, "timestamp"
    ] - pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="strictly increasing"):
        add_session_features(non_monotonic)

    flat = _make_regular_intraday_df(periods=8, flat=True)
    flat_result = add_session_features(flat)
    assert flat_result["session_position_in_range_so_far"].isna().all()


def test_nan_rows_and_repeated_equal_highs_lows_do_not_break_causal_outputs() -> None:
    df = _make_regular_intraday_df(periods=12)
    df.loc[5, "atr_14"] = np.nan
    df.loc[3:6, "high"] = 111.0
    df.loc[3:6, "low"] = 109.0

    result = add_session_features(df)

    assert np.isnan(result.loc[5, "session_close_change_from_open_atr"])
    assert result["session_asia_flag"].sum() == 8
    assert result["session_london_flag"].sum() == 4


def test_weekend_gap_starts_new_session_group_for_running_and_prev_ny_state() -> None:
    timestamp = pd.to_datetime(
        [
            "2026-03-27 17:00:00+00:00",
            "2026-03-27 18:00:00+00:00",
            "2026-03-27 19:00:00+00:00",
            "2026-03-27 20:00:00+00:00",
            "2026-03-27 21:00:00+00:00",
            "2026-03-29 21:00:00+00:00",
            "2026-03-29 22:00:00+00:00",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": [100, 101, 102, 103, 104, 90, 91],
            "high": [101, 102, 103, 104, 110, 95, 92],
            "low": [99, 100, 101, 102, 98, 89, 90],
            "close": [100.5, 101.5, 102.5, 103.5, 99.0, 94.0, 91.5],
            "atr_14": [10.0] * 7,
        }
    )

    result = add_session_features(df)

    sunday_reopen = result.iloc[5]
    assert sunday_reopen["session_code"] == 3
    assert sunday_reopen["bars_since_session_open"] == 0
    assert sunday_reopen["prev_ny_high"] == pytest.approx(110.0)
    assert sunday_reopen["prev_ny_low"] == pytest.approx(98.0)
