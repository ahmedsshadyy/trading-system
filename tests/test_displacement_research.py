from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.research.displacement_research import (
    build_displacement_research_table,
    summarize_displacement_research,
)
from src.indicators.smc.displacement import add_displacement_candle


def _make_research_df() -> pd.DataFrame:
    rows = [
        (100.0, 112.0, 99.0, 111.0),
        (111.0, 112.0, 105.0, 108.0),
        (108.0, 120.0, 107.0, 110.0),
        (110.0, 113.0, 109.0, 111.0),
        (120.0, 121.0, 108.0, 109.0),
        (109.0, 116.0, 108.0, 110.0),
        (120.0, 122.0, 119.0, 121.5),
        (121.5, 123.0, 120.5, 121.0),
        (130.0, 142.0, 129.0, 141.0),
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h", tz="UTC")
    df["volume"] = np.arange(1000, 1000 + len(df), dtype=np.int64)
    df["atr_14"] = np.full(len(df), 5.0, dtype=float)
    df["trend_state"] = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1], dtype=np.int8)
    df["trend_bias_state"] = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1], dtype=np.int8)
    df["bos_direction"] = np.array([0, 0, 1, 0, -1, 0, 0, 0, 1], dtype=np.int8)
    df["choch_direction"] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    df["session"] = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int8)
    df["regime"] = np.array([2, 2, 2, 1, 2, 2, 1, 1, 2], dtype=np.int8)
    df["vol_ratio"] = np.array([1.6, 0.9, 1.1, 0.8, 1.5, 1.2, 0.7, 0.6, 1.8])
    df["adx_14"] = np.array([28.0, 27.0, 29.0, 24.0, 31.0, 30.0, 26.0, 25.0, 33.0])
    df["rsi_14"] = np.array([62.0, 54.0, 57.0, 56.0, 39.0, 44.0, 58.0, 52.0, 66.0])
    return add_displacement_candle(df)


def test_displacement_research_table_maps_one_row_per_detector_event() -> None:
    df = _make_research_df()
    original = df.copy(deep=True)

    research = build_displacement_research_table(df)

    pd.testing.assert_frame_equal(df, original)
    assert int(df["displacement_flag"].sum()) == 3
    assert len(research) == 3
    assert research["displacement_event_id"].tolist() == [1, 2, 3]
    assert research["displacement_detect_idx"].tolist() == [0, 4, 8]
    assert research["displacement_side"].tolist() == ["bull", "bear", "bull"]
    assert research["displacement_direction"].tolist() == [1, -1, 1]
    assert research["displacement_trend_state_on_event"].tolist() == [1.0, -1.0, 1.0]
    assert research["displacement_trend_bias_state_on_event"].tolist() == [
        1.0,
        -1.0,
        1.0,
    ]
    assert research["displacement_session_on_event"].tolist() == ["1", "2", "3"]
    assert research["displacement_volume_ratio_on_event"].tolist() == [1.6, 1.5, 1.8]
    assert np.isclose(
        research.loc[0, "displacement_body_atr"], df.loc[0, "displacement_body_atr"]
    )
    assert np.isclose(
        research.loc[1, "displacement_body_frac"], df.loc[4, "displacement_body_frac"]
    )


def test_displacement_research_horizon_logic_retest_and_outcomes_are_frozen() -> None:
    research = build_displacement_research_table(_make_research_df())

    bull = research.iloc[0]
    assert np.isclose(bull["displacement_mfe_3_atr"], 1.8)
    assert np.isclose(bull["displacement_mae_3_atr"], 1.2)
    assert np.isclose(bull["displacement_excursion_ratio_3"], 1.5)
    assert bull["displacement_hold_3"] is True
    assert bull["displacement_failed_3"] is False
    assert bull["displacement_reversal_3"] is False
    assert bull["displacement_retest_ever_3"] is True
    assert bull["displacement_first_retest_idx"] == 1.0
    assert bull["displacement_first_retest_delay"] == 1.0
    assert np.isclose(bull["displacement_first_retest_depth_frac"], 7.0 / 13.0)
    assert bull["displacement_retest_count_3"] == 1
    assert bull["displacement_hold_after_retest_3"] is True
    assert bull["displacement_continuation_3_1.5atr"] is True
    assert bull["displacement_final_outcome_3"] == "weak_continuation"

    bear = research.iloc[1]
    assert np.isclose(bear["displacement_mfe_3_atr"], 0.2)
    assert np.isclose(bear["displacement_mae_3_atr"], 2.8)
    assert bear["displacement_hold_3"] is False
    assert bear["displacement_failed_3"] is True
    assert bear["displacement_reversal_3"] is False
    assert bear["displacement_retest_ever_3"] is True
    assert bear["displacement_first_retest_idx"] == 5.0
    assert bear["displacement_retest_count_3"] == 1
    assert bear["displacement_hold_after_retest_3"] is False
    assert bear["displacement_final_outcome_3"] == "failed"

    late = research.iloc[2]
    assert np.isnan(late["displacement_mfe_3_atr"])
    assert pd.isna(late["displacement_hold_1"])
    assert late["displacement_final_outcome_3"] == "insufficient_horizon"
    assert late["displacement_final_outcome_20"] == "insufficient_horizon"


def test_displacement_research_scores_and_summary_are_consistent() -> None:
    research = build_displacement_research_table(_make_research_df())
    summary = summarize_displacement_research(research)

    for col in [
        "displacement_quality_score",
        "displacement_follow_through_score",
        "displacement_tradeable_score",
        "displacement_failure_severity_score",
    ]:
        clean = research[col].dropna()
        assert ((clean >= 0.0) & (clean <= 1.0)).all(), col

    assert np.isfinite(research.loc[0, "displacement_follow_through_score"])
    assert np.isnan(research.loc[1, "displacement_follow_through_score"])
    assert np.isnan(research.loc[2, "displacement_follow_through_score"])
    assert (
        research.loc[1, "displacement_tradeable_score"]
        == research.loc[1, "displacement_quality_score"]
    )
    assert summary["event_count"] == 3
    assert summary["bull_count"] == 2
    assert summary["bear_count"] == 1
    assert summary["recommended_excursion_ratio_variant"] == "capped"
    assert np.isclose(summary["hold_rates"]["3"], 0.5)
    assert np.isclose(summary["failure_rates"]["3"], 0.5)
    assert np.isclose(summary["reversal_rates"]["3"], 0.0)
    assert summary["outcome_distributions"]["3"]["weak_continuation"] == 1
    assert summary["outcome_distributions"]["3"]["failed"] == 1
    assert summary["outcome_distributions"]["3"]["insufficient_horizon"] == 1
    assert summary["retest_reaction_quality_count"] == 1
    assert summary["no_retest_count"] == 1
    assert summary["invalid_retest_reaction_count"] == 1
    assert summary["consistency_checks"]["non_negative_mfe"] is True
    assert summary["consistency_checks"]["non_negative_mae"] is True
    assert summary["consistency_checks"]["non_negative_excursion_ratio"] is True
    assert (
        summary["outcome_reconciliation"]["3"]["final_outcomes_sum_to_event_count"]
        is True
    )
    assert (
        summary["outcome_reconciliation"]["3"][
            "failed_events_reconcile_with_failed_flag"
        ]
        is True
    )
    assert summary["consistency_checks"]["unique_event_ids"] is True
    assert summary["consistency_checks"]["one_row_per_detect_idx"] is True


def test_displacement_research_excursions_are_non_negative_and_ratios_are_stable() -> (
    None
):
    negative_mfe_rows = [
        (100.0, 112.0, 99.0, 111.0),
        (109.0, 110.0, 108.0, 109.0),
        (109.0, 109.5, 107.5, 108.0),
    ]
    negative_mfe_df = pd.DataFrame(
        negative_mfe_rows, columns=["open", "high", "low", "close"]
    )
    negative_mfe_df["timestamp"] = pd.date_range(
        "2024-01-01", periods=len(negative_mfe_df), freq="4h", tz="UTC"
    )
    negative_mfe_df["volume"] = np.arange(100, 100 + len(negative_mfe_df))
    negative_mfe_df["atr_14"] = np.full(len(negative_mfe_df), 5.0, dtype=float)
    negative_mfe_df = add_displacement_candle(negative_mfe_df)
    negative_mfe_research = build_displacement_research_table(negative_mfe_df)
    assert negative_mfe_research.loc[0, "displacement_mfe_1_atr"] == 0.0
    assert negative_mfe_research.loc[0, "displacement_mae_1_atr"] >= 0.0

    zero_mae_rows = [
        (100.0, 112.0, 99.0, 111.0),
        (112.0, 115.0, 112.0, 114.0),
        (114.0, 116.0, 113.0, 115.0),
    ]
    zero_mae_df = pd.DataFrame(zero_mae_rows, columns=["open", "high", "low", "close"])
    zero_mae_df["timestamp"] = pd.date_range(
        "2024-02-01", periods=len(zero_mae_df), freq="4h", tz="UTC"
    )
    zero_mae_df["volume"] = np.arange(200, 200 + len(zero_mae_df))
    zero_mae_df["atr_14"] = np.full(len(zero_mae_df), 5.0, dtype=float)
    zero_mae_df = add_displacement_candle(zero_mae_df)
    zero_mae_research = build_displacement_research_table(zero_mae_df)
    assert zero_mae_research.loc[0, "displacement_mae_1_atr"] == 0.0
    assert np.isclose(zero_mae_research.loc[0, "displacement_excursion_ratio_1"], 8.0)
    assert np.isclose(
        zero_mae_research.loc[0, "displacement_excursion_ratio_1_capped"], 8.0
    )
    assert np.isclose(
        zero_mae_research.loc[0, "displacement_excursion_ratio_1_capped"],
        min(zero_mae_research.loc[0, "displacement_excursion_ratio_1"], 10.0),
    )
    assert zero_mae_research.loc[0, "displacement_excursion_ratio_1_capped"] <= 10.0


def test_displacement_research_horizon_20_failed_and_reversed_outcomes_reconcile() -> (
    None
):
    base_rows = [(101.0, 102.0, 100.0, 101.0)] * 22
    base_rows[0] = (100.0, 112.0, 99.0, 111.0)

    failed_rows = list(base_rows)
    failed_rows[20] = (100.5, 101.0, 98.4, 98.8)
    failed_df = pd.DataFrame(failed_rows, columns=["open", "high", "low", "close"])
    failed_df["timestamp"] = pd.date_range(
        "2024-03-01", periods=len(failed_df), freq="4h", tz="UTC"
    )
    failed_df["volume"] = np.arange(300, 300 + len(failed_df))
    failed_df["atr_14"] = np.full(len(failed_df), 5.0, dtype=float)
    failed_df = add_displacement_candle(failed_df)
    failed_df.loc[:, "displacement_flag"] = 0
    failed_df.loc[:, "displacement_bull"] = 0
    failed_df.loc[:, "displacement_bear"] = 0
    failed_df.loc[0, "displacement_flag"] = 1
    failed_df.loc[0, "displacement_bull"] = 1
    failed_df.loc[0, "displacement_direction"] = 1
    failed_research = build_displacement_research_table(failed_df)
    assert bool(failed_research.loc[0, "displacement_failed_20"]) is True
    assert bool(failed_research.loc[0, "displacement_reversal_20"]) is False
    assert failed_research.loc[0, "displacement_final_outcome_20"] == "failed"

    reversed_rows = list(base_rows)
    reversed_rows[20] = (99.5, 100.0, 95.5, 96.0)
    reversed_df = pd.DataFrame(reversed_rows, columns=["open", "high", "low", "close"])
    reversed_df["timestamp"] = pd.date_range(
        "2024-04-01", periods=len(reversed_df), freq="4h", tz="UTC"
    )
    reversed_df["volume"] = np.arange(400, 400 + len(reversed_df))
    reversed_df["atr_14"] = np.full(len(reversed_df), 5.0, dtype=float)
    reversed_df = add_displacement_candle(reversed_df)
    reversed_df.loc[:, "displacement_flag"] = 0
    reversed_df.loc[:, "displacement_bull"] = 0
    reversed_df.loc[:, "displacement_bear"] = 0
    reversed_df.loc[0, "displacement_flag"] = 1
    reversed_df.loc[0, "displacement_bull"] = 1
    reversed_df.loc[0, "displacement_direction"] = 1
    reversed_research = build_displacement_research_table(reversed_df)
    assert bool(reversed_research.loc[0, "displacement_failed_20"]) is True
    assert bool(reversed_research.loc[0, "displacement_reversal_20"]) is True
    assert reversed_research.loc[0, "displacement_final_outcome_20"] == "reversed"


def test_displacement_research_requires_canonical_volume_features_for_volume_ratio() -> (
    None
):
    n = 24
    rows = [(100.0, 112.0, 99.0, 111.0)] + [(111.0, 112.0, 110.0, 111.0)] * (n - 1)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["tickVolume"] = np.arange(1, n + 1, dtype=float) * 10.0
    df["atr_14"] = np.full(n, 5.0, dtype=float)
    df = add_displacement_candle(df)
    df = df.drop(columns=["vol_ratio"], errors="ignore")

    research = build_displacement_research_table(df)
    assert np.isnan(research.loc[0, "displacement_volume_ratio_on_event"])

    detect_idx_late = 20
    df.loc[:, "displacement_flag"] = 0
    df.loc[:, "displacement_bull"] = 0
    df.loc[:, "displacement_bear"] = 0
    df.loc[detect_idx_late, "displacement_flag"] = 1
    df.loc[detect_idx_late, "displacement_bull"] = 1
    df.loc[detect_idx_late, "displacement_direction"] = 1
    late_research = build_displacement_research_table(df)
    assert np.isnan(late_research.loc[0, "displacement_volume_ratio_on_event"])
