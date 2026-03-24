from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.research.fvg_research import (
    build_fvg_research_table,
    summarize_fvg_research,
)
from src.indicators.smc.fvg import (
    ALL_FVG_CORE_COLUMNS,
    MAX_ACTIVE_AGE_BARS,
    STATE_ACTIVE_TOUCHED,
    STATE_EXPIRED,
    STATE_INVALIDATED,
    STATE_MERGED,
    add_fvg,
    collect_fvg_debug_tables,
)


def _make_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h")
    return df[["timestamp", "open", "high", "low", "close"]]


def _run_fvg(
    rows: list[tuple[float, float, float, float]],
    *,
    min_width_atr: float = 0.0,
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
    pattern_mode: str = "wick",
) -> pd.DataFrame:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    return add_fvg(
        df,
        atr_length=3,
        min_width_atr=min_width_atr,
        max_active_age_bars=max_active_age_bars,
        pattern_mode=pattern_mode,
    )


def test_fvg_core_schema_and_compatibility_surface_exist() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
    ]
    result = _run_fvg(rows)

    assert len(ALL_FVG_CORE_COLUMNS) == 112
    for col in ALL_FVG_CORE_COLUMNS:
        assert col in result.columns

    for col in [
        "fvg_bull",
        "fvg_bear",
        "fvg_size_atr",
        "fvg_bull_confirm_idx",
        "fvg_bear_confirm_idx",
        "fvg_confirm_delay",
        "fvg_mid_body_atr",
        "fvg_bos_bull",
        "fvg_bos_bear",
    ]:
        assert col in result.columns


def test_bull_fvg_detects_on_right_bar_only_without_same_bar_touch() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (110.0, 111.0, 100.0, 101.0),
    ]
    result = _run_fvg(rows)

    assert result.loc[1, "fvg_bull_origin_flag"] == 1
    assert result.loc[1, "fvg_bull_origin_idx"] == 1.0
    assert result.loc[2, "fvg_bull_detect_flag"] == 1
    assert result.loc[2, "fvg_bull_event_id"] == 1
    assert result.loc[2, "fvg_bull_detect_idx"] == 2.0
    assert result.loc[2, "fvg_bull_active"] == 1
    assert result.loc[2, "fvg_bull_active_age"] == 0.0
    assert result.loc[2, "fvg_bull_active_age_bars"] == 0.0
    assert result.loc[2, "fvg_bull_active_age_decay"] == 1.0
    assert 0.0 <= result.loc[2, "fvg_bull_active_base_quality"] <= 1.0
    assert 0.0 <= result.loc[2, "fvg_bull_active_effective_significance"] <= 1.0
    assert result.loc[2, "fvg_bull_active_fill_pct"] == 0.0
    assert np.isnan(result.loc[0, "fvg_bull_detect_idx"])
    assert result.loc[0, "fvg_bull_event_id"] == 0
    assert result.loc[1, "fvg_bull_event_id"] == 0
    assert result.loc[2, "fvg_bull_first_touch_idx"] == 3.0
    assert result.loc[2, "fvg_bull_first_partial_fill_idx"] == 3.0
    assert result.loc[2, "fvg_bull_full_fill_idx"] == 4.0
    assert result.loc[2, "fvg_bull_terminal_state"] != 0
    assert result.loc[3, "fvg_bull_active"] == 1
    assert result.loc[4, "fvg_bull_active"] == 0


def test_bear_fvg_detects_on_right_bar_only() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
    ]
    result = _run_fvg(rows)

    assert result.loc[1, "fvg_bear_origin_flag"] == 1
    assert result.loc[2, "fvg_bear_detect_flag"] == 1
    assert result.loc[2, "fvg_bear_event_id"] == 1
    assert result.loc[2, "fvg_bear_low"] == 89.0
    assert result.loc[2, "fvg_bear_high"] == 99.0
    assert result.loc[2, "fvg_bear_active"] == 1
    assert result.loc[2, "fvg_bear_active_age"] == 0.0
    assert result.loc[2, "fvg_bear_active_age_bars"] == 0.0
    assert result.loc[2, "fvg_bear_active_age_decay"] == 1.0
    assert 0.0 <= result.loc[2, "fvg_bear_active_base_quality"] <= 1.0
    assert 0.0 <= result.loc[2, "fvg_bear_active_effective_significance"] <= 1.0
    assert result.loc[2, "fvg_bear_active_fill_pct"] == 0.0
    assert result.loc[0, "fvg_bear_event_id"] == 0


def test_invalidation_is_one_shot_and_beats_full_fill() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (103.0, 104.0, 99.0, 100.0),
        (100.0, 101.0, 98.0, 99.0),
    ]
    result = _run_fvg(rows)

    assert result.loc[2, "fvg_bull_invalidation_idx"] == 4.0
    assert result.loc[2, "fvg_bull_terminal_state"] == STATE_INVALIDATED
    assert result.loc[4, "fvg_bull_zone_invalidated"] == 1
    assert result.loc[5, "fvg_bull_zone_invalidated"] == 0
    assert result.loc[4, "fvg_bull_active"] == 0


def test_first_retest_flag_only_fires_once_on_first_touch_transition() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (105.5, 106.0, 105.0, 105.8),
        (106.0, 108.0, 105.0, 106.0),
        (105.5, 107.0, 104.0, 105.0),
    ]
    result = _run_fvg(rows)

    assert np.isfinite(result.loc[2, "fvg_bull_first_touch_idx"])
    assert np.isfinite(result.loc[2, "fvg_bull_first_partial_fill_idx"])
    assert result.loc[3, "fvg_bull_active_state"] == STATE_ACTIVE_TOUCHED
    assert result.loc[3, "fvg_bull_first_retest_flag"] == 1
    assert result.loc[4, "fvg_bull_first_retest_flag"] == 0
    assert result.loc[4, "fvg_bull_active_touch_count"] == 2


def test_single_source_zone_can_expire_without_touch() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (104.0, 105.0, 103.0, 104.5),
        (104.2, 105.0, 104.0, 104.6),
        (104.5, 106.0, 104.2, 105.0),
        (104.8, 106.0, 104.5, 105.2),
    ]
    result = _run_fvg(rows, max_active_age_bars=1)

    assert result.loc[2, "fvg_bull_expiry_idx"] == 3.0
    assert result.loc[2, "fvg_bull_terminal_state"] == STATE_EXPIRED
    assert result.loc[4, "fvg_bull_active"] == 0


def test_gap_cleanliness_is_nan_for_degenerate_middle_candle_range() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (105.0, 105.0, 105.0, 105.0),
        (106.0, 107.0, 106.0, 106.5),
        (106.0, 107.0, 104.0, 104.5),
    ]
    result = _run_fvg(rows)

    assert result.loc[2, "fvg_bull_detect_flag"] == 1
    assert np.isnan(result.loc[2, "fvg_bull_gap_cleanliness"])


def test_same_side_merge_marks_raw_events_merged_and_keeps_one_active_rep() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (106.0, 106.0, 104.0, 105.0),
        (105.0, 107.0, 104.5, 106.5),
        (108.0, 109.0, 108.0, 108.5),
        (107.0, 108.0, 106.0, 106.5),
    ]
    result = _run_fvg(rows)

    bull_detect_rows = result.index[result["fvg_bull_detect_flag"] == 1].tolist()
    assert bull_detect_rows == [2, 5]
    assert result.loc[2, "fvg_bull_terminal_state"] == STATE_MERGED
    assert result.loc[5, "fvg_bull_terminal_state"] == STATE_MERGED
    assert result.loc[5, "fvg_bull_active_count"] == 1
    assert result.loc[5, "fvg_bull_active_low"] == 101.0
    assert result.loc[5, "fvg_bull_active_high"] == 111.0


def test_merge_does_not_reset_active_age_anchor_or_decay_clock() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (106.0, 106.0, 104.0, 105.0),
        (105.0, 107.0, 104.5, 106.5),
        (108.0, 109.0, 108.0, 108.5),
        (107.0, 108.0, 106.0, 106.5),
    ]
    result = _run_fvg(rows)

    assert result.loc[5, "fvg_bull_active_age"] == 3.0
    assert result.loc[5, "fvg_bull_active_age_bars"] == 3.0
    assert result.loc[5, "fvg_bull_active_age_decay"] < 1.0
    assert result.loc[5, "fvg_bull_active_weighted_count"] > 0.0


def test_debug_active_member_counts_match_dense_export() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (106.0, 106.0, 104.0, 105.0),
        (105.0, 107.0, 104.5, 106.5),
        (108.0, 109.0, 108.0, 108.5),
        (107.0, 108.0, 106.0, 106.5),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(df, atr_length=3, min_width_atr=0.0)
    result = debug["frame"]
    members = debug["active_member_table"]

    bull_counts = (
        members[members["side"] == "bull"].groupby("row_idx")["active_id"].nunique()
    )
    bull_counts = bull_counts.reindex(result.index, fill_value=0)
    assert (bull_counts.to_numpy() == result["fvg_bull_active_count"].to_numpy()).all()
    assert members["active_id"].notna().all()


def test_disjoint_same_side_components_only_owned_component_receives_price_deltas() -> (
    None
):
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (126.0, 128.0, 120.0, 121.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=10,
    )
    members = debug["active_member_table"]
    bull_members = members[members["side"] == "bull"].copy()

    multi_component_rows = bull_members.groupby("row_idx")["component_id"].nunique()
    candidate_rows = multi_component_rows[multi_component_rows > 1].index.tolist()
    assert candidate_rows

    delta_rows = bull_members[
        (bull_members["row_idx"].isin(candidate_rows))
        & ((bull_members["touch_delta"] > 0) | (bull_members["fill_delta"] > 0))
    ]
    assert not delta_rows.empty

    for row_idx, scoped in bull_members[
        bull_members["row_idx"].isin(candidate_rows)
    ].groupby("row_idx"):
        owned = scoped[scoped["component_owned_this_row"] == 1]
        non_owned = scoped[scoped["component_owned_this_row"] != 1]
        assert len(owned["component_id"].unique()) <= 1
        if ((owned["touch_delta"] > 0) | (owned["fill_delta"] > 0)).any():
            assert (non_owned["touch_delta"] == 0).all()
            assert (non_owned["fill_delta"] == 0).all()


def test_non_owned_disjoint_bull_component_can_still_full_fill() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (126.0, 128.0, 100.0, 110.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=10,
    )
    result = debug["frame"]

    assert result.loc[5, "fvg_bull_active_count"] == 2
    assert result.loc[2, "fvg_bull_full_fill_idx"] == 6.0
    assert result.loc[2, "fvg_bull_terminal_state"] == 4


def test_non_owned_disjoint_bull_component_can_still_invalidate() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (126.0, 128.0, 98.0, 99.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=10,
    )
    result = debug["frame"]

    assert result.loc[5, "fvg_bull_active_count"] == 2
    assert result.loc[2, "fvg_bull_invalidation_idx"] == 6.0
    assert result.loc[2, "fvg_bull_terminal_state"] == STATE_INVALIDATED


def test_fvg_research_overlay_produces_event_table_and_meaningful_rates() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (105.5, 106.0, 105.0, 105.8),
        (106.0, 108.0, 105.0, 106.0),
        (111.0, 112.0, 110.5, 111.5),
        (112.0, 113.0, 111.0, 112.5),
        (113.0, 114.0, 112.0, 113.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(df, atr_length=3, min_width_atr=0.0)
    research = build_fvg_research_table(debug["frame"], debug_tables=debug)
    summary = summarize_fvg_research(research)

    assert not research.empty
    assert "fvg_r_time_to_first_touch" in research.columns
    assert "fvg_r_hold_after_touch_5" in research.columns
    assert "fvg_r_final_outcome" in research.columns
    assert "fvg_terminal_label" in research.columns
    assert "fvg_r_continuation_without_touch_flag" in research.columns
    assert "fvg_r_touched_rejected_flag" in research.columns
    assert 0.0 <= summary["touch_rate"] <= 1.0
    assert "cleanliness_bucket_tradeable" in summary
    assert "never_touched_audit" in summary
    assert "touched_audit" in summary
    assert "reconciliation_summary" in summary
    assert "core_vs_research_audit" in summary
    assert "core_terminal_vs_research_final_crosstab" in summary
    assert "state_family_reconciliation" in summary
    assert "breakdown_reconciliation_checks" in summary
    assert "consistency_checks" in summary
    clean_bucket = summary["cleanliness_bucket_tradeable"]["dirty"]
    assert "eligible_count" in clean_bucket
    assert "success_count" in clean_bucket
    assert "rate" in clean_bucket
    for breakdown in summary["continuation_without_touch_audit"]["breakdowns"].values():
        for bucket in breakdown.values():
            assert "eligible_count" in bucket
            assert "success_count" in bucket
            assert "rate" in bucket
    assert summary["consistency_checks"]["final_outcomes_sum_to_event_count"] is True
    assert (
        summary["consistency_checks"]["research_final_outcome_is_terminal_faithful"]
        is True
    )


def test_research_final_outcome_mirrors_core_terminal_label() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (104.0, 105.0, 103.0, 104.5),
        (104.2, 105.0, 104.0, 104.6),
        (104.5, 106.0, 104.2, 105.0),
        (104.8, 106.0, 104.5, 105.2),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(
        df, atr_length=3, min_width_atr=0.0, max_active_age_bars=1
    )
    research = build_fvg_research_table(debug["frame"], debug_tables=debug)
    summary = summarize_fvg_research(research)

    assert (research["fvg_terminal_label"] == research["fvg_r_final_outcome"]).all()
    crosstab = summary["core_terminal_vs_research_final_crosstab"]
    assert crosstab["expired"]["expired"] == 1
    assert (
        summary["core_vs_research_audit"]["exact_one_mapping_or_exclusion_check"]
        is True
    )


def test_continuation_without_touch_flag_can_coexist_with_expired_final_outcome() -> (
    None
):
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 112.2, 101.0, 111.8),
        (111.0, 112.0, 110.5, 111.2),
        (111.3, 121.5, 111.0, 120.8),
        (120.8, 121.0, 111.5, 112.0),
        (112.0, 112.5, 111.0, 111.5),
    ] + [(111.5, 112.0, 111.0, 111.4)] * 20
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(
        df, atr_length=3, min_width_atr=0.0, max_active_age_bars=2
    )
    frame = debug["frame"].copy()
    frame["atr_14"] = frame["atr_3"]
    research = build_fvg_research_table(frame, debug_tables=debug)

    expired_event = research.iloc[0]
    assert expired_event["fvg_terminal_label"] == "expired"
    assert expired_event["fvg_r_final_outcome"] == "expired"
    assert bool(expired_event["fvg_r_continuation_without_touch_flag"]) is True


def test_touched_rejected_flag_can_coexist_with_invalidated_final_outcome() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 112.2, 101.0, 111.8),
        (111.0, 112.0, 110.5, 111.2),
        (111.5, 112.0, 110.4, 112.0),
        (111.8, 113.5, 110.8, 112.5),
        (103.0, 104.0, 99.0, 100.0),
    ] + [(100.0, 101.0, 99.0, 100.0)] * 20
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(df, atr_length=3, min_width_atr=0.0)
    frame = debug["frame"].copy()
    frame["atr_14"] = frame["atr_3"]
    research = build_fvg_research_table(frame, debug_tables=debug)

    event = research[research["fvg_terminal_label"] == "invalidated"].iloc[0]
    assert event["fvg_terminal_label"] == "invalidated"
    assert event["fvg_r_final_outcome"] == "invalidated"
    assert bool(event["fvg_r_touched_rejected_flag"]) is True
