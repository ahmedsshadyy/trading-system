from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.foundation.volatility import add_atr
from src.indicators.smc.fvg import collect_fvg_debug_tables
from src.indicators.smc.fvg_fill import add_fvg_fill
from src.indicators.smc.ifvg import (
    IFVG_STATE_EXPIRED,
    IFVG_STATE_FULLY_RECLAIMED,
    IFVG_STATE_ACTIVE_UNTESTED,
    MAX_IFVG_ACTIVE_AGE_BARS,
    _IfvgEvent,
    _build_invalidated_source_candidates,
    _select_active_ifvg,
    add_ifvg,
)
from src.validation.indicators.fvg import summarize_fvg


def _make_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h")
    return df[["timestamp", "open", "high", "low", "close"]]


def _run_ifvg(
    rows: list[tuple[float, float, float, float]],
    *,
    min_source_width_atr: float = 0.1,
    min_source_age_bars: int = 2,
    min_inversion_body_atr: float = 0.0,
    min_cross_distance_atr: float = 0.0,
    max_ifvg_age_bars: int | None = MAX_IFVG_ACTIVE_AGE_BARS,
    selector: str = "significance",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug_tables = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=20,
    )
    result = add_ifvg(
        debug_tables["frame"],
        debug_tables=debug_tables,
        atr_length=3,
        min_source_width_atr=min_source_width_atr,
        min_source_age_bars=min_source_age_bars,
        min_inversion_body_atr=min_inversion_body_atr,
        min_cross_distance_atr=min_cross_distance_atr,
        max_ifvg_age_bars=max_ifvg_age_bars,
        selector=selector,
    )
    return result, debug_tables


def test_bull_fvg_invalidated_creates_bear_ifvg() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (103.0, 104.0, 99.0, 100.0),
    ]
    result, _debug = _run_ifvg(rows)

    assert result.loc[4, "ifvg_bear_detect_flag"] == 1
    assert result.loc[4, "ifvg_bear_source_fvg_event_id"] == 1
    assert result.loc[4, "ifvg_bear_source_fvg_direction"] == 1
    assert result.loc[4, "ifvg_bear_direction"] == -1
    assert result.loc[4, "ifvg_bear_low"] == 101.0
    assert result.loc[4, "ifvg_bear_high"] == 111.0


def test_bear_fvg_invalidated_creates_bull_ifvg() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, _debug = _run_ifvg(rows)

    assert result.loc[4, "ifvg_bull_detect_flag"] == 1
    assert result.loc[4, "ifvg_bull_source_fvg_event_id"] == 1
    assert result.loc[4, "ifvg_bull_source_fvg_direction"] == -1
    assert result.loc[4, "ifvg_bull_direction"] == 1
    assert result.loc[4, "ifvg_bull_low"] == 89.0
    assert result.loc[4, "ifvg_bull_high"] == 99.0


def test_ifvg_detect_idx_equals_source_invalidation_idx() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, debug = _run_ifvg(rows)
    event_table = debug["event_table"]
    source = event_table[event_table["terminal_state"] == 5].iloc[0]

    assert result.loc[4, "ifvg_bull_detect_idx"] == source["invalidation_idx"]


def test_no_ifvg_active_before_detect() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, _debug = _run_ifvg(rows)

    assert (result.loc[:3, "ifvg_bull_active"] == 0).all()
    assert result.loc[4, "ifvg_bull_active"] == 1


def test_source_fvg_too_young_no_ifvg() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_age_bars=2)

    assert result["ifvg_bull_detect_flag"].sum() == 0
    assert result["ifvg_bear_detect_flag"].sum() == 0


def test_source_fvg_too_small_no_ifvg() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_width_atr=5.0, min_source_age_bars=1)

    assert result["ifvg_bull_detect_flag"].sum() == 0


def test_source_fill_pct_at_inversion_comes_from_previous_row() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_age_bars=1)

    assert np.isclose(result.loc[4, "ifvg_bull_source_fill_pct_at_inversion"], 0.5)


def test_first_retest_flag_is_one_shot_and_test_count_uses_episode_starts() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (100.0, 101.0, 95.0, 96.0),
        (99.5, 100.5, 96.0, 98.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_age_bars=1)

    assert result.loc[5, "ifvg_bull_first_retest_flag"] == 1
    assert result.loc[6, "ifvg_bull_first_retest_flag"] == 0
    assert result.loc[5, "ifvg_bull_active_test_count"] == 1
    assert result.loc[6, "ifvg_bull_active_test_count"] == 1


def test_retest_depth_frac_bullish() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (100.0, 101.0, 95.0, 96.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_age_bars=1)

    assert np.isclose(result.loc[5, "ifvg_bull_retest_depth_frac"], 0.4)


def test_ifvg_fully_reclaimed_terminal() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (100.0, 101.0, 95.0, 96.0),
        (90.0, 91.0, 87.0, 88.0),
    ]
    result, _debug = _run_ifvg(rows, min_source_age_bars=1)

    assert result.loc[4, "ifvg_bull_fully_reclaimed_idx"] == 6.0
    assert result.loc[4, "ifvg_bull_terminal_state"] == IFVG_STATE_FULLY_RECLAIMED
    assert result.loc[6, "ifvg_bull_zone_fully_reclaimed"] == 1
    assert result.loc[6, "ifvg_bull_active"] == 0


def test_ifvg_expired_terminal() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (102.0, 103.0, 101.0, 102.0),
        (103.0, 104.0, 102.0, 103.0),
    ]
    result, _debug = _run_ifvg(
        rows,
        min_source_age_bars=1,
        max_ifvg_age_bars=2,
    )

    assert result.loc[4, "ifvg_bull_expiry_idx"] == 6.0
    assert result.loc[4, "ifvg_bull_terminal_state"] == IFVG_STATE_EXPIRED


def test_default_on_expiry_removes_stale_ifvg_from_active_export() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ] + [(102.0, 103.0, 101.0, 102.0)] * 50
    result, _debug = _run_ifvg(rows, min_source_age_bars=1)

    expiry_row = 4 + MAX_IFVG_ACTIVE_AGE_BARS
    assert result.loc[expiry_row - 1, "ifvg_bull_active"] == 1
    assert result.loc[expiry_row, "ifvg_bull_active"] == 0
    assert result.loc[4, "ifvg_bull_expiry_idx"] == float(expiry_row)
    assert result.loc[4, "ifvg_bull_terminal_state"] == IFVG_STATE_EXPIRED


def test_max_ifvg_age_bars_must_be_positive() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug_tables = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=20,
    )

    with pytest.raises(ValueError, match="max_ifvg_age_bars"):
        add_ifvg(debug_tables["frame"], debug_tables=debug_tables, max_ifvg_age_bars=0)

    with pytest.raises(ValueError, match="max_ifvg_age_bars"):
        add_ifvg(
            debug_tables["frame"], debug_tables=debug_tables, max_ifvg_age_bars=None
        )


def test_fully_reclaimed_wins_over_expiry_on_same_bar() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (102.0, 103.0, 101.0, 102.0),
        (90.0, 91.0, 87.0, 88.0),
    ]
    result, _debug = _run_ifvg(
        rows,
        min_source_age_bars=1,
        max_ifvg_age_bars=2,
    )

    assert result.loc[4, "ifvg_bull_terminal_state"] == IFVG_STATE_FULLY_RECLAIMED
    assert result.loc[4, "ifvg_bull_fully_reclaimed_idx"] == 6.0
    assert np.isnan(result.loc[4, "ifvg_bull_expiry_idx"])


def test_active_count_membership_unchanged_by_selector_before_expiry() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
        (102.0, 104.0, 101.0, 103.0),
        (104.0, 105.0, 103.0, 104.0),
    ]
    nearest, debug = _run_ifvg(
        rows,
        min_source_age_bars=1,
        selector="nearest",
    )
    significance = add_ifvg(
        debug["frame"],
        debug_tables=debug,
        atr_length=3,
        min_source_age_bars=1,
        selector="significance",
    )

    assert np.array_equal(
        nearest["ifvg_bull_active_count"].to_numpy(),
        significance["ifvg_bull_active_count"].to_numpy(),
    )
    assert np.array_equal(
        nearest["ifvg_bear_active_count"].to_numpy(),
        significance["ifvg_bear_active_count"].to_numpy(),
    )


def test_significance_selector_prefers_fresher_higher_quality_event() -> None:
    older = _IfvgEvent(
        event_id=1,
        side="bull",
        direction=1,
        detect_idx=10,
        detect_ts=pd.Timestamp("2024-01-01", tz="UTC"),
        low=98.0,
        high=102.0,
        mid=100.0,
        width=4.0,
        width_atr=1.0,
        source_fvg_event_id=11,
        source_fvg_detect_idx=9,
        source_fvg_direction=-1,
        source_age_bars=5,
        source_width_atr=1.0,
        source_fill_pct_at_inversion=0.0,
        inversion_body_atr=0.5,
        inversion_range_atr=1.0,
        inversion_body_frac=0.5,
        inversion_close_location=0.5,
        inversion_conviction_location=0.5,
        cross_distance_atr=0.5,
        inversion_score=0.25,
        active_since_idx=10,
    )
    fresher = _IfvgEvent(
        event_id=2,
        side="bull",
        direction=1,
        detect_idx=95,
        detect_ts=pd.Timestamp("2024-01-02", tz="UTC"),
        low=90.0,
        high=94.0,
        mid=92.0,
        width=4.0,
        width_atr=1.0,
        source_fvg_event_id=22,
        source_fvg_detect_idx=94,
        source_fvg_direction=-1,
        source_age_bars=5,
        source_width_atr=1.0,
        source_fill_pct_at_inversion=0.0,
        inversion_body_atr=1.0,
        inversion_range_atr=1.0,
        inversion_body_frac=0.7,
        inversion_close_location=0.7,
        inversion_conviction_location=0.7,
        cross_distance_atr=1.0,
        inversion_score=0.95,
        active_since_idx=95,
    )
    active_rows = [
        (older, {"state_for_row": IFVG_STATE_ACTIVE_UNTESTED}),
        (fresher, {"state_for_row": IFVG_STATE_ACTIVE_UNTESTED}),
    ]

    nearest = _select_active_ifvg(
        side="bull",
        close_value=100.0,
        atr_value=1.0,
        current_idx=100,
        active_rows=active_rows,
        selector="nearest",
    )
    significance = _select_active_ifvg(
        side="bull",
        close_value=100.0,
        atr_value=1.0,
        current_idx=100,
        active_rows=active_rows,
        selector="significance",
    )

    assert nearest is not None and nearest[0].event_id == 1
    assert significance is not None and significance[0].event_id == 2


def test_ifvg_summary_reconciles_canonical_and_dense_universes() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ] + [(102.0, 103.0, 101.0, 102.0)] * 50
    result, debug = _run_ifvg(rows, min_source_age_bars=1)
    result = add_fvg_fill(result, debug_tables=debug)
    summary = summarize_fvg(
        result,
        full_df=result,
        debug_tables=debug,
        research_table=None,
        old_no_expiry_research_table=None,
    )
    ifvg_summary = summary["ifvg_summary"]
    reconciliation = summary["ifvg_universe_reconciliation"]
    active_policy = summary["ifvg_active_pool_policy_audit"]
    lifecycle = summary["ifvg_lifecycle_audit"]

    assert ifvg_summary["canonical_ifvg_count"] == (
        ifvg_summary["bull_detect_count"] + ifvg_summary["bear_detect_count"]
    )
    assert (
        reconciliation["canonical_ifvg_count"]
        == reconciliation["event_distribution_eligible_count"]
    )
    assert active_policy["not_causality_error"] is True
    assert lifecycle["no_fully_reclaimed_rep_still_active"] is True
    assert lifecycle["no_expired_rep_still_active"] is True
    assert lifecycle["no_invalid_terminal_rep_still_active"] is True


def test_candidate_builder_has_no_orphan_ifvgs() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
        (89.0, 94.0, 88.0, 93.0),
        (100.0, 102.0, 99.0, 101.0),
    ]
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug = collect_fvg_debug_tables(df, atr_length=3, min_width_atr=0.0)
    candidates = _build_invalidated_source_candidates(
        debug["frame"],
        debug_tables=debug,
        atr_length=3,
        min_source_age_bars=1,
    )

    assert not candidates.empty
    invalidated_sources = set(
        debug["event_table"]
        .loc[debug["event_table"]["terminal_state"] == 5, "event_id"]
        .astype(int)
        .tolist()
    )
    accepted_sources = set(
        candidates.loc[candidates["accepted"], "source_fvg_event_id"]
        .astype(int)
        .tolist()
    )
    assert accepted_sources.issubset(invalidated_sources)
