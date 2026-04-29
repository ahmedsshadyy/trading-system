from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.ob import (
    OB_STATE_ACTIVE_FRESH,
    OB_STATE_INVALIDATED,
    OB_STATE_MITIGATED_FULL,
    add_ob,
)
from src.indicators.smc.ob_mitigation import add_ob_mitigation
from src.validation.indicators.ob import summarize_ob


def _base_ob_frame() -> pd.DataFrame:
    rows = [
        (100.0, 101.0, 99.5, 100.5),
        (100.5, 101.5, 100.0, 101.0),
        (102.0, 103.0, 99.0, 100.0),  # source bearish candle
        (100.0, 103.0, 99.5, 102.5),
        (102.5, 104.0, 101.5, 103.5),
        (103.5, 105.0, 103.0, 104.5),  # parent BOS row
        (104.5, 108.0, 104.0, 107.0),  # displacement / activation row
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h", tz="UTC")
    df["volume"] = 1000
    df["atr_14"] = 1.0
    df["bos_bull"] = 0
    df["bos_bear"] = 0
    df["bos_source_idx"] = np.nan
    df["bos_displacement_score"] = np.nan
    df["displacement_flag"] = 0
    df["displacement_direction"] = 0
    df["displacement_score"] = np.nan

    df.loc[5, "bos_bull"] = 1
    df.loc[5, "bos_source_idx"] = 1
    df.loc[5, "bos_displacement_score"] = 0.62
    df.loc[6, "displacement_flag"] = 1
    df.loc[6, "displacement_direction"] = 1
    df.loc[6, "displacement_score"] = 0.81
    return df


def _frame_with_full_mitigation() -> pd.DataFrame:
    df = _base_ob_frame().copy()
    tail = pd.DataFrame(
        [
            (107.0, 110.0, 106.0, 109.0),
            (103.0, 103.5, 100.5, 102.5),  # first touch, partial, midpoint
            (101.0, 103.0, 98.5, 100.5),  # full mitigation, not invalidated
        ],
        columns=["open", "high", "low", "close"],
    )
    tail["timestamp"] = pd.date_range(
        df["timestamp"].iloc[-1] + pd.Timedelta(hours=4),
        periods=len(tail),
        freq="4h",
        tz="UTC",
    )
    tail["volume"] = 1000
    tail["atr_14"] = 1.0
    tail["bos_bull"] = 0
    tail["bos_bear"] = 0
    tail["bos_source_idx"] = np.nan
    tail["bos_displacement_score"] = np.nan
    tail["displacement_flag"] = 0
    tail["displacement_direction"] = 0
    tail["displacement_score"] = np.nan
    return pd.concat([df, tail], ignore_index=True)


def _frame_with_invalidation() -> pd.DataFrame:
    df = _base_ob_frame().copy()
    tail = pd.DataFrame(
        [
            (107.0, 110.0, 106.0, 109.0),
            (100.5, 102.5, 98.0, 98.5),  # touch + full + invalidation
        ],
        columns=["open", "high", "low", "close"],
    )
    tail["timestamp"] = pd.date_range(
        df["timestamp"].iloc[-1] + pd.Timedelta(hours=4),
        periods=len(tail),
        freq="4h",
        tz="UTC",
    )
    tail["volume"] = 1000
    tail["atr_14"] = 1.0
    tail["bos_bull"] = 0
    tail["bos_bear"] = 0
    tail["bos_source_idx"] = np.nan
    tail["bos_displacement_score"] = np.nan
    tail["displacement_flag"] = 0
    tail["displacement_direction"] = 0
    tail["displacement_score"] = np.nan
    return pd.concat([df, tail], ignore_index=True)


def test_add_ob_uses_source_bos_and_activation_doctrine() -> None:
    result = add_ob(_base_ob_frame())

    event_rows = result[result["ob_id"] > 0]
    assert len(event_rows) == 1

    row = event_rows.iloc[0]
    assert int(row["ob_id"]) == 1
    assert row["ob_family"] == "bos"
    assert row["ob_parent_event_type"] == "bos"
    assert int(row["ob_side"]) == 1
    assert int(row["ob_source_idx"]) == 2
    assert int(row["ob_parent_bos_idx"]) == 5
    assert int(row["ob_activate_idx"]) == 5
    assert row["ob_parent_bos_ts"] == result.loc[5, "timestamp"]
    assert row["ob_source_ts"] == result.loc[2, "timestamp"]
    assert row["ob_activate_ts"] == result.loc[5, "timestamp"]
    assert int(row["ob_traceback_start_idx"]) == 3
    assert int(row["ob_traceback_end_idx"]) == 5
    assert int(row["ob_source_is_opposing_candle_bool"]) == 1
    assert row["ob_source_selection_reason"] == "last_opposing_before_displacement_leg"
    assert row["ob_bull"] == 1
    assert row["ob_bear"] == 0
    assert np.isclose(row["ob_zone_high"], 103.0)
    assert np.isclose(row["ob_zone_low"], 99.0)
    assert np.isclose(row["ob_zone_mid"], 101.0)
    assert np.isclose(row["ob_zone_height_abs"], 4.0)
    assert np.isclose(row["ob_width_atr"], 4.0)
    assert np.isclose(row["ob_body_high"], 102.0)
    assert np.isclose(row["ob_body_low"], 100.0)
    assert np.isclose(row["ob_parent_displacement_score"], 0.62)
    assert int(row["ob_state"]) == OB_STATE_ACTIVE_FRESH


def test_add_ob_mitigation_tracks_partial_and_full_without_invalidation() -> None:
    result = add_ob(_frame_with_full_mitigation())
    result = add_ob_mitigation(result)

    row = result[result["ob_id"] == 1].iloc[0]
    assert int(row["ob_state"]) == OB_STATE_MITIGATED_FULL
    assert int(row["ob_has_been_touched"]) == 1
    assert int(row["ob_has_partial_mitigation"]) == 1
    assert int(row["ob_has_full_mitigation"]) == 1
    assert int(row["ob_first_touch_idx"]) == 8
    assert int(row["ob_first_partial_mitigation_idx"]) == 8
    assert int(row["ob_first_full_mitigation_idx"]) == 9
    assert row["ob_first_touch_ts"] == result.loc[8, "timestamp"]
    assert row["ob_first_full_mitigation_ts"] == result.loc[9, "timestamp"]
    assert np.isnan(row["ob_invalidation_idx"])
    assert np.isclose(row["ob_mitigation_penetration_frac"], 1.0)
    assert int(row["ob_touch_count"]) == 2
    assert int(row["ob_mitigation_count"]) == 2
    assert int(row["ob_midpoint_touch_flag"]) == 1
    assert int(row["ob_midpoint_touch_idx"]) == 8

    assert int(result.loc[7, "ob_unmitigated_bull"]) == 1
    assert int(result.loc[8, "ob_first_retest"]) == 1
    assert int(result.loc[8, "ob_unmitigated_bull"]) == 0
    assert int(result.loc[8, "ob_bull_active"]) == 1
    assert int(result.loc[8, "ob_bull_active_state"]) >= 2
    assert int(result.loc[9, "ob_bull_active"]) == 0


def test_add_ob_mitigation_invalidation_has_precedence() -> None:
    result = add_ob(_frame_with_invalidation())
    result = add_ob_mitigation(result)

    row = result[result["ob_id"] == 1].iloc[0]
    assert int(row["ob_state"]) == OB_STATE_INVALIDATED
    assert int(row["ob_is_invalidated"]) == 1
    assert int(row["ob_has_been_touched"]) == 1
    assert int(row["ob_first_touch_idx"]) == 8
    assert int(row["ob_invalidation_idx"]) == 8
    assert int(row["ob_first_full_mitigation_idx"]) == 8


def test_summarize_ob_reports_clean_contract() -> None:
    result = add_ob(_frame_with_full_mitigation())
    result = add_ob_mitigation(result)
    summary = summarize_ob(result)

    assert summary["event_counts"]["ob_count"] == 1
    assert summary["event_counts"]["qualified_canonical_ob_count"] == 1
    assert summary["sanity_checks"]["every_ob_has_parent_bos"] is True
    assert summary["sanity_checks"]["geometry_positive"] is True
    assert summary["sanity_checks"]["bull_source_candle_is_bearish"] is True
    assert summary["sanity_checks"]["geometry_full_range_consistency"] is True
    assert summary["sanity_checks"]["activation_equals_parent_confirmation"] is True
    assert summary["mitigation_checks"]["no_touch_before_activation"] is True
    assert summary["mitigation_checks"]["full_not_before_touch"] is True
