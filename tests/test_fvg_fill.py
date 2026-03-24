from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.smc.fvg import (
    MAX_ACTIVE_AGE_BARS,
    collect_fvg_debug_tables,
)
from src.indicators.smc.fvg_fill import (
    FILL_TYPE_AGGRESSIVE,
    FILL_TYPE_NONE,
    FILL_TYPE_ORDERLY,
    FILL_TYPE_WICK_ONLY,
    add_fvg_fill,
)
from src.validation.indicators.fvg import (
    _summarize_fill,
    _summarize_fill_selection_reconciliation,
    _summarize_overlap_fill_forensics,
)


def _make_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h")
    return df[["timestamp", "open", "high", "low", "close"]]


def _run_fill(
    rows: list[tuple[float, float, float, float]],
    *,
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
) -> pd.DataFrame:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug_tables = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=max_active_age_bars,
    )
    return add_fvg_fill(debug_tables["frame"], debug_tables=debug_tables, atr_length=3)


def _run_fill_with_debug(
    rows: list[tuple[float, float, float, float]],
    *,
    max_active_age_bars: int | None = MAX_ACTIVE_AGE_BARS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    debug_tables = collect_fvg_debug_tables(
        df,
        atr_length=3,
        min_width_atr=0.0,
        max_active_age_bars=max_active_age_bars,
    )
    filled = add_fvg_fill(
        debug_tables["frame"], debug_tables=debug_tables, atr_length=3
    )
    return filled, debug_tables


def _blank_fill_input(n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": np.zeros(n, dtype=float),
            "high": np.zeros(n, dtype=float),
            "low": np.zeros(n, dtype=float),
            "close": np.zeros(n, dtype=float),
            "atr_3": np.ones(n, dtype=float),
        }
    )
    for side in ("bull", "bear"):
        df[f"fvg_{side}_active"] = 0
        df[f"fvg_{side}_active_id"] = 0
        df[f"fvg_{side}_active_low"] = np.nan
        df[f"fvg_{side}_active_high"] = np.nan
        df[f"fvg_{side}_active_mid"] = np.nan
        df[f"fvg_{side}_active_width"] = np.nan
        df[f"fvg_{side}_active_width_atr"] = np.nan
        df[f"fvg_{side}_active_fill_pct"] = np.nan
        df[f"fvg_{side}_active_touch_count"] = 0
        df[f"fvg_{side}_active_state"] = 0
        df[f"fvg_{side}_active_count"] = 0
        df[f"fvg_{side}_active_age_bars"] = np.nan
        df[f"fvg_{side}_first_retest_flag"] = 0
        df[f"fvg_{side}_detect_flag"] = 0
        df[f"fvg_{side}_event_id"] = 0
        df[f"fvg_{side}_low"] = np.nan
        df[f"fvg_{side}_high"] = np.nan
        df[f"fvg_{side}_width"] = np.nan
        df[f"fvg_{side}_first_touch_idx"] = np.nan
        df[f"fvg_{side}_full_fill_idx"] = np.nan
        df[f"fvg_{side}_invalidation_idx"] = np.nan
        df[f"fvg_{side}_expiry_idx"] = np.nan
        df[f"fvg_{side}_terminal_state"] = 0
    return df


def _run_fill_manual(df: pd.DataFrame) -> pd.DataFrame:
    active_members: list[dict[str, object]] = []
    for side in ("bull", "bear"):
        prev_touch = 0
        prev_fill = 0.0
        for row_idx, row in df.iterrows():
            if int(row[f"fvg_{side}_active"]) != 1:
                prev_touch = 0
                prev_fill = 0.0
                continue
            touch_count = int(row[f"fvg_{side}_active_touch_count"])
            fill_pct = float(row[f"fvg_{side}_active_fill_pct"])
            zone_low = float(row[f"fvg_{side}_active_low"])
            zone_high = float(row[f"fvg_{side}_active_high"])
            close_value = float(row["close"])
            touched = touch_count > 0
            closed_outside = (
                (
                    (close_value >= zone_high)
                    if side == "bull"
                    else (close_value <= zone_low)
                )
                if touched
                else False
            )
            active_members.append(
                {
                    "row_idx": int(row_idx),
                    "side": side,
                    "active_id": int(row[f"fvg_{side}_active_id"]),
                    "detect_idx": int(
                        row[f"fvg_{side}_detect_flag"] == 1 and row_idx or row_idx - 1
                    ),
                    "age_anchor_idx": int(
                        row_idx - float(row[f"fvg_{side}_active_age_bars"])
                    ),
                    "active_since_idx": int(
                        row_idx - float(row[f"fvg_{side}_active_age_bars"])
                    ),
                    "low": zone_low,
                    "high": zone_high,
                    "base_quality": 0.0,
                    "touch_count": touch_count,
                    "fill_pct": fill_pct,
                    "state": int(row[f"fvg_{side}_active_state"]),
                    "selected": 1,
                    "first_touch_idx": (row_idx if touch_count > 0 else np.nan),
                    "last_touch_idx": (row_idx if touch_count > 0 else np.nan),
                    "first_partial_fill_idx": (row_idx if fill_pct > 0 else np.nan),
                    "max_single_bar_fill_jump": max(fill_pct - prev_fill, 0.0),
                    "all_touch_bars_closed_outside": int(closed_outside),
                    "last_touch_rejection_flag": (
                        1.0
                        if touched and closed_outside
                        else 0.0 if touched else np.nan
                    ),
                    "bounce_raw_max": 0.0,
                    "component_id": 1,
                    "component_owned_this_row": 1,
                    "touch_delta": max(touch_count - prev_touch, 0),
                    "fill_delta": max(fill_pct - prev_fill, 0.0),
                    "entered_this_row": int(touch_count > 0 or fill_pct > 0),
                }
            )
            prev_touch = touch_count
            prev_fill = fill_pct
    debug_tables = {
        "event_table": pd.DataFrame(),
        "active_rep_table": pd.DataFrame(),
        "active_member_table": pd.DataFrame(active_members),
    }
    return add_fvg_fill(df, debug_tables=debug_tables, atr_length=3)


def _bull_base_rows() -> list[tuple[float, float, float, float]]:
    return [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
    ]


def _bear_base_rows() -> list[tuple[float, float, float, float]]:
    return [
        (100.0, 101.0, 99.0, 100.0),
        (98.0, 100.0, 90.0, 91.0),
        (88.0, 89.0, 87.0, 88.0),
    ]


def test_bull_no_fill_untouched_zone() -> None:
    rows = _bull_base_rows() + [
        (112.5, 114.0, 112.0, 113.0),
        (113.0, 115.0, 112.5, 114.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_fill_type"] == FILL_TYPE_NONE
    assert result.loc[3, "fvg_fill_bull_remaining_low"] == 101.0
    assert result.loc[3, "fvg_fill_bull_remaining_high"] == 112.0
    assert result.loc[3, "fvg_fill_bull_remaining_width"] == 11.0
    assert result.loc[3, "fvg_fill_bull_remaining_frac"] == 1.0
    assert result.loc[3, "fvg_fill_bull_untouched_count"] == 1


def test_bear_no_fill_untouched_zone() -> None:
    rows = _bear_base_rows() + [
        (87.5, 88.5, 86.5, 87.0),
        (86.8, 88.0, 85.5, 86.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bear_fill_type"] == FILL_TYPE_NONE
    assert result.loc[3, "fvg_fill_bear_remaining_low"] == 88.5
    assert result.loc[3, "fvg_fill_bear_remaining_high"] == 99.0
    assert result.loc[3, "fvg_fill_bear_remaining_width"] == 10.5
    assert result.loc[3, "fvg_fill_bear_remaining_frac"] == 1.0
    assert result.loc[3, "fvg_fill_bear_untouched_count"] == 1


def test_bull_partial_fill_remaining_geometry() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
    ]
    result = _run_fill(rows)

    assert np.isclose(result.loc[3, "fvg_bull_active_fill_pct"], 0.3)
    assert result.loc[3, "fvg_fill_bull_remaining_low"] == 101.0
    assert np.isclose(result.loc[3, "fvg_fill_bull_remaining_high"], 108.0)
    assert np.isclose(result.loc[3, "fvg_fill_bull_remaining_width"], 7.0)
    assert np.isclose(result.loc[3, "fvg_fill_bull_remaining_frac"], 0.7)


def test_bear_partial_fill_remaining_geometry() -> None:
    rows = _bear_base_rows() + [
        (89.0, 94.0, 88.0, 93.0),
    ]
    result = _run_fill(rows)

    assert np.isclose(result.loc[3, "fvg_bear_active_fill_pct"], 0.5)
    assert np.isclose(result.loc[3, "fvg_fill_bear_remaining_low"], 94.0)
    assert result.loc[3, "fvg_fill_bear_remaining_high"] == 99.0
    assert np.isclose(result.loc[3, "fvg_fill_bear_remaining_width"], 5.0)
    assert np.isclose(result.loc[3, "fvg_fill_bear_remaining_frac"], 0.5)


def test_bull_full_fill_remaining_becomes_nan() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (110.0, 111.0, 100.0, 101.0),
    ]
    result = _run_fill(rows)

    assert np.isnan(result.loc[4, "fvg_fill_bull_remaining_low"])
    assert np.isnan(result.loc[4, "fvg_fill_bull_remaining_width"])


def test_bear_full_fill_remaining_becomes_nan() -> None:
    rows = _bear_base_rows() + [
        (89.0, 94.0, 88.0, 93.0),
        (94.0, 100.0, 93.0, 99.0),
    ]
    result = _run_fill(rows)

    assert np.isnan(result.loc[4, "fvg_fill_bear_remaining_low"])
    assert np.isnan(result.loc[4, "fvg_fill_bear_remaining_width"])


def test_bull_fill_velocity_same_zone() -> None:
    rows = _bull_base_rows() + [
        (111.0, 114.0, 110.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (110.0, 113.0, 106.0, 107.0),
    ]
    result = _run_fill(rows)

    assert np.isclose(result.loc[4, "fvg_fill_bull_fill_pct_delta_1"], 0.2)
    assert np.isclose(result.loc[5, "fvg_fill_bull_fill_pct_delta_1"], 0.2)
    assert np.isclose(result.loc[5, "fvg_fill_bull_fill_pct_delta_3"], 0.5)


def test_bull_fill_velocity_zone_change_resets() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    assert result.loc[5, "fvg_bull_active_id"] != result.loc[4, "fvg_bull_active_id"]
    assert result.loc[5, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"
    assert np.isclose(result.loc[5, "fvg_fill_bull_fill_pct_delta_1"], 0.0)


def test_bull_fill_type_wick_only() -> None:
    df = _blank_fill_input(4)
    df.loc[:, ["open", "high", "low", "close"]] = [
        [100.0, 101.0, 99.0, 100.0],
        [102.0, 110.0, 101.0, 109.0],
        [112.0, 113.0, 111.0, 112.0],
        [112.0, 114.0, 111.0, 112.5],
    ]
    df.loc[
        2,
        [
            "fvg_bull_detect_flag",
            "fvg_bull_event_id",
            "fvg_bull_low",
            "fvg_bull_high",
            "fvg_bull_width",
        ],
    ] = [
        1,
        1,
        101.0,
        111.0,
        10.0,
    ]
    df.loc[2:3, "fvg_bull_active"] = 1
    df.loc[2:3, "fvg_bull_active_id"] = 1
    df.loc[2:3, "fvg_bull_active_low"] = 101.0
    df.loc[2:3, "fvg_bull_active_high"] = 111.0
    df.loc[2:3, "fvg_bull_active_mid"] = 106.0
    df.loc[2:3, "fvg_bull_active_width"] = 10.0
    df.loc[2:3, "fvg_bull_active_width_atr"] = 10.0
    df.loc[2:3, "fvg_bull_active_count"] = 1
    df.loc[2, "fvg_bull_active_fill_pct"] = 0.0
    df.loc[3, "fvg_bull_active_fill_pct"] = 0.0
    df.loc[2, "fvg_bull_active_touch_count"] = 0
    df.loc[3, "fvg_bull_active_touch_count"] = 1
    df.loc[2, "fvg_bull_active_state"] = 1
    df.loc[3, "fvg_bull_active_state"] = 2
    df.loc[2, "fvg_bull_active_age_bars"] = 0.0
    df.loc[3, "fvg_bull_active_age_bars"] = 1.0
    result = _run_fill_manual(df)

    assert result.loc[3, "fvg_fill_bull_fill_type"] == FILL_TYPE_WICK_ONLY


def test_bull_fill_type_orderly() -> None:
    rows = _bull_base_rows() + [
        (111.0, 114.0, 110.8, 112.0),
        (112.0, 114.0, 109.6, 111.0),
    ]
    result = _run_fill(rows)

    assert result.loc[4, "fvg_fill_bull_fill_type"] == FILL_TYPE_ORDERLY


def test_bull_fill_type_aggressive() -> None:
    rows = _bull_base_rows() + [
        (111.0, 114.0, 105.0, 106.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_fill_type"] == FILL_TYPE_AGGRESSIVE


def test_fill_type_monotonic_no_revert() -> None:
    rows = _bull_base_rows() + [
        (111.0, 114.0, 105.0, 106.0),
        (110.0, 112.0, 104.0, 111.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_fill_type"] == FILL_TYPE_AGGRESSIVE
    assert result.loc[4, "fvg_fill_bull_fill_type"] == FILL_TYPE_AGGRESSIVE


def test_bull_bounce_strength_after_touch() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (109.0, 118.0, 108.5, 116.0),
    ]
    result = _run_fill(rows)

    assert result.loc[4, "fvg_fill_bull_bounce_strength_atr"] > 0.0


def test_bull_touch_rejection_flag() -> None:
    df = _blank_fill_input(4)
    df.loc[:, ["open", "high", "low", "close"]] = [
        [100.0, 101.0, 99.0, 100.0],
        [102.0, 110.0, 101.0, 109.0],
        [112.0, 113.0, 111.0, 112.0],
        [112.0, 114.0, 111.0, 112.5],
    ]
    df.loc[
        2,
        [
            "fvg_bull_detect_flag",
            "fvg_bull_event_id",
            "fvg_bull_low",
            "fvg_bull_high",
            "fvg_bull_width",
        ],
    ] = [
        1,
        1,
        101.0,
        111.0,
        10.0,
    ]
    df.loc[2:3, "fvg_bull_active"] = 1
    df.loc[2:3, "fvg_bull_active_id"] = 1
    df.loc[2:3, "fvg_bull_active_low"] = 101.0
    df.loc[2:3, "fvg_bull_active_high"] = 111.0
    df.loc[2:3, "fvg_bull_active_mid"] = 106.0
    df.loc[2:3, "fvg_bull_active_width"] = 10.0
    df.loc[2:3, "fvg_bull_active_width_atr"] = 10.0
    df.loc[2:3, "fvg_bull_active_count"] = 1
    df.loc[2, "fvg_bull_active_fill_pct"] = 0.0
    df.loc[3, "fvg_bull_active_fill_pct"] = 0.0
    df.loc[2, "fvg_bull_active_touch_count"] = 0
    df.loc[3, "fvg_bull_active_touch_count"] = 1
    df.loc[2, "fvg_bull_active_state"] = 1
    df.loc[3, "fvg_bull_active_state"] = 2
    df.loc[2, "fvg_bull_active_age_bars"] = 0.0
    df.loc[3, "fvg_bull_active_age_bars"] = 1.0
    result = _run_fill_manual(df)

    assert result.loc[3, "fvg_fill_bull_touch_rejection_flag"] == 1.0


def test_bull_touch_acceptance_flag() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_touch_rejection_flag"] == 0.0


def test_bull_bars_since_touch() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (111.2, 113.0, 111.1, 112.0),
        (111.3, 112.5, 111.2, 112.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_bars_since_touch"] == 0.0
    assert result.loc[4, "fvg_fill_bull_bars_since_touch"] == 1.0
    assert result.loc[5, "fvg_fill_bull_bars_since_touch"] == 2.0


def test_distance_to_remaining_zone() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
    ]
    result = _run_fill(rows)
    remaining_low = result.loc[3, "fvg_fill_bull_remaining_low"]
    remaining_high = result.loc[3, "fvg_fill_bull_remaining_high"]
    atr = result.loc[3, "atr_3"]

    expected_mid = (
        abs(result.loc[3, "close"] - (remaining_low + remaining_high) / 2.0) / atr
    )
    expected_near = (result.loc[3, "close"] - remaining_high) / atr
    expected_far = (result.loc[3, "close"] - remaining_low) / atr

    assert np.isclose(
        result.loc[3, "fvg_fill_bull_dist_to_remaining_mid_atr"], expected_mid
    )
    assert np.isclose(
        result.loc[3, "fvg_fill_bull_dist_to_remaining_near_atr"], expected_near
    )
    assert np.isclose(
        result.loc[3, "fvg_fill_bull_dist_to_remaining_far_atr"], expected_far
    )


def test_multi_zone_mean_fill_pct() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    # mean_fill_pct and max_fill_pct are now scoped to touched zones only so that
    # price-disjoint untouched zones don't dilute the signal.
    # One zone has 30% fill, one is untouched → mean and max of the touched pool = 0.3.
    assert np.isclose(result.loc[5, "fvg_fill_bull_mean_fill_pct"], 0.3)
    assert np.isclose(result.loc[5, "fvg_fill_bull_max_fill_pct"], 0.3)
    # pool_* columns retain the original pool-wide averages across ALL active reps.
    assert np.isclose(result.loc[5, "fvg_fill_bull_pool_mean_fill_pct"], 0.15)
    assert np.isclose(result.loc[5, "fvg_fill_bull_pool_max_fill_pct"], 0.3)


def test_multi_zone_untouched_count() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    assert result.loc[5, "fvg_fill_bull_untouched_count"] == 1


def test_terminal_zone_fill_columns_become_nan() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (110.0, 111.0, 100.0, 101.0),
        (101.0, 102.0, 100.0, 101.5),
    ]
    result = _run_fill(rows)

    assert np.isnan(result.loc[5, "fvg_fill_bull_remaining_low"])
    assert np.isnan(result.loc[5, "fvg_fill_bull_bars_since_touch"])
    assert result.loc[5, "fvg_fill_bull_fill_type"] == FILL_TYPE_NONE


def test_no_active_zone_all_nan() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.5, 101.5, 99.5, 100.2),
        (100.4, 101.4, 99.6, 100.1),
        (100.3, 101.3, 99.7, 100.0),
    ]
    result = _run_fill(rows)

    assert result["fvg_fill_bull_remaining_low"].isna().all()
    assert result["fvg_fill_bear_remaining_low"].isna().all()
    assert (result["fvg_fill_bull_fill_type"] == FILL_TYPE_NONE).all()
    assert (result["fvg_fill_bear_fill_type"] == FILL_TYPE_NONE).all()


def test_remaining_frac_equals_one_minus_fill_pct() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
    ]
    result = _run_fill(rows)

    assert np.isclose(
        result.loc[3, "fvg_fill_bull_remaining_frac"],
        1.0 - result.loc[3, "fvg_bull_active_fill_pct"],
    )


def test_bars_since_first_touch_increments() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (109.0, 113.0, 108.5, 111.0),
        (111.0, 112.5, 110.5, 112.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_bars_since_first_touch"] == 0.0
    assert result.loc[4, "fvg_fill_bull_bars_since_first_touch"] == 1.0
    assert result.loc[5, "fvg_fill_bull_bars_since_first_touch"] == 2.0


def test_selected_zone_features_reference_current_active_id() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    assert result.loc[5, "fvg_bull_active"] == 1
    assert np.isfinite(result.loc[5, "fvg_fill_bull_remaining_width"])


def test_fill_velocity_resets_when_active_id_changes() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    assert result.loc[5, "fvg_bull_active_id"] != result.loc[4, "fvg_bull_active_id"]
    assert result.loc[5, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"
    assert np.isclose(result.loc[5, "fvg_fill_bull_fill_pct_delta_1"], 0.0)


def test_bounce_strength_is_non_negative() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (109.0, 118.0, 108.5, 116.0),
    ]
    result = _run_fill(rows)

    assert result.loc[4, "fvg_fill_bull_bounce_strength_atr"] >= 0.0


def test_multi_zone_untouched_count_does_not_exceed_active_count() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)

    assert (
        result.loc[5, "fvg_fill_bull_untouched_count"]
        <= result.loc[5, "fvg_bull_active_count"]
    )


def test_overlap_older_untouched_newer_touched_keeps_older_untouched() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (113.0, 115.0, 112.5, 114.0),
        (116.0, 124.0, 115.0, 123.0),
        (125.0, 126.0, 124.5, 125.5),
        (124.0, 125.0, 114.2, 115.0),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    active_members = debug_tables["active_member_table"]
    overlap_row = active_members[
        (active_members["row_idx"] == 6) & (active_members["side"] == "bull")
    ].sort_values("detect_idx")

    older = overlap_row.iloc[0]
    newer = overlap_row.iloc[1]
    assert older["fill_pct"] == 0.0
    assert older["touch_count"] == 0
    assert older["state"] == 1
    assert newer["fill_pct"] > 0.0
    assert newer["touch_count"] == 1
    assert result.loc[6, "fvg_bull_active_count"] == 2
    assert result.loc[6, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"
    assert result.loc[6, "fvg_fill_bull_rep_matches_structural_selected"] == 0
    assert result.loc[6, "fvg_fill_bull_rep_fill_pct"] > 0.0
    assert result.loc[6, "fvg_fill_bull_rep_touch_count"] == 1


def test_overlap_older_partially_filled_newer_touch_does_not_change_older_without_reentry() -> (
    None
):
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (124.0, 125.0, 114.2, 115.0),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    active_members = debug_tables["active_member_table"]
    older_rows = active_members[
        (active_members["side"] == "bull") & (active_members["active_id"] == 1)
    ].sort_values("row_idx")

    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 5, "fill_pct"].iloc[0], 0.3
    )
    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 6, "fill_pct"].iloc[0], 0.0
    )
    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 6, "historical_fill_pct"].iloc[0], 0.3
    )

    newer_row = active_members[
        (active_members["row_idx"] == 6)
        & (active_members["side"] == "bull")
        & (active_members["active_id"] == 3)
    ].iloc[0]
    assert newer_row["fill_pct"] > 0.0
    assert result.loc[6, "fvg_fill_bull_rep_id"] == 3
    assert result.loc[6, "fvg_fill_bull_rep_fill_pct"] > 0.0
    assert result.loc[6, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"


def test_disjoint_overlap_keeps_historical_fill_but_single_live_fill() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (124.0, 125.0, 114.2, 115.0),
    ]
    _result, debug_tables = _run_fill_with_debug(rows)
    scoped = debug_tables["active_member_table"]
    scoped = scoped[(scoped["row_idx"] == 6) & (scoped["side"] == "bull")].copy()
    by_component = (
        scoped.groupby("component_id", sort=True)
        .agg(
            live_fill_pct=("fill_pct", "max"),
            historical_fill_pct=("historical_fill_pct", "max"),
        )
        .reset_index()
    )

    assert (by_component["live_fill_pct"] > 0).sum() == 1
    assert (by_component["historical_fill_pct"] > 0).sum() == 2


def test_stale_disjoint_live_fill_demotes_after_persistence_window() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (126.2, 127.1, 124.5, 126.4),
        (126.3, 127.2, 124.6, 126.5),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    active_members = debug_tables["active_member_table"]
    older_rows = active_members[
        (active_members["side"] == "bull") & (active_members["active_id"] == 1)
    ].sort_values("row_idx")

    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 5, "fill_pct"].iloc[0], 0.3
    )
    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 6, "fill_pct"].iloc[0], 0.0
    )
    assert np.isclose(
        older_rows.loc[older_rows["row_idx"] == 6, "historical_fill_pct"].iloc[0], 0.3
    )
    assert result.loc[5, "fvg_fill_bull_rep_id"] == 1
    assert result.loc[5, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"
    assert result.loc[6, "fvg_fill_bull_rep_id"] == result.loc[6, "fvg_bull_active_id"]
    assert result.loc[6, "fvg_fill_bull_rep_selection_mode"] == "structural_fallback"


def test_fill_rep_can_differ_from_structural_selected_active_id() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    active_members = debug_tables["active_member_table"]
    overlap_row = active_members[
        (active_members["row_idx"] == 5) & (active_members["side"] == "bull")
    ]
    selected_member = overlap_row[overlap_row["selected"] == 1].iloc[0]

    assert result.loc[5, "fvg_bull_active_id"] == selected_member["active_id"]
    assert result.loc[5, "fvg_fill_bull_rep_id"] != result.loc[5, "fvg_bull_active_id"]
    assert result.loc[5, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"
    assert result.loc[5, "fvg_fill_bull_rep_fill_pct"] == 0.3
    assert np.isfinite(result.loc[5, "fvg_fill_bull_remaining_width"])


def test_fill_rep_uses_structural_fallback_when_no_touched_candidates_exist() -> None:
    rows = _bull_base_rows() + [
        (112.5, 114.0, 112.0, 113.0),
        (113.0, 115.0, 112.5, 114.0),
    ]
    result = _run_fill(rows)

    assert result.loc[3, "fvg_fill_bull_rep_selection_mode"] == "structural_fallback"
    assert result.loc[3, "fvg_fill_bull_rep_matches_structural_selected"] == 1
    assert result.loc[3, "fvg_fill_bull_rep_fill_pct"] == 0.0


def test_true_untouched_excludes_zero_penetration_wick_touch_state() -> None:
    df = _blank_fill_input(4)
    df.loc[:, ["open", "high", "low", "close"]] = [
        [100.0, 101.0, 99.0, 100.0],
        [102.0, 110.0, 101.0, 109.0],
        [112.0, 113.0, 111.0, 112.0],
        [112.0, 114.0, 111.0, 112.5],
    ]
    df.loc[
        2,
        [
            "fvg_bull_detect_flag",
            "fvg_bull_event_id",
            "fvg_bull_low",
            "fvg_bull_high",
            "fvg_bull_width",
        ],
    ] = [1, 1, 101.0, 111.0, 10.0]
    df.loc[2:3, "fvg_bull_active"] = 1
    df.loc[2:3, "fvg_bull_active_id"] = 1
    df.loc[2:3, "fvg_bull_active_low"] = 101.0
    df.loc[2:3, "fvg_bull_active_high"] = 111.0
    df.loc[2:3, "fvg_bull_active_mid"] = 106.0
    df.loc[2:3, "fvg_bull_active_width"] = 10.0
    df.loc[2:3, "fvg_bull_active_width_atr"] = 10.0
    df.loc[2:3, "fvg_bull_active_count"] = 1
    df.loc[2:3, "fvg_bull_active_fill_pct"] = 0.0
    df.loc[2, "fvg_bull_active_touch_count"] = 0
    df.loc[3, "fvg_bull_active_touch_count"] = 1
    df.loc[2, "fvg_bull_active_state"] = 1
    df.loc[3, "fvg_bull_active_state"] = 1
    df.loc[2, "fvg_bull_active_age_bars"] = 0.0
    df.loc[3, "fvg_bull_active_age_bars"] = 1.0
    result = _run_fill_manual(df)

    assert result.loc[3, "fvg_fill_bull_untouched_count"] == 0
    assert result.loc[3, "fvg_fill_bull_rep_selection_mode"] == "touched_pool"


def test_overlap_forensics_summary_reports_zero_overlap_violations() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (102.0, 110.0, 101.0, 109.0),
        (112.0, 113.0, 111.0, 112.0),
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
        (124.0, 125.0, 114.2, 115.0),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    summary = _summarize_overlap_fill_forensics(
        result, active_members=debug_tables["active_member_table"]
    )

    assert summary["rows_with_multi_active_bull"] > 0
    assert summary["no_untouched_rep_with_nonzero_fill"] is True
    assert summary["no_untouched_rep_with_nonzero_touch_count"] is True
    assert summary["no_untouched_rep_fill_changed_without_entry"] is True
    assert summary["no_non_selected_untouched_rep_marked_as_filling"] is True
    assert summary["no_rep_with_nonzero_fill_before_first_entry"] is True


def test_fill_selection_reconciliation_reports_structural_fill_mismatch() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result, debug_tables = _run_fill_with_debug(rows)
    summary = _summarize_fill_selection_reconciliation(
        result, active_members=debug_tables["active_member_table"]
    )

    assert summary is not None
    assert summary["bull"]["rows_with_fill_rep_differs_from_structural_selected"] > 0
    assert summary["bull"]["rows_where_fill_rep_differs_from_structural_selected"] > 0
    assert summary["bull"]["rows_where_visual_confusion_pattern_occurs"] > 0
    assert summary["bull"]["rows_with_touched_candidates_present"] > 0
    assert summary["bull"]["invariants"]["fill_rep_matches_chosen_core_member"] is True
    row_audit = summary["bull"]["row_level_mismatch_audit"]
    assert row_audit
    assert row_audit[0]["structural_selected_id"] != row_audit[0]["fill_rep_id"]
    assert np.isfinite(row_audit[0]["structural_selected_remaining_width"])
    assert np.isfinite(row_audit[0]["fill_rep_remaining_width"])


def test_fill_summary_invariants_hold_on_structural_vs_fill_mismatch_path() -> None:
    rows = _bull_base_rows() + [
        (112.0, 114.0, 108.0, 109.0),
        (115.0, 123.0, 114.0, 122.0),
        (126.0, 127.0, 124.0, 126.0),
    ]
    result = _run_fill(rows)
    summary = _summarize_fill(result)

    assert summary is not None
    audit = summary["bull"]["invariant_audit"]
    assert audit["remaining_width_non_negative"] is True
    assert audit["remaining_width_lte_original"] is True
    assert audit["remaining_frac_matches_one_minus_fill_pct_max_abs_err"] < 1e-12
