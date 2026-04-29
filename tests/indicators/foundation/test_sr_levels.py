from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import validate_sr_levels as vsr
from src.indicators.foundation.sr_levels import (
    ABSORB_TOL_ATR,
    BREAK_CONFIRM_CLOSES,
    FAMILY_MAX_AGE,
    FAMILY_PRIOR,
    MAX_AGE_BARS,
    SR_FAMILY_DAY,
    SR_FAMILY_EQHL,
    SR_FAMILY_SWING,
    SR_STATE_ACTIVE,
    SR_STATE_ACTIVE_WEAKENED,
    SR_STATE_INVALIDATED,
    SR_STATE_RETIRED,
    SR_SIDE_RESISTANCE,
    SR_SIDE_SUPPORT,
    add_sr_levels,
    build_sr_level_registry,
    extract_sr_source_events,
    project_sr_context,
    update_sr_lifecycle,
)
from src.validation.indicators.sr_levels import summarize_sr_levels

ATR = 5.0


def _base(n: int, close: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.full(n, close),
            "high": np.full(n, close + 1.0),
            "low": np.full(n, close - 1.0),
            "close": np.full(n, close),
            "volume": np.ones(n),
            "atr_14": np.full(n, ATR),
        }
    )


def _ensure_swing_cols(df: pd.DataFrame) -> None:
    if "swing_high_confirm_flag" not in df.columns:
        df["swing_high_confirm_flag"] = 0
        df["swing_high_confirm_price"] = np.nan
        df["swing_high_confirm_origin_idx"] = np.nan
        df["swing_low_confirm_flag"] = 0
        df["swing_low_confirm_price"] = np.nan
        df["swing_low_confirm_origin_idx"] = np.nan


def _swing_confirm(
    df: pd.DataFrame,
    *,
    bar: int,
    price: float,
    side: int,
    origin_bar: int | None = None,
) -> None:
    _ensure_swing_cols(df)
    if origin_bar is None:
        origin_bar = max(0, bar - 3)
    if side == SR_SIDE_SUPPORT:
        df.at[bar, "swing_low_confirm_flag"] = 1
        df.at[bar, "swing_low_confirm_price"] = price
        df.at[bar, "swing_low_confirm_origin_idx"] = origin_bar
    else:
        df.at[bar, "swing_high_confirm_flag"] = 1
        df.at[bar, "swing_high_confirm_price"] = price
        df.at[bar, "swing_high_confirm_origin_idx"] = origin_bar


def _emitted_supports(registry: dict[int, object]) -> list[object]:
    return [
        lev
        for lev in registry.values()
        if lev.side == SR_SIDE_SUPPORT and getattr(lev, "emitted_zone_flag", False)
    ]


def test_source_absorption_merges_nearby_supports() -> None:
    df = _base(40)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(df, bar=7, price=95.6, side=SR_SIDE_SUPPORT, origin_bar=3)

    registry = build_sr_level_registry(df)
    project_sr_context(df, registry)

    emitted = _emitted_supports(registry)
    absorbed = [
        lev
        for lev in registry.values()
        if lev.side == SR_SIDE_SUPPORT and lev.absorbed_by
    ]
    assert len(emitted) == 1
    assert len(absorbed) == 1
    assert emitted[0].anchor_count == 2
    assert abs(emitted[0].level_price - 95.3) < ABSORB_TOL_ATR * ATR


def test_multi_anchor_zone_width_expands() -> None:
    single_df = _base(40)
    _swing_confirm(single_df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    single_registry = build_sr_level_registry(single_df)
    project_sr_context(single_df, single_registry)
    single_width = _emitted_supports(single_registry)[0].zone_half_width_atr

    df = _base(40)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(df, bar=7, price=96.2, side=SR_SIDE_SUPPORT, origin_bar=3)

    registry = build_sr_level_registry(df)
    project_sr_context(df, registry)

    zone = _emitted_supports(registry)[0]
    assert zone.anchor_count == 2
    assert zone.zone_half_width_atr > single_width


def test_family_aware_absorption_preserves_eqhl_best_family() -> None:
    df = _base(50)
    df["prev_day_low"] = np.nan
    df.loc[10:, "prev_day_low"] = 99.0
    df["eql_detect_flag"] = 0
    df["eql_level_on_detect"] = np.nan
    df["eql_origin_idx"] = np.nan
    df["eql_score_on_detect"] = np.nan
    df["eql_member_count_on_detect"] = np.nan
    df.at[12, "eql_detect_flag"] = 1
    df.at[12, "eql_level_on_detect"] = 99.1
    df.at[12, "eql_origin_idx"] = 8
    df.at[12, "eql_score_on_detect"] = 0.95
    df.at[12, "eql_member_count_on_detect"] = 3

    registry = build_sr_level_registry(df)
    project_sr_context(df, registry)

    emitted = _emitted_supports(registry)
    assert emitted
    zone = emitted[0]
    assert zone.family_mix_counts[SR_FAMILY_DAY] >= 1
    assert zone.family_mix_counts[SR_FAMILY_EQHL] >= 1
    assert zone.best_source_family == SR_FAMILY_EQHL


def test_width_model_distinguishes_tight_vs_broad_anchor_dispersion() -> None:
    tight_df = _base(40)
    _swing_confirm(tight_df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(tight_df, bar=7, price=95.1, side=SR_SIDE_SUPPORT, origin_bar=3)
    tight_registry = build_sr_level_registry(tight_df)
    project_sr_context(tight_df, tight_registry)
    tight_width = _emitted_supports(tight_registry)[0].zone_width_atr

    broad_df = _base(40)
    _swing_confirm(broad_df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(broad_df, bar=7, price=96.1, side=SR_SIDE_SUPPORT, origin_bar=3)
    broad_registry = build_sr_level_registry(broad_df)
    project_sr_context(broad_df, broad_registry)
    broad_width = _emitted_supports(broad_registry)[0].zone_width_atr

    assert broad_width > tight_width


def test_break_pending_then_reclaim() -> None:
    df = _base(40)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    df.loc[10, ["close", "low", "high"]] = [94.0, 93.9, 95.5]
    df.loc[11, ["close", "low", "high"]] = [95.2, 94.8, 96.0]

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)
    zone = _emitted_supports(registry)[0]

    assert zone.invalidation_idx == -1
    assert zone.reclaim_count == 1
    assert zone.failed_break_count == 1
    assert zone.state in {SR_STATE_ACTIVE, SR_STATE_ACTIVE_WEAKENED}
    assert int(out["sr_reclaim_this_bar_flag"].iloc[11]) == 1
    assert int(out["support_broken_this_bar"].iloc[10]) == 0


def test_confirmed_invalidation_after_two_break_closes() -> None:
    df = _base(40)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    df.loc[10, ["close", "low", "high"]] = [94.1, 94.0, 95.2]
    df.loc[11, ["close", "low", "high"]] = [94.0, 93.9, 95.1]

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)
    zone = _emitted_supports(registry)[0]

    assert BREAK_CONFIRM_CLOSES == 2
    assert zone.state in {SR_STATE_INVALIDATED, SR_STATE_RETIRED}
    assert zone.invalidation_idx == 11
    assert int(out["support_broken_this_bar"].iloc[11]) == 1


def test_vp_dwell_and_shift_suppresses_small_noise() -> None:
    df = _base(20)
    df["vp_val"] = np.nan
    df.loc[2:8, "vp_val"] = 90.0
    df.loc[9:12, "vp_val"] = 90.5
    df.loc[13:18, "vp_val"] = 92.0

    events = extract_sr_source_events(df)
    vp_supports = [
        lev
        for lev in events
        if lev.side == SR_SIDE_SUPPORT and lev.source_family == "vp"
    ]
    assert len(vp_supports) == 2


def test_primary_zone_can_differ_from_geometric_nearest() -> None:
    df = _base(60)
    df["prev_day_low"] = np.nan
    df.loc[10:, "prev_day_low"] = 99.0
    _swing_confirm(df, bar=12, price=97.4, side=SR_SIDE_SUPPORT, origin_bar=6)
    if "eql_detect_flag" not in df.columns:
        df["eql_detect_flag"] = 0
        df["eql_level_on_detect"] = np.nan
        df["eql_origin_idx"] = np.nan
        df["eql_score_on_detect"] = np.nan
        df["eql_member_count_on_detect"] = np.nan
    df.at[13, "eql_detect_flag"] = 1
    df.at[13, "eql_level_on_detect"] = 97.6
    df.at[13, "eql_origin_idx"] = 8
    df.at[13, "eql_score_on_detect"] = 0.92
    df.at[13, "eql_member_count_on_detect"] = 3
    for bar in (18, 22, 26):
        df.loc[bar, ["low", "close", "high"]] = [97.5, 100.0, 101.0]

    out = add_sr_levels(df, include_research_only=False)
    probe = out.iloc[30]
    assert probe["nearest_support_price"] == pytest.approx(99.0)
    assert probe["primary_support_zone_mid"] < probe["nearest_support_price"]
    assert probe["primary_support_zone_score"] >= probe["nearest_support_strength"]


def test_reaction_quality_penalizes_weak_pierce_churn() -> None:
    clean_df = _base(60)
    _swing_confirm(clean_df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    for bar in (12, 16, 20, 24):
        clean_df.loc[bar, ["low", "close", "high"]] = [95.05, 95.4, 100.0]
    clean_registry = build_sr_level_registry(clean_df)
    project_sr_context(clean_df, clean_registry)
    clean_zone = _emitted_supports(clean_registry)[0]

    weak_df = _base(60)
    _swing_confirm(weak_df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    for bar in (12, 16, 20, 24):
        weak_df.loc[bar, ["low", "close", "high"]] = [94.2, 95.15, 100.0]
    weak_registry = build_sr_level_registry(weak_df)
    project_sr_context(weak_df, weak_registry)
    weak_zone = _emitted_supports(weak_registry)[0]

    assert clean_zone.clean_touch_count > 0
    assert weak_zone.weak_touch_count > 0
    assert weak_zone.reaction_quality_score < clean_zone.reaction_quality_score


def test_primary_zone_falls_back_to_nearest_when_score_edge_is_small() -> None:
    df = _base(60)
    df["prev_day_low"] = np.nan
    df.loc[10:, "prev_day_low"] = 99.0
    _swing_confirm(df, bar=12, price=97.6, side=SR_SIDE_SUPPORT, origin_bar=6)

    out = add_sr_levels(df, include_research_only=False)
    probe = out.iloc[30]
    assert probe["nearest_support_price"] == pytest.approx(99.0)
    assert probe["primary_support_zone_mid"] == pytest.approx(
        probe["nearest_support_price"]
    )


def test_live_research_columns_remain_backward_compatible() -> None:
    df = _base(40)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    df["prev_day_high"] = np.nan
    df.loc[10:, "prev_day_high"] = 108.0

    live_df = add_sr_levels(df, include_research_only=False)
    research_df = add_sr_levels(df, include_research_only=True)

    legacy_cols = [
        "nearest_support_price",
        "nearest_support_distance_atr",
        "nearest_support_strength",
        "nearest_resistance_price",
        "nearest_resistance_distance_atr",
        "nearest_resistance_strength",
        "active_support_count",
        "active_resistance_count",
    ]
    for col in legacy_cols:
        assert col in live_df.columns
        assert col in research_df.columns
    live_cols = [col for col in research_df.columns if not col.startswith("r_")]
    pd.testing.assert_frame_equal(
        live_df[live_cols],
        research_df[live_cols],
        check_dtype=False,
    )
    assert "r_sr_touch_event_id" in research_df.columns
    assert "r_sr_score_quintile_calibration_json" in research_df.columns


def test_input_dataframe_not_mutated() -> None:
    df = _base(20)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    original = df.copy(deep=True)
    _ = add_sr_levels(df, include_research_only=False)
    pd.testing.assert_frame_equal(df, original)


def test_retires_very_old_zone() -> None:
    df = _base(MAX_AGE_BARS + 25)
    _swing_confirm(df, bar=0, price=90.0, side=SR_SIDE_SUPPORT, origin_bar=0)
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    zone = _emitted_supports(registry)[0]
    assert zone.state == SR_STATE_RETIRED


def test_family_prior_still_available_for_day_levels() -> None:
    assert FAMILY_PRIOR[SR_FAMILY_DAY] > FAMILY_PRIOR[SR_FAMILY_SWING]


def test_hard_expiry_sets_terminal_reason_and_removes_zone_from_ladder() -> None:
    df = _base(FAMILY_MAX_AGE[SR_FAMILY_DAY] + 20)
    df["prev_day_low"] = 99.0

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)
    zone = _emitted_supports(registry)[0]

    assert zone.expiry_idx == FAMILY_MAX_AGE[SR_FAMILY_DAY]
    assert zone.terminal_reason == "expired_hard"
    assert pd.notna(out.loc[zone.expiry_idx - 1, "sr_support_l1_id"])
    assert pd.isna(out.loc[zone.expiry_idx, "sr_support_l1_id"])


def test_ladder_exports_visible_deeper_support_backups_in_distance_order() -> None:
    df = _base(80)
    df["prev_day_low"] = np.nan
    df.loc[0:, "prev_day_low"] = 99.0
    _swing_confirm(df, bar=5, price=97.5, side=SR_SIDE_SUPPORT, origin_bar=2)
    df["eql_detect_flag"] = 0
    df["eql_level_on_detect"] = np.nan
    df["eql_origin_idx"] = np.nan
    df["eql_score_on_detect"] = np.nan
    df["eql_member_count_on_detect"] = np.nan
    df.at[8, "eql_detect_flag"] = 1
    df.at[8, "eql_level_on_detect"] = 96.0
    df.at[8, "eql_origin_idx"] = 6
    df.at[8, "eql_score_on_detect"] = 0.9
    df.at[8, "eql_member_count_on_detect"] = 3

    out = add_sr_levels(df, include_research_only=False)
    probe = out.iloc[20]

    assert probe["sr_support_l1_id"] == probe["nearest_support_zone_id"]
    assert (
        probe["sr_support_l1_mid"]
        > probe["sr_support_l2_mid"]
        > probe["sr_support_l3_mid"]
    )
    assert probe["sr_support_l1_score"] >= 0.0
    assert probe["sr_support_l3_expiry_bars_remaining"] >= 0.0


def test_invalidated_zone_is_visible_before_break_and_absent_after_terminal_bar() -> (
    None
):
    df = _base(50)
    _swing_confirm(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    df.loc[10, ["close", "low", "high"]] = [94.1, 94.0, 95.2]
    df.loc[11, ["close", "low", "high"]] = [94.0, 93.9, 95.1]

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)
    zone = _emitted_supports(registry)[0]

    assert zone.invalidation_idx == 11
    assert zone.terminal_reason == "invalidated"
    assert pd.notna(out.loc[9, "nearest_support_zone_id"])
    assert pd.isna(out.loc[10, "nearest_support_zone_id"])
    assert pd.isna(out.loc[11, "nearest_support_zone_id"])
    assert pd.isna(out.loc[20, "sr_support_l1_id"])


def test_summary_splits_structure_pass_from_score_fail() -> None:
    df = _base(60)
    df["prev_day_low"] = 95.0
    df["prev_day_high"] = 105.0

    live_df = add_sr_levels(df, include_research_only=False)
    research_df = add_sr_levels(df, include_research_only=True)
    registry = build_sr_level_registry(df)
    summary = summarize_sr_levels(research_df, registry, live_df=live_df)

    assert summary["structure_status"]["label"] == "pass"
    assert summary["score_status"]["label"] == "fail"
    assert summary["score_status"]["flat"] is True


def test_validator_chart_renders_per_zone_rectangles(monkeypatch) -> None:
    df = _base(60)
    df["prev_day_low"] = 95.0
    df["prev_day_high"] = 105.0
    # Add a swing high (high-info family) so the chart has at least one zone
    # passing the structural-quality filter applied in _select_visible_zones.
    _swing_confirm(df, bar=10, price=104.5, side=SR_SIDE_RESISTANCE, origin_bar=8)
    _swing_confirm(df, bar=12, price=95.5, side=SR_SIDE_SUPPORT, origin_bar=10)

    # The structural-score floor in the chart filter is a tunable knob;
    # synthetic single-anchor swings here can't reach prod thresholds.
    # Loosen the floor for this test so it tests the rendering contract,
    # not a specific tuning value.
    monkeypatch.setattr(vsr, "_STRUCTURAL_SCORE_MIN", 0.30)

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)  # mutates registry via update_sr_lifecycle
    fig = vsr._build_sr_chart(
        out,
        out,
        registry,
        title="test",
        date_from=str(out["timestamp"].iloc[0].date()),
    )

    rect_shapes = [shape for shape in (fig.layout.shapes or ()) if shape.type == "rect"]
    assert rect_shapes, "expected at least one zone rectangle on the price subplot"
    trace_names = {trace.name for trace in fig.data}
    assert any(
        name in trace_names for name in ("Support zone", "Resistance zone")
    ), f"expected zone-anchor markers in legend; got {trace_names}"
    assert all(
        "Support L1" not in name for name in trace_names
    ), "ladder bands should be removed in the rebuilt chart"
