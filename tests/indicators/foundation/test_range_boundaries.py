from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.range_boundaries import (
    ALL_RANGE_BOUNDARY_COLUMNS,
    RANGE_STATE_ACCEPTED_BREAKOUT,
    RANGE_STATE_ACTIVE_INTACT,
    RANGE_STATE_ACTIVE_WEAKENED,
    RANGE_STATE_BROKEN_UNACCEPTED,
    RANGE_STATE_EXPIRED,
    RANGE_STATE_SUPERSEDED,
    add_range_boundaries,
    collect_range_boundary_debug_tables,
)
from src.indicators.foundation.volatility import add_atr


def _make_ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h", tz="UTC")
    return df[["timestamp", "open", "high", "low", "close"]]


def _run_range_boundaries(
    rows: list[tuple[float, float, float, float]],
    **kwargs: object,
) -> pd.DataFrame:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    params: dict[str, object] = {
        "atr_length": 3,
        "range_lookback_bars": 5,
        "min_confirm_bars": 2,
        "min_candidate_dwell_bars": 2,
        "viability_lookback_bars": 3,
        "max_width_atr": 2.0,
        "min_close_inside_frac": 0.5,
    }
    params.update(kwargs)
    params.setdefault("candidate_lookback_bars", (int(params["range_lookback_bars"]),))
    return add_range_boundaries(df, **params)


def _run_range_debug(
    rows: list[tuple[float, float, float, float]],
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    df = _make_ohlc(rows)
    df = add_atr(df, period=3)
    params: dict[str, object] = {
        "atr_length": 3,
        "range_lookback_bars": 5,
        "min_confirm_bars": 2,
        "min_candidate_dwell_bars": 2,
        "viability_lookback_bars": 3,
        "max_width_atr": 2.0,
        "min_close_inside_frac": 0.5,
    }
    params.update(kwargs)
    params.setdefault("candidate_lookback_bars", (int(params["range_lookback_bars"]),))
    return collect_range_boundary_debug_tables(df, **params)


def _pressure_imbalance_v2_from_frame(
    frame: pd.DataFrame,
    *,
    confirm_idx: int,
    low: float,
    high: float,
    lookback_bars: int,
) -> float:
    width = max(high - low, 1e-9)
    start = max(0, confirm_idx - lookback_bars + 1)
    window = frame.iloc[start : confirm_idx + 1]
    close_pos = ((window["close"] - low) / width).clip(0.0, 1.0)
    close_span = float(close_pos.max() - close_pos.min())
    mean_edge_bias = abs(float(close_pos.mean()) - 0.5) * 2.0
    last_edge_bias = abs(float(close_pos.iloc[-1]) - 0.5) * 2.0
    return float(
        min(
            max(
                (1.0 - close_span) * (0.5 + 0.5 * max(mean_edge_bias, last_edge_bias)),
                0.0,
            ),
            1.0,
        )
    )


def test_range_boundary_schema_exists() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
    ]
    result = _run_range_boundaries(rows)

    for col in ALL_RANGE_BOUNDARY_COLUMNS:
        assert col in result.columns


def test_confirm_bar_activates_range_without_prior_activation_or_same_bar_break_logic() -> (
    None
):
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
    ]
    result = _run_range_boundaries(rows)

    assert (result.loc[:4, "range_active"] == 0).all()
    assert result.loc[5, "range_detect_flag"] == 1
    assert result.loc[5, "range_active"] == 1
    assert result.loc[5, "range_state"] == RANGE_STATE_ACTIVE_INTACT
    assert result.loc[5, "range_confirm_idx"] == 5.0
    assert result.loc[5, "range_breakout_pending_flag"] == 0
    assert result.loc[5, "range_upper_active"] == 1
    assert result.loc[5, "range_lower_active"] == 1
    assert result.loc[5, "range_upper_source_idx"] == 5.0
    assert result.loc[5, "range_upper_age_bars"] == 0.0
    assert result.loc[5, "range_viability_gate_pass"] == 1
    assert 0.0 <= result.loc[5, "range_strength_formation"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_viability"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_structure"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_monitorability"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_semantic"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_legacy"] <= 1.0
    assert 0.0 <= result.loc[5, "range_strength_viability_legacy"] <= 1.0


def test_false_break_reclaim_returns_to_weakened_active_state() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 101.05, 99.43, 100.09),
        (100.09, 100.90, 99.50, 100.00),
        (100.07, 100.88, 99.55, 100.02),
    ]
    result = _run_range_boundaries(rows)

    assert result.loc[6, "range_state"] == RANGE_STATE_BROKEN_UNACCEPTED
    assert result.loc[6, "range_breakout_pending_flag"] == 1
    assert result.loc[7, "range_state"] == RANGE_STATE_ACTIVE_WEAKENED
    assert result.loc[7, "range_weakened_flag"] == 1
    assert result.loc[7, "range_active"] == 1
    assert np.isnan(result.loc[5, "range_end_idx"])


def test_close_based_breakout_acceptance_terminates_range_on_terminal_row() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
        (100.09, 100.93, 99.41, 100.07),
        (100.07, 100.95, 99.39, 100.11),
        (100.10, 101.40, 100.8, 101.25),
        (101.2, 101.7, 101.0, 101.55),
    ]
    result = _run_range_boundaries(rows)

    assert result.loc[9, "range_state"] == RANGE_STATE_ACCEPTED_BREAKOUT
    assert result.loc[9, "range_active"] == 0
    assert result.loc[9, "range_accepted_breakout_flag"] == 1
    assert result.loc[5, "range_end_idx"] == 9.0


def test_drifting_directional_trend_does_not_qualify_as_range() -> None:
    rows = [
        (100.0, 100.9, 99.8, 100.8),
        (100.8, 101.6, 100.5, 101.4),
        (101.4, 102.1, 101.0, 101.9),
        (101.9, 102.7, 101.5, 102.5),
        (102.5, 103.2, 102.1, 103.0),
        (103.0, 103.8, 102.7, 103.6),
        (103.6, 104.4, 103.2, 104.1),
        (104.1, 104.9, 103.8, 104.7),
    ]
    result = _run_range_boundaries(rows)

    assert result["range_detect_flag"].sum() == 0
    assert result["range_active"].sum() == 0


def test_one_sided_compression_does_not_qualify_as_range() -> None:
    rows = [
        (100.0, 101.0, 99.9, 100.8),
        (100.8, 101.0, 100.1, 100.9),
        (100.9, 101.0, 100.2, 100.95),
        (100.95, 101.0, 100.3, 100.97),
        (100.97, 101.0, 100.4, 100.99),
        (100.99, 101.0, 100.5, 100.98),
        (100.98, 101.0, 100.6, 100.99),
        (100.99, 101.0, 100.7, 101.0),
    ]
    result = _run_range_boundaries(rows)

    assert result["range_detect_flag"].sum() == 0


def test_expiry_deactivates_stale_range() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
        (100.09, 100.93, 99.41, 100.07),
        (100.07, 100.95, 99.39, 100.11),
    ] + [(100.1, 100.8, 99.6, 100.1)] * 5
    result = _run_range_boundaries(rows, max_active_age_bars=3)

    assert result.loc[8, "range_state"] == RANGE_STATE_EXPIRED
    assert result.loc[8, "range_expired_flag"] == 1
    assert result.loc[8, "range_active"] == 0
    assert result.loc[5, "range_end_idx"] == 8.0


def test_newer_tighter_overlapping_range_supersedes_older_event_in_debug_table() -> (
    None
):
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.20, 101.10, 99.60, 100.50),
        (100.45, 100.75, 100.05, 100.40),
        (100.38, 100.72, 100.08, 100.42),
        (100.40, 100.70, 100.10, 100.41),
        (100.41, 100.69, 100.12, 100.43),
        (100.43, 100.71, 100.11, 100.44),
        (100.44, 100.70, 100.12, 100.43),
    ]
    debug = _run_range_debug(rows)
    result = debug["frame"]
    events = debug["event_table"].sort_values("range_id").reset_index(drop=True)

    assert result["range_detect_flag"].sum() == 2
    assert events.loc[0, "state"] == RANGE_STATE_SUPERSEDED
    assert events.loc[0, "end_idx"] == 12.0
    assert events.loc[1, "state"] == RANGE_STATE_ACTIVE_INTACT
    assert result.loc[12, "range_id"] == 2
    assert result.loc[12, "range_width_atr"] < events.loc[0, "width_atr"]


def test_source_metadata_is_causal_and_monotone_while_range_is_active() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
        (100.09, 100.93, 99.41, 100.07),
        (100.07, 100.95, 99.39, 100.11),
    ]
    result = _run_range_boundaries(rows)
    active = result[result["range_active"] == 1]

    assert not active.empty
    assert (active["range_upper_source_idx"] == active["range_confirm_idx"]).all()
    assert active["range_upper_age_bars"].is_monotonic_increasing
    assert active["range_lower_age_bars"].is_monotonic_increasing
    assert active["range_upper_source_timestamp"].notna().all()
    assert active["range_lower_source_timestamp"].notna().all()


def test_soft_viability_gating_allows_confirmation_when_pressure_threshold_fails_but_score_passes() -> (
    None
):
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.0),
        (100.0, 100.92, 99.42, 100.0),
        (100.0, 100.91, 99.43, 99.7),
        (99.7, 100.90, 99.44, 99.7),
    ]
    debug = _run_range_debug(rows)
    candidates = debug["candidate_table"]

    assert debug["frame"]["range_detect_flag"].sum() == 1
    assert int(candidates.iloc[-1]["confirmed_flag"]) == 1
    assert int(candidates.iloc[-1]["viability_fail_due_to_pressure"]) == 1
    assert int(candidates.iloc[-1]["viability_fail_due_to_score_threshold"]) == 0
    assert int(candidates.iloc[-1]["range_viability_gate_pass"]) == 1


def test_expansion_veto_remains_a_hard_confirmation_reject() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.1),
        (100.2, 101.0, 100.1, 101.0),
        (101.0, 101.05, 100.95, 101.02),
    ]
    debug = _run_range_debug(rows)
    candidates = debug["candidate_table"]

    assert debug["frame"]["range_detect_flag"].sum() == 0
    assert not candidates.empty
    assert int(candidates.iloc[-1]["expansion_veto_seen_flag"]) == 1
    assert int(candidates.iloc[-1]["range_viability_gate_pass"]) == 0
    assert int(candidates.iloc[-1]["confirmed_flag"]) == 0


def test_lineage_grace_preserves_same_lineage_when_geometry_recovers_cleanly() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.85, 99.0, 100.9),
        (100.9, 100.9, 100.0, 99.4),
        (99.4, 101.2, 99.0, 100.1),
        (100.1, 100.9, 99.4, 99.2),
        (99.2, 100.9, 99.45, 100.9),
    ]
    debug_no_grace = _run_range_debug(
        rows,
        range_lookback_bars=4,
        lineage_grace_bars=0,
        min_confirm_bars=3,
    )
    debug_with_grace = _run_range_debug(
        rows,
        range_lookback_bars=4,
        lineage_grace_bars=1,
        min_confirm_bars=3,
    )

    candidates_no_grace = debug_no_grace["candidate_table"]
    candidates_with_grace = debug_with_grace["candidate_table"]

    assert len(candidates_no_grace) == 2
    assert len(candidates_with_grace) == 1
    assert int(candidates_with_grace.iloc[0]["failed_same_lineage_continuation"]) == 0


def test_lineage_grace_resets_when_recovery_fails_same_lineage_constraints() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 101.5, 99.2, 98.8),
        (98.8, 101.8, 100.6, 101.3),
        (101.3, 101.6982, 100.70, 101.3),
        (101.3, 101.5964, 100.72, 101.3),
    ]
    debug = _run_range_debug(
        rows,
        range_lookback_bars=3,
        lineage_grace_bars=1,
    )
    candidates = debug["candidate_table"].reset_index(drop=True)

    assert len(candidates) >= 2
    assert int(candidates.loc[0, "failed_same_lineage_continuation"]) == 1
    assert int(candidates.loc[0, "failed_candidate_eligibility_before_maturity"]) == 1


def test_multi_window_raw_formation_confirms_on_short_lookback_but_not_long_lookback() -> (
    None
):
    rows = [
        (100.0, 100.7, 99.5, 100.1),
        (100.1, 100.8, 99.4, 100.0),
        (100.0, 100.75, 99.45, 100.15),
        (100.15, 100.78, 99.42, 100.05),
        (100.05, 100.76, 99.44, 100.10),
        (100.10, 100.77, 99.43, 100.08),
        (100.08, 100.79, 99.41, 100.09),
        (100.09, 100.81, 99.40, 100.07),
        (100.07, 101.6, 100.6, 101.4),
        (101.4, 101.8, 101.1, 101.6),
        (101.6, 101.9, 101.3, 101.7),
        (101.7, 102.0, 101.5, 101.8),
    ]
    debug_short = _run_range_debug(rows, candidate_lookback_bars=(5,))
    debug_long = _run_range_debug(rows, candidate_lookback_bars=(12,))
    debug_multi = _run_range_debug(rows, candidate_lookback_bars=(5, 12))

    assert len(debug_short["event_table"]) == 1
    assert int(debug_short["event_table"].iloc[0]["candidate_lookback_bars"]) == 5
    assert debug_long["event_table"].empty
    assert len(debug_multi["event_table"]) == 1
    assert int(debug_multi["event_table"].iloc[0]["candidate_lookback_bars"]) == 5


def test_duplicate_suppression_collapses_overlapping_confirms_from_two_lookbacks() -> (
    None
):
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
        (100.09, 100.93, 99.41, 100.07),
        (100.07, 100.95, 99.39, 100.11),
        (100.11, 100.94, 99.40, 100.10),
        (100.10, 100.96, 99.38, 100.09),
        (100.09, 100.97, 99.37, 100.08),
    ]
    debug_5 = _run_range_debug(rows, candidate_lookback_bars=(5,))
    debug_8 = _run_range_debug(rows, candidate_lookback_bars=(8,))
    debug_multi = _run_range_debug(rows, candidate_lookback_bars=(5, 8))

    assert len(debug_5["event_table"]) == 1
    assert len(debug_8["event_table"]) == 1
    assert len(debug_multi["event_table"]) == 1
    assert int(debug_multi["event_table"].iloc[0]["candidate_lookback_bars"]) == 8
    assert int(debug_multi["frame"]["range_detect_flag"].sum()) == 1


def test_nested_ranges_from_different_lookbacks_are_preserved() -> None:
    rows = [
        (100.0, 100.8, 99.2, 100.0),
        (100.0, 100.85, 99.15, 100.05),
        (100.05, 100.82, 99.18, 99.98),
        (99.98, 100.83, 99.17, 100.02),
        (100.02, 100.81, 99.19, 100.0),
        (100.0, 100.82, 99.18, 100.01),
        (100.01, 100.45, 99.55, 100.00),
        (100.00, 100.44, 99.56, 100.01),
        (100.01, 100.43, 99.57, 99.99),
        (99.99, 100.44, 99.56, 100.00),
        (100.00, 100.43, 99.57, 100.01),
        (100.01, 100.44, 99.56, 100.00),
    ]
    debug = _run_range_debug(rows, candidate_lookback_bars=(5, 8, 12))
    events = debug["event_table"].sort_values("confirm_idx").reset_index(drop=True)

    assert len(events) == 2
    assert list(events["candidate_lookback_bars"]) == [8, 5]
    assert float(events.loc[1, "width_atr"]) < float(events.loc[0, "width_atr"])


def test_production_pressure_metric_matches_promoted_v2_formula() -> None:
    rows = [
        (100.0, 100.8, 99.4, 100.1),
        (100.1, 100.9, 99.3, 100.0),
        (100.0, 100.85, 99.35, 100.15),
        (100.15, 100.95, 99.45, 100.05),
        (100.05, 100.90, 99.40, 100.10),
        (100.10, 100.92, 99.42, 100.08),
        (100.08, 100.91, 99.43, 100.09),
    ]
    debug = _run_range_debug(rows, candidate_lookback_bars=(5,))
    event = debug["event_table"].iloc[0]
    expected = _pressure_imbalance_v2_from_frame(
        debug["frame"],
        confirm_idx=int(event["confirm_idx"]),
        low=float(event["low"]),
        high=float(event["high"]),
        lookback_bars=3,
    )

    assert np.isclose(
        float(event["range_recent_pressure_imbalance"]),
        expected,
        atol=1e-9,
    )
