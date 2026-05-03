from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_PRODUCTION_COLUMNS,
    FINAL_SWEEPS_RESEARCH_COLUMNS,
    add_final_sweeps,
)
from src.indicators.smc.sweeps.unified_sources import LIQ_LADDER_DEPTH
from src.validation.indicators.final_sweeps import build_final_sweeps_diagnostics


def _frame_with_one_source(
    rows: list[tuple[float, float, float, float]],
    *,
    side: str,
    source_level: float,
    source_strength: float = 0.8,
    source_family: str = "resistance",
    source_age_start: int = 5,
) -> pd.DataFrame:
    n = len(rows)
    cols: dict[str, object] = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "atr_14": np.full(n, 1.0, dtype=float),
    }

    prefix = f"liq_{side}_l1"
    cols[f"{prefix}_cluster_id"] = np.full(n, 101.0, dtype=float)
    cols[f"{prefix}_level"] = np.full(n, source_level, dtype=float)
    cols[f"{prefix}_zone_low"] = np.full(n, source_level, dtype=float)
    cols[f"{prefix}_zone_high"] = np.full(n, source_level, dtype=float)
    cols[f"{prefix}_is_zone"] = np.zeros(n, dtype=float)
    cols[f"{prefix}_width_abs"] = np.zeros(n, dtype=float)
    cols[f"{prefix}_width_atr"] = np.zeros(n, dtype=float)
    cols[f"{prefix}_strength"] = np.full(n, source_strength, dtype=float)
    cols[f"{prefix}_state"] = np.full(n, 2.0, dtype=float)
    cols[f"{prefix}_age_bars"] = np.arange(
        source_age_start, source_age_start + n, dtype=float
    )
    cols[f"{prefix}_freshness"] = np.full(n, 0.5, dtype=float)
    cols[f"{prefix}_touch_count"] = np.zeros(n, dtype=float)
    cols[f"{prefix}_signed_dist_atr"] = np.full(n, 1.0, dtype=float)
    cols[f"{prefix}_member_count"] = np.ones(n, dtype=float)
    cols[f"{prefix}_origin_idx"] = np.full(n, -float(source_age_start), dtype=float)
    cols[f"{prefix}_active_start_idx"] = np.full(
        n, -float(source_age_start), dtype=float
    )
    cols[f"{prefix}_primary_family"] = np.array([source_family] * n, dtype=object)
    cols[f"{prefix}_attribution_families"] = np.array([source_family] * n, dtype=object)

    for other_side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            if other_side == side and rank == 1:
                continue
            other_prefix = f"liq_{other_side}_l{rank}"
            for field in (
                "cluster_id",
                "level",
                "zone_low",
                "zone_high",
                "is_zone",
                "width_abs",
                "width_atr",
                "strength",
                "state",
                "age_bars",
                "freshness",
                "touch_count",
                "signed_dist_atr",
                "member_count",
                "origin_idx",
                "active_start_idx",
            ):
                cols[f"{other_prefix}_{field}"] = np.full(n, np.nan, dtype=float)
            cols[f"{other_prefix}_primary_family"] = np.array([""] * n, dtype=object)
            cols[f"{other_prefix}_attribution_families"] = np.array(
                [""] * n, dtype=object
            )

    cols["liq_active_total_count"] = np.full(n, 1.0, dtype=float)
    cols["liq_active_above_count"] = np.full(n, 1.0 if side == "above" else 0.0)
    cols["liq_active_below_count"] = np.full(n, 1.0 if side == "below" else 0.0)
    cols["liq_dropped_by_crowding_count"] = np.zeros(n, dtype=float)
    cols["liq_dropped_by_dominance_count"] = np.zeros(n, dtype=float)
    cols["liq_nearest_above_dist_atr"] = np.full(
        n, 1.0 if side == "above" else np.nan, dtype=float
    )
    cols["liq_nearest_below_dist_atr"] = np.full(
        n, 1.0 if side == "below" else np.nan, dtype=float
    )
    cols["liq_top_above_strength"] = np.full(
        n, source_strength if side == "above" else np.nan, dtype=float
    )
    cols["liq_top_below_strength"] = np.full(
        n, source_strength if side == "below" else np.nan, dtype=float
    )
    cols["liq_source_timeframe"] = np.array(["H4"] * n, dtype=object)
    cols["liq_mtf_policy"] = np.array(["same_timeframe_only"] * n, dtype=object)
    return pd.DataFrame(cols)


def test_forward_outcomes_respect_direction_symmetry() -> None:
    above = _frame_with_one_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (101.2, 102.0, 99.8, 100.5),
            (100.3, 100.4, 99.0, 99.3),
        ],
        side="above",
        source_level=101.0,
        source_family="resistance",
    )
    below = _frame_with_one_source(
        [
            (100.0, 100.5, 99.5, 100.2),
            (99.8, 100.4, 98.2, 99.6),
            (99.7, 101.0, 99.6, 100.8),
        ],
        side="below",
        source_level=99.0,
        source_family="support",
    )
    above_out = add_final_sweeps(above)
    below_out = add_final_sweeps(below)
    assert above_out["sweep_fwd_close_ret_atr_1"].iloc[1] > 0.0
    assert below_out["sweep_fwd_close_ret_atr_1"].iloc[1] > 0.0


def test_first_hit_favorable_first() -> None:
    rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.4, 100.6, 99.2, 99.3),
    ] + [(99.3, 99.6, 99.0, 99.2)] * 20
    out = add_final_sweeps(
        _frame_with_one_source(rows, side="above", source_level=101.0)
    )
    assert out["sweep_first_favorable_1p0_bar"].iloc[1] == 1.0
    assert np.isnan(out["sweep_first_adverse_1p0_bar"].iloc[1])
    assert out["sweep_fwd_path_label_1"].iloc[1] == "clean_reversal"
    assert out["sweep_reversal_speed_bucket"].iloc[1] == "immediate"


def test_first_hit_adverse_first() -> None:
    rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.6, 102.2, 100.4, 101.7),
    ] + [(101.7, 101.9, 101.2, 101.5)] * 20
    out = add_final_sweeps(
        _frame_with_one_source(rows, side="above", source_level=101.0)
    )
    assert out["sweep_first_adverse_1p0_bar"].iloc[1] == 1.0
    assert np.isnan(out["sweep_first_favorable_1p0_bar"].iloc[1])
    assert out["sweep_fwd_path_label_1"].iloc[1] == "continuation"
    assert out["sweep_continuation_speed_bucket"].iloc[1] == "immediate"


def test_first_hit_both_same_bar_and_neither_and_unavailable() -> None:
    both_rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.5, 102.0, 99.0, 100.4),
    ] + [(100.4, 100.6, 100.0, 100.3)] * 20
    both_out = add_final_sweeps(
        _frame_with_one_source(both_rows, side="above", source_level=101.0)
    )
    assert both_out["sweep_first_favorable_1p0_bar"].iloc[1] == 1.0
    assert both_out["sweep_first_adverse_1p0_bar"].iloc[1] == 1.0
    assert both_out["sweep_fwd_path_label_1"].iloc[1] == "two_sided_volatile"

    neither_rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.4, 100.8, 100.1, 100.4),
    ] + [(100.3, 100.7, 100.1, 100.4)] * 20
    neither_out = add_final_sweeps(
        _frame_with_one_source(neither_rows, side="above", source_level=101.0)
    )
    assert neither_out["sweep_fwd_path_label_1"].iloc[1] == "chop_no_resolution"

    unavailable_rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.4, 100.6, 99.2, 99.3),
    ]
    unavailable_out = add_final_sweeps(
        _frame_with_one_source(unavailable_rows, side="above", source_level=101.0)
    )
    assert np.isnan(unavailable_out["sweep_first_favorable_1p0_bar"].iloc[1])
    assert unavailable_out["sweep_fwd_path_label_1"].iloc[1] == "clean_reversal"
    assert pd.isna(unavailable_out["sweep_fwd_path_label_2"].iloc[1])


def test_forward_columns_are_research_only_schema() -> None:
    assert not any(
        col.startswith(("sweep_fwd_", "sweep_first_"))
        for col in FINAL_SWEEPS_PRODUCTION_COLUMNS
    )
    assert any(
        col.startswith(("sweep_fwd_", "sweep_first_"))
        for col in FINAL_SWEEPS_RESEARCH_COLUMNS
    )


def test_step11h_diagnostics_tables_exist() -> None:
    rows = [
        (99.0, 100.0, 98.5, 99.5),
        (101.2, 102.0, 99.8, 100.5),
        (100.4, 100.6, 99.2, 99.3),
    ] + [(99.3, 99.6, 99.0, 99.2)] * 35
    df = _frame_with_one_source(rows, side="above", source_level=101.0)
    df["regime"] = np.full(len(df), 1.0, dtype=float)
    df["regime_label"] = np.array(["TRANSITIONAL"] * len(df), dtype=object)
    df["session_name"] = np.array(["london"] * len(df), dtype=object)
    out = add_final_sweeps(df)
    diagnostics = build_final_sweeps_diagnostics(out)
    assert diagnostics["post_path_summary"]["total_confirmed_sweeps"] == 1
    assert not diagnostics["outcome_by_class_table"].empty
    assert not diagnostics["outcome_by_family_table"].empty
    assert not diagnostics["outcome_by_side_table"].empty
    assert not diagnostics["outcome_by_regime_table"].empty
    assert not diagnostics["outcome_by_session_phase_table"].empty
    assert not diagnostics["outcome_by_quality_bucket_table"].empty
    assert not diagnostics["outcome_by_source_age_bucket_table"].empty
    assert not diagnostics["outcome_by_distance_bucket_table"].empty
