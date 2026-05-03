"""Causality + acceptance-gate tests for the canonical sweeps validator.

These tests exercise
:mod:`src.validation.indicators.canonical_sweeps` against synthetic
frames that come straight from the unit-test fixtures used by the
detector itself, so they pin both the report shape and the gate logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
    add_final_sweeps,
    step11e_default_kwargs,
)
from src.indicators.smc.sweeps.unified_sources import LIQ_LADDER_DEPTH
from src.validation.indicators.canonical_sweeps import (
    ACCEPTED_SOURCE_FAMILIES,
    build_canonical_sweeps_report,
    report_passed,
)


def _ladder_frame(rows, *, source_level: float, source_family: str = "resistance"):
    n = len(rows)
    cols: dict[str, object] = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "atr_14": np.full(n, 1.0, dtype=float),
    }
    for side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            for fname in (
                "cluster_id",
                "level",
                "zone_low",
                "zone_high",
                "strength",
                "age_bars",
                "origin_idx",
                "active_start_idx",
                "primary_family",
                "attribution_families",
            ):
                col = f"liq_{side}_l{rank}_{fname}"
                if fname.endswith("family") or fname.endswith("families"):
                    cols[col] = [""] * n
                else:
                    cols[col] = [np.nan] * n
    df = pd.DataFrame(cols)
    df["liq_above_l1_cluster_id"] = 1.0
    df["liq_above_l1_level"] = source_level
    df["liq_above_l1_zone_low"] = source_level
    df["liq_above_l1_zone_high"] = source_level
    df["liq_above_l1_strength"] = 0.6
    df["liq_above_l1_origin_idx"] = 0.0
    df["liq_above_l1_active_start_idx"] = 0.0
    df["liq_above_l1_age_bars"] = [max(5 + i, 0) for i in range(n)]
    df["liq_above_l1_primary_family"] = source_family
    df["liq_above_l1_attribution_families"] = source_family
    return df


def _bearish_sweep_frame() -> pd.DataFrame:
    rows = [
        (95.0, 96.0, 94.0, 95.5),
        (95.5, 96.5, 95.0, 96.0),
        (96.0, 97.0, 95.5, 96.5),
        (96.5, 100.5, 96.0, 99.5),
        (99.5, 99.8, 99.0, 99.3),
        (99.3, 99.5, 98.0, 98.5),
        (98.5, 99.0, 97.5, 98.0),
    ]
    return _ladder_frame(rows, source_level=100.0)


def _run(df: pd.DataFrame) -> pd.DataFrame:
    return add_final_sweeps(df, **step11e_default_kwargs(cooldown_bars=10))


def test_report_passes_all_acceptance_gates_on_clean_fixture():
    out = _run(_bearish_sweep_frame())
    report = build_canonical_sweeps_report(out)
    assert report["total_sweep_count"] >= 1
    assert report_passed(report) is True


def test_alias_columns_must_be_present_or_gate_fails():
    out = _run(_bearish_sweep_frame()).copy()
    # Drop a required alias column → schema invariant fails.
    out = out.drop(columns=["sweep_direction"])
    # The report itself raises now because we marked direction as required.
    raised = False
    try:
        build_canonical_sweeps_report(out)
    except ValueError:
        raised = True
    assert raised


def test_legacy_columns_present_fails_acceptance_gate():
    out = _run(_bearish_sweep_frame()).copy()
    # Inject the legacy detector's flag columns; gate must reject.
    out["sweep_high"] = 0
    out["sweep_low"] = 0
    report = build_canonical_sweeps_report(out)
    assert report["schema_invariants"]["no_legacy_columns"] is False
    assert report_passed(report) is False


def test_future_columns_required_must_be_empty():
    out = _run(_bearish_sweep_frame())
    report = build_canonical_sweeps_report(out)
    assert report["future_columns_required"] == []


def test_causality_counters_are_zero_on_clean_fixture():
    out = _run(_bearish_sweep_frame())
    report = build_canonical_sweeps_report(out)
    causality = report["causality_violations"]
    for key in (
        "source_after_breach",
        "breach_after_confirm",
        "swept_source_idx_negative",
        "swept_source_idx_out_of_range",
        "ambiguous_direction_label",
        "swept_source_family_unknown",
        "missing_source_metadata",
        "bullish_with_above_swept_side",
        "bearish_with_below_swept_side",
        "duplicate_canonical_event",
    ):
        assert causality[key] == 0, key


def test_injected_post_breach_source_idx_is_caught():
    out = _run(_bearish_sweep_frame()).copy()
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    assert not sweep_rows.empty
    # Force a violation: rewrite swept_source_idx to a value AFTER the
    # breach bar. The detector's sanitization fired at compute time, so
    # we have to rewrite both the alias AND the diagnostic flag here to
    # simulate a regression.
    target = sweep_rows.index[0]
    breach = int(out.at[target, "sweep_breach_idx"])
    out.at[target, "swept_source_idx"] = float(breach + 1)
    report = build_canonical_sweeps_report(out)
    assert report["causality_violations"]["source_after_breach"] >= 1
    assert report_passed(report) is False


def test_invalid_direction_label_is_caught():
    out = _run(_bearish_sweep_frame()).copy()
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    target = sweep_rows.index[0]
    out.at[target, "sweep_direction"] = "sideways"
    report = build_canonical_sweeps_report(out)
    assert report["causality_violations"]["ambiguous_direction_label"] >= 1
    assert report_passed(report) is False


def test_unknown_source_family_is_flagged():
    out = _run(_bearish_sweep_frame()).copy()
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    target = sweep_rows.index[0]
    out.at[target, "swept_source_family"] = "not_a_real_family"
    report = build_canonical_sweeps_report(out)
    assert report["causality_violations"]["swept_source_family_unknown"] >= 1


def test_bull_flag_with_above_side_is_caught():
    out = _run(_bearish_sweep_frame()).copy()
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    target = sweep_rows.index[0]
    # Lie: claim a bullish sweep but keep swept_source_side = +1.
    out.at[target, "bullish_sweep_flag"] = 1.0
    out.at[target, "bearish_sweep_flag"] = 0.0
    report = build_canonical_sweeps_report(out)
    assert report["causality_violations"]["bullish_with_above_swept_side"] >= 1
    assert report_passed(report) is False


def test_accepted_source_families_match_canonical_list():
    expected = {
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
        "session_high",
        "session_low",
        "swing_high",
        "swing_low",
        "equal_high",
        "equal_low",
        "resistance",
        "support",
        "range_high",
        "range_low",
    }
    assert set(ACCEPTED_SOURCE_FAMILIES) == expected


def test_distribution_keys_present():
    out = _run(_bearish_sweep_frame())
    report = build_canonical_sweeps_report(out)
    for key in (
        "breach_atr_distribution",
        "close_reclaim_atr_distribution",
        "distance_at_start_atr_distribution",
    ):
        dist = report[key]
        for stat in (
            "count",
            "mean",
            "median",
            "min",
            "max",
            "p10",
            "p25",
            "p75",
            "p90",
        ):
            assert stat in dist, f"{key}.{stat}"


def test_alias_column_set_matches_module_export():
    out = _run(_bearish_sweep_frame())
    for col in FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS:
        assert col in out.columns, col


def test_upstream_origin_idx_invalid_count_present():
    out = _run(_bearish_sweep_frame())
    report = build_canonical_sweeps_report(out)
    # Field present and integer-typed.
    assert "upstream_origin_idx_invalid_count" in report
    assert isinstance(report["upstream_origin_idx_invalid_count"], int)
