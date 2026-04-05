from __future__ import annotations

import pandas as pd

from scripts.validate_range_boundaries import (
    _add_contract_scores,
    _add_path_c2_candidate_scores,
    _build_agreement_matrix,
    _build_bucket_lift_report,
    _build_geometry_audit,
    _build_geometry_candidate_active_truth_summary,
    _build_geometry_candidate_comparison,
    _build_geometry_candidate_downstream_summary,
    _build_geometry_candidate_gate_report,
    _build_geometry_ranking_preservation_report,
    _build_path_c2_candidate_report,
    _build_path_c2_archetype_report,
    _build_ranking_rebase_comparison_report,
    _compute_geometry_candidates,
    _assign_contract_bucket_labels,
    _assess_rung,
    _evaluate_path_c2_candidates,
    _primary_path_from_reports,
    _select_best_assessment,
)


def _build_event_table(*, good_long: bool, confirm_latency: float) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for idx in range(10):
        rows.append(
            {
                "range_id": idx + 1,
                "confirm_idx": float(idx * 10),
                "end_idx": float(idx * 10 + 1),
                "confirm_latency_bars": confirm_latency,
                "upper_touches": 3 + (idx % 2),
                "lower_touches": 3,
                "width_atr": 1.6 + 0.05 * idx,
                "bars_to_first_breach": 1.0,
                "bars_to_breakout_accept": 1.0,
                "reclaimed_count": 0.0,
                "break_pending_count": 0.0,
                "strength": 0.72 + 0.01 * idx,
                "strength_legacy": 0.78 + 0.01 * idx,
                "range_strength_structure": 0.70 + 0.01 * idx,
                "range_strength_monitorability": 0.38 + 0.01 * idx,
                "range_strength_semantic": 0.36 + 0.01 * idx,
                "range_strength_viability": 0.40 + 0.01 * idx,
                "range_strength_viability_legacy": 0.76 + 0.01 * idx,
                "confirm_regime": 0.0,
            }
        )
    for idx in range(10):
        if good_long:
            rows.append(
                {
                    "range_id": idx + 11,
                    "confirm_idx": float(200 + idx * 10),
                    "end_idx": float(200 + idx * 10 + 14 + idx),
                    "confirm_latency_bars": max(confirm_latency, 3.0),
                    "upper_touches": 2 + (idx % 2),
                    "lower_touches": 2 + ((idx + 1) % 2),
                    "width_atr": 2.1 + 0.04 * idx,
                    "bars_to_first_breach": 7.0 + idx,
                    "bars_to_breakout_accept": 14.0 + idx,
                    "reclaimed_count": 2.0,
                    "break_pending_count": 2.0,
                    "strength": 0.60 + 0.005 * idx,
                    "strength_legacy": 0.58 + 0.005 * idx,
                    "range_strength_structure": 0.62 + 0.005 * idx,
                    "range_strength_monitorability": 0.76 + 0.005 * idx,
                    "range_strength_semantic": 0.78 + 0.005 * idx,
                    "range_strength_viability": 0.74 + 0.005 * idx,
                    "range_strength_viability_legacy": 0.56 + 0.005 * idx,
                    "confirm_regime": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "range_id": idx + 11,
                    "confirm_idx": float(200 + idx * 10),
                    "end_idx": float(200 + idx * 10 + 8),
                    "confirm_latency_bars": 7.0,
                    "upper_touches": 2.0,
                    "lower_touches": 2.0,
                    "width_atr": 2.5,
                    "bars_to_first_breach": 1.0,
                    "bars_to_breakout_accept": 1.0,
                    "reclaimed_count": 0.0,
                    "break_pending_count": 0.0,
                    "strength": 0.60 + 0.005 * idx,
                    "strength_legacy": 0.58 + 0.005 * idx,
                    "range_strength_structure": 0.62 + 0.005 * idx,
                    "range_strength_monitorability": 0.30 + 0.005 * idx,
                    "range_strength_semantic": 0.28 + 0.005 * idx,
                    "range_strength_viability": 0.32 + 0.005 * idx,
                    "range_strength_viability_legacy": 0.56 + 0.005 * idx,
                    "confirm_regime": 0.0,
                }
            )
    return _add_contract_scores(pd.DataFrame(rows))


def test_contract_scores_rank_durable_range_above_instant_fail_micro_box() -> None:
    scored = _build_event_table(good_long=True, confirm_latency=3.0)
    short = scored.iloc[:10]
    long = scored.iloc[10:]

    assert float(long["rb_micro_box_risk_score"].mean()) < float(
        short["rb_micro_box_risk_score"].mean()
    )
    assert float(long["rb_late_confirm_fragility_score"].mean()) < float(
        short["rb_late_confirm_fragility_score"].mean()
    )
    assert float(long["rb_monitor_worthiness_score"].mean()) > float(
        short["rb_monitor_worthiness_score"].mean()
    )
    assert float(long["rb_plausibility_score"].mean()) > float(
        short["rb_plausibility_score"].mean()
    )


def test_rung_with_good_coverage_but_latency_one_is_rejected() -> None:
    assessment = _assess_rung(
        "latency_bad",
        {
            "summary": {
                "event_counts": {"confirmed_ranges": 180, "active_rows": 900},
                "confirmation_timing": {"confirm_latency_bars": {"median": 1.0}},
            },
            "event_table": _build_event_table(good_long=True, confirm_latency=1.0),
        },
    )

    assert assessment["coverage_in_band"] is True
    assert assessment["latency_ok"] is False
    assert assessment["valid"] is False


def test_rung_with_inverted_plausibility_is_rejected() -> None:
    event_table = _build_event_table(good_long=False, confirm_latency=3.0)
    event_table.loc[event_table.index[:10], "rb_plausibility_score"] = 0.40
    event_table.loc[event_table.index[10:], "rb_plausibility_score"] = 0.20
    assessment = _assess_rung(
        "plausibility_bad",
        {
            "summary": {
                "event_counts": {"confirmed_ranges": 180, "active_rows": 900},
                "confirmation_timing": {"confirm_latency_bars": {"median": 3.0}},
            },
            "event_table": event_table,
        },
    )

    assert assessment["coverage_in_band"] is True
    assert assessment["plausibility_aligned"] is False
    assert assessment["valid"] is False


def test_valid_rung_is_selected_over_looser_invalid_rung() -> None:
    valid = {
        "label": "mid_good",
        "valid": True,
        "score": 0.40,
    }
    looser_invalid = {
        "label": "mid_loose",
        "valid": False,
        "score": 0.05,
    }

    selected, has_valid = _select_best_assessment([looser_invalid, valid])

    assert has_valid is True
    assert selected["label"] == "mid_good"


def test_contract_bucket_labels_assign_expected_family() -> None:
    scored = _assign_contract_bucket_labels(
        _build_event_table(good_long=True, confirm_latency=3.0)
    )

    short = scored.iloc[:10]
    long = scored.iloc[10:]

    assert set(short["contract_bucket"]).issubset(
        {
            "weak_false_positive",
            "strong_false_positive",
            "fragile_but_plausible",
            "unclassified",
        }
    )
    assert "durable_plausible" in set(long["contract_bucket"])


def test_primary_path_prefers_controlled_refresh_when_staleness_dominates() -> None:
    active_truth = pd.DataFrame(
        {
            "range_id": [1, 2, 3],
            "failure_classification": [
                "frozen_box_became_stale",
                "frozen_box_became_stale",
                "confirm_too_late",
            ],
        }
    )
    doctrine = pd.DataFrame(
        {
            "range_id": [1, 1, 2, 2],
            "doctrine": [
                "frozen",
                "bounded_continuity_refresh",
                "frozen",
                "bounded_continuity_refresh",
            ],
            "visual_plausibility_score": [0.40, 0.60, 0.45, 0.62],
        }
    )
    ranking = pd.DataFrame(
        {
            "ranking_metric": ["strength", "rb_plausibility_score"],
            "durable_plausible": [2, 3],
            "strong_false_positive": [3, 1],
        }
    )
    downstream = {"fraction_presence_improved_interpretation": 0.50}
    geometry = pd.DataFrame(
        {"geometry_review_bucket_suggested": ["edge_matches_chart_well"]}
    )

    out = _primary_path_from_reports(
        active_truth, doctrine, ranking, downstream, geometry
    )

    assert out["primary_next_path"] == "Path B — controlled refresh fix"


def test_ranking_rebase_reports_show_rebased_mix_closer_to_truth() -> None:
    scored = _assign_contract_bucket_labels(
        _build_event_table(good_long=True, confirm_latency=3.0)
    )

    rebase = _build_ranking_rebase_comparison_report(scored, top_n=10)
    agreement = _build_agreement_matrix(scored, top_n=10)
    lift = _build_bucket_lift_report(scored, top_n=10)

    legacy = rebase[rebase["ranking_family"] == "legacy_strength"].iloc[0]
    rebased = rebase[rebase["ranking_family"] == "rebased_strength"].iloc[0]

    assert int(rebased["durable_plausible"]) >= int(legacy["durable_plausible"])
    assert float(rebased["mean_plausibility"]) >= float(legacy["mean_plausibility"])
    assert float(rebased["mean_micro_box_risk"]) <= float(legacy["mean_micro_box_risk"])
    assert not agreement.empty
    durable_lift = lift[
        (lift["ranking_label"] == "failed_path_c")
        & (lift["bucket"] == "durable_plausible")
    ].iloc[0]
    assert "candidate_minus_legacy" in durable_lift.index


def test_path_c2_candidate_report_prefers_truth_gated_candidate_on_fixture() -> None:
    scored = _assign_contract_bucket_labels(
        _add_path_c2_candidate_scores(
            _build_event_table(good_long=True, confirm_latency=3.0)
        )
    )
    report = _build_path_c2_candidate_report(scored, top_ns=(10,))
    gates, recommendation = _evaluate_path_c2_candidates(scored, top_n=10)
    archetypes = _build_path_c2_archetype_report(scored)

    legacy = report[report["candidate_label"] == "legacy"].iloc[0]
    repair = report[report["candidate_label"] == "repair_v2_eligibility_gated"].iloc[0]

    assert float(repair["mean_monitor_worthiness"]) >= float(
        legacy["mean_monitor_worthiness"]
    )
    assert float(repair["mean_plausibility"]) >= float(legacy["mean_plausibility"])
    assert float(repair["mean_micro_box_risk"]) <= float(legacy["mean_micro_box_risk"])
    assert recommendation == "repair_v2_eligibility_gated"
    assert (
        bool(
            gates[gates["candidate_label"] == "repair_v2_eligibility_gated"][
                "passed"
            ].iloc[0]
        )
        is True
    )
    short = archetypes[archetypes["cohort"] == "short_lived_high_strength"].iloc[0]
    long = archetypes[archetypes["cohort"] == "long_lived_medium_strength"].iloc[0]
    assert float(long["strength_repair_v2"]) >= float(short["strength_repair_v2"])


def _build_geometry_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="4h", tz="UTC"),
            "open": [100.5, 101.0, 101.8, 101.4, 101.1, 100.9, 101.7, 102.2],
            "high": [101.0, 102.2, 102.6, 102.0, 101.8, 101.7, 102.9, 103.1],
            "low": [99.8, 100.9, 101.3, 100.8, 100.7, 100.5, 101.2, 101.8],
            "close": [100.7, 101.9, 101.5, 101.2, 101.0, 101.6, 102.5, 102.9],
            "atr": [0.8] * 8,
            "swing_high_confirm_flag": [0, 1, 0, 0, 0, 0, 1, 0],
            "swing_high_confirm_price": [
                pd.NA,
                102.2,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                102.9,
                pd.NA,
            ],
            "swing_low_confirm_flag": [1, 0, 0, 0, 1, 0, 0, 0],
            "swing_low_confirm_price": [
                99.8,
                pd.NA,
                pd.NA,
                pd.NA,
                100.7,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "bos_bull": [0, 0, 0, 0, 0, 0, 1, 0],
            "bos_bear": [0] * 8,
            "choch_bull": [0] * 8,
            "choch_bear": [0] * 8,
        }
    )
    events = pd.DataFrame(
        {
            "range_id": [1],
            "birth_idx": [1.0],
            "confirm_idx": [4.0],
            "end_idx": [6.0],
            "state": [4],
            "high": [101.8],
            "low": [100.8],
            "width_atr": [1.25],
            "confirm_latency_bars": [3.0],
            "candidate_lookback_bars": [5.0],
            "upper_touches": [3.0],
            "lower_touches": [3.0],
            "bars_to_first_breach": [4.0],
            "bars_to_breakout_accept": [5.0],
            "reclaimed_count": [1.0],
            "break_pending_count": [1.0],
            "duration_bars": [2.0],
            "strength": [0.55],
            "strength_legacy": [0.58],
            "range_strength_structure": [0.52],
            "range_strength_monitorability": [0.80],
            "range_strength_semantic": [0.82],
            "range_strength_viability": [0.74],
            "range_strength_viability_legacy": [0.57],
            "confirm_regime": [0.0],
        }
    )
    scored = _add_contract_scores(events)
    scored["strength_repair_v2"] = 0.88
    return frame, scored


def test_geometry_candidates_add_deterministic_upper_and_lower_columns() -> None:
    frame, events = _build_geometry_fixture()
    out = _compute_geometry_candidates(frame, events)

    for high_col, low_col in (
        ("range_high_g2", "range_low_g2"),
        ("range_high_g3", "range_low_g3"),
        ("range_high_g4", "range_low_g4"),
        ("range_high_g5", "range_low_g5"),
    ):
        assert high_col in out.columns
        assert low_col in out.columns
        assert float(out.iloc[0][high_col]) >= float(out.iloc[0]["high"])
        assert float(out.iloc[0][low_col]) <= float(out.iloc[0]["low"])


def test_geometry_candidate_comparison_builds_shifts_and_buckets() -> None:
    frame, events = _build_geometry_fixture()
    events = _compute_geometry_candidates(frame, events)
    geometry_audit = _build_geometry_audit(frame, events, pd.DataFrame())
    comparison, summary = _build_geometry_candidate_comparison(frame, geometry_audit)

    assert set(comparison["candidate_family"]) == {
        "g1_legacy",
        "g2_envelope_extended_extrema",
        "g3_touch_cluster_envelope",
        "g4_quantile_envelope",
        "g5_widened_compact_core_with_guardrails",
    }
    assert "width_delta_atr" in comparison.columns
    assert "geometry_review_bucket_suggested" in comparison.columns
    assert "faithful_or_partial_share" in summary.columns


def test_geometry_candidate_downstream_ranking_and_gate_reports_build() -> None:
    frame, events = _build_geometry_fixture()
    events = _compute_geometry_candidates(frame, events)
    geometry_audit = _build_geometry_audit(frame, events, pd.DataFrame())
    comparison, summary = _build_geometry_candidate_comparison(frame, geometry_audit)
    active_truth = pd.DataFrame(
        {
            "range_id": [1],
            "failure_classification": ["confirm_too_late"],
            "frozen_visual_plausibility_score": [0.40],
        }
    )
    truth_summary = _build_geometry_candidate_active_truth_summary(
        frame, geometry_audit, active_truth
    )
    downstream_summary = _build_geometry_candidate_downstream_summary(
        frame, geometry_audit
    )
    forensics = _assign_contract_bucket_labels(_add_path_c2_candidate_scores(events))
    ranking_preservation = _build_geometry_ranking_preservation_report(
        forensics, comparison, top_n=1
    )
    gate_report, recommendation = _build_geometry_candidate_gate_report(
        summary,
        truth_summary,
        downstream_summary,
        ranking_preservation,
    )

    assert not truth_summary.empty
    assert not downstream_summary.empty
    assert not ranking_preservation.empty
    assert not gate_report.empty
    assert recommendation in {
        "no_candidate_passed",
        "g2_envelope_extended_extrema",
        "g3_touch_cluster_envelope",
        "g4_quantile_envelope",
        "g5_widened_compact_core_with_guardrails",
    }
