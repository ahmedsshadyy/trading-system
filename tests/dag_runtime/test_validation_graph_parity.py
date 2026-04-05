from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime.builtin_graphs import (
    build_range_boundaries_graph,
    get_builtin_graph,
)
from scripts import validate_range_boundaries as vrb
from scripts import validate_regime as vreg
from scripts import validate_sr_levels as vsr
from scripts import validate_trend_state as vtrend


def _sample_ohlcv(rows: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    close = pd.Series(range(rows), dtype=float).mul(0.6).add(2000.0)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": [1000 + i for i in range(rows)],
        }
    )


def _normalize(value):
    if isinstance(value, pd.DataFrame):
        out = value.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], utc=True)
        return out.reset_index(drop=True)
    if isinstance(value, dict):
        return {str(key): _normalize(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize(child) for child in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        pd.testing.assert_frame_equal(
            _normalize(left), _normalize(right), check_dtype=False
        )
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left.keys()) == set(right.keys())
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert _normalize(left) == _normalize(right)


def test_regime_validation_graph_matches_direct_summary(tmp_path: Path) -> None:
    raw = _sample_ohlcv(240)
    graph = get_builtin_graph("validate_regime", instrument="XAU_USD", timeframe="H4")
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "plot_rows": vreg.PLOT_ROWS,
            "full": False,
            "html": False,
            "out_dir": str(tmp_path),
        },
        cache_root=tmp_path,
        force=True,
        invalidate_cache=True,
    )
    dag = (
        execute_graph(graph, context=context, target="regime_validation_bundle")
        .output()
        .payload
    )

    live_df = vreg._build_context(raw.copy(), include_research_only=False)
    research_df = vreg._build_context(raw.copy(), include_research_only=True)
    direct = vreg.validate_regime(
        research_df.tail(vreg.PLOT_ROWS),
        summary_df=research_df,
        live_df=live_df,
        research_df=research_df,
        outpath=None,
        title="Regime Validation — XAU_USD H4",
        synthetic_summary=vreg._synthetic_fixture_summary(),
    )

    _assert_nested_equal(dag["summary"], direct["summary"])
    assert dag["html_path"] is None


def test_trend_state_validation_graph_matches_direct_summary(tmp_path: Path) -> None:
    raw = _sample_ohlcv(240)
    graph = get_builtin_graph(
        "validate_trend_state", instrument="XAU_USD", timeframe="H4"
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "plot_rows": vtrend.PLOT_ROWS,
            "full": False,
            "html": False,
            "out_dir": str(tmp_path),
        },
        cache_root=tmp_path,
        force=True,
        invalidate_cache=True,
    )
    dag = (
        execute_graph(graph, context=context, target="trend_state_validation_bundle")
        .output()
        .payload
    )

    full_df = vtrend._build_context(raw.copy())
    direct = vtrend.validate_trend_state(
        full_df.tail(vtrend.PLOT_ROWS),
        summary_df=full_df,
        outpath=None,
        title="Trend State Validation — XAU_USD H4",
        n_windows=5,
    )

    _assert_nested_equal(dag["summary"], direct["summary"])
    assert dag["html_path"] is None


def test_sr_levels_validation_graph_matches_direct_summary(tmp_path: Path) -> None:
    raw = _sample_ohlcv(240)
    graph = get_builtin_graph(
        "validate_sr_levels", instrument="XAU_USD", timeframe="H4"
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={"html": False, "out_dir": str(tmp_path)},
        cache_root=tmp_path,
        force=True,
        invalidate_cache=True,
    )
    dag = (
        execute_graph(graph, context=context, target="sr_validation_bundle")
        .output()
        .payload
    )

    enriched = (
        raw.copy()
        .pipe(vsr.normalize_candle_schema, require_volume=True)
        .pipe(vsr.add_atr)
        .pipe(vsr.add_swings)
        .pipe(vsr.add_equal_hl)
        .pipe(vsr.add_prev_day_hl)
        .pipe(vsr.add_prev_week_hl)
        .pipe(vsr.add_session_features)
        .pipe(vsr.add_volume_profile)
    )
    registry = vsr.build_sr_level_registry(enriched)
    live_df = vsr.project_sr_context(enriched.copy(), registry)
    direct_summary = vsr.summarize_sr_levels(live_df, registry, live_df=live_df)

    _assert_nested_equal(dag["summary"], direct_summary)
    assert dag["row_count"] == len(live_df)
    assert dag["html_path"] is None


def test_range_boundaries_graph_skips_retune_when_step8e_a_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _sample_ohlcv(60)
    calls = {"count": 0}

    monkeypatch.setattr(
        vrb, "_load_canonical_live_context", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(vrb, "_build_context", lambda df: df.copy())
    monkeypatch.setattr(
        vrb, "_build_recovery_ladder", lambda: [("mid_a", {}), ("mid_b", {})]
    )

    def fake_run_debug_with_params(
        df: pd.DataFrame, params: dict[str, object]
    ) -> dict[str, object]:
        calls["count"] += 1
        event_table = pd.DataFrame(
            {
                "range_id": [1],
                "confirm_idx": [1.0],
                "end_idx": [5.0],
                "confirm_latency_bars": [3.0],
                "upper_touches": [2],
                "lower_touches": [2],
                "width_atr": [1.5],
                "bars_to_first_breach": [4.0],
                "bars_to_breakout_accept": [6.0],
                "reclaimed_count": [1.0],
                "break_pending_count": [1.0],
                "strength": [0.7],
                "strength_legacy": [0.7],
                "range_strength_structure": [0.7],
                "range_strength_monitorability": [0.7],
                "range_strength_semantic": [0.7],
                "range_strength_viability": [0.7],
                "range_strength_viability_legacy": [0.7],
                "confirm_regime": [0.0],
                "rb_micro_box_risk_score": [0.1],
                "rb_plausibility_score": [0.8],
                "rb_monitor_worthiness_score": [0.8],
                "duration_bars": [3.0],
            }
        )
        return {
            "summary": {
                "event_counts": {"confirmed_ranges": 185, "active_rows": 1050},
                "confirmation_timing": {"confirm_latency_bars": {"median": 3.0}},
                "promotion_funnel": {
                    "raw_candidate_count": 1,
                    "maturity_pass_count": 1,
                    "viability_pass_count": 1,
                },
            },
            "frame": df.copy(),
            "event_table": event_table,
            "candidate_table": event_table.copy(),
        }

    def fake_assess_rung(label: str, result: dict[str, object]) -> dict[str, object]:
        return {
            "label": label,
            "confirmed_ranges": 185,
            "active_rows": 1050,
            "confirm_latency_median": 3.0,
            "short_lived_high_strength_duration_mean": 2.0,
            "coverage_in_band": True,
            "active_in_band": True,
            "latency_ok": True,
            "short_lived_ok": True,
            "plausibility_aligned": True,
            "monitor_aligned": True,
            "micro_box_ok": True,
            "strength_not_badly_inverted": True,
            "valid": label.startswith("step8e_a/"),
            "score": 0.0 if label.startswith("step8e_a/") else 10.0,
            "contract_bucket_summary": {},
        }

    monkeypatch.setattr(vrb, "_run_debug_with_params", fake_run_debug_with_params)
    monkeypatch.setattr(vrb, "_assess_rung", fake_assess_rung)

    graph = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={"html": False, "write_csv": False, "out_dir": str(tmp_path)},
        cache_root=tmp_path,
        force=True,
        invalidate_cache=True,
    )
    result = (
        execute_graph(graph, context=context, target="range_selected_rung")
        .output()
        .payload
    )

    assert calls["count"] == 2
    assert result["used_retune"] is False
    assert result["reporting_label"].startswith("step8e_a/")


def test_range_boundaries_graph_matches_legacy_selection_and_summaries(
    tmp_path: Path,
) -> None:
    raw_path = Path("data/raw/XAU_USD_H4.parquet")
    if not raw_path.exists():
        pytest.skip("raw validation fixture not available")

    raw = pd.read_parquet(raw_path).tail(1200).reset_index(drop=True)
    raw = vrb.normalize_candle_schema(raw, require_volume=False)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "date_from": str(raw["timestamp"].iloc[0].date()),
            "plot_rows": 300,
            "full": False,
            "html": False,
            "write_csv": False,
            "out_dir": str(tmp_path),
        },
        cache_root=tmp_path,
        features_root="data/features",
        force=True,
        invalidate_cache=True,
    )
    dag_result = execute_graph(graph, context=context, target="range_validation_bundle")
    dag = dag_result.output().payload

    context_df = vrb._build_context(raw)
    rung_results: list[tuple[str, dict[str, object]]] = []
    assessments: list[dict[str, object]] = []
    ladder = vrb._build_recovery_ladder()
    for label, overrides in ladder:
        params = {**vrb.BASE_RECOVERY_PARAMS, **overrides}
        phase_label = f"step8e_a/{label}"
        result = vrb._run_debug_with_params(context_df, params)
        rung_results.append((phase_label, result))
        assessments.append(vrb._assess_rung(phase_label, result))
    best_assessment, has_valid_rung = vrb._select_best_assessment(assessments)
    used_retune = False
    if not has_valid_rung:
        used_retune = True
        for label, overrides in ladder:
            params = {
                **vrb.BASE_RECOVERY_PARAMS,
                **overrides,
                **vrb.STEP8E_B_RETUNE_PARAMS,
            }
            phase_label = f"step8e_b/{label}"
            result = vrb._run_debug_with_params(context_df, params)
            rung_results.append((phase_label, result))
            assessments.append(vrb._assess_rung(phase_label, result))
        best_assessment, has_valid_rung = vrb._select_best_assessment(assessments)

    reporting_label = str(best_assessment["label"])
    selected_label = reporting_label if has_valid_rung else "no_valid_contract_rung"
    selected_result = next(
        result for label, result in rung_results if label == reporting_label
    )
    full_df = selected_result["frame"]
    event_table = selected_result["event_table"]
    candidate_table = selected_result["candidate_table"]
    forensics, short_high, long_medium = vrb._build_forensics_tables(event_table)
    forensics = vrb._assign_contract_bucket_labels(
        vrb._add_path_c2_candidate_scores(forensics)
    )
    geometry_audit = vrb._build_geometry_audit(full_df, event_table, candidate_table)
    active_truth_audit, doctrine_report = vrb._build_active_truth_audit(
        full_df, geometry_audit
    )
    ranking_report = vrb._build_ranking_disagreement_report(forensics)
    ranking_repair_gates, ranking_repair_recommendation = (
        vrb._evaluate_path_c2_candidates(forensics)
    )
    downstream_usefulness, downstream_summary = vrb._build_downstream_usefulness_report(
        full_df, forensics
    )
    path_summary = vrb._primary_path_from_reports(
        active_truth_audit,
        doctrine_report,
        ranking_report,
        downstream_summary,
        geometry_audit,
    )
    diagnostics = {
        "contract_bucket_summary": vrb._build_contract_bucket_summary(forensics),
        "interpretability_summary": vrb._build_interpretability_metrics_summary(
            forensics
        ),
        "archetype_summary": vrb._build_archetype_summary(short_high, long_medium),
        "alignment_audit": vrb._build_viability_alignment_audit(
            short_high, long_medium
        ),
        "pressure_audit": vrb._build_pressure_alignment_audit(short_high, long_medium),
        "path_summary": path_summary,
    }

    assert dag["selected_rung"]["reporting_label"] == reporting_label
    assert dag["selected_rung"]["selected_label"] == selected_label
    assert dag["selected_rung"]["used_retune"] == used_retune
    _assert_nested_equal(dag["selected_summary"], selected_result["summary"])
    _assert_nested_equal(dag["downstream_summary"], downstream_summary)
    _assert_nested_equal(dag["diagnostics"], diagnostics)
    assert dag["artifacts"]["range_main_chart"]["html_path"] is None
    assert dag["artifacts"]["range_csv_bundle"]["artifact_paths"] == {}
    assert (
        ranking_repair_recommendation
        == dag_result.node_results["range_ranking_bundle"].output.payload[
            "ranking_repair_recommendation"
        ]
    )
