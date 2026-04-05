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


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
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


def _load_real_raw(rows: int = 1200) -> pd.DataFrame:
    raw_path = Path("data/raw/XAU_USD_H4.parquet")
    if not raw_path.exists():
        pytest.skip("raw validation fixture not available")
    raw = pd.read_parquet(raw_path).tail(rows).reset_index(drop=True)
    raw = vrb.normalize_candle_schema(raw, require_volume=False)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    return raw


def _make_context(
    graph_name: str,
    raw: pd.DataFrame,
    cache_root: Path,
    out_dir: Path,
    *,
    html: bool = False,
    write_csv: bool = False,
    force: bool = False,
    invalidate_cache: bool = False,
) -> GraphRunContext:
    return GraphRunContext(
        graph_name=graph_name,
        symbol="XAU_USD",
        timeframe="H4",
        inputs={"raw_input": raw},
        config={
            "date_from": str(raw["timestamp"].iloc[0].date()),
            "plot_rows": 300,
            "full": False,
            "html": html,
            "write_csv": write_csv,
            "out_dir": str(out_dir),
        },
        cache_root=cache_root,
        features_root="data/features",
        force=force,
        invalidate_cache=invalidate_cache,
    )


def _node_detail(summary: dict[str, object], node_name: str) -> dict[str, object]:
    for node in summary["nodes"]:
        if node["node_name"] == node_name:
            return node["details"]
    raise AssertionError(f"node not found in profiler summary: {node_name}")


def test_range_rung_profiler_exposes_substage_timings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _sample_ohlcv(60)

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
            "profile_details": {
                "substage_seconds": {
                    "debug_collect": 1.0,
                    "pressure_imbalance_legacy": 2.0,
                    "pressure_imbalance_v2": 3.0,
                    "contract_scores": 4.0,
                    "summary_build": 5.0,
                },
                "skipped": False,
            },
        }

    def fake_assess_rung(label: str, result: dict[str, object]) -> dict[str, object]:
        return {
            "label": label,
            "confirmed_ranges": 185,
            "active_rows": 1050,
            "confirm_latency_median": 3.0,
            "short_lived_high_strength_duration_mean": 2.0,
            "plausibility_aligned": True,
            "monitor_aligned": True,
            "micro_box_ok": True,
            "strength_not_badly_inverted": True,
            "valid": label.startswith("step8e_a/"),
            "score": 0.0,
        }

    monkeypatch.setattr(vrb, "_run_debug_with_params", fake_run_debug_with_params)
    monkeypatch.setattr(vrb, "_assess_rung", fake_assess_rung)

    graph = build_range_boundaries_graph(instrument="XAU_USD", timeframe="H4")
    result = execute_graph(
        graph,
        context=GraphRunContext(
            graph_name=graph.graph_name,
            symbol="XAU_USD",
            timeframe="H4",
            inputs={"raw_input": raw},
            cache_root=tmp_path,
            force=True,
            invalidate_cache=True,
        ),
        target="range_selected_debug",
    )
    summary = result.profiler.summary()

    rung_details = _node_detail(summary, "range_rung_debug__step8e_a__mid_a")
    assert rung_details["label"] == "step8e_a/mid_a"
    assert rung_details["skipped"] is False
    assert set(rung_details["substage_seconds"]) == {
        "debug_collect",
        "pressure_imbalance_legacy",
        "pressure_imbalance_v2",
        "contract_scores",
        "summary_build",
    }

    skipped_details = _node_detail(summary, "range_rung_debug__step8e_b__mid_a")
    assert skipped_details["skipped"] is True
    assert skipped_details["substage_seconds"]["debug_collect"] == 0.0
    assert skipped_details["substage_seconds"]["summary_build"] == 0.0

    selected_debug_details = _node_detail(summary, "range_selected_debug")
    assert selected_debug_details["label"] == "selected_debug"
    assert "cache_write_seconds" in selected_debug_details
    assert "selected_debug_cache_write" in selected_debug_details


def test_range_boundaries_selection_rerun_uses_cache_and_skips_post_selection_analytics(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    out_dir = tmp_path / "reports"
    context = _make_context(graph.graph_name, raw, tmp_path / "cache", out_dir)

    first = execute_graph(graph, context=context, target="range_selection_bundle")
    second = execute_graph(graph, context=context, target="range_selection_bundle")

    forbidden_nodes = {
        "range_selected_debug",
        "range_forensics",
        "range_geometry_audit",
        "range_active_truth_audit",
        "range_coverage_regime_report",
        "range_ranking_bundle",
        "range_downstream_usefulness",
        "range_diagnostics_bundle",
        "range_chart_bundle",
        "range_csv_bundle",
    }
    assert forbidden_nodes.isdisjoint(first.closure_nodes)
    assert forbidden_nodes.isdisjoint(second.closure_nodes)
    assert set(second.executed_nodes).issubset({"raw_input", "range_selection_bundle"})


def test_range_boundaries_geometry_target_executes_minimal_closure_after_warmup(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    warm_context = _make_context(graph.graph_name, raw, cache_root, out_dir)
    execute_graph(graph, context=warm_context, target="range_analysis_bundle")

    result = execute_graph(
        graph,
        context=_make_context(graph.graph_name, raw, cache_root, out_dir),
        target="range_geometry_audit",
        invalidate_nodes={"range_geometry_audit"},
    )

    assert set(result.executed_nodes).issubset({"raw_input", "range_geometry_audit"})
    assert "range_downstream_usefulness" not in result.closure_nodes
    assert "range_chart_bundle" not in result.closure_nodes
    assert "range_csv_bundle" not in result.closure_nodes


def test_range_boundaries_active_truth_target_executes_minimal_closure_after_warmup(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    warm_context = _make_context(graph.graph_name, raw, cache_root, out_dir)
    execute_graph(graph, context=warm_context, target="range_analysis_bundle")

    result = execute_graph(
        graph,
        context=_make_context(graph.graph_name, raw, cache_root, out_dir),
        target="range_active_truth_audit",
        invalidate_nodes={"range_active_truth_audit"},
    )

    assert set(result.executed_nodes).issubset(
        {"raw_input", "range_active_truth_audit"}
    )
    assert "range_downstream_usefulness" not in result.closure_nodes
    assert "range_chart_bundle" not in result.closure_nodes
    assert "range_csv_bundle" not in result.closure_nodes


def test_range_boundaries_downstream_target_executes_minimal_closure_after_warmup(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    warm_context = _make_context(graph.graph_name, raw, cache_root, out_dir)
    execute_graph(graph, context=warm_context, target="range_analysis_bundle")

    result = execute_graph(
        graph,
        context=_make_context(graph.graph_name, raw, cache_root, out_dir),
        target="range_downstream_usefulness",
        invalidate_nodes={"range_downstream_usefulness"},
    )

    assert set(result.executed_nodes).issubset(
        {"raw_input", "range_downstream_usefulness"}
    )
    assert "range_chart_bundle" not in result.closure_nodes
    assert "range_csv_bundle" not in result.closure_nodes


def test_range_boundaries_chart_target_does_not_recompute_upstream_compute_on_warm_cache(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    execute_graph(
        graph,
        context=_make_context(graph.graph_name, raw, cache_root, out_dir, html=False),
        target="range_analysis_bundle",
    )

    result = execute_graph(
        graph,
        context=_make_context(graph.graph_name, raw, cache_root, out_dir, html=True),
        target="range_chart_bundle",
    )

    forbidden_compute = {
        "range_context",
        "range_selected_rung",
        "range_selected_debug",
        "range_forensics",
        "range_geometry_audit",
        "range_active_truth_audit",
        "range_coverage_regime_report",
        "range_ranking_bundle",
        "range_downstream_usefulness",
        "range_diagnostics_bundle",
    }
    assert forbidden_compute.isdisjoint(result.executed_nodes)
    artifact_nodes = {
        record["node_name"] for record in result.profiler.summary()["artifacts_written"]
    }
    assert artifact_nodes
    assert artifact_nodes.issubset(
        {
            "range_main_chart",
            "range_geometry_chart_pack",
            "range_refresh_chart_pack",
            "range_downstream_chart_pack",
        }
    )


def test_range_boundaries_csv_target_does_not_execute_chart_nodes_on_warm_cache(
    tmp_path: Path,
) -> None:
    raw = _load_real_raw(800)
    graph = get_builtin_graph(
        "validate_range_boundaries", instrument="XAU_USD", timeframe="H4"
    )
    cache_root = tmp_path / "cache"
    out_dir = tmp_path / "reports"
    execute_graph(
        graph,
        context=_make_context(
            graph.graph_name, raw, cache_root, out_dir, html=False, write_csv=False
        ),
        target="range_analysis_bundle",
    )

    result = execute_graph(
        graph,
        context=_make_context(
            graph.graph_name, raw, cache_root, out_dir, html=False, write_csv=True
        ),
        target="range_csv_bundle",
    )

    chart_nodes = {
        "range_main_chart",
        "range_geometry_chart_pack",
        "range_refresh_chart_pack",
        "range_downstream_chart_pack",
        "range_chart_bundle",
    }
    forbidden_compute = {
        "range_context",
        "range_selected_rung",
        "range_selected_debug",
        "range_forensics",
        "range_geometry_audit",
        "range_active_truth_audit",
        "range_coverage_regime_report",
        "range_ranking_bundle",
        "range_downstream_usefulness",
        "range_diagnostics_bundle",
    }
    assert chart_nodes.isdisjoint(result.executed_nodes)
    assert forbidden_compute.isdisjoint(result.executed_nodes)
    artifact_nodes = {
        record["node_name"] for record in result.profiler.summary()["artifacts_written"]
    }
    assert artifact_nodes == {"range_csv_bundle"}
