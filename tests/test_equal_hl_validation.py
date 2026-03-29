from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.indicators.research.equal_hl_research import build_equal_hl_research_table
from src.validation.indicators.equal_hl import (
    apply_eqhl_score_candidate,
    equal_hl_candidate_passes_acceptance,
    summarize_equal_hl,
    validate_equal_hl,
)
from tests.test_equal_hl_research import _build_eqh_research_frame

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_summary_uses_full_dataset_when_provided() -> None:
    full_df = _build_eqh_research_frame()
    plot_df = full_df.iloc[-3:].copy()
    research = build_equal_hl_research_table(full_df)

    summary = summarize_equal_hl(plot_df, full_df=full_df, research_table=research)

    assert summary["rows"] == len(full_df)
    assert summary["eqh_detect_count"] == 1
    assert "structural_score_stats" in summary
    assert "tradeable_live_score_stats" in summary
    assert "formation_delay_stats" in summary
    assert "top_quartile_trade_metrics" in summary
    assert "score_calibration" in summary
    assert "width_reference" in summary
    assert "tradeable_live_score_distribution_audit" in summary
    assert "tradeable_live_score_robustness" in summary
    assert "formation_delay_bucket_by_side_trade_metrics" in summary
    assert "formation_delay_bucket_by_structural_quartile_trade_metrics" in summary
    assert summary["selector_mode"] == "structural_first"


def test_validate_equal_hl_returns_tables_and_summary_without_writing() -> None:
    full_df = _build_eqh_research_frame()
    plot_df = full_df.iloc[-3:].copy()
    research = build_equal_hl_research_table(full_df)

    result = validate_equal_hl(plot_df, full_df=full_df, research_table=research)

    assert result["html_path"] is None
    assert "summary" in result
    assert "tables" in result
    assert isinstance(result["tables"]["strongest_clusters"], pd.DataFrame)
    assert isinstance(result["tables"]["structurally_strongest_clusters"], pd.DataFrame)
    assert isinstance(result["tables"]["fresh_tradeable_clusters"], pd.DataFrame)
    assert isinstance(result["tables"]["calibration_events"], pd.DataFrame)
    assert isinstance(result["tables"]["active_snapshots"], pd.DataFrame)
    assert isinstance(result["tables"]["component_direction_audit"], pd.DataFrame)
    assert isinstance(result["tables"]["width_reference"], pd.DataFrame)


def test_equal_hl_candidate_acceptance_requires_inventory_and_score_separation() -> (
    None
):
    baseline = {
        "event_count": 467,
        "score_repair_accepted": True,
        "top_quartile_avg_r": 0.10,
        "top_decile_avg_r": 0.15,
        "bottom_decile_avg_r": -0.05,
        "top_quartile_win_rate": 0.58,
        "formation_delay_mean": 18.0,
        "active_age_mean": 67.0,
        "eqh_top_quartile_avg_r": 0.09,
        "eql_top_quartile_avg_r": 0.11,
    }
    candidate = {
        "event_count": 500,
        "score_repair_accepted": True,
        "top_quartile_avg_r": 0.11,
        "top_decile_avg_r": 0.18,
        "bottom_decile_avg_r": -0.01,
        "top_quartile_win_rate": 0.60,
        "formation_delay_mean": 20.0,
        "active_age_mean": 72.0,
        "eqh_top_quartile_avg_r": 0.08,
        "eql_top_quartile_avg_r": 0.12,
    }
    assert equal_hl_candidate_passes_acceptance(candidate, baseline) is True

    stale = dict(candidate)
    stale["active_age_mean"] = 90.0
    assert equal_hl_candidate_passes_acceptance(stale, baseline) is False


def test_tune_equal_hl_script_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("tune_equal_hl")
    raw = pd.DataFrame({"tickVolume": [1.0]})
    sweep = pd.DataFrame(
        {
            "accepted": [True],
            "atr_tolerance": [0.15],
            "lookback_swings": [40],
        }
    )

    monkeypatch.setattr(module.pd, "read_parquet", lambda _: raw.copy())
    monkeypatch.setattr(
        module, "build_equal_hl_analysis_base_frame", lambda df: df.copy()
    )
    monkeypatch.setattr(
        module,
        "run_tuning",
        lambda base_df: {
            "baseline_candidate": {"event_count": 467},
            "score_calibration": {
                "component_direction_audit": pd.DataFrame({"a": [1]}),
                "coarse_candidates": pd.DataFrame({"b": [1]}),
                "refined_candidates": pd.DataFrame({"c": [1]}),
                "selected_candidate": {"accepted": True},
            },
            "score_repair_accepted": True,
            "selected_candidate": {"event_count": 530, "accepted": True},
            "results_df": sweep,
        },
    )
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SWEEP_CSV", tmp_path / "sweep.csv")
    monkeypatch.setattr(module, "SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(module, "COMPONENT_AUDIT_CSV", tmp_path / "audit.csv")
    monkeypatch.setattr(module, "COARSE_CANDIDATES_CSV", tmp_path / "coarse.csv")
    monkeypatch.setattr(module, "REFINED_CANDIDATES_CSV", tmp_path / "refined.csv")

    module.main()

    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "audit.csv").exists()
    assert (tmp_path / "coarse.csv").exists()
    assert (tmp_path / "refined.csv").exists()


def test_apply_eqhl_score_candidate_recalibrates_width_component_from_distribution() -> (
    None
):
    calibration_events = pd.DataFrame(
        {
            "eqhl_event_id": [1, 2, 3, 4, 5],
            "side": ["eqh"] * 5,
            "detect_idx": [0, 1, 2, 3, 4],
            "touch_count": [2, 2, 2, 2, 2],
            "width_atr": [0.02, 0.04, 0.06, 0.08, 0.10],
            "formation_delay": [5, 5, 5, 5, 5],
            "structural_score": [0.50, 0.50, 0.50, 0.50, 0.50],
            "tradeable_live_score": [0.0] * 5,
            "touch_component": [0.0] * 5,
            "width_component": [0.0] * 5,
            "active_age_component": [0.5] * 5,
            "formation_delay_component": [0.5] * 5,
            "structural_component": [0.5] * 5,
            "realized_r": [0.1, 0.2, -0.1, 0.0, -0.2],
            "exit_reason": ["time"] * 5,
            "touch_bucket": ["2-touch"] * 5,
            "formation_delay_bucket": ["0_10"] * 5,
        }
    )
    candidate = {
        "name": "width_only",
        "signs": {
            "touch_component": "current",
            "width_component": "current",
            "active_age_component": "current",
            "formation_delay_component": "current",
            "structural_component": "current",
        },
        "touch_component": 0.0,
        "width_component": 1.0,
        "active_age_component": 0.0,
        "formation_delay_component": 0.0,
        "structural_component": 0.0,
    }

    scored = apply_eqhl_score_candidate(calibration_events, candidate_row=candidate)

    assert scored["tradeable_live_score"].nunique() > 1
    assert (
        scored["tradeable_live_score"].iloc[0] > scored["tradeable_live_score"].iloc[-1]
    )
