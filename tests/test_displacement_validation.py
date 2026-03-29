from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.indicators.displacement import (
    displacement_candidate_passes_acceptance,
    extract_displacement_overlap_tables,
    summarize_displacement_overlap,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_overlap_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC"),
            "displacement_flag": np.array([0, 0, 1, 0, 0, 1], dtype=np.int8),
            "bos_bull": np.array([0, 0, 1, 0, 1, 0], dtype=np.int8),
            "bos_bear": np.zeros(6, dtype=np.int8),
            "bos_quality_score": np.array([np.nan, np.nan, 0.20, np.nan, 0.90, np.nan]),
            "bos_tradeable_score": np.array(
                [np.nan, np.nan, 0.80, np.nan, 0.40, np.nan]
            ),
            "bos_direction": np.array([0, 0, 1, 0, 1, 0], dtype=np.int8),
            "choch_bull": np.array([0, 1, 0, 0, 0, 0], dtype=np.int8),
            "choch_bear": np.array([0, 0, 0, 1, 0, 0], dtype=np.int8),
            "choch_quality_score": np.array(
                [np.nan, 0.20, np.nan, 0.80, np.nan, np.nan]
            ),
            "choch_tradeable_score": np.array(
                [np.nan, 0.70, np.nan, 0.90, np.nan, np.nan]
            ),
            "choch_direction": np.array([0, 1, 0, -1, 0, 0], dtype=np.int8),
        }
    )
    return df


def test_displacement_overlap_summary_counts_are_deterministic() -> None:
    summary = summarize_displacement_overlap(_make_overlap_df())

    assert summary["bos"]["event_count"] == 2
    assert np.isclose(summary["bos"]["same_bar"]["rate"], 0.5)
    assert np.isclose(summary["bos"]["pm1"]["rate"], 1.0)
    assert np.isclose(summary["bos"]["pm2"]["rate"], 1.0)
    assert summary["choch"]["event_count"] == 2
    assert np.isclose(summary["choch"]["same_bar"]["rate"], 0.0)
    assert np.isclose(summary["choch"]["pm1"]["rate"], 1.0)
    assert np.isclose(summary["choch"]["pm2"]["rate"], 1.0)
    assert summary["bos"]["top_quartile"]["quality"]["event_count"] == 1
    assert np.isclose(
        summary["bos"]["top_quartile"]["quality"]["same_bar"]["rate"], 0.0
    )
    assert np.isclose(
        summary["bos"]["top_quartile"]["tradeable"]["same_bar"]["rate"], 1.0
    )
    assert summary["choch"]["top_quartile"]["quality"]["event_count"] == 1
    assert np.isclose(summary["choch"]["top_quartile"]["quality"]["pm1"]["rate"], 1.0)


def test_displacement_overlap_tables_include_top_quartile_thresholds() -> None:
    tables = extract_displacement_overlap_tables(_make_overlap_df())

    assert len(tables["bos_events"]) == 2
    assert len(tables["choch_events"]) == 2
    assert set(tables["top_quartile_summary"]["event_type"]) == {"bos", "choch"}
    assert set(tables["top_quartile_summary"]["score_variant"]) == {
        "quality",
        "tradeable",
    }
    bos_quality = (
        tables["top_quartile_summary"]
        .loc[
            (tables["top_quartile_summary"]["event_type"] == "bos")
            & (tables["top_quartile_summary"]["score_variant"] == "quality")
        ]
        .iloc[0]
    )
    assert np.isclose(bos_quality["threshold"], 0.725)
    assert bos_quality["event_count"] == 1
    assert np.isclose(bos_quality["pm1_rate"], 1.0)


def test_displacement_overlap_summary_uses_consistent_index_space_on_slices() -> None:
    df = _make_overlap_df()
    df.index = np.arange(100, 106, dtype=int)

    summary = summarize_displacement_overlap(df)

    assert summary["bos"]["event_count"] == 2
    assert np.isclose(summary["bos"]["same_bar"]["rate"], 0.5)
    assert np.isclose(summary["bos"]["pm1"]["rate"], 1.0)
    assert np.isclose(summary["bos"]["pm2"]["rate"], 1.0)
    assert summary["choch"]["event_count"] == 2
    assert np.isclose(summary["choch"]["same_bar"]["rate"], 0.0)
    assert np.isclose(summary["choch"]["pm1"]["rate"], 1.0)
    assert np.isclose(summary["choch"]["pm2"]["rate"], 1.0)


def test_displacement_candidate_acceptance_enforces_overlap_guardrails() -> None:
    baseline = {
        "bos_quality_pm1_rate": 0.60,
        "bos_tradeable_pm1_rate": 0.62,
        "choch_quality_pm1_rate": 0.59,
        "choch_tradeable_pm1_rate": 0.67,
    }
    candidate = {
        **baseline,
        "event_count": 600,
        "body_atr_hold_monotonicity_ok": True,
        "tradeable_bucket_order_ok": True,
    }
    assert displacement_candidate_passes_acceptance(candidate, baseline) is True

    too_low = dict(candidate)
    too_low["bos_quality_pm1_rate"] = 0.54
    assert displacement_candidate_passes_acceptance(too_low, baseline) is False


def test_analyze_displacement_overlap_script_writes_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module("analyze_displacement_overlap")
    raw = pd.DataFrame({"tickVolume": [1.0]})
    bos_events = pd.DataFrame({"event_idx": [1]})
    choch_events = pd.DataFrame({"event_idx": [2]})
    top_quartile = pd.DataFrame({"event_type": ["bos"], "score_variant": ["quality"]})

    monkeypatch.setattr(module.pd, "read_parquet", lambda _: raw.copy())
    monkeypatch.setattr(
        module,
        "build_displacement_analysis_base_frame",
        lambda df, instrument="XAU_USD": df.copy(),
    )
    monkeypatch.setattr(
        module,
        "build_displacement_analysis_frame",
        lambda df, **kwargs: df.copy(),
    )
    monkeypatch.setattr(
        module,
        "build_displacement_research_table",
        lambda df, atr_length=14: pd.DataFrame(),
    )
    monkeypatch.setattr(
        module,
        "summarize_displacement",
        lambda df, research_table=None, **kwargs: {
            "candidate_comparison": {"event_count": 10},
            "overlap_summary": {"bos": {"event_count": 1}},
            "research_summary": {"recommended_excursion_ratio_variant": "capped"},
        },
    )
    monkeypatch.setattr(
        module,
        "extract_displacement_overlap_tables",
        lambda df: {
            "bos_events": bos_events,
            "choch_events": choch_events,
            "top_quartile_summary": top_quartile,
        },
    )
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(module, "BOS_EVENTS_CSV", tmp_path / "bos.csv")
    monkeypatch.setattr(module, "CHOCH_EVENTS_CSV", tmp_path / "choch.csv")
    monkeypatch.setattr(module, "TOP_QUARTILE_CSV", tmp_path / "top.csv")

    module.main()

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "bos.csv").exists()
    assert (tmp_path / "choch.csv").exists()
    assert (tmp_path / "top.csv").exists()


def test_tune_displacement_script_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module("tune_displacement")
    raw = pd.DataFrame({"tickVolume": [1.0]})
    sweep = pd.DataFrame(
        {
            "accepted": [True],
            "event_distance_to_target": [0.0],
            "total_looseness": [0.1],
        }
    )

    monkeypatch.setattr(module.pd, "read_parquet", lambda _: raw.copy())
    monkeypatch.setattr(
        module,
        "build_displacement_analysis_base_frame",
        lambda df, instrument="XAU_USD": df.copy(),
    )
    monkeypatch.setattr(
        module,
        "run_tuning",
        lambda base_df: {
            "baseline_candidate": {"event_count": 478},
            "selected_candidate": {"event_count": 590, "accepted": True},
            "stopped_stage": "body_atr_mult",
            "results_df": sweep,
        },
    )
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SWEEP_CSV", tmp_path / "sweep.csv")
    monkeypatch.setattr(module, "SUMMARY_JSON", tmp_path / "summary.json")

    module.main()

    assert (tmp_path / "sweep.csv").exists()
    assert (tmp_path / "summary.json").exists()
