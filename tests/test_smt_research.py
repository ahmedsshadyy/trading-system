from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.research.smt_research import (
    build_smt_research_table,
    summarize_smt_research,
)


def _research_frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [99.0, 99.5, 100.0, 103.0, 102.0],
            "high": [100.0, 100.5, 102.0, 105.0, 107.0],
            "low": [98.0, 99.0, 99.0, 100.0, 101.0],
            "close": [99.5, 100.0, 104.0, 100.0, 106.0],
            "atr_14": np.full(5, 2.0, dtype=float),
            "xasset_smt_dxy_dir": [0, 1, 0, 0, 0],
            "xasset_smt_dxy_score": [np.nan, 0.8, np.nan, np.nan, np.nan],
            "xasset_smt_dxy_expected_relation": [-1, -1, -1, -1, -1],
            "xasset_smt_usd_jpy_dir": [0, 0, -1, 0, 0],
            "xasset_smt_usd_jpy_score": [np.nan, np.nan, 0.6, np.nan, np.nan],
            "xasset_smt_usd_jpy_expected_relation": [1, 1, 1, 1, 1],
        }
    )


def test_build_smt_research_table_classifies_failed_and_reversed_outcomes() -> None:
    research = build_smt_research_table(_research_frame(), horizons=(2,))

    assert len(research) == 2

    bull = research.loc[research["smt_partner"] == "dxy"].iloc[0]
    assert bull["smt_direction"] == 1
    assert bull["smt_divergence_type"] == "bullish"
    assert bool(bull["smt_hold_2"]) is False
    assert bool(bull["smt_failed_2"]) is True
    assert bool(bull["smt_reversal_2"]) is False
    assert bull["smt_final_outcome_2"] == "failed"
    assert bull["smt_mfe_2_atr"] == 2.5
    assert bull["smt_mae_2_atr"] == 0.5

    bear = research.loc[research["smt_partner"] == "usd_jpy"].iloc[0]
    assert bear["smt_direction"] == -1
    assert bear["smt_divergence_type"] == "bearish"
    assert bool(bear["smt_hold_2"]) is False
    assert bool(bear["smt_failed_2"]) is True
    assert bool(bear["smt_reversal_2"]) is True
    assert bear["smt_final_outcome_2"] == "reversed"
    assert bear["smt_mfe_2_atr"] == 2.0
    assert bear["smt_mae_2_atr"] == 1.5


def test_build_smt_research_table_marks_insufficient_horizon() -> None:
    frame = _research_frame().tail(2).reset_index(drop=True)
    frame.loc[:, "xasset_smt_dxy_dir"] = 0
    frame.loc[:, "xasset_smt_dxy_score"] = np.nan
    frame.loc[1, "xasset_smt_dxy_dir"] = 1
    frame.loc[1, "xasset_smt_dxy_score"] = 0.7

    research = build_smt_research_table(frame, horizons=(5,))

    assert len(research) == 1
    row = research.iloc[0]
    assert bool(row["smt_hold_5"]) is False
    assert bool(row["smt_failed_5"]) is False
    assert bool(row["smt_reversal_5"]) is False
    assert row["smt_final_outcome_5"] == "insufficient"
    assert np.isnan(row["smt_mfe_5_atr"])
    assert np.isnan(row["smt_mae_5_atr"])


def test_summarize_smt_research_aggregates_partner_stats() -> None:
    research = build_smt_research_table(_research_frame(), horizons=(2,))

    summary = summarize_smt_research(research)

    assert summary["event_count"] == 2
    assert summary["pairs"]["primary__dxy"]["events"] == 1
    assert summary["pairs"]["primary__dxy"]["mean_score"] == 0.8
    assert summary["pairs"]["primary__dxy"]["bullish_pct"] == 1.0
    assert summary["pairs"]["primary__usd_jpy"]["events"] == 1
    assert summary["pairs"]["primary__usd_jpy"]["mean_score"] == 0.6
    assert summary["pairs"]["primary__usd_jpy"]["bullish_pct"] == 0.0
