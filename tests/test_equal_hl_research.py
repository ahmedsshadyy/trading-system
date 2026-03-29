from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.pipelines.build_live import build_live_indicators
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.equal_hl_research import (
    build_equal_hl_research_table,
    summarize_equal_hl_research,
)
from src.indicators.smc.equal_hl import (
    EQHL_ACTIVE_COLUMNS,
    EQHL_DETECT_COLUMNS,
    EQHL_DIAGNOSTIC_COLUMNS,
    EQHL_LEGACY_COLUMNS,
    EQHL_LIFECYCLE_COLUMNS,
    EQHL_RANKED_ACTIVE_COLUMNS,
    add_equal_hl,
)
from tests.test_equal_hl import _make_eqhl_df


def _build_pipeline_df(n: int = 120) -> pd.DataFrame:
    idx = np.arange(n, dtype=float)
    base = 100.0 + np.sin(idx / 7.0) * 2.0 + (idx * 0.02)
    close = base + np.cos(idx / 5.0) * 0.3
    high = np.maximum(base, close) + 0.4
    low = np.minimum(base, close) - 0.4
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tickVolume": 1000 + np.arange(n, dtype=float) * 5.0,
        }
    )


def _build_eqh_research_frame() -> pd.DataFrame:
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
            (99.9, 100.1, 99.5, 99.7),
            (99.7, 100.15, 99.4, 99.6),
            (99.6, 99.95, 98.8, 99.0),
            (99.0, 100.02, 98.7, 99.2),
            (99.2, 99.8, 98.5, 98.9),
        ],
        sh_conf=[0, 1, 0, 1, 0, 0, 0],
        sh_price=[None, 100.0, None, 100.05, None, None, None],
        sh_origin=[None, 0, None, 2, None, None, None],
        sh_strength=[0.8, 0.8, 0.7, 0.7, None, None, None],
    )
    df["trend_state"] = 1.0
    df["trend_bias_state"] = -1.0
    df["bos_direction"] = 0.0
    df["choch_direction"] = 0.0
    df["session"] = "london"
    df["regime"] = 2.0
    df["vol_ratio"] = 1.2
    df["adx"] = 25.0
    df["rsi"] = 48.0
    return add_equal_hl(df)


def _build_eql_research_frame() -> pd.DataFrame:
    df = _make_eqhl_df(
        [
            (100.5, 101.0, 99.7, 100.4),
            (100.4, 100.8, 99.6, 100.2),
            (100.2, 100.6, 99.8, 100.1),
            (100.1, 100.7, 99.55, 100.3),
            (100.3, 100.9, 99.4, 99.45),
            (99.45, 100.2, 99.2, 99.4),
            (99.4, 100.1, 99.1, 99.3),
        ],
        sl_conf=[0, 1, 0, 1, 0, 0, 0],
        sl_price=[None, 99.7, None, 99.65, None, None, None],
        sl_origin=[None, 0, None, 2, None, None, None],
        sl_strength=[0.9, 0.9, 0.8, 0.8, None, None, None],
    )
    return add_equal_hl(df)


def test_research_table_maps_detect_events_exactly() -> None:
    eqh_df = _build_eqh_research_frame()
    eql_df = _build_eql_research_frame()
    combined = pd.concat([eqh_df, eql_df], ignore_index=True)
    research = build_equal_hl_research_table(combined)

    detect_count = int(
        combined["eqh_detect_flag"].sum() + combined["eql_detect_flag"].sum()
    )
    assert len(research) == detect_count
    assert research["eqhl_event_id"].is_unique
    assert sorted(research["eqhl_r_side"].unique().tolist()) == ["eqh", "eql"]


def test_eqh_follow_through_retest_and_excursions_are_coherent() -> None:
    df = _build_eqh_research_frame()
    research = build_equal_hl_research_table(df, horizons=(1, 2, 3, 5))

    row = research.iloc[0]
    assert row["eqhl_r_side"] == "eqh"
    assert row["eqhl_r_age_on_detect"] == 2
    assert row["eqhl_r_formation_delay"] == 2
    assert row["eqhl_r_structural_score"] >= 0.0
    assert row["eqhl_r_tradeable_live_score_on_detect"] >= 0.0
    assert row["eqhl_r_hold_3"] == 1
    assert row["eqhl_r_failed_3"] == 0
    assert row["eqhl_r_swept_3"] == 0
    assert row["eqhl_r_retest_ever_3"] == 1
    assert row["eqhl_r_first_retest_idx"] == 5.0
    assert row["eqhl_r_first_retest_delay"] == 2.0
    assert row["eqhl_r_mfe_3_atr"] >= 0.0
    assert row["eqhl_r_mae_3_atr"] >= 0.0
    assert row["eqhl_r_excursion_ratio_3"] >= 0.0


def test_eql_failure_and_sweep_semantics_are_separate_from_hold() -> None:
    df = _build_eql_research_frame()
    research = build_equal_hl_research_table(df, horizons=(1, 2, 3))

    row = research.iloc[0]
    assert row["eqhl_r_side"] == "eql"
    assert row["eqhl_r_hold_1"] == 0
    assert row["eqhl_r_failed_1"] == 1
    assert row["eqhl_r_swept_1"] == 1
    assert row["eqhl_r_final_outcome_3"] in {"failed", "swept"}


def test_research_summary_is_populated() -> None:
    research = build_equal_hl_research_table(_build_eqh_research_frame())
    summary = summarize_equal_hl_research(research)

    assert summary["event_count"] == 1
    assert "structural_score_stats" in summary
    assert "tradeable_live_score_on_detect_stats" in summary
    assert "follow_through_rates" in summary
    assert "outcome_distribution" in summary


def test_equal_hl_live_research_parity_and_no_contamination() -> None:
    df = _build_pipeline_df()
    live = build_live_indicators(df, instrument="XAU_USD", include_vp=False)
    research = build_research_indicators(
        df,
        instrument="XAU_USD",
        include_vp=False,
        include_avwap=False,
    )

    eqhl_live_columns = (
        EQHL_DETECT_COLUMNS
        + EQHL_ACTIVE_COLUMNS
        + EQHL_RANKED_ACTIVE_COLUMNS
        + EQHL_LIFECYCLE_COLUMNS
        + EQHL_DIAGNOSTIC_COLUMNS
        + EQHL_LEGACY_COLUMNS
    )
    for column in eqhl_live_columns:
        pd.testing.assert_series_equal(
            live[column],
            research[column],
            check_names=False,
            check_dtype=False,
        )

    assert not any(col.startswith("eqhl_r_") for col in live.columns)
    assert not any(col.startswith("eqhl_r_") for col in research.columns)
    banned_tokens = ("mfe", "mae", "hold", "fail", "retest", "outcome")
    for frame in (live, research):
        eqhl_cols = [
            col for col in frame.columns if col.startswith(("eqh_", "eql_", "equal_"))
        ]
        assert not any(
            any(token in col for token in banned_tokens) for col in eqhl_cols
        )
