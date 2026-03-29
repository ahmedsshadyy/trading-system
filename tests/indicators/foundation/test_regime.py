from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.regime import (
    add_regime,
    _assign_raw_regime,
    _stabilize_regime,
)
from src.validation.indicators.regime import validate_regime


def _make_regime_input_df(n: int = 12) -> pd.DataFrame:
    close = np.linspace(100.0, 105.0, n)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": close - 0.3,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "atr_14": np.full(n, 2.0, dtype=float),
            "adx_14": np.full(n, 25.0, dtype=float),
            "bb_width": np.full(n, 0.05, dtype=float),
            "bb_width_pct_rank_100": np.full(n, 0.50, dtype=float),
            "ema_20_slope": np.full(n, 0.2, dtype=float),
            "ema_20_slope_atr": np.full(n, 0.2, dtype=float),
            "trend_state": np.ones(n, dtype=np.int8),
            "trend_confidence": np.full(n, 2, dtype=np.int8),
            "hh_count": np.full(n, 3, dtype=np.int16),
            "ll_count": np.zeros(n, dtype=np.int16),
            "trend_bias_state": np.ones(n, dtype=np.int8),
            "trend_strength_ema": np.full(n, 0.5, dtype=float),
        }
    )
    return df


def _make_stabilization_frame(
    raw_regime: list[int], raw_margin: list[float]
) -> pd.DataFrame:
    n = len(raw_regime)
    return pd.DataFrame(
        {
            "regime_input_ready": np.ones(n, dtype=np.int8),
            "range_regime_score": np.where(np.array(raw_regime) == 0, 0.80, 0.20),
            "transition_regime_score": np.where(np.array(raw_regime) == 1, 0.80, 0.20),
            "trend_regime_score": np.where(np.array(raw_regime) == 2, 0.80, 0.20),
            "regime_confidence": np.full(n, 0.80, dtype=float),
            "regime_margin": np.array(raw_margin, dtype=float),
        }
    )


def test_add_regime_is_pure_and_preserves_index() -> None:
    raw = _make_regime_input_df()
    raw.index = pd.Index(np.arange(100, 100 + len(raw)), name="row_id")
    original = raw.copy(deep=True)

    result = add_regime(raw, include_research_only=False)

    pd.testing.assert_frame_equal(raw, original)
    assert result.index.equals(raw.index)
    for col in raw.columns:
        pd.testing.assert_series_equal(result[col], raw[col], check_names=False)


def test_regime_warmup_is_explicit_until_all_inputs_are_valid() -> None:
    raw = _make_regime_input_df(8)
    raw.loc[:2, "adx_14"] = np.nan
    raw.loc[:4, "bb_width_pct_rank_100"] = np.nan
    raw.loc[:1, "ema_20_slope_atr"] = np.nan
    raw.loc[:3, "trend_confidence"] = np.nan

    result = add_regime(raw)

    assert result.loc[:4, "regime"].isna().all()
    assert result.loc[:4, "regime_label"].isna().all()
    assert (
        result.loc[
            :4, ["regime_is_ranging", "regime_is_transitional", "regime_is_trending"]
        ]
        .sum()
        .sum()
        == 0
    )
    assert result.loc[5, "regime"] in (0, 1, 2)


def test_regime_fixture_a_clean_range_maps_to_ranging() -> None:
    raw = _make_regime_input_df(6)
    raw["adx_14"] = 12.0
    raw["bb_width_pct_rank_100"] = 0.05
    raw["ema_20_slope_atr"] = 0.02
    raw["trend_state"] = 0
    raw["trend_confidence"] = 0
    raw["hh_count"] = 0
    raw["ll_count"] = 0
    raw["trend_bias_state"] = 0

    result = add_regime(raw)
    valid = result["regime"].dropna()

    assert not valid.empty
    assert (valid.astype(int) == 0).all()
    assert (
        result.loc[valid.index, "range_regime_score"]
        > result.loc[valid.index, "trend_regime_score"]
    ).all()


def test_regime_fixture_b_clean_trend_maps_to_trending() -> None:
    raw = _make_regime_input_df(6)
    raw["adx_14"] = 38.0
    raw["bb_width_pct_rank_100"] = 0.85
    raw["ema_20_slope_atr"] = 0.50
    raw["trend_state"] = 1
    raw["trend_confidence"] = 2
    raw["hh_count"] = 4
    raw["ll_count"] = 0

    result = add_regime(raw)
    valid = result["regime"].dropna()

    assert not valid.empty
    assert (valid.astype(int) == 2).all()
    assert (
        result.loc[valid.index, "trend_regime_score"]
        > result.loc[valid.index, "range_regime_score"]
    ).all()


def test_regime_fixture_c_mixed_state_maps_to_transitional_and_boundary() -> None:
    raw = _make_regime_input_df(6)
    raw["adx_14"] = 26.0
    raw["bb_width_pct_rank_100"] = 0.45
    raw["ema_20_slope_atr"] = 0.15
    raw["trend_state"] = 1
    raw["trend_confidence"] = 0.5
    raw["hh_count"] = 1
    raw["ll_count"] = 0
    raw["trend_bias_state"] = 1

    result = add_regime(raw)
    valid_idx = result["regime"].dropna().index
    assert not valid_idx.empty
    assert (result.loc[valid_idx, "regime"] == 1).all()
    assert (
        result.loc[valid_idx, "transition_regime_score"]
        >= result.loc[valid_idx, ["trend_regime_score", "range_regime_score"]].max(
            axis=1
        )
    ).all()


def test_regime_transition_sequence_resets_bars_and_enter_flags() -> None:
    raw = pd.concat(
        [
            _make_regime_input_df(3).assign(
                adx_14=12.0,
                bb_width_pct_rank_100=0.05,
                ema_20_slope_atr=0.02,
                trend_state=0,
                trend_confidence=0,
                hh_count=0,
                ll_count=0,
                trend_bias_state=0,
            ),
            _make_regime_input_df(3).assign(
                adx_14=26.0,
                bb_width_pct_rank_100=0.45,
                ema_20_slope_atr=0.15,
                trend_state=1,
                trend_confidence=0.5,
                hh_count=1,
                ll_count=0,
                trend_bias_state=1,
            ),
            _make_regime_input_df(3).assign(
                adx_14=38.0,
                bb_width_pct_rank_100=0.85,
                ema_20_slope_atr=0.50,
                trend_state=1,
                trend_confidence=2,
                hh_count=4,
                ll_count=0,
                trend_bias_state=1,
            ),
        ],
        ignore_index=True,
    )
    raw["timestamp"] = pd.date_range(
        "2024-01-01", periods=len(raw), freq="1h", tz="UTC"
    )

    result = add_regime(raw)

    assert result.loc[0, "regime_enter_ranging"] == 1
    assert result.loc[3, "regime_enter_transitional"] == 1
    assert result.loc[6, "regime_enter_trending"] == 1
    assert result.loc[0, "bars_in_regime"] == 1
    assert result.loc[3, "bars_in_regime"] == 1
    assert result.loc[6, "bars_in_regime"] == 1
    assert result.loc[1, "bars_in_regime"] == 2
    assert result.loc[4, "bars_in_regime"] == 2
    assert result.loc[7, "bars_in_regime"] == 2


def test_regime_scores_bounds_and_one_hot_contract_hold() -> None:
    result = add_regime(_make_regime_input_df(20))
    valid = result["regime"].notna()

    for col in (
        "trend_regime_score",
        "range_regime_score",
        "transition_regime_score",
        "regime_confidence",
        "regime_margin",
        "raw_regime_confidence",
        "raw_regime_margin",
    ):
        values = pd.to_numeric(result.loc[valid, col], errors="coerce")
        assert values.between(0.0, 1.0).all()

    flags = result.loc[
        valid, ["regime_is_ranging", "regime_is_transitional", "regime_is_trending"]
    ]
    assert flags.sum(axis=1).eq(1).all()


def test_regime_live_research_parity_and_research_gating_hold() -> None:
    raw = _make_regime_input_df(20)
    live = add_regime(raw, include_research_only=False)
    research = add_regime(raw, include_research_only=True)

    assert not any(col.startswith("r_") for col in live.columns)
    assert "r_regime_forward_5_return_abs" in research.columns
    assert "r_regime_transition_type" in research.columns
    non_research_cols = [col for col in research.columns if not col.startswith("r_")]
    pd.testing.assert_frame_equal(
        live[non_research_cols], research[non_research_cols], check_dtype=False
    )


def test_low_margin_raw_flip_does_not_exit_extreme_immediately() -> None:
    base = _make_stabilization_frame([0, 0, 2], [0.30, 0.30, 0.09])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    assert int(result.loc[2, "raw_regime"]) == 2
    assert int(result.loc[2, "regime"]) == 0
    assert result.loc[2, "regime_stabilized_from_raw"] == 1
    assert result.loc[2, "regime_boundary_flag"] == 1


def test_weak_transitional_signal_does_not_exit_extreme_immediately() -> None:
    base = _make_stabilization_frame([2, 2, 1], [0.25, 0.25, 0.08])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    assert [
        int(x) for x in pd.to_numeric(result["regime"], errors="coerce").tolist()
    ] == [2, 2, 2]
    assert result.loc[2, "regime_boundary_flag"] == 1


def test_direct_ranging_to_trending_jump_is_blocked_when_not_decisive() -> None:
    base = _make_stabilization_frame([0, 0, 0, 2], [0.25, 0.25, 0.25, 0.15])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    assert int(result.loc[3, "raw_regime"]) == 2
    assert int(result.loc[3, "regime"]) == 1
    assert result.loc[3, "regime_direct_extreme_jump"] == 0
    assert result.loc[3, "regime_forced_transitional"] == 1


def test_forced_transitional_needs_stronger_persistence_to_leave() -> None:
    base = _make_stabilization_frame([0, 2, 2, 2], [0.25, 0.12, 0.12, 0.12])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    observed = [
        int(x) for x in pd.to_numeric(result["regime"], errors="coerce").tolist()
    ]
    assert observed == [0, 1, 1, 2]
    assert result.loc[1, "regime_forced_transitional"] == 1
    assert result.loc[2, "regime_forced_transitional"] == 1


def test_decisive_direct_extreme_jump_remains_allowed() -> None:
    base = _make_stabilization_frame([0, 0, 0, 2], [0.25, 0.25, 0.25, 0.22])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    assert int(result.loc[3, "regime"]) == 2
    assert result.loc[3, "regime_direct_extreme_jump"] == 1
    assert result.loc[3, "regime_forced_transitional"] == 0


def test_trend_continuation_is_not_over_smoothed_into_transitional() -> None:
    base = _make_stabilization_frame([2, 2, 2, 2], [0.25, 0.05, 0.04, 0.03])
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    assert (pd.to_numeric(result["regime"], errors="coerce").astype(int) == 2).all()
    assert result["regime_forced_transitional"].sum() == 0


def test_short_oscillation_is_smoothed_by_stabilization() -> None:
    base = _make_stabilization_frame(
        [0, 2, 0, 2, 0, 2], [0.12, 0.12, 0.11, 0.11, 0.12, 0.12]
    )
    staged = _assign_raw_regime(base)
    result = _stabilize_regime(staged)

    raw_changes = int(
        pd.to_numeric(staged["raw_regime"], errors="coerce")
        .ne(pd.to_numeric(staged["raw_regime"], errors="coerce").shift(1))
        .iloc[1:]
        .sum()
    )
    final_changes = int(
        pd.to_numeric(result["regime"], errors="coerce")
        .ne(pd.to_numeric(result["regime"], errors="coerce").shift(1))
        .iloc[1:]
        .sum()
    )
    assert final_changes < raw_changes


def test_regime_validator_reports_extended_contract() -> None:
    live = add_regime(_make_regime_input_df(40), include_research_only=False)
    research = add_regime(_make_regime_input_df(40), include_research_only=True)

    result = validate_regime(
        research.tail(20),
        summary_df=research,
        live_df=live,
        research_df=research,
        synthetic_summary={"passed": 4, "total": 4},
    )
    summary = result["summary"]

    assert summary["checks"]["required_columns_present"] is True
    assert summary["checks"]["score_bounds_ok"] is True
    assert summary["checks"]["regime_values_ok"] is True
    assert summary["checks"]["one_hot_exclusive_ok"] is True
    assert summary["checks"]["transition_contract_ok"] is True
    assert summary["checks"]["persistence_bounds_ok"] is True
    assert summary["checks"]["live_research_parity_ok"] is True
    assert summary["checks"]["no_label_contamination_ok"] is True
    assert summary["current_regime_snapshot"]["regime"] in {
        "RANGING",
        "TRANSITIONAL",
        "TRENDING",
    }
    assert "boundary_flag_rate_pct" in summary["boundary_diagnostics"]
    assert "context_caution_rate_pct" in summary["boundary_diagnostics"]
    assert "caution_source_breakdown" in summary["boundary_diagnostics"]
    assert "caution_overlap_counts" in summary["boundary_diagnostics"]
    assert "confidence_bucket_counts" in summary["boundary_diagnostics"]
    assert (
        "trending_with_neutral_trend_state" in summary["extreme_misalignment_profiles"]
    )
    assert (
        "ranging_with_directional_trend_state"
        in summary["extreme_misalignment_profiles"]
    )
    assert "RANGING" in summary["transition_matrix"]
    assert "direct_extreme_jump_count" in summary["flicker_diagnostics"]
    assert "raw_regime_counts" in summary["raw_vs_stabilized_audit"]
    assert (
        summary["downstream_caution_contract"]["canonical_flag"]
        == "regime_context_caution"
    )
    assert summary["synthetic_fixture_summary"]["passed"] == 4
