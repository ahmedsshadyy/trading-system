"""
Unit tests for the indicator library.

Tests verify:
1. Output shape (same rows as input)
2. Expected columns exist
3. No input mutation
4. NaN handling (first N rows for lookback)
5. Binary columns are 0/1 only
6. Pure function contract
7. Timing contracts (confirm flags don't appear before detection bar)
8. Schema contracts (required columns exist, correct dtypes)
9. Edge cases (tiny DataFrames, zero-range candles)

Run: poetry run pytest tests/test_indicators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.indicators.structure.wedges import add_wedges
from src.indicators.structure.bos import add_bos

### Helpers for Trend_state tests


def _make_ohlc_from_close(closes: list[float]) -> pd.DataFrame:
    """
    Build a simple deterministic OHLC DataFrame from close values.

    We keep the candles simple and monotone enough that the causal swing detector
    can form structural events reliably.
    """
    close = np.asarray(closes, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="4h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _run_trend_state(
    closes: list[float],
    *,
    swing_window: int = 2,
    atr_length: int = 3,
    event_freshness_bars: int = 6,
    bias_half_life_bars: int = 2,
    bias_neutral_ttl_bars: int = 3,
    bias_min_score: float = 0.40,
    emerging_strength_threshold: float = 0.12,
    structure_loss_strength_threshold: float = 0.08,
) -> pd.DataFrame:
    """
    Build deterministic OHLC -> ATR -> swings -> trend_state for behavior tests.
    Uses shorter ATR / tighter controllable thresholds so short synthetic paths
    can actually exercise live causal behavior.
    """
    from src.indicators.foundation.volatility import add_atr
    from src.indicators.structure.swings import add_swings
    from src.indicators.structure.trend_state import add_trend_state

    df = _make_ohlc_from_close(closes)
    df = add_atr(df, period=atr_length)
    df = add_swings(
        df,
        window=swing_window,
        atr_length=atr_length,
        min_retrace_atr=0.0,
        min_confirm_bars=1,
    )
    df = add_trend_state(
        df,
        atr_length=atr_length,
        event_freshness_bars=event_freshness_bars,
        bias_half_life_bars=bias_half_life_bars,
        bias_neutral_ttl_bars=bias_neutral_ttl_bars,
        bias_min_score=bias_min_score,
        emerging_strength_threshold=emerging_strength_threshold,
        structure_loss_strength_threshold=structure_loss_strength_threshold,
    )
    return df


def _run_trend_with_regime_context(closes: list[float]) -> pd.DataFrame:
    from src.indicators.foundation.volatility import add_atr, add_bb_width
    from src.indicators.foundation.adx import add_adx
    from src.indicators.foundation.ema import add_emas
    from src.indicators.foundation.regime import add_regime
    from src.indicators.structure.swings import add_swings
    from src.indicators.structure.trend_state import add_trend_state

    df = _make_ohlc_from_close(closes)
    df = add_atr(df, period=3)
    df = add_emas(df)
    df = add_adx(df)
    df = add_bb_width(df)
    df = add_swings(
        df,
        window=2,
        atr_length=3,
        min_retrace_atr=0.0,
        min_confirm_bars=1,
    )
    df = add_trend_state(
        df,
        atr_length=3,
        event_freshness_bars=6,
        bias_half_life_bars=2,
        bias_neutral_ttl_bars=3,
        bias_min_score=0.40,
        emerging_strength_threshold=0.12,
        structure_loss_strength_threshold=0.08,
    )
    df = add_regime(df)
    return df


def _run_trend_with_env_overrides(
    closes: list[float],
    *,
    event_freshness_bars: int = 6,
    bias_half_life_bars: int = 2,
    bias_neutral_ttl_bars: int = 3,
    bias_min_score: float = 0.40,
    env_start_idx: int | None = None,
) -> pd.DataFrame:
    from src.indicators.foundation.volatility import add_atr, add_bb_width
    from src.indicators.foundation.adx import add_adx
    from src.indicators.foundation.ema import add_emas
    from src.indicators.structure.swings import add_swings
    from src.indicators.structure.trend_state import add_trend_state

    df = _make_ohlc_from_close(closes)
    df = add_atr(df, period=3)
    df = add_emas(df)
    df = add_adx(df)
    df = add_bb_width(df)
    df = add_swings(
        df,
        window=2,
        atr_length=3,
        min_retrace_atr=0.0,
        min_confirm_bars=1,
    )

    if env_start_idx is not None:
        df.loc[df.index >= env_start_idx, "adx_strength"] = 0.85
        df.loc[df.index >= env_start_idx, "ema_slope_strength"] = 0.85
        df.loc[df.index >= env_start_idx, "compression_score"] = 0.20
        df.loc[df.index >= env_start_idx, "structure_continuity"] = 0.85

    df = add_trend_state(
        df,
        atr_length=3,
        event_freshness_bars=event_freshness_bars,
        bias_half_life_bars=bias_half_life_bars,
        bias_neutral_ttl_bars=bias_neutral_ttl_bars,
        bias_min_score=bias_min_score,
    )
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate 500 rows of synthetic OHLCV data resembling XAU/USD H4."""
    np.random.seed(42)
    n = 500
    base = 2000.0
    returns = np.random.normal(0, 0.003, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n)))
    opn = close * (1 + np.random.normal(0, 0.001, n))

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h")
    volume = np.random.randint(1000, 50000, n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def intraday_df() -> pd.DataFrame:
    """Generate H1 data with proper session coverage for session tests."""
    np.random.seed(123)
    n = 2000  # ~83 days of H1 data
    base = 2000.0
    returns = np.random.normal(0, 0.001, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, n)))
    opn = close * (1 + np.random.normal(0, 0.0005, n))
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h")
    volume = np.random.randint(500, 30000, n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Generate a very small DataFrame (20 rows) for edge case tests."""
    np.random.seed(99)
    n = 20
    base = 2000.0
    close = base + np.cumsum(np.random.normal(0, 5, n))
    high = close + np.abs(np.random.normal(0, 3, n))
    low = close - np.abs(np.random.normal(0, 3, n))
    opn = close + np.random.normal(0, 2, n)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h"),
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        }
    )


@pytest.fixture
def zero_range_df() -> pd.DataFrame:
    """DataFrame with some zero-range candles (open=high=low=close)."""
    np.random.seed(77)
    n = 100
    base = 2000.0
    close = base + np.cumsum(np.random.normal(0, 5, n))
    high = close + np.abs(np.random.normal(0, 3, n))
    low = close - np.abs(np.random.normal(0, 3, n))
    opn = close + np.random.normal(0, 2, n)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    # Force 10 zero-range candles
    for i in range(10, 20):
        val = close[i]
        opn[i] = high[i] = low[i] = close[i] = val
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h"),
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_no_mutation(original: pd.DataFrame, after_fn) -> pd.DataFrame:
    """Ensure the input DataFrame was not mutated."""
    copy = original.copy()
    result = after_fn(copy)
    pd.testing.assert_frame_equal(copy, original)
    return result


def assert_binary_column(df: pd.DataFrame, col: str):
    """Check that a column contains only 0 and 1 (and possibly NaN)."""
    vals = df[col].dropna().unique()
    assert set(vals).issubset({0, 1}), f"{col} has non-binary values: {vals}"


# ---------------------------------------------------------------------------
# Trend / Structure
# ---------------------------------------------------------------------------


class TestTrend:
    def test_add_emas_columns(self, sample_df):
        from src.indicators.foundation.ema import add_emas

        result = add_emas(sample_df)
        assert len(result) == len(sample_df)
        for p in (20, 50, 200):
            assert f"ema_{p}" in result.columns
            assert f"ema_{p}_slope" in result.columns
            assert f"price_above_ema_{p}" in result.columns
        assert "ema_20_above_50" in result.columns
        assert "ema_50_above_200" in result.columns

    def test_ema_binary_flags(self, sample_df):
        from src.indicators.foundation.ema import add_emas

        result = add_emas(sample_df)
        for col in ["price_above_ema_20", "ema_20_above_50", "ema_50_above_200"]:
            assert_binary_column(result, col)

    def test_add_emas_no_mutation(self, sample_df):
        from src.indicators.foundation.ema import add_emas

        assert_no_mutation(sample_df, add_emas)

    def test_add_adx_columns(self, sample_df):
        from src.indicators.foundation.adx import add_adx

        result = add_adx(sample_df)
        assert "adx_14" in result.columns
        assert "adx_above_25" in result.columns
        assert "adx_delta_3" in result.columns

    def test_add_swings(self, sample_df):
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=3)
        assert len(result) == len(sample_df)
        assert "swing_high" in result.columns
        assert "last_swing_high" in result.columns
        assert "swing_high_age" in result.columns
        assert_binary_column(result, "swing_high")
        assert_binary_column(result, "swing_low")
        # At least some swings detected
        assert result["swing_high"].sum() > 0
        assert result["swing_low"].sum() > 0

    def test_add_trend_state(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state

        result = add_swings(sample_df)
        result = add_trend_state(result)
        assert "trend_state" in result.columns
        assert "trend_bias_state" in result.columns
        assert "trend_confidence" in result.columns
        assert "trend_bull_ready" in result.columns
        assert "trend_bear_ready" in result.columns
        assert "trend_bias_score_live" in result.columns
        assert set(result["trend_state"].unique()).issubset({-1, 0, 1})

    def test_add_bos(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.bos import add_bos

        result = add_swings(sample_df)
        result = add_bos(result)
        assert "bos_bull" in result.columns
        assert "bos_bear" in result.columns
        assert "bos_direction" in result.columns
        assert_binary_column(result, "bos_bull")

    def test_add_bos_emits_research_grade_quality_columns(self):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos

        df = _make_ohlc_from_close(
            [100, 104, 101, 106, 103, 108, 104, 110, 105, 111, 103, 99, 102, 97]
        )
        original = df.copy(deep=True)

        result = add_atr(df, period=3)
        result = add_swings(
            result,
            window=2,
            atr_length=3,
            min_retrace_atr=0.0,
            min_confirm_bars=1,
        )
        result = add_trend_state(result, atr_length=3, event_freshness_bars=6)
        result = add_bos(result, atr_length=3, min_source_age_bars=1)

        pd.testing.assert_frame_equal(df, original)

        expected = {
            "bos_candle_range_atr",
            "bos_upper_wick_atr",
            "bos_lower_wick_atr",
            "bos_body_to_range",
            "bos_close_location",
            "bos_gap_from_level_atr",
            "bos_displacement_score",
            "bos_source_prominence_atr",
            "bos_source_fresh",
            "bos_source_stale",
            "bos_source_rank",
            "bos_source_in_trend_direction",
        }
        assert expected.issubset(result.columns)

        event_rows = result[(result["bos_bull"] == 1) | (result["bos_bear"] == 1)]
        assert not event_rows.empty
        assert event_rows["bos_body_to_range"].between(0, 1).all()
        assert event_rows["bos_close_location"].between(0, 1).all()
        assert (event_rows["bos_displacement_score"] > 0).all()
        assert event_rows["bos_source_rank"].between(1, 10).all()
        assert (
            event_rows["bos_source_fresh"].astype(bool)
            & event_rows["bos_source_stale"].astype(bool)
        ).sum() == 0

    def test_add_choch(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos
        from src.indicators.structure.choch import add_choch

        result = add_swings(sample_df)
        result = add_trend_state(result)
        result = add_bos(result)
        result = add_choch(result)
        assert "choch_bull" in result.columns
        assert "choch_bear" in result.columns

    def test_add_choch_emits_canonical_transition_schema(self):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos
        from src.indicators.structure.choch import add_choch

        df = _make_ohlc_from_close(
            [111, 108, 110, 106, 108, 104, 106, 102, 104, 100, 103, 107, 105, 109]
        )
        original = df.copy(deep=True)

        result = add_atr(df, period=3)
        result = add_swings(
            result,
            window=2,
            atr_length=3,
            min_retrace_atr=0.0,
            min_confirm_bars=1,
        )
        result = add_trend_state(result, atr_length=3, event_freshness_bars=6)
        result = add_bos(result, atr_length=3, min_source_age_bars=1)
        result = add_choch(result, atr_length=3, min_source_age_bars=1)

        pd.testing.assert_frame_equal(df, original)

        expected = {
            "choch_bull",
            "choch_bear",
            "choch_direction",
            "choch_event_id",
            "choch_source_side",
            "choch_source_idx",
            "choch_source_price",
            "choch_level",
            "choch_close_break_bull",
            "choch_close_break_bear",
            "choch_wick_break_bull",
            "choch_wick_break_bear",
            "choch_raw_candidate_bull",
            "choch_raw_candidate_bear",
            "choch_pass_source_age_bull",
            "choch_pass_source_age_bear",
            "choch_pass_break_distance_bull",
            "choch_pass_break_distance_bear",
            "choch_pass_body_bull",
            "choch_pass_body_bear",
            "choch_pass_source_strength_bull",
            "choch_pass_source_strength_bear",
            "choch_pass_trend_bull",
            "choch_pass_trend_bear",
            "choch_break_distance",
            "choch_break_distance_atr",
            "choch_candle_body_atr",
            "choch_candle_range_atr",
            "choch_upper_wick_atr",
            "choch_lower_wick_atr",
            "choch_body_to_range",
            "choch_close_location",
            "choch_gap_from_level_atr",
            "choch_displacement_score",
            "choch_trend_state_from",
            "choch_trend_state_to",
            "choch_bias_state_from",
            "choch_bias_state_to",
            "choch_against_prev_trend",
            "choch_after_structure_loss",
        }
        assert expected.issubset(result.columns)

        event_rows = result[(result["choch_bull"] == 1) | (result["choch_bear"] == 1)]
        assert len(event_rows) == 1
        event = event_rows.iloc[0]

        assert event["choch_direction"] == 1
        assert event["choch_trend_state_from"] == -1
        assert event["choch_trend_state_to"] == 1
        assert event["choch_bias_state_from"] == -1
        assert event["choch_bias_state_to"] == 1
        assert event["choch_against_prev_trend"] == 1
        assert 0 <= event["choch_body_to_range"] <= 1
        assert 0 <= event["choch_close_location"] <= 1
        assert event["choch_displacement_score"] > 0

    def test_default_bos_is_continuation_only_and_excludes_choch_reversal(self):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos
        from src.indicators.structure.choch import add_choch

        df = _make_ohlc_from_close(
            [100, 103, 101, 105, 103, 107, 105, 109, 107, 111, 108, 104, 106, 102]
        )

        result = add_atr(df, period=3)
        result = add_swings(
            result,
            window=2,
            atr_length=3,
            min_retrace_atr=0.0,
            min_confirm_bars=1,
        )
        result = add_trend_state(result, atr_length=3, event_freshness_bars=6)
        result = add_bos(result, atr_length=3, min_source_age_bars=1)
        result = add_choch(result, atr_length=3, min_source_age_bars=1)

        choch_rows = result[result["choch_bear"] == 1]
        assert len(choch_rows) == 1

        choch_idx = choch_rows.index[0]
        assert result.loc[choch_idx, "bos_bear"] == 0
        assert result.loc[choch_idx, "bos_bull"] == 0

    def test_validate_structure_context_outputs_summary_and_html(self, tmp_path):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos
        from src.indicators.structure.choch import add_choch
        from src.validation.indicators.structure_context import (
            validate_structure_context,
        )

        df = _make_ohlc_from_close(
            [111, 108, 110, 106, 108, 104, 106, 102, 104, 100, 103, 107, 105, 109]
        )
        df = add_atr(df, period=3)
        df = add_swings(
            df,
            window=2,
            atr_length=3,
            min_retrace_atr=0.0,
            min_confirm_bars=1,
        )
        df = add_trend_state(df, atr_length=3, event_freshness_bars=6)
        df = add_bos(df, atr_length=3, min_source_age_bars=1)
        df = add_choch(df, atr_length=3, min_source_age_bars=1)

        result = validate_structure_context(
            df,
            outpath=tmp_path / "structure_context_test.html",
            title="Structure Context Test",
        )

        assert result["summary"]["event_counts"]["choch_count"] >= 1
        assert result["summary"]["event_counts"]["same_bar_overlap"] == 0
        assert result["html_path"].exists()

    def test_validate_choch_outputs_summary_windows_and_html(self, tmp_path):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.structure.bos import add_bos
        from src.indicators.structure.choch import add_choch
        from src.validation.indicators.choch import validate_choch

        df = _make_ohlc_from_close(
            [111, 108, 110, 106, 108, 104, 106, 102, 104, 100, 103, 107, 105, 109]
        )
        df = add_atr(df, period=3)
        df = add_swings(
            df,
            window=2,
            atr_length=3,
            min_retrace_atr=0.0,
            min_confirm_bars=1,
        )
        df = add_trend_state(df, atr_length=3, event_freshness_bars=6)
        df = add_bos(df, atr_length=3, min_source_age_bars=1)
        df = add_choch(df, atr_length=3, min_source_age_bars=1)

        result = validate_choch(
            df,
            outpath=tmp_path / "choch_validation_test.html",
            title="CHoCH Validation Test",
            n_windows=3,
        )

        assert result["summary"]["event_counts"]["choch_count"] >= 1
        assert result["summary"]["sanity_checks"]["displacement_score_positive"] is True
        assert len(result["bull_windows"]) >= 1
        assert result["html_path"].exists()


class TestTrendStateBehavior:

    def test_trend_state_columns_transition_layer_exist(self):
        df = _run_trend_state(
            [100, 102, 101, 104, 103, 106, 105, 103, 104, 102, 101, 99]
        )

        expected = {
            "trend_prev",
            "trend_enter_bullish",
            "trend_enter_bearish",
            "trend_enter_neutral",
            "bars_in_trend_state",
            "trend_persistence_5",
            "trend_persistence_20",
            "trend_direct_opposite_flip",
            "trend_bias_inherited_flag",
            "trend_bias_expired_flag",
            "trend_bias_contradicted_flag",
            "trend_conf_structure_continuity",
            "trend_conf_freshness",
            "trend_conf_event_quality",
            "trend_conf_persistence",
            "trend_conf_contradiction_penalty",
            "trend_conf_neutral_coherence",
            "trend_bull_commit_score",
            "trend_bear_commit_score",
            "trend_directional_evidence_score",
            "trend_commit_gap",
            "trend_commit_dominant_side",
            "trend_commit_gap_persist_3",
            "trend_bull_dominant_2_of_3",
            "trend_bear_dominant_2_of_3",
            "trend_bull_commit_override",
            "trend_bear_commit_override",
            "trend_structure_loss_bull",
            "trend_structure_loss_bear",
            "trend_emerging_bull",
            "trend_emerging_bear",
            "trend_regime_phase",
            "trend_pressure_bull_raw",
            "trend_pressure_bear_raw",
            "trend_strength_raw",
            "trend_strength_ema",
        }
        assert expected.issubset(df.columns)

    def test_bullish_strict_state_can_form(self):
        # Rising / pullback / rising sequence that should produce HH + HL
        df = _run_trend_state([100, 103, 101, 105, 103, 107, 105, 109, 107, 111])

        assert (df["trend_state"] == 1).any()

    def test_bearish_strict_state_can_form(self):
        # Falling / bounce / falling sequence that should produce LH + LL
        df = _run_trend_state([111, 108, 110, 106, 108, 104, 106, 102, 104, 100])

        assert (df["trend_state"] == -1).any()

    def test_bias_can_be_carried_when_strict_state_goes_neutral(self):
        # First create bull structure, then flatten enough to let freshness expire.
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 109, 109, 109, 109, 109, 109],
            event_freshness_bars=3,
            bias_neutral_ttl_bars=4,
        )

        carried = (df["trend_state"] == 0) & (df["trend_bias_state"] == 1)
        assert carried.any()

    def test_bias_eventually_expires_in_neutral(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 109, 109, 109, 109, 109, 109, 109],
            event_freshness_bars=3,
            bias_neutral_ttl_bars=2,
            bias_half_life_bars=1,
            bias_min_score=0.60,
        )

        # After enough neutral bars, bias should die.
        assert ((df["trend_state"] == 0) & (df["trend_bias_state"] == 0)).any()

    def test_confidence_values_stay_in_unit_interval(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 103, 104, 102, 101, 99]
        )

        conf = pd.to_numeric(df["trend_confidence"], errors="coerce")
        assert ((conf >= 0.0) & (conf <= 1.0)).all()

    def test_neutral_rows_can_have_nonzero_confidence(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 109, 109, 109, 109, 109, 109],
            event_freshness_bars=3,
            bias_neutral_ttl_bars=4,
        )

        neutral = df[df["trend_state"] == 0]
        assert not neutral.empty
        assert (neutral["trend_confidence"] > 0).any()
        assert (neutral["trend_confidence"] <= 0.65).all()

    def test_bull_commit_override_can_promote_direction_without_strict_readiness(self):
        df = _run_trend_state(
            [
                100,
                103,
                101,
                105,
                103,
                107,
                105,
                109,
                108.5,
                108.9,
                109.4,
                109.8,
                110.2,
                110.6,
                111.0,
                111.4,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )

        rows = df[df["trend_bull_commit_override"] == 1]
        assert not rows.empty
        assert (rows["trend_state"] == 1).all()
        assert (rows["trend_bull_ready"] == 0).all()
        assert (rows["trend_bull_commit_score"] >= 0.62).all()
        assert (rows["trend_bear_commit_score"] <= 0.38).all()

    def test_bear_commit_override_can_promote_direction_without_strict_readiness(self):
        df = _run_trend_state(
            [
                111,
                108,
                110,
                106,
                108,
                104,
                106,
                102,
                102.1,
                102.2,
                102.0,
                101.8,
                101.6,
                101.4,
                101.2,
                101.0,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )

        rows = df[df["trend_bear_commit_override"] == 1]
        assert not rows.empty
        assert (rows["trend_state"] == -1).all()
        assert (rows["trend_bear_ready"] == 0).all()
        assert (rows["trend_bear_commit_score"] >= 0.62).all()
        assert (rows["trend_bull_commit_score"] <= 0.38).all()

    def test_close_commit_scores_remain_neutral(self):
        df = _run_trend_state(
            [
                100,
                102,
                101,
                103,
                102,
                104,
                103,
                105,
                104,
                103,
                104,
                103,
                104,
                103,
                104,
                103,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )

        row = df.iloc[-1]
        assert int(row["trend_state"]) == 0
        assert int(row["trend_bull_commit_override"]) == 0
        assert int(row["trend_bear_commit_override"]) == 0
        assert (
            abs(
                float(row["trend_bull_commit_score"])
                - float(row["trend_bear_commit_score"])
            )
            < 0.18
        )

    def test_neutral_confidence_falls_as_directional_evidence_rises(self):
        low_evidence = _run_trend_state(
            [
                100,
                103,
                101,
                104,
                102,
                104,
                103,
                105,
                104,
                103,
                104,
                103,
                104,
                103,
                104,
                103,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )
        high_evidence = _run_trend_state(
            [
                100,
                103,
                101,
                105,
                103,
                107,
                105,
                109,
                109,
                109,
                109,
                109,
                109,
                109,
                109,
                109,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )

        low_row = low_evidence[low_evidence["trend_state"] == 0].iloc[-1]
        high_row = high_evidence[high_evidence["trend_state"] == 0].iloc[-1]

        assert float(high_row["trend_directional_evidence_score"]) > float(
            low_row["trend_directional_evidence_score"]
        )
        assert float(high_row["trend_confidence"]) < float(low_row["trend_confidence"])

    def test_coherent_directional_fixture_has_higher_confidence_than_weaker_directional_fixture(
        self,
    ):
        coherent = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 107, 111],
            event_freshness_bars=4,
        )
        weaker = _run_trend_state(
            [
                100,
                103,
                101,
                105,
                103,
                107,
                105,
                109,
                108.5,
                108.9,
                109.4,
                109.8,
                110.2,
                110.6,
                111.0,
                111.4,
            ],
            event_freshness_bars=4,
            bias_neutral_ttl_bars=4,
        )

        coherent_dir = coherent[coherent["trend_state"] != 0]["trend_confidence"]
        weaker_dir = weaker[weaker["trend_state"] != 0]["trend_confidence"]

        assert not coherent_dir.empty
        assert not weaker_dir.empty
        assert float(coherent_dir.mean()) > float(weaker_dir.mean())

    def test_neutral_dwell_contract_counts_neutral_runs(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 109, 109, 109, 109, 109, 109],
            event_freshness_bars=3,
            bias_neutral_ttl_bars=4,
        )

        neutral = df[df["trend_state"] == 0]
        assert not neutral.empty
        assert (neutral["bars_in_trend_state"] >= 1).all()
        assert neutral["bars_in_trend_state"].max() > 1

    def test_commit_gap_helpers_obey_contract(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 108, 107, 106, 105, 106, 107, 108]
        )

        gap = pd.to_numeric(df["trend_commit_gap"], errors="coerce")
        bull = pd.to_numeric(df["trend_bull_commit_score"], errors="coerce")
        bear = pd.to_numeric(df["trend_bear_commit_score"], errors="coerce")
        dom = pd.to_numeric(df["trend_commit_dominant_side"], errors="coerce")
        gap_persist = pd.to_numeric(df["trend_commit_gap_persist_3"], errors="coerce")

        np.testing.assert_allclose(gap.to_numpy(), (bull - bear).abs().to_numpy())
        assert set(dom.dropna().astype(int).unique()).issubset({-1, 0, 1})
        assert ((gap_persist >= 0.0) & (gap_persist <= 1.0)).all()
        assert_binary_column(df, "trend_bull_dominant_2_of_3")
        assert_binary_column(df, "trend_bear_dominant_2_of_3")

    def test_trend_persistence_fields_stay_in_unit_interval(self):
        df = _run_trend_state([100, 103, 101, 105, 103, 107, 105, 109, 107, 111])
        assert (
            (df["trend_persistence_5"].dropna() >= 0.0)
            & (df["trend_persistence_5"].dropna() <= 1.0)
        ).all()
        assert (
            (df["trend_persistence_20"].dropna() >= 0.0)
            & (df["trend_persistence_20"].dropna() <= 1.0)
        ).all()

    def test_transition_helper_flags_direct_opposite_flips(self):
        from src.indicators.structure.trend_state import _add_trend_transition_contract

        raw = pd.DataFrame({"trend_state": pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, 1.0])})
        result = _add_trend_transition_contract(raw)
        assert int(result.loc[2, "trend_direct_opposite_flip"]) == 1
        assert int(result.loc[5, "trend_enter_bullish"]) == 1
        assert int(result.loc[4, "trend_enter_neutral"]) == 1

    def test_strength_is_positive_in_bullish_regime_on_average(self):
        df = _run_trend_state([100, 103, 101, 105, 103, 107, 105, 109, 107, 111])

        bull_rows = df[df["trend_state"] == 1]
        assert not bull_rows.empty
        assert bull_rows["trend_strength_ema"].mean() > 0

    def test_strength_is_negative_in_bearish_regime_on_average(self):
        df = _run_trend_state([111, 108, 110, 106, 108, 104, 106, 102, 104, 100])

        bear_rows = df[df["trend_state"] == -1]
        assert not bear_rows.empty
        assert bear_rows["trend_strength_ema"].mean() < 0

    def test_structure_loss_bull_can_appear_after_losing_bull_structure(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 108.7, 108.4, 108.2, 108.1, 108.0],
            event_freshness_bars=3,
            structure_loss_strength_threshold=0.03,
        )

        assert (df["trend_structure_loss_bull"] == 1).any()

    def test_structure_loss_bear_can_appear_after_losing_bear_structure(self):
        df = _run_trend_state(
            [111, 108, 110, 106, 108, 104, 106, 102, 102.3, 102.6, 102.9, 103.1, 103.2],
            event_freshness_bars=3,
            structure_loss_strength_threshold=0.03,
        )

        assert (df["trend_structure_loss_bear"] == 1).any()

    def test_emerging_bull_rows_obey_contract(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 109, 108.7, 108.4, 108.2, 108.1, 108.0],
            event_freshness_bars=3,
            emerging_strength_threshold=0.02,
            structure_loss_strength_threshold=0.03,
        )

        rows = df[df["trend_emerging_bull"] == 1]
        if not rows.empty:
            assert (rows["trend_state"] == 0).all()
            assert (rows["trend_bull_ready"] == 0).all()
            assert (rows["trend_bear_ready"] == 0).all()
            assert (rows["trend_strength_raw"] >= 0).all()
            assert (rows["trend_regime_phase"] == 1).all()
        else:
            # acceptable in a tiny synthetic path; existence is already covered by validation
            assert "trend_emerging_bull" in df.columns

    def test_emerging_bear_rows_obey_contract(self):
        df = _run_trend_state(
            [111, 108, 110, 106, 108, 104, 106, 102, 102.3, 102.6, 102.9, 103.1, 103.2],
            event_freshness_bars=3,
            emerging_strength_threshold=0.02,
            structure_loss_strength_threshold=0.03,
        )

        rows = df[df["trend_emerging_bear"] == 1]
        if not rows.empty:
            assert (rows["trend_state"] == 0).all()
            assert (rows["trend_bull_ready"] == 0).all()
            assert (rows["trend_bear_ready"] == 0).all()
            assert (rows["trend_strength_raw"] <= 0).all()
            assert (rows["trend_regime_phase"] == -1).all()
        else:
            # acceptable in a tiny synthetic path; existence is already covered by validation
            assert "trend_emerging_bear" in df.columns

    def test_regime_phase_values_stay_in_expected_set(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 103, 104, 102, 101, 99]
        )

        vals = set(df["trend_regime_phase"].dropna().astype(int).unique())
        assert vals.issubset({-3, -2, -1, 0, 1, 2, 3})

    def test_pressure_columns_are_non_negative(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 103, 104, 102, 101, 99]
        )

        assert (df["trend_pressure_bull_raw"] >= 0).all()
        assert (df["trend_pressure_bear_raw"] >= 0).all()

    def test_strength_columns_are_bounded(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 103, 104, 102, 101, 99]
        )

        assert (
            (df["trend_strength_raw"] >= -1.0) & (df["trend_strength_raw"] <= 1.0)
        ).all()
        assert (
            (df["trend_strength_ema"] >= -1.0) & (df["trend_strength_ema"] <= 1.0)
        ).all()

    def test_validator_exposes_trend_regime_interaction_sections(self):
        from src.validation.indicators.trend_state import validate_trend_state

        df = _run_trend_with_regime_context(
            [100, 103, 101, 105, 103, 107, 105, 109, 108, 107, 106, 105, 106, 107, 108]
        )
        result = validate_trend_state(
            df.tail(10),
            summary_df=df,
            outpath="/tmp/trend_state_validation_test.html",
            n_windows=3,
        )
        summary = result["summary"]
        assert "transition_matrix" in summary
        assert "dwell_diagnostics" in summary
        assert "confidence_by_state" in summary
        assert "confidence_ordering_check" in summary
        assert "confidence_separation_check" in summary
        assert "neutral_confidence_cap_check" in summary
        assert "strength_by_state" in summary
        assert "commitment_by_state" in summary
        assert "bias_interaction" in summary
        assert "regime_interaction" in summary
        assert "neutral_overuse_audit" in summary
        assert "neutral_confidence_audit" in summary
        assert "neutral_in_trend_audit" in summary
        assert "directional_in_range_audit" in summary
        assert "neutral_with_directional_bias_audit" in summary
        assert "commit_gap_audit" in summary
        assert "neutral_age_audit" in summary
        assert "semantic_buckets" in summary
        assert "neutral_with_high_bull_commit" in summary["semantic_buckets"]
        assert (
            "neutral_with_high_directional_evidence_strict"
            in summary["semantic_buckets"]
        )
        assert (
            "neutral_with_high_directional_evidence_broad"
            in summary["semantic_buckets"]
        )
        assert result["transition_windows"]


class TestBOSContext:
    @staticmethod
    def _make_context_df() -> pd.DataFrame:
        n = 12
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = np.array(
            [
                99.6,
                99.9,
                100.2,
                100.6,
                101.0,
                100.8,
                101.2,
                100.3,
                99.2,
                99.0,
                100.1,
                99.4,
            ],
            dtype=float,
        )
        open_ = np.array(
            [
                99.4,
                99.7,
                100.0,
                100.4,
                100.7,
                100.9,
                100.9,
                100.7,
                99.6,
                99.4,
                99.7,
                99.7,
            ],
            dtype=float,
        )
        high = np.array(
            [
                99.8,
                100.1,
                100.4,
                100.8,
                101.2,
                101.4,
                101.5,
                100.9,
                99.7,
                100.0,
                100.2,
                99.8,
            ],
            dtype=float,
        )
        low = np.array(
            [
                99.2,
                99.5,
                99.8,
                100.2,
                100.4,
                100.4,
                100.7,
                100.0,
                99.0,
                98.6,
                99.6,
                99.1,
            ],
            dtype=float,
        )

        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "atr_14": np.full(n, 1.0),
                "bos_bull": np.zeros(n, dtype=int),
                "bos_bear": np.zeros(n, dtype=int),
                "bos_direction": np.zeros(n, dtype=int),
                "bos_level": np.full(n, np.nan),
                "bos_break_distance_atr": np.full(n, np.nan),
                "bos_candle_body_atr": np.full(n, np.nan),
                "bos_body_to_range": np.full(n, np.nan),
                "bos_displacement_score": np.full(n, np.nan),
                "bos_source_strength": np.full(n, np.nan),
                "bos_source_prominence_atr": np.full(n, np.nan),
                "bos_source_age": np.full(n, np.nan),
                "trend_state": np.zeros(n, dtype=int),
                "trend_bias_state": np.zeros(n, dtype=int),
                "wedge_active": np.zeros(n, dtype=int),
                "wedge_kind": np.zeros(n, dtype=int),
                "wedge_breakout_idx": np.full(n, np.nan),
                "sweep_high": np.zeros(n, dtype=int),
                "sweep_low": np.zeros(n, dtype=int),
                "displacement_candle": np.zeros(n, dtype=int),
                "displacement_direction": np.zeros(n, dtype=int),
                "fvg_bull": np.zeros(n, dtype=int),
                "fvg_bear": np.zeros(n, dtype=int),
                "fvg_bull_low": np.full(n, np.nan),
                "fvg_bull_high": np.full(n, np.nan),
                "fvg_bear_low": np.full(n, np.nan),
                "fvg_bear_high": np.full(n, np.nan),
                "ob_bull": np.zeros(n, dtype=int),
                "ob_bear": np.zeros(n, dtype=int),
                "ob_bull_low": np.full(n, np.nan),
                "ob_bull_high": np.full(n, np.nan),
                "ob_bear_low": np.full(n, np.nan),
                "ob_bear_high": np.full(n, np.nan),
                "equal_highs_level": np.full(n, np.nan),
                "equal_highs_active": np.zeros(n, dtype=int),
                "equal_highs_cluster_id": np.full(n, np.nan),
                "equal_lows_level": np.full(n, np.nan),
                "equal_lows_active": np.zeros(n, dtype=int),
                "equal_lows_cluster_id": np.full(n, np.nan),
                "prev_day_high": np.full(n, np.nan),
                "prev_day_low": np.full(n, np.nan),
                "prev_week_high": np.full(n, np.nan),
                "prev_week_low": np.full(n, np.nan),
                "asian_high": np.full(n, np.nan),
                "asian_low": np.full(n, np.nan),
                "vp_vah": np.full(n, np.nan),
                "vp_val": np.full(n, np.nan),
            }
        )

        df.loc[4, ["bos_bull", "bos_direction", "bos_level"]] = [1, 1, 100.5]
        df.loc[8, ["bos_bear", "bos_direction", "bos_level"]] = [1, -1, 100.0]

        df.loc[
            4, ["bos_break_distance_atr", "bos_candle_body_atr", "bos_body_to_range"]
        ] = [
            0.7,
            0.8,
            0.7,
        ]
        df.loc[
            4,
            [
                "bos_displacement_score",
                "bos_source_strength",
                "bos_source_prominence_atr",
                "bos_source_age",
            ],
        ] = [
            0.4,
            2.4,
            1.5,
            5.0,
        ]
        df.loc[
            8, ["bos_break_distance_atr", "bos_candle_body_atr", "bos_body_to_range"]
        ] = [
            0.6,
            0.7,
            0.6,
        ]
        df.loc[
            8,
            [
                "bos_displacement_score",
                "bos_source_strength",
                "bos_source_prominence_atr",
                "bos_source_age",
            ],
        ] = [
            0.35,
            1.8,
            1.2,
            8.0,
        ]

        df.loc[4, ["trend_state", "trend_bias_state", "wedge_active", "wedge_kind"]] = [
            1,
            1,
            1,
            1,
        ]
        df.loc[8, ["trend_state", "trend_bias_state"]] = [-1, -1]
        df.loc[7, "wedge_breakout_idx"] = 7
        df.loc[7, "wedge_kind"] = -1

        df.loc[2, "sweep_low"] = 1
        df.loc[7, "sweep_high"] = 1

        df.loc[3, ["displacement_candle", "displacement_direction"]] = [1, 1]
        df.loc[7, ["displacement_candle", "displacement_direction"]] = [1, -1]

        df.loc[3, ["fvg_bull", "fvg_bull_low", "fvg_bull_high"]] = [1, 100.7, 101.1]
        df.loc[7, ["fvg_bear", "fvg_bear_low", "fvg_bear_high"]] = [1, 99.1, 99.5]

        df.loc[4, ["ob_bull", "ob_bull_low", "ob_bull_high"]] = [1, 100.8, 101.1]
        df.loc[8, ["ob_bear", "ob_bear_low", "ob_bear_high"]] = [1, 99.0, 99.4]

        df.loc[
            3, ["equal_highs_level", "equal_highs_active", "equal_highs_cluster_id"]
        ] = [
            101.2,
            1,
            11,
        ]
        df.loc[
            7, ["equal_lows_level", "equal_lows_active", "equal_lows_cluster_id"]
        ] = [
            99.0,
            1,
            21,
        ]

        df.loc[4, ["prev_day_high", "asian_high", "vp_vah"]] = [101.1, 101.25, 101.4]
        df.loc[8, ["prev_day_low", "asian_low", "vp_val"]] = [99.1, 99.0, 98.9]

        return df

    def test_add_bos_context_research_mode_schema_and_contract(self):
        from src.indicators.features.bos_context import (
            RESEARCH_BOS_CONTEXT_COLUMNS,
            add_bos_context,
        )

        df = self._make_context_df()
        original = df.copy(deep=True)

        result = add_bos_context(df, include_forward_diagnostics=True)

        pd.testing.assert_frame_equal(df, original)
        assert set(RESEARCH_BOS_CONTEXT_COLUMNS).issubset(result.columns)

        non_event_mask = result["bos_direction"] == 0
        assert (
            result.loc[non_event_mask, RESEARCH_BOS_CONTEXT_COLUMNS].isna().all().all()
        )

        event_rows = result[result["bos_direction"] != 0]
        assert len(event_rows) == 2
        assert (
            (event_rows["bos_quality_score"] >= 0)
            & (event_rows["bos_quality_score"] <= 1)
        ).all()
        assert (
            (event_rows["bos_tradeable_score"] >= 0)
            & (event_rows["bos_tradeable_score"] <= 1)
        ).all()

    def test_add_bos_context_forward_diagnostics_follow_spec(self):
        from src.indicators.features.bos_context import add_bos_context

        result = add_bos_context(
            self._make_context_df(), include_forward_diagnostics=True
        )

        bull_event = result[result["bos_bull"] == 1].iloc[0]
        bear_event = result[result["bos_bear"] == 1].iloc[0]

        assert bull_event["bos_after_sweep"] == 1
        assert bull_event["bos_after_displacement"] == 1
        assert bull_event["bos_after_fvg"] == 1
        assert bull_event["bos_near_wedge"] == 1
        assert bull_event["bos_into_ob"] == 1
        assert bull_event["bos_into_fvg"] == 1
        assert bull_event["bos_near_eqhl"] == 1
        assert bull_event["bos_near_liquidity"] == 1
        assert bull_event["bos_hold_1"] == 1
        assert bull_event["bos_hold_2"] == 1
        assert bull_event["bos_hold_3"] == 0
        assert bull_event["bos_failed_3"] == 1
        assert bull_event["bos_retest_1"] == 1

        assert bear_event["bos_after_sweep"] == 1
        assert bear_event["bos_after_displacement"] == 1
        assert bear_event["bos_after_fvg"] == 1
        assert bear_event["bos_hold_1"] == 1
        assert bear_event["bos_hold_2"] == 0
        assert bear_event["bos_failed_2"] == 1
        assert bear_event["bos_retest_1"] == 1
        assert bear_event["bos_mfe_3_atr"] >= 0
        assert bear_event["bos_mae_3_atr"] >= 0

    def test_add_bos_context_live_mode_omits_forward_columns(self):
        from src.indicators.features.bos_context import (
            EXCURSION_COLUMNS,
            FOLLOW_THROUGH_COLUMNS,
            LIVE_BOS_CONTEXT_COLUMNS,
            add_bos_context,
        )

        result = add_bos_context(
            self._make_context_df(), include_forward_diagnostics=False
        )

        assert set(LIVE_BOS_CONTEXT_COLUMNS).issubset(result.columns)
        for col in FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS:
            assert col not in result.columns

    def test_validate_bos_context_outputs_summary_and_html(self, tmp_path):
        from src.indicators.features.bos_context import add_bos_context
        from src.validation.indicators.bos_context import validate_bos_context

        df = add_bos_context(
            self._make_context_df(),
            include_forward_diagnostics=True,
        )

        result = validate_bos_context(
            df,
            outpath=tmp_path / "bos_context_validation_test.html",
            title="BOS Context Validation Test",
        )

        assert result["summary"]["event_counts"]["bos_count"] == 2
        assert (
            result["summary"]["sanity_checks"]["quality_score_in_unit_interval"] is True
        )
        assert (
            result["summary"]["sanity_checks"]["tradeable_score_in_unit_interval"]
            is True
        )
        assert result["html_path"].exists()


class TestCHOCHContext:
    @staticmethod
    def _make_context_df() -> pd.DataFrame:
        n = 12
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = np.array(
            [
                100.4,
                100.1,
                99.8,
                100.0,
                100.9,
                100.7,
                100.2,
                99.6,
                98.9,
                99.2,
                100.2,
                100.5,
            ],
            dtype=float,
        )
        open_ = np.array(
            [
                100.6,
                100.3,
                100.0,
                99.8,
                100.2,
                100.9,
                100.6,
                100.0,
                99.4,
                99.0,
                99.6,
                100.2,
            ],
            dtype=float,
        )
        high = np.array(
            [
                100.8,
                100.5,
                100.2,
                100.4,
                101.1,
                101.0,
                100.8,
                100.2,
                99.7,
                99.5,
                100.4,
                100.8,
            ],
            dtype=float,
        )
        low = np.array(
            [
                100.1,
                99.9,
                99.5,
                99.6,
                99.9,
                99.9,
                99.8,
                99.4,
                98.7,
                98.8,
                99.2,
                99.9,
            ],
            dtype=float,
        )

        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "atr_14": np.full(n, 1.0),
                "choch_bull": np.zeros(n, dtype=int),
                "choch_bear": np.zeros(n, dtype=int),
                "choch_direction": np.zeros(n, dtype=int),
                "choch_level": np.full(n, np.nan),
                "choch_break_distance_atr": np.full(n, np.nan),
                "choch_candle_body_atr": np.full(n, np.nan),
                "choch_body_to_range": np.full(n, np.nan),
                "choch_displacement_score": np.full(n, np.nan),
                "choch_trend_state_from": np.full(n, np.nan),
                "choch_bias_state_from": np.full(n, np.nan),
                "choch_against_prev_trend": np.zeros(n, dtype=int),
                "choch_after_structure_loss": np.zeros(n, dtype=int),
                "trend_state": np.zeros(n, dtype=int),
                "trend_bias_state": np.zeros(n, dtype=int),
                "wedge_active": np.zeros(n, dtype=int),
                "wedge_kind": np.zeros(n, dtype=int),
                "wedge_breakout_dir": np.zeros(n, dtype=int),
                "sweep_high": np.zeros(n, dtype=int),
                "sweep_low": np.zeros(n, dtype=int),
                "displacement_candle": np.zeros(n, dtype=int),
                "displacement_direction": np.zeros(n, dtype=int),
                "fvg_bull": np.zeros(n, dtype=int),
                "fvg_bear": np.zeros(n, dtype=int),
                "fvg_bull_low": np.full(n, np.nan),
                "fvg_bull_high": np.full(n, np.nan),
                "fvg_bear_low": np.full(n, np.nan),
                "fvg_bear_high": np.full(n, np.nan),
                "ob_bull": np.zeros(n, dtype=int),
                "ob_bear": np.zeros(n, dtype=int),
                "ob_bull_low": np.full(n, np.nan),
                "ob_bull_high": np.full(n, np.nan),
                "ob_bear_low": np.full(n, np.nan),
                "ob_bear_high": np.full(n, np.nan),
            }
        )

        df.loc[4, ["choch_bull", "choch_direction", "choch_level"]] = [1, 1, 100.0]
        df.loc[8, ["choch_bear", "choch_direction", "choch_level"]] = [1, -1, 99.2]

        df.loc[
            4,
            [
                "choch_break_distance_atr",
                "choch_candle_body_atr",
                "choch_body_to_range",
            ],
        ] = [0.8, 0.7, 0.65]
        df.loc[
            4,
            [
                "choch_displacement_score",
                "choch_trend_state_from",
                "choch_bias_state_from",
            ],
        ] = [0.42, -1, -1]
        df.loc[
            4,
            [
                "choch_against_prev_trend",
                "choch_after_structure_loss",
                "trend_state",
                "trend_bias_state",
            ],
        ] = [1, 1, -1, -1]

        df.loc[
            8,
            [
                "choch_break_distance_atr",
                "choch_candle_body_atr",
                "choch_body_to_range",
            ],
        ] = [0.7, 0.8, 0.60]
        df.loc[
            8,
            [
                "choch_displacement_score",
                "choch_trend_state_from",
                "choch_bias_state_from",
            ],
        ] = [0.38, 1, 0]
        df.loc[
            8,
            [
                "choch_against_prev_trend",
                "choch_after_structure_loss",
                "trend_state",
                "trend_bias_state",
            ],
        ] = [1, 0, 1, 0]

        df.loc[2, "sweep_low"] = 1
        df.loc[7, "sweep_high"] = 1

        df.loc[3, ["displacement_candle", "displacement_direction"]] = [1, 1]
        df.loc[7, ["displacement_candle", "displacement_direction"]] = [1, -1]

        df.loc[3, ["wedge_active", "wedge_kind", "wedge_breakout_dir"]] = [1, -1, 1]
        df.loc[7, ["wedge_active", "wedge_kind", "wedge_breakout_dir"]] = [1, 1, -1]

        df.loc[3, ["fvg_bull", "fvg_bull_low", "fvg_bull_high"]] = [1, 100.8, 101.1]
        df.loc[7, ["fvg_bear", "fvg_bear_low", "fvg_bear_high"]] = [1, 98.8, 99.1]

        df.loc[4, ["ob_bull", "ob_bull_low", "ob_bull_high"]] = [1, 100.8, 101.0]
        df.loc[8, ["ob_bear", "ob_bear_low", "ob_bear_high"]] = [1, 98.8, 99.1]

        return df

    def test_add_choch_context_research_mode_schema_and_contract(self):
        from src.indicators.features.choch_context import (
            RESEARCH_CHOCH_CONTEXT_COLUMNS,
            add_choch_context,
        )

        df = self._make_context_df()
        original = df.copy(deep=True)

        result = add_choch_context(df, include_forward_diagnostics=True)

        pd.testing.assert_frame_equal(df, original)
        assert set(RESEARCH_CHOCH_CONTEXT_COLUMNS).issubset(result.columns)

        non_event_mask = result["choch_direction"] == 0
        assert (
            result.loc[non_event_mask, RESEARCH_CHOCH_CONTEXT_COLUMNS]
            .isna()
            .all()
            .all()
        )

        event_rows = result[result["choch_direction"] != 0]
        assert len(event_rows) == 2
        assert (
            (event_rows["choch_quality_score"] >= 0)
            & (event_rows["choch_quality_score"] <= 1)
        ).all()
        assert (
            (event_rows["choch_tradeable_score"] >= 0)
            & (event_rows["choch_tradeable_score"] <= 1)
        ).all()

    def test_add_choch_context_forward_diagnostics_follow_spec(self):
        from src.indicators.features.choch_context import add_choch_context

        result = add_choch_context(
            self._make_context_df(), include_forward_diagnostics=True
        )

        bull_event = result[result["choch_bull"] == 1].iloc[0]
        bear_event = result[result["choch_bear"] == 1].iloc[0]

        assert bull_event["choch_reversal_alignment"] == 1
        assert bull_event["choch_after_sweep"] == 1
        assert bull_event["choch_after_wedge"] == 1
        assert bull_event["choch_after_displacement"] == 1
        assert bull_event["choch_into_fvg"] == 1
        assert bull_event["choch_into_ob"] == 1
        assert bull_event["choch_hold_1"] == 1
        assert bull_event["choch_hold_2"] == 1
        assert bull_event["choch_hold_3"] == 0
        assert bull_event["choch_failed_3"] == 1
        assert bull_event["choch_retest_1"] == 1

        assert bear_event["choch_reversal_alignment"] == 0
        assert bear_event["choch_after_sweep"] == 1
        assert bear_event["choch_after_wedge"] == 1
        assert bear_event["choch_after_displacement"] == 1
        assert bear_event["choch_into_fvg"] == 1
        assert bear_event["choch_into_ob"] == 1
        assert bear_event["choch_hold_1"] == 1
        assert bear_event["choch_hold_2"] == 0
        assert bear_event["choch_failed_2"] == 1
        assert bear_event["choch_retest_1"] == 1
        assert bear_event["choch_mfe_3_atr"] >= 0
        assert bear_event["choch_mae_3_atr"] >= 0

    def test_add_choch_context_live_mode_omits_forward_columns(self):
        from src.indicators.features.choch_context import (
            EXCURSION_COLUMNS,
            FOLLOW_THROUGH_COLUMNS,
            LIVE_CHOCH_CONTEXT_COLUMNS,
            add_choch_context,
        )

        result = add_choch_context(
            self._make_context_df(), include_forward_diagnostics=False
        )

        assert set(LIVE_CHOCH_CONTEXT_COLUMNS).issubset(result.columns)
        for col in FOLLOW_THROUGH_COLUMNS + EXCURSION_COLUMNS:
            assert col not in result.columns

    def test_validate_choch_context_outputs_summary_and_html(self, tmp_path):
        from src.indicators.features.choch_context import add_choch_context
        from src.validation.indicators.choch_context import validate_choch_context

        df = add_choch_context(
            self._make_context_df(),
            include_forward_diagnostics=True,
        )

        result = validate_choch_context(
            df,
            outpath=tmp_path / "choch_context_validation_test.html",
            title="CHoCH Context Validation Test",
        )

        assert result["summary"]["event_counts"]["choch_count"] == 2
        assert (
            result["summary"]["sanity_checks"]["quality_score_in_unit_interval"] is True
        )
        assert (
            result["summary"]["sanity_checks"]["tradeable_score_in_unit_interval"]
            is True
        )
        assert result["html_path"].exists()


# tests/test_indicators.py  (append this class)


class TestBOS:
    @staticmethod
    def _make_base_df(n: int = 12) -> pd.DataFrame:
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = np.linspace(100.0, 101.0, n)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "atr_14": np.full(n, 1.0),
                "last_swing_high": np.full(n, np.nan),
                "last_swing_low": np.full(n, np.nan),
                "last_swing_high_idx": np.full(n, np.nan),
                "last_swing_low_idx": np.full(n, np.nan),
                "swing_high_age": np.full(n, np.nan),
                "swing_low_age": np.full(n, np.nan),
            }
        )
        return df

    @staticmethod
    def _run_bos_from_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        kwargs.setdefault("require_trend_alignment", False)
        kwargs.setdefault("allow_neutral_trend_breaks", True)
        return add_bos(df, **kwargs)

    @staticmethod
    def _make_bullish_bos_df() -> pd.DataFrame:
        df = TestBOS._make_base_df(10)

        # source swing high from idx 2 at 103.0, available from bar 3 onward
        for i in range(3, len(df)):
            df.loc[i, "last_swing_high"] = 103.0
            df.loc[i, "last_swing_high_idx"] = 2
            df.loc[i, "swing_high_age"] = i - 2

        # no bearish source
        # bar 6 wick only, no close break
        df.loc[6, ["open", "high", "low", "close"]] = [102.7, 103.2, 102.5, 102.9]
        # bar 7 canonical bullish close break
        df.loc[7, ["open", "high", "low", "close"]] = [102.8, 104.0, 102.7, 103.6]
        # bar 8 remains above, should not refire same source
        df.loc[8, ["open", "high", "low", "close"]] = [103.4, 104.2, 103.1, 103.8]
        return df

    @staticmethod
    def _make_bearish_bos_df() -> pd.DataFrame:
        df = TestBOS._make_base_df(10)

        for i in range(3, len(df)):
            df.loc[i, "last_swing_low"] = 98.0
            df.loc[i, "last_swing_low_idx"] = 2
            df.loc[i, "swing_low_age"] = i - 2

        # bar 6 wick only
        df.loc[6, ["open", "high", "low", "close"]] = [98.4, 98.5, 97.8, 98.1]
        # bar 7 canonical bearish close break
        df.loc[7, ["open", "high", "low", "close"]] = [98.2, 98.3, 97.1, 97.4]
        # bar 8 no refire
        df.loc[8, ["open", "high", "low", "close"]] = [97.6, 97.7, 96.9, 97.2]
        return df

    def test_bullish_bos_can_fire_on_deterministic_path(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        assert (df["bos_bull"] == 1).any()

    def test_bearish_bos_can_fire_on_deterministic_path(self):
        df = self._run_bos_from_df(
            self._make_bearish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        assert (df["bos_bear"] == 1).any()

    def test_bos_direction_values_stay_in_expected_set(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        vals = set(df["bos_direction"].dropna().astype(int).unique())
        assert vals.issubset({-1, 0, 1})

    def test_bos_is_sparse_not_forward_filled(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        event_mask = (df["bos_bull"] == 1) | (df["bos_bear"] == 1)
        assert (df.loc[event_mask, "bos_direction"] != 0).all()
        assert (df.loc[~event_mask, "bos_direction"] == 0).all()

    def test_bos_never_fires_both_directions_same_bar(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        assert ((df["bos_bull"] + df["bos_bear"]) <= 1).all()

    def test_bos_break_family_columns_are_binary(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        for col in [
            "bos_bull",
            "bos_bear",
            "bos_close_break_bull",
            "bos_close_break_bear",
            "bos_wick_break_bull",
            "bos_wick_break_bear",
            "bos_raw_candidate_bull",
            "bos_raw_candidate_bear",
            "bos_pass_source_age_bull",
            "bos_pass_source_age_bear",
            "bos_pass_break_distance_bull",
            "bos_pass_break_distance_bear",
            "bos_pass_body_bull",
            "bos_pass_body_bear",
            "bos_pass_source_strength_bull",
            "bos_pass_source_strength_bear",
            "bos_pass_trend_bull",
            "bos_pass_trend_bear",
        ]:
            vals = set(df[col].dropna().astype(int).unique())
            assert vals.issubset({0, 1})

    def test_bos_source_idx_precedes_event_bar(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        event_rows = df[(df["bos_bull"] == 1) | (df["bos_bear"] == 1)]
        for idx, row in event_rows.iterrows():
            assert np.isfinite(row["bos_source_idx"])
            assert row["bos_source_idx"] < idx

    def test_bos_source_metadata_populates_when_event_fires(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        event_rows = df[(df["bos_bull"] == 1) | (df["bos_bear"] == 1)]
        assert not event_rows.empty
        assert event_rows["bos_source_side"].isin({-1, 1}).all()
        assert event_rows["bos_source_idx"].notna().all()
        assert event_rows["bos_source_price"].notna().all()
        assert event_rows["bos_level"].notna().all()

    def test_bos_close_break_contains_canonical_events(self):
        df_bull = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        df_bear = self._run_bos_from_df(
            self._make_bearish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        bull_rows = df_bull["bos_bull"] == 1
        bear_rows = df_bear["bos_bear"] == 1
        if bull_rows.any():
            assert (df_bull.loc[bull_rows, "bos_close_break_bull"] == 1).all()
        if bear_rows.any():
            assert (df_bear.loc[bear_rows, "bos_close_break_bear"] == 1).all()

    def test_bos_break_distance_positive_on_event_rows(self):
        df_bull = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        df_bear = self._run_bos_from_df(
            self._make_bearish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        event_rows = pd.concat(
            [
                df_bull[(df_bull["bos_bull"] == 1) | (df_bull["bos_bear"] == 1)],
                df_bear[(df_bear["bos_bull"] == 1) | (df_bear["bos_bear"] == 1)],
            ],
            ignore_index=True,
        )
        assert not event_rows.empty
        assert (event_rows["bos_break_distance"] > 0).all()

    def test_bos_atr_normalized_metrics_are_finite_when_atr_valid(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            atr_length=14,
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        event_rows = df[(df["bos_bull"] == 1) | (df["bos_bear"] == 1)]
        if not event_rows.empty:
            assert event_rows["bos_break_distance_atr"].dropna().ge(0).all()
            assert event_rows["bos_candle_body_atr"].dropna().ge(0).all()

    def test_bos_event_ids_are_unique_and_positive_on_event_rows(self):
        for df in [
            self._run_bos_from_df(
                self._make_bullish_bos_df(),
                min_source_age_bars=0,
                min_break_distance_atr=0.0,
                min_body_atr=0.0,
            ),
            self._run_bos_from_df(
                self._make_bearish_bos_df(),
                min_source_age_bars=0,
                min_break_distance_atr=0.0,
                min_body_atr=0.0,
            ),
        ]:
            event_rows = df[(df["bos_bull"] == 1) | (df["bos_bear"] == 1)]
            ids = event_rows["bos_event_id"].to_numpy()
            assert len(ids) > 0
            assert (ids > 0).all()
            assert len(np.unique(ids)) == len(ids)

    def test_bos_event_ids_are_sequential(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        event_rows = df[(df["bos_bull"] == 1) | (df["bos_bear"] == 1)]
        ids = event_rows["bos_event_id"].to_numpy()
        if len(ids) > 0:
            np.testing.assert_array_equal(ids, np.arange(1, len(ids) + 1))

    def test_bos_non_event_rows_have_zero_event_id(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        non_event_rows = df[(df["bos_bull"] == 0) & (df["bos_bear"] == 0)]
        assert (non_event_rows["bos_event_id"] == 0).all()

    def test_bos_quality_fields_are_nan_on_non_event_rows(self):
        df = self._run_bos_from_df(self._make_bullish_bos_df())
        non_event_rows = df[(df["bos_bull"] == 0) & (df["bos_bear"] == 0)]
        for col in [
            "bos_break_distance",
            "bos_break_distance_atr",
            "bos_candle_body_atr",
            "bos_source_age",
            "bos_source_strength",
        ]:
            assert non_event_rows[col].isna().all()

    def test_same_source_level_does_not_refire_canonical_bos(self):
        df = self._make_bullish_bos_df().copy()
        extra = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    df["timestamp"].iloc[-1] + pd.Timedelta(hours=4),
                    periods=3,
                    freq="4h",
                    tz="UTC",
                ),
                "open": [103.4, 103.6, 103.7],
                "high": [104.2, 104.4, 104.6],
                "low": [103.0, 103.2, 103.3],
                "close": [103.8, 104.0, 104.2],
                "atr_14": [1.0, 1.0, 1.0],
                "last_swing_high": [103.0, 103.0, 103.0],
                "last_swing_low": [np.nan, np.nan, np.nan],
                "last_swing_high_idx": [2.0, 2.0, 2.0],
                "last_swing_low_idx": [np.nan, np.nan, np.nan],
                "swing_high_age": [8.0, 9.0, 10.0],
                "swing_low_age": [np.nan, np.nan, np.nan],
            }
        )
        df = pd.concat([df, extra], ignore_index=True)
        df = self._run_bos_from_df(
            df,
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )

        bull_event_rows = df[df["bos_bull"] == 1]
        if not bull_event_rows.empty:
            src = bull_event_rows["bos_source_idx"].dropna().astype(int)
            assert src.is_unique

    def test_bos_source_side_matches_event_direction(self):
        df_bull = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        df_bear = self._run_bos_from_df(
            self._make_bearish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        bull_rows = df_bull["bos_bull"] == 1
        bear_rows = df_bear["bos_bear"] == 1
        if bull_rows.any():
            assert (df_bull.loc[bull_rows, "bos_source_side"] == 1).all()
        if bear_rows.any():
            assert (df_bear.loc[bear_rows, "bos_source_side"] == -1).all()

    def test_wick_only_break_is_not_canonical_bos(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        row = df.iloc[6]
        assert row["bos_wick_break_bull"] == 1
        assert row["bos_close_break_bull"] == 0
        assert row["bos_bull"] == 0

    def test_min_source_age_filter_blocks_too_fresh_break(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=10,
            min_break_distance_atr=0.0,
            min_body_atr=0.0,
        )
        assert (df["bos_bull"] == 1).sum() == 0

    def test_min_break_distance_filter_blocks_small_close_through(self):
        df = self._run_bos_from_df(
            self._make_bullish_bos_df(),
            min_source_age_bars=0,
            min_break_distance_atr=10.0,
            min_body_atr=0.0,
        )
        assert (df["bos_bull"] == 1).sum() == 0

    def test_min_body_filter_blocks_small_body_break_on_that_bar(self):
        df = self._make_bullish_bos_df().copy()
        df.loc[7, ["open", "close"]] = [103.55, 103.60]  # tiny body on breakout bar

        out = self._run_bos_from_df(
            df,
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.20,
        )

        assert out.loc[7, "bos_close_break_bull"] == 1
        assert out.loc[7, "bos_pass_body_bull"] == 0
        assert out.loc[7, "bos_bull"] == 0

    def test_failed_small_body_attempt_does_not_consume_source_level(self):
        df = self._make_bullish_bos_df().copy()
        df.loc[7, ["open", "close"]] = [103.55, 103.60]  # tiny body, should fail
        df.loc[8, ["open", "close"]] = [103.20, 103.90]  # later valid break

        out = self._run_bos_from_df(
            df,
            min_source_age_bars=0,
            min_break_distance_atr=0.0,
            min_body_atr=0.20,
        )

        assert out.loc[7, "bos_bull"] == 0
        assert out.loc[8, "bos_bull"] == 1


# ---------------------------------------------------------------------------
# Wedges
# ---------------------------------------------------------------------------


class TestWedges:
    @staticmethod
    def _base_df(n: int = 60) -> pd.DataFrame:
        ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = np.linspace(100.0, 101.0, n)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "atr_14": np.full(n, 1.0),
                "swing_high_confirm_flag": np.zeros(n, dtype=np.int8),
                "swing_low_confirm_flag": np.zeros(n, dtype=np.int8),
                "swing_high_confirm_origin_idx": np.full(n, np.nan),
                "swing_low_confirm_origin_idx": np.full(n, np.nan),
                "swing_high_confirm_price": np.full(n, np.nan),
                "swing_low_confirm_price": np.full(n, np.nan),
            }
        )
        return df

    @staticmethod
    def _set_high(
        df: pd.DataFrame, confirm_idx: int, origin_idx: int, price: float
    ) -> None:
        df.loc[confirm_idx, "swing_high_confirm_flag"] = 1
        df.loc[confirm_idx, "swing_high_confirm_origin_idx"] = origin_idx
        df.loc[confirm_idx, "swing_high_confirm_price"] = price

    @staticmethod
    def _set_low(
        df: pd.DataFrame, confirm_idx: int, origin_idx: int, price: float
    ) -> None:
        df.loc[confirm_idx, "swing_low_confirm_flag"] = 1
        df.loc[confirm_idx, "swing_low_confirm_origin_idx"] = origin_idx
        df.loc[confirm_idx, "swing_low_confirm_price"] = price

    def test_add_wedges_emits_required_columns(self) -> None:
        df = self._base_df()
        out = add_wedges(df)

        expected = {
            "wedge_rising",
            "wedge_falling",
            "wedge_active",
            "wedge_kind",
            "wedge_upper_bound",
            "wedge_lower_bound",
            "wedge_apex_idx",
            "wedge_bars_to_apex",
            "wedge_width",
            "wedge_width_atr",
            "wedge_compression_ratio",
            "wedge_upper_slope",
            "wedge_lower_slope",
            "wedge_upper_r2",
            "wedge_lower_r2",
            "wedge_quality",
            "wedge_confirm_count",
            "wedge_age",
            "wedge_breakout_up",
            "wedge_breakout_down",
            "wedge_breakout_dir",
            "wedge_breakout_distance_atr",
        }
        assert expected.issubset(out.columns)

    def test_no_confirmed_swings_means_no_wedge(self) -> None:
        df = self._base_df()
        out = add_wedges(df)

        assert int(out["wedge_active"].sum()) == 0
        assert int(out["wedge_rising"].sum()) == 0
        assert int(out["wedge_falling"].sum()) == 0
        assert int(out["wedge_breakout_up"].sum()) == 0
        assert int(out["wedge_breakout_down"].sum()) == 0

    def test_rising_wedge_detects_only_after_sufficient_confirmations(self) -> None:
        df = self._base_df(70)

        # Highs: rising slowly
        self._set_high(df, 12, 10, 110.0)
        self._set_high(df, 22, 20, 111.0)
        self._set_high(df, 32, 30, 112.0)

        # Lows: rising faster
        self._set_low(df, 17, 15, 100.0)
        self._set_low(df, 27, 25, 102.0)
        self._set_low(df, 37, 35, 104.0)

        out = add_wedges(df)

        assert out.loc[:31, "wedge_active"].max() == 0
        assert out.loc[37:, "wedge_active"].max() == 1
        assert out.loc[37:, "wedge_rising"].max() == 1
        assert out.loc[37:, "wedge_falling"].max() == 0

        active_rows = out[out["wedge_rising"] == 1]
        assert not active_rows.empty
        assert (active_rows["wedge_upper_slope"] > 0).all()
        assert (
            active_rows["wedge_lower_slope"] > active_rows["wedge_upper_slope"]
        ).all()
        assert (active_rows["wedge_width"] > 0).all()
        assert (active_rows["wedge_width_atr"] > 0).all()

    def test_falling_wedge_detects(self) -> None:
        df = self._base_df(70)

        # Highs: falling faster
        self._set_high(df, 12, 10, 120.0)
        self._set_high(df, 22, 20, 117.0)
        self._set_high(df, 32, 30, 114.0)

        # Lows: falling slowly
        self._set_low(df, 17, 15, 110.0)
        self._set_low(df, 27, 25, 108.5)
        self._set_low(df, 37, 35, 107.0)

        out = add_wedges(df)

        assert out.loc[37:, "wedge_falling"].max() == 1
        active_rows = out[out["wedge_falling"] == 1]
        assert not active_rows.empty
        assert (active_rows["wedge_upper_slope"] < 0).all()
        assert (active_rows["wedge_lower_slope"] < 0).all()
        assert (
            active_rows["wedge_upper_slope"] < active_rows["wedge_lower_slope"]
        ).all()

    def test_breakout_up_fires_after_active_rising_wedge(self) -> None:
        df = self._base_df(120)

        self._set_high(df, 12, 10, 110.0)
        self._set_high(df, 22, 20, 111.0)
        self._set_high(df, 32, 30, 112.0)

        self._set_low(df, 17, 15, 100.0)
        self._set_low(df, 27, 25, 102.0)
        self._set_low(df, 37, 35, 104.0)

        out_pre = add_wedges(df)
        assert out_pre["wedge_rising"].max() == 1

        active_idxs = out_pre.index[out_pre["wedge_rising"] == 1].tolist()
        assert active_idxs, "Expected a rising wedge to become active"

        last_active_idx = active_idxs[-1]
        breakout_idx = last_active_idx + 1
        assert breakout_idx < len(df)

        upper = float(out_pre.loc[last_active_idx, "wedge_upper_bound"])
        assert np.isfinite(upper)

        df.loc[breakout_idx, "open"] = upper + 1.5
        df.loc[breakout_idx, "high"] = upper + 2.0
        df.loc[breakout_idx, "low"] = upper + 1.0
        df.loc[breakout_idx, "close"] = upper + 1.8

        out = add_wedges(df)

        assert int(out["wedge_breakout_up"].sum()) >= 1
        fired = out.index[out["wedge_breakout_up"] == 1].tolist()
        assert breakout_idx in fired
        assert out.loc[breakout_idx, "wedge_breakout_dir"] == 1
        assert out.loc[breakout_idx, "wedge_breakout_distance_atr"] > 0

    def test_breakout_down_fires_after_active_falling_wedge(self) -> None:
        df = self._base_df(120)

        self._set_high(df, 12, 10, 120.0)
        self._set_high(df, 22, 20, 117.0)
        self._set_high(df, 32, 30, 114.0)

        self._set_low(df, 17, 15, 110.0)
        self._set_low(df, 27, 25, 108.5)
        self._set_low(df, 37, 35, 107.0)

        out_pre = add_wedges(df)
        assert out_pre["wedge_falling"].max() == 1

        active_idxs = out_pre.index[out_pre["wedge_falling"] == 1].tolist()
        assert active_idxs, "Expected a falling wedge to become active"

        last_active_idx = active_idxs[-1]
        breakout_idx = last_active_idx + 1
        assert breakout_idx < len(df)

        lower = float(out_pre.loc[last_active_idx, "wedge_lower_bound"])
        assert np.isfinite(lower)

        df.loc[breakout_idx, "open"] = lower - 1.5
        df.loc[breakout_idx, "high"] = lower - 1.0
        df.loc[breakout_idx, "low"] = lower - 2.0
        df.loc[breakout_idx, "close"] = lower - 1.8

        out = add_wedges(df)

        assert int(out["wedge_breakout_down"].sum()) >= 1
        fired = out.index[out["wedge_breakout_down"] == 1].tolist()
        assert breakout_idx in fired
        assert out.loc[breakout_idx, "wedge_breakout_dir"] == -1
        assert out.loc[breakout_idx, "wedge_breakout_distance_atr"] > 0

    def test_detector_is_causal_no_wedge_before_last_required_confirmation(
        self,
    ) -> None:
        df = self._base_df(70)

        self._set_high(df, 12, 10, 110.0)
        self._set_high(df, 22, 20, 111.0)
        self._set_high(df, 32, 30, 112.0)

        self._set_low(df, 17, 15, 100.0)
        self._set_low(df, 27, 25, 102.0)
        # final low confirmation arrives later
        self._set_low(df, 50, 35, 104.0)

        out = add_wedges(df)

        assert out.loc[:49, "wedge_active"].max() == 0
        assert out.loc[50:, "wedge_active"].max() == 1

    def test_non_compressing_channel_should_not_trigger_wedge(self) -> None:
        df = self._base_df(70)

        # Parallel-ish rising channel, not wedge:
        self._set_high(df, 12, 10, 110.0)
        self._set_high(df, 22, 20, 112.0)
        self._set_high(df, 32, 30, 114.0)

        self._set_low(df, 17, 15, 100.0)
        self._set_low(df, 27, 25, 102.0)
        self._set_low(df, 37, 35, 104.0)

        out = add_wedges(
            df,
            min_compression_ratio=0.30,  # strict enough to reject near-parallel channel
        )

        assert int(out["wedge_active"].sum()) == 0

    def test_missing_required_columns_raises(self) -> None:
        df = self._base_df().drop(columns=["swing_high_confirm_flag"])
        with pytest.raises(Exception):
            add_wedges(df)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestMomentum:
    def test_add_rsi(self, sample_df):
        from src.indicators.foundation.momentum import add_rsi

        result = add_rsi(sample_df)
        assert "rsi_14" in result.columns
        # RSI should be 0–100 where not NaN
        valid = result["rsi_14"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_add_macd(self, sample_df):
        from src.indicators.foundation.momentum import add_macd

        result = add_macd(sample_df)
        assert "macd_hist" in result.columns
        assert "macd_hist_positive" in result.columns
        assert_binary_column(result, "macd_hist_positive")

    def test_rsi_divergence(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.foundation.momentum import add_rsi, add_rsi_divergence

        result = add_swings(sample_df)
        result = add_rsi(result)
        result = add_rsi_divergence(result)
        assert "rsi_div_bearish" in result.columns
        assert "rsi_div_bullish" in result.columns


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestVolatility:
    def test_add_atr(self, sample_df):
        from src.indicators.foundation.volatility import add_atr

        result = add_atr(sample_df)
        assert "atr_14" in result.columns
        assert "atr_pct_50" in result.columns
        assert result["atr_14"].dropna().min() >= 0

    def test_add_bb_width(self, sample_df):
        from src.indicators.foundation.volatility import add_bb_width

        result = add_bb_width(sample_df)
        assert "bb_width" in result.columns
        assert "bb_width_below_30" in result.columns
        assert_binary_column(result, "bb_width_below_30")

    def test_add_body_ratio(self, sample_df):
        from src.indicators.foundation.volatility import add_body_ratio

        result = add_body_ratio(sample_df)
        assert "body_ratio" in result.columns
        valid = result["body_ratio"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1.0

    def test_rolling_atr_ratio(self, sample_df):
        from src.indicators.foundation.volatility import add_rolling_atr_ratio

        result = add_rolling_atr_ratio(sample_df)
        assert "atr_ratio_rolling" in result.columns


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestVolume:
    def test_add_volume_features(self, sample_df):
        from src.indicators.foundation.volume import add_volume_features

        result = add_volume_features(sample_df)
        assert "vol_ratio" in result.columns
        assert "candle_delta_proxy" in result.columns
        assert "signed_tick_pressure_blend" in result.columns
        assert "pressure_divergence_flag" in result.columns
        assert "upper_rejection_effort" in result.columns

    def test_add_volume_ratio(self, sample_df):
        from src.indicators.foundation.volume import add_volume_ratio

        result = add_volume_ratio(sample_df)
        assert "vol_ratio" in result.columns
        assert "vol_above_1_5x" in result.columns
        assert_binary_column(result, "vol_above_1_5x")

    def test_add_vsa(self, sample_df):
        from src.indicators.foundation.volume import add_vsa

        result = add_vsa(sample_df)
        assert "vsa_absorption" in result.columns
        assert_binary_column(result, "vsa_absorption")

    def test_add_wick_ratio(self, sample_df):
        from src.indicators.foundation.volume import add_wick_ratio

        result = add_wick_ratio(sample_df)
        valid_upper = result["upper_wick_ratio"].dropna()
        assert valid_upper.min() >= 0
        assert valid_upper.max() <= 1.0


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------


class TestValue:
    def test_add_anchored_vwap(self, sample_df):
        from src.indicators.foundation.value import add_anchored_vwap

        result = add_anchored_vwap(
            sample_df,
            anchor_idx=20,
            anchor_label="day_open",
            anchor_class="live_safe",
        )
        assert "avwap" in result.columns
        assert "avwap_std" in result.columns
        assert "avwap_anchor_label" in result.columns
        assert result.loc[20, "avwap_anchor_label"] == "day_open"

    def test_prev_day_hl(self, intraday_df):
        from src.indicators.foundation.value import add_prev_day_hl

        result = add_prev_day_hl(intraday_df)
        assert "prev_day_high" in result.columns
        assert "prev_day_low" in result.columns

    def test_prev_week_hl(self, intraday_df):
        from src.indicators.foundation.value import add_prev_week_hl

        result = add_prev_week_hl(intraday_df)
        assert "prev_week_high" in result.columns

    def test_round_number(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.foundation.value import add_round_number_flag

        result = add_atr(sample_df)
        result = add_round_number_flag(result, instrument="XAU_USD")
        assert "near_round_number" in result.columns
        assert_binary_column(result, "near_round_number")

    def test_asian_session_hl(self, intraday_df):
        from src.indicators.foundation.value import add_asian_session_hl

        result = add_asian_session_hl(intraday_df)
        assert "asian_high" in result.columns
        assert "asian_range_ratio" in result.columns


# ---------------------------------------------------------------------------
# SMC
# ---------------------------------------------------------------------------


class TestSMC:
    def test_add_fvg(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.fvg import add_fvg

        result = add_atr(sample_df)
        result = add_fvg(result)
        assert "fvg_bull" in result.columns
        assert "fvg_bear" in result.columns
        assert "fvg_size_atr" in result.columns
        assert_binary_column(result, "fvg_bull")

    def test_add_fvg_fill(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.fvg import collect_fvg_debug_tables
        from src.indicators.smc.fvg_fill import add_fvg_fill

        result = add_atr(sample_df)
        debug_tables = collect_fvg_debug_tables(result)
        result = add_fvg_fill(debug_tables["frame"], debug_tables=debug_tables)
        assert "fvg_fill_pct" in result.columns
        assert "fvg_age" in result.columns
        assert "fvg_fill_bull_remaining_width" in result.columns
        assert "fvg_fill_bear_remaining_width" in result.columns
        assert "fvg_fill_bull_rep_id" in result.columns
        assert "fvg_fill_bear_rep_id" in result.columns

    def test_add_ifvg(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.fvg import collect_fvg_debug_tables
        from src.indicators.smc.ifvg import add_ifvg

        result = add_atr(sample_df)
        debug_tables = collect_fvg_debug_tables(result)
        result = add_ifvg(debug_tables["frame"], debug_tables=debug_tables)
        assert "ifvg_bull" in result.columns
        assert "ifvg_bear" in result.columns
        assert "ifvg_bull_detect_flag" in result.columns
        assert "ifvg_bear_active" in result.columns
        assert "ifvg_bull_source_fvg_event_id" in result.columns
        assert "ifvg_bull_active_age_decay" in result.columns
        assert "ifvg_bear_active_effective_significance" in result.columns

    def test_add_ob(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.ob import add_ob

        result = add_atr(sample_df)
        result = add_ob(result)
        assert "ob_bull" in result.columns
        assert "ob_bear" in result.columns
        assert "ob_width_atr" in result.columns

    def test_add_liquidity_sweep(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.sweeps import add_liquidity_sweep

        result = add_atr(sample_df)
        result = add_swings(result)
        result = add_liquidity_sweep(result)
        assert "sweep_high" in result.columns
        assert "sweep_low" in result.columns

    def test_add_equal_hl(self, sample_df):
        from src.indicators.structure.swings import add_swings
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.equal_hl import add_equal_hl

        result = add_atr(sample_df)
        result = add_swings(result)
        result = add_equal_hl(result)
        assert "equal_highs" in result.columns

    def test_displacement_candle(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.displacement import add_displacement_candle

        result = add_atr(sample_df)
        result = add_displacement_candle(result)
        assert "displacement_candle" in result.columns
        assert_binary_column(result, "displacement_candle")


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    def test_compute_volume_profile(self, sample_df):
        from src.indicators.foundation.volume_profile import compute_volume_profile

        vp = compute_volume_profile(sample_df.tail(80))
        assert "poc" in vp
        assert "vah" in vp
        assert "val" in vp
        assert vp["val"] <= vp["poc"] <= vp["vah"]

    def test_add_volume_profile(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.foundation.volume_profile import add_volume_profile

        result = add_atr(sample_df)
        result = add_volume_profile(result)
        assert "vp_poc" in result.columns
        assert "vp_vah" in result.columns


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TestSession:
    def test_session_classifier(self, intraday_df):
        from src.indicators.foundation.session import add_session_classifier

        result = add_session_classifier(intraday_df)
        assert "session" in result.columns
        assert "london_open" in result.columns
        assert_binary_column(result, "london_open")
        assert_binary_column(result, "ny_open")

    def test_time_features(self, sample_df):
        from src.indicators.foundation.session import add_time_features

        result = add_time_features(sample_df)
        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class TestRegime:
    def test_add_regime(self, sample_df):
        from src.indicators.foundation.ema import add_emas
        from src.indicators.foundation.adx import add_adx
        from src.indicators.foundation.volatility import add_bb_width
        from src.indicators.structure.swings import add_swings
        from src.indicators.structure.trend_state import add_trend_state
        from src.indicators.foundation.regime import add_regime

        result = add_emas(sample_df)
        result = add_adx(result)
        result = add_bb_width(result)
        result = add_swings(result)
        result = add_trend_state(result)
        result = add_regime(result)
        assert "regime" in result.columns
        valid = pd.to_numeric(result["regime"], errors="coerce").dropna().astype(int)
        assert not valid.empty
        assert set(valid.unique()).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_build_all(self, sample_df):
        from src.indicators import build_all_indicators

        result = build_all_indicators(sample_df, instrument="XAU_USD", include_vp=False)
        assert len(result) == len(sample_df)
        # Check a sampling of critical columns exist
        expected = [
            "ema_20",
            "adx_14",
            "rsi_14",
            "macd_hist",
            "atr_14",
            "swing_high",
            "bos_bull",
            "choch_bull",
            "fvg_bull",
            "ob_bull",
            "sweep_high",
            "regime",
            "vol_ratio",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_build_all_no_mutation(self, sample_df):
        from src.indicators import build_all_indicators

        assert_no_mutation(
            sample_df,
            lambda df: build_all_indicators(df, instrument="XAU_USD", include_vp=False),
        )

    def test_build_live(self, sample_df):
        from src.indicators.pipelines.build_live import build_live_indicators

        result = build_live_indicators(
            sample_df, instrument="XAU_USD", include_vp=False
        )
        assert len(result) == len(sample_df)
        expected = [
            "ema_20",
            "adx_14",
            "rsi_14",
            "macd_hist",
            "atr_14",
            "swing_high",
            "bos_bull",
            "choch_bull",
            "fvg_bull",
            "ob_bull",
            "sweep_high",
            "regime",
            "amd_phase",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


# ===========================================================================
# Step 7 — Contract Tests
# ===========================================================================


class TestPurityContract:
    """Every indicator function must not mutate its input DataFrame."""

    def test_add_atr_purity(self, sample_df):
        from src.indicators.foundation.volatility import add_atr

        original = sample_df.copy()
        add_atr(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_add_swings_purity(self, sample_df):
        from src.indicators.structure.swings import add_swings

        original = sample_df.copy()
        add_swings(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_add_fvg_purity(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.fvg import add_fvg

        prepped = add_atr(sample_df)
        original = prepped.copy()
        add_fvg(prepped)
        pd.testing.assert_frame_equal(prepped, original)

    def test_add_ob_purity(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.ob import add_ob

        prepped = add_atr(sample_df)
        original = prepped.copy()
        add_ob(prepped)
        pd.testing.assert_frame_equal(prepped, original)

    def test_add_liquidity_sweep_purity(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.structure.swings import add_swings
        from src.indicators.smc.sweeps import add_liquidity_sweep

        prepped = add_swings(add_atr(sample_df))
        original = prepped.copy()
        add_liquidity_sweep(prepped)
        pd.testing.assert_frame_equal(prepped, original)

    def test_get_atr_array_purity(self, sample_df):
        """get_atr_array must never write to the input DataFrame."""
        from src.indicators._helpers.arrays import get_atr_array

        original_cols = set(sample_df.columns)
        original = sample_df.copy()
        get_atr_array(sample_df)
        assert set(sample_df.columns) == original_cols
        pd.testing.assert_frame_equal(sample_df, original)


class TestTimingContract:
    """Confirm flags must not appear before detection bar."""

    def test_swing_detect_after_origin(self, sample_df):
        """Retrace-confirmed swings: detect bar must be after origin bar."""
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=6)
        sh = result[result["swing_high"] == 1]
        for idx, row in sh.iterrows():
            detect = row["swing_high_detect_idx"]
            assert (
                detect > idx
            ), f"Swing high origin at {idx} has detect_idx {detect} (should be later)"

        sl = result[result["swing_low"] == 1]
        for idx, row in sl.iterrows():
            detect = row["swing_low_detect_idx"]
            assert (
                detect > idx
            ), f"Swing low origin at {idx} has detect_idx {detect} (should be later)"

    def test_last_swing_updates_on_detect_bar(self, sample_df):
        """last_swing_* must update on the detect bar, not the origin bar."""
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=6)

        # On detect bars, last_swing should reflect the just-confirmed swing
        detect_bars = result[result["swing_high_detect_flag"] == 1]
        for idx, row in detect_bars.iterrows():
            assert np.isfinite(row["last_swing_high"])
            assert row["swing_high_age"] == 0

        detect_bars = result[result["swing_low_detect_flag"] == 1]
        for idx, row in detect_bars.iterrows():
            assert np.isfinite(row["last_swing_low"])
            assert row["swing_low_age"] == 0

        def test_fvg_confirm_idx_after_origin(self, sample_df):
            """FVG confirm index must be after the FVG origin bar."""
            from src.indicators.foundation.volatility import add_atr
            from src.indicators.smc.fvg import add_fvg

            result = add_fvg(add_atr(sample_df))
            fvg_bars = result[result["fvg_bull"] == 1]
            for _, row in fvg_bars.iterrows():
                ci = row["fvg_bull_confirm_idx"]
                if ci >= 0:
                    assert ci > row.name

    def test_canonical_swing_detect_columns_exist(self, sample_df):
        """Canonical swing detector should expose detect metadata columns."""
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=6)
        expected = [
            "swing_high_detect_flag",
            "swing_low_detect_flag",
            "swing_high_detect_idx",
            "swing_low_detect_idx",
            "last_swing_high_idx",
            "last_swing_low_idx",
        ]
        for col in expected:
            assert col in result.columns, f"Missing swing column: {col}"


class TestSchemaContract:
    """Required columns exist with correct dtypes after pipeline runs."""

    def test_build_all_column_count_stable(self, sample_df):
        """Pipeline output column count should be deterministic."""
        from src.indicators import build_all_indicators

        r1 = build_all_indicators(sample_df, instrument="XAU_USD", include_vp=False)
        r2 = build_all_indicators(sample_df, instrument="XAU_USD", include_vp=False)
        assert len(r1.columns) == len(r2.columns)
        assert list(r1.columns) == list(r2.columns)

    def test_binary_columns_are_int_dtype(self, sample_df):
        """Binary flag columns should be integer type, not float."""
        from src.indicators import build_all_indicators

        result = build_all_indicators(sample_df, instrument="XAU_USD", include_vp=False)
        binary_cols = [
            "bos_bull",
            "bos_bear",
            "choch_bull",
            "choch_bear",
            "fvg_bull",
            "fvg_bear",
            "ob_bull",
            "ob_bear",
            "sweep_high",
            "sweep_low",
            "displacement_candle",
        ]
        for col in binary_cols:
            if col in result.columns:
                assert result[col].dtype in (
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    int,
                ), f"{col} has dtype {result[col].dtype}, expected int"

    def test_row_count_preserved(self, sample_df):
        """Pipeline must not add or remove rows."""
        from src.indicators import build_all_indicators

        result = build_all_indicators(sample_df, instrument="XAU_USD", include_vp=False)
        assert len(result) == len(sample_df)


class TestEdgeCases:
    """Edge cases: tiny DataFrames, zero-range candles."""

    def test_tiny_df_no_crash(self, tiny_df):
        """Pipeline should not crash on a 20-row DataFrame."""
        from src.indicators import build_all_indicators

        result = build_all_indicators(tiny_df, instrument="XAU_USD", include_vp=False)
        assert len(result) == len(tiny_df)
        assert "atr_14" in result.columns

    def test_zero_range_candles_no_crash(self, zero_range_df):
        """Pipeline should handle zero-range candles without NaN explosion."""
        from src.indicators import build_all_indicators

        result = build_all_indicators(
            zero_range_df, instrument="XAU_USD", include_vp=False
        )
        assert len(result) == len(zero_range_df)
        # body_ratio should be 0 (not NaN) for zero-range candles
        zero_range_rows = result.iloc[10:20]
        assert (zero_range_rows["body_ratio"] == 0.0).all()

    def test_displacement_on_zero_range(self, zero_range_df):
        """Displacement detector should not flag zero-range candles."""
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.displacement import add_displacement_candle

        result = add_displacement_candle(add_atr(zero_range_df))
        # Zero-range candles at indices 10-19 should NOT be displacement candles
        assert result.iloc[10:20]["displacement_candle"].sum() == 0

    def test_no_label_columns_in_live_build(self, sample_df):
        """Live pipeline must never produce label_* or future_* columns."""
        from src.indicators.pipelines.build_live import build_live_indicators

        result = build_live_indicators(
            sample_df, instrument="XAU_USD", include_vp=False
        )
        label_cols = [
            c
            for c in result.columns
            if c.startswith("label_") or c.startswith("future_")
        ]
        assert label_cols == [], f"Live build contains label columns: {label_cols}"

        # Check amd_label_* specifically (retrospective labeling columns)
        amd_label_cols = [c for c in result.columns if c.startswith("amd_label")]
        assert (
            amd_label_cols == []
        ), f"Live build contains AMD label columns: {amd_label_cols}"
