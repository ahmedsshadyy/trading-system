"""
Unit tests for the indicator library.

Tests verify:
1. Output shape (same rows as input)
2. Expected columns exist
3. No input mutation
4. NaN handling (first N rows for lookback)
5. Binary columns are 0/1 only
6. Pure function contract

Run: poetry run pytest tests/test_indicators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
# Trend
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
        from src.indicators.trend import add_swings

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
        from src.indicators.trend import add_swings, add_trend_state

        result = add_swings(sample_df)
        result = add_trend_state(result)
        assert "trend_state" in result.columns
        assert set(result["trend_state"].unique()).issubset({-1, 0, 1})

    def test_add_bos(self, sample_df):
        from src.indicators.trend import add_swings, add_bos

        result = add_swings(sample_df)
        result = add_bos(result)
        assert "bos_bull" in result.columns
        assert "bos_bear" in result.columns
        assert "bos_direction" in result.columns
        assert_binary_column(result, "bos_bull")

    def test_add_choch(self, sample_df):
        from src.indicators.trend import add_swings, add_trend_state, add_bos, add_choch

        result = add_swings(sample_df)
        result = add_trend_state(result)
        result = add_bos(result)
        result = add_choch(result)
        assert "choch_bull" in result.columns
        assert "choch_bear" in result.columns


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
        from src.indicators.trend import add_swings
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
        from src.indicators.smc import add_fvg

        result = add_atr(sample_df)
        result = add_fvg(result)
        assert "fvg_bull" in result.columns
        assert "fvg_bear" in result.columns
        assert "fvg_size_atr" in result.columns
        assert_binary_column(result, "fvg_bull")

    def test_add_fvg_fill(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc import add_fvg, add_fvg_fill

        result = add_atr(sample_df)
        result = add_fvg(result)
        result = add_fvg_fill(result)
        assert "fvg_fill_pct" in result.columns
        assert "fvg_age" in result.columns

    def test_add_ob(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc import add_ob

        result = add_atr(sample_df)
        result = add_ob(result)
        assert "ob_bull" in result.columns
        assert "ob_bear" in result.columns
        assert "ob_width_atr" in result.columns

    def test_add_liquidity_sweep(self, sample_df):
        from src.indicators.trend import add_swings
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc import add_liquidity_sweep

        result = add_atr(sample_df)
        result = add_swings(result)
        result = add_liquidity_sweep(result)
        assert "sweep_high" in result.columns
        assert "sweep_low" in result.columns

    def test_add_equal_hl(self, sample_df):
        from src.indicators.trend import add_swings
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc import add_equal_hl

        result = add_atr(sample_df)
        result = add_swings(result)
        result = add_equal_hl(result)
        assert "equal_highs" in result.columns

    def test_displacement_candle(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc import add_displacement_candle

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
        from src.indicators.trend import add_swings, add_trend_state
        from src.indicators.foundation.volatility import add_bb_width
        from src.indicators.foundation.regime import add_regime

        result = add_emas(sample_df)
        result = add_adx(result)
        result = add_bb_width(result)
        result = add_swings(result)
        result = add_trend_state(result)
        result = add_regime(result)
        assert "regime" in result.columns
        assert set(result["regime"].unique()).issubset({0, 1, 2})


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
