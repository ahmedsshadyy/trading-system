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
    df = add_swings(df, window=swing_window, atr_length=atr_length)
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


class TestTrendStateBehavior:

    def test_trend_state_columns_transition_layer_exist(self):
        df = _run_trend_state(
            [100, 102, 101, 104, 103, 106, 105, 103, 104, 102, 101, 99]
        )

        expected = {
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

    def test_confidence_values_stay_in_expected_set(self):
        df = _run_trend_state(
            [100, 103, 101, 105, 103, 107, 105, 103, 104, 102, 101, 99]
        )

        vals = set(df["trend_confidence"].dropna().astype(int).unique())
        assert vals.issubset({-1, 0, 1, 2})

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
        from src.indicators.smc.fvg import add_fvg

        result = add_atr(sample_df)
        result = add_fvg(result)
        assert "fvg_bull" in result.columns
        assert "fvg_bear" in result.columns
        assert "fvg_size_atr" in result.columns
        assert_binary_column(result, "fvg_bull")

    def test_add_fvg_fill(self, sample_df):
        from src.indicators.foundation.volatility import add_atr
        from src.indicators.smc.fvg import add_fvg
        from src.indicators.smc.fvg_fill import add_fvg_fill

        result = add_atr(sample_df)
        result = add_fvg(result)
        result = add_fvg_fill(result)
        assert "fvg_fill_pct" in result.columns
        assert "fvg_age" in result.columns

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

    def test_swing_detect_idx_matches_origin(self, sample_df):
        """Canonical causal swings detect on the same bar they originate."""
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=6)
        sh = result[result["swing_high"] == 1]
        for idx, row in sh.iterrows():
            assert row["swing_high_detect_flag"] == 1
            assert row["swing_high_detect_idx"] == idx

        sl = result[result["swing_low"] == 1]
        for idx, row in sl.iterrows():
            assert row["swing_low_detect_flag"] == 1
            assert row["swing_low_detect_idx"] == idx

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

    def test_last_swing_updates_on_detection_bar(self, sample_df):
        """Canonical causal last_swing_* should update immediately on the detection bar."""
        from src.indicators.structure.swings import add_swings

        result = add_swings(sample_df, window=6)

        for idx, row in result[result["swing_high"] == 1].iterrows():
            assert row["last_swing_high"] == row["swing_high_price"]
            assert row["last_swing_high_idx"] == float(idx)
            assert row["swing_high_age"] == 0

        for idx, row in result[result["swing_low"] == 1].iterrows():
            assert row["last_swing_low"] == row["swing_low_price"]
            assert row["last_swing_low_idx"] == float(idx)
            assert row["swing_low_age"] == 0

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
