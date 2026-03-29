from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.pipelines.build_live import build_live_indicators
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.smc.displacement import (
    DISPLACEMENT_CANONICAL_COLUMNS,
    DISPLACEMENT_FLAG_COLUMNS,
    DISPLACEMENT_FLOAT_COLUMNS,
    DISPLACEMENT_LEGACY_COLUMNS,
    DISPLACEMENT_LIVE_SAFE_COLUMNS,
    add_displacement_candle,
)


def _make_displacement_df(
    rows: list[tuple[float, float, float, float]],
    *,
    atr_values: list[float] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="4h", tz="UTC")
    df["volume"] = np.arange(1000, 1000 + len(df), dtype=np.int64)
    if atr_values is not None:
        df["atr_14"] = np.asarray(atr_values, dtype=float)
        return df[["timestamp", "open", "high", "low", "close", "volume", "atr_14"]]
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _sample_pipeline_df(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = 2000.0
    returns = rng.normal(0.0, 0.003, n)
    close = base * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0.0, 0.002, n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0.0, 0.002, n)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 50000, size=n),
        }
    )


def test_displacement_purity_preserves_input_and_existing_schema() -> None:
    df = _make_displacement_df(
        [
            (100.0, 110.0, 99.0, 109.0),
            (110.0, 111.0, 98.0, 99.0),
        ],
        atr_values=[5.0, 5.0],
    )
    df["marker"] = np.array([7, 8], dtype=np.int64)
    original = df.copy(deep=True)

    result = add_displacement_candle(df)

    pd.testing.assert_frame_equal(df, original)
    assert result.index.equals(df.index)
    assert list(result.columns[: len(df.columns)]) == list(df.columns)
    for col in df.columns:
        assert result[col].dtype == df[col].dtype


def test_displacement_schema_contains_canonical_and_legacy_columns() -> None:
    df = _make_displacement_df([(100.0, 110.0, 99.0, 109.0)], atr_values=[5.0])

    result = add_displacement_candle(df)

    for col in DISPLACEMENT_CANONICAL_COLUMNS + DISPLACEMENT_LEGACY_COLUMNS:
        assert col in result.columns, f"Missing displacement column: {col}"


def test_displacement_dtypes_are_frozen() -> None:
    df = _make_displacement_df([(100.0, 110.0, 99.0, 109.0)], atr_values=[5.0])

    result = add_displacement_candle(df)

    for col in DISPLACEMENT_FLAG_COLUMNS + DISPLACEMENT_LEGACY_COLUMNS:
        assert result[col].dtype == np.int8, f"{col} dtype is {result[col].dtype}"
    for col in DISPLACEMENT_FLOAT_COLUMNS:
        assert np.issubdtype(
            result[col].dtype, np.floating
        ), f"{col} dtype is {result[col].dtype}"


def test_textbook_bullish_displacement_passes_with_expected_geometry() -> None:
    df = _make_displacement_df([(100.0, 112.0, 99.0, 111.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_flag"] == 1
    assert row["displacement_bull"] == 1
    assert row["displacement_bear"] == 0
    assert row["displacement_direction"] == 1
    assert np.isclose(row["displacement_body"], 11.0)
    assert np.isclose(row["displacement_range"], 13.0)
    assert np.isclose(row["displacement_upper_wick"], 1.0)
    assert np.isclose(row["displacement_lower_wick"], 1.0)
    assert np.isclose(row["displacement_body_atr"], 2.2)
    assert np.isclose(row["displacement_range_atr"], 2.6)
    assert np.isclose(row["displacement_signed_body_atr"], 2.2)
    assert np.isclose(row["displacement_body_frac"], 11.0 / 13.0)
    assert np.isclose(row["displacement_close_to_extreme_frac"], 1.0 / 13.0)
    assert np.isclose(row["displacement_opposite_wick_frac"], 1.0 / 13.0)


def test_textbook_bearish_displacement_passes_with_negative_signed_body() -> None:
    df = _make_displacement_df([(110.0, 111.0, 98.0, 99.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_flag"] == 1
    assert row["displacement_bull"] == 0
    assert row["displacement_bear"] == 1
    assert row["displacement_direction"] == -1
    assert row["displacement_signed_body_atr"] < 0
    assert np.isclose(row["displacement_close_to_extreme_frac"], 1.0 / 13.0)
    assert np.isclose(row["displacement_opposite_wick_frac"], 1.0 / 13.0)


def test_doji_fails_and_score_is_zero() -> None:
    df = _make_displacement_df([(100.0, 105.0, 95.0, 100.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_direction"] == 0
    assert row["displacement_flag"] == 0
    assert row["displacement_pass_valid_inputs"] == 0
    assert row["displacement_score"] == 0.0


def test_bad_close_location_fails_even_with_large_body() -> None:
    df = _make_displacement_df([(100.0, 114.0, 99.0, 110.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_pass_body_atr"] == 1
    assert row["displacement_pass_body_frac"] == 1
    assert row["displacement_pass_close_to_extreme"] == 0
    assert row["displacement_flag"] == 0
    assert row["displacement_score"] > 0.0


def test_large_opposite_wick_fails() -> None:
    df = _make_displacement_df([(104.0, 114.0, 99.0, 113.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_pass_body_atr"] == 1
    assert row["displacement_pass_body_frac"] == 1
    assert row["displacement_pass_close_to_extreme"] == 1
    assert row["displacement_pass_opposite_wick"] == 0
    assert row["displacement_flag"] == 0


def test_insufficient_body_fraction_fails() -> None:
    df = _make_displacement_df([(100.0, 120.0, 99.0, 111.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_pass_body_atr"] == 1
    assert row["displacement_pass_body_frac"] == 0
    assert row["displacement_flag"] == 0


def test_invalid_atr_suppresses_displacement() -> None:
    df = _make_displacement_df([(100.0, 112.0, 99.0, 111.0)], atr_values=[np.nan])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_pass_valid_inputs"] == 0
    assert row["displacement_pass_body_atr"] == 0
    assert row["displacement_flag"] == 0
    assert np.isnan(row["displacement_body_atr"])
    assert row["displacement_score"] == 0.0


def test_threshold_boundaries_are_inclusive() -> None:
    df = _make_displacement_df([(102.0, 110.0, 100.0, 108.0)], atr_values=[4.0])

    row = add_displacement_candle(df, body_atr_mult=1.5).iloc[0]

    assert np.isclose(row["displacement_body_atr"], 1.5)
    assert np.isclose(row["displacement_body_frac"], 0.6)
    assert np.isclose(row["displacement_close_to_extreme_frac"], 0.2)
    assert np.isclose(row["displacement_opposite_wick_frac"], 0.2)
    assert row["displacement_flag"] == 1


def test_just_below_threshold_fails() -> None:
    df = _make_displacement_df([(102.0, 110.0, 100.0, 108.0)], atr_values=[4.01])

    row = add_displacement_candle(df, body_atr_mult=1.5).iloc[0]

    assert row["displacement_pass_body_atr"] == 0
    assert row["displacement_flag"] == 0


def test_zero_range_bar_is_rejected_without_crashing() -> None:
    df = _make_displacement_df([(100.0, 100.0, 100.0, 100.0)], atr_values=[5.0])

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_flag"] == 0
    assert row["displacement_pass_valid_inputs"] == 0
    assert np.isnan(row["displacement_body_frac"])
    assert row["displacement_score"] == 0.0


def test_nan_ohlc_is_rejected_gracefully() -> None:
    df = _make_displacement_df([(100.0, 112.0, 99.0, 111.0)], atr_values=[5.0])
    df.loc[0, "high"] = np.nan

    row = add_displacement_candle(df).iloc[0]

    assert row["displacement_flag"] == 0
    assert row["displacement_pass_valid_inputs"] == 0
    assert row["displacement_score"] == 0.0


def test_score_is_bounded_and_orders_stronger_candles_higher() -> None:
    df = _make_displacement_df(
        [
            (100.0, 112.0, 99.0, 111.0),
            (102.0, 110.0, 100.0, 108.0),
            (100.0, 114.0, 99.0, 110.0),
        ],
        atr_values=[5.0, 4.0, 5.0],
    )

    result = add_displacement_candle(df)

    assert (
        (result["displacement_score"] >= 0.0) & (result["displacement_score"] <= 1.0)
    ).all()
    assert result.loc[0, "displacement_flag"] == 1
    assert result.loc[1, "displacement_flag"] == 1
    assert result.loc[0, "displacement_score"] > result.loc[1, "displacement_score"]
    assert result.loc[2, "displacement_flag"] == 0
    assert result.loc[2, "displacement_score"] > 0.0


def test_atr_ready_flag_tracks_warmup() -> None:
    df = _make_displacement_df(
        [
            (100.0, 112.0, 99.0, 111.0),
            (110.0, 111.0, 98.0, 99.0),
            (100.0, 114.0, 99.0, 110.0),
        ],
        atr_values=[5.0, 5.0, 5.0],
    )

    result = add_displacement_candle(df, atr_length=3)

    assert result["displacement_atr_ready"].tolist() == [0, 0, 1]


def test_live_and_research_pipelines_match_on_all_displacement_outputs() -> None:
    df = _sample_pipeline_df()

    live = build_live_indicators(df, instrument="XAU_USD", include_vp=False)
    research = build_research_indicators(df, instrument="XAU_USD", include_vp=False)

    for col in DISPLACEMENT_LIVE_SAFE_COLUMNS:
        assert col in live.columns
        assert col in research.columns
        if col in DISPLACEMENT_FLOAT_COLUMNS:
            np.testing.assert_allclose(
                live[col].to_numpy(dtype=float),
                research[col].to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(
                live[col].to_numpy(),
                research[col].to_numpy(),
            )


def test_no_label_or_future_columns_in_displacement_output() -> None:
    df = _make_displacement_df([(100.0, 112.0, 99.0, 111.0)], atr_values=[5.0])

    result = add_displacement_candle(df)

    forbidden = [
        col
        for col in result.columns
        if col.startswith("label_") or col.startswith("future_")
    ]
    assert forbidden == [], f"Forbidden forward-looking columns found: {forbidden}"


def test_stricter_thresholds_produce_fewer_or_equal_displacement_events() -> None:
    """Monotonicity: tightening any threshold must not increase event count."""
    rng = np.random.default_rng(0)
    n = 300
    close = 2000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.004, n))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(close, open_) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = np.minimum(close, open_) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(1000, 50000, size=n),
        }
    )

    base_count = int(
        add_displacement_candle(df, body_atr_mult=1.35)["displacement_flag"].sum()
    )
    stricter_body_atr = int(
        add_displacement_candle(df, body_atr_mult=1.80)["displacement_flag"].sum()
    )
    stricter_body_frac = int(
        add_displacement_candle(df, min_body_frac=0.75)["displacement_flag"].sum()
    )
    stricter_close = int(
        add_displacement_candle(df, close_extreme_frac=0.10)["displacement_flag"].sum()
    )
    stricter_opp_wick = int(
        add_displacement_candle(df, max_opposite_wick_frac=0.10)[
            "displacement_flag"
        ].sum()
    )

    assert (
        stricter_body_atr <= base_count
    ), f"Stricter body_atr_mult raised event count: {base_count} -> {stricter_body_atr}"
    assert (
        stricter_body_frac <= base_count
    ), f"Stricter min_body_frac raised event count: {base_count} -> {stricter_body_frac}"
    assert (
        stricter_close <= base_count
    ), f"Stricter close_extreme_frac raised event count: {base_count} -> {stricter_close}"
    assert (
        stricter_opp_wick <= base_count
    ), f"Stricter max_opposite_wick_frac raised event count: {base_count} -> {stricter_opp_wick}"
