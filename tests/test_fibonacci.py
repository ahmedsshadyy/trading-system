"""
tests/test_fibonacci.py

Unit tests for the canonical causal Fibonacci level engine.

Tests cover:
1. Causality: no fib structures active before both anchors are confirmed
2. Bull structure detection on swing high confirmation
3. Bear structure detection on swing low confirmation
4. Fib level price arithmetic correctness
5. Hard expiry after max_age_bars
6. Bull structure invalidation on close below anchor_low
7. Bear structure invalidation on close above anchor_high
8. Nearest level projection correctness
9. Multiple concurrent structures: best quality is selected
10. Pure function: input df is never mutated
11. Touch zone detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.foundation.fibonacci import (
    FIB_RATIOS,
    FIB_RATIO_LABELS,
    add_fibonacci_levels,
    FIB_CANONICAL_COLUMNS,
)

# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def _make_fib_df(
    rows: list[tuple[float, float, float, float]],
    *,
    sh_conf: list[int] | None = None,
    sl_conf: list[int] | None = None,
    sh_price: list[float | None] | None = None,
    sl_price: list[float | None] | None = None,
    sh_origin: list[int | None] | None = None,
    sl_origin: list[int | None] | None = None,
    sh_strength: list[float | None] | None = None,
    sl_strength: list[float | None] | None = None,
    atr_values: list[float] | None = None,
) -> pd.DataFrame:
    n = len(rows)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["atr_14"] = np.array(
        atr_values if atr_values is not None else [1.0] * n, dtype=float
    )

    df["swing_high_confirm_flag"] = np.array(sh_conf or [0] * n, dtype=np.int8)
    df["swing_low_confirm_flag"] = np.array(sl_conf or [0] * n, dtype=np.int8)
    df["swing_high_confirm_price"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_price or [None] * n)],
        dtype=float,
    )
    df["swing_low_confirm_price"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_price or [None] * n)],
        dtype=float,
    )
    df["swing_high_confirm_origin_idx"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_origin or [None] * n)],
        dtype=float,
    )
    df["swing_low_confirm_origin_idx"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_origin or [None] * n)],
        dtype=float,
    )
    df["swing_high_strength"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_strength or [None] * n)],
        dtype=float,
    )
    df["swing_low_strength"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_strength or [None] * n)],
        dtype=float,
    )
    return df


# ---------------------------------------------------------------------------
# Test 1: Causality
# ---------------------------------------------------------------------------


def test_no_fib_active_before_both_anchors_confirmed() -> None:
    """No fib structure should be active until both swing anchors are confirmed."""
    df = _make_fib_df(
        [
            (99.0, 100.0, 98.0, 99.5),  # row 0: no swings
            (99.5, 101.0, 99.0, 100.5),  # row 1: no swings
            (100.5, 102.0, 100.0, 101.5),  # row 2: swing low confirm only
            (
                101.5,
                103.0,
                101.0,
                102.5,
            ),  # row 3: swing high confirm → bull structure created
            (102.5, 103.5, 101.5, 102.0),  # row 4: active
        ],
        sl_conf=[0, 0, 1, 0, 0],
        sl_price=[None, None, 99.0, None, None],
        sl_origin=[None, None, 0, None, None],
        sh_conf=[0, 0, 0, 1, 0],
        sh_price=[None, None, None, 103.0, None],
        sh_origin=[None, None, None, 2, None],
    )
    result = add_fibonacci_levels(df)

    # Before detect bar (row 3): no bull active
    assert result["fib_bull_active"].iloc[:3].sum() == 0
    # No detect flag before row 3
    assert result["fib_bull_detect_flag"].iloc[:3].sum() == 0
    # On detect bar (row 3): detect flag set, active starts
    assert result.loc[3, "fib_bull_detect_flag"] == 1
    assert result.loc[3, "fib_bull_active"] == 1


# ---------------------------------------------------------------------------
# Test 2: Bull structure detection
# ---------------------------------------------------------------------------


def test_bull_fib_detect_on_swing_high_confirm() -> None:
    """Bull fib structure is created when a swing high confirms after a swing low."""
    anchor_low = 98.0
    anchor_high = 105.0
    df = _make_fib_df(
        [
            (100.0, 101.0, 98.0, 100.5),  # row 0: swing low origin
            (100.5, 102.0, 99.5, 101.5),  # row 1: swing low confirm
            (101.5, 104.0, 101.0, 103.5),  # row 2: rising
            (103.5, 105.5, 103.0, 104.5),  # row 3: swing high confirm → bull detect
            (104.5, 105.0, 103.5, 103.8),  # row 4
        ],
        sl_conf=[0, 1, 0, 0, 0],
        sl_price=[None, anchor_low, None, None, None],
        sl_origin=[None, 0, None, None, None],
        sh_conf=[0, 0, 0, 1, 0],
        sh_price=[None, None, None, anchor_high, None],
        sh_origin=[None, None, None, 2, None],
    )
    result = add_fibonacci_levels(df)

    assert result.loc[3, "fib_bull_detect_flag"] == 1
    assert result.loc[3, "fib_bull_detect_idx"] == 3.0
    assert result.loc[3, "fib_bull_origin_idx"] == 1.0
    assert np.isclose(result.loc[3, "fib_bull_anchor_a_price"], anchor_low)
    assert np.isclose(result.loc[3, "fib_bull_anchor_b_price"], anchor_high)
    expected_range = anchor_high - anchor_low
    assert np.isclose(result.loc[3, "fib_bull_range"], expected_range)
    assert result.loc[3, "fib_bull_active"] == 1


# ---------------------------------------------------------------------------
# Test 3: Bear structure detection
# ---------------------------------------------------------------------------


def test_bear_fib_detect_on_swing_low_confirm() -> None:
    """Bear fib structure is created when a swing low confirms after a swing high."""
    anchor_high = 110.0
    anchor_low = 103.0
    df = _make_fib_df(
        [
            (108.0, 111.0, 107.0, 109.5),  # row 0: swing high origin
            (109.5, 110.5, 108.5, 110.0),  # row 1: swing high confirm
            (110.0, 110.2, 107.0, 107.5),  # row 2: declining
            (107.5, 108.0, 103.5, 104.0),  # row 3: swing low confirm → bear detect
            (104.0, 105.0, 103.5, 104.5),  # row 4
        ],
        sh_conf=[0, 1, 0, 0, 0],
        sh_price=[None, anchor_high, None, None, None],
        sh_origin=[None, 0, None, None, None],
        sl_conf=[0, 0, 0, 1, 0],
        sl_price=[None, None, None, anchor_low, None],
        sl_origin=[None, None, None, 2, None],
    )
    result = add_fibonacci_levels(df)

    assert result.loc[3, "fib_bear_detect_flag"] == 1
    assert result.loc[3, "fib_bear_detect_idx"] == 3.0
    assert result.loc[3, "fib_bear_origin_idx"] == 1.0
    assert np.isclose(result.loc[3, "fib_bear_anchor_a_price"], anchor_high)
    assert np.isclose(result.loc[3, "fib_bear_anchor_b_price"], anchor_low)
    assert result.loc[3, "fib_bear_active"] == 1


# ---------------------------------------------------------------------------
# Test 4: Fib level price arithmetic
# ---------------------------------------------------------------------------


def test_fib_level_prices_are_correct() -> None:
    """Verify fib level prices are computed correctly for bull structure."""
    anchor_low = 100.0
    anchor_high = 110.0
    rng = anchor_high - anchor_low  # 10.0

    df = _make_fib_df(
        [
            (100.0, 101.0, 100.0, 100.5),
            (100.5, 105.0, 100.0, 102.0),
            (102.0, 110.0, 101.0, 108.0),
            (108.0, 111.0, 107.0, 109.5),
            (109.5, 110.5, 108.0, 109.0),
        ],
        sl_conf=[0, 1, 0, 0, 0],
        sl_price=[None, anchor_low, None, None, None],
        sl_origin=[None, 0, None, None, None],
        sh_conf=[0, 0, 0, 1, 0],
        sh_price=[None, None, None, anchor_high, None],
        sh_origin=[None, None, None, 1, None],
    )
    result = add_fibonacci_levels(df)

    detect_row = result[result["fib_bull_detect_flag"] == 1].iloc[0]
    # For bull: level(r) = anchor_high - r * range
    expected = {
        "0000": anchor_high - 0.000 * rng,  # 110.0
        "0236": anchor_high - 0.236 * rng,  # 107.64
        "0382": anchor_high - 0.382 * rng,  # 106.18
        "0500": anchor_high - 0.500 * rng,  # 105.0
        "0618": anchor_high - 0.618 * rng,  # 103.82
        "0786": anchor_high - 0.786 * rng,  # 102.14
        "1000": anchor_high - 1.000 * rng,  # 100.0
        "1272": anchor_high - 1.272 * rng,  # 97.28
        "1618": anchor_high - 1.618 * rng,  # 93.82
    }
    for lbl, expected_price in expected.items():
        col = f"fib_bull_l{lbl}"
        actual = float(detect_row[col])
        assert np.isclose(
            actual, expected_price, atol=1e-6
        ), f"Level {lbl}: expected {expected_price:.4f}, got {actual:.4f}"


# ---------------------------------------------------------------------------
# Test 5: Hard expiry
# ---------------------------------------------------------------------------


def test_fib_structure_expires_after_max_age_bars() -> None:
    """Structure should expire exactly at max_age_bars bars after detection."""
    max_age = 5
    n_total = 15

    rows = [
        (100.0 + i * 0.1, 101.0 + i * 0.1, 99.5 + i * 0.1, 100.5 + i * 0.1)
        for i in range(n_total)
    ]
    sl_conf = [0] * n_total
    sl_price_list = [None] * n_total
    sl_origin_list = [None] * n_total
    sh_conf = [0] * n_total
    sh_price_list = [None] * n_total
    sh_origin_list = [None] * n_total

    sl_conf[1] = 1
    sl_price_list[1] = 100.0
    sl_origin_list[1] = 0

    sh_conf[4] = 1
    sh_price_list[4] = 105.0
    sh_origin_list[4] = 2

    df = _make_fib_df(
        rows,
        sl_conf=sl_conf,
        sl_price=sl_price_list,
        sl_origin=sl_origin_list,
        sh_conf=sh_conf,
        sh_price=sh_price_list,
        sh_origin=sh_origin_list,
    )
    result = add_fibonacci_levels(df, max_age_bars=max_age)

    # Detect at row 4 → active from row 4 onwards
    # max_age=5: age_bars increments each bar AFTER detect
    # age_bars=0 at detect, age_bars=1 at row 5, ..., age_bars=4 at row 8
    # expires when age_bars >= max_age (=5), so at row 9
    assert result.loc[4, "fib_bull_active"] == 1  # on detect bar
    assert result.loc[8, "fib_bull_active"] == 1  # last active bar (age_bars = 4)
    # By row 9 (age_bars = 5 >= max_age), should be expired
    assert result.loc[9, "fib_bull_active"] == 0


# ---------------------------------------------------------------------------
# Test 6: Bull invalidation
# ---------------------------------------------------------------------------


def test_bull_fib_invalidated_on_close_below_anchor_low() -> None:
    """Bull structure should invalidate when close drops below anchor_low with buffer."""
    anchor_low = 100.0
    anchor_high = 110.0
    buffer = 0.0  # zero buffer for simplicity

    rows = [
        (100.0, 101.0, 100.0, 100.5),  # 0
        (100.5, 105.0, 100.0, 102.0),  # 1: swing low confirm
        (102.0, 110.0, 101.0, 108.0),  # 2
        (108.0, 111.0, 107.0, 109.5),  # 3: swing high confirm → bull detect
        (109.5, 110.0, 108.0, 109.0),  # 4: still active
        (109.0, 110.0, 98.0, 99.0),  # 5: close < anchor_low → invalidation
        (99.0, 100.0, 98.5, 99.5),  # 6: should be inactive
    ]
    df = _make_fib_df(
        rows,
        sl_conf=[0, 1, 0, 0, 0, 0, 0],
        sl_price=[None, anchor_low, None, None, None, None, None],
        sl_origin=[None, 0, None, None, None, None, None],
        sh_conf=[0, 0, 0, 1, 0, 0, 0],
        sh_price=[None, None, None, anchor_high, None, None, None],
        sh_origin=[None, None, None, 1, None, None, None],
        atr_values=[1.0] * 7,
    )
    result = add_fibonacci_levels(df, invalidation_buffer_atr=buffer)

    assert result.loc[3, "fib_bull_active"] == 1  # detect
    assert result.loc[4, "fib_bull_active"] == 1  # still active
    assert result.loc[6, "fib_bull_active"] == 0  # after invalidation


# ---------------------------------------------------------------------------
# Test 7: Bear invalidation
# ---------------------------------------------------------------------------


def test_bear_fib_invalidated_on_close_above_anchor_high() -> None:
    """Bear structure should invalidate when close rises above anchor_high with buffer."""
    anchor_high = 110.0
    anchor_low = 103.0

    rows = [
        (108.0, 111.0, 107.0, 109.5),  # 0
        (109.5, 110.5, 108.5, 110.0),  # 1: swing high confirm
        (110.0, 110.2, 107.0, 107.5),  # 2
        (107.5, 108.0, 103.5, 104.0),  # 3: swing low confirm → bear detect
        (104.0, 106.0, 103.5, 105.0),  # 4: still active
        (105.0, 112.0, 104.5, 111.5),  # 5: close > anchor_high → invalidation
        (111.5, 112.0, 110.0, 111.0),  # 6: should be inactive
    ]
    df = _make_fib_df(
        rows,
        sh_conf=[0, 1, 0, 0, 0, 0, 0],
        sh_price=[None, anchor_high, None, None, None, None, None],
        sh_origin=[None, 0, None, None, None, None, None],
        sl_conf=[0, 0, 0, 1, 0, 0, 0],
        sl_price=[None, None, None, anchor_low, None, None, None],
        sl_origin=[None, None, None, 1, None, None, None],
        atr_values=[1.0] * 7,
    )
    result = add_fibonacci_levels(df, invalidation_buffer_atr=0.0)

    assert result.loc[3, "fib_bear_active"] == 1
    assert result.loc[4, "fib_bear_active"] == 1
    assert result.loc[6, "fib_bear_active"] == 0


# ---------------------------------------------------------------------------
# Test 8: Nearest level projection
# ---------------------------------------------------------------------------


def test_nearest_level_projection_is_causal() -> None:
    """Nearest fib level columns should be NaN before any structure is active."""
    df = _make_fib_df(
        [(100.0, 101.0, 99.0, 100.5)] * 10,
    )
    result = add_fibonacci_levels(df)

    assert result["fib_nearest_bull_level"].isna().all()
    assert result["fib_nearest_bear_level"].isna().all()
    assert result["fib_bull_active"].sum() == 0
    assert result["fib_bear_active"].sum() == 0


# ---------------------------------------------------------------------------
# Test 9: Multiple concurrent structures
# ---------------------------------------------------------------------------


def test_multiple_concurrent_structures_selects_highest_quality() -> None:
    """When multiple bull structures are active, the highest quality score is selected."""
    # Create two bull structures at different times
    n = 15
    rows = [
        (100.0 + i * 0.5, 101.0 + i * 0.5, 99.5 + i * 0.5, 100.3 + i * 0.5)
        for i in range(n)
    ]

    sl_conf = [0] * n
    sl_price_list = [None] * n
    sl_origin_list = [None] * n
    sh_conf = [0] * n
    sh_price_list = [None] * n
    sh_origin_list = [None] * n
    sh_strength_list = [None] * n
    sl_strength_list = [None] * n

    # Structure 1: small range (lower quality)
    sl_conf[1] = 1
    sl_price_list[1] = 100.0
    sl_origin_list[1] = 0
    sl_strength_list[1] = 0.5
    sh_conf[3] = 1
    sh_price_list[3] = 101.0
    sh_origin_list[3] = 2
    sh_strength_list[3] = 0.5  # range = 1.0 ATR

    # Structure 2: larger range (higher quality)
    sl_conf[5] = 1
    sl_price_list[5] = 102.0
    sl_origin_list[5] = 4
    sl_strength_list[5] = 0.9
    sh_conf[8] = 1
    sh_price_list[8] = 110.0
    sh_origin_list[8] = 6
    sh_strength_list[8] = 0.9  # range = 8.0 ATR

    df = _make_fib_df(
        rows,
        sl_conf=sl_conf,
        sl_price=sl_price_list,
        sl_origin=sl_origin_list,
        sh_conf=sh_conf,
        sh_price=sh_price_list,
        sh_origin=sh_origin_list,
        sh_strength=sh_strength_list,
        sl_strength=sl_strength_list,
        atr_values=[1.0] * n,
    )
    result = add_fibonacci_levels(df, max_age_bars=100)

    # At row 9 onwards, both structures should be active
    # The selected one (fib_bull_quality_score) should be from structure 2 (higher quality)
    assert result.loc[9, "fib_active_bull_count"] >= 2
    # Structure 2 has range_atr = 8.0, structure 1 has range_atr = 1.0
    # Quality is primarily driven by range, so structure 2 should win
    assert (
        result.loc[9, "fib_bull_quality_score"]
        > result.loc[4, "fib_bull_quality_score"]
    )


# ---------------------------------------------------------------------------
# Test 10: Pure function
# ---------------------------------------------------------------------------


def test_pure_function_does_not_mutate_input() -> None:
    """add_fibonacci_levels must not mutate the input DataFrame."""
    df = _make_fib_df(
        [(100.0, 101.0, 99.0, 100.5)] * 5,
        sl_conf=[0, 1, 0, 0, 0],
        sl_price=[None, 99.0, None, None, None],
        sh_conf=[0, 0, 0, 1, 0],
        sh_price=[None, None, None, 105.0, None],
    )
    original_cols = list(df.columns)
    original_close = df["close"].copy()

    _ = add_fibonacci_levels(df)

    assert list(df.columns) == original_cols
    pd.testing.assert_series_equal(df["close"], original_close)


# ---------------------------------------------------------------------------
# Test 11: Touch zone detection
# ---------------------------------------------------------------------------


def test_inside_bull_zone_flag_set_when_price_within_tolerance() -> None:
    """fib_inside_bull_zone should be 1 when close is within touch_tol of any fib level."""
    anchor_low = 100.0
    anchor_high = 110.0
    # 0.618 retracement for bull = 110 - 0.618 * 10 = 103.82
    golden_level = anchor_high - 0.618 * (anchor_high - anchor_low)  # 103.82

    rows = [
        (100.0, 101.0, 100.0, 100.5),  # 0
        (100.5, 105.0, 100.0, 102.0),  # 1: swing low confirm
        (102.0, 110.0, 101.0, 108.0),  # 2
        (
            108.0,
            111.0,
            107.0,
            109.5,
        ),  # 3: bull detect (anchor_low=100, anchor_high=110)
        # close near the golden ratio level
        (109.5, 110.0, 103.7, golden_level + 0.05),  # 4: close ~= golden level
        (
            golden_level,
            golden_level + 1.0,
            golden_level - 0.1,
            golden_level + 0.5,
        ),  # 5: clearly at level
        (107.0, 108.0, 106.0, 107.0),  # 6: far from any level
    ]
    df = _make_fib_df(
        rows,
        sl_conf=[0, 1, 0, 0, 0, 0, 0],
        sl_price=[None, anchor_low, None, None, None, None, None],
        sl_origin=[None, 0, None, None, None, None, None],
        sh_conf=[0, 0, 0, 1, 0, 0, 0],
        sh_price=[None, None, None, anchor_high, None, None, None],
        sh_origin=[None, None, None, 1, None, None, None],
        atr_values=[1.0] * 7,
    )
    result = add_fibonacci_levels(df, touch_tol_atr=0.20)

    # Row 5: close is at golden_level, tolerance = 0.20 * ATR = 0.20
    # Distance = 0 < 0.20 → inside zone
    assert result.loc[5, "fib_inside_bull_zone"] == 1
    # Row 6: close = 107.0, nearest level check
    # Levels: 0.236 = 107.64, 0.382 = 106.18 → nearest is 0.236 at distance 0.64 > 0.20
    assert result.loc[6, "fib_inside_bull_zone"] == 0
