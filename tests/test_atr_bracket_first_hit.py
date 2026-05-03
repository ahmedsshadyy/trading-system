"""Tests for Step 11T — ATR bracket first-hit audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.indicators.atr_bracket_first_hit import (
    ATR_BRACKET_HORIZONS,
    build_atr_bracket_events,
    build_atr_bracket_first_hit_diagnostics,
)


def _empty_swing_columns(n: int) -> dict[str, np.ndarray]:
    return {
        "swing_high_confirm_flag": np.zeros(n),
        "swing_low_confirm_flag": np.zeros(n),
    }


def _empty_sweep_columns(n: int) -> dict[str, np.ndarray | list]:
    return {
        "sweep_flag": np.zeros(n),
        "sweep_side": np.full(n, np.nan),
        "sweep_class": np.full(n, np.nan),
        "sweep_selectivity_class": [""] * n,
        "sweep_primary_family": [""] * n,
        "sweep_is_displacement_confirmed": np.zeros(n),
        "sweep_is_tradeable_candidate": np.zeros(n),
    }


def _frame(highs, lows, closes, *, atr=1.0):
    n = len(highs)
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": list(closes),
        "high": list(highs),
        "low": list(lows),
        "close": list(closes),
        "atr_14": [atr] * n,
        "regime_label": ["trend"] * n,
        "session_name": ["london"] * n,
    }
    base.update(_empty_swing_columns(n))
    base.update(_empty_sweep_columns(n))
    return pd.DataFrame(base)


def _set_swing_low(df: pd.DataFrame, idx: int) -> None:
    df.loc[idx, "swing_low_confirm_flag"] = 1.0


def _set_swing_high(df: pd.DataFrame, idx: int) -> None:
    df.loc[idx, "swing_high_confirm_flag"] = 1.0


def _set_sweep(
    df: pd.DataFrame,
    idx: int,
    *,
    side: int,
    selectivity: str = "standard_liquidity_sweep",
    family: str = "swing_low",
    displacement: bool = False,
    tradeable: bool = False,
) -> None:
    df.loc[idx, "sweep_flag"] = 1.0
    df.loc[idx, "sweep_side"] = float(side)
    df.loc[idx, "sweep_selectivity_class"] = selectivity
    df.loc[idx, "sweep_primary_family"] = family
    df.loc[idx, "sweep_is_displacement_confirmed"] = 1.0 if displacement else 0.0
    df.loc[idx, "sweep_is_tradeable_candidate"] = 1.0 if tradeable else 0.0
    df.loc[idx, "sweep_class"] = 1.0


def test_bullish_swing_tp_first_locks_in_outcome():
    # Confirm at idx 5, entry close=100, atr=1 → TP=100.5, SL=99.5.
    # Bar 6 high=100.6 (TP hit), low=99.8 (SL not hit) → tp_first.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 100.6
    lows[6] = 99.8
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5)

    events = build_atr_bracket_events(df)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["signal_entity_type"] == "swing"
    assert row["signal_side"] == "bullish_reversal"
    assert row["swing_side"] == "swing_low"
    assert row["entry_close"] == 100.0
    assert row["tp_price"] == 100.5
    assert row["sl_price"] == 99.5
    for h in ATR_BRACKET_HORIZONS:
        if h >= 1:
            assert row[f"bracket_outcome_{h}"] == "tp_first"
            assert row[f"first_tp_hit_bar_{h}"] == 1.0
            assert pd.isna(row[f"first_sl_hit_bar_{h}"])
            assert row[f"tp_before_sl_flag_{h}"] == 1.0
            assert row[f"sl_before_tp_flag_{h}"] == 0.0
            assert row[f"ambiguous_same_bar_flag_{h}"] == 0.0


def test_bearish_swing_sl_first():
    # swing_high → bearish: TP=entry-0.5, SL=entry+0.5.
    # Bar 6 SL hits via high=100.6 (>= SL=100.5), TP unmet.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 100.6
    lows[6] = 99.8
    df = _frame(highs, lows, closes)
    _set_swing_high(df, 5)

    events = build_atr_bracket_events(df)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["signal_side"] == "bearish_reversal"
    assert row["tp_price"] == 99.5
    assert row["sl_price"] == 100.5
    for h in ATR_BRACKET_HORIZONS:
        assert row[f"bracket_outcome_{h}"] == "sl_first"
        assert row[f"first_sl_hit_bar_{h}"] == 1.0
        assert pd.isna(row[f"first_tp_hit_bar_{h}"])


def test_same_bar_ambiguous_never_resolves_to_tp_or_sl():
    # Bar 6 hits both TP=100.5 (high=100.6) and SL=99.5 (low=99.4).
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 100.6
    lows[6] = 99.4
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5)

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    for h in ATR_BRACKET_HORIZONS:
        assert row[f"bracket_outcome_{h}"] == "ambiguous_same_bar"
        assert row[f"ambiguous_same_bar_flag_{h}"] == 1.0
        assert row[f"tp_before_sl_flag_{h}"] == 0.0
        assert row[f"sl_before_tp_flag_{h}"] == 0.0
        assert row[f"first_tp_hit_bar_{h}"] == 1.0
        assert row[f"first_sl_hit_bar_{h}"] == 1.0


def test_neither_when_horizon_quiet():
    # No bar in [6..10] crosses ±0.5 ATR.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5)

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    for h in ATR_BRACKET_HORIZONS:
        assert row[f"bracket_outcome_{h}"] == "neither"
        assert pd.isna(row[f"first_tp_hit_bar_{h}"])
        assert pd.isna(row[f"first_sl_hit_bar_{h}"])


def test_insufficient_future_when_too_few_bars_and_no_hit():
    # Confirm at n-3 → only 2 future bars. Horizon 5 → insufficient_future.
    n = 8
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, n - 3)

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    assert row["bracket_outcome_1"] == "neither"  # 2 bars > 1, but no hit
    assert row["bracket_outcome_2"] == "neither"
    for h in (3, 4, 5):
        assert row[f"bracket_outcome_{h}"] == "insufficient_future"


def test_early_hit_locks_in_horizon_even_with_short_future():
    # Confirm at n-3 → only 2 future bars. Bar n-2 hits TP at delay 1.
    # Horizon 5 should still be tp_first (bracket position closed at bar 1).
    n = 8
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[n - 2] = 100.6
    df = _frame(highs, lows, closes)
    _set_swing_low(df, n - 3)

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    for h in ATR_BRACKET_HORIZONS:
        assert row[f"bracket_outcome_{h}"] == "tp_first"
        assert row[f"first_tp_hit_bar_{h}"] == 1.0


def test_bullish_sweep_below_sell_side_uses_long_bracket():
    # sweep_side = -1 (below_sell_side) → bullish_reversal: TP=entry+0.5.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 100.6
    df = _frame(highs, lows, closes)
    _set_sweep(
        df,
        5,
        side=-1,
        selectivity="displacement_confirmed_sweep",
        displacement=True,
        tradeable=True,
        family="prev_day_low",
    )

    events = build_atr_bracket_events(df)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["signal_entity_type"] == "sweep"
    assert row["signal_side"] == "bullish_reversal"
    assert row["sweep_side_label"] == "below_sell_side"
    assert row["sweep_selectivity_class"] == "displacement_confirmed_sweep"
    assert row["sweep_primary_family"] == "prev_day_low"
    assert row["bracket_outcome_5"] == "tp_first"


def test_bearish_sweep_above_buy_side_uses_short_bracket():
    # sweep_side = 1 (above_buy_side) → bearish_reversal: TP=entry-0.5.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    lows[6] = 99.4
    df = _frame(highs, lows, closes)
    _set_sweep(
        df, 5, side=1, selectivity="standard_liquidity_sweep", family="prev_week_high"
    )

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    assert row["signal_side"] == "bearish_reversal"
    assert row["sweep_side_label"] == "above_buy_side"
    assert row["bracket_outcome_5"] == "tp_first"


def test_no_lookahead_prior_bars_ignored():
    # Confirm at idx=5; bars 0..4 contain extreme values that would trigger
    # both TP and SL but must be ignored.
    n = 12
    closes = [100.0] * n
    highs = [200.0] * 5 + [100.1] * (n - 5)  # huge highs before confirm
    lows = [50.0] * 5 + [99.9] * (n - 5)  # tiny lows before confirm
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5)

    events = build_atr_bracket_events(df)
    row = events.iloc[0]
    for h in ATR_BRACKET_HORIZONS:
        assert row[f"bracket_outcome_{h}"] == "neither"


def test_diagnostics_summary_shape_and_keys():
    n = 30
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 100.6  # TP hit for swing 1
    highs[16] = 100.6  # TP hit for swing 2
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5)
    _set_swing_low(df, 15)
    _set_sweep(
        df,
        10,
        side=-1,
        selectivity="displacement_confirmed_sweep",
        displacement=True,
        tradeable=True,
    )

    diag = build_atr_bracket_first_hit_diagnostics(df)
    summary = diag["summary"]
    assert summary["swings_total"] == 2
    assert summary["sweeps_total"] == 1
    # All confirmed swings hit TP at bar 1 in this fixture.
    assert summary["swings_tp_first_rate_5"] == 1.0
    assert summary["swings_sl_first_rate_5"] == 0.0
    assert summary["swings_win_rate_ex_ambiguous_5"] == 1.0
    # by_entity table has both rows.
    by_entity = diag["by_entity"]
    assert set(by_entity["signal_entity_type"]) == {"swing", "sweep"}


def test_win_rate_with_ambiguous_half_credit():
    # Build 2 tp_first + 2 ambiguous + 0 sl_first events:
    #   half_credit = (2 + 0.5*2) / (2 + 0 + 2) = 0.75
    # ex_ambiguous = 2 / (2 + 0) = 1.0
    n = 40
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    # tp_first bars
    highs[6] = 100.6
    highs[11] = 100.6
    # ambiguous bars (both TP and SL hit)
    highs[16] = 100.6
    lows[16] = 99.4
    highs[21] = 100.6
    lows[21] = 99.4
    df = _frame(highs, lows, closes)
    for confirm in (5, 10, 15, 20):
        _set_swing_low(df, confirm)

    diag = build_atr_bracket_first_hit_diagnostics(df)
    summary = diag["summary"]
    by_entity = diag["by_entity"]
    swing_row = by_entity[by_entity["signal_entity_type"] == "swing"].iloc[0]
    assert swing_row["tp_first_count_5"] == 2
    assert swing_row["ambiguous_count_5"] == 2
    assert swing_row["sl_first_count_5"] == 0
    assert swing_row["win_rate_ex_ambiguous_5"] == 1.0
    assert swing_row["win_rate_with_ambiguous_half_credit_5"] == 0.75
    assert summary["swings_win_rate_ex_ambiguous_5"] == 1.0
