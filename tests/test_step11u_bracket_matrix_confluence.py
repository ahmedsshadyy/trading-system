"""Tests for Step 11U — bracket matrix + confluence edge audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.indicators.bracket_matrix_confluence import (
    STEP11U_BRACKET_PROFILES,
    STEP11U_HORIZONS,
    build_step11u_diagnostics,
    build_step11u_events,
)


def _empty_arrays(n: int) -> dict[str, object]:
    return {
        "swing_high_confirm_flag": np.zeros(n),
        "swing_low_confirm_flag": np.zeros(n),
        "swing_high_confirm_price": np.full(n, np.nan),
        "swing_low_confirm_price": np.full(n, np.nan),
        "sweep_flag": np.zeros(n),
        "sweep_side": np.full(n, np.nan),
        "sweep_class": np.full(n, np.nan),
        "sweep_source_level": np.full(n, np.nan),
        "sweep_selectivity_class": [""] * n,
        "sweep_primary_family": [""] * n,
        "sweep_is_displacement_confirmed": np.zeros(n),
        "sweep_is_tradeable_candidate": np.zeros(n),
        "displacement_bull": np.zeros(n, dtype=int),
        "displacement_bear": np.zeros(n, dtype=int),
    }


def _frame(highs, lows, closes, *, atr=1.0, vol_ratio=None) -> pd.DataFrame:
    n = len(highs)
    base = {
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": list(closes),
        "high": list(highs),
        "low": list(lows),
        "close": list(closes),
        "atr_14": [atr] * n,
        "regime_label": ["TRENDING"] * n,
        "session_name": ["NY"] * n,
        "vol_ratio": [1.0] * n if vol_ratio is None else list(vol_ratio),
    }
    base.update(_empty_arrays(n))
    return pd.DataFrame(base)


def _set_swing_low(df: pd.DataFrame, idx: int, *, price: float) -> None:
    df.loc[idx, "swing_low_confirm_flag"] = 1.0
    df.loc[idx, "swing_low_confirm_price"] = price


def _set_swing_high(df: pd.DataFrame, idx: int, *, price: float) -> None:
    df.loc[idx, "swing_high_confirm_flag"] = 1.0
    df.loc[idx, "swing_high_confirm_price"] = price


def _set_sweep(
    df: pd.DataFrame,
    idx: int,
    *,
    side: int,
    selectivity: str = "standard_liquidity_sweep",
    family: str = "session_low",
    source_level: float = 0.0,
    displacement_confirmed: bool = False,
) -> None:
    df.loc[idx, "sweep_flag"] = 1.0
    df.loc[idx, "sweep_side"] = float(side)
    df.loc[idx, "sweep_class"] = 1.0
    df.loc[idx, "sweep_source_level"] = source_level
    df.loc[idx, "sweep_selectivity_class"] = selectivity
    df.loc[idx, "sweep_primary_family"] = family
    df.loc[idx, "sweep_is_displacement_confirmed"] = (
        1.0 if displacement_confirmed else 0.0
    )


def test_bullish_swing_with_displacement_attached_when_within_window():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5, price=99.0)
    df.loc[6, "displacement_bull"] = 1  # 1-bar lookahead

    events = build_step11u_events(df)
    swing_rows = events[events["signal_entity_type"] == "swing"]
    assert len(swing_rows) == 1
    row = swing_rows.iloc[0]
    assert bool(row["displacement_confirmed"])
    assert row["signal_confluence_type"] == "swing_displacement_confirmed"


def test_swing_high_requires_bearish_displacement():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_high(df, 5, price=101.0)
    # Bullish displacement — must NOT count for swing_high.
    df.loc[6, "displacement_bull"] = 1

    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    assert bool(row["displacement_confirmed"]) is False
    assert row["signal_confluence_type"] == "swing_all"


def test_displacement_window_respects_1_to_3_bar_bounds():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5, price=99.0)
    # Bar 9 (4 bars after) — outside [1,3] window.
    df.loc[9, "displacement_bull"] = 1
    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    assert bool(row["displacement_confirmed"]) is False


def test_sweep_swing_confluence_uses_only_already_confirmed_swing():
    n = 14
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    # Confirmed swing low BEFORE the sweep — must be picked up.
    _set_swing_low(df, 4, price=99.0)
    # Sweep at idx 6, source level very near the prior swing low.
    _set_sweep(df, 6, side=-1, source_level=99.05, family="session_low")

    events = build_step11u_events(df)
    sweep_row = events[events["signal_entity_type"] == "sweep"].iloc[0]
    assert bool(sweep_row["swing_confluent"]) is True
    assert sweep_row["swing_confluence_type"] == "proximity"
    # |99.05 - 99.0| / atr(1.0) ≈ 0.05 ≤ 0.25
    assert abs(sweep_row["swing_confluence_distance_atr"] - 0.05) < 1e-9


def test_sweep_swing_confluence_does_not_use_future_swing():
    n = 14
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    # Sweep at idx 6, swing only confirmed AFTER the sweep — must not match.
    _set_sweep(df, 6, side=-1, source_level=99.05, family="session_low")
    _set_swing_low(df, 8, price=99.0)

    events = build_step11u_events(df)
    sweep_row = events[events["signal_entity_type"] == "sweep"].iloc[0]
    assert bool(sweep_row["swing_confluent"]) is False
    assert sweep_row["swing_confluence_type"] == "none"
    # No matching pool yet → distance NaN.
    assert pd.isna(sweep_row["swing_confluence_distance_atr"])


def test_sweep_direct_source_confluence_when_family_is_swing():
    n = 14
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_sweep(df, 6, side=-1, source_level=99.05, family="swing_low")
    events = build_step11u_events(df)
    sweep_row = events[events["signal_entity_type"] == "sweep"].iloc[0]
    assert bool(sweep_row["swing_confluent"]) is True
    assert sweep_row["swing_confluence_type"] == "direct_source"


def test_sweep_swing_displacement_confluence_label():
    n = 14
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 4, price=99.0)
    _set_sweep(
        df,
        6,
        side=-1,
        source_level=99.0,
        family="session_low",
        displacement_confirmed=True,
    )
    events = build_step11u_events(df)
    sweep_row = events[events["signal_entity_type"] == "sweep"].iloc[0]
    assert sweep_row["signal_confluence_type"] == "sweep_swing_displacement_confluence"


def test_bullish_bracket_tp1p0_sl0p5_directionality():
    # Long with TP=+1.0 ATR, SL=-0.5 ATR.
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 101.05  # TP=101.0 hit
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5, price=99.0)

    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    assert row["tp1p0_sl0p5_tp_price"] == 101.0
    assert row["tp1p0_sl0p5_sl_price"] == 99.5
    assert row["tp1p0_sl0p5_bracket_outcome_5"] == "tp_first"


def test_short_bracket_tp0p5_sl1p0_directionality():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    highs[6] = 101.05  # SL=101.0 hit (short)
    df = _frame(highs, lows, closes)
    _set_swing_high(df, 5, price=101.0)

    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    assert row["tp0p5_sl1p0_tp_price"] == 99.5
    assert row["tp0p5_sl1p0_sl_price"] == 101.0
    assert row["tp0p5_sl1p0_bracket_outcome_5"] == "sl_first"


def test_same_bar_ambiguous_explicit_for_all_profiles():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    # Bar 6 hits both extremes.
    highs[6] = 101.5
    lows[6] = 98.5
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5, price=99.0)

    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    for label, _, _ in STEP11U_BRACKET_PROFILES:
        for h in STEP11U_HORIZONS:
            assert row[f"{label}_bracket_outcome_{h}"] == "ambiguous"
            assert row[f"{label}_ambiguous_same_bar_flag_{h}"] == 1.0
            assert row[f"{label}_tp_before_sl_flag_{h}"] == 0.0
            assert row[f"{label}_sl_before_tp_flag_{h}"] == 0.0


def test_no_lookahead_bars_before_confirm_ignored():
    n = 12
    closes = [100.0] * n
    highs = [200.0] * 5 + [100.1] * (n - 5)
    lows = [50.0] * 5 + [99.9] * (n - 5)
    df = _frame(highs, lows, closes)
    _set_swing_low(df, 5, price=99.0)
    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    for label, _, _ in STEP11U_BRACKET_PROFILES:
        for h in STEP11U_HORIZONS:
            assert row[f"{label}_bracket_outcome_{h}"] == "neither"


def test_insufficient_horizon_when_no_hit_and_too_few_bars():
    n = 8
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    _set_swing_low(df, n - 3, price=99.0)
    events = build_step11u_events(df)
    row = events[events["signal_entity_type"] == "swing"].iloc[0]
    # 2 future bars available
    assert row["tp0p5_sl0p5_bracket_outcome_2"] == "neither"
    assert row["tp0p5_sl0p5_bracket_outcome_3"] == "insufficient"
    assert row["tp0p5_sl0p5_bracket_outcome_5"] == "insufficient"


def test_diagnostics_universes_overlap_and_low_flags_present():
    # 4 swings, 1 sweep_swing_displacement_confluence event. Lots of
    # neither outcomes but enough rows to test universe filters.
    n = 60
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    # Swings at 5, 15, 25, 35; one with displacement
    df = _frame(highs, lows, closes)
    for i in (5, 15, 25, 35):
        _set_swing_low(df, i, price=99.0)
    df.loc[6, "displacement_bull"] = 1  # makes swing 5 displacement-confirmed

    # Sweep at 45 is direct_source confluence + displacement-confirmed
    _set_sweep(
        df,
        45,
        side=-1,
        source_level=99.0,
        family="swing_low",
        displacement_confirmed=True,
    )

    diag = build_step11u_diagnostics(df)
    by_conf = diag["by_confluence_type"]
    counts = dict(zip(by_conf["signal_confluence_type"], by_conf["count"]))
    assert counts["swing_all"] == 4
    assert counts["swing_displacement_confirmed"] == 1
    assert counts["sweep_all"] == 1
    assert counts["sweep_displacement_confirmed"] == 1
    assert counts["sweep_swing_confluence"] == 1
    assert counts["sweep_swing_displacement_confluence"] == 1
    # low_sample column added
    assert "low_sample" in by_conf.columns
    # primary horizon resolution flag column added per profile
    assert "low_resolution_tp0p5_sl0p5_h5" in by_conf.columns


def test_volume_confirmed_flag_uses_vol_ratio_threshold():
    n = 12
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    vols = [0.5] * n
    vols[5] = 1.5
    df = _frame(highs, lows, closes, vol_ratio=vols)
    _set_swing_low(df, 5, price=99.0)
    _set_swing_low(df, 7, price=99.0)
    events = build_step11u_events(df)
    swings = events[events["signal_entity_type"] == "swing"].sort_values("signal_idx")
    assert bool(swings.iloc[0]["volume_confirmed"]) is True
    assert bool(swings.iloc[1]["volume_confirmed"]) is False


def test_best_group_skips_low_sample_in_main_path():
    n = 80
    closes = [100.0] * n
    highs = [100.1] * n
    lows = [99.9] * n
    df = _frame(highs, lows, closes)
    # 2 swing events — count=2, below 100 sample threshold.
    _set_swing_low(df, 5, price=99.0)
    _set_swing_high(df, 15, price=101.0)
    diag = build_step11u_diagnostics(df)
    summary = diag["summary"]
    # All groups have low sample; best_group_* should be empty strings.
    assert summary["best_group_by_tp0p5_sl0p5"] == ""
    assert summary["best_group_with_min_count_300"] == ""
