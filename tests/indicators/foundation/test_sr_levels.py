"""
tests/indicators/foundation/test_sr_levels.py

Full test coverage for src/indicators/foundation/sr_levels.py.

Five synthetic fixtures cover the canonical lifecycle scenarios:

  A — clean support hold (touch_count increments, state stays active)
  B — support weakening then invalidation (weaken → break)
  C — deduplication / supersession of nearby same-side levels
  D — range context (between_nearest_sr_flag, flags, distance)
  E — prior-period family (prev_day_high / prev_day_low activation)

ATR for all fixtures: fixed at 5.0 (supplied as column ``atr_14``).
Derived thresholds:
  touch zone    ±0.75  (TOUCH_TOL_ATR=0.15 × 5.0)
  invalidation  ±0.50  (INVALIDATION_BUFFER_ATR=0.10 × 5.0)
  merge tol      1.00  (MERGE_TOL_ATR=0.20 × 5.0)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.indicators.foundation.sr_levels import (
    FAMILY_PRIOR,
    INVALIDATION_BUFFER_ATR,
    MAX_AGE_BARS,
    MAX_WEAKEN_COUNT,
    MERGE_TOL_ATR,
    SR_FAMILY_DAY,
    SR_FAMILY_SWING,
    SR_SIDE_RESISTANCE,
    SR_SIDE_SUPPORT,
    SR_STATE_ACTIVE,
    SR_STATE_INACTIVE_PRE_LIVE,
    SR_STATE_INVALIDATED,
    SR_STATE_RETIRED,
    TOUCH_TOL_ATR,
    SRLevel,
    add_sr_levels,
    build_sr_level_registry,
    extract_sr_source_events,
    project_sr_context,
    update_sr_lifecycle,
    _compute_strength,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATR = 5.0
TOUCH_TOL = TOUCH_TOL_ATR * ATR  # 0.75
INV_BUF = INVALIDATION_BUFFER_ATR * ATR  # 0.50
MERGE_TOL = MERGE_TOL_ATR * ATR  # 1.00


def _base_candle(
    n: int, *, open_=100.0, high=101.0, low=99.0, close=100.0
) -> pd.DataFrame:
    """Minimal OHLC frame with fixed ATR column."""
    return pd.DataFrame(
        {
            "open": np.full(n, open_),
            "high": np.full(n, high),
            "low": np.full(n, low),
            "close": np.full(n, close),
            "volume": np.ones(n),
            "atr_14": np.full(n, ATR),
        }
    )


def _swing_confirm_at(
    df: pd.DataFrame,
    bar: int,
    price: float,
    *,
    side: int,
    origin_bar: int | None = None,
) -> None:
    """Inject a single swing confirm event into ``df`` at ``bar``."""
    if origin_bar is None:
        origin_bar = max(0, bar - 3)

    if side == SR_SIDE_RESISTANCE:
        if "swing_high_confirm_flag" not in df.columns:
            df["swing_high_confirm_flag"] = 0
            df["swing_high_confirm_price"] = np.nan
            df["swing_high_confirm_origin_idx"] = np.nan
            df["swing_low_confirm_flag"] = 0
            df["swing_low_confirm_price"] = np.nan
            df["swing_low_confirm_origin_idx"] = np.nan
        df.at[bar, "swing_high_confirm_flag"] = 1
        df.at[bar, "swing_high_confirm_price"] = price
        df.at[bar, "swing_high_confirm_origin_idx"] = origin_bar
    else:
        if "swing_low_confirm_flag" not in df.columns:
            df["swing_high_confirm_flag"] = 0
            df["swing_high_confirm_price"] = np.nan
            df["swing_high_confirm_origin_idx"] = np.nan
            df["swing_low_confirm_flag"] = 0
            df["swing_low_confirm_price"] = np.nan
            df["swing_low_confirm_origin_idx"] = np.nan
        df.at[bar, "swing_low_confirm_flag"] = 1
        df.at[bar, "swing_low_confirm_price"] = price
        df.at[bar, "swing_low_confirm_origin_idx"] = origin_bar


# ---------------------------------------------------------------------------
# Fixture A — clean support hold
# ---------------------------------------------------------------------------


def _fixture_a() -> pd.DataFrame:
    """
    50-bar frame.  Swing low at bar 5 (price = 95.0).
    Bars 10, 15, 20: candle low touches support zone (low=95.3, close=100.0).
    Price never falls below invalidation threshold.
    Expected: level active throughout, touch_count = 3 by bar 20.
    """
    n = 50
    df = _base_candle(n, close=100.0, low=99.0, high=101.0)
    # Add swing columns
    _swing_confirm_at(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    # Touch bars: low enters touch zone (≤ 95.0 + 0.75 = 95.75) but close holds
    for bar in (10, 15, 20):
        df.at[bar, "low"] = 95.3  # inside touch zone
        df.at[bar, "high"] = 101.0
        df.at[bar, "close"] = 100.0
    return df


def test_fixture_a_level_created():
    df = _fixture_a()
    events = extract_sr_source_events(df)
    sup_events = [e for e in events if e.side == SR_SIDE_SUPPORT]
    assert len(sup_events) == 1
    assert sup_events[0].level_price == pytest.approx(95.0)
    assert sup_events[0].source_family == SR_FAMILY_SWING
    assert sup_events[0].source_live_from_idx == 5


def test_fixture_a_state_active_after_activation():
    df = _fixture_a()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    lev = next(l for l in registry.values() if l.side == SR_SIDE_SUPPORT)
    assert lev.state == SR_STATE_ACTIVE


def test_fixture_a_touch_count():
    df = _fixture_a()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    lev = next(l for l in registry.values() if l.side == SR_SIDE_SUPPORT)
    assert lev.touch_count == 3


def test_fixture_a_purity():
    """add_sr_levels must not mutate the original DataFrame."""
    df = _fixture_a()
    original_cols = list(df.columns)
    original_close = df["close"].copy()
    _ = add_sr_levels(df, include_research_only=False)
    assert list(df.columns) == original_cols
    pd.testing.assert_series_equal(df["close"], original_close)


def test_fixture_a_projection_nearest_support():
    df = _fixture_a()
    out = add_sr_levels(df, include_research_only=False)
    # From bar 5 onward there should be a nearest support active
    assert int(out["nearest_support_active"].iloc[6]) == 1
    assert float(out["nearest_support_price"].iloc[10]) == pytest.approx(95.0)


def test_fixture_a_no_label_contamination():
    df = _fixture_a()
    out = add_sr_levels(df, include_research_only=False)
    bad = [c for c in out.columns if c.startswith("label_") or c.startswith("future_")]
    assert bad == []


# ---------------------------------------------------------------------------
# Fixture B — support weakening then invalidation
# ---------------------------------------------------------------------------


def _fixture_b() -> pd.DataFrame:
    """
    60-bar frame.  Swing low at bar 5 (price = 95.0).
    Bars 10–12: low pierces touch zone but close holds → weaken events.
    Bar 25: close falls below 95.0 − INV_BUF (= 94.5) → invalidation.
    """
    n = 60
    df = _base_candle(n, close=100.0, low=99.0, high=101.0)
    _swing_confirm_at(df, bar=5, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)

    # Weaken events: low < 95.0 - TOUCH_TOL = 94.25 but close stays above 94.5
    for bar in (10, 11, 12):
        df.at[bar, "low"] = 94.0  # low < 94.25 → weak pierce
        df.at[bar, "close"] = 95.2  # close > 94.5 → holds, not invalidated

    # Invalidation: close < 94.5
    df.at[25, "close"] = 94.0
    df.at[25, "low"] = 93.5
    # After invalidation keep close below (levels remain invalidated)
    for bar in range(26, 60):
        df.at[bar, "close"] = 94.0
        df.at[bar, "low"] = 93.5

    return df


def test_fixture_b_weaken_count():
    df = _fixture_b()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    lev = next(l for l in registry.values() if l.side == SR_SIDE_SUPPORT)
    # weaken_count is capped by MAX_WEAKEN_COUNT=4 but we only trigger 3 events
    assert lev.weaken_count == 3


def test_fixture_b_invalidation_state():
    df = _fixture_b()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    lev = next(l for l in registry.values() if l.side == SR_SIDE_SUPPORT)
    assert lev.state in (SR_STATE_INVALIDATED, SR_STATE_RETIRED)
    assert lev.invalidation_idx == 25


def test_fixture_b_support_broken_flag():
    df = _fixture_b()
    out = add_sr_levels(df, include_research_only=False)
    assert int(out["support_broken_this_bar"].iloc[25]) == 1
    # Not set before bar 25
    assert int(out["support_broken_this_bar"].iloc[24]) == 0


def test_fixture_b_invalidation_bars_before_confirm():
    """A level must NOT appear at bars before its source_live_from_idx."""
    df = _fixture_b()
    out = add_sr_levels(df, include_research_only=False)
    # Support confirm at bar 5 — bars 0-4 must have no nearest support
    for bar in range(5):
        assert int(out["nearest_support_active"].iloc[bar]) == 0


# ---------------------------------------------------------------------------
# Fixture C — deduplication / supersession
# ---------------------------------------------------------------------------


def _fixture_c() -> pd.DataFrame:
    """
    40-bar frame.  Two swing highs confirmed close together at bars 10 and 11.
    First: price=105.0 (weaker source, origin close to confirm).
    Second: price=105.5 (stronger source — larger origin-to-confirm distance).

    MERGE_TOL = 1.0 ATR-unit price distance.
    |105.5 − 105.0| = 0.5 < 1.0 → deduplication fires.
    The weaker level is retired as superseded.
    """
    n = 40
    df = _base_candle(n, close=100.0, low=99.0, high=106.0)
    # Swing high 1 at bar 10: origin at bar 10 (same bar → mag_atr=1.5 default)
    # Swing high 2 at bar 11: origin at bar 5 (5 bars back, price diff = 6.0 → mag=1.2 ATR)
    # Stronger source: let origin_bar differ significantly so mag_atr > 1.5

    # high_1 at bar 10: origin_bar=10 (no origin distance → src_strength=1.5/3=0.50)
    if "swing_high_confirm_flag" not in df.columns:
        df["swing_high_confirm_flag"] = 0
        df["swing_high_confirm_price"] = np.nan
        df["swing_high_confirm_origin_idx"] = np.nan
        df["swing_low_confirm_flag"] = 0
        df["swing_low_confirm_price"] = np.nan
        df["swing_low_confirm_origin_idx"] = np.nan

    df.at[10, "swing_high_confirm_flag"] = 1
    df.at[10, "swing_high_confirm_price"] = 105.0
    df.at[10, "swing_high_confirm_origin_idx"] = 10  # same → mag default=1.5 → src=0.50

    # high_2 at bar 11: origin at bar 5, close[5]=100 → mag=(105.5-100)/5=1.1 → src=0.367
    # So high_1 (src=0.50) is STRONGER than high_2 (src=0.367)
    df.at[11, "swing_high_confirm_flag"] = 1
    df.at[11, "swing_high_confirm_price"] = 105.5
    df.at[11, "swing_high_confirm_origin_idx"] = 5

    return df


def test_fixture_c_dedup_one_survivor():
    df = _fixture_c()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    res_levels = [l for l in registry.values() if l.side == SR_SIDE_RESISTANCE]
    active = [l for l in res_levels if l.state == SR_STATE_ACTIVE]
    retired = [l for l in res_levels if l.state == SR_STATE_RETIRED]
    # Exactly one active, one retired (superseded)
    assert len(active) == 1
    assert len(retired) == 1


def test_fixture_c_superseded_by_set():
    df = _fixture_c()
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    res_levels = [l for l in registry.values() if l.side == SR_SIDE_RESISTANCE]
    retired = [l for l in res_levels if l.state == SR_STATE_RETIRED]
    assert len(retired) == 1
    assert retired[0].superseded_by is not None


def test_fixture_c_levels_not_created_before_confirm():
    """No resistance projected before bar 10."""
    df = _fixture_c()
    out = add_sr_levels(df, include_research_only=False)
    for bar in range(10):
        assert int(out["nearest_resistance_active"].iloc[bar]) == 0


# ---------------------------------------------------------------------------
# Fixture D — range context flags
# ---------------------------------------------------------------------------


def _fixture_d() -> pd.DataFrame:
    """
    30-bar frame with:
      - swing low  at bar 5 → support at 90.0
      - swing high at bar 5 → resistance at 110.0
    Bars 10+: price held in range at close=100 — bounded between 90 and 110.
    Expected: between_nearest_sr_flag=1, flags correct.
    """
    n = 30
    df = _base_candle(n, close=100.0, low=99.0, high=101.0)

    df["swing_high_confirm_flag"] = 0
    df["swing_high_confirm_price"] = np.nan
    df["swing_high_confirm_origin_idx"] = np.nan
    df["swing_low_confirm_flag"] = 0
    df["swing_low_confirm_price"] = np.nan
    df["swing_low_confirm_origin_idx"] = np.nan

    df.at[5, "swing_low_confirm_flag"] = 1
    df.at[5, "swing_low_confirm_price"] = 90.0
    df.at[5, "swing_low_confirm_origin_idx"] = 2

    df.at[5, "swing_high_confirm_flag"] = 1
    df.at[5, "swing_high_confirm_price"] = 110.0
    df.at[5, "swing_high_confirm_origin_idx"] = 2

    return df


def test_fixture_d_between_flag():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=False)
    # From bar 6 onward (after activation at bar 5) both levels exist
    assert int(out["between_nearest_sr_flag"].iloc[6]) == 1
    assert int(out["between_nearest_sr_flag"].iloc[29]) == 1


def test_fixture_d_not_above_resistance():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=False)
    # Price=100 is below resistance at 110 → above_nearest_resistance_flag=0
    assert int(out["above_nearest_resistance_flag"].iloc[10]) == 0


def test_fixture_d_not_below_support():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=False)
    # Price=100 is above support at 90 → below_nearest_support_flag=0
    assert int(out["below_nearest_support_flag"].iloc[10]) == 0


def test_fixture_d_distances_correct():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=False)
    bar = 10
    # Support distance = 100 - 90 = 10.0
    assert float(out["nearest_support_distance"].iloc[bar]) == pytest.approx(10.0)
    # Resistance distance = 110 - 100 = 10.0
    assert float(out["nearest_resistance_distance"].iloc[bar]) == pytest.approx(10.0)
    # ATR-normalized: 10 / 5 = 2.0
    assert float(out["nearest_support_distance_atr"].iloc[bar]) == pytest.approx(2.0)
    assert float(out["nearest_resistance_distance_atr"].iloc[bar]) == pytest.approx(2.0)


def test_fixture_d_above_resistance_when_price_escapes():
    """When close > resistance price → above_nearest_resistance_flag = 1."""
    df = _fixture_d()
    # Push price above resistance
    for bar in range(20, 30):
        df.at[bar, "close"] = 115.0
        df.at[bar, "high"] = 116.0
        df.at[bar, "low"] = 114.0
    out = add_sr_levels(df, include_research_only=False)
    # Resistance should be invalidated by bar 20 (close=115 > 110+0.5)
    assert int(out["above_nearest_resistance_flag"].iloc[25]) == 1


# ---------------------------------------------------------------------------
# Fixture E — prior-period family (day H/L)
# ---------------------------------------------------------------------------


def _fixture_e() -> pd.DataFrame:
    """
    40-bar frame.  No swing columns — only prev_day_high / prev_day_low.
    Period 1: bars 0-9  → no value (NaN)
    Period 2: bars 10-19 → prev_day_high=108.0, prev_day_low=92.0
    Period 3: bars 20-39 → prev_day_high=109.0, prev_day_low=91.0
    """
    n = 40
    df = _base_candle(n, close=100.0, low=99.0, high=101.0)
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_high[10:20] = 108.0
    prev_day_low[10:20] = 92.0
    prev_day_high[20:] = 109.0
    prev_day_low[20:] = 91.0
    df["prev_day_high"] = prev_day_high
    df["prev_day_low"] = prev_day_low
    return df


def test_fixture_e_no_level_before_first_value():
    df = _fixture_e()
    out = add_sr_levels(df, include_research_only=False)
    for bar in range(10):
        assert int(out["nearest_resistance_active"].iloc[bar]) == 0
        assert int(out["nearest_support_active"].iloc[bar]) == 0


def test_fixture_e_level_active_at_first_value():
    df = _fixture_e()
    out = add_sr_levels(df, include_research_only=False)
    # At bar 10 levels activate (source_live_from_idx=10)
    assert int(out["nearest_resistance_active"].iloc[10]) == 1
    assert int(out["nearest_support_active"].iloc[10]) == 1


def test_fixture_e_correct_prices():
    df = _fixture_e()
    out = add_sr_levels(df, include_research_only=False)
    assert float(out["nearest_resistance_price"].iloc[15]) == pytest.approx(108.0)
    assert float(out["nearest_support_price"].iloc[15]) == pytest.approx(92.0)


def test_fixture_e_family_is_day():
    df = _fixture_e()
    out = add_sr_levels(df, include_research_only=False)
    assert out["nearest_resistance_source_family"].iloc[15] == SR_FAMILY_DAY
    assert out["nearest_support_source_family"].iloc[15] == SR_FAMILY_DAY


def test_fixture_e_new_period_creates_new_level():
    df = _fixture_e()
    registry = build_sr_level_registry(df)
    # Should have 4 source events: 2 for period 2, 2 for period 3
    day_events = [e for e in registry.values() if e.source_family == SR_FAMILY_DAY]
    assert len(day_events) == 4


# ---------------------------------------------------------------------------
# Research / live parity
# ---------------------------------------------------------------------------


def test_parity_live_columns_match_research():
    """Live columns in the research build must equal those in the live build."""
    df = _fixture_a()
    live_df = add_sr_levels(df, include_research_only=False)
    research_df = add_sr_levels(df, include_research_only=True)
    live_cols = [c for c in research_df.columns if not c.startswith("r_")]
    shared = [c for c in live_cols if c in live_df.columns]
    pd.testing.assert_frame_equal(live_df[shared], research_df[shared])


def test_research_has_r_prefix_columns():
    df = _fixture_a()
    out = add_sr_levels(df, include_research_only=True)
    r_cols = [c for c in out.columns if c.startswith("r_")]
    assert len(r_cols) > 0


def test_live_has_no_r_prefix_columns():
    df = _fixture_a()
    out = add_sr_levels(df, include_research_only=False)
    r_cols = [c for c in out.columns if c.startswith("r_")]
    assert r_cols == []


# ---------------------------------------------------------------------------
# Edge cases — missing families / empty frame
# ---------------------------------------------------------------------------


def test_empty_frame_returns_empty():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "atr_14"])
    out = add_sr_levels(df, include_research_only=False)
    assert len(out) == 0


def test_no_swing_columns_no_swing_levels():
    df = _base_candle(20)
    events = extract_sr_source_events(df)
    swing_events = [e for e in events if e.source_family == SR_FAMILY_SWING]
    assert swing_events == []


def test_warmup_rows_have_no_nearest_sr():
    """Bars before the first confirm row have no active nearest SR."""
    df = _fixture_a()  # confirm at bar 5
    out = add_sr_levels(df, include_research_only=False)
    for bar in range(5):
        assert int(out["nearest_support_active"].iloc[bar]) == 0
        assert int(out["nearest_resistance_active"].iloc[bar]) == 0


# ---------------------------------------------------------------------------
# Strength model
# ---------------------------------------------------------------------------


def test_strength_in_unit_interval():
    lev = SRLevel(
        level_id=1,
        side=SR_SIDE_SUPPORT,
        level_price=100.0,
        source_family=SR_FAMILY_SWING,
        source_strength_initial=0.8,
    )
    lev.age_bars = 10
    lev.touch_count = 2
    lev.weaken_count = 0
    s = _compute_strength(lev)
    assert 0.0 <= s <= 1.0


def test_strength_decreases_with_weakens():
    lev = SRLevel(
        level_id=1,
        side=SR_SIDE_SUPPORT,
        level_price=100.0,
        source_family=SR_FAMILY_SWING,
        source_strength_initial=0.8,
    )
    lev.age_bars = 5
    s_no_weaken = _compute_strength(lev)
    lev.weaken_count = 3
    s_weakened = _compute_strength(lev)
    assert s_weakened < s_no_weaken


def test_strength_increases_with_touches():
    base = SRLevel(
        level_id=1,
        side=SR_SIDE_SUPPORT,
        level_price=100.0,
        source_family=SR_FAMILY_SWING,
        source_strength_initial=0.5,
    )
    base.age_bars = 5
    s_zero_touches = _compute_strength(base)
    base.touch_count = 4
    s_four_touches = _compute_strength(base)
    assert s_four_touches > s_zero_touches


def test_family_prior_used_in_strength():
    """Week-family level should have higher base strength than swing-family."""
    from src.indicators.foundation.sr_levels import SR_FAMILY_WEEK

    sw = SRLevel(
        level_id=1,
        side=SR_SIDE_RESISTANCE,
        level_price=100.0,
        source_family=SR_FAMILY_SWING,
        source_strength_initial=0.5,
    )
    wk = SRLevel(
        level_id=2,
        side=SR_SIDE_RESISTANCE,
        level_price=100.0,
        source_family=SR_FAMILY_WEEK,
        source_strength_initial=0.5,
    )
    assert _compute_strength(wk) > _compute_strength(sw)


# ---------------------------------------------------------------------------
# Retirement by age
# ---------------------------------------------------------------------------


def test_retirement_by_max_age():
    """A level that ages beyond MAX_AGE_BARS must be retired."""
    n = MAX_AGE_BARS + 20
    df = _base_candle(n, close=100.0, low=99.0, high=101.0)
    _swing_confirm_at(df, bar=0, price=90.0, side=SR_SIDE_SUPPORT, origin_bar=0)
    registry = build_sr_level_registry(df)
    update_sr_lifecycle(df, registry)
    lev = next(l for l in registry.values() if l.side == SR_SIDE_SUPPORT)
    assert lev.state == SR_STATE_RETIRED


# ---------------------------------------------------------------------------
# Causality (no future look-ahead)
# ---------------------------------------------------------------------------


def test_no_level_projected_before_source_live_from():
    """Nearest support must not appear before its source_live_from_idx."""
    df = _fixture_a()  # support live from bar 5
    out = add_sr_levels(df, include_research_only=False)
    # Bars 0-4: before activation — must be inactive
    for bar in range(5):
        assert int(out["nearest_support_active"].iloc[bar]) == 0
    # Bar 5: activation row — level is immediately projected (correct causal behaviour)
    assert int(out["nearest_support_active"].iloc[5]) == 1


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------


def test_no_label_future_columns_in_live():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=False)
    bad = [c for c in out.columns if c.startswith("label_") or c.startswith("future_")]
    assert bad == []


def test_no_label_future_columns_in_research():
    df = _fixture_d()
    out = add_sr_levels(df, include_research_only=True)
    bad = [c for c in out.columns if c.startswith("label_") or c.startswith("future_")]
    assert bad == []
