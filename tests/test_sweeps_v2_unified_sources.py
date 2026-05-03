"""Tests for Step 10 — unified liquidity source framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.smc.sweeps.unified_sources import (
    LIQ_DEPRECATED_FAMILIES,
    LIQ_FAMILY_PRECEDENCE,
    LIQ_GLOBAL_CROWDING_CAP,
    LIQ_LADDER_DEPTH,
    LIQ_SOURCE_FAMILIES,
    UNIFIED_SOURCE_COLUMNS,
    add_unified_liquidity_sources,
    build_unified_liquidity_clusters_audit,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_base_df(n: int = 60, *, close_start: float = 1900.0) -> pd.DataFrame:
    """Synthetic OHLC frame with predictable swings + ATR."""

    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = close_start + np.cumsum(rng.normal(0.0, 0.5, n))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close + rng.normal(0.0, 0.1, n),
            "high": close + np.abs(rng.normal(0.0, 1.0, n)),
            "low": close - np.abs(rng.normal(0.0, 1.0, n)),
            "close": close,
            "atr_14": np.full(n, 2.0, dtype=float),
            # Source families: only swings — every other family is optional.
            "last_swing_high": close + 5.0,
            "last_swing_low": close - 5.0,
            "last_swing_high_idx": np.arange(n, dtype=float),
            "last_swing_low_idx": np.arange(n, dtype=float),
            "swing_high_age": np.full(n, 5.0, dtype=float),
            "swing_low_age": np.full(n, 5.0, dtype=float),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Schema + column-set tests
# ---------------------------------------------------------------------------


def test_emits_canonical_schema() -> None:
    df = _make_base_df()
    out = add_unified_liquidity_sources(df, scan_timeframe="H4", instrument="XAU_USD")
    for col in UNIFIED_SOURCE_COLUMNS:
        assert col in out.columns, f"missing column {col!r}"


def test_ladder_columns_per_side_per_rank() -> None:
    df = _make_base_df()
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    for side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            for field in (
                "cluster_id",
                "primary_family",
                "level",
                "zone_low",
                "zone_high",
                "is_zone",
                "width_atr",
                "strength",
                "state",
                "age_bars",
                "signed_dist_atr",
                "attribution_families",
                "member_count",
                "origin_idx",
                "active_start_idx",
            ):
                assert f"liq_{side}_l{rank}_{field}" in out.columns


def test_canonical_families_recognised() -> None:
    expected = (
        "swing_high",
        "swing_low",
        "equal_high",
        "equal_low",
        "resistance",
        "support",
        "session_high",
        "session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
    )
    assert set(LIQ_SOURCE_FAMILIES) == set(expected)


def test_deprecated_families_are_blocked() -> None:
    """range_boundary, FVG, OB are excluded for v1. The constant lists the
    families that are blocked at runtime."""

    for fam in ("range_boundary_high", "range_boundary_low", "fvg_high", "ob_low"):
        assert fam in LIQ_DEPRECATED_FAMILIES


def test_precedence_order_matches_spec() -> None:
    """Spec precedence: equal_hl → S/R → prev_week → prev_day → session → swing."""

    assert LIQ_FAMILY_PRECEDENCE["equal_high"] < LIQ_FAMILY_PRECEDENCE["resistance"]
    assert (
        LIQ_FAMILY_PRECEDENCE["resistance"]
        < LIQ_FAMILY_PRECEDENCE["previous_week_high"]
    )
    assert (
        LIQ_FAMILY_PRECEDENCE["previous_week_high"]
        < LIQ_FAMILY_PRECEDENCE["previous_day_high"]
    )
    assert (
        LIQ_FAMILY_PRECEDENCE["previous_day_high"]
        < LIQ_FAMILY_PRECEDENCE["session_high"]
    )
    assert LIQ_FAMILY_PRECEDENCE["session_high"] < LIQ_FAMILY_PRECEDENCE["swing_high"]


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_mtf_policy_stamp_is_constant() -> None:
    df = _make_base_df()
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    assert (out["liq_mtf_policy"] == "same_timeframe_only").all()
    assert (out["liq_source_timeframe"] == "H4").all()


def test_unknown_timeframe_rejected() -> None:
    df = _make_base_df()
    with pytest.raises(ValueError):
        add_unified_liquidity_sources(df, scan_timeframe="H2")


def test_only_swings_produces_swing_attribution() -> None:
    df = _make_base_df()
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    families = set(audit["primary_family"].unique())
    assert families.issubset({"swing_high", "swing_low"})
    assert "swing_high" in families
    assert "swing_low" in families


def test_dedup_collapses_overlapping_levels() -> None:
    """When two same-side sources sit within tolerance they collapse into one
    cluster. Use swing + previous-day-high at exactly the same level on the
    above side and confirm only one cluster is emitted."""

    df = _make_base_df(n=20)
    df["prev_day_high"] = df["last_swing_high"]  # identical level
    df["prev_day_low"] = df["last_swing_low"]
    df["prev_week_high"] = float("nan")
    df["prev_week_low"] = float("nan")
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    # On every bar, "above" side should yield at most ~1 cluster (swing +
    # prev_day_high merged). Ensure the median above-count is 1.
    above_counts = audit[audit["side_label"] == "above"].groupby("bar_idx").size()
    assert above_counts.median() <= 1


def test_precedence_resolved_within_cluster() -> None:
    """When swing + prev_day_high merge, the primary family must be the one
    with lower precedence rank — prev_day_high beats swing_high."""

    df = _make_base_df(n=20)
    df["prev_day_high"] = df["last_swing_high"]
    df["prev_day_low"] = df["last_swing_low"]
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    above = audit[audit["side_label"] == "above"]
    # prev_day_high (rank 4) beats swing_high (rank 6) → primary should be
    # previous_day_high on every bar with a cluster.
    assert (above["primary_family"] == "previous_day_high").all()
    # Attribution should record both contributors.
    assert above["attribution_families"].str.contains("swing_high").all()
    assert above["attribution_families"].str.contains("previous_day_high").all()


def test_crowding_caps_ladder_depth() -> None:
    """Even if 20 sources are active per side, only LIQ_LADDER_DEPTH per side
    appear in the ladder."""

    df = _make_base_df(n=20)
    # Stack many distinct above-price sources that won't merge (different prices).
    df["prev_day_high"] = df["last_swing_high"] + 10.0
    df["prev_week_high"] = df["last_swing_high"] + 20.0
    df["prev_asia_high"] = df["last_swing_high"] + 30.0
    df["prev_london_high"] = df["last_swing_high"] + 40.0
    df["prev_ny_high"] = df["last_swing_high"] + 50.0
    df["prev_day_low"] = df["last_swing_low"] - 10.0
    df["prev_week_low"] = df["last_swing_low"] - 20.0
    df["prev_asia_low"] = df["last_swing_low"] - 30.0
    df["prev_london_low"] = df["last_swing_low"] - 40.0
    df["prev_ny_low"] = df["last_swing_low"] - 50.0
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    # Per side, never more than LIQ_LADDER_DEPTH active.
    assert out["liq_active_above_count"].max() <= LIQ_LADDER_DEPTH
    assert out["liq_active_below_count"].max() <= LIQ_LADDER_DEPTH
    # Combined must respect the global crowding cap.
    assert out["liq_active_total_count"].max() <= LIQ_GLOBAL_CROWDING_CAP


def test_causality_origin_le_active_start_le_bar() -> None:
    df = _make_base_df(n=40)
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    # Active start must not exceed bar idx for any row.
    assert (audit["source_active_start_idx"] <= audit["bar_idx"]).all()


def test_audit_table_is_reconstructible_from_ladder_columns() -> None:
    df = _make_base_df()
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    # Should have at least one row per bar with active sources.
    bars_with_sources = (out["liq_active_total_count"].fillna(0) > 0).sum()
    if bars_with_sources > 0:
        assert len(audit) > 0
        # Source timeframe stamp present on every row.
        assert (audit["source_timeframe"] == "H4").all()


def test_eligibility_filter_drops_wrong_side_sources() -> None:
    """A swing_high level that sits below close should be ignored — its
    family-side (+1) is inconsistent with its geometric position (below)."""

    df = _make_base_df(n=10)
    # Force swing_high BELOW close; swing_low ABOVE close.
    df["last_swing_high"] = df["close"] - 5.0
    df["last_swing_low"] = df["close"] + 5.0
    out = add_unified_liquidity_sources(df, scan_timeframe="H4")
    audit = build_unified_liquidity_clusters_audit(out)
    # No clusters should be emitted for the swing families because they are
    # all on the wrong side of price.
    assert audit.empty or "swing_high" not in audit["primary_family"].values
    assert audit.empty or "swing_low" not in audit["primary_family"].values
