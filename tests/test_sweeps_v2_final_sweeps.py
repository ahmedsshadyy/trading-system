"""Tests for Step 11 — final sweeps detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.sweeps_v2.final_sweeps import (
    FINAL_SWEEPS_COLUMNS,
    SWEEP_CLASS_ACCEPTED_BREAKOUT,
    SWEEP_CLASS_DELAYED_REJECTION,
    SWEEP_CLASS_SAME_BAR,
    SWEEP_CLASS_UNRESOLVED,
    add_final_sweeps,
    step11d_default_kwargs,
    step11e_default_kwargs,
)
from src.indicators.sweeps_v2.unified_sources import (
    LIQ_LADDER_DEPTH,
    add_unified_liquidity_sources,
)

# ---------------------------------------------------------------------------
# Synthetic frame builder — direct ladder construction (no upstream pipeline)
# ---------------------------------------------------------------------------


def _frame_with_one_above_source(
    rows: list[tuple[float, float, float, float]],
    *,
    source_level: float,
    source_zone_width: float = 0.0,
    source_strength: float = 0.6,
    source_family: str = "resistance",
    source_age_start: int = 5,
) -> pd.DataFrame:
    """Build a minimal frame with a single liquidity cluster in the
    ``liq_above_l1_*`` slot at every bar.
    """

    n = len(rows)
    cols: dict[str, object] = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "atr_14": np.full(n, 1.0, dtype=float),
    }

    half = source_zone_width / 2.0
    cols["liq_above_l1_cluster_id"] = np.full(n, 101.0, dtype=float)
    cols["liq_above_l1_level"] = np.full(n, source_level, dtype=float)
    cols["liq_above_l1_zone_low"] = np.full(n, source_level - half, dtype=float)
    cols["liq_above_l1_zone_high"] = np.full(n, source_level + half, dtype=float)
    cols["liq_above_l1_is_zone"] = np.full(n, 1.0 if source_zone_width > 0 else 0.0)
    cols["liq_above_l1_width_abs"] = np.full(n, source_zone_width, dtype=float)
    cols["liq_above_l1_width_atr"] = np.full(n, source_zone_width, dtype=float)
    cols["liq_above_l1_strength"] = np.full(n, source_strength, dtype=float)
    cols["liq_above_l1_state"] = np.full(n, 2.0, dtype=float)  # ACTIVE
    cols["liq_above_l1_age_bars"] = np.arange(
        source_age_start, source_age_start + n, dtype=float
    )
    cols["liq_above_l1_freshness"] = np.full(n, 0.5, dtype=float)
    cols["liq_above_l1_touch_count"] = np.zeros(n, dtype=float)
    cols["liq_above_l1_signed_dist_atr"] = np.full(n, 1.0, dtype=float)
    cols["liq_above_l1_member_count"] = np.ones(n, dtype=float)
    cols["liq_above_l1_origin_idx"] = np.full(n, -float(source_age_start), dtype=float)
    cols["liq_above_l1_active_start_idx"] = np.full(
        n, -float(source_age_start), dtype=float
    )
    cols["liq_above_l1_primary_family"] = np.array([source_family] * n, dtype=object)
    cols["liq_above_l1_attribution_families"] = np.array(
        [source_family] * n, dtype=object
    )

    # All other slots are empty: cluster_id NaN signals "not active".
    for side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            if side == "above" and rank == 1:
                continue
            for f in (
                "cluster_id",
                "level",
                "zone_low",
                "zone_high",
                "is_zone",
                "width_abs",
                "width_atr",
                "strength",
                "state",
                "age_bars",
                "freshness",
                "touch_count",
                "signed_dist_atr",
                "member_count",
                "origin_idx",
                "active_start_idx",
            ):
                cols[f"liq_{side}_l{rank}_{f}"] = np.full(n, np.nan, dtype=float)
            cols[f"liq_{side}_l{rank}_primary_family"] = np.array(
                [""] * n, dtype=object
            )
            cols[f"liq_{side}_l{rank}_attribution_families"] = np.array(
                [""] * n, dtype=object
            )

    cols["liq_active_total_count"] = np.full(n, 1.0)
    cols["liq_active_above_count"] = np.full(n, 1.0)
    cols["liq_active_below_count"] = np.full(n, 0.0)
    cols["liq_dropped_by_crowding_count"] = np.full(n, 0.0)
    cols["liq_dropped_by_dominance_count"] = np.full(n, 0.0)
    cols["liq_nearest_above_dist_atr"] = np.full(n, 1.0)
    cols["liq_nearest_below_dist_atr"] = np.full(n, np.nan)
    cols["liq_top_above_strength"] = np.full(n, source_strength)
    cols["liq_top_below_strength"] = np.full(n, np.nan)
    cols["liq_source_timeframe"] = np.array(["H4"] * n, dtype=object)
    cols["liq_mtf_policy"] = np.array(["same_timeframe_only"] * n, dtype=object)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_emits_canonical_columns() -> None:
    df = _frame_with_one_above_source(
        [(99.0, 100.0, 98.5, 99.5)] * 5, source_level=101.0
    )
    out = add_final_sweeps(df)
    for col in FINAL_SWEEPS_COLUMNS:
        assert col in out.columns, f"missing {col!r}"


# ---------------------------------------------------------------------------
# Same-bar sweep
# ---------------------------------------------------------------------------


def test_same_bar_sweep_when_wick_pierces_and_close_returns() -> None:
    """Bar 1: high pierces 101 but close stays below → same-bar sweep.

    Bar 0 establishes context; bar 1 is the breach + same-bar reclaim.
    """

    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),  # bar 0: no breach
            (99.5, 102.0, 99.0, 100.5),  # bar 1: wick > 101, close 100.5 < 101
            (100.0, 101.0, 99.0, 100.0),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    # Bar 1 must mark a sweep, class = same_bar.
    assert out["sweep_flag"].iloc[1] == 1.0
    assert int(out["sweep_class"].iloc[1]) == SWEEP_CLASS_SAME_BAR
    assert int(out["sweep_breach_idx"].iloc[1]) == 1
    assert int(out["sweep_confirm_idx"].iloc[1]) == 1
    assert out["sweep_latency_bars"].iloc[1] == 0.0
    # Source attribution recorded
    assert out["sweep_primary_family"].iloc[1] == "resistance"
    # Mechanics: penetration_abs > 0
    assert out["penetration_abs"].iloc[1] > 0.0


# ---------------------------------------------------------------------------
# Delayed-rejection sweep
# ---------------------------------------------------------------------------


def test_delayed_rejection_sweep_within_window() -> None:
    """Bar 1 closes above (acceptance), bar 2 reclaims back below.

    Within the 3-bar default window → delayed_rejection_sweep.
    """

    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.5, 99.0, 102.0),  # close above source
            (101.5, 102.0, 100.0, 100.5),  # reclaim back below
            (100.0, 101.0, 99.5, 100.0),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    # Confirm bar = bar 2.
    assert out["sweep_flag"].iloc[2] == 1.0
    assert int(out["sweep_class"].iloc[2]) == SWEEP_CLASS_DELAYED_REJECTION
    assert int(out["sweep_breach_idx"].iloc[2]) == 1
    assert int(out["sweep_confirm_idx"].iloc[2]) == 2
    assert out["sweep_latency_bars"].iloc[2] == 1.0


# ---------------------------------------------------------------------------
# Accepted breakout
# ---------------------------------------------------------------------------


def test_accepted_breakout_when_no_reclaim_in_window() -> None:
    """Bar 1 closes above the source; bars 2-5 stay above. After the 3-bar
    window closes without reclaim → accepted_breakout."""

    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.5, 99.0, 102.0),  # breach + close above
            (102.0, 103.0, 101.5, 102.5),
            (102.5, 103.5, 102.0, 103.0),
            (103.0, 104.0, 102.5, 103.5),  # window closed
            (103.5, 104.5, 103.0, 104.0),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    # The accepted-breakout terminal stamp should appear at bar 5 (one after
    # the confirmation_window expires at bar 4 = breach 1 + 3).
    accepted_bars = out.index[
        out["sweep_class"] == SWEEP_CLASS_ACCEPTED_BREAKOUT
    ].tolist()
    assert len(accepted_bars) >= 1
    # Sweep flag must NOT be 1 on the accepted-breakout bar (no confirmed
    # rejection).
    assert (out.loc[accepted_bars, "sweep_flag"].fillna(0) == 0).all()


# ---------------------------------------------------------------------------
# Unresolved breach
# ---------------------------------------------------------------------------


def test_unresolved_breach_when_window_open_at_end() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.0, 99.0, 101.5),  # breach close above
            (101.5, 102.0, 101.0, 101.7),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    # Last bar should have UNRESOLVED stamp.
    # Either marked unresolved on last bar OR the breach class still pending.
    classes = out["sweep_class"].dropna().astype(int).tolist()
    assert SWEEP_CLASS_UNRESOLVED in classes


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_causality_breach_le_confirm() -> None:
    """For every confirmed sweep, breach_idx <= confirm_idx."""

    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.0, 99.0, 100.5),
            (100.0, 102.5, 99.5, 102.0),
            (102.0, 102.5, 101.0, 101.5),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    confirmed = out[out["sweep_flag"].fillna(0) > 0]
    # At least one confirmed sweep, all of which must respect causality.
    for _, row in confirmed.iterrows():
        assert row["sweep_breach_idx"] <= row["sweep_confirm_idx"]


# ---------------------------------------------------------------------------
# No sources → no sweep
# ---------------------------------------------------------------------------


def test_no_sources_means_no_sweep() -> None:
    """If every ladder slot is empty, the detector emits zero sweeps."""

    df = _frame_with_one_above_source(
        [(99.0, 100.0, 98.5, 99.5)] * 5, source_level=101.0
    )
    # Wipe the only source.
    df["liq_above_l1_cluster_id"] = np.nan
    out = add_final_sweeps(df)
    assert out["sweep_flag"].fillna(0).sum() == 0
    assert out["sweep_breach_flag"].fillna(0).sum() == 0


def test_fresh_resistance_is_not_yet_sweep_eligible() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.0, 99.0, 100.5),
            (100.0, 101.0, 99.0, 100.0),
        ],
        source_level=101.0,
        source_age_start=0,
    )
    out = add_final_sweeps(df)
    assert out["sweep_flag"].fillna(0).sum() == 0
    assert out["sweep_breach_flag"].fillna(0).sum() == 0


# ---------------------------------------------------------------------------
# Tradeable filter
# ---------------------------------------------------------------------------


def test_tradeable_flag_set_only_when_quality_passes() -> None:
    """High-strength source + clean rejection → tradeable_candidate=1."""

    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.0, 99.0, 100.5),  # same-bar sweep
            (100.0, 101.0, 99.0, 100.0),
        ],
        source_level=101.0,
        source_strength=0.8,
    )
    out = add_final_sweeps(df)
    assert int(out["sweep_is_tradeable_candidate"].iloc[1]) == 1


def test_consumed_source_instance_does_not_refire_without_reactivation() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.2, 99.0, 100.4),  # confirmed same-bar sweep
            (99.8, 100.4, 99.2, 99.9),
            (99.7, 102.3, 99.1, 100.3),  # would otherwise sweep again
            (99.8, 100.5, 99.0, 99.9),
        ],
        source_level=101.0,
        source_strength=0.8,
    )
    out = add_final_sweeps(df)
    assert int(out["sweep_flag"].fillna(0).sum()) == 1


def test_step11d_session_requires_prior_separation() -> None:
    df = _frame_with_one_above_source(
        [
            (100.0, 100.1, 99.8, 100.0),
            (100.0, 100.1, 99.8, 100.0),
            (100.0, 100.1, 99.8, 100.0),
            (100.0, 100.1, 99.8, 100.0),  # prior distance only 0.2 ATR throughout
            (100.0, 100.9, 99.7, 100.1),  # would otherwise same-bar sweep
            (100.0, 100.2, 99.8, 100.0),
        ],
        source_level=100.2,
        source_strength=0.8,
        source_family="session_high",
        source_age_start=0,
    )
    out = add_final_sweeps(df, **step11d_default_kwargs())
    assert int(out["sweep_flag"].fillna(0).sum()) == 0


def test_step11d_session_tradeable_requires_volume_or_displacement() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 99.2, 98.8, 99.0),
            (99.0, 99.3, 98.9, 99.0),
            (99.1, 99.4, 98.9, 99.1),
            (99.1, 99.5, 99.0, 99.2),
            (99.0, 101.8, 98.9, 100.8),  # same-bar sweep, strong enough otherwise
            (100.0, 100.4, 99.6, 100.0),
        ],
        source_level=101.0,
        source_strength=0.8,
        source_family="session_high",
        source_age_start=0,
    )
    out = add_final_sweeps(df, **step11d_default_kwargs())
    assert int(out["sweep_flag"].iloc[4]) == 1
    assert int(out["sweep_is_tradeable_candidate"].iloc[4]) == 0


def test_step11e_labels_close_interaction_as_micro_and_not_tradeable() -> None:
    df = _frame_with_one_above_source(
        [
            (100.70, 100.90, 100.60, 100.85),
            (100.85, 101.30, 100.70, 100.90),  # breach from only 0.15 ATR away
            (100.80, 100.95, 100.70, 100.82),
        ],
        source_level=101.0,
        source_strength=0.8,
    )
    df["volume_ratio_20"] = np.full(len(df), 1.5, dtype=float)
    out = add_final_sweeps(df, **step11e_default_kwargs())
    assert int(out["sweep_flag"].iloc[1]) == 1
    assert int(out["sweep_is_micro_interaction"].iloc[1]) == 1
    assert out["sweep_selectivity_class"].iloc[1] == "micro_interaction_sweep"
    assert int(out["sweep_is_tradeable_candidate"].iloc[1]) == 0


def test_step11e_exceptional_impulse_can_escape_micro_bucket() -> None:
    df = _frame_with_one_above_source(
        [
            (100.70, 100.90, 100.60, 100.85),
            (
                100.85,
                101.80,
                100.10,
                100.20,
            ),  # deep same-bar sweep with strong rejection
            (100.20, 100.60, 100.00, 100.30),
        ],
        source_level=101.0,
        source_strength=0.8,
    )
    df["volume_ratio_20"] = np.full(len(df), 1.5, dtype=float)
    out = add_final_sweeps(df, **step11e_default_kwargs())
    assert int(out["sweep_flag"].iloc[1]) == 1
    assert int(out["sweep_is_standard_liquidity"].iloc[1]) == 1
    assert out["sweep_selectivity_class"].iloc[1] == "tradeable_sweep_candidate"
    assert int(out["sweep_is_tradeable_candidate"].iloc[1]) == 1


def test_step11e_upgrades_standard_sweep_to_displacement_confirmed() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.5, 99.0, 102.0),  # breach close above
            (101.5, 102.0, 100.0, 100.5),  # reclaim
            (100.4, 100.5, 99.4, 99.5),  # bearish displacement after confirm
            (99.4, 99.6, 99.0, 99.2),
            (99.2, 99.4, 98.8, 99.0),
            (99.0, 99.3, 98.7, 98.9),
            (98.9, 99.2, 98.6, 98.8),
        ],
        source_level=101.0,
        source_strength=0.8,
    )
    df["displacement_flag"] = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    out = add_final_sweeps(df, **step11e_default_kwargs())
    assert int(out["sweep_flag"].iloc[2]) == 1
    assert int(out["sweep_is_standard_liquidity"].iloc[2]) == 1
    assert int(out["sweep_is_displacement_confirmed"].iloc[2]) == 1
    assert out["sweep_selectivity_class"].iloc[2] == "displacement_confirmed_sweep"


# ---------------------------------------------------------------------------
# Quality components stay in [0,1]
# ---------------------------------------------------------------------------


def test_quality_score_bounded_zero_to_one() -> None:
    df = _frame_with_one_above_source(
        [
            (99.0, 100.0, 98.5, 99.5),
            (99.5, 102.0, 99.0, 100.5),
        ],
        source_level=101.0,
    )
    out = add_final_sweeps(df)
    qs = out["sweep_quality_score"].dropna()
    assert ((qs >= 0.0) & (qs <= 1.0)).all()


# ---------------------------------------------------------------------------
# End-to-end: composition with unified_sources
# ---------------------------------------------------------------------------


def test_full_stack_unified_then_final_sweeps_no_range_boundary() -> None:
    """Sanity check: drive both stages, verify no range_boundary attribution."""

    n = 50
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close + rng.normal(0.0, 0.1, n),
            "high": close + np.abs(rng.normal(0.0, 1.0, n)),
            "low": close - np.abs(rng.normal(0.0, 1.0, n)),
            "close": close,
            "atr_14": np.full(n, 1.5, dtype=float),
            "last_swing_high": close + 3.0,
            "last_swing_low": close - 3.0,
            "last_swing_high_idx": np.arange(n, dtype=float),
            "last_swing_low_idx": np.arange(n, dtype=float),
            "swing_high_age": np.full(n, 5.0, dtype=float),
            "swing_low_age": np.full(n, 5.0, dtype=float),
        }
    )
    out = add_unified_liquidity_sources(df, scan_timeframe="H4", instrument="XAU_USD")
    out = add_final_sweeps(out)
    fams = (
        out["sweep_primary_family"]
        .astype(str)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    for fam in fams:
        assert not fam.startswith("range_")
        assert not fam.startswith("fvg_")
        assert not fam.startswith("ob_")
