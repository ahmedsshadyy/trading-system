"""Canonical sweep schema tests (Step-frozen alias contract).

These tests guard the additive Step-frozen contract surfaced by
:mod:`src.indicators.smc.sweeps.final_sweeps`. They exercise:

* every alias column listed in ``FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS``
  is emitted on every sweep_flag row,
* directional invariants (bullish ↔ swept-side below price, bearish ↔
  above price; flags are exclusive),
* derived numeric columns (close-reclaim ATR, wick-rejection ratio,
  body-reclaim ratio) read only the confirm bar,
* ``swept_source_idx`` <= ``sweep_breach_idx`` <= ``sweep_confirm_idx``
  (causality),
* ``swept_source_timestamp`` resolves through the source bar.

The fixture mirrors ``tests/test_sweeps_v2_final_sweeps.py`` — a minimal
direct-ladder frame so the detector runs without the upstream pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
    SWEEPS_CANONICAL_THRESHOLDS,
    add_final_sweeps,
    step11e_default_kwargs,
)
from src.indicators.smc.sweeps.unified_sources import LIQ_LADDER_DEPTH


def _frame_with_one_above_source(
    rows,
    *,
    source_level: float,
    source_zone_width: float = 0.0,
    source_strength: float = 0.6,
    source_family: str = "resistance",
    source_age_start: int = 5,
) -> pd.DataFrame:
    n = len(rows)
    cols: dict[str, object] = {
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "atr_14": np.full(n, 1.0, dtype=float),
    }
    for side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            for fname in (
                "cluster_id",
                "level",
                "zone_low",
                "zone_high",
                "strength",
                "age_bars",
                "origin_idx",
                "active_start_idx",
                "primary_family",
                "attribution_families",
            ):
                col = f"liq_{side}_l{rank}_{fname}"
                if fname.endswith("family") or fname.endswith("families"):
                    cols[col] = [""] * n
                else:
                    cols[col] = [np.nan] * n
    df = pd.DataFrame(cols)
    df["liq_above_l1_cluster_id"] = 1.0
    df["liq_above_l1_level"] = source_level
    df["liq_above_l1_zone_low"] = source_level - source_zone_width / 2.0
    df["liq_above_l1_zone_high"] = source_level + source_zone_width / 2.0
    df["liq_above_l1_strength"] = source_strength
    df["liq_above_l1_origin_idx"] = 0.0
    df["liq_above_l1_active_start_idx"] = 0.0
    df["liq_above_l1_age_bars"] = [max(source_age_start + i, 0) for i in range(n)]
    df["liq_above_l1_primary_family"] = source_family
    df["liq_above_l1_attribution_families"] = source_family
    return df


def _frame_with_one_below_source(
    rows,
    *,
    source_level: float,
    source_zone_width: float = 0.0,
    source_strength: float = 0.6,
    source_family: str = "support",
    source_age_start: int = 5,
) -> pd.DataFrame:
    df = _frame_with_one_above_source(
        rows,
        source_level=source_level,
        source_zone_width=source_zone_width,
        source_strength=source_strength,
        source_family=source_family,
        source_age_start=source_age_start,
    )
    # Move the source from the above ladder to the below ladder.
    for fname in (
        "cluster_id",
        "level",
        "zone_low",
        "zone_high",
        "strength",
        "age_bars",
        "origin_idx",
        "active_start_idx",
        "primary_family",
        "attribution_families",
    ):
        df[f"liq_below_l1_{fname}"] = df[f"liq_above_l1_{fname}"]
        df[f"liq_above_l1_{fname}"] = (
            np.nan if not isinstance(df[f"liq_below_l1_{fname}"].iloc[0], str) else ""
        )
    return df


def _bearish_sweep_frame() -> pd.DataFrame:
    rows = [
        (95.0, 96.0, 94.0, 95.5),
        (95.5, 96.5, 95.0, 96.0),
        (96.0, 97.0, 95.5, 96.5),
        (96.5, 100.5, 96.0, 99.5),  # wick through 100, close back inside
        (99.5, 99.8, 99.0, 99.3),
        (99.3, 99.5, 98.0, 98.5),
        (98.5, 99.0, 97.5, 98.0),
    ]
    return _frame_with_one_above_source(rows, source_level=100.0)


def _bullish_sweep_frame() -> pd.DataFrame:
    rows = [
        (105.0, 106.0, 104.0, 105.5),
        (105.5, 106.5, 105.0, 106.0),
        (106.0, 107.0, 105.5, 106.5),
        (105.5, 105.8, 99.5, 100.5),  # wick through 100 (below), close back above
        (100.5, 101.0, 100.2, 100.8),
        (100.8, 101.5, 100.0, 101.2),
        (101.2, 102.0, 100.8, 101.5),
    ]
    return _frame_with_one_below_source(rows, source_level=100.0)


def _run(df: pd.DataFrame) -> pd.DataFrame:
    return add_final_sweeps(df, **step11e_default_kwargs(cooldown_bars=10))


def test_canonical_alias_columns_emitted():
    out = _run(_bearish_sweep_frame())
    for col in FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS:
        assert col in out.columns


def test_bearish_sweep_writes_bearish_flag_and_direction():
    out = _run(_bearish_sweep_frame())
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    assert not sweep_rows.empty
    row = sweep_rows.iloc[0]
    assert row["sweep_direction"] == "bearish"
    assert row["bearish_sweep_flag"] == 1.0
    assert row["bullish_sweep_flag"] == 0.0
    assert row["swept_source_side"] == 1.0
    assert row["swept_source_family"] == "resistance"
    assert float(row["swept_level"]) == 100.0
    assert float(row["swept_source_strength"]) == 0.6
    # source_idx lookup picks up the origin bar.
    assert int(row["swept_source_idx"]) == 0
    # Source timestamp is the timestamp at the source bar.
    assert pd.Timestamp(row["swept_source_timestamp"]) == pd.Timestamp(
        out["timestamp"].iloc[0]
    )


def test_bullish_sweep_writes_bullish_flag_and_direction():
    out = _run(_bullish_sweep_frame())
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    assert not sweep_rows.empty
    row = sweep_rows.iloc[0]
    assert row["sweep_direction"] == "bullish"
    assert row["bearish_sweep_flag"] == 0.0
    assert row["bullish_sweep_flag"] == 1.0
    assert row["swept_source_side"] == -1.0
    assert row["swept_source_family"] == "support"
    assert float(row["swept_level"]) == 100.0


def test_bullish_and_bearish_flags_are_mutually_exclusive():
    # On any non-sweep bar both flags must be zero. On any sweep bar
    # exactly one is one and the other is zero — never both nonzero.
    for fixture in (_bearish_sweep_frame(), _bullish_sweep_frame()):
        out = _run(fixture)
        bull = pd.to_numeric(out["bullish_sweep_flag"], errors="coerce").fillna(0)
        bear = pd.to_numeric(out["bearish_sweep_flag"], errors="coerce").fillna(0)
        # No bar has both flags set.
        assert int(((bull > 0) & (bear > 0)).sum()) == 0
        sweep_mask = pd.to_numeric(out["sweep_flag"], errors="coerce").fillna(0) > 0
        # Every sweep bar has exactly one flag set.
        assert int(((bull[sweep_mask] + bear[sweep_mask]) != 1).sum()) == 0
        # No flag fires on a non-sweep bar.
        assert int(bull[~sweep_mask].sum()) == 0
        assert int(bear[~sweep_mask].sum()) == 0


def test_breach_and_distance_aliases_match_internal_columns():
    out = _run(_bearish_sweep_frame())
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    assert not sweep_rows.empty
    for confirm_idx, row in sweep_rows.iterrows():
        # ``penetration_atr`` is emitted at the breach bar, while the
        # canonical ``sweep_breach_atr`` is surfaced at the confirm bar
        # for downstream tooling. Resolve via ``sweep_breach_idx``.
        breach_idx = int(row["sweep_breach_idx"])
        breach_pen = float(out["penetration_atr"].iloc[breach_idx])
        assert float(row["sweep_breach_atr"]) == breach_pen
        # ``sweep_distance_at_start_atr`` and the internal pre-breach
        # distance share the confirm bar.
        assert float(row["sweep_distance_at_start_atr"]) == float(
            row["sweep_pre_breach_distance_atr"]
        )


def test_close_reclaim_atr_directionality():
    # Bearish sweep: TP below, swept_level above close → reclaim positive.
    out = _run(_bearish_sweep_frame())
    row = out[out["sweep_flag"].fillna(0) > 0].iloc[0]
    expected = (float(row["swept_level"]) - float(row["close"])) / 1.0
    assert abs(float(row["sweep_close_reclaim_atr"]) - expected) < 1e-9

    # Bullish sweep: swept_level below close → reclaim positive.
    out_b = _run(_bullish_sweep_frame())
    row_b = out_b[out_b["sweep_flag"].fillna(0) > 0].iloc[0]
    expected_b = (float(row_b["close"]) - float(row_b["swept_level"])) / 1.0
    assert abs(float(row_b["sweep_close_reclaim_atr"]) - expected_b) < 1e-9


def test_wick_and_body_ratios_use_confirm_bar_only():
    out = _run(_bearish_sweep_frame())
    row = out[out["sweep_flag"].fillna(0) > 0].iloc[0]
    high = float(row["high"])
    low = float(row["low"])
    o = float(row["open"])
    c = float(row["close"])
    rng = high - low
    upper_wick = high - max(o, c)
    body_reclaim = max(float(row["swept_level"]) - c, 0.0)
    assert abs(float(row["sweep_wick_rejection_ratio"]) - upper_wick / rng) < 1e-9
    assert abs(float(row["sweep_body_reclaim_ratio"]) - body_reclaim / rng) < 1e-9
    # Ratios are bounded.
    assert 0 <= float(row["sweep_wick_rejection_ratio"]) <= 1
    assert 0 <= float(row["sweep_body_reclaim_ratio"]) <= 1


def test_source_idx_is_causal_and_resolved_to_timestamp():
    out = _run(_bearish_sweep_frame())
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    for _, row in sweep_rows.iterrows():
        # Causality chain: source ≤ breach ≤ confirm.
        assert int(row["swept_source_idx"]) <= int(row["sweep_breach_idx"])
        assert int(row["sweep_breach_idx"]) <= int(row["sweep_confirm_idx"])
        # Resolved timestamp matches the source bar.
        ts_resolved = pd.Timestamp(row["swept_source_timestamp"])
        ts_expected = pd.Timestamp(out["timestamp"].iloc[int(row["swept_source_idx"])])
        assert ts_resolved == ts_expected


def test_rank_and_duplicate_group_id_default_to_singleton():
    out = _run(_bearish_sweep_frame())
    sweep_rows = out[out["sweep_flag"].fillna(0) > 0]
    for _, row in sweep_rows.iterrows():
        assert int(row["sweep_level_rank"]) == 1
        assert int(row["sweep_duplicate_group_id"]) == int(row["sweep_event_id"])


def test_canonical_threshold_registry_keys():
    expected = {
        "breach_tolerance_atr",
        "min_breach_atr",
        "min_close_reclaim_atr",
        "max_source_distance_atr",
        "min_source_age_bars",
        "micro_breach_atr_threshold",
        "strong_breach_atr_threshold",
        "strong_reclaim_atr_threshold",
        "min_wick_rejection_ratio",
        "min_quality_score_tradeable",
    }
    assert expected.issubset(set(SWEEPS_CANONICAL_THRESHOLDS.keys()))
    for entry in SWEEPS_CANONICAL_THRESHOLDS.values():
        assert "scope" in entry
        assert "description" in entry
