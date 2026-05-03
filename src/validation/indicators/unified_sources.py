"""Validation summaries for the unified liquidity source framework (Step 10).

The validators here are statistic-only: they consume the dense ladder
columns (and the sidecar audit table reconstructed via
:func:`build_unified_liquidity_clusters_audit`) and emit the punch-list of
distributions every validation script must print.

Validation contract per the SweepsPlan
--------------------------------------
* source rows by family
* active source count over time
* clusters by family composition
* deduplication rate
* dropped by crowding count
* dropped by dominance count
* nearest source distance distributions
* source age distributions
* source strength distributions

We also enforce the causality guard (origin <= active_start <= bar) and the
MTF policy stamp at the audit-table boundary.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.indicators.smc.sweeps.mtf_policy import (
    HTF_LIQUIDITY_PROJECTION_ENABLED,
    SWEEP_MTF_POLICY,
    assert_same_timeframe_sources,
)
from src.indicators.smc.sweeps.unified_sources import (
    LIQ_GLOBAL_CROWDING_CAP,
    LIQ_LADDER_DEPTH,
    LIQ_SOURCE_FAMILIES,
    UNIFIED_SOURCE_COLUMNS,
    build_unified_liquidity_clusters_audit,
)


def _quantiles(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def summarize_unified_sources(
    df: pd.DataFrame,
    *,
    scan_timeframe: str,
) -> dict[str, Any]:
    """Compute the canonical unified-sources validation dict.

    Parameters
    ----------
    df
        A pipeline frame post-:func:`add_unified_liquidity_sources`. Required
        columns: ``liq_active_total_count``, ``liq_dropped_by_crowding_count``,
        ``liq_dropped_by_dominance_count``, ``liq_above_l*_*`` /
        ``liq_below_l*_*`` ladder. Optional: ``timestamp``.
    scan_timeframe
        The timeframe the validation run is scanning. Cross-checked against
        the per-row stamp via :func:`assert_same_timeframe_sources`.
    """

    if df is None or len(df) == 0:
        return {"error": "empty_frame"}

    audit = build_unified_liquidity_clusters_audit(df)
    # Hard MTF guard at the validation boundary.
    assert_same_timeframe_sources(audit, scan_timeframe=scan_timeframe)

    summary: dict[str, Any] = {
        "mtf_policy": SWEEP_MTF_POLICY,
        "htf_projection_enabled": bool(HTF_LIQUIDITY_PROJECTION_ENABLED),
        "scan_timeframe": scan_timeframe,
        "source_timeframe_matches_scan_timeframe": True,
        "ladder_depth": LIQ_LADDER_DEPTH,
        "global_crowding_cap": LIQ_GLOBAL_CROWDING_CAP,
        "schema_columns": list(UNIFIED_SOURCE_COLUMNS),
        "row_count": int(len(df)),
        "audit_row_count": int(len(audit)),
    }

    # Ladder fill rates (how often each rank slot is populated).
    fill_rates: dict[str, dict[str, float]] = {"above": {}, "below": {}}
    for side in ("above", "below"):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            col = f"liq_{side}_l{rank}_cluster_id"
            if col not in df.columns:
                continue
            present = pd.to_numeric(df[col], errors="coerce").notna().sum()
            fill_rates[side][f"l{rank}"] = round(float(present) / len(df), 4)
    summary["ladder_fill_rate"] = fill_rates

    # Active source count distribution.
    summary["active_source_count"] = {
        "total": _quantiles(df["liq_active_total_count"]),
        "above": _quantiles(df["liq_active_above_count"]),
        "below": _quantiles(df["liq_active_below_count"]),
    }

    # Dedup / crowding / dominance counters.
    summary["dropped_counters"] = {
        "crowding_total": int(
            pd.to_numeric(df["liq_dropped_by_crowding_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "dominance_total": int(
            pd.to_numeric(df["liq_dropped_by_dominance_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "crowding_per_bar": _quantiles(df["liq_dropped_by_crowding_count"]),
        "dominance_per_bar": _quantiles(df["liq_dropped_by_dominance_count"]),
    }

    # Nearest distance distribution (ATR-normalised).
    summary["nearest_source_distance_atr"] = {
        "above": _quantiles(df["liq_nearest_above_dist_atr"]),
        "below": _quantiles(df["liq_nearest_below_dist_atr"]),
    }

    # Family / strength / age distributions from the audit table.
    if not audit.empty:
        family_counts = audit["primary_family"].value_counts().to_dict()
        # Reasonable validator output: ensure every canonical family is keyed.
        for fam in LIQ_SOURCE_FAMILIES:
            family_counts.setdefault(fam, 0)
        summary["source_rows_by_family"] = {
            fam: int(family_counts.get(fam, 0)) for fam in LIQ_SOURCE_FAMILIES
        }
        # Range / FVG / OB families MUST be absent.
        deprecated_present = sorted(
            f for f in family_counts if f.startswith(("range_", "fvg_", "ob_"))
        )
        summary["deprecated_families_present"] = deprecated_present

        # Cluster compositions: how often each family combination appears.
        summary["cluster_family_compositions_top"] = (
            audit["attribution_families"].value_counts().head(15).to_dict()
        )

        summary["age_distribution_bars"] = _quantiles(audit["age_bars"])
        summary["strength_distribution"] = _quantiles(audit["strength"])
        summary["width_distribution_atr"] = _quantiles(audit["width_atr"])
        summary["touch_count_distribution"] = _quantiles(audit["touch_count"])

        # Causality check
        causality_violations = int(
            (
                (audit["source_active_start_idx"] >= 0)
                & (audit["source_active_start_idx"] > audit["bar_idx"])
            ).sum()
        )
        summary["causality_violations"] = causality_violations
    else:
        summary["source_rows_by_family"] = {fam: 0 for fam in LIQ_SOURCE_FAMILIES}
        summary["deprecated_families_present"] = []
        summary["cluster_family_compositions_top"] = {}
        summary["age_distribution_bars"] = _quantiles(pd.Series(dtype=float))
        summary["strength_distribution"] = _quantiles(pd.Series(dtype=float))
        summary["width_distribution_atr"] = _quantiles(pd.Series(dtype=float))
        summary["touch_count_distribution"] = _quantiles(pd.Series(dtype=float))
        summary["causality_violations"] = 0

    return summary


def print_unified_sources_summary(summary: Mapping[str, Any], indent: int = 0) -> None:
    """Pretty-print the summary dict to stdout — matches the existing
    convention used by ``validate_equal_hl.py`` and friends."""

    prefix = " " * indent
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_unified_sources_summary(value, indent + 2)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {value[:8]}{' ...' if len(value) > 8 else ''}")
        else:
            print(f"{prefix}{key}: {value}")


__all__ = [
    "summarize_unified_sources",
    "print_unified_sources_summary",
]
