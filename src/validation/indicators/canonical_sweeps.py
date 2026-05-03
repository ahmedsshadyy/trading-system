"""Canonical sweeps validation report (Step-frozen contract).

Produces the 18-point report defined in
``docs/indicator_contracts/sweeps.md``. Reads the canonical alias
columns surfaced by :mod:`src.indicators.smc.sweeps.final_sweeps`; never
reaches into legacy ``sweep_high`` / ``sweep_low`` schema (the legacy
detector was retired).

This module is import-light: it only consumes the frame produced by the
canonical pipeline and returns a plain ``dict`` summary plus a few small
``pandas.DataFrame`` tables. Pretty-printing lives in the script
``scripts/validate_sweeps.py``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
    SWEEPS_CANONICAL_THRESHOLDS,
)

#: Columns required from the frame for the canonical report.
REQUIRED_CANONICAL_EVENT_COLUMNS: tuple[str, ...] = (
    "sweep_flag",
    "sweep_direction",
    "bullish_sweep_flag",
    "bearish_sweep_flag",
    "swept_level",
    "swept_source_family",
    "swept_source_side",
    "swept_source_strength",
    "swept_source_idx",
    "swept_source_age_bars",
    "swept_source_timestamp",
    "sweep_breach_atr",
    "sweep_close_reclaim_atr",
    "sweep_distance_at_start_atr",
    "sweep_wick_rejection_ratio",
    "sweep_body_reclaim_ratio",
    "sweep_quality_score",
    "sweep_selectivity_class",
    "sweep_primary_family",
    "sweep_breach_idx",
    "sweep_confirm_idx",
    "sweep_event_id",
    "sweep_level_rank",
    "sweep_duplicate_group_id",
)

#: Columns required at the breach bar (to verify causality).
REQUIRED_FRAME_COLUMNS: tuple[str, ...] = (
    "high",
    "low",
    "close",
    "atr_14",
)

ACCEPTED_SOURCE_FAMILIES: tuple[str, ...] = (
    "previous_day_high",
    "previous_day_low",
    "previous_week_high",
    "previous_week_low",
    "session_high",
    "session_low",
    "swing_high",
    "swing_low",
    "equal_high",
    "equal_low",
    "resistance",
    "support",
    "range_high",
    "range_low",
)


def _value_counts_dict(series: pd.Series) -> dict[str, int]:
    """Return ``value: count`` ordered by descending count."""

    if series.empty:
        return {}
    counts = series.fillna("").astype(str).value_counts(dropna=False)
    return {str(idx): int(val) for idx, val in counts.items()}


def _continuous_distribution(series: pd.Series) -> dict[str, float | int]:
    """Return summary stats for a continuous distribution."""

    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    quantiles = cleaned.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(cleaned.size),
        "mean": float(cleaned.mean()),
        "median": float(quantiles.loc[0.50]),
        "std": float(cleaned.std(ddof=0)),
        "min": float(cleaned.min()),
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
        "max": float(cleaned.max()),
    }


def _causality_violations(df: pd.DataFrame, sweep_rows: pd.DataFrame) -> dict[str, int]:
    """Count violations of the canonical causality contract."""

    violations: dict[str, int] = {
        "source_after_breach": 0,
        "breach_after_confirm": 0,
        "swept_source_idx_negative": 0,
        "swept_source_idx_out_of_range": 0,
        "ambiguous_direction_label": 0,
        "swept_source_family_unknown": 0,
        "missing_source_metadata": 0,
        "bullish_with_above_swept_side": 0,
        "bearish_with_below_swept_side": 0,
        "duplicate_canonical_event": 0,
    }
    if sweep_rows.empty:
        return violations

    n_total = len(df)
    swept_idx = pd.to_numeric(sweep_rows["swept_source_idx"], errors="coerce")
    breach_idx = pd.to_numeric(sweep_rows["sweep_breach_idx"], errors="coerce")
    confirm_idx = pd.to_numeric(sweep_rows["sweep_confirm_idx"], errors="coerce")

    finite_swept = swept_idx.dropna()
    violations["source_after_breach"] = int(
        ((finite_swept > breach_idx.loc[finite_swept.index])).sum()
    )
    violations["breach_after_confirm"] = int(
        (breach_idx.dropna() > confirm_idx.dropna()).sum()
    )
    violations["swept_source_idx_negative"] = int((finite_swept < 0).sum())
    violations["swept_source_idx_out_of_range"] = int((finite_swept >= n_total).sum())

    direction = sweep_rows["sweep_direction"].astype(str).fillna("")
    valid_directions = {"bullish", "bearish"}
    violations["ambiguous_direction_label"] = int(
        (~direction.isin(valid_directions)).sum()
    )

    family = sweep_rows["swept_source_family"].astype(str).fillna("")
    violations["swept_source_family_unknown"] = int(
        ((~family.isin(ACCEPTED_SOURCE_FAMILIES)) & (family != "")).sum()
    )

    # Required metadata not NaN on every confirmed sweep. ``swept_source_idx``
    # is excluded here because the canonical alias deliberately NaN-s it
    # when the upstream ladder breaks causality; that case is reported via
    # ``upstream_origin_idx_invalid_count`` instead.
    required_meta = (
        "swept_level",
        "swept_source_family",
        "swept_source_side",
        "sweep_breach_idx",
        "sweep_confirm_idx",
    )
    missing_mask = pd.Series(False, index=sweep_rows.index)
    for col in required_meta:
        if col == "swept_source_family":
            missing_mask |= sweep_rows[col].astype(str).fillna("") == ""
        else:
            missing_mask |= pd.to_numeric(sweep_rows[col], errors="coerce").isna()
    violations["missing_source_metadata"] = int(missing_mask.sum())

    side_numeric = pd.to_numeric(sweep_rows["swept_source_side"], errors="coerce")
    bull_flag = pd.to_numeric(sweep_rows["bullish_sweep_flag"], errors="coerce").fillna(
        0
    )
    bear_flag = pd.to_numeric(sweep_rows["bearish_sweep_flag"], errors="coerce").fillna(
        0
    )
    # Bullish sweeps must reference a below-price source (side == -1).
    violations["bullish_with_above_swept_side"] = int(
        ((bull_flag > 0) & (side_numeric != -1)).sum()
    )
    violations["bearish_with_below_swept_side"] = int(
        ((bear_flag > 0) & (side_numeric != +1)).sum()
    )

    # Same-bar / same-direction duplicates.
    if "sweep_duplicate_group_id" in sweep_rows.columns:
        same_bar = sweep_rows.groupby(
            [confirm_idx, sweep_rows["sweep_direction"].astype(str)],
            dropna=False,
            observed=False,
        ).size()
        violations["duplicate_canonical_event"] = int(int((same_bar > 1).sum()))
    return violations


def build_canonical_sweeps_report(
    df: pd.DataFrame,
    *,
    expected_alias_columns: Iterable[str] = FINAL_SWEEPS_CANONICAL_ALIAS_COLUMNS,
) -> dict[str, object]:
    """Return the 18-point canonical sweep validation report."""

    if df is None or len(df) == 0:
        return {
            "total_rows": 0,
            "total_sweep_count": 0,
            "bullish_sweep_count": 0,
            "bearish_sweep_count": 0,
            "by_source_family": {},
            "by_selectivity_class": {},
            "by_session_phase": {},
            "by_regime_label": {},
            "by_volume_confirmed": {},
            "by_displacement_confirmed": {},
            "breach_atr_distribution": _continuous_distribution(pd.Series(dtype=float)),
            "close_reclaim_atr_distribution": _continuous_distribution(
                pd.Series(dtype=float)
            ),
            "distance_at_start_atr_distribution": _continuous_distribution(
                pd.Series(dtype=float)
            ),
            "valid_source_metadata_pct": float("nan"),
            "source_family_share_pct": {},
            "schema_invariants": {
                "alias_columns_present": False,
                "missing_alias_columns": list(expected_alias_columns),
                "no_legacy_columns": True,
            },
            "causality_violations": {
                "source_after_breach": 0,
                "breach_after_confirm": 0,
                "swept_source_idx_negative": 0,
                "swept_source_idx_out_of_range": 0,
                "ambiguous_direction_label": 0,
                "swept_source_family_unknown": 0,
                "missing_source_metadata": 0,
                "bullish_with_above_swept_side": 0,
                "bearish_with_below_swept_side": 0,
                "duplicate_canonical_event": 0,
            },
            "upstream_origin_idx_invalid_count": 0,
            "future_columns_required": [],
            "thresholds": dict(SWEEPS_CANONICAL_THRESHOLDS),
        }

    missing_frame = [c for c in REQUIRED_FRAME_COLUMNS if c not in df.columns]
    missing_event = [c for c in REQUIRED_CANONICAL_EVENT_COLUMNS if c not in df.columns]
    if missing_frame or missing_event:
        raise ValueError(
            "Canonical sweep report requires columns "
            f"{sorted(set(missing_frame + missing_event))}; pipeline "
            "did not produce them. Did you call ``add_final_sweeps`` "
            "after ``add_unified_liquidity_sources``?"
        )

    sweep_mask = pd.to_numeric(df["sweep_flag"], errors="coerce").fillna(0) > 0
    sweep_rows = df.loc[sweep_mask].copy()

    bullish_count = int(
        pd.to_numeric(sweep_rows["bullish_sweep_flag"], errors="coerce").fillna(0).sum()
    )
    bearish_count = int(
        pd.to_numeric(sweep_rows["bearish_sweep_flag"], errors="coerce").fillna(0).sum()
    )

    by_source_family = _value_counts_dict(sweep_rows["swept_source_family"])
    by_selectivity_class = _value_counts_dict(sweep_rows["sweep_selectivity_class"])
    by_session_phase = (
        _value_counts_dict(sweep_rows["session_name"])
        if "session_name" in sweep_rows.columns
        else {}
    )
    by_regime_label = (
        _value_counts_dict(sweep_rows["regime_label"])
        if "regime_label" in sweep_rows.columns
        else {}
    )
    by_volume_confirmed = (
        _value_counts_dict(
            (
                pd.to_numeric(
                    sweep_rows["sweep_q_volume_confirmation"], errors="coerce"
                )
                > 0.5
            ).astype(int)
        )
        if "sweep_q_volume_confirmation" in sweep_rows.columns
        else {}
    )
    by_displacement_confirmed = (
        _value_counts_dict(
            pd.to_numeric(
                sweep_rows["sweep_is_displacement_confirmed"], errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )
        if "sweep_is_displacement_confirmed" in sweep_rows.columns
        else {}
    )

    breach_atr_distribution = _continuous_distribution(sweep_rows["sweep_breach_atr"])
    close_reclaim_atr_distribution = _continuous_distribution(
        sweep_rows["sweep_close_reclaim_atr"]
    )
    distance_at_start_atr_distribution = _continuous_distribution(
        sweep_rows["sweep_distance_at_start_atr"]
    )

    # Coverage: % of confirmed sweeps with non-null required metadata.
    required_meta = (
        "swept_level",
        "swept_source_family",
        "swept_source_idx",
        "sweep_breach_idx",
    )
    valid_meta = pd.Series(True, index=sweep_rows.index)
    for col in required_meta:
        if col == "swept_source_family":
            valid_meta &= sweep_rows[col].astype(str).fillna("") != ""
        else:
            valid_meta &= pd.to_numeric(sweep_rows[col], errors="coerce").notna()
    valid_pct = (
        float(valid_meta.mean() * 100.0) if not sweep_rows.empty else float("nan")
    )

    total_sweeps = int(sweep_rows.shape[0])
    family_share = (
        {
            family: float(count / total_sweeps * 100.0)
            for family, count in by_source_family.items()
        }
        if total_sweeps
        else {}
    )

    expected = list(expected_alias_columns)
    missing_aliases = [c for c in expected if c not in df.columns]
    legacy_legacy_columns = [c for c in ("sweep_high", "sweep_low") if c in df.columns]

    causality = _causality_violations(df, sweep_rows)
    upstream_invalid_count = (
        int(
            pd.to_numeric(
                sweep_rows["sweep_origin_idx_upstream_invalid"], errors="coerce"
            )
            .fillna(0)
            .sum()
        )
        if "sweep_origin_idx_upstream_invalid" in sweep_rows.columns
        else 0
    )

    return {
        "total_rows": int(len(df)),
        "total_sweep_count": total_sweeps,
        "bullish_sweep_count": bullish_count,
        "bearish_sweep_count": bearish_count,
        "by_source_family": by_source_family,
        "by_selectivity_class": by_selectivity_class,
        "by_session_phase": by_session_phase,
        "by_regime_label": by_regime_label,
        "by_volume_confirmed": by_volume_confirmed,
        "by_displacement_confirmed": by_displacement_confirmed,
        "breach_atr_distribution": breach_atr_distribution,
        "close_reclaim_atr_distribution": close_reclaim_atr_distribution,
        "distance_at_start_atr_distribution": distance_at_start_atr_distribution,
        "valid_source_metadata_pct": valid_pct,
        "source_family_share_pct": family_share,
        "schema_invariants": {
            "alias_columns_present": not missing_aliases,
            "missing_alias_columns": missing_aliases,
            "no_legacy_columns": not legacy_legacy_columns,
            "legacy_columns_present": legacy_legacy_columns,
        },
        "causality_violations": causality,
        "upstream_origin_idx_invalid_count": upstream_invalid_count,
        # No forward / future-bar columns are required for this report.
        "future_columns_required": [],
        "thresholds": dict(SWEEPS_CANONICAL_THRESHOLDS),
    }


def report_passed(report: dict[str, object]) -> bool:
    """Return True iff every canonical acceptance gate passed."""

    causality = report.get("causality_violations", {}) or {}
    if any(int(v) for v in causality.values()):
        return False
    schema = report.get("schema_invariants", {}) or {}
    if not schema.get("alias_columns_present", False):
        return False
    if not schema.get("no_legacy_columns", False):
        return False
    return True


__all__ = [
    "ACCEPTED_SOURCE_FAMILIES",
    "REQUIRED_CANONICAL_EVENT_COLUMNS",
    "REQUIRED_FRAME_COLUMNS",
    "build_canonical_sweeps_report",
    "report_passed",
]
