"""
validation/indicators/sr_levels.py

Validation summary and chart helpers for the S/R zone engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators.foundation.sr_levels import (
    SR_FAMILY_DAY,
    SR_FAMILY_EQHL,
    SR_FAMILY_SESSION,
    SR_FAMILY_SWING,
    SR_FAMILY_VP,
    SR_FAMILY_WEEK,
    SR_STATE_ACTIVE,
    SR_STATE_ACTIVE_WEAKENED,
    SR_STATE_BREAK_PENDING,
    SR_STATE_INVALIDATED,
    SR_STATE_RETIRED,
    SR_SIDE_RESISTANCE,
    SR_SIDE_SUPPORT,
    SRLevel,
    build_sr_touch_audit_table,
)
from src.validation.common.chart_core import create_candlestick_figure, save_figure_html

ALL_FAMILIES = (
    SR_FAMILY_SWING,
    SR_FAMILY_EQHL,
    SR_FAMILY_SESSION,
    SR_FAMILY_DAY,
    SR_FAMILY_WEEK,
    SR_FAMILY_VP,
)

LIVE_SAFE_COLUMNS: frozenset[str] = frozenset(
    {
        "nearest_support_price",
        "nearest_support_distance",
        "nearest_support_distance_atr",
        "nearest_support_age_bars",
        "nearest_support_strength",
        "nearest_support_source_family",
        "nearest_support_source_idx",
        "nearest_support_zone_id",
        "nearest_support_touch_count",
        "nearest_support_refresh_count",
        "nearest_support_weaken_count",
        "nearest_support_active",
        "nearest_resistance_price",
        "nearest_resistance_distance",
        "nearest_resistance_distance_atr",
        "nearest_resistance_age_bars",
        "nearest_resistance_strength",
        "nearest_resistance_source_family",
        "nearest_resistance_source_idx",
        "nearest_resistance_zone_id",
        "nearest_resistance_touch_count",
        "nearest_resistance_refresh_count",
        "nearest_resistance_weaken_count",
        "nearest_resistance_active",
        "inside_sr_band_flag",
        "between_nearest_sr_flag",
        "above_nearest_resistance_flag",
        "below_nearest_support_flag",
        "support_broken_this_bar",
        "resistance_broken_this_bar",
        "active_support_count",
        "active_resistance_count",
        "support_cluster_density_atr",
        "resistance_cluster_density_atr",
        "primary_support_zone_low",
        "primary_support_zone_high",
        "primary_support_zone_mid",
        "primary_support_zone_score",
        "primary_support_zone_anchor_count",
        "primary_support_zone_family_count",
        "primary_support_zone_width_atr",
        "primary_support_zone_id",
        "primary_resistance_zone_low",
        "primary_resistance_zone_high",
        "primary_resistance_zone_mid",
        "primary_resistance_zone_score",
        "primary_resistance_zone_anchor_count",
        "primary_resistance_zone_family_count",
        "primary_resistance_zone_width_atr",
        "primary_resistance_zone_id",
        "inside_primary_support_zone_flag",
        "inside_primary_resistance_zone_flag",
        "sr_break_pending_flag",
        "sr_reclaim_this_bar_flag",
    }
)
for _side in ("support", "resistance"):
    for _slot in range(1, 4):
        _prefix = f"sr_{_side}_l{_slot}"
        LIVE_SAFE_COLUMNS = LIVE_SAFE_COLUMNS.union(
            {
                f"{_prefix}_id",
                f"{_prefix}_low",
                f"{_prefix}_high",
                f"{_prefix}_mid",
                f"{_prefix}_score",
                f"{_prefix}_family",
                f"{_prefix}_state",
                f"{_prefix}_age_bars",
                f"{_prefix}_expiry_bars_remaining",
            }
        )


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
    }


def _is_live_zone(level: SRLevel) -> bool:
    return level.emitted_zone_flag and level.state in {
        SR_STATE_ACTIVE,
        SR_STATE_ACTIVE_WEAKENED,
        SR_STATE_BREAK_PENDING,
    }


def _first_active_row(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    mask = pd.to_numeric(df[column], errors="coerce").fillna(0).eq(1)
    if not bool(mask.any()):
        return None
    return int(np.flatnonzero(mask.to_numpy())[0])


def _no_inf_in_live_columns(df: pd.DataFrame) -> bool:
    live_cols = [col for col in LIVE_SAFE_COLUMNS if col in df.columns]
    if not live_cols:
        return True
    numeric = df[live_cols].select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return bool(np.isfinite(numeric[~np.isnan(numeric)]).all())


def _touch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "r_sr_touch_event_id" not in df.columns:
        return pd.DataFrame()
    mask = pd.to_numeric(df["r_sr_touch_event_id"], errors="coerce").notna()
    cols = [col for col in df.columns if col.startswith("r_sr_touch_")]
    if not bool(mask.any()):
        return pd.DataFrame(columns=cols)
    return df.loc[mask, cols].copy()


_OUTCOME_METRIC_COLUMNS: tuple[str, ...] = ("drift", "held", "signed")
_OUTCOME_METRIC_TO_TEMPLATE: dict[str, str] = {
    "drift": "outcome_{h}",
    "held": "outcome_held_{h}",
    "signed": "outcome_signed_{h}",
}


def _empty_quintile_skeleton(touch_count: int) -> dict[str, object]:
    empty_metric = {
        metric: {str(h): None for h in (4, 8, 12)} for metric in _OUTCOME_METRIC_COLUMNS
    }
    return {
        "touch_count": int(touch_count),
        "quintiles": {},
        "monotonicity": {str(h): None for h in (4, 8, 12)},
        "top_vs_bottom_delta": {str(h): None for h in (4, 8, 12)},
        "monotonicity_by_metric": {k: dict(v) for k, v in empty_metric.items()},
        "top_vs_bottom_delta_by_metric": {k: dict(v) for k, v in empty_metric.items()},
    }


def _quintile_summary(
    touches: pd.DataFrame, score_col: str = "score"
) -> dict[str, object]:
    if touches.empty or score_col not in touches.columns:
        return _empty_quintile_skeleton(len(touches))
    scores = pd.to_numeric(touches[score_col], errors="coerce")
    ranked = touches.loc[scores.notna()].copy()
    ranked[score_col] = scores.loc[scores.notna()]
    if ranked.empty or ranked[score_col].nunique() < 2:
        return _empty_quintile_skeleton(len(ranked))
    ranked["quintile"] = pd.qcut(
        ranked[score_col],
        q=min(5, ranked[score_col].nunique()),
        labels=False,
        duplicates="drop",
    )
    quintiles: dict[str, dict[str, object]] = {}
    monotonicity: dict[str, bool | None] = {}
    deltas: dict[str, float | None] = {}
    monotonicity_by_metric: dict[str, dict[str, bool | None]] = {
        metric: {} for metric in _OUTCOME_METRIC_COLUMNS
    }
    deltas_by_metric: dict[str, dict[str, float | None]] = {
        metric: {} for metric in _OUTCOME_METRIC_COLUMNS
    }
    for quintile, group in ranked.groupby("quintile", dropna=True):
        label = str(int(quintile) + 1)
        quintiles[label] = {
            "count": int(len(group)),
            "score_mean": float(
                pd.to_numeric(group[score_col], errors="coerce").mean()
            ),
        }
        for horizon in (4, 8, 12):
            quintiles[label][f"outcome_rate_{horizon}"] = float(
                pd.to_numeric(group[f"outcome_{horizon}"], errors="coerce").mean()
            )
            quintiles[label][f"mfe_atr_mean_{horizon}"] = float(
                pd.to_numeric(group[f"mfe_atr_{horizon}"], errors="coerce").mean()
            )
            quintiles[label][f"mae_atr_mean_{horizon}"] = float(
                pd.to_numeric(group[f"mae_atr_{horizon}"], errors="coerce").mean()
            )
            for metric in _OUTCOME_METRIC_COLUMNS:
                col = _OUTCOME_METRIC_TO_TEMPLATE[metric].format(h=horizon)
                if col not in group.columns:
                    quintiles[label][f"outcome_rate_{metric}_{horizon}"] = None
                    continue
                quintiles[label][f"outcome_rate_{metric}_{horizon}"] = float(
                    pd.to_numeric(group[col], errors="coerce").mean()
                )
    for horizon in (4, 8, 12):
        for metric in _OUTCOME_METRIC_COLUMNS:
            metric_rates = [
                quintiles[key].get(f"outcome_rate_{metric}_{horizon}")
                for key in sorted(quintiles, key=int)
            ]
            metric_clean = [
                float(v) for v in metric_rates if v is not None and np.isfinite(v)
            ]
            monotonicity_by_metric[metric][str(horizon)] = (
                bool(all(a <= b for a, b in zip(metric_clean, metric_clean[1:])))
                if len(metric_clean) >= 2
                else None
            )
            deltas_by_metric[metric][str(horizon)] = (
                float(metric_clean[-1] - metric_clean[0]) if metric_clean else None
            )
        rates = [
            quintiles[key].get(f"outcome_rate_{horizon}")
            for key in sorted(quintiles, key=int)
        ]
        clean_rates = [float(v) for v in rates if v is not None and np.isfinite(v)]
        monotonicity[str(horizon)] = (
            bool(all(a <= b for a, b in zip(clean_rates, clean_rates[1:])))
            if len(clean_rates) >= 2
            else None
        )
        if "1" in quintiles and str(len(quintiles)) in quintiles:
            top = quintiles[str(len(quintiles))].get(f"outcome_rate_{horizon}")
            bottom = quintiles["1"].get(f"outcome_rate_{horizon}")
            deltas[str(horizon)] = (
                float(top - bottom) if top is not None and bottom is not None else None
            )
        else:
            deltas[str(horizon)] = None
    return {
        "touch_count": int(len(ranked)),
        "quintiles": quintiles,
        "monotonicity": monotonicity,
        "top_vs_bottom_delta": deltas,
        "monotonicity_by_metric": monotonicity_by_metric,
        "top_vs_bottom_delta_by_metric": deltas_by_metric,
    }


def _segment_calibration(touches: pd.DataFrame, column: str) -> dict[str, object]:
    if touches.empty or column not in touches.columns:
        return {}
    summary: dict[str, object] = {}
    grouped = touches.dropna(subset=[column]).groupby(column, dropna=True)
    for value, group in grouped:
        summary[str(value)] = _quintile_summary(group)
    return summary


def _rank_corr(series: pd.Series, target: pd.Series) -> float | None:
    valid = (
        pd.to_numeric(series, errors="coerce").notna()
        & pd.to_numeric(target, errors="coerce").notna()
    )
    if int(valid.sum()) < 3:
        return None
    left = pd.to_numeric(series[valid], errors="coerce").rank(pct=True)
    right = pd.to_numeric(target[valid], errors="coerce").rank(pct=True)
    corr = left.corr(right, method="pearson")
    return float(corr) if corr is not None and np.isfinite(corr) else None


def _component_audit(touches: pd.DataFrame) -> dict[str, object]:
    components = (
        "source_quality_score",
        "confluence_score",
        "reaction_quality_score",
        "freshness_score",
        "family_prior_score",
        "width_quality_score",
        "score_penalty_value",
        "score",
    )
    out: dict[str, object] = {}
    for column in components:
        if column not in touches.columns:
            continue
        out[column] = {
            "rank_corr": {
                str(h): _rank_corr(touches[column], touches[f"outcome_{h}"])
                for h in (4, 8, 12)
            },
            "top_vs_bottom_delta": _quintile_summary(touches, score_col=column).get(
                "top_vs_bottom_delta", {}
            ),
        }
    return out


_HIGH_INFO_FAMILIES: tuple[str, ...] = ("eqhl", "swing")
_HIGH_INFO_DELTA_MIN: float = 0.10
_HIGH_INFO_MIN_TOUCHES: int = 30


def _extract_metric_delta(
    block: dict[str, object], metric: str, horizon: str
) -> float | None:
    deltas = (block.get("top_vs_bottom_delta_by_metric") or {}).get(metric, {})
    if not isinstance(deltas, dict):
        return None
    value = deltas.get(horizon)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _high_info_predictive(calibration: dict[str, object]) -> dict[str, object]:
    """Per-family signed-OR-held top-vs-bottom delta on the high-info families.

    The pooled headline is dominated by session/day touches that are noise by
    construction. The gold-standard gate evaluates predictive power on the
    high-info families (eqhl, swing) using two complementary outcome metrics:

    - ``signed`` = price moved by k*ATR in the level's implied direction.
    - ``held``   = close never crossed the level's far edge over the horizon.

    A family passes the gate if EITHER metric's top-vs-bottom delta clears
    the threshold AND the touch count is large enough to be statistically
    meaningful. Either gate passing => predictive_high_info=True.
    """
    by_family = calibration.get("by_source_family", {}) or {}
    horizon = "8"
    family_results: dict[str, dict[str, object]] = {}
    for family in _HIGH_INFO_FAMILIES:
        block = by_family.get(family) if isinstance(by_family, dict) else None
        if not isinstance(block, dict):
            family_results[family] = {
                "touch_count": 0,
                "signed_delta_8": None,
                "held_delta_8": None,
                "passes": False,
            }
            continue
        signed_delta = _extract_metric_delta(block, "signed", horizon)
        held_delta = _extract_metric_delta(block, "held", horizon)
        touch_count = int(block.get("touch_count", 0))
        any_metric_passes = (
            signed_delta is not None and signed_delta >= _HIGH_INFO_DELTA_MIN
        ) or (held_delta is not None and held_delta >= _HIGH_INFO_DELTA_MIN)
        passes = bool(any_metric_passes and touch_count >= _HIGH_INFO_MIN_TOUCHES)
        family_results[family] = {
            "touch_count": touch_count,
            "signed_delta_8": signed_delta,
            "held_delta_8": held_delta,
            "passes": passes,
        }
    any_high_info_pass = any(item["passes"] for item in family_results.values())
    return {
        "predictive": any_high_info_pass,
        "by_family": family_results,
        "min_delta_8": _HIGH_INFO_DELTA_MIN,
        "min_touch_count": _HIGH_INFO_MIN_TOUCHES,
    }


def _headline_verdict(
    checks: dict[str, object],
    *,
    nearest_support_availability: float,
    nearest_resistance_availability: float,
    calibration: dict[str, object],
) -> dict[str, object]:
    parity_ok = checks.get("live_research_parity_ok")
    structure_status = {
        "causal": bool(
            parity_ok is True
            and checks.get("no_pre_live_support_projection")
            and checks.get("no_pre_live_resistance_projection")
            and checks.get("warmup_has_no_primary_zone_before_first_live")
        ),
        "clean": bool(
            checks.get("contamination_clean") and checks.get("no_inf_in_live_columns")
        ),
        "available": bool(
            nearest_support_availability >= 0.95
            and nearest_resistance_availability >= 0.95
        ),
    }
    structure_status["label"] = "pass" if all(structure_status.values()) else "fail"

    deltas = calibration.get("top_vs_bottom_delta", {})
    monotonicity = calibration.get("monotonicity", {})
    valid_deltas = [
        float(value)
        for value in deltas.values()
        if value is not None and np.isfinite(value)
    ]
    mean_delta = float(np.mean(valid_deltas)) if valid_deltas else None
    monotonic_true_count = int(
        sum(1 for value in monotonicity.values() if value is True)
    )
    predictive_pooled = bool(
        mean_delta is not None and mean_delta >= 0.02 and monotonic_true_count >= 2
    )
    inverted = bool(
        mean_delta is not None and mean_delta <= -0.02 and monotonic_true_count == 0
    )
    high_info = _high_info_predictive(calibration)
    predictive_high_info = bool(high_info.get("predictive", False))
    # Headline label: gold-standard gate is the high-info-family signed-delta
    # criterion. The pooled drift gate is retained as a diagnostic only — it
    # gets dominated by low-info session/day touches that are random by
    # construction.
    score_status = {
        "predictive": predictive_high_info,
        "predictive_high_info": predictive_high_info,
        "predictive_pooled": predictive_pooled,
        "flat": bool(not predictive_high_info and not inverted),
        "inverted": inverted,
        "mean_top_vs_bottom_delta": mean_delta,
        "monotonic_true_count": monotonic_true_count,
        "high_info": high_info,
    }
    score_status["label"] = "pass" if predictive_high_info else "fail"
    return {
        "structure_status": structure_status,
        "score_status": score_status,
    }


def _primary_selection_summary(
    df: pd.DataFrame, touches: pd.DataFrame
) -> dict[str, object]:
    rows: list[pd.DataFrame] = []
    for side in ("support", "resistance"):
        primary_col = f"primary_{side}_zone_id"
        nearest_col = f"nearest_{side}_zone_id"
        primary_score_col = f"primary_{side}_zone_score"
        nearest_score_col = (
            "nearest_support_strength"
            if side == "support"
            else "nearest_resistance_strength"
        )
        if primary_col not in df.columns or nearest_col not in df.columns:
            continue
        side_rows = pd.DataFrame(
            {
                "side": side,
                "primary_zone_id": pd.to_numeric(df[primary_col], errors="coerce"),
                "nearest_zone_id": pd.to_numeric(df[nearest_col], errors="coerce"),
                "primary_score": pd.to_numeric(df[primary_score_col], errors="coerce"),
                "nearest_score": pd.to_numeric(df[nearest_score_col], errors="coerce"),
                "row": np.arange(len(df)),
            }
        )
        rows.append(side_rows)
    if not rows:
        return {}
    joined = pd.concat(rows, ignore_index=True)
    valid = joined["primary_zone_id"].notna() & joined["nearest_zone_id"].notna()
    diff_rate = (
        float(
            (
                joined.loc[valid, "primary_zone_id"]
                != joined.loc[valid, "nearest_zone_id"]
            ).mean()
        )
        if bool(valid.any())
        else None
    )
    if touches.empty or "row" not in touches.columns:
        touch_joined = joined.iloc[0:0].copy()
        for column in ("outcome_4", "outcome_8", "outcome_12"):
            touch_joined[column] = np.nan
    else:
        touch_joined = touches.merge(
            joined, on=["row"], how="left", suffixes=("", "_row")
        )
    for horizon in (4, 8, 12):
        if f"outcome_{horizon}" not in touch_joined.columns:
            touch_joined[f"outcome_{horizon}"] = np.nan
    return {
        "primary_diff_rate": diff_rate,
        "score_quality": {
            "primary_score": {
                "rank_corr": {
                    str(h): _rank_corr(
                        touch_joined["primary_score"], touch_joined[f"outcome_{h}"]
                    )
                    for h in (4, 8, 12)
                },
                "top_vs_bottom_delta": _quintile_summary(
                    touch_joined.drop(columns=["score"], errors="ignore")
                    .dropna(subset=["primary_score"])
                    .rename(columns={"primary_score": "score"})
                ).get("top_vs_bottom_delta", {}),
            },
            "nearest_score": {
                "rank_corr": {
                    str(h): _rank_corr(
                        touch_joined["nearest_score"], touch_joined[f"outcome_{h}"]
                    )
                    for h in (4, 8, 12)
                },
                "top_vs_bottom_delta": _quintile_summary(
                    touch_joined.drop(columns=["score"], errors="ignore")
                    .dropna(subset=["nearest_score"])
                    .rename(columns={"nearest_score": "score"})
                ).get("top_vs_bottom_delta", {}),
            },
        },
        "touch_match_rate": {
            "primary": (
                float(
                    pd.to_numeric(
                        touches["primary_matches_touch"], errors="coerce"
                    ).mean()
                )
                if "primary_matches_touch" in touches.columns and not touches.empty
                else None
            ),
            "nearest": (
                float(
                    pd.to_numeric(
                        touches["nearest_matches_touch"], errors="coerce"
                    ).mean()
                )
                if "nearest_matches_touch" in touches.columns and not touches.empty
                else None
            ),
        },
    }


def summarize_sr_levels(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
    *,
    live_df: pd.DataFrame | None = None,
    touch_rows: pd.DataFrame | None = None,
) -> dict:
    n = len(df)
    all_levels = list(registry.values())
    emitted = [lev for lev in all_levels if lev.emitted_zone_flag]
    absorbed = [lev for lev in all_levels if lev.absorbed_by is not None]
    active = [lev for lev in emitted if _is_live_zone(lev)]
    invalidated = [lev for lev in emitted if lev.state == SR_STATE_INVALIDATED]
    retired = [lev for lev in emitted if lev.state == SR_STATE_RETIRED]

    raw_by_family = {family: 0 for family in ALL_FAMILIES}
    absorbed_by_family = {family: 0 for family in ALL_FAMILIES}
    emitted_by_family = {family: 0 for family in ALL_FAMILIES}
    emitted_by_best_family = {family: 0 for family in ALL_FAMILIES}
    absorption_matrix = {
        family: {target_family: 0 for target_family in ALL_FAMILIES}
        for family in ALL_FAMILIES
    }
    for lev in all_levels:
        raw_by_family[lev.source_family] = raw_by_family.get(lev.source_family, 0) + 1
        if lev.absorbed_by is not None:
            absorbed_by_family[lev.source_family] = (
                absorbed_by_family.get(lev.source_family, 0) + 1
            )
            target_family = lev.absorbed_into_best_family or lev.absorbed_into_family
            if target_family in absorption_matrix:
                absorption_matrix[lev.source_family][target_family] += 1
    for lev in emitted:
        emitted_by_family[lev.source_family] = (
            emitted_by_family.get(lev.source_family, 0) + 1
        )
        best_family = lev.best_source_family or lev.source_family
        emitted_by_best_family[best_family] = (
            emitted_by_best_family.get(best_family, 0) + 1
        )

    ns_avail = float(df["nearest_support_active"].sum()) / n if n else 0.0
    nr_avail = float(df["nearest_resistance_active"].sum()) / n if n else 0.0

    first_support_live = min(
        (lev.source_live_from_idx for lev in emitted if lev.side == SR_SIDE_SUPPORT),
        default=None,
    )
    first_resistance_live = min(
        (lev.source_live_from_idx for lev in emitted if lev.side == SR_SIDE_RESISTANCE),
        default=None,
    )
    first_support_proj = _first_active_row(df, "nearest_support_active")
    first_resistance_proj = _first_active_row(df, "nearest_resistance_active")

    parity_ok: bool | str = "skipped"
    if live_df is not None:
        live_cols = [c for c in df.columns if not c.startswith("r_")]
        shared = [c for c in live_cols if c in live_df.columns]
        try:
            parity_ok = bool(df[shared].equals(live_df[shared]))
        except Exception:
            parity_ok = "error"

    contamination_clean = not any(
        c.startswith("label_") or c.startswith("future_") for c in df.columns
    )
    checks = {
        "live_research_parity_ok": parity_ok,
        "contamination_clean": contamination_clean,
        "no_pre_live_support_projection": (
            first_support_proj is None
            or first_support_live is None
            or first_support_proj >= first_support_live
        ),
        "no_pre_live_resistance_projection": (
            first_resistance_proj is None
            or first_resistance_live is None
            or first_resistance_proj >= first_resistance_live
        ),
        "no_inf_in_live_columns": _no_inf_in_live_columns(df),
        "warmup_has_no_primary_zone_before_first_live": (
            (
                first_support_live is None
                or first_support_live == 0
                or pd.to_numeric(
                    df.iloc[:first_support_live]["primary_support_zone_id"],
                    errors="coerce",
                )
                .dropna()
                .empty
            )
            and (
                first_resistance_live is None
                or first_resistance_live == 0
                or pd.to_numeric(
                    df.iloc[:first_resistance_live]["primary_resistance_zone_id"],
                    errors="coerce",
                )
                .dropna()
                .empty
            )
        ),
    }

    source_funnel = {
        "raw_source_count": int(len(all_levels)),
        "absorbed_source_count": int(len(absorbed)),
        "emitted_zone_count": int(len(emitted)),
        "active_zone_count": int(len(active)),
        "invalidated_zone_count": int(len(invalidated)),
        "retired_zone_count": int(len(retired)),
        "raw_sources_by_family": raw_by_family,
        "absorbed_sources_by_family": absorbed_by_family,
        "emitted_zones_by_family": emitted_by_family,
        "emitted_zones_by_best_family": emitted_by_best_family,
        "absorbed_into_family_matrix": absorption_matrix,
    }

    zone_geometry = {
        "width_atr": _continuous_stats(
            pd.Series([lev.zone_width_atr for lev in emitted], dtype=float)
        ),
        "anchor_count": _continuous_stats(
            pd.Series([lev.anchor_count for lev in emitted], dtype=float)
        ),
        "family_count": _continuous_stats(
            pd.Series([lev.family_count for lev in emitted], dtype=float)
        ),
        "family_mix_count_distribution": {
            int(k): int(v)
            for k, v in pd.Series([lev.family_count for lev in emitted], dtype=float)
            .value_counts()
            .sort_index()
            .items()
        },
        "width_bucket_distribution": {
            str(k): int(v)
            for k, v in pd.Series(
                [
                    (
                        "<=0.30"
                        if lev.zone_width_atr <= 0.30
                        else "0.30-0.45" if lev.zone_width_atr <= 0.45 else ">0.45"
                    )
                    for lev in emitted
                ],
                dtype=object,
            )
            .value_counts()
            .sort_index()
            .items()
        },
    }

    pending_count = int(
        sum(lev.failed_break_count + lev.accepted_break_count for lev in emitted)
    )
    interaction_quality = {
        "zones_with_touch_rate": (
            float(sum(1 for lev in emitted if lev.touch_count > 0)) / len(emitted)
            if emitted
            else 0.0
        ),
        "reclaim_rate": (
            float(sum(lev.reclaim_count for lev in emitted)) / pending_count
            if pending_count
            else None
        ),
        "false_break_rate": (
            float(sum(lev.failed_break_count for lev in emitted)) / pending_count
            if pending_count
            else None
        ),
        "accepted_break_rate": (
            float(sum(lev.accepted_break_count for lev in emitted)) / pending_count
            if pending_count
            else None
        ),
        "total_touch_count": int(sum(lev.touch_count for lev in emitted)),
        "total_reclaim_count": int(sum(lev.reclaim_count for lev in emitted)),
    }

    if touch_rows is None:
        touch_rows = build_sr_touch_audit_table(df, registry)
    calibration_json = None
    if "r_sr_score_quintile_calibration_json" in df.columns and not df.empty:
        try:
            calibration_json = json.loads(
                str(df["r_sr_score_quintile_calibration_json"].iloc[0])
            )
        except Exception:
            calibration_json = None
    score_calibration = calibration_json or {
        "touch_count": int(len(touch_rows)),
        "quintiles": {},
        "monotonicity": {},
        "top_vs_bottom_delta": {},
    }
    score_calibration["by_source_family"] = _segment_calibration(
        touch_rows, "best_source_family"
    )
    score_calibration["by_anchor_count_bucket"] = _segment_calibration(
        touch_rows, "anchor_count_bucket"
    )
    score_calibration["by_family_count_bucket"] = _segment_calibration(
        touch_rows, "family_count_bucket"
    )
    score_calibration["by_width_bucket"] = _segment_calibration(
        touch_rows, "width_bucket"
    )
    score_calibration["by_age_bucket"] = _segment_calibration(touch_rows, "age_bucket")
    score_calibration["by_touch_type"] = _segment_calibration(touch_rows, "touch_type")
    score_calibration["by_side"] = _segment_calibration(touch_rows, "side")
    score_calibration["component_audit"] = _component_audit(touch_rows)
    headline_verdict = _headline_verdict(
        checks,
        nearest_support_availability=ns_avail,
        nearest_resistance_availability=nr_avail,
        calibration=score_calibration,
    )

    distance_summary = {
        "support_distance_atr": _continuous_stats(df["nearest_support_distance_atr"]),
        "resistance_distance_atr": _continuous_stats(
            df["nearest_resistance_distance_atr"]
        ),
    }
    strength_by_side = {
        "support": {
            family: {
                "count": int(len(group)),
                "mean_strength": float(np.mean([lev.level_strength for lev in group])),
                "mean_source_quality": float(
                    np.mean([lev.source_quality_score for lev in group])
                ),
            }
            for family in ALL_FAMILIES
            for group in [
                [
                    lev
                    for lev in emitted
                    if lev.side == SR_SIDE_SUPPORT and lev.source_family == family
                ]
            ]
            if group
        },
        "resistance": {
            family: {
                "count": int(len(group)),
                "mean_strength": float(np.mean([lev.level_strength for lev in group])),
                "mean_source_quality": float(
                    np.mean([lev.source_quality_score for lev in group])
                ),
            }
            for family in ALL_FAMILIES
            for group in [
                [
                    lev
                    for lev in emitted
                    if lev.side == SR_SIDE_RESISTANCE and lev.source_family == family
                ]
            ]
            if group
        },
    }

    state_counts = {
        "active": int(sum(1 for lev in emitted if lev.state == SR_STATE_ACTIVE)),
        "active_weakened": int(
            sum(1 for lev in emitted if lev.state == SR_STATE_ACTIVE_WEAKENED)
        ),
        "break_pending": int(
            sum(1 for lev in emitted if lev.state == SR_STATE_BREAK_PENDING)
        ),
        "invalidated": int(len(invalidated)),
        "retired": int(len(retired)),
        "inactive_pre_live": int(sum(1 for lev in emitted if lev.state == 0)),
    }

    diagnostics = {
        "score_breakdown": {
            "source_family": _segment_calibration(touch_rows, "best_source_family"),
            "anchor_count_bucket": _segment_calibration(
                touch_rows, "anchor_count_bucket"
            ),
            "family_count_bucket": _segment_calibration(
                touch_rows, "family_count_bucket"
            ),
            "width_bucket": _segment_calibration(touch_rows, "width_bucket"),
            "age_bucket": _segment_calibration(touch_rows, "age_bucket"),
            "touch_type_bucket": _segment_calibration(touch_rows, "touch_type"),
        },
        "primary_selection": _primary_selection_summary(df, touch_rows),
        "absorption_diagnostics": {
            "raw_sources_by_family": raw_by_family,
            "absorbed_sources_by_family": absorbed_by_family,
            "emitted_zones_by_best_family": emitted_by_best_family,
            "absorbed_into_family_matrix": absorption_matrix,
        },
    }

    return {
        "row_count": int(n),
        **headline_verdict,
        "checks": checks,
        "source_funnel": source_funnel,
        "zone_geometry": zone_geometry,
        "interaction_quality": interaction_quality,
        "score_calibration": score_calibration,
        "state_counts": state_counts,
        "nearest_support_availability_rate": round(ns_avail, 4),
        "nearest_resistance_availability_rate": round(nr_avail, 4),
        "distance_summary": distance_summary,
        "strength_by_side": strength_by_side,
        "diagnostics": diagnostics,
        "benchmark_targets": {
            "baseline_raw_sources": 42116,
            "baseline_active_end_zones": 32,
            "emitted_reduction_vs_baseline": (
                1.0 - (len(emitted) / 42116.0) if emitted else 1.0
            ),
            "absorbed_share_of_raw_sources": (
                float(len(absorbed)) / len(all_levels) if all_levels else 0.0
            ),
        },
    }


_SUP_FILL = "rgba(80, 145, 230, 0.10)"
_RES_FILL = "rgba(224, 112, 89, 0.10)"
_SUP_LINE = "rgba(80, 145, 230, 0.34)"
_RES_LINE = "rgba(224, 112, 89, 0.34)"
_PRIMARY_SUP_LINE = "rgba(0, 109, 119, 0.96)"
_PRIMARY_SUP_FILL = "rgba(0, 109, 119, 0.18)"
_PRIMARY_RES_LINE = "rgba(188, 108, 37, 0.96)"
_PRIMARY_RES_FILL = "rgba(188, 108, 37, 0.18)"
_NEAREST_SUP_LINE = "rgba(38, 70, 83, 0.95)"
_NEAREST_RES_LINE = "rgba(84, 84, 84, 0.95)"


def _visible_zone_end(level: SRLevel, last_ts: Any, x_values: pd.Series) -> Any:
    if level.invalidation_idx >= 0 and level.invalidation_idx < len(x_values):
        return x_values.iloc[level.invalidation_idx]
    if level.retirement_idx >= 0 and level.retirement_idx < len(x_values):
        return x_values.iloc[level.retirement_idx]
    return last_ts


def _add_zone_rectangles(
    fig: go.Figure,
    x_values: pd.Series,
    plot_start: Any,
    plot_end: Any,
    registry: dict[int, SRLevel],
    *,
    row: int,
    col: int,
    max_per_side: int = 6,
) -> int:
    visible: list[tuple[SRLevel, Any, Any]] = []
    for level in registry.values():
        if not level.emitted_zone_flag:
            continue
        if level.source_live_from_idx >= len(x_values):
            continue
        x0 = x_values.iloc[level.source_live_from_idx]
        x1 = _visible_zone_end(level, plot_end, x_values)
        if x1 < plot_start:
            continue
        visible.append((level, max(x0, plot_start), x1))

    overlays = 0
    for side, fill, line in (
        (SR_SIDE_SUPPORT, _SUP_FILL, _SUP_LINE),
        (SR_SIDE_RESISTANCE, _RES_FILL, _RES_LINE),
    ):
        side_levels = [item for item in visible if item[0].side == side]
        side_levels.sort(key=lambda item: item[0].level_strength, reverse=True)
        for rank, (level, x0, x1) in enumerate(side_levels[:max_per_side], start=1):
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=level.zone_low,
                y1=level.zone_high,
                fillcolor=fill,
                line=dict(color=line, width=1),
                row=row,
                col=col,
                layer="below",
            )
            fig.add_trace(
                go.Scatter(
                    x=[x1],
                    y=[level.level_price],
                    mode="markers+text",
                    marker=dict(size=6, color=line),
                    text=[
                        f"{'S' if side == SR_SIDE_SUPPORT else 'R'}{rank} "
                        f"{level.best_source_family or level.source_family} "
                        f"{level.level_strength:.2f}"
                    ],
                    textposition="middle right",
                    textfont=dict(size=10, color=line),
                    name=f"{'Support' if side == SR_SIDE_SUPPORT else 'Resistance'} Zone",
                    showlegend=False,
                    hovertemplate=(
                        f"zone={level.level_id}<br>"
                        f"best_family={level.best_source_family or level.source_family}<br>"
                        f"score={level.level_strength:.3f}<br>"
                        f"anchors={level.anchor_count}<br>"
                        f"width_atr={level.zone_width_atr:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            overlays += 1
    return overlays


def _add_primary_zone_band(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    side: str,
    row: int,
    col: int,
) -> None:
    low_col = f"primary_{side}_zone_low"
    high_col = f"primary_{side}_zone_high"
    score_col = f"primary_{side}_zone_score"
    if low_col not in df.columns or high_col not in df.columns:
        return
    color = _PRIMARY_SUP_FILL if side == "support" else _PRIMARY_RES_FILL
    line = _PRIMARY_SUP_LINE if side == "support" else _PRIMARY_RES_LINE
    low = pd.to_numeric(df[low_col], errors="coerce")
    high = pd.to_numeric(df[high_col], errors="coerce")
    score = (
        pd.to_numeric(df[score_col], errors="coerce")
        if score_col in df.columns
        else None
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=high,
            mode="lines",
            line=dict(color=line, width=2.4),
            name=f"Primary {side.title()} High",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=low,
            mode="lines",
            line=dict(color=line, width=2.4),
            fill="tonexty",
            fillcolor=color,
            name=f"Primary {side.title()} Zone",
            showlegend=False,
            hovertemplate=(
                f"Primary {side}<br>"
                "low=%{y:.4f}<br>"
                + (
                    "score=%{customdata:.3f}<br><extra></extra>"
                    if score is not None
                    else "<extra></extra>"
                )
            ),
            customdata=score if score is not None else None,
        ),
        row=row,
        col=col,
    )


def validate_sr_levels(
    plot_df: pd.DataFrame,
    *,
    registry: dict[int, SRLevel],
    summary_df: pd.DataFrame,
    live_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "S/R Level Validation",
) -> dict:
    summary = summarize_sr_levels(summary_df, registry, live_df=live_df)
    if outpath is None:
        return summary

    fig = create_candlestick_figure(plot_df, title=title)
    x_values = pd.to_datetime(plot_df["timestamp"], utc=True, errors="coerce")
    plot_start = x_values.iloc[0]
    plot_end = x_values.iloc[-1]
    overlays = _add_zone_rectangles(
        fig,
        x_values.reset_index(drop=True),
        plot_start,
        plot_end,
        registry,
        row=1,
        col=1,
    )
    _add_primary_zone_band(fig, plot_df, side="support", row=1, col=1)
    _add_primary_zone_band(fig, plot_df, side="resistance", row=1, col=1)
    for col, color, name in (
        ("nearest_support_price", _NEAREST_SUP_LINE, "Nearest Support"),
        ("nearest_resistance_price", _NEAREST_RES_LINE, "Nearest Resistance"),
    ):
        if col not in plot_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=pd.to_numeric(plot_df[col], errors="coerce"),
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8, dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.005,
        y=0.995,
        text="Pale blocks: context zones | Teal/Orange bands: primary zones | Gray dashed: nearest levels",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(90,90,90,0.25)",
        borderwidth=1,
        font=dict(size=11, color="rgba(45,55,72,0.95)"),
    )
    save_figure_html(fig, outpath)
    summary["chart_path"] = str(outpath)
    summary["chart_overlay_count"] = int(overlays)
    return summary
