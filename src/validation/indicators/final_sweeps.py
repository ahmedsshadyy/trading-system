"""Validation summaries for the final sweeps detector (Step 11).

Validation contract (from the SweepsPlan)
-----------------------------------------
Print:
* sweeps by source family
* sweeps by side
* same-bar vs delayed sweeps
* accepted breakouts vs sweeps
* unresolved breaches
* source attribution distribution
* penetration distributions
* rejection latency distributions
* sweep quality distributions
* overlap with displacement / BOS / CHoCH
* regime-conditional sweep counts
* volume-confirmed sweep counts

This module also enforces the v1 doctrine boundary checks:
* range_boundary, FVG, OB sources must be absent from sweep attribution
* every confirmed sweep has a non-empty source family
* sweep_breach_idx <= sweep_confirm_idx (causality)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.indicators.sweeps_v2.final_sweeps import (
    FINAL_SWEEPS_COLUMNS,
    SWEEP_CLASS_ACCEPTED_BREAKOUT,
    SWEEP_CLASS_DELAYED_REJECTION,
    SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM,
    SWEEP_CLASS_NO_INTERACTION,
    SWEEP_CLASS_PROBED,
    SWEEP_CLASS_SAME_BAR,
    SWEEP_CLASS_SWEEP_THEN_BREAK,
    SWEEP_CLASS_UNRESOLVED,
)
from src.indicators.sweeps_v2.funnel_audit import build_interaction_audit
from src.indicators.sweeps_v2.unified_sources import (
    LIQ_SOURCE_FAMILIES,
    build_unified_liquidity_clusters_audit,
)

_CLASS_NAMES: dict[int, str] = {
    SWEEP_CLASS_NO_INTERACTION: "no_interaction",
    SWEEP_CLASS_PROBED: "probed",
    SWEEP_CLASS_UNRESOLVED: "unresolved_breach",
    SWEEP_CLASS_SAME_BAR: "same_bar_sweep",
    SWEEP_CLASS_DELAYED_REJECTION: "delayed_rejection_sweep",
    SWEEP_CLASS_ACCEPTED_BREAKOUT: "accepted_breakout",
    SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM: "failed_breakout_reclaim",
    SWEEP_CLASS_SWEEP_THEN_BREAK: "sweep_then_break",
}

_AGE_BUCKETS: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, np.inf)
_STRENGTH_BUCKETS: tuple[float, ...] = (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01)
_DISTANCE_BUCKETS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 1.00, 2.00, np.inf)
_PENETRATION_BUCKETS: tuple[float, ...] = (
    0.0,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    np.inf,
)
_FAMILY_GROUPS: dict[str, str] = {
    "support": "s_r",
    "resistance": "s_r",
    "equal_high": "eqh_eql",
    "equal_low": "eqh_eql",
    "session_high": "session",
    "session_low": "session",
    "previous_day_high": "prior_period",
    "previous_day_low": "prior_period",
    "previous_week_high": "prior_period",
    "previous_week_low": "prior_period",
    "swing_high": "swing",
    "swing_low": "swing",
}
_SESSION_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "session_high": ("prev_asia_high", "prev_london_high", "prev_ny_high"),
    "session_low": ("prev_asia_low", "prev_london_low", "prev_ny_low"),
}


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


def _bucket_labels(edges: tuple[float, ...]) -> list[str]:
    labels: list[str] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if np.isinf(right):
            labels.append(f"[{left:.2f}, inf)")
        else:
            labels.append(f"[{left:.2f}, {right:.2f})")
    return labels


def _bucketize(values: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    return pd.cut(values, bins=list(edges), right=False, labels=_bucket_labels(edges))


def _safe_int(value: object) -> int | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return int(num)


def build_sweep_source_event_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct one row per confirmed sweep with source metadata as it
    existed on the breach bar.

    ``sweep_source_id`` is a per-(bar, slot) projection id, not a stable
    cross-bar cluster id. For reuse diagnostics we therefore build both the
    raw projected id and a stable source-instance key using
    ``family|side|rounded_level|active_start_idx``.
    """

    if df is None or len(df) == 0 or "sweep_flag" not in df.columns:
        return pd.DataFrame()

    sweeps = df[df["sweep_flag"].fillna(0) > 0].copy()
    if sweeps.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for confirm_idx, row in sweeps.iterrows():
        breach_idx = _safe_int(row.get("sweep_breach_idx"))
        source_id = pd.to_numeric(row.get("sweep_source_id"), errors="coerce")
        if (
            breach_idx is None
            or pd.isna(source_id)
            or breach_idx < 0
            or breach_idx >= len(df)
        ):
            continue

        matched: dict[str, object] | None = None
        for side_label in ("above", "below"):
            for rank in range(1, 6):
                prefix = f"liq_{side_label}_l{rank}"
                cid = pd.to_numeric(
                    df.at[breach_idx, f"{prefix}_cluster_id"], errors="coerce"
                )
                if pd.isna(cid) or abs(float(cid) - float(source_id)) > 1e-9:
                    continue
                matched = {
                    "confirm_idx": int(confirm_idx),
                    "breach_idx": int(breach_idx),
                    "side_label": side_label,
                    "source_cluster_id_raw": float(source_id),
                    "family": str(df.at[breach_idx, f"{prefix}_primary_family"] or ""),
                    "level": float(
                        pd.to_numeric(
                            df.at[breach_idx, f"{prefix}_level"], errors="coerce"
                        )
                    ),
                    "strength": float(
                        pd.to_numeric(
                            df.at[breach_idx, f"{prefix}_strength"], errors="coerce"
                        )
                    ),
                    "age_bars": float(
                        pd.to_numeric(
                            df.at[breach_idx, f"{prefix}_age_bars"], errors="coerce"
                        )
                    ),
                    "origin_idx": _safe_int(df.at[breach_idx, f"{prefix}_origin_idx"]),
                    "active_start_idx": _safe_int(
                        df.at[breach_idx, f"{prefix}_active_start_idx"]
                    ),
                    "sweep_class": _safe_int(row.get("sweep_class")),
                    "tradeable": int(
                        pd.to_numeric(
                            row.get("sweep_is_tradeable_candidate"), errors="coerce"
                        )
                        > 0
                    ),
                    "selectivity_class": str(row.get("sweep_selectivity_class") or ""),
                    "micro_interaction": int(
                        pd.to_numeric(
                            row.get("sweep_is_micro_interaction"), errors="coerce"
                        )
                        > 0
                    ),
                    "standard_liquidity": int(
                        pd.to_numeric(
                            row.get("sweep_is_standard_liquidity"), errors="coerce"
                        )
                        > 0
                    ),
                    "displacement_confirmed": int(
                        pd.to_numeric(
                            row.get("sweep_is_displacement_confirmed"), errors="coerce"
                        )
                        > 0
                    ),
                    "quality": float(
                        pd.to_numeric(row.get("sweep_quality_score"), errors="coerce")
                    ),
                    "pre_breach_distance_atr": float(
                        pd.to_numeric(
                            row.get("sweep_pre_breach_distance_atr"), errors="coerce"
                        )
                    ),
                    "followed_by_displacement": int(
                        pd.to_numeric(
                            row.get("sweep_followed_by_displacement"), errors="coerce"
                        )
                        > 0
                    ),
                    "followed_by_bos": int(
                        pd.to_numeric(row.get("sweep_followed_by_bos"), errors="coerce")
                        > 0
                    ),
                    "followed_by_choch": int(
                        pd.to_numeric(
                            row.get("sweep_followed_by_choch"), errors="coerce"
                        )
                        > 0
                    ),
                    "followthrough_available": int(
                        pd.to_numeric(
                            row.get("sweep_research_followthrough_available"),
                            errors="coerce",
                        )
                        > 0
                    ),
                    "penetration_atr": float(
                        pd.to_numeric(
                            df.at[breach_idx, "penetration_atr"], errors="coerce"
                        )
                    ),
                    "active_source_count": float(
                        pd.to_numeric(
                            df.at[breach_idx, "liq_active_total_count"], errors="coerce"
                        )
                    ),
                }
                nearest_col = (
                    "liq_nearest_above_dist_atr"
                    if side_label == "above"
                    else "liq_nearest_below_dist_atr"
                )
                nearest_dist = float(
                    pd.to_numeric(df.at[breach_idx, nearest_col], errors="coerce")
                )
                matched["nearest_source_distance_atr_signed"] = nearest_dist
                matched["nearest_source_distance_atr_abs"] = (
                    abs(nearest_dist) if np.isfinite(nearest_dist) else np.nan
                )
                if matched["family"] in _SESSION_SOURCE_COLUMNS:
                    session_subtype = "unmatched_session_source"
                    for col in _SESSION_SOURCE_COLUMNS[matched["family"]]:
                        if col not in df.columns:
                            continue
                        ref = float(
                            pd.to_numeric(df.at[breach_idx, col], errors="coerce")
                        )
                        if (
                            np.isfinite(ref)
                            and np.isfinite(matched["level"])
                            and abs(ref - matched["level"]) <= 1e-9
                        ):
                            session_subtype = col
                            break
                    matched["session_source_type"] = session_subtype
                    matched["session_source_scope"] = "completed_previous_session_only"
                else:
                    matched["session_source_type"] = ""
                    matched["session_source_scope"] = ""
                matched["breach_session_phase"] = (
                    str(df.at[breach_idx, "session_name"])
                    if "session_name" in df.columns
                    else ""
                )
                active_start_idx = matched["active_start_idx"]
                history_max = float("nan")
                if (
                    active_start_idx is not None
                    and active_start_idx >= 0
                    and breach_idx > active_start_idx
                ):
                    distances: list[float] = []
                    for j in range(int(active_start_idx), int(breach_idx)):
                        atr = float(pd.to_numeric(df.at[j, "atr_14"], errors="coerce"))
                        close = float(pd.to_numeric(df.at[j, "close"], errors="coerce"))
                        if not (np.isfinite(atr) and atr > 0 and np.isfinite(close)):
                            continue
                        dist = (
                            (matched["level"] - close) / atr
                            if side_label == "above"
                            else (close - matched["level"]) / atr
                        )
                        if np.isfinite(dist):
                            distances.append(dist)
                    if distances:
                        history_max = float(max(distances))
                matched["historical_max_distance_atr"] = history_max
                break
            if matched is not None:
                break

        if matched is None:
            continue

        active_start_idx = matched.get("active_start_idx")
        stable_level = (
            round(float(matched["level"]), 4)
            if np.isfinite(matched["level"])
            else np.nan
        )
        matched["source_level_key"] = (
            f"{matched['family']}|{matched['side_label']}|{stable_level}"
        )
        matched["source_instance_key"] = (
            f"{matched['family']}|{matched['side_label']}|{stable_level}|"
            f"{active_start_idx if active_start_idx is not None else -1}"
        )
        matched["family_group"] = _FAMILY_GROUPS.get(str(matched["family"]), "other")
        rows.append(matched)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["confirm_idx", "breach_idx"])
        .reset_index(drop=True)
    )


def build_final_sweeps_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    if df is None or len(df) == 0:
        return {
            "family_funnel": pd.DataFrame(),
            "age_buckets": pd.DataFrame(),
            "strength_buckets": pd.DataFrame(),
            "distance_buckets": pd.DataFrame(),
            "selectivity_class_table": pd.DataFrame(),
            "selectivity_family_table": pd.DataFrame(),
            "selectivity_distance_table": pd.DataFrame(),
            "selectivity_penetration_table": pd.DataFrame(),
            "selectivity_quality_table": pd.DataFrame(),
            "session_funnel": pd.DataFrame(),
            "session_age_buckets": pd.DataFrame(),
            "session_distance_buckets": pd.DataFrame(),
            "session_penetration_buckets": pd.DataFrame(),
            "session_phase_table": pd.DataFrame(),
            "session_source_type_table": pd.DataFrame(),
            "repeated_sweeps": pd.DataFrame(),
            "family_group_rates": pd.DataFrame(),
            "source_events": pd.DataFrame(),
            "negative_distance_explanation": "",
        }

    audit = build_interaction_audit(df)
    source_events = build_sweep_source_event_table(df)
    active_audit = build_unified_liquidity_clusters_audit(df)

    active_counts = (
        active_audit.groupby("primary_family").size().rename("active_source_rows")
        if not active_audit.empty
        else pd.Series(dtype="int64")
    )
    probe_counts = (
        audit.groupby("family")["stage"]
        .apply(lambda s: int((s != "eligible_only").sum()))
        .rename("probe_count")
        if not audit.empty
        else pd.Series(dtype="int64")
    )
    breach_counts = (
        df.loc[df["sweep_breach_flag"].fillna(0) > 0, "sweep_breach_source_family"]
        .astype(str)
        .value_counts()
        .rename("breach_count")
    )
    sweep_counts = (
        source_events["family"].value_counts().rename("sweep_count")
        if not source_events.empty
        else pd.Series(dtype="int64")
    )
    tradeable_counts = (
        source_events.loc[source_events["tradeable"] > 0, "family"]
        .value_counts()
        .rename("tradeable_count")
        if not source_events.empty
        else pd.Series(dtype="int64")
    )

    family_funnel = (
        pd.DataFrame(index=list(LIQ_SOURCE_FAMILIES))
        .join(
            [
                active_counts,
                probe_counts,
                breach_counts,
                sweep_counts,
                tradeable_counts,
            ],
            how="left",
        )
        .fillna(0.0)
    )
    family_funnel = family_funnel.reset_index().rename(columns={"index": "family"})
    for col in (
        "active_source_rows",
        "probe_count",
        "breach_count",
        "sweep_count",
        "tradeable_count",
    ):
        family_funnel[col] = family_funnel[col].astype(int)
    family_funnel["probe_per_active"] = family_funnel["probe_count"] / family_funnel[
        "active_source_rows"
    ].replace(0, np.nan)
    family_funnel["breach_per_probe"] = family_funnel["breach_count"] / family_funnel[
        "probe_count"
    ].replace(0, np.nan)
    family_funnel["sweep_per_breach"] = family_funnel["sweep_count"] / family_funnel[
        "breach_count"
    ].replace(0, np.nan)
    family_funnel["tradeable_per_sweep"] = family_funnel[
        "tradeable_count"
    ] / family_funnel["sweep_count"].replace(0, np.nan)
    family_funnel["sweep_per_active"] = family_funnel["sweep_count"] / family_funnel[
        "active_source_rows"
    ].replace(0, np.nan)

    def _bucket_count_table(
        column: str, edges: tuple[float, ...], bucket_name: str
    ) -> pd.DataFrame:
        if source_events.empty:
            return pd.DataFrame(columns=[bucket_name, "sweep_count", "tradeable_count"])
        bucketed = source_events.copy()
        bucketed[bucket_name] = _bucketize(bucketed[column], edges)
        out = (
            bucketed.groupby(bucket_name, observed=False)
            .agg(
                sweep_count=("family", "size"),
                tradeable_count=("tradeable", "sum"),
            )
            .reset_index()
        )
        out[bucket_name] = out[bucket_name].astype(str)
        return out

    age_buckets = _bucket_count_table("age_bars", _AGE_BUCKETS, "age_bucket")
    strength_buckets = _bucket_count_table(
        "strength", _STRENGTH_BUCKETS, "strength_bucket"
    )
    distance_buckets = _bucket_count_table(
        "nearest_source_distance_atr_abs",
        _DISTANCE_BUCKETS,
        "nearest_distance_atr_bucket",
    )

    if not source_events.empty:
        selectivity = source_events.copy()
        selectivity["selectivity_class"] = selectivity["selectivity_class"].replace(
            "", "unclassified"
        )
        selectivity_class_table = (
            selectivity.groupby("selectivity_class", as_index=False)
            .agg(
                sweep_count=("family", "size"),
                tradeable_count=("tradeable", "sum"),
                followed_by_displacement_count=("followed_by_displacement", "sum"),
                followed_by_bos_count=("followed_by_bos", "sum"),
                followed_by_choch_count=("followed_by_choch", "sum"),
                quality_mean=("quality", "mean"),
                quality_p50=("quality", "median"),
                quality_p90=("quality", lambda s: float(pd.Series(s).quantile(0.90))),
            )
            .sort_values("sweep_count", ascending=False)
            .reset_index(drop=True)
        )
        selectivity_class_table["tradeable_rate"] = selectivity_class_table[
            "tradeable_count"
        ] / selectivity_class_table["sweep_count"].replace(0, np.nan)
        selectivity_class_table["displacement_rate"] = selectivity_class_table[
            "followed_by_displacement_count"
        ] / selectivity_class_table["sweep_count"].replace(0, np.nan)
        selectivity_class_table["bos_rate"] = selectivity_class_table[
            "followed_by_bos_count"
        ] / selectivity_class_table["sweep_count"].replace(0, np.nan)
        selectivity_class_table["choch_rate"] = selectivity_class_table[
            "followed_by_choch_count"
        ] / selectivity_class_table["sweep_count"].replace(0, np.nan)
        selectivity_family_table = selectivity.pivot_table(
            index="family",
            columns="selectivity_class",
            values="confirm_idx",
            aggfunc="count",
            fill_value=0,
        ).reset_index()
        selectivity_distance_table = (
            selectivity.assign(
                pre_breach_distance_bucket=_bucketize(
                    selectivity["pre_breach_distance_atr"], _DISTANCE_BUCKETS
                )
            )
            .groupby(
                ["pre_breach_distance_bucket", "selectivity_class"],
                observed=False,
                as_index=False,
            )
            .agg(sweep_count=("family", "size"), tradeable_count=("tradeable", "sum"))
        )
        selectivity_distance_table["pre_breach_distance_bucket"] = (
            selectivity_distance_table["pre_breach_distance_bucket"].astype(str)
        )
        selectivity_penetration_table = (
            selectivity.assign(
                penetration_bucket=_bucketize(
                    selectivity["penetration_atr"], _PENETRATION_BUCKETS
                )
            )
            .groupby(
                ["penetration_bucket", "selectivity_class"],
                observed=False,
                as_index=False,
            )
            .agg(sweep_count=("family", "size"), tradeable_count=("tradeable", "sum"))
        )
        selectivity_penetration_table["penetration_bucket"] = (
            selectivity_penetration_table["penetration_bucket"].astype(str)
        )
        quality_rows: list[dict[str, object]] = []
        for cls, sub in selectivity.groupby("selectivity_class"):
            quality_rows.append(
                {"selectivity_class": cls, **_quantiles(sub["quality"])}
            )
        selectivity_quality_table = pd.DataFrame(quality_rows)
    else:
        selectivity_class_table = pd.DataFrame()
        selectivity_family_table = pd.DataFrame()
        selectivity_distance_table = pd.DataFrame()
        selectivity_penetration_table = pd.DataFrame()
        selectivity_quality_table = pd.DataFrame()

    session_events = (
        source_events[
            source_events["family"].isin(["session_high", "session_low"])
        ].copy()
        if not source_events.empty
        else pd.DataFrame()
    )
    if not session_events.empty:
        session_families = family_funnel[
            family_funnel["family"].isin(["session_high", "session_low"])
        ].reset_index(drop=True)
        session_funnel = pd.concat(
            [
                session_families,
                pd.DataFrame(
                    [
                        {
                            "family": "session_total",
                            "active_source_rows": int(
                                session_families["active_source_rows"].sum()
                            ),
                            "probe_count": int(session_families["probe_count"].sum()),
                            "breach_count": int(session_families["breach_count"].sum()),
                            "sweep_count": int(session_families["sweep_count"].sum()),
                            "tradeable_count": int(
                                session_families["tradeable_count"].sum()
                            ),
                            "probe_per_active": (
                                float(session_families["probe_count"].sum())
                                / max(
                                    float(session_families["active_source_rows"].sum()),
                                    1.0,
                                )
                            ),
                            "breach_per_probe": (
                                float(session_families["breach_count"].sum())
                                / max(float(session_families["probe_count"].sum()), 1.0)
                            ),
                            "sweep_per_breach": (
                                float(session_families["sweep_count"].sum())
                                / max(
                                    float(session_families["breach_count"].sum()), 1.0
                                )
                            ),
                            "tradeable_per_sweep": (
                                float(session_families["tradeable_count"].sum())
                                / max(float(session_families["sweep_count"].sum()), 1.0)
                            ),
                            "sweep_per_active": (
                                float(session_families["sweep_count"].sum())
                                / max(
                                    float(session_families["active_source_rows"].sum()),
                                    1.0,
                                )
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        session_age_buckets = (
            session_events.assign(
                age_bucket=_bucketize(session_events["age_bars"], _AGE_BUCKETS)
            )
            .groupby("age_bucket", observed=False)
            .agg(sweep_count=("family", "size"), tradeable_count=("tradeable", "sum"))
            .reset_index()
        )
        session_age_buckets["age_bucket"] = session_age_buckets["age_bucket"].astype(
            str
        )
        session_distance_buckets = (
            session_events.assign(
                nearest_distance_atr_bucket=_bucketize(
                    session_events["nearest_source_distance_atr_abs"], _DISTANCE_BUCKETS
                )
            )
            .groupby("nearest_distance_atr_bucket", observed=False)
            .agg(sweep_count=("family", "size"), tradeable_count=("tradeable", "sum"))
            .reset_index()
        )
        session_distance_buckets["nearest_distance_atr_bucket"] = (
            session_distance_buckets["nearest_distance_atr_bucket"].astype(str)
        )
        penetration_edges = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.60, np.inf)
        session_penetration_buckets = (
            session_events.assign(
                penetration_bucket=_bucketize(
                    session_events["penetration_atr"], penetration_edges
                )
            )
            .groupby("penetration_bucket", observed=False)
            .agg(sweep_count=("family", "size"), tradeable_count=("tradeable", "sum"))
            .reset_index()
        )
        session_penetration_buckets["penetration_bucket"] = session_penetration_buckets[
            "penetration_bucket"
        ].astype(str)
        session_phase_table = (
            session_events.groupby("breach_session_phase", as_index=False)
            .agg(
                sweep_count=("family", "size"),
                tradeable_count=("tradeable", "sum"),
            )
            .sort_values("sweep_count", ascending=False)
            .reset_index(drop=True)
        )
        session_source_type_table = (
            session_events.groupby(
                ["session_source_scope", "session_source_type"], as_index=False
            )
            .agg(
                sweep_count=("family", "size"),
                tradeable_count=("tradeable", "sum"),
            )
            .sort_values("sweep_count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        session_funnel = pd.DataFrame()
        session_age_buckets = pd.DataFrame()
        session_distance_buckets = pd.DataFrame()
        session_penetration_buckets = pd.DataFrame()
        session_phase_table = pd.DataFrame()
        session_source_type_table = pd.DataFrame()

    repeated_rows: list[dict[str, object]] = []
    if not source_events.empty:
        ordered = source_events.sort_values("confirm_idx")
        for key_name in (
            "source_cluster_id_raw",
            "source_level_key",
            "source_instance_key",
        ):
            for window in (5, 10, 20):
                repeat_count = 0
                for _, sub in ordered.groupby(key_name):
                    bars = sub["confirm_idx"].to_numpy(dtype=int)
                    if bars.size <= 1:
                        continue
                    repeat_count += int(((bars[1:] - bars[:-1]) <= window).sum())
                repeated_rows.append(
                    {
                        "key_type": key_name,
                        "window_bars": window,
                        "repeat_count": int(repeat_count),
                    }
                )
    repeated_sweeps = pd.DataFrame(repeated_rows)

    if not family_funnel.empty:
        family_groups = family_funnel.copy()
        family_groups["family_group"] = family_groups["family"].map(
            lambda fam: _FAMILY_GROUPS.get(str(fam), "other")
        )
        family_group_rates = family_groups.groupby("family_group", as_index=False)[
            ["active_source_rows", "sweep_count", "tradeable_count"]
        ].sum()
        family_group_rates["sweep_per_active"] = family_group_rates[
            "sweep_count"
        ] / family_group_rates["active_source_rows"].replace(0, np.nan)
        family_group_rates["tradeable_per_sweep"] = family_group_rates[
            "tradeable_count"
        ] / family_group_rates["sweep_count"].replace(0, np.nan)
    else:
        family_group_rates = pd.DataFrame()

    negative_distance_explanation = (
        "Expected, not a bug: unified sources are admitted on the correct side of "
        "price at bar start (open / prior close fallback), but nearest distances "
        "are measured against the bar close. If price closes through a source "
        "intrabar, the signed nearest above/below distance can go negative."
    )

    return {
        "family_funnel": family_funnel,
        "age_buckets": age_buckets,
        "strength_buckets": strength_buckets,
        "distance_buckets": distance_buckets,
        "selectivity_class_table": selectivity_class_table,
        "selectivity_family_table": selectivity_family_table,
        "selectivity_distance_table": selectivity_distance_table,
        "selectivity_penetration_table": selectivity_penetration_table,
        "selectivity_quality_table": selectivity_quality_table,
        "session_funnel": session_funnel,
        "session_age_buckets": session_age_buckets,
        "session_distance_buckets": session_distance_buckets,
        "session_penetration_buckets": session_penetration_buckets,
        "session_phase_table": session_phase_table,
        "session_source_type_table": session_source_type_table,
        "repeated_sweeps": repeated_sweeps,
        "family_group_rates": family_group_rates,
        "source_events": source_events,
        "negative_distance_explanation": negative_distance_explanation,
    }


def summarize_final_sweeps(
    df: pd.DataFrame,
    *,
    scan_timeframe: str,
) -> dict[str, Any]:
    """Compute the canonical final-sweeps validation dict."""

    if df is None or len(df) == 0:
        return {"error": "empty_frame"}
    for col in ("sweep_flag", "sweep_class", "sweep_primary_family"):
        if col not in df.columns:
            return {"error": f"missing_column:{col}"}

    sweeps = df[df["sweep_flag"].fillna(0) > 0].copy()
    breaches = df[df["sweep_breach_flag"].fillna(0) > 0].copy()

    # Causality: every confirmed sweep has breach_idx <= confirm_idx.
    causality_violations = int(
        (
            (
                sweeps["sweep_breach_idx"].fillna(-1)
                > sweeps["sweep_confirm_idx"].fillna(-1)
            )
        ).sum()
    )

    # Family attribution
    family_counts = (
        sweeps["sweep_primary_family"]
        .astype(str)
        .replace("", "none")
        .value_counts()
        .to_dict()
    )
    deprecated_in_attr = []
    if "sweep_attribution_families" in sweeps.columns:
        for raw in sweeps["sweep_attribution_families"].astype(str).fillna(""):
            for fam in raw.split("|"):
                if fam.startswith(("range_", "fvg_", "ob_")):
                    deprecated_in_attr.append(fam)
    deprecated_in_attr = sorted(set(deprecated_in_attr))

    # Class distribution
    class_counts_raw = df["sweep_class"].dropna().astype(int).value_counts().to_dict()
    class_counts = {
        _CLASS_NAMES.get(k, f"class_{k}"): int(v) for k, v in class_counts_raw.items()
    }

    # Side distribution
    side_counts = (
        sweeps["sweep_side"]
        .dropna()
        .astype(int)
        .map({+1: "above_buy_side", -1: "below_sell_side"})
        .value_counts()
        .to_dict()
    )

    # Same-bar vs delayed
    same_bar = int((sweeps["sweep_class"] == SWEEP_CLASS_SAME_BAR).sum())
    delayed = int((sweeps["sweep_class"] == SWEEP_CLASS_DELAYED_REJECTION).sum())

    # Accepted breakouts vs sweeps
    accepted_breakouts = int((df["sweep_class"] == SWEEP_CLASS_ACCEPTED_BREAKOUT).sum())
    confirmed_sweeps = int(
        df["sweep_class"]
        .isin([SWEEP_CLASS_SAME_BAR, SWEEP_CLASS_DELAYED_REJECTION])
        .sum()
    )
    unresolved = int((df["sweep_class"] == SWEEP_CLASS_UNRESOLVED).sum())

    # Penetration distributions (at breach bar)
    penetration_atr_dist = _quantiles(breaches["penetration_atr"])
    penetration_abs_dist = _quantiles(breaches["penetration_abs"])
    pen_width_frac_dist = _quantiles(breaches["penetration_source_width_frac"])

    # Rejection latency
    latency_dist = _quantiles(sweeps["sweep_latency_bars"])

    # Quality distributions
    quality_dist = _quantiles(sweeps["sweep_quality_score"])
    component_dists: dict[str, dict[str, float]] = {}
    for c in (
        "sweep_q_source_strength",
        "sweep_q_penetration",
        "sweep_q_rejection",
        "sweep_q_displacement_followthrough",
        "sweep_q_regime_context",
        "sweep_q_volume_confirmation",
        "sweep_q_crowding",
    ):
        if c in df.columns:
            component_dists[c] = _quantiles(df[c])

    # Follow-through overlap
    followthrough = {}
    for col in (
        "sweep_followed_by_displacement",
        "sweep_followed_by_bos",
        "sweep_followed_by_choch",
    ):
        if col in df.columns:
            followthrough[col] = int(df.loc[df["sweep_flag"] > 0, col].fillna(0).sum())

    # Volume / regime confluence (binary tagging at confirm bars)
    volume_confirmed = 0
    if "vol_ratio" in df.columns:
        volume_confirmed = int(
            ((df["sweep_flag"] > 0) & (df["vol_ratio"].fillna(0) > 1.0)).sum()
        )

    regime_conditional = {}
    if "regime" in df.columns:
        regime_conditional = (
            df.loc[df["sweep_flag"] > 0, "regime"]
            .dropna()
            .astype(int)
            .value_counts()
            .to_dict()
        )
    if "regime_label" in df.columns:
        regime_conditional_labels = (
            df.loc[df["sweep_flag"] > 0, "regime_label"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )
    else:
        regime_conditional_labels = {}

    # Tradeable-candidate filter rate
    tradeable = int(sweeps["sweep_is_tradeable_candidate"].fillna(0).sum())
    tradeable_rate = round(tradeable / max(len(sweeps), 1), 4) if len(sweeps) else 0.0

    # Interaction phase distribution (per side, top rank)
    interaction_dist: dict[str, dict[str, int]] = {}
    for side in ("above", "below"):
        col = f"sweep_interaction_phase_{side}_l1"
        if col in df.columns:
            counts = df[col].dropna().astype(int).value_counts().to_dict()
            interaction_dist[side] = counts

    return {
        "scan_timeframe": scan_timeframe,
        "row_count": int(len(df)),
        "confirmed_sweeps_count": confirmed_sweeps,
        "accepted_breakouts_count": accepted_breakouts,
        "unresolved_breaches_count": unresolved,
        "breach_count_total": int(len(breaches)),
        "same_bar_sweeps": same_bar,
        "delayed_rejection_sweeps": delayed,
        "tradeable_candidates": tradeable,
        "tradeable_rate": tradeable_rate,
        "causality_violations": causality_violations,
        "deprecated_families_in_attribution": deprecated_in_attr,
        "sweeps_by_class": class_counts,
        "sweeps_by_side": side_counts,
        "sweeps_by_primary_family": family_counts,
        "penetration_atr": penetration_atr_dist,
        "penetration_abs": penetration_abs_dist,
        "penetration_source_width_frac": pen_width_frac_dist,
        "latency_bars": latency_dist,
        "sweep_quality_score": quality_dist,
        "quality_components": component_dists,
        "followthrough_overlap_count": followthrough,
        "volume_confirmed_sweeps": volume_confirmed,
        "regime_conditional_sweep_counts": regime_conditional,
        "regime_conditional_sweep_labels": regime_conditional_labels,
        "interaction_phase_distribution_l1": interaction_dist,
        "schema_columns": list(FINAL_SWEEPS_COLUMNS),
    }


def print_final_sweeps_summary(summary: Mapping[str, Any], indent: int = 0) -> None:
    """Pretty-print the summary dict — matches the existing convention used
    by ``validate_equal_hl.py`` and friends.
    """

    prefix = " " * indent
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_final_sweeps_summary(value, indent + 2)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {value[:8]}{' ...' if len(value) > 8 else ''}")
        else:
            print(f"{prefix}{key}: {value}")


__all__ = [
    "build_sweep_source_event_table",
    "build_final_sweeps_diagnostics",
    "summarize_final_sweeps",
    "print_final_sweeps_summary",
]
