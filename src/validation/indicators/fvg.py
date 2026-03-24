from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.research.fvg_research import (
    summarize_fvg_research,
    summarize_old_no_expiry_unresolved_audit,
)
from src.indicators.smc.fvg import ALL_FVG_CORE_COLUMNS, STATE_INVALIDATED
from src.indicators.smc.fvg_fill import (
    _build_member_snapshots_from_core,
    _is_touched_candidate,
    _pick_fill_active_rep,
)
from src.indicators.smc.ifvg import (
    IFVG_STATE_ACTIVE_FAILING,
    IFVG_STATE_ACTIVE_HOLDING,
    IFVG_STATE_EXPIRED,
    IFVG_STATE_FULLY_RECLAIMED,
    MAX_IFVG_ACTIVE_AGE_BARS,
    _IfvgEvent,
    _advance_ifvg_event,
    _build_invalidated_source_candidates,
)

ACTIVE_STATES = {1, 2, 3}
CLEANLINESS_SUMMARY_CAP = 5.0
FORENSIC_WINDOW_START = pd.Timestamp("2026-01-11", tz="UTC")
FORENSIC_WINDOW_END = pd.Timestamp("2026-03-13 23:59:59", tz="UTC")


def _continuous_stats(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _capped_cleanliness_stats(series: pd.Series) -> dict[str, float | int]:
    clean = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .clip(upper=CLEANLINESS_SUMMARY_CAP)
    )
    return _continuous_stats(clean)


def _value_counts(series: pd.Series) -> dict[int | float | str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    out: dict[int | float | str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            out["NaN"] = int(value)
        elif isinstance(key, (np.integer, int)):
            out[int(key)] = int(value)
        elif isinstance(key, (np.floating, float)):
            out[float(key)] = int(value)
        else:
            out[str(key)] = int(value)
    return out


def _bar_delta(ts: pd.Series) -> pd.Timedelta:
    diffs = pd.Series(pd.to_datetime(ts, utc=True)).diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=4)
    return diffs.mode().iloc[0]


def _remaining_geometry_from_zone(
    *,
    side: str,
    low: object,
    high: object,
    fill_pct: object,
) -> dict[str, float]:
    if pd.isna(low) or pd.isna(high) or pd.isna(fill_pct):
        return {
            "remaining_low": np.nan,
            "remaining_high": np.nan,
            "remaining_width": np.nan,
            "remaining_frac": np.nan,
        }

    zone_low = float(low)
    zone_high = float(high)
    zone_width = zone_high - zone_low
    if not np.isfinite(zone_width) or zone_width < 0:
        return {
            "remaining_low": np.nan,
            "remaining_high": np.nan,
            "remaining_width": np.nan,
            "remaining_frac": np.nan,
        }

    clipped_fill_pct = float(np.clip(float(fill_pct), 0.0, 1.0))
    remaining_width = max(zone_width * (1.0 - clipped_fill_pct), 0.0)
    if side == "bull":
        remaining_low = zone_low
        remaining_high = zone_high - (clipped_fill_pct * zone_width)
    else:
        remaining_low = zone_low + (clipped_fill_pct * zone_width)
        remaining_high = zone_high
    return {
        "remaining_low": remaining_low,
        "remaining_high": remaining_high,
        "remaining_width": remaining_width,
        "remaining_frac": (remaining_width / zone_width if zone_width > 0 else np.nan),
    }


def _pool_side_diagnostics(
    *,
    side: str,
    window_rows: pd.Index,
    full_df: pd.DataFrame,
    active_members: pd.DataFrame,
) -> dict[str, object]:
    member_side = active_members[
        (active_members["side"] == side)
        & (active_members["row_idx"] >= int(window_rows.min()))
        & (active_members["row_idx"] <= int(window_rows.max()))
    ].copy()
    all_rows = pd.Index(window_rows, name="row_idx")

    if member_side.empty:
        dense_count = (
            full_df.loc[window_rows, f"fvg_{side}_active_count"].fillna(0).astype(int)
        )
        dense_selected = (
            full_df.loc[window_rows, f"fvg_{side}_active"].fillna(0).astype(int)
        )
        return {
            "active_count_on_first_row": int(dense_count.iloc[0]),
            "active_count_on_last_row": int(dense_count.iloc[-1]),
            "carried_in_active_count_on_first_row": 0,
            "carried_in_active_count_on_last_row": 0,
            "carried_in_active_count_after_expiry_filter": 0,
            "max_active_count": int(dense_count.max()),
            "max_active_count_from_in_window_events_only": 0,
            "max_active_count_from_pre_window_carried_events": 0,
            "max_active_count_excluding_merged_sources": 0,
            "max_active_count_merged_reps_only": 0,
            "max_active_age_in_pool": 0,
            "terminal_but_still_in_active_pool_count": 0,
            "active_pool_unique_ids_max": 0,
            "active_pool_rowwise_unique_ids_consistent": True,
            "active_count_matches_dense_export": bool((dense_count == 0).all()),
            "selected_rep_count_matches_dense_export": bool(
                (dense_selected == 0).all()
            ),
        }

    start_idx = int(window_rows.min())
    grouped = member_side.groupby("row_idx").agg(
        active_member_count=("active_id", "size"),
        unique_active_ids=("active_id", "nunique"),
        carried_in_count=("detect_idx", lambda s: int((s < start_idx).sum())),
        in_window_count=("detect_idx", lambda s: int((s >= start_idx).sum())),
        merged_rep_count=("source_count", lambda s: int((s > 1).sum())),
        non_merged_rep_count=("source_count", lambda s: int((s == 1).sum())),
        terminal_member_count=("state", lambda s: int((~s.isin(ACTIVE_STATES)).sum())),
        selected_rep_count=("selected", "sum"),
    )
    grouped = grouped.reindex(all_rows, fill_value=0)
    age_anchor = (
        member_side["age_anchor_idx"]
        if "age_anchor_idx" in member_side.columns
        else member_side["detect_idx"]
    )
    member_side["active_age"] = member_side["row_idx"] - age_anchor
    grouped["max_active_age"] = (
        member_side.groupby("row_idx")["active_age"]
        .max()
        .reindex(all_rows, fill_value=0)
    )

    dense_count = (
        full_df.loc[window_rows, f"fvg_{side}_active_count"].fillna(0).astype(int)
    )
    dense_selected = (
        full_df.loc[window_rows, f"fvg_{side}_active"].fillna(0).astype(int)
    )

    return {
        "active_count_on_first_row": int(grouped["active_member_count"].iloc[0]),
        "active_count_on_last_row": int(grouped["active_member_count"].iloc[-1]),
        "carried_in_active_count_on_first_row": int(
            grouped["carried_in_count"].iloc[0]
        ),
        "carried_in_active_count_on_last_row": int(
            grouped["carried_in_count"].iloc[-1]
        ),
        "carried_in_active_count_after_expiry_filter": int(
            grouped["carried_in_count"].iloc[0]
        ),
        "max_active_count": int(grouped["active_member_count"].max()),
        "max_active_count_from_in_window_events_only": int(
            grouped["in_window_count"].max()
        ),
        "max_active_count_from_pre_window_carried_events": int(
            grouped["carried_in_count"].max()
        ),
        "max_active_count_excluding_merged_sources": int(
            grouped["non_merged_rep_count"].max()
        ),
        "max_active_count_merged_reps_only": int(grouped["merged_rep_count"].max()),
        "max_active_age_in_pool": int(grouped["max_active_age"].max()),
        "terminal_but_still_in_active_pool_count": int(
            grouped["terminal_member_count"].sum()
        ),
        "active_pool_unique_ids_max": int(grouped["unique_active_ids"].max()),
        "active_pool_rowwise_unique_ids_consistent": bool(
            (grouped["active_member_count"] == grouped["unique_active_ids"]).all()
        ),
        "active_count_matches_dense_export": bool(
            (grouped["active_member_count"].to_numpy() == dense_count.to_numpy()).all()
        ),
        "selected_rep_count_matches_dense_export": bool(
            (
                grouped["selected_rep_count"].to_numpy() == dense_selected.to_numpy()
            ).all()
        ),
    }


def _forensic_window_reconciliation(
    research_table: pd.DataFrame,
    *,
    start: pd.Timestamp = FORENSIC_WINDOW_START,
    end: pd.Timestamp = FORENSIC_WINDOW_END,
) -> dict[str, object]:
    if research_table.empty:
        return {
            "window": {"start": str(start), "end": str(end), "event_count": 0},
            "core_terminal_distribution": {},
            "research_final_distribution": {},
            "core_terminal_vs_research_final_crosstab": {},
            "expired_event_reconciliation_table": [],
        }

    detect_ts = pd.to_datetime(research_table["fvg_detect_ts"], utc=True)
    window_events = research_table[(detect_ts >= start) & (detect_ts <= end)].copy()
    if window_events.empty:
        return {
            "window": {"start": str(start), "end": str(end), "event_count": 0},
            "core_terminal_distribution": {},
            "research_final_distribution": {},
            "core_terminal_vs_research_final_crosstab": {},
            "expired_event_reconciliation_table": [],
        }

    crosstab = pd.crosstab(
        window_events["fvg_terminal_label"],
        window_events["fvg_r_final_outcome"],
        dropna=False,
    )
    expired_events = window_events[window_events["fvg_terminal_label"] == "expired"][
        [
            "fvg_event_id",
            "fvg_side",
            "fvg_detect_ts",
            "fvg_expiry_idx",
            "fvg_r_ever_touched",
            "fvg_r_ever_partial_fill",
            "fvg_r_continuation_without_touch_flag",
            "fvg_r_touched_rejected_flag",
            "fvg_terminal_label",
            "fvg_r_final_outcome",
        ]
    ].copy()
    expired_events = expired_events.rename(
        columns={
            "fvg_side": "side",
            "fvg_expiry_idx": "core_expiry_idx",
            "fvg_r_ever_touched": "ever_touched",
            "fvg_r_ever_partial_fill": "ever_partial_fill",
        }
    )

    return {
        "window": {
            "start": str(start),
            "end": str(end),
            "event_count": int(len(window_events)),
        },
        "core_terminal_distribution": {
            str(k): int(v)
            for k, v in window_events["fvg_terminal_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "research_final_distribution": {
            str(k): int(v)
            for k, v in window_events["fvg_r_final_outcome"]
            .value_counts()
            .sort_index()
            .items()
        },
        "core_terminal_vs_research_final_crosstab": {
            str(idx): {str(col): int(val) for col, val in row.items()}
            for idx, row in crosstab.to_dict(orient="index").items()
        },
        "expired_event_reconciliation_table": expired_events.to_dict(orient="records"),
    }


def _summarize_overlap_fill_forensics(
    full_df: pd.DataFrame,
    *,
    active_members: pd.DataFrame,
    max_samples: int = 8,
) -> dict[str, object]:
    base = {
        "rows_with_multi_active_bull": 0,
        "rows_with_multi_active_bear": 0,
        "rows_with_selected_filling_and_other_same_side_active_bull": 0,
        "rows_with_selected_filling_and_other_same_side_active_bear": 0,
        "rows_with_non_selected_untouched_while_selected_filling_bull": 0,
        "rows_with_non_selected_untouched_while_selected_filling_bear": 0,
        "rows_with_selected_untouched_and_other_same_side_filling_bull": 0,
        "rows_with_selected_untouched_and_other_same_side_filling_bear": 0,
        "untouched_rep_with_nonzero_fill_count": 0,
        "untouched_rep_with_nonzero_touch_count_count": 0,
        "state_active_untouched_with_prior_touch_count_count": 0,
        "untouched_rep_fill_changed_without_entry_count": 0,
        "non_selected_untouched_rep_marked_as_filling_count": 0,
        "rep_with_nonzero_fill_before_first_entry_count": 0,
        "no_untouched_rep_with_nonzero_fill": True,
        "no_untouched_rep_with_nonzero_touch_count": True,
        "no_untouched_rep_fill_changed_without_entry": True,
        "no_non_selected_untouched_rep_marked_as_filling": True,
        "no_rep_with_nonzero_fill_before_first_entry": True,
        "state_active_untouched_with_prior_touch_count_note": (
            "core_state_1_can_persist_after_zero_penetration_wick_touches;"
            "_this_is_reported_separately_from_true_untouched_fill_invariants"
        ),
        "sample_forensic_cases": [],
        "selected_untouched_other_filling_sample_cases": [],
    }
    if active_members.empty:
        return base

    members = active_members.copy()
    members["row_idx"] = members["row_idx"].astype(int)
    members["detect_idx"] = members["detect_idx"].astype(int)
    row_idx = members["row_idx"].to_numpy(dtype=int)
    bar_low = pd.to_numeric(full_df.loc[row_idx, "low"], errors="coerce").to_numpy()
    bar_high = pd.to_numeric(full_df.loc[row_idx, "high"], errors="coerce").to_numpy()
    members["timestamp"] = pd.to_datetime(
        full_df.loc[row_idx, "timestamp"].to_numpy(), utc=True
    )
    members["price_entered_zone_on_row"] = (
        bar_high >= pd.to_numeric(members["low"], errors="coerce").to_numpy()
    ) & (bar_low <= pd.to_numeric(members["high"], errors="coerce").to_numpy())
    members["currently_untouched"] = (
        (members["state"] == 1)
        & (members["touch_count"] == 0)
        & np.isclose(
            pd.to_numeric(members["fill_pct"], errors="coerce").fillna(0.0).to_numpy(),
            0.0,
        )
    )

    sample_cases: list[dict[str, object]] = []
    selected_untouched_cases: list[dict[str, object]] = []
    totals = {
        "untouched_rep_with_nonzero_fill_count": 0,
        "untouched_rep_with_nonzero_touch_count_count": 0,
        "state_active_untouched_with_prior_touch_count_count": 0,
        "untouched_rep_fill_changed_without_entry_count": 0,
        "non_selected_untouched_rep_marked_as_filling_count": 0,
        "rep_with_nonzero_fill_before_first_entry_count": 0,
    }

    for side in ("bull", "bear"):
        side_members = (
            members[members["side"] == side]
            .copy()
            .sort_values(["active_id", "row_idx"])
            .reset_index(drop=True)
        )
        if side_members.empty:
            continue

        selected_rows = (
            side_members[side_members["selected"] == 1][
                [
                    "row_idx",
                    "timestamp",
                    "active_id",
                    "detect_idx",
                    "fill_pct",
                    "touch_count",
                    "state",
                ]
            ]
            .drop_duplicates(subset=["row_idx"])
            .rename(
                columns={
                    "active_id": "selected_active_id",
                    "detect_idx": "selected_detect_idx",
                    "fill_pct": "selected_fill_pct",
                    "touch_count": "selected_touch_count",
                    "state": "selected_state",
                }
            )
        )
        side_members = side_members.merge(
            selected_rows,
            on=["row_idx", "timestamp"],
            how="left",
        )
        active_counts = side_members.groupby("row_idx")["active_id"].size()
        side_members["row_active_count"] = side_members["row_idx"].map(active_counts)
        side_members["row_has_multi_active"] = side_members["row_active_count"] >= 2
        side_members["older_than_selected"] = side_members[
            "selected_detect_idx"
        ].notna() & (side_members["detect_idx"] < side_members["selected_detect_idx"])
        side_members["non_selected_untouched_while_selected_filling"] = (
            side_members["row_has_multi_active"]
            & (~side_members["selected"].astype(bool))
            & side_members["currently_untouched"]
            & (pd.to_numeric(side_members["selected_fill_pct"], errors="coerce") > 0)
        )

        dense_active_id = pd.to_numeric(
            full_df.loc[side_members["row_idx"], f"fvg_{side}_active_id"],
            errors="coerce",
        ).fillna(0)
        dense_active_id_array = dense_active_id.to_numpy()
        dense_fill_pct = pd.to_numeric(
            full_df.loc[side_members["row_idx"], f"fvg_{side}_active_fill_pct"],
            errors="coerce",
        ).fillna(0.0)
        dense_fill_pct_array = dense_fill_pct.to_numpy()

        side_members["prev_fill_pct"] = side_members.groupby("active_id")[
            "fill_pct"
        ].shift(1)
        side_members["prev_touch_count"] = side_members.groupby("active_id")[
            "touch_count"
        ].shift(1)
        side_members["prev_state"] = side_members.groupby("active_id")["state"].shift(1)
        prev_untouched = (
            side_members["prev_state"].eq(1)
            & side_members["prev_touch_count"].fillna(0).eq(0)
            & np.isclose(
                pd.to_numeric(side_members["prev_fill_pct"], errors="coerce")
                .fillna(0.0)
                .to_numpy(),
                0.0,
            )
        )
        fill_changed_without_entry = (
            prev_untouched
            & (~side_members["price_entered_zone_on_row"])
            & (
                (
                    pd.to_numeric(side_members["fill_pct"], errors="coerce")
                    > pd.to_numeric(
                        side_members["prev_fill_pct"], errors="coerce"
                    ).fillna(0.0)
                )
                | (
                    pd.to_numeric(side_members["touch_count"], errors="coerce")
                    > pd.to_numeric(
                        side_members["prev_touch_count"], errors="coerce"
                    ).fillna(0.0)
                )
                | side_members["state"].ne(1)
            )
        )

        first_fill_rows = (
            side_members[pd.to_numeric(side_members["fill_pct"], errors="coerce") > 0]
            .sort_values(["active_id", "row_idx"])
            .drop_duplicates(subset=["active_id"], keep="first")
        )

        untouched_state = side_members["state"] == 1
        true_untouched = side_members["currently_untouched"]
        totals["untouched_rep_with_nonzero_fill_count"] += int(
            (
                true_untouched
                & (pd.to_numeric(side_members["fill_pct"], errors="coerce") > 0)
            ).sum()
        )
        totals["untouched_rep_with_nonzero_touch_count_count"] += int(
            (
                true_untouched
                & (pd.to_numeric(side_members["touch_count"], errors="coerce") > 0)
            ).sum()
        )
        totals["state_active_untouched_with_prior_touch_count_count"] += int(
            (
                untouched_state
                & (pd.to_numeric(side_members["touch_count"], errors="coerce") > 0)
                & np.isclose(
                    pd.to_numeric(side_members["fill_pct"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(),
                    0.0,
                )
            ).sum()
        )
        totals["untouched_rep_fill_changed_without_entry_count"] += int(
            fill_changed_without_entry.sum()
        )
        totals["non_selected_untouched_rep_marked_as_filling_count"] += int(
            (
                (~side_members["selected"].astype(bool))
                & side_members["currently_untouched"]
                & (dense_active_id_array == side_members["active_id"].to_numpy())
                & (dense_fill_pct_array > 0)
            ).sum()
        )
        totals["rep_with_nonzero_fill_before_first_entry_count"] += int(
            (~first_fill_rows["price_entered_zone_on_row"]).sum()
        )

        selected_rows = selected_rows.join(
            active_counts.rename("active_member_count"), on="row_idx"
        )
        selected_rows = selected_rows.set_index("row_idx", drop=False)
        selected_rows["row_has_multi_active"] = (
            selected_rows["active_member_count"].fillna(0).astype(int) >= 2
        )
        selected_rows["selected_is_filling"] = (
            pd.to_numeric(selected_rows["selected_fill_pct"], errors="coerce") > 0
        )
        selected_rows["selected_is_untouched"] = (
            selected_rows["selected_state"].eq(1)
            & pd.to_numeric(selected_rows["selected_touch_count"], errors="coerce")
            .fillna(0)
            .eq(0)
            & np.isclose(
                pd.to_numeric(selected_rows["selected_fill_pct"], errors="coerce")
                .fillna(0.0)
                .to_numpy(),
                0.0,
            )
        )

        non_selected_untouched_rows = set(
            side_members.loc[
                (~side_members["selected"].astype(bool))
                & side_members["currently_untouched"],
                "row_idx",
            ].tolist()
        )
        non_selected_filling_rows = set(
            side_members.loc[
                (~side_members["selected"].astype(bool))
                & (pd.to_numeric(side_members["fill_pct"], errors="coerce") > 0),
                "row_idx",
            ].tolist()
        )

        base[f"rows_with_multi_active_{side}"] = int(
            (selected_rows["row_has_multi_active"]).sum()
        )
        base[f"rows_with_selected_filling_and_other_same_side_active_{side}"] = int(
            (
                selected_rows["row_has_multi_active"]
                & selected_rows["selected_is_filling"]
            ).sum()
        )
        base[f"rows_with_non_selected_untouched_while_selected_filling_{side}"] = int(
            (
                selected_rows["row_has_multi_active"]
                & selected_rows["selected_is_filling"]
                & selected_rows.index.isin(non_selected_untouched_rows)
            ).sum()
        )
        base[f"rows_with_selected_untouched_and_other_same_side_filling_{side}"] = int(
            (
                selected_rows["row_has_multi_active"]
                & selected_rows["selected_is_untouched"]
                & selected_rows.index.isin(non_selected_filling_rows)
            ).sum()
        )

        sample_row_idx = selected_rows.index[
            selected_rows["row_has_multi_active"]
            & selected_rows["selected_is_filling"]
            & selected_rows.index.isin(non_selected_untouched_rows)
        ].tolist()
        for row_idx_value in sample_row_idx[:max_samples]:
            scoped = side_members[side_members["row_idx"] == row_idx_value].copy()
            sample_cases.append(
                {
                    "row_idx": int(row_idx_value),
                    "timestamp": str(
                        pd.to_datetime(scoped["timestamp"].iloc[0], utc=True)
                    ),
                    "side": side,
                    "selected_active_id": int(
                        selected_rows.loc[row_idx_value, "selected_active_id"]
                    ),
                    "selected_fill_pct": float(
                        selected_rows.loc[row_idx_value, "selected_fill_pct"]
                    ),
                    "other_active_ids": [
                        int(v)
                        for v in scoped.loc[
                            ~scoped["selected"].astype(bool), "active_id"
                        ].tolist()
                    ],
                    "other_untouched_ids": [
                        int(v)
                        for v in scoped.loc[
                            (~scoped["selected"].astype(bool))
                            & scoped["currently_untouched"],
                            "active_id",
                        ].tolist()
                    ],
                }
            )

        selected_untouched_row_idx = selected_rows.index[
            selected_rows["row_has_multi_active"]
            & selected_rows["selected_is_untouched"]
            & selected_rows.index.isin(non_selected_filling_rows)
        ].tolist()
        for row_idx_value in selected_untouched_row_idx[:max_samples]:
            scoped = side_members[side_members["row_idx"] == row_idx_value].copy()
            selected_untouched_cases.append(
                {
                    "row_idx": int(row_idx_value),
                    "timestamp": str(
                        pd.to_datetime(scoped["timestamp"].iloc[0], utc=True)
                    ),
                    "side": side,
                    "selected_active_id": int(
                        selected_rows.loc[row_idx_value, "selected_active_id"]
                    ),
                    "selected_fill_pct": float(
                        selected_rows.loc[row_idx_value, "selected_fill_pct"]
                    ),
                    "other_filling_ids": [
                        int(v)
                        for v in scoped.loc[
                            (~scoped["selected"].astype(bool))
                            & (pd.to_numeric(scoped["fill_pct"], errors="coerce") > 0),
                            "active_id",
                        ].tolist()
                    ],
                    "other_fill_pcts": [
                        float(v)
                        for v in scoped.loc[
                            (~scoped["selected"].astype(bool))
                            & (pd.to_numeric(scoped["fill_pct"], errors="coerce") > 0),
                            "fill_pct",
                        ].tolist()
                    ],
                }
            )

    base.update(totals)
    base["no_untouched_rep_with_nonzero_fill"] = (
        base["untouched_rep_with_nonzero_fill_count"] == 0
    )
    base["no_untouched_rep_with_nonzero_touch_count"] = (
        base["untouched_rep_with_nonzero_touch_count_count"] == 0
    )
    base["no_untouched_rep_fill_changed_without_entry"] = (
        base["untouched_rep_fill_changed_without_entry_count"] == 0
    )
    base["no_non_selected_untouched_rep_marked_as_filling"] = (
        base["non_selected_untouched_rep_marked_as_filling_count"] == 0
    )
    base["no_rep_with_nonzero_fill_before_first_entry"] = (
        base["rep_with_nonzero_fill_before_first_entry_count"] == 0
    )
    sample_cases = sorted(
        sample_cases,
        key=lambda item: (item["timestamp"], item["side"], item["row_idx"]),
    )
    selected_untouched_cases = sorted(
        selected_untouched_cases,
        key=lambda item: (item["timestamp"], item["side"], item["row_idx"]),
    )
    base["sample_forensic_cases"] = sample_cases[:max_samples]
    base["selected_untouched_other_filling_sample_cases"] = selected_untouched_cases[
        :max_samples
    ]
    return base


def _summarize_fill_selection_reconciliation(
    full_df: pd.DataFrame,
    *,
    active_members: pd.DataFrame,
    max_samples: int = 8,
) -> dict[str, object] | None:
    required = {
        "fvg_fill_bull_rep_id",
        "fvg_fill_bull_rep_fill_pct",
        "fvg_fill_bull_rep_state",
        "fvg_fill_bull_rep_matches_structural_selected",
        "fvg_fill_bull_rep_selection_mode",
        "fvg_fill_bear_rep_id",
        "fvg_fill_bear_rep_fill_pct",
        "fvg_fill_bear_rep_state",
        "fvg_fill_bear_rep_matches_structural_selected",
        "fvg_fill_bear_rep_selection_mode",
    }
    if not required.issubset(full_df.columns):
        return None

    out: dict[str, object] = {}
    rep_snapshots = _build_member_snapshots_from_core(
        {"active_member_table": active_members},
        n_rows=len(full_df),
    )
    atr_col = next((col for col in full_df.columns if col.startswith("atr_")), None)
    atr_values = (
        pd.to_numeric(full_df[atr_col], errors="coerce").fillna(np.nan).to_numpy()
        if atr_col is not None
        else np.full(len(full_df), np.nan, dtype=float)
    )
    close_values = pd.to_numeric(full_df["close"], errors="coerce").to_numpy(
        dtype=float
    )
    for side in ("bull", "bear"):
        structural_active = pd.to_numeric(
            full_df[f"fvg_{side}_active"], errors="coerce"
        ).fillna(0)
        structural_selected_id = pd.to_numeric(
            full_df[f"fvg_{side}_active_id"], errors="coerce"
        ).fillna(0)
        fill_rep_id = pd.to_numeric(
            full_df[f"fvg_fill_{side}_rep_id"], errors="coerce"
        ).fillna(0)
        fill_rep_fill_pct = pd.to_numeric(
            full_df[f"fvg_fill_{side}_rep_fill_pct"], errors="coerce"
        )
        fill_rep_state = pd.to_numeric(
            full_df[f"fvg_fill_{side}_rep_state"], errors="coerce"
        )
        structural_touch_count = pd.to_numeric(
            full_df[f"fvg_{side}_active_touch_count"], errors="coerce"
        )
        matches = pd.to_numeric(
            full_df[f"fvg_fill_{side}_rep_matches_structural_selected"], errors="coerce"
        ).fillna(0)
        selection_mode = full_df[f"fvg_fill_{side}_rep_selection_mode"].fillna("none")
        structural_remaining_width = pd.Series(np.nan, index=full_df.index, dtype=float)
        structural_remaining_frac = pd.Series(np.nan, index=full_df.index, dtype=float)
        for row_idx, row in full_df.iterrows():
            geometry = _remaining_geometry_from_zone(
                side=side,
                low=row.get(f"fvg_{side}_active_low"),
                high=row.get(f"fvg_{side}_active_high"),
                fill_pct=row.get(f"fvg_{side}_active_fill_pct"),
            )
            structural_remaining_width.loc[row_idx] = geometry["remaining_width"]
            structural_remaining_frac.loc[row_idx] = geometry["remaining_frac"]
        fill_rep_remaining_width = pd.to_numeric(
            full_df[f"fvg_fill_{side}_remaining_width"], errors="coerce"
        )
        fill_rep_remaining_frac = pd.to_numeric(
            full_df[f"fvg_fill_{side}_remaining_frac"], errors="coerce"
        )

        touched_row_map: dict[int, list[int]] = {}
        fill_rep_in_active_pool_flags: list[bool] = []
        touched_candidate_rows_choose_fill_rep_flags: list[bool] = []
        fallback_rows_semantically_match_structural_flags: list[bool] = []
        none_mode_empty_flags: list[bool] = []
        fill_rep_matches_chosen_core_member_flags: list[bool] = []
        chosen_core_mismatch_rows: list[int] = []
        row_level_mismatch_audit: list[dict[str, object]] = []
        for row_idx, snapshot in enumerate(rep_snapshots):
            side_reps = [rep for rep in snapshot.values() if rep.side == side]
            side_rep_ids = [int(rep.active_id) for rep in side_reps]
            touched_candidate_ids = [
                int(rep.active_id) for rep in side_reps if _is_touched_candidate(rep)
            ]
            if touched_candidate_ids:
                touched_row_map[row_idx] = touched_candidate_ids

            current_fill_rep_id = int(fill_rep_id.iloc[row_idx])
            current_selection_mode = str(selection_mode.iloc[row_idx])
            current_structural_id = int(structural_selected_id.iloc[row_idx])
            chosen_core_rep, _ = _pick_fill_active_rep(
                current_idx=row_idx,
                close_value=float(close_values[row_idx]),
                atr_value=float(atr_values[row_idx]),
                structural_selected_id=current_structural_id,
                representations=side_reps,
            )
            chosen_core_rep_id = (
                int(chosen_core_rep.active_id) if chosen_core_rep is not None else 0
            )

            fill_rep_in_active_pool_flags.append(
                current_fill_rep_id == 0 or current_fill_rep_id in side_rep_ids
            )
            if touched_candidate_ids:
                touched_candidate_rows_choose_fill_rep_flags.append(
                    current_fill_rep_id in touched_candidate_ids
                )
            if current_selection_mode == "structural_fallback":
                fallback_rows_semantically_match_structural_flags.append(
                    current_structural_id == 0 or bool(matches.iloc[row_idx] == 1)
                )
            if current_selection_mode == "none":
                none_mode_empty_flags.append(current_fill_rep_id == 0)
            fill_rep_matches_chosen_core_member_flags.append(
                current_fill_rep_id == chosen_core_rep_id
            )
            if current_fill_rep_id != chosen_core_rep_id:
                chosen_core_mismatch_rows.append(row_idx)
            if (
                current_fill_rep_id > 0
                and current_structural_id > 0
                and current_fill_rep_id != current_structural_id
            ):
                row_level_mismatch_audit.append(
                    {
                        "row_idx": int(row_idx),
                        "timestamp": str(
                            pd.to_datetime(full_df.loc[row_idx, "timestamp"], utc=True)
                        ),
                        "structural_selected_id": current_structural_id,
                        "fill_rep_id": current_fill_rep_id,
                        "structural_selected_fill_pct": (
                            float(full_df.loc[row_idx, f"fvg_{side}_active_fill_pct"])
                            if np.isfinite(
                                full_df.loc[row_idx, f"fvg_{side}_active_fill_pct"]
                            )
                            else np.nan
                        ),
                        "fill_rep_fill_pct": (
                            float(fill_rep_fill_pct.iloc[row_idx])
                            if np.isfinite(fill_rep_fill_pct.iloc[row_idx])
                            else np.nan
                        ),
                        "structural_selected_remaining_width": (
                            float(structural_remaining_width.iloc[row_idx])
                            if np.isfinite(structural_remaining_width.iloc[row_idx])
                            else np.nan
                        ),
                        "fill_rep_remaining_width": (
                            float(fill_rep_remaining_width.iloc[row_idx])
                            if np.isfinite(fill_rep_remaining_width.iloc[row_idx])
                            else np.nan
                        ),
                        "structural_selected_remaining_frac": (
                            float(structural_remaining_frac.iloc[row_idx])
                            if np.isfinite(structural_remaining_frac.iloc[row_idx])
                            else np.nan
                        ),
                        "fill_rep_remaining_frac": (
                            float(fill_rep_remaining_frac.iloc[row_idx])
                            if np.isfinite(fill_rep_remaining_frac.iloc[row_idx])
                            else np.nan
                        ),
                    }
                )

        active_mask = structural_active == 1
        fill_mask = fill_rep_id > 0
        mismatch_mask = fill_mask & (matches == 0)
        structural_untouched_mask = structural_touch_count.fillna(0).eq(
            0
        ) & pd.to_numeric(
            full_df[f"fvg_{side}_active_fill_pct"], errors="coerce"
        ).fillna(
            0.0
        ).eq(
            0.0
        )
        materially_different_geometry_mask = (
            (
                structural_remaining_width.fillna(np.nan)
                - fill_rep_remaining_width.fillna(np.nan)
            ).abs()
            > 1e-9
        ) | (
            (
                structural_remaining_frac.fillna(np.nan)
                - fill_rep_remaining_frac.fillna(np.nan)
            ).abs()
            > 1e-9
        )
        visual_confusion_mask = mismatch_mask & (
            structural_untouched_mask | materially_different_geometry_mask
        )
        sample_rows = full_df.loc[mismatch_mask].head(max_samples)
        samples = []
        for row in sample_rows.itertuples():
            samples.append(
                {
                    "row_idx": int(row.Index),
                    "timestamp": str(pd.to_datetime(row.timestamp, utc=True)),
                    "structural_selected_id": int(
                        getattr(row, f"fvg_{side}_active_id")
                    ),
                    "structural_selected_fill_pct": (
                        float(getattr(row, f"fvg_{side}_active_fill_pct"))
                        if np.isfinite(getattr(row, f"fvg_{side}_active_fill_pct"))
                        else np.nan
                    ),
                    "fill_rep_id": int(getattr(row, f"fvg_fill_{side}_rep_id")),
                    "fill_rep_fill_pct": (
                        float(getattr(row, f"fvg_fill_{side}_rep_fill_pct"))
                        if np.isfinite(getattr(row, f"fvg_fill_{side}_rep_fill_pct"))
                        else np.nan
                    ),
                    "structural_selected_remaining_width": (
                        float(structural_remaining_width.loc[int(row.Index)])
                        if np.isfinite(structural_remaining_width.loc[int(row.Index)])
                        else np.nan
                    ),
                    "fill_rep_remaining_width": (
                        float(fill_rep_remaining_width.loc[int(row.Index)])
                        if np.isfinite(fill_rep_remaining_width.loc[int(row.Index)])
                        else np.nan
                    ),
                    "structural_selected_remaining_frac": (
                        float(structural_remaining_frac.loc[int(row.Index)])
                        if np.isfinite(structural_remaining_frac.loc[int(row.Index)])
                        else np.nan
                    ),
                    "fill_rep_remaining_frac": (
                        float(fill_rep_remaining_frac.loc[int(row.Index)])
                        if np.isfinite(fill_rep_remaining_frac.loc[int(row.Index)])
                        else np.nan
                    ),
                    "touched_candidate_ids": touched_row_map.get(int(row.Index), []),
                }
            )

        out[side] = {
            "rows_with_active_rep": int(active_mask.sum()),
            "rows_with_fill_rep": int(fill_mask.sum()),
            "rows_with_fill_rep_matches_structural_selected": int(
                (fill_mask & (matches == 1)).sum()
            ),
            "rows_with_fill_rep_differs_from_structural_selected": int(
                mismatch_mask.sum()
            ),
            "rows_where_fill_rep_differs_from_structural_selected": int(
                mismatch_mask.sum()
            ),
            "rows_where_visual_confusion_pattern_occurs": int(
                visual_confusion_mask.sum()
            ),
            "rows_with_fill_rep_not_equal_to_chosen_core_member": int(
                len(chosen_core_mismatch_rows)
            ),
            "rows_with_touched_candidates_present": int(
                selection_mode.eq("touched_pool").sum()
            ),
            "rows_using_structural_fallback": int(
                selection_mode.eq("structural_fallback").sum()
            ),
            "rows_with_fill_rep_untouched": int(
                (
                    fill_mask & fill_rep_state.eq(1) & fill_rep_fill_pct.fillna(0).eq(0)
                ).sum()
            ),
            "rows_with_fill_rep_touched": int((fill_mask & fill_rep_state.eq(2)).sum()),
            "rows_with_fill_rep_partial_fill": int(
                (fill_mask & fill_rep_state.eq(3)).sum()
            ),
            "fill_rep_matches_structural_selected_rate": (
                float((fill_mask & (matches == 1)).sum() / max(int(fill_mask.sum()), 1))
                if fill_mask.any()
                else np.nan
            ),
            "fill_rep_differs_from_structural_selected_rate": (
                float(mismatch_mask.sum() / max(int(fill_mask.sum()), 1))
                if fill_mask.any()
                else np.nan
            ),
            "structural_fallback_rate_when_active": (
                float(
                    selection_mode.eq("structural_fallback").sum()
                    / max(int(active_mask.sum()), 1)
                )
                if active_mask.any()
                else np.nan
            ),
            "invariants": {
                "fill_rep_always_in_active_pool": bool(
                    all(fill_rep_in_active_pool_flags)
                ),
                "touched_candidate_rows_choose_touched_pool_rep": bool(
                    all(touched_candidate_rows_choose_fill_rep_flags)
                ),
                "structural_fallback_rows_semantically_match_structural_selected": bool(
                    all(fallback_rows_semantically_match_structural_flags)
                ),
                "none_mode_rows_have_zero_fill_rep_id": bool(
                    all(none_mode_empty_flags)
                ),
                "fill_rep_matches_chosen_core_member": bool(
                    all(fill_rep_matches_chosen_core_member_flags)
                ),
            },
            "mismatch_sample_rows": samples,
            "row_level_mismatch_audit": row_level_mismatch_audit,
        }
    return out


def _summarize_core_fill_ownership_reconciliation(
    full_df: pd.DataFrame,
    *,
    active_members: pd.DataFrame,
    max_samples: int = 8,
) -> dict[str, object] | None:
    required = {
        "row_idx",
        "side",
        "active_id",
        "component_id",
        "component_low",
        "component_high",
        "component_entered_this_row",
        "component_owned_this_row",
        "touch_delta",
        "fill_delta",
        "entered_this_row",
    }
    if active_members.empty or not required.issubset(active_members.columns):
        return None

    out: dict[str, object] = {}
    rep_snapshots = _build_member_snapshots_from_core(
        {"active_member_table": active_members},
        n_rows=len(full_df),
    )
    atr_col = next((col for col in full_df.columns if col.startswith("atr_")), None)
    atr_values = (
        pd.to_numeric(full_df[atr_col], errors="coerce").fillna(np.nan).to_numpy()
        if atr_col is not None
        else np.full(len(full_df), np.nan, dtype=float)
    )
    close_values = pd.to_numeric(full_df["close"], errors="coerce").to_numpy(
        dtype=float
    )

    for side in ("bull", "bear"):
        side_members = active_members[active_members["side"] == side].copy()
        if side_members.empty:
            out[side] = {
                "rows_with_multi_component_active": 0,
                "rows_with_owned_component": 0,
                "rows_with_disjoint_multi_component_entry_attempt": 0,
                "rows_with_disjoint_multi_component_positive_fill_delta": 0,
                "rows_with_disjoint_multi_component_positive_touch_delta": 0,
                "rows_with_non_owned_component_positive_fill_delta": 0,
                "rows_with_non_owned_component_positive_touch_delta": 0,
                "rows_with_fill_rep_not_equal_to_chosen_core_member": 0,
                "invariants": {
                    "no_disjoint_multi_component_fill_delta": True,
                    "no_disjoint_multi_component_touch_delta": True,
                    "no_non_owned_component_fill_delta": True,
                    "no_non_owned_component_touch_delta": True,
                    "fill_rep_matches_chosen_core_member": True,
                    "no_fill_delta_without_entry": True,
                    "no_touch_delta_without_entry": True,
                },
                "sample_violations": [],
            }
            continue

        grouped = side_members.groupby("row_idx", sort=True)
        multi_component_rows = 0
        owned_component_rows = 0
        disjoint_entry_rows = 0
        disjoint_fill_rows = 0
        disjoint_touch_rows = 0
        non_owned_fill_rows = 0
        non_owned_touch_rows = 0
        fill_without_entry_rows = 0
        touch_without_entry_rows = 0
        fill_rep_mismatch_rows = 0
        sample_violations: list[dict[str, object]] = []

        for row_idx, scoped in grouped:
            component_count = int(scoped["component_id"].nunique())
            if component_count >= 2:
                multi_component_rows += 1

            component_view = (
                scoped.groupby("component_id", sort=True)
                .agg(
                    component_low=("component_low", "first"),
                    component_high=("component_high", "first"),
                    entered=("component_entered_this_row", "max"),
                    owned=("component_owned_this_row", "max"),
                    fill_delta_sum=("fill_delta", "sum"),
                    touch_delta_sum=("touch_delta", "sum"),
                )
                .reset_index()
            )
            owned_components = component_view[
                pd.to_numeric(component_view["owned"], errors="coerce").fillna(0) > 0
            ]
            if not owned_components.empty:
                owned_component_rows += 1
            entered_components = component_view[
                pd.to_numeric(component_view["entered"], errors="coerce").fillna(0) > 0
            ]
            if component_count >= 2 and len(entered_components) >= 2:
                disjoint_entry_rows += 1
            positive_fill_components = component_view[
                pd.to_numeric(component_view["fill_delta_sum"], errors="coerce") > 0
            ]
            positive_touch_components = component_view[
                pd.to_numeric(component_view["touch_delta_sum"], errors="coerce") > 0
            ]
            if component_count >= 2 and len(positive_fill_components) >= 2:
                disjoint_fill_rows += 1
            if component_count >= 2 and len(positive_touch_components) >= 2:
                disjoint_touch_rows += 1

            non_owned_fill = scoped[
                (
                    pd.to_numeric(scoped["component_owned_this_row"], errors="coerce")
                    == 0
                )
                & (pd.to_numeric(scoped["fill_delta"], errors="coerce") > 0)
            ]
            non_owned_touch = scoped[
                (
                    pd.to_numeric(scoped["component_owned_this_row"], errors="coerce")
                    == 0
                )
                & (pd.to_numeric(scoped["touch_delta"], errors="coerce") > 0)
            ]
            if not non_owned_fill.empty:
                non_owned_fill_rows += 1
            if not non_owned_touch.empty:
                non_owned_touch_rows += 1

            no_entry_fill = scoped[
                (pd.to_numeric(scoped["entered_this_row"], errors="coerce") == 0)
                & (pd.to_numeric(scoped["fill_delta"], errors="coerce") > 0)
            ]
            no_entry_touch = scoped[
                (pd.to_numeric(scoped["entered_this_row"], errors="coerce") == 0)
                & (pd.to_numeric(scoped["touch_delta"], errors="coerce") > 0)
            ]
            if not no_entry_fill.empty:
                fill_without_entry_rows += 1
            if not no_entry_touch.empty:
                touch_without_entry_rows += 1

            structural_selected_id = int(full_df.loc[row_idx, f"fvg_{side}_active_id"])
            chosen_core_rep, _ = _pick_fill_active_rep(
                current_idx=int(row_idx),
                close_value=float(close_values[int(row_idx)]),
                atr_value=float(atr_values[int(row_idx)]),
                structural_selected_id=structural_selected_id,
                representations=[
                    rep
                    for rep in rep_snapshots[int(row_idx)].values()
                    if rep.side == side
                ],
            )
            chosen_core_rep_id = (
                int(chosen_core_rep.active_id) if chosen_core_rep is not None else 0
            )
            current_fill_rep_id = int(full_df.loc[row_idx, f"fvg_fill_{side}_rep_id"])
            if current_fill_rep_id != chosen_core_rep_id:
                fill_rep_mismatch_rows += 1

            if len(sample_violations) >= max_samples:
                continue
            if (
                (
                    component_count >= 2
                    and (
                        len(positive_fill_components) >= 2
                        or len(positive_touch_components) >= 2
                    )
                )
                or not non_owned_fill.empty
                or not non_owned_touch.empty
                or not no_entry_fill.empty
                or not no_entry_touch.empty
            ):
                owned_component = (
                    owned_components.iloc[0] if not owned_components.empty else None
                )
                violating = scoped[
                    (pd.to_numeric(scoped["fill_delta"], errors="coerce") > 0)
                    | (pd.to_numeric(scoped["touch_delta"], errors="coerce") > 0)
                ]
                sample_violations.append(
                    {
                        "row_idx": int(row_idx),
                        "timestamp": str(
                            pd.to_datetime(full_df.loc[row_idx, "timestamp"], utc=True)
                        ),
                        "side": side,
                        "owned_component_id": (
                            int(owned_component["component_id"])
                            if owned_component is not None
                            else 0
                        ),
                        "owned_component_bounds": (
                            [
                                float(owned_component["component_low"]),
                                float(owned_component["component_high"]),
                            ]
                            if owned_component is not None
                            else []
                        ),
                        "violating_active_ids": [
                            int(v) for v in violating["active_id"].tolist()
                        ],
                        "violating_fill_deltas": [
                            float(v) for v in violating["fill_delta"].tolist()
                        ],
                        "violating_touch_deltas": [
                            int(v) for v in violating["touch_delta"].tolist()
                        ],
                    }
                )

        out[side] = {
            "rows_with_multi_component_active": int(multi_component_rows),
            "rows_with_owned_component": int(owned_component_rows),
            "rows_with_disjoint_multi_component_entry_attempt": int(
                disjoint_entry_rows
            ),
            "rows_with_disjoint_multi_component_positive_fill_delta": int(
                disjoint_fill_rows
            ),
            "rows_with_disjoint_multi_component_positive_touch_delta": int(
                disjoint_touch_rows
            ),
            "rows_with_non_owned_component_positive_fill_delta": int(
                non_owned_fill_rows
            ),
            "rows_with_non_owned_component_positive_touch_delta": int(
                non_owned_touch_rows
            ),
            "rows_with_fill_rep_not_equal_to_chosen_core_member": int(
                fill_rep_mismatch_rows
            ),
            "invariants": {
                "no_disjoint_multi_component_fill_delta": disjoint_fill_rows == 0,
                "no_disjoint_multi_component_touch_delta": disjoint_touch_rows == 0,
                "no_non_owned_component_fill_delta": non_owned_fill_rows == 0,
                "no_non_owned_component_touch_delta": non_owned_touch_rows == 0,
                "fill_rep_matches_chosen_core_member": fill_rep_mismatch_rows == 0,
                "no_fill_delta_without_entry": fill_without_entry_rows == 0,
                "no_touch_delta_without_entry": touch_without_entry_rows == 0,
            },
            "sample_violations": sample_violations,
        }
    return out


def _summarize_live_vs_historical_fill_split(
    active_members: pd.DataFrame,
    *,
    max_samples: int = 10,
) -> dict[str, object] | None:
    required = {
        "row_idx",
        "timestamp",
        "side",
        "active_id",
        "component_id",
        "fill_pct",
        "historical_fill_pct",
    }
    if active_members.empty or not required.issubset(active_members.columns):
        return None

    out: dict[str, object] = {}
    for side in ("bull", "bear"):
        side_members = active_members[active_members["side"] == side].copy()
        if side_members.empty:
            out[side] = {
                "rows_with_live_fill_on_multiple_components": 0,
                "rows_with_historical_fill_on_multiple_components": 0,
                "rows_with_historical_only_coexistence": 0,
                "invariants": {
                    "no_live_fill_on_multiple_components": True,
                },
                "sample_historical_only_rows": [],
            }
            continue

        sample_rows: list[dict[str, object]] = []
        live_multi = 0
        historical_multi = 0
        historical_only = 0

        for row_idx, scoped in side_members.groupby("row_idx", sort=True):
            component_view = (
                scoped.groupby("component_id", sort=True)
                .agg(
                    live_fill_max=("fill_pct", "max"),
                    historical_fill_max=("historical_fill_pct", "max"),
                )
                .reset_index()
            )
            live_components = component_view[
                pd.to_numeric(component_view["live_fill_max"], errors="coerce") > 0
            ]
            historical_components = component_view[
                pd.to_numeric(component_view["historical_fill_max"], errors="coerce")
                > 0
            ]
            live_multi_flag = len(live_components) >= 2
            historical_multi_flag = len(historical_components) >= 2
            historical_only_flag = historical_multi_flag and not live_multi_flag
            live_multi += int(live_multi_flag)
            historical_multi += int(historical_multi_flag)
            historical_only += int(historical_only_flag)

            if historical_only_flag and len(sample_rows) < max_samples:
                sample_rows.append(
                    {
                        "row_idx": int(row_idx),
                        "timestamp": str(
                            pd.to_datetime(scoped["timestamp"].iloc[0], utc=True)
                        ),
                        "side": side,
                        "live_fill_active_ids": [
                            int(v)
                            for v in scoped.loc[
                                pd.to_numeric(scoped["fill_pct"], errors="coerce") > 0,
                                "active_id",
                            ].tolist()
                        ],
                        "historical_fill_active_ids": [
                            int(v)
                            for v in scoped.loc[
                                pd.to_numeric(
                                    scoped["historical_fill_pct"], errors="coerce"
                                )
                                > 0,
                                "active_id",
                            ].tolist()
                        ],
                    }
                )

        out[side] = {
            "rows_with_live_fill_on_multiple_components": int(live_multi),
            "rows_with_historical_fill_on_multiple_components": int(historical_multi),
            "rows_with_historical_only_coexistence": int(historical_only),
            "invariants": {
                "no_live_fill_on_multiple_components": bool(live_multi == 0),
            },
            "sample_historical_only_rows": sample_rows,
        }
    return out


def _summarize_fill(
    df: pd.DataFrame,
    *,
    research_table: pd.DataFrame | None = None,
) -> dict[str, object] | None:
    required = {
        "fvg_fill_bull_remaining_width",
        "fvg_fill_bear_remaining_width",
        "fvg_fill_bull_fill_pct_delta_3",
        "fvg_fill_bear_fill_pct_delta_3",
        "fvg_fill_bull_fill_type",
        "fvg_fill_bear_fill_type",
    }
    if not required.issubset(df.columns):
        return None

    def _side(side: str) -> dict[str, object]:
        remaining_width = pd.to_numeric(
            df[f"fvg_fill_{side}_remaining_width"], errors="coerce"
        )
        remaining_frac = pd.to_numeric(
            df[f"fvg_fill_{side}_remaining_frac"], errors="coerce"
        )
        fill_delta_3 = pd.to_numeric(
            df[f"fvg_fill_{side}_fill_pct_delta_3"], errors="coerce"
        )
        fill_type = df[f"fvg_fill_{side}_fill_type"]
        bounce = pd.to_numeric(
            df[f"fvg_fill_{side}_bounce_strength_atr"], errors="coerce"
        )
        bars_since_touch = pd.to_numeric(
            df[f"fvg_fill_{side}_bars_since_touch"], errors="coerce"
        )
        bars_since_first_touch = pd.to_numeric(
            df[f"fvg_fill_{side}_bars_since_first_touch"], errors="coerce"
        )
        touch_rejection = pd.to_numeric(
            df[f"fvg_fill_{side}_touch_rejection_flag"], errors="coerce"
        )
        # Use fill_rep's own fill_pct (not structural selected) so invariants are
        # self-consistent: remaining_* columns are computed from fill_rep geometry.
        fill_pct = pd.to_numeric(
            df.get(f"fvg_fill_{side}_rep_fill_pct", pd.Series(dtype=float)),
            errors="coerce",
        )
        active_count = pd.to_numeric(df[f"fvg_{side}_active_count"], errors="coerce")
        untouched_count = pd.to_numeric(
            df[f"fvg_fill_{side}_untouched_count"], errors="coerce"
        )

        touch_rows = df[bars_since_touch == 0]
        remaining_full_fill = pd.Series(dtype=float)
        if research_table is not None and not research_table.empty:
            full_fill_events = research_table[
                (research_table["fvg_side"] == side)
                & (research_table["fvg_terminal_label"] == "full_fill")
                & research_table["fvg_full_fill_idx"].notna()
            ]
            prev_rows = (full_fill_events["fvg_full_fill_idx"].astype(int) - 1).clip(
                lower=0
            )
            prev_rows = prev_rows[prev_rows.isin(df.index)]
            if not prev_rows.empty:
                remaining_full_fill = pd.to_numeric(
                    df.loc[prev_rows, f"fvg_fill_{side}_remaining_width_atr"],
                    errors="coerce",
                )

        remaining_alignment = (remaining_frac - (1.0 - fill_pct)).abs()
        return {
            "remaining_width_at_touch_atr": _continuous_stats(
                touch_rows[f"fvg_fill_{side}_remaining_width_atr"]
            ),
            "remaining_width_before_full_fill_atr": _continuous_stats(
                remaining_full_fill
            ),
            "remaining_frac": _continuous_stats(remaining_frac),
            "fill_pct_delta_3": _continuous_stats(fill_delta_3),
            "fill_type_distribution": _value_counts(fill_type),
            "bounce_strength_atr": _continuous_stats(bounce),
            "touch_rejection_rate": (
                float(touch_rejection.dropna().mean())
                if touch_rejection.notna().any()
                else np.nan
            ),
            "bars_since_touch": _continuous_stats(bars_since_touch),
            "bars_since_first_touch": _continuous_stats(bars_since_first_touch),
            "mean_active_zone_count_when_active": (
                float(active_count[active_count > 0].mean())
                if (active_count > 0).any()
                else np.nan
            ),
            "touched_vs_untouched_pool_rows": {
                "touched_rows": int((active_count > untouched_count).sum()),
                "fully_untouched_rows": int(
                    ((active_count > 0) & (active_count == untouched_count)).sum()
                ),
            },
            "invariant_audit": {
                "remaining_width_non_negative": bool(
                    (remaining_width.dropna() >= 0).all()
                ),
                # remaining_frac = remaining_width / fill_rep_zone_width, so
                # remaining_frac <= 1.0 is the self-consistent check without
                # needing the structural active_width (which may differ from
                # fill_rep zone width when fill_rep ≠ structural selected).
                "remaining_width_lte_original": bool(
                    (remaining_frac.dropna() <= 1.0 + 1e-6).all()
                ),
                "fill_pct_delta_3_non_negative": bool(
                    (fill_delta_3.dropna() >= 0).all()
                ),
                "bounce_non_negative": bool((bounce.dropna() >= 0).all()),
                "bars_since_touch_non_negative": bool(
                    (bars_since_touch.dropna() >= 0).all()
                ),
                "bars_since_first_touch_non_negative": bool(
                    (bars_since_first_touch.dropna() >= 0).all()
                ),
                "untouched_count_lte_active_count": bool(
                    (
                        untouched_count.fillna(0).to_numpy()
                        <= active_count.fillna(0).to_numpy()
                    ).all()
                ),
                "remaining_frac_matches_one_minus_fill_pct_max_abs_err": (
                    float(remaining_alignment.dropna().max())
                    if remaining_alignment.notna().any()
                    else np.nan
                ),
            },
        }

    return {
        "bull": _side("bull"),
        "bear": _side("bear"),
    }


def _summarize_ifvg(
    df: pd.DataFrame,
    *,
    debug_tables: dict[str, pd.DataFrame],
) -> dict[str, object] | None:
    required = {
        "ifvg_bull_detect_flag",
        "ifvg_bear_detect_flag",
        "ifvg_bull_active",
        "ifvg_bear_active",
    }
    if not required.issubset(df.columns):
        return None

    candidates = _build_invalidated_source_candidates(df, debug_tables=debug_tables)
    canonical = (
        candidates[candidates["accepted"]].copy()
        if not candidates.empty
        else pd.DataFrame()
    )
    invalidated_sources = debug_tables["event_table"][
        debug_tables["event_table"]["terminal_state"] == STATE_INVALIDATED
    ].copy()

    bull_detect = (
        pd.to_numeric(df["ifvg_bull_detect_flag"], errors="coerce").fillna(0) == 1
    )
    bear_detect = (
        pd.to_numeric(df["ifvg_bear_detect_flag"], errors="coerce").fillna(0) == 1
    )
    bull_active = pd.to_numeric(df["ifvg_bull_active"], errors="coerce").fillna(0) == 1
    bear_active = pd.to_numeric(df["ifvg_bear_active"], errors="coerce").fillna(0) == 1
    canonical_event_rows: list[dict[str, object]] = []
    active_pool_rows: list[dict[str, object]] = []
    terminal_leak_counts = {
        "fully_reclaimed": 0,
        "expired": 0,
        "invalid_terminal": 0,
    }
    if not canonical.empty:
        events: list[_IfvgEvent] = []
        for row in canonical.itertuples(index=False):
            events.append(
                _IfvgEvent(
                    event_id=int(row.ifvg_event_id),
                    side=str(row.ifvg_side),
                    direction=int(row.direction),
                    detect_idx=int(row.detect_idx),
                    detect_ts=pd.to_datetime(row.detect_ts, utc=True),
                    low=float(row.low),
                    high=float(row.high),
                    mid=float(row.mid),
                    width=float(row.width),
                    width_atr=float(row.width_atr),
                    source_fvg_event_id=int(row.source_fvg_event_id),
                    source_fvg_detect_idx=int(row.source_fvg_detect_idx),
                    source_fvg_direction=int(row.source_fvg_direction),
                    source_age_bars=int(row.source_age_bars),
                    source_width_atr=float(row.source_width_atr),
                    source_fill_pct_at_inversion=float(
                        row.source_fill_pct_at_inversion
                    ),
                    inversion_body_atr=float(row.inversion_body_atr),
                    inversion_range_atr=float(row.inversion_range_atr),
                    inversion_body_frac=float(row.inversion_body_frac),
                    inversion_close_location=float(row.inversion_close_location),
                    inversion_conviction_location=float(
                        row.inversion_conviction_location
                    ),
                    cross_distance_atr=float(row.cross_distance_atr),
                    inversion_score=float(row.inversion_score),
                    active_since_idx=int(row.detect_idx),
                )
            )

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        timestamps = pd.to_datetime(df["timestamp"], utc=True)

        for row_idx in range(len(df)):
            for event in events:
                row_metrics = _advance_ifvg_event(
                    event,
                    row_idx=row_idx,
                    close_value=float(close[row_idx]),
                    high_value=float(high[row_idx]),
                    low_value=float(low[row_idx]),
                    max_ifvg_age_bars=MAX_IFVG_ACTIVE_AGE_BARS,
                )
                if bool(row_metrics["active_after_row"]):
                    if event.current_state == IFVG_STATE_FULLY_RECLAIMED:
                        terminal_leak_counts["fully_reclaimed"] += 1
                    if event.current_state == IFVG_STATE_EXPIRED:
                        terminal_leak_counts["expired"] += 1
                    if event.current_state not in {
                        1,
                        2,
                        3,
                        4,
                    }:
                        terminal_leak_counts["invalid_terminal"] += 1
                    active_pool_rows.append(
                        {
                            "row_idx": int(row_idx),
                            "timestamp": timestamps.iloc[row_idx],
                            "side": event.side,
                            "ifvg_event_id": int(event.event_id),
                            "detect_idx": int(event.detect_idx),
                            "active_age": int(row_idx - event.detect_idx),
                            "state": int(row_metrics["state_for_row"]),
                            "source_fvg_direction": int(event.source_fvg_direction),
                            "source_age_bars": int(event.source_age_bars),
                            "source_fill_pct_at_inversion": float(
                                event.source_fill_pct_at_inversion
                            ),
                            "carried_in": int(event.detect_idx < row_idx),
                        }
                    )

        for event in events:
            canonical_event_rows.append(
                {
                    "ifvg_event_id": int(event.event_id),
                    "side": event.side,
                    "detect_idx": int(event.detect_idx),
                    "source_fvg_event_id": int(event.source_fvg_event_id),
                    "source_fvg_detect_idx": int(event.source_fvg_detect_idx),
                    "source_fvg_direction": int(event.source_fvg_direction),
                    "source_age_bars": int(event.source_age_bars),
                    "source_fill_pct_at_inversion": float(
                        event.source_fill_pct_at_inversion
                    ),
                    "inversion_score": float(event.inversion_score),
                    "first_retest_idx": (
                        float(event.first_retest_idx)
                        if event.first_retest_idx is not None
                        else np.nan
                    ),
                    "fully_reclaimed_idx": (
                        float(event.fully_reclaimed_idx)
                        if event.fully_reclaimed_idx is not None
                        else np.nan
                    ),
                    "expiry_idx": (
                        float(event.expiry_idx)
                        if event.expiry_idx is not None
                        else np.nan
                    ),
                    "terminal_state": int(event.terminal_state),
                }
            )

    canonical_events = pd.DataFrame(canonical_event_rows)
    active_pool = pd.DataFrame(active_pool_rows)

    exported_detect_rows = pd.concat(
        [
            df.loc[
                bull_detect,
                [
                    "ifvg_bull_event_id",
                    "ifvg_bull_detect_idx",
                    "ifvg_bull_source_fvg_event_id",
                ],
            ].rename(columns=lambda col: col.replace("ifvg_bull_", "")),
            df.loc[
                bear_detect,
                [
                    "ifvg_bear_event_id",
                    "ifvg_bear_detect_idx",
                    "ifvg_bear_source_fvg_event_id",
                ],
            ].rename(columns=lambda col: col.replace("ifvg_bear_", "")),
        ],
        axis=0,
        ignore_index=True,
    )
    exported_detect_event_ids = (
        set(exported_detect_rows["event_id"].astype(int).tolist())
        if not exported_detect_rows.empty
        else set()
    )
    dense_excluded = (
        canonical_events[
            ~canonical_events["ifvg_event_id"]
            .astype(int)
            .isin(exported_detect_event_ids)
        ].copy()
        if not canonical_events.empty
        else pd.DataFrame()
    )
    if not dense_excluded.empty:
        dense_group_sizes = (
            canonical.groupby(["detect_idx", "ifvg_side"], sort=False)
            .size()
            .rename("same_side_same_row_canonical_count")
            .reset_index()
            .rename(columns={"ifvg_side": "side"})
        )
        dense_excluded = dense_excluded.merge(
            dense_group_sizes, on=["detect_idx", "side"], how="left"
        )
        dense_excluded["exclusion_reason"] = (
            "dense_detect_surface_exports_only_one_ifvg_per_side_per_row"
        )

    invalidated_source_ids = set(invalidated_sources["event_id"].astype(int).tolist())
    orphan_count = 0
    detect_idx_match = True
    if not canonical.empty:
        invalidation_lookup = {
            int(row.event_id): int(float(row.invalidation_idx))
            for row in invalidated_sources.itertuples(index=False)
            if np.isfinite(row.invalidation_idx)
        }
        canonical_source_ids = canonical["source_fvg_event_id"].astype(int)
        orphan_count = int((~canonical_source_ids.isin(invalidated_source_ids)).sum())
        detect_idx_match = bool(
            (
                canonical["detect_idx"].astype(int)
                == canonical_source_ids.map(invalidation_lookup).astype(int)
            ).all()
        )

    detect_rows = canonical_events.copy()
    rejection_counts = (
        candidates.loc[~candidates["accepted"], "rejection_reason"]
        .value_counts()
        .to_dict()
        if not candidates.empty
        else {}
    )

    def _active_age_stats(side: str) -> dict[str, float | int]:
        if active_pool.empty:
            return {
                "max": 0,
                "mean": np.nan,
                "median": np.nan,
                "p90": np.nan,
            }
        scoped = pd.to_numeric(
            active_pool.loc[active_pool["side"] == side, "active_age"], errors="coerce"
        ).dropna()
        if scoped.empty:
            return {
                "max": 0,
                "mean": np.nan,
                "median": np.nan,
                "p90": np.nan,
            }
        return {
            "max": int(scoped.max()),
            "mean": float(scoped.mean()),
            "median": float(scoped.median()),
            "p90": float(scoped.quantile(0.9)),
        }

    def _window_active_profile(side: str) -> dict[str, int]:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        window_mask = (ts >= FORENSIC_WINDOW_START) & (ts <= FORENSIC_WINDOW_END)
        window_idx = df.index[window_mask]
        if len(window_idx) == 0:
            return {
                "active_rows": 0,
                "total_rows": 0,
                "rows_with_zero_active_count": 0,
                "active_count_on_first_row": 0,
                "active_count_on_last_row": 0,
                "carried_in_active_count_on_first_row": 0,
                "carried_in_active_count_on_last_row": 0,
            }
        if active_pool.empty:
            counts = pd.Series(0, index=window_idx, dtype=int)
            carried = pd.Series(0, index=window_idx, dtype=int)
        else:
            scoped = active_pool[
                (active_pool["side"] == side)
                & active_pool["row_idx"].between(
                    int(window_idx.min()), int(window_idx.max())
                )
            ].copy()
            counts = (
                scoped.groupby("row_idx")["ifvg_event_id"]
                .size()
                .reindex(window_idx, fill_value=0)
            )
            carried = (
                scoped.groupby("row_idx")["carried_in"]
                .sum()
                .reindex(window_idx, fill_value=0)
            )
        return {
            "active_rows": int((counts > 0).sum()),
            "total_rows": int(len(window_idx)),
            "rows_with_zero_active_count": int((counts == 0).sum()),
            "active_count_on_first_row": int(counts.iloc[0]),
            "active_count_on_last_row": int(counts.iloc[-1]),
            "carried_in_active_count_on_first_row": int(carried.iloc[0]),
            "carried_in_active_count_on_last_row": int(carried.iloc[-1]),
        }

    def _bucketed_reclaim_rate(
        column: str, bins: list[float], labels: list[str]
    ) -> dict[str, object]:
        if canonical_events.empty:
            return {}
        scoped = canonical_events.copy()
        values = pd.to_numeric(scoped[column], errors="coerce")
        bucket = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
        scoped["bucket"] = bucket.astype("object").fillna("NaN")
        out: dict[str, object] = {}
        for bucket_name, rows in scoped.groupby("bucket", dropna=False):
            out[str(bucket_name)] = {
                "count": int(len(rows)),
                "reclaim_rate": float(
                    (rows["terminal_state"] == IFVG_STATE_FULLY_RECLAIMED).mean()
                ),
            }
        return out

    def _regime_proxy() -> pd.Series:
        delta = pd.to_numeric(df["close"], errors="coerce").diff(20)
        regime = pd.Series("insufficient_history", index=df.index, dtype=object)
        regime.loc[delta > 0] = "up_20bar"
        regime.loc[delta < 0] = "down_20bar"
        regime.loc[delta.eq(0)] = "flat_20bar"
        return regime

    regime_proxy = _regime_proxy()

    ifvg_summary = {
        "canonical_ifvg_count": int(len(canonical)),
        "bull_detect_count": int((canonical["ifvg_side"] == "bull").sum()),
        "bear_detect_count": int((canonical["ifvg_side"] == "bear").sum()),
        "bull_active_rows": int(bull_active.sum()),
        "bear_active_rows": int(bear_active.sum()),
        "active_row_counts": {
            "bull_active_rows": int(bull_active.sum()),
            "bear_active_rows": int(bear_active.sum()),
        },
    }
    source_linkage_audit = {
        "canonical_ifvg_count": int(len(canonical)),
        "source_invalidated_fvg_count": int(len(invalidated_sources)),
        "orphan_ifvg_count": int(orphan_count),
        "all_ifvgs_have_valid_invalidated_source": orphan_count == 0,
        "detect_idx_equals_source_invalidation_idx_check": bool(detect_idx_match),
    }
    candidate_filter_audit = {
        "raw_invalidation_candidate_count": int(len(candidates)),
        "canonical_accepted_count": int(len(canonical)),
        "filtered_out_count": (
            int((~candidates["accepted"]).sum()) if not candidates.empty else 0
        ),
        "filtered_by_reason": {
            "source_too_small": int(rejection_counts.get("source_too_small", 0)),
            "source_too_young": int(rejection_counts.get("source_too_young", 0)),
            "atr_unavailable": int(rejection_counts.get("atr_unavailable", 0)),
            "inversion_body_too_small": int(
                rejection_counts.get("inversion_body_too_small", 0)
            ),
            "cross_distance_too_small": int(
                rejection_counts.get("cross_distance_too_small", 0)
            ),
            "invalid_width": int(rejection_counts.get("invalid_width", 0)),
        },
    }
    if not canonical_events.empty:
        canonical_events["time_to_reclaim"] = pd.to_numeric(
            canonical_events["fully_reclaimed_idx"], errors="coerce"
        ) - pd.to_numeric(canonical_events["detect_idx"], errors="coerce")
    lifecycle_audit = {
        "tested_count": (
            int(detect_rows["first_retest_idx"].notna().sum())
            if not detect_rows.empty
            else 0
        ),
        "holding_row_count": int(
            (
                pd.to_numeric(df["ifvg_bull_active_state"], errors="coerce").fillna(0)
                == 3
            ).sum()
            + (
                pd.to_numeric(df["ifvg_bear_active_state"], errors="coerce").fillna(0)
                == 3
            ).sum()
        ),
        "failing_row_count": int(
            (
                pd.to_numeric(df["ifvg_bull_active_state"], errors="coerce").fillna(0)
                == 4
            ).sum()
            + (
                pd.to_numeric(df["ifvg_bear_active_state"], errors="coerce").fillna(0)
                == 4
            ).sum()
        ),
        "fully_reclaimed_count": (
            int((detect_rows["terminal_state"] == IFVG_STATE_FULLY_RECLAIMED).sum())
            if not detect_rows.empty
            else 0
        ),
        "expired_count": (
            int((detect_rows["terminal_state"] == IFVG_STATE_EXPIRED).sum())
            if not detect_rows.empty
            else 0
        ),
        "first_retest_rate": (
            float(detect_rows["first_retest_idx"].notna().mean())
            if not detect_rows.empty
            else np.nan
        ),
        "reclaim_rate": (
            float((detect_rows["terminal_state"] == IFVG_STATE_FULLY_RECLAIMED).mean())
            if not detect_rows.empty
            else np.nan
        ),
        "reclaim_rate_by_side": (
            {
                side: float(
                    (rows["terminal_state"] == IFVG_STATE_FULLY_RECLAIMED).mean()
                )
                for side, rows in canonical_events.groupby("side")
            }
            if not canonical_events.empty
            else {}
        ),
        "time_to_reclaim": _continuous_stats(
            canonical_events.loc[
                canonical_events["terminal_state"] == IFVG_STATE_FULLY_RECLAIMED,
                "time_to_reclaim",
            ]
            if not canonical_events.empty
            else pd.Series(dtype=float)
        ),
        "reclaim_rate_by_source_age_bucket": _bucketed_reclaim_rate(
            "source_age_bars",
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=["0_2", "3_5", "6_10", "11_plus"],
        ),
        "reclaim_rate_by_source_fill_bucket": _bucketed_reclaim_rate(
            "source_fill_pct_at_inversion",
            bins=[-np.inf, 0.0, 0.25, 0.5, 0.75, np.inf],
            labels=["0", "0_25", "25_50", "50_75", "75_plus"],
        ),
        "no_test_on_detect_invariant": (
            bool(
                (
                    detect_rows["first_retest_idx"].dropna()
                    > detect_rows.loc[
                        detect_rows["first_retest_idx"].notna(), "detect_idx"
                    ]
                ).all()
            )
            if not detect_rows.empty and detect_rows["first_retest_idx"].notna().any()
            else True
        ),
        "no_fully_reclaimed_rep_still_active": terminal_leak_counts["fully_reclaimed"]
        == 0,
        "no_expired_rep_still_active": terminal_leak_counts["expired"] == 0,
        "no_invalid_terminal_rep_still_active": terminal_leak_counts["invalid_terminal"]
        == 0,
    }
    distributional_fingerprint = {
        "ifvg_per_1000_bars": (
            float(len(canonical) * 1000.0 / len(df)) if len(df) else np.nan
        ),
        "bull_bear_ratio": (
            float(int(bull_detect.sum()) / int(bear_detect.sum()))
            if int(bear_detect.sum()) > 0
            else np.nan
        ),
        "ifvg_to_fvg_ratio": (
            float(len(canonical) / len(invalidated_sources))
            if len(invalidated_sources) > 0
            else np.nan
        ),
        "source_age_bars": _continuous_stats(
            detect_rows["source_age_bars"]
            if not detect_rows.empty
            else pd.Series(dtype=float)
        ),
        "source_fill_pct_at_inversion": _continuous_stats(
            detect_rows["source_fill_pct_at_inversion"]
            if not detect_rows.empty
            else pd.Series(dtype=float)
        ),
        "inversion_score": _continuous_stats(
            detect_rows["inversion_score"]
            if not detect_rows.empty
            else pd.Series(dtype=float)
        ),
        "active_count_distribution": {
            "bull": _continuous_stats(
                pd.to_numeric(df["ifvg_bull_active_count"], errors="coerce")
            ),
            "bear": _continuous_stats(
                pd.to_numeric(df["ifvg_bear_active_count"], errors="coerce")
            ),
        },
    }
    dense_side_detect_total = int(bull_detect.sum()) + int(bear_detect.sum())
    event_distribution_eligible_count = int(len(detect_rows))
    universe_reconciliation = {
        "canonical_ifvg_count": int(len(canonical_events)),
        "canonical_detect_total_by_side": int(
            (canonical["ifvg_side"] == "bull").sum()
            + (canonical["ifvg_side"] == "bear").sum()
        ),
        "dense_side_detect_total": int(dense_side_detect_total),
        "event_distribution_eligible_count": int(event_distribution_eligible_count),
        "excluded_from_event_distribution_count": int(
            len(canonical_events) - event_distribution_eligible_count
        ),
        "excluded_from_legacy_sparse_detect_universe_count": int(len(dense_excluded)),
        "exclusion_reasons": (
            {
                "dense_detect_surface_exports_only_one_ifvg_per_side_per_row": int(
                    len(dense_excluded)
                )
            }
            if not dense_excluded.empty
            else {}
        ),
        "excluded_events_table": (
            dense_excluded[
                [
                    "ifvg_event_id",
                    "side",
                    "detect_idx",
                    "source_fvg_event_id",
                    "same_side_same_row_canonical_count",
                    "exclusion_reason",
                ]
            ].to_dict(orient="records")
            if not dense_excluded.empty
            else []
        ),
    }
    bull_window_profile = _window_active_profile("bull")
    bear_window_profile = _window_active_profile("bear")
    if active_pool.empty:
        asymmetry_by_regime = {}
    else:
        row_side_counts = (
            active_pool.groupby(["row_idx", "side"])["ifvg_event_id"]
            .size()
            .unstack(fill_value=0)
            .reindex(df.index, fill_value=0)
        )
        row_side_counts["regime"] = regime_proxy
        asymmetry_by_regime = {
            str(regime): {
                "rows": int(len(rows)),
                "bull_mean_active_count": (
                    float(rows.get("bull", pd.Series(dtype=float)).mean())
                    if "bull" in rows
                    else 0.0
                ),
                "bear_mean_active_count": (
                    float(rows.get("bear", pd.Series(dtype=float)).mean())
                    if "bear" in rows
                    else 0.0
                ),
            }
            for regime, rows in row_side_counts.groupby("regime", dropna=False)
        }
    bull_age = _active_age_stats("bull")
    bear_age = _active_age_stats("bear")
    active_pool_policy_audit = {
        "bull_active_rows": int(bull_window_profile["active_rows"]),
        "total_rows": int(bull_window_profile["total_rows"]),
        "rows_with_zero_bull_active_count": int(
            bull_window_profile["rows_with_zero_active_count"]
        ),
        "active_count_on_first_row": int(
            bull_window_profile["active_count_on_first_row"]
        ),
        "active_count_on_last_row": int(
            bull_window_profile["active_count_on_last_row"]
        ),
        "carried_in_active_bull_count_on_first_row": int(
            bull_window_profile["carried_in_active_count_on_first_row"]
        ),
        "carried_in_active_bull_count_on_last_row": int(
            bull_window_profile["carried_in_active_count_on_last_row"]
        ),
        "max_active_age_in_pool_bull": int(bull_age["max"]),
        "mean_active_age_in_pool_bull": bull_age["mean"],
        "median_active_age_in_pool_bull": bull_age["median"],
        "p90_active_age_in_pool_bull": bull_age["p90"],
        "max_active_age_in_pool_bear": int(bear_age["max"]),
        "mean_active_age_in_pool_bear": bear_age["mean"],
        "median_active_age_in_pool_bear": bear_age["median"],
        "p90_active_age_in_pool_bear": bear_age["p90"],
        "clipped_window_bull": bull_window_profile,
        "clipped_window_bear": bear_window_profile,
        "global_active_age_bull": bull_age,
        "global_active_age_bear": bear_age,
        "active_count_asymmetry": {
            "bull_mean_active_count": float(
                pd.to_numeric(df["ifvg_bull_active_count"], errors="coerce").mean()
            ),
            "bear_mean_active_count": float(
                pd.to_numeric(df["ifvg_bear_active_count"], errors="coerce").mean()
            ),
            "by_regime_proxy_20bar": asymmetry_by_regime,
            "by_source_direction_note": (
                "source_direction is deterministic by IFVG side "
                "(bull IFVG <- bear FVG, bear IFVG <- bull FVG), so side "
                "asymmetry already encodes source-direction asymmetry."
            ),
        },
        "bull_active_every_row_not_causality_error": bool(
            bull_window_profile["active_rows"] == bull_window_profile["total_rows"]
            and lifecycle_audit["no_test_on_detect_invariant"]
            and source_linkage_audit["detect_idx_equals_source_invalidation_idx_check"]
        ),
        "not_causality_error": bool(
            lifecycle_audit["no_test_on_detect_invariant"]
            and source_linkage_audit["detect_idx_equals_source_invalidation_idx_check"]
            and lifecycle_audit["no_invalid_terminal_rep_still_active"]
        ),
        "bull_active_every_row_main_driver": (
            "lack_of_expiry_plus_carried_inventory_and_frequent_new_bull_ifvgs"
            if bull_window_profile["active_rows"] == bull_window_profile["total_rows"]
            else "not_always_active"
        ),
        "active_pool_policy_problem": bool(
            bull_window_profile["active_rows"] == bull_window_profile["total_rows"]
            and terminal_leak_counts["invalid_terminal"] == 0
        ),
        "recommendation": {
            "canonical_hard_expiry_for_active_state_export": True,
            "age_based_significance_decay": True,
            "note": (
                "Keep hard expiry and age-based decay enabled in the canonical "
                "export path."
                if not (
                    bull_window_profile["active_rows"]
                    == bull_window_profile["total_rows"]
                    and terminal_leak_counts["invalid_terminal"] == 0
                )
                else (
                    "This still looks like an active-pool policy problem rather "
                    "than a causality error. Hard expiry limits pathological "
                    "persistence; age-based decay improves selection even before "
                    "expiry."
                )
            ),
        },
    }
    return {
        "ifvg_summary": ifvg_summary,
        "ifvg_source_linkage_audit": source_linkage_audit,
        "ifvg_candidate_filter_audit": candidate_filter_audit,
        "ifvg_lifecycle_audit": lifecycle_audit,
        "ifvg_distributional_fingerprint": distributional_fingerprint,
        "ifvg_universe_reconciliation": universe_reconciliation,
        "ifvg_active_pool_policy_audit": active_pool_policy_audit,
    }


def summarize_fvg(
    window_df: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    debug_tables: dict[str, pd.DataFrame],
    research_table: pd.DataFrame | None = None,
    old_no_expiry_research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    required = {"timestamp", "open", "high", "low", "close", *ALL_FVG_CORE_COLUMNS}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing FVG columns: {sorted(missing)}")

    bull_origin = window_df["fvg_bull_origin_flag"] == 1
    bear_origin = window_df["fvg_bear_origin_flag"] == 1
    bull_detect = window_df["fvg_bull_detect_flag"] == 1
    bear_detect = window_df["fvg_bear_detect_flag"] == 1
    detect_rows = bull_detect | bear_detect
    bull_events = window_df.loc[bull_detect]
    bear_events = window_df.loc[bear_detect]
    window_rows = window_df.index
    event_table = debug_tables["event_table"]
    event_window = event_table[
        event_table["detect_idx"].between(
            int(window_rows.min()), int(window_rows.max())
        )
    ].copy()

    def _expiry_stats(side: str) -> dict[str, float | int]:
        side_events = event_table[
            (event_table["side"] == side)
            & event_table["detect_idx"].between(
                int(window_rows.min()), int(window_rows.max())
            )
        ].copy()
        if side_events.empty:
            return {
                "expired_count": 0,
                "expired_before_touch_rate": np.nan,
                "expired_after_touch_rate": np.nan,
                "expired_after_partial_fill_rate": np.nan,
            }
        expired = side_events["terminal_state"] == 6
        return {
            "expired_count": int(expired.sum()),
            "expired_before_touch_rate": float(
                (expired & side_events["first_touch_idx"].isna()).mean()
            ),
            "expired_after_touch_rate": float(
                (expired & side_events["first_touch_idx"].notna()).mean()
            ),
            "expired_after_partial_fill_rate": float(
                (expired & side_events["first_partial_fill_idx"].notna()).mean()
            ),
        }

    pool = {
        "bull": _pool_side_diagnostics(
            side="bull",
            window_rows=window_rows,
            full_df=full_df,
            active_members=debug_tables["active_member_table"],
        ),
        "bear": _pool_side_diagnostics(
            side="bear",
            window_rows=window_rows,
            full_df=full_df,
            active_members=debug_tables["active_member_table"],
        ),
    }

    research_summary = None
    global_reconciliation_report = None
    window_forensic_reconciliation = None
    if research_table is not None and not research_table.empty:
        global_research_summary = summarize_fvg_research(research_table)
        research_window = research_table[
            research_table["fvg_detect_idx"].between(
                int(window_rows.min()), int(window_rows.max())
            )
        ].copy()
        research_summary = summarize_fvg_research(research_window)
        global_reconciliation_report = global_research_summary
        window_forensic_reconciliation = _forensic_window_reconciliation(research_table)
        if (
            old_no_expiry_research_table is not None
            and not old_no_expiry_research_table.empty
        ):
            global_reconciliation_report["old_no_expiry_unresolved_audit"] = (
                summarize_old_no_expiry_unresolved_audit(old_no_expiry_research_table)
            )
    fill_summary = _summarize_fill(window_df, research_table=research_table)
    ifvg_summary_blocks = _summarize_ifvg(full_df, debug_tables=debug_tables)
    overlap_fill_forensics = _summarize_overlap_fill_forensics(
        full_df,
        active_members=debug_tables["active_member_table"],
    )
    core_fill_ownership_reconciliation = _summarize_core_fill_ownership_reconciliation(
        full_df,
        active_members=debug_tables["active_member_table"],
    )
    live_historical_fill_split = _summarize_live_vs_historical_fill_split(
        debug_tables["active_member_table"]
    )
    fill_selection_reconciliation = _summarize_fill_selection_reconciliation(
        full_df,
        active_members=debug_tables["active_member_table"],
    )

    return {
        "window": {
            "start": str(pd.to_datetime(window_df["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(window_df["timestamp"], utc=True).max()),
            "rows": int(len(window_df)),
        },
        "schema": {
            "expected_fvg_columns": len(ALL_FVG_CORE_COLUMNS),
            "present_fvg_columns": int(
                sum(col in full_df.columns for col in ALL_FVG_CORE_COLUMNS)
            ),
            "missing_fvg_columns": sorted(
                set(ALL_FVG_CORE_COLUMNS) - set(full_df.columns)
            ),
        },
        "event_counts": {
            "fvg_count": int(detect_rows.sum()),
            "fvg_bull_count": int(bull_detect.sum()),
            "fvg_bear_count": int(bear_detect.sum()),
            "fvg_rate": float(detect_rows.mean()) if len(window_df) else np.nan,
        },
        "bull_event_stats": {
            "width_atr": _continuous_stats(bull_events["fvg_bull_width_atr"]),
            "mid_body_atr": _continuous_stats(bull_events["fvg_bull_mid_body_atr"]),
            "gap_cleanliness": _capped_cleanliness_stats(
                bull_events["fvg_bull_gap_cleanliness"]
            ),
            "selected_effective_significance": _continuous_stats(
                window_df.loc[
                    window_df["fvg_bull_active"] == 1,
                    "fvg_bull_active_effective_significance",
                ]
            ),
            "terminal_state_counts": _value_counts(
                bull_events["fvg_bull_terminal_state"]
            ),
            **_expiry_stats("bull"),
        },
        "bear_event_stats": {
            "width_atr": _continuous_stats(bear_events["fvg_bear_width_atr"]),
            "mid_body_atr": _continuous_stats(bear_events["fvg_bear_mid_body_atr"]),
            "gap_cleanliness": _capped_cleanliness_stats(
                bear_events["fvg_bear_gap_cleanliness"]
            ),
            "selected_effective_significance": _continuous_stats(
                window_df.loc[
                    window_df["fvg_bear_active"] == 1,
                    "fvg_bear_active_effective_significance",
                ]
            ),
            "terminal_state_counts": _value_counts(
                bear_events["fvg_bear_terminal_state"]
            ),
            **_expiry_stats("bear"),
        },
        "active_pool": pool,
        "fill": fill_summary,
        "ifvg_summary": (
            None if ifvg_summary_blocks is None else ifvg_summary_blocks["ifvg_summary"]
        ),
        "ifvg_source_linkage_audit": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_source_linkage_audit"]
        ),
        "ifvg_candidate_filter_audit": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_candidate_filter_audit"]
        ),
        "ifvg_lifecycle_audit": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_lifecycle_audit"]
        ),
        "ifvg_distributional_fingerprint": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_distributional_fingerprint"]
        ),
        "ifvg_universe_reconciliation": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_universe_reconciliation"]
        ),
        "ifvg_active_pool_policy_audit": (
            None
            if ifvg_summary_blocks is None
            else ifvg_summary_blocks["ifvg_active_pool_policy_audit"]
        ),
        "overlap_fill_forensics": overlap_fill_forensics,
        "core_fill_ownership_reconciliation": core_fill_ownership_reconciliation,
        "live_historical_fill_split": live_historical_fill_split,
        "fill_selection_reconciliation": fill_selection_reconciliation,
        "research": research_summary,
        "global_reconciliation_report": global_reconciliation_report,
        "window_forensic_reconciliation": window_forensic_reconciliation,
        "sanity_checks": {
            "event_id_zero_off_bull_detect_rows": bool(
                (window_df.loc[~bull_detect, "fvg_bull_event_id"] == 0).all()
            ),
            "event_id_zero_off_bear_detect_rows": bool(
                (window_df.loc[~bear_detect, "fvg_bear_event_id"] == 0).all()
            ),
            "bull_detect_delay_is_one": bool(
                (window_df.loc[bull_detect, "fvg_bull_detect_delay"] == 1).all()
                if bull_detect.any()
                else True
            ),
            "bear_detect_delay_is_one": bool(
                (window_df.loc[bear_detect, "fvg_bear_detect_delay"] == 1).all()
                if bear_detect.any()
                else True
            ),
            "bull_origin_annotations_consistent": bool(
                (
                    window_df.loc[bull_origin, "fvg_bull_origin_idx"]
                    == window_df.loc[bull_origin, "fvg_bull_mid_idx"]
                ).all()
                and (
                    window_df.loc[bull_origin, "fvg_bull_right_idx"]
                    > window_df.loc[bull_origin, "fvg_bull_origin_idx"]
                ).all()
                if bull_origin.any()
                else True
            ),
            "bear_origin_annotations_consistent": bool(
                (
                    window_df.loc[bear_origin, "fvg_bear_origin_idx"]
                    == window_df.loc[bear_origin, "fvg_bear_mid_idx"]
                ).all()
                and (
                    window_df.loc[bear_origin, "fvg_bear_right_idx"]
                    > window_df.loc[bear_origin, "fvg_bear_origin_idx"]
                ).all()
                if bear_origin.any()
                else True
            ),
            "bull_first_touch_after_detect": bool(
                (
                    window_df.loc[
                        bull_detect & window_df["fvg_bull_first_touch_idx"].notna(),
                        "fvg_bull_first_touch_idx",
                    ]
                    > window_df.loc[
                        bull_detect & window_df["fvg_bull_first_touch_idx"].notna(),
                        "fvg_bull_detect_idx",
                    ]
                ).all()
                if (bull_detect & window_df["fvg_bull_first_touch_idx"].notna()).any()
                else True
            ),
            "bear_first_touch_after_detect": bool(
                (
                    window_df.loc[
                        bear_detect & window_df["fvg_bear_first_touch_idx"].notna(),
                        "fvg_bear_first_touch_idx",
                    ]
                    > window_df.loc[
                        bear_detect & window_df["fvg_bear_first_touch_idx"].notna(),
                        "fvg_bear_detect_idx",
                    ]
                ).all()
                if (bear_detect & window_df["fvg_bear_first_touch_idx"].notna()).any()
                else True
            ),
            "no_same_bar_fill_on_bull_detect_rows": bool(
                (
                    event_window.loc[
                        (event_window["side"] == "bull")
                        & event_window["first_partial_fill_idx"].notna(),
                        "first_partial_fill_idx",
                    ]
                    > event_window.loc[
                        (event_window["side"] == "bull")
                        & event_window["first_partial_fill_idx"].notna(),
                        "detect_idx",
                    ]
                ).all()
                if (
                    (event_window["side"] == "bull")
                    & event_window["first_partial_fill_idx"].notna()
                ).any()
                else True
            ),
            "no_same_bar_fill_on_bear_detect_rows": bool(
                (
                    event_window.loc[
                        (event_window["side"] == "bear")
                        & event_window["first_partial_fill_idx"].notna(),
                        "first_partial_fill_idx",
                    ]
                    > event_window.loc[
                        (event_window["side"] == "bear")
                        & event_window["first_partial_fill_idx"].notna(),
                        "detect_idx",
                    ]
                ).all()
                if (
                    (event_window["side"] == "bear")
                    & event_window["first_partial_fill_idx"].notna()
                ).any()
                else True
            ),
            "active_count_consistent_with_debug_bull": pool["bull"][
                "active_count_matches_dense_export"
            ],
            "active_count_consistent_with_debug_bear": pool["bear"][
                "active_count_matches_dense_export"
            ],
            "no_terminal_reps_still_counted_bull": pool["bull"][
                "terminal_but_still_in_active_pool_count"
            ]
            == 0,
            "no_terminal_reps_still_counted_bear": pool["bear"][
                "terminal_but_still_in_active_pool_count"
            ]
            == 0,
        },
    }


def fvg_event_windows(
    df: pd.DataFrame,
    *,
    side: str = "all",
    bars_before: int = 6,
    bars_after: int = 8,
    limit: int = 5,
) -> list[pd.DataFrame]:
    if side not in {"all", "bull", "bear"}:
        raise ValueError("side must be one of {'all', 'bull', 'bear'}")

    if side == "bull":
        event_pos = np.flatnonzero(df["fvg_bull_detect_flag"].fillna(0).to_numpy())[
            :limit
        ]
    elif side == "bear":
        event_pos = np.flatnonzero(df["fvg_bear_detect_flag"].fillna(0).to_numpy())[
            :limit
        ]
    else:
        event_pos = np.flatnonzero(
            (
                (df["fvg_bull_detect_flag"] == 1) | (df["fvg_bear_detect_flag"] == 1)
            ).to_numpy()
        )[:limit]

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "fvg_bull_detect_flag",
        "fvg_bull_event_id",
        "fvg_bull_low",
        "fvg_bull_high",
        "fvg_bull_width_atr",
        "fvg_bull_terminal_state",
        "fvg_bull_active_count",
        "fvg_bear_detect_flag",
        "fvg_bear_event_id",
        "fvg_bear_low",
        "fvg_bear_high",
        "fvg_bear_width_atr",
        "fvg_bear_terminal_state",
        "fvg_bear_active_count",
    ]
    cols = [col for col in cols if col in df.columns]

    windows: list[pd.DataFrame] = []
    for pos in event_pos:
        start = max(0, pos - bars_before)
        stop = min(len(df), pos + bars_after + 1)
        windows.append(df.iloc[start:stop][cols].copy())
    return windows


def _add_rectangle_traces(
    fig: go.Figure,
    *,
    rep_table: pd.DataFrame,
    full_df: pd.DataFrame,
    window_start_idx: int,
    window_end_idx: int,
    row: int,
    col: int,
) -> None:
    if rep_table.empty:
        return

    delta = _bar_delta(full_df.loc[window_start_idx:window_end_idx, "timestamp"])
    bull_marker_x: list[pd.Timestamp] = []
    bull_marker_y: list[float] = []
    bull_custom: list[list[object]] = []
    bear_marker_x: list[pd.Timestamp] = []
    bear_marker_y: list[float] = []
    bear_custom: list[list[object]] = []

    for rep in rep_table.itertuples(index=False):
        start_idx = max(int(rep.start_idx), window_start_idx)
        end_idx = min(int(rep.end_idx), window_end_idx)
        if start_idx > end_idx:
            continue

        x0 = pd.to_datetime(full_df.loc[start_idx, "timestamp"], utc=True)
        x1 = pd.to_datetime(full_df.loc[end_idx, "timestamp"], utc=True) + delta
        carried_in = int(rep.start_idx) < window_start_idx
        merged_rep = int(rep.source_count) > 1

        if rep.side == "bull":
            fill = (
                "rgba(34, 197, 94, 0.07)"
                if not carried_in
                else "rgba(34, 197, 94, 0.03)"
            )
            line = "#16a34a"
        else:
            fill = (
                "rgba(239, 68, 68, 0.07)"
                if not carried_in
                else "rgba(239, 68, 68, 0.03)"
            )
            line = "#dc2626"

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=float(rep.low),
            y1=float(rep.high),
            xref=f"x{'' if row == 1 else row}",
            yref=f"y{'' if row == 1 else row}",
            fillcolor=fill,
            line=dict(color=line, width=0.9, dash="dot" if merged_rep else "solid"),
            layer="below",
            row=row,
            col=col,
        )

        if (
            int(rep.start_idx) >= window_start_idx
            and int(rep.start_idx) <= window_end_idx
        ):
            start_ts = pd.to_datetime(
                full_df.loc[int(rep.start_idx), "timestamp"], utc=True
            )
            custom = [
                int(rep.active_id),
                int(rep.source_count),
                str(rep.terminal_reason),
                float(rep.width_atr),
                int(rep.detect_idx),
            ]
            if rep.side == "bull":
                bull_marker_x.append(start_ts)
                bull_marker_y.append(float(rep.high))
                bull_custom.append(custom)
            else:
                bear_marker_x.append(start_ts)
                bear_marker_y.append(float(rep.low))
                bear_custom.append(custom)

    if bull_marker_x:
        bull_custom_arr = np.asarray(bull_custom, dtype=object)
        fig.add_trace(
            go.Scatter(
                x=bull_marker_x,
                y=bull_marker_y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
                customdata=bull_custom_arr,
                hovertemplate=(
                    "Bull Active Rep<br>"
                    "active_id=%{customdata[0]}<br>"
                    "source_count=%{customdata[1]}<br>"
                    "terminal=%{customdata[2]}<br>"
                    "width_atr=%{customdata[3]}<br>"
                    "detect_idx=%{customdata[4]}<extra></extra>"
                ),
                name="Bull Rep Start",
            ),
            row=row,
            col=col,
        )
    if bear_marker_x:
        bear_custom_arr = np.asarray(bear_custom, dtype=object)
        fig.add_trace(
            go.Scatter(
                x=bear_marker_x,
                y=bear_marker_y,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
                customdata=bear_custom_arr,
                hovertemplate=(
                    "Bear Active Rep<br>"
                    "active_id=%{customdata[0]}<br>"
                    "source_count=%{customdata[1]}<br>"
                    "terminal=%{customdata[2]}<br>"
                    "width_atr=%{customdata[3]}<br>"
                    "detect_idx=%{customdata[4]}<extra></extra>"
                ),
                name="Bear Rep Start",
            ),
            row=row,
            col=col,
        )


def _add_selected_remaining_overlays(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    needed = {
        "fvg_fill_bull_remaining_low",
        "fvg_fill_bull_remaining_high",
        "fvg_fill_bear_remaining_low",
        "fvg_fill_bear_remaining_high",
    }
    if not needed.issubset(window_df.columns):
        return

    delta = _bar_delta(window_df["timestamp"])
    for side, fill, line in (
        ("bull", "rgba(34, 197, 94, 0.30)", "#15803d"),
        ("bear", "rgba(239, 68, 68, 0.30)", "#b91c1c"),
    ):
        low_col = f"fvg_fill_{side}_remaining_low"
        high_col = f"fvg_fill_{side}_remaining_high"
        hover_cols = [
            "timestamp",
            low_col,
            high_col,
            f"fvg_fill_{side}_remaining_width_atr",
            f"fvg_fill_{side}_remaining_frac",
            f"fvg_fill_{side}_fill_pct_delta_1",
            f"fvg_fill_{side}_fill_pct_delta_3",
            f"fvg_fill_{side}_fill_type",
            f"fvg_fill_{side}_bounce_strength_atr",
            f"fvg_fill_{side}_bars_since_touch",
            f"fvg_fill_{side}_bars_since_first_touch",
            f"fvg_fill_{side}_touch_rejection_flag",
            f"fvg_fill_{side}_rep_id",
            f"fvg_fill_{side}_rep_fill_pct",
            f"fvg_fill_{side}_rep_state",
            f"fvg_fill_{side}_rep_touch_count",
            f"fvg_{side}_active_count",
            f"fvg_fill_{side}_rep_detect_idx",
            f"fvg_fill_{side}_rep_matches_structural_selected",
            f"fvg_fill_{side}_rep_selection_mode",
            f"fvg_{side}_active_id",
            f"fvg_{side}_active_fill_pct",
        ]
        scoped = window_df[
            window_df[low_col].notna() & window_df[high_col].notna()
        ].copy()
        if scoped.empty:
            continue

        selection_mode = (
            scoped[f"fvg_fill_{side}_rep_selection_mode"].fillna("").astype(str)
        )
        rep_touch_count = pd.to_numeric(
            scoped[f"fvg_fill_{side}_rep_touch_count"], errors="coerce"
        ).fillna(0)
        rep_fill_pct = pd.to_numeric(
            scoped[f"fvg_fill_{side}_rep_fill_pct"], errors="coerce"
        ).fillna(0.0)
        real_fill_context = (
            selection_mode.eq("touched_pool")
            | rep_touch_count.gt(0)
            | rep_fill_pct.gt(0.0)
        )
        visual_scoped = scoped[real_fill_context].copy()

        for row_data in visual_scoped.itertuples(index=False):
            x0 = pd.to_datetime(getattr(row_data, "timestamp"), utc=True)
            x1 = x0 + delta
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=float(getattr(row_data, low_col)),
                y1=float(getattr(row_data, high_col)),
                xref=f"x{'' if row == 1 else row}",
                yref=f"y{'' if row == 1 else row}",
                fillcolor=fill,
                line=dict(color=line, width=0.9),
                layer="below",
                row=row,
                col=col,
            )

        if not visual_scoped.empty:
            custom = visual_scoped[hover_cols].to_numpy(dtype=object)
            fig.add_trace(
                go.Scatter(
                    x=visual_scoped["timestamp"],
                    y=(visual_scoped[low_col] + visual_scoped[high_col]) / 2.0,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=line,
                        symbol="diamond-open",
                        line=dict(color=line, width=1.2),
                    ),
                    customdata=custom,
                    hovertemplate=(
                        f"Fill Rep {side.title()} Marker + Remaining Zone<br>"
                        "fill_rep_id=%{customdata[12]}<br>"
                        "fill_rep_fill_pct=%{customdata[13]}<br>"
                        "fill_rep_state=%{customdata[14]}<br>"
                        "fill_rep_touch_count=%{customdata[15]}<br>"
                        "side_active_count=%{customdata[16]}<br>"
                        "fill_rep_detect_idx=%{customdata[17]}<br>"
                        "matches_structural_selected=%{customdata[18]}<br>"
                        "selection_mode=%{customdata[19]}<br>"
                        "structural_selected_id=%{customdata[20]}<br>"
                        "structural_fill_pct=%{customdata[21]}<br>"
                        "low=%{customdata[1]}<br>"
                        "high=%{customdata[2]}<br>"
                        "remaining_width_atr=%{customdata[3]}<br>"
                        "remaining_frac=%{customdata[4]}<br>"
                        "fill_pct_delta_1=%{customdata[5]}<br>"
                        "fill_pct_delta_3=%{customdata[6]}<br>"
                        "fill_type=%{customdata[7]}<br>"
                        "bounce_strength_atr=%{customdata[8]}<br>"
                        "bars_since_touch=%{customdata[9]}<br>"
                        "bars_since_first_touch=%{customdata[10]}<br>"
                        "touch_rejection_flag=%{customdata[11]}<extra></extra>"
                    ),
                    name=f"Fill Rep {side.title()}",
                ),
                row=row,
                col=col,
            )

        mismatch = visual_scoped[
            pd.to_numeric(
                visual_scoped[f"fvg_fill_{side}_rep_matches_structural_selected"],
                errors="coerce",
            ).fillna(0)
            == 0
        ].copy()
        if not mismatch.empty:
            mismatch_custom = mismatch[
                [
                    f"fvg_{side}_active_id",
                    f"fvg_{side}_active_fill_pct",
                    f"fvg_fill_{side}_rep_id",
                    f"fvg_fill_{side}_rep_fill_pct",
                    f"fvg_fill_{side}_rep_selection_mode",
                    f"fvg_fill_{side}_remaining_width",
                    f"fvg_fill_{side}_remaining_frac",
                ]
            ].to_numpy(dtype=object)
            fig.add_trace(
                go.Scatter(
                    x=mismatch["timestamp"],
                    y=(mismatch[low_col] + mismatch[high_col]) / 2.0,
                    mode="markers",
                    marker=dict(
                        size=9,
                        symbol="x",
                        color=line,
                        line=dict(color="white", width=1.0),
                    ),
                    customdata=mismatch_custom,
                    hovertemplate=(
                        f"{side.title()} Structural vs Fill Mismatch<br>"
                        "structural_selected_id=%{customdata[0]}<br>"
                        "structural_fill_pct=%{customdata[1]}<br>"
                        "fill_rep_id=%{customdata[2]}<br>"
                        "fill_rep_fill_pct=%{customdata[3]}<br>"
                        "selection_mode=%{customdata[4]}<br>"
                        "fill_rep_remaining_width=%{customdata[5]}<br>"
                        "fill_rep_remaining_frac=%{customdata[6]}<extra></extra>"
                    ),
                    name=f"{side.title()} Structural/Fill Mismatch",
                ),
                row=row,
                col=col,
            )


def _add_structural_selected_markers(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    for side, color in (("bull", "#065f46"), ("bear", "#7f1d1d")):
        scoped = window_df[
            pd.to_numeric(window_df[f"fvg_{side}_active"], errors="coerce").fillna(0)
            == 1
        ].copy()
        if scoped.empty:
            continue
        custom = scoped[
            [
                f"fvg_{side}_active_id",
                f"fvg_{side}_active_fill_pct",
                f"fvg_{side}_active_touch_count",
                f"fvg_{side}_active_state",
                f"fvg_{side}_active_since_idx",
            ]
        ].to_numpy(dtype=object)
        fig.add_trace(
            go.Scatter(
                x=scoped["timestamp"],
                y=(scoped[f"fvg_{side}_active_low"] + scoped[f"fvg_{side}_active_high"])
                / 2.0,
                mode="markers",
                marker=dict(
                    size=5,
                    color="white",
                    symbol="circle-open",
                    line=dict(color=color, width=1.2),
                ),
                customdata=custom,
                hovertemplate=(
                    f"Structural Selected {side.title()} Rep<br>"
                    "structural_active_id=%{customdata[0]}<br>"
                    "structural_fill_pct=%{customdata[1]}<br>"
                    "structural_touch_count=%{customdata[2]}<br>"
                    "structural_state=%{customdata[3]}<br>"
                    "structural_active_since_idx=%{customdata[4]}<extra></extra>"
                ),
                name=f"Structural Selected {side.title()}",
            ),
            row=row,
            col=col,
        )


def _add_touch_episode_markers(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    needed = {
        "fvg_fill_bull_bars_since_touch",
        "fvg_fill_bear_bars_since_touch",
    }
    if not needed.issubset(window_df.columns):
        return

    for side, color, symbol, y_col in (
        ("bull", "#15803d", "star", "high"),
        ("bear", "#b91c1c", "star", "low"),
    ):
        scoped = window_df[
            (
                pd.to_numeric(
                    window_df[f"fvg_fill_{side}_bars_since_touch"], errors="coerce"
                )
                == 0
            )
            & pd.to_numeric(
                window_df[f"fvg_fill_{side}_bounce_strength_atr"], errors="coerce"
            ).notna()
        ].copy()
        if scoped.empty:
            continue
        custom = scoped[
            [
                f"fvg_{side}_active_id",
                f"fvg_fill_{side}_bounce_strength_atr",
                f"fvg_fill_{side}_fill_type",
                f"fvg_fill_{side}_touch_rejection_flag",
            ]
        ].to_numpy(dtype=object)
        fig.add_trace(
            go.Scatter(
                x=scoped["timestamp"],
                y=scoped[y_col],
                mode="markers",
                marker=dict(color=color, size=9, symbol=symbol),
                customdata=custom,
                hovertemplate=(
                    f"{side.title()} Touch Episode<br>"
                    "active_id=%{customdata[0]}<br>"
                    "bounce_strength_atr=%{customdata[1]}<br>"
                    "fill_type=%{customdata[2]}<br>"
                    "touch_rejection_flag=%{customdata[3]}<extra></extra>"
                ),
                name=f"{side.title()} Touch",
            ),
            row=row,
            col=col,
        )


def _add_event_markers(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    research_table: pd.DataFrame | None,
    row: int,
    col: int,
) -> None:
    if research_table is None or research_table.empty:
        return

    window_events = research_table[
        research_table["fvg_detect_idx"].between(
            int(window_df.index.min()), int(window_df.index.max())
        )
    ].copy()
    if window_events.empty:
        return

    for side, color, symbol, y_col in (
        ("bull", "#16a34a", "circle", "high"),
        ("bear", "#dc2626", "circle", "low"),
    ):
        side_events = window_events[window_events["fvg_side"] == side].copy()
        if side_events.empty:
            continue
        side_events["marker_y"] = window_df.loc[
            side_events["fvg_detect_idx"], y_col
        ].to_numpy(dtype=float)
        custom = np.column_stack(
            [
                side_events["fvg_event_id"].to_numpy(),
                side_events["fvg_terminal_label"].to_numpy(dtype=object),
                side_events["fvg_r_final_outcome"].to_numpy(dtype=object),
                side_events["fvg_expiry_idx"].to_numpy(),
                side_events["fvg_r_continuation_without_touch_flag"].to_numpy(),
                side_events["fvg_r_touched_rejected_flag"].to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=side_events["fvg_detect_ts"],
                y=side_events["marker_y"],
                mode="markers",
                marker=dict(color=color, size=7, symbol=symbol, line=dict(width=1)),
                customdata=custom,
                hovertemplate=(
                    f"{side.title()} FVG Detect<br>"
                    "event_id=%{customdata[0]}<br>"
                    "core_terminal=%{customdata[1]}<br>"
                    "research_final=%{customdata[2]}<br>"
                    "expiry_idx=%{customdata[3]}<br>"
                    "cwt_flag=%{customdata[4]}<br>"
                    "touched_rejected_flag=%{customdata[5]}<extra></extra>"
                ),
                name=f"{side.title()} Detect",
            ),
            row=row,
            col=col,
        )


def _add_terminal_event_markers(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    full_df: pd.DataFrame,
    research_table: pd.DataFrame | None,
    row: int,
    col: int,
) -> None:
    if research_table is None or research_table.empty:
        return

    window_start_idx = int(window_df.index.min())
    window_end_idx = int(window_df.index.max())
    terminal_specs = (
        ("full_fill", "fvg_full_fill_idx", "#2563eb", "diamond", "Full Fill"),
        ("invalidated", "fvg_invalidation_idx", "#111827", "x", "Invalidation"),
        ("expired", "fvg_expiry_idx", "#f59e0b", "square", "Expiry"),
        ("merged", "fvg_merge_idx", "#7c3aed", "hexagon", "Merge"),
    )

    for outcome, idx_col, color, symbol, label in terminal_specs:
        scoped = research_table[
            (research_table["fvg_terminal_label"] == outcome)
            & research_table[idx_col].notna()
        ].copy()
        if scoped.empty:
            continue

        scoped["terminal_idx_int"] = scoped[idx_col].astype(int)
        scoped = scoped[
            scoped["terminal_idx_int"].between(window_start_idx, window_end_idx)
        ].copy()
        if scoped.empty:
            continue

        y_vals: list[float] = []
        for event in scoped.itertuples(index=False):
            terminal_idx = int(event.terminal_idx_int)
            if event.fvg_side == "bull":
                y_vals.append(float(full_df.loc[terminal_idx, "high"]))
            else:
                y_vals.append(float(full_df.loc[terminal_idx, "low"]))

        scoped["marker_y"] = y_vals
        custom = np.column_stack(
            [
                scoped["fvg_event_id"].to_numpy(),
                scoped["fvg_side"].to_numpy(dtype=object),
                scoped["fvg_terminal_label"].to_numpy(dtype=object),
                scoped["fvg_r_final_outcome"].to_numpy(dtype=object),
                scoped["fvg_detect_idx"].to_numpy(),
                scoped["fvg_r_ever_touched"].to_numpy(),
                scoped["fvg_r_ever_partial_fill"].to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(
                    full_df.loc[scoped["terminal_idx_int"], "timestamp"].to_numpy(),
                    utc=True,
                ),
                y=scoped["marker_y"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=9,
                    symbol=symbol,
                    line=dict(color="white", width=0.8),
                ),
                customdata=custom,
                hovertemplate=(
                    f"{label}<br>"
                    "event_id=%{customdata[0]}<br>"
                    "side=%{customdata[1]}<br>"
                    "core_terminal=%{customdata[2]}<br>"
                    "research_final=%{customdata[3]}<br>"
                    "detect_idx=%{customdata[4]}<br>"
                    "ever_touched=%{customdata[5]}<br>"
                    "ever_partial_fill=%{customdata[6]}<extra></extra>"
                ),
                name=label,
            ),
            row=row,
            col=col,
        )


def _add_ifvg_active_overlays(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    required = {
        "ifvg_bull_active",
        "ifvg_bull_active_low",
        "ifvg_bull_active_high",
        "ifvg_bear_active",
        "ifvg_bear_active_low",
        "ifvg_bear_active_high",
    }
    if not required.issubset(window_df.columns):
        return

    delta = _bar_delta(window_df["timestamp"])
    for side, fill, line in (
        ("bull", "rgba(37, 99, 235, 0.20)", "#2563eb"),
        ("bear", "rgba(245, 158, 11, 0.20)", "#f59e0b"),
    ):
        scoped = window_df[
            pd.to_numeric(window_df[f"ifvg_{side}_active"], errors="coerce").fillna(0)
            == 1
        ].copy()
        if scoped.empty:
            continue

        for row_data in scoped.itertuples(index=False):
            x0 = pd.to_datetime(getattr(row_data, "timestamp"), utc=True)
            x1 = x0 + delta
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=float(getattr(row_data, f"ifvg_{side}_active_low")),
                y1=float(getattr(row_data, f"ifvg_{side}_active_high")),
                xref=f"x{'' if row == 1 else row}",
                yref=f"y{'' if row == 1 else row}",
                fillcolor=fill,
                line=dict(color=line, width=0.8),
                layer="below",
                row=row,
                col=col,
            )

        for state_value, state_label, symbol, marker_color in (
            (IFVG_STATE_ACTIVE_HOLDING, "Holding", "triangle-up", line),
            (IFVG_STATE_ACTIVE_FAILING, "Failing", "triangle-down", "#7c2d12"),
        ):
            state_rows = scoped[
                pd.to_numeric(scoped[f"ifvg_{side}_active_state"], errors="coerce")
                == state_value
            ].copy()
            if state_rows.empty:
                continue
            state_custom = state_rows[
                [
                    f"ifvg_{side}_active_id",
                    f"ifvg_{side}_active_state",
                    f"ifvg_{side}_active_test_count",
                    f"ifvg_{side}_active_hold_count",
                    f"ifvg_{side}_retest_depth_frac",
                    f"ifvg_{side}_active_age",
                    f"ifvg_{side}_active_age_decay",
                    f"ifvg_{side}_active_effective_significance",
                ]
            ].to_numpy(dtype=object)
            fig.add_trace(
                go.Scatter(
                    x=state_rows["timestamp"],
                    y=(
                        state_rows[f"ifvg_{side}_active_low"]
                        + state_rows[f"ifvg_{side}_active_high"]
                    )
                    / 2.0,
                    mode="markers",
                    marker=dict(size=8, color=marker_color, symbol=symbol),
                    customdata=state_custom,
                    hovertemplate=(
                        f"{side.title()} IFVG {state_label}<br>"
                        "ifvg_event_id=%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "test_count=%{customdata[2]}<br>"
                        "hold_count=%{customdata[3]}<br>"
                        "retest_depth_frac=%{customdata[4]}<br>"
                        "active_age=%{customdata[5]}<br>"
                        "active_age_decay=%{customdata[6]}<br>"
                        "active_effective_significance=%{customdata[7]}<extra></extra>"
                    ),
                    name=f"{side.title()} IFVG {state_label}",
                ),
                row=row,
                col=col,
            )


def _add_ifvg_event_markers(
    fig: go.Figure,
    *,
    window_df: pd.DataFrame,
    full_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    required = {
        "ifvg_bull_detect_flag",
        "ifvg_bear_detect_flag",
        "ifvg_bull_source_fvg_event_id",
        "ifvg_bear_source_fvg_event_id",
    }
    if not required.issubset(window_df.columns):
        return

    for side, color, symbol, y_col in (
        ("bull", "#2563eb", "diamond", "low"),
        ("bear", "#f59e0b", "diamond", "high"),
    ):
        detect_mask = (
            pd.to_numeric(
                window_df[f"ifvg_{side}_detect_flag"], errors="coerce"
            ).fillna(0)
            == 1
        )
        scoped = window_df[detect_mask].copy()
        if scoped.empty:
            continue

        custom = scoped[
            [
                f"ifvg_{side}_event_id",
                f"ifvg_{side}_detect_idx",
                f"ifvg_{side}_source_fvg_event_id",
                f"ifvg_{side}_source_fvg_detect_idx",
                f"ifvg_{side}_source_fill_pct_at_inversion",
                f"ifvg_{side}_inversion_score",
            ]
        ].to_numpy(dtype=object)
        fig.add_trace(
            go.Scatter(
                x=scoped["timestamp"],
                y=scoped[y_col],
                mode="markers",
                marker=dict(size=8, color=color, symbol=symbol),
                customdata=custom,
                hovertemplate=(
                    f"{side.title()} IFVG Detect<br>"
                    "ifvg_event_id=%{customdata[0]}<br>"
                    "ifvg_detect_idx=%{customdata[1]}<br>"
                    "source_fvg_event_id=%{customdata[2]}<br>"
                    "source_fvg_detect_idx=%{customdata[3]}<br>"
                    "source_fill_pct_at_inversion=%{customdata[4]}<br>"
                    "inversion_score=%{customdata[5]}<extra></extra>"
                ),
                name=f"{side.title()} IFVG Detect",
            ),
            row=row,
            col=col,
        )

        retest_mask = (
            pd.to_numeric(
                scoped[f"ifvg_{side}_first_retest_flag"], errors="coerce"
            ).fillna(0)
            == 1
        )
        if retest_mask.any():
            retests = scoped[retest_mask].copy()
            retest_custom = retests[
                [
                    f"ifvg_{side}_event_id",
                    f"ifvg_{side}_source_fvg_event_id",
                    f"ifvg_{side}_active_state",
                    f"ifvg_{side}_retest_depth_frac",
                ]
            ].to_numpy(dtype=object)
            fig.add_trace(
                go.Scatter(
                    x=retests["timestamp"],
                    y=retests["close"],
                    mode="markers",
                    marker=dict(size=9, color=color, symbol="star"),
                    customdata=retest_custom,
                    name=f"{side.title()} IFVG First Retest",
                    hovertemplate=(
                        f"{side.title()} IFVG First Retest<br>"
                        "ifvg_event_id=%{customdata[0]}<br>"
                        "source_fvg_event_id=%{customdata[1]}<br>"
                        "state=%{customdata[2]}<br>"
                        "retest_depth_frac=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        reclaimed_rows = window_df[
            pd.to_numeric(
                window_df[f"ifvg_{side}_zone_fully_reclaimed"], errors="coerce"
            ).fillna(0)
            == 1
        ].copy()
        if not reclaimed_rows.empty:
            reclaimed_custom = reclaimed_rows[
                [
                    f"ifvg_{side}_active_id",
                    f"ifvg_{side}_active_state",
                    f"ifvg_{side}_active_test_count",
                    f"ifvg_{side}_active_hold_count",
                ]
            ].to_numpy(dtype=object)
            fig.add_trace(
                go.Scatter(
                    x=reclaimed_rows["timestamp"],
                    y=reclaimed_rows["close"],
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="square"),
                    customdata=reclaimed_custom,
                    name=f"{side.title()} IFVG Fully Reclaimed",
                    hovertemplate=(
                        f"{side.title()} IFVG Fully Reclaimed<br>"
                        "ifvg_event_id=%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "test_count=%{customdata[2]}<br>"
                        "hold_count=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )


def make_fvg_figure(
    window_df: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    debug_tables: dict[str, pd.DataFrame],
    research_table: pd.DataFrame | None,
    title: str,
) -> go.Figure:
    window_start_idx = int(window_df.index.min())
    window_end_idx = int(window_df.index.max())
    rep_table = debug_tables["active_rep_table"]
    rep_table = rep_table[
        (rep_table["end_idx"] >= window_start_idx)
        & (rep_table["start_idx"] <= window_end_idx)
    ].copy()
    members = debug_tables["active_member_table"]
    left_boundary_selected = members[
        (members["row_idx"] == window_start_idx) & (members["selected"] == 1)
    ][["side", "active_id"]].drop_duplicates()
    carried_ids = set(left_boundary_selected["active_id"].tolist())
    rep_table = rep_table[
        rep_table["start_idx"].between(window_start_idx, window_end_idx)
        | rep_table["active_id"].isin(carried_ids)
    ].copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=window_df["timestamp"],
            open=window_df["open"],
            high=window_df["high"],
            low=window_df["low"],
            close=window_df["close"],
            name="Price",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )

    _add_rectangle_traces(
        fig,
        rep_table=rep_table,
        full_df=full_df,
        window_start_idx=window_start_idx,
        window_end_idx=window_end_idx,
        row=1,
        col=1,
    )
    _add_selected_remaining_overlays(
        fig,
        window_df=window_df,
        row=1,
        col=1,
    )
    _add_structural_selected_markers(
        fig,
        window_df=window_df,
        row=1,
        col=1,
    )
    _add_touch_episode_markers(
        fig,
        window_df=window_df,
        row=1,
        col=1,
    )
    _add_event_markers(
        fig,
        window_df=window_df,
        research_table=research_table,
        row=1,
        col=1,
    )
    _add_ifvg_active_overlays(
        fig,
        window_df=window_df,
        row=1,
        col=1,
    )
    _add_ifvg_event_markers(
        fig,
        window_df=window_df,
        full_df=full_df,
        row=1,
        col=1,
    )
    _add_terminal_event_markers(
        fig,
        window_df=window_df,
        full_df=full_df,
        research_table=research_table,
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=window_df["timestamp"],
            y=window_df["fvg_bull_active_count"],
            mode="lines",
            line=dict(color="#16a34a", width=2),
            name="Bull Active Count",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=window_df["timestamp"],
            y=-window_df["fvg_bear_active_count"],
            mode="lines",
            line=dict(color="#dc2626", width=2),
            name="Bear Active Count",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Active Count", row=2, col=1)
    fig.update_layout(
        title=title,
        height=980,
        width=1750,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=50, r=40, t=85, b=40),
    )
    return fig


def validate_fvg(
    window_df: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    debug_tables: dict[str, pd.DataFrame],
    outpath: str | Path,
    title: str = "FVG Validation",
    n_windows: int = 5,
    research_table: pd.DataFrame | None = None,
    old_no_expiry_research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    summary = summarize_fvg(
        window_df,
        full_df=full_df,
        debug_tables=debug_tables,
        research_table=research_table,
        old_no_expiry_research_table=old_no_expiry_research_table,
    )
    bull_windows = fvg_event_windows(
        window_df.reset_index(drop=True), side="bull", limit=n_windows
    )
    bear_windows = fvg_event_windows(
        window_df.reset_index(drop=True), side="bear", limit=n_windows
    )

    fig = make_fvg_figure(
        window_df,
        full_df=full_df,
        debug_tables=debug_tables,
        research_table=research_table,
        title=title,
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath)

    return {
        "summary": summary,
        "bull_windows": bull_windows,
        "bear_windows": bear_windows,
        "html_path": str(outpath.resolve()),
    }
