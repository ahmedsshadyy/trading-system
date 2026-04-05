from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.validation.common.chart_core import save_figure_html


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
    }


def _value_counts(series: pd.Series) -> dict[int | str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    out: dict[int | str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            out["NaN"] = int(value)
        elif isinstance(key, (np.integer, int)):
            out[int(key)] = int(value)
        else:
            out[str(key)] = int(value)
    return out


def _year_counts(series: pd.Series) -> dict[int, int]:
    ts = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if ts.empty:
        return {}
    counts = ts.dt.year.value_counts().sort_index()
    return {int(year): int(count) for year, count in counts.items()}


def _counts_by_lookback(
    series: pd.Series | None,
    mask: pd.Series | None = None,
) -> dict[int, int]:
    if series is None:
        return {}
    values = pd.to_numeric(series, errors="coerce")
    if mask is not None:
        values = values[mask.fillna(False)]
    values = values.dropna().astype(int)
    if values.empty:
        return {}
    counts = values.value_counts().sort_index()
    return {int(lookback): int(count) for lookback, count in counts.items()}


def _active_count_series(
    frame: pd.DataFrame, event_table: pd.DataFrame | None
) -> pd.Series:
    count = pd.Series(np.zeros(len(frame), dtype=np.int32), index=frame.index)
    if event_table is None or event_table.empty:
        return count
    for _, event in event_table.iterrows():
        confirm_idx = pd.to_numeric(
            pd.Series([event.get("confirm_idx")]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(confirm_idx):
            continue
        start = int(confirm_idx)
        end_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end = int(end_value) - 1 if np.isfinite(end_value) else len(frame) - 1
        if end < start:
            continue
        count.loc[start:end] += 1
    return count


def _overlap_rate(event_table: pd.DataFrame | None) -> float | None:
    if event_table is None or len(event_table) < 2:
        return 0.0 if event_table is not None else None
    pairs = 0
    overlaps = 0
    table = event_table.reset_index(drop=True)
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            pairs += 1
            low_i = float(table.loc[i, "low"])
            high_i = float(table.loc[i, "high"])
            low_j = float(table.loc[j, "low"])
            high_j = float(table.loc[j, "high"])
            if max(low_i, low_j) < min(high_i, high_j):
                overlaps += 1
    return overlaps / pairs if pairs else 0.0


def _regime_quality_table(
    event_table: pd.DataFrame | None,
) -> dict[int | str, dict[str, object]]:
    if (
        event_table is None
        or event_table.empty
        or "confirm_regime" not in event_table.columns
    ):
        return {}
    out: dict[int | str, dict[str, object]] = {}
    for key, group in event_table.groupby("confirm_regime", dropna=False):
        label: int | str
        if pd.isna(key):
            label = "NaN"
        else:
            label = int(key)
        out[label] = {
            "confirm_count": int(len(group)),
            "mean_duration_bars": (
                float(
                    pd.to_numeric(group["end_idx"], errors="coerce")
                    .sub(pd.to_numeric(group["confirm_idx"], errors="coerce"))
                    .dropna()
                    .mean()
                )
                if pd.to_numeric(group["end_idx"], errors="coerce").notna().any()
                else None
            ),
            "accepted_breakout_rate": float(
                pd.to_numeric(group["state"], errors="coerce").eq(4).mean()
            ),
            "mean_width_atr": float(
                pd.to_numeric(group["width_atr"], errors="coerce").dropna().mean()
            ),
            "mean_strength": float(
                pd.to_numeric(group["strength"], errors="coerce").dropna().mean()
            ),
        }
    return out


def summarize_range_boundaries(
    df: pd.DataFrame,
    *,
    event_table: pd.DataFrame | None = None,
    candidate_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    required = {
        "timestamp",
        "close",
        "range_detect_flag",
        "range_active",
        "range_state",
        "range_width_atr",
        "range_strength",
        "range_confirm_idx",
        "range_breakout_pending_flag",
        "range_accepted_breakout_flag",
        "range_expired_flag",
        "range_superseded_flag",
        "range_invalidated_flag",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing range boundary columns: {sorted(missing)}")

    out = _ensure_datetime(df)
    detect_rows = out[out["range_detect_flag"] == 1].copy()
    active_rows = out[out["range_active"] == 1].copy()
    active_counts = _active_count_series(out, event_table)

    durations = pd.Series(dtype=float)
    terminal_state_counts: dict[int | str, int] = {}
    if event_table is not None and not event_table.empty:
        confirm_idx = pd.to_numeric(event_table["confirm_idx"], errors="coerce")
        end_idx = pd.to_numeric(event_table["end_idx"], errors="coerce")
        durations = end_idx - confirm_idx
        terminal_state_counts = _value_counts(event_table["state"])

    first_confirm = (
        int(detect_rows["range_confirm_idx"].min())
        if not detect_rows["range_confirm_idx"].dropna().empty
        else None
    )
    pre_confirm_active = 0
    if first_confirm is not None:
        pre_confirm_active = int(out.loc[: first_confirm - 1, "range_active"].sum())

    checks = {
        "no_active_before_first_confirm": pre_confirm_active == 0,
        "no_same_bar_break_pending_on_confirm_rows": bool(
            (detect_rows["range_breakout_pending_flag"] == 0).all()
            if not detect_rows.empty
            else True
        ),
        "detect_rows_are_active": bool(
            (detect_rows["range_active"] == 1).all() if not detect_rows.empty else True
        ),
        "source_idx_matches_confirm_idx_on_active_rows": bool(
            (
                active_rows["range_upper_source_idx"]
                == active_rows["range_confirm_idx"]
            ).all()
            and (
                active_rows["range_lower_source_idx"]
                == active_rows["range_confirm_idx"]
            ).all()
            if not active_rows.empty
            else True
        ),
        "range_strength_in_unit_interval": bool(
            out["range_strength"].dropna().between(0.0, 1.0).all()
        ),
        "range_width_atr_positive_on_detect_rows": bool(
            (detect_rows["range_width_atr"] > 0).all()
            if not detect_rows.empty
            else True
        ),
        "no_flat_or_inverted_geometry_on_detect_rows": bool(
            (detect_rows["range_high"] > detect_rows["range_low"]).all()
            if not detect_rows.empty
            else True
        ),
    }

    regime_confirm_counts: dict[int | str, int] = {}
    if "regime" in out.columns and not detect_rows.empty:
        regime_confirm_counts = _value_counts(detect_rows["regime"])

    reclaim_stats: dict[str, object] = {}
    if event_table is not None and not event_table.empty:
        pending_counts = pd.to_numeric(
            event_table.get("break_pending_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        reclaimed_counts = pd.to_numeric(
            event_table.get("reclaimed_count", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        pending_bars = pd.to_numeric(
            event_table.get("total_pending_bars", pd.Series(dtype=float)),
            errors="coerce",
        )
        events_with_pending = pending_counts.gt(0)
        reclaim_stats = {
            "events_with_break_pending": int(events_with_pending.sum()),
            "reclaim_events": int(reclaimed_counts.gt(0).sum()),
            "reclaim_rate_given_break_pending": (
                float(reclaimed_counts.gt(0).loc[events_with_pending].mean())
                if events_with_pending.any()
                else None
            ),
            "mean_pending_duration_bars": (
                float(
                    pending_bars.loc[events_with_pending].dropna().sum()
                    / pending_counts.loc[events_with_pending].sum()
                )
                if events_with_pending.any()
                and pending_counts.loc[events_with_pending].sum() > 0
                else None
            ),
        }

    viability_diagnostics: dict[str, object] = {}
    promotion_funnel: dict[str, object] = {}
    if candidate_table is not None and not candidate_table.empty:
        rejected_before_maturity = candidate_table[
            pd.to_numeric(
                candidate_table.get("maturity_pass_flag", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        ].copy()
        viability_diagnostics = {
            "range_strength_viability": _continuous_stats(
                event_table["range_strength_viability"]
                if event_table is not None
                and "range_strength_viability" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "range_recent_pressure_imbalance": _continuous_stats(
                event_table["range_recent_pressure_imbalance"]
                if event_table is not None
                and "range_recent_pressure_imbalance" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "range_recent_equilibrium_score": _continuous_stats(
                event_table["range_recent_equilibrium_score"]
                if event_table is not None
                and "range_recent_equilibrium_score" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "range_recent_two_sided_freshness_score": _continuous_stats(
                event_table["range_recent_two_sided_freshness_score"]
                if event_table is not None
                and "range_recent_two_sided_freshness_score" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "expansion_veto_reject_count": int(
                pd.to_numeric(
                    candidate_table["expansion_veto_seen_flag"], errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "low_viability_reject_count": int(
                pd.to_numeric(
                    candidate_table["low_viability_reject_seen_flag"], errors="coerce"
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_fail_due_to_pressure": int(
                pd.to_numeric(
                    candidate_table.get(
                        "viability_fail_due_to_pressure", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_fail_due_to_equilibrium": int(
                pd.to_numeric(
                    candidate_table.get(
                        "viability_fail_due_to_equilibrium", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_fail_due_to_freshness": int(
                pd.to_numeric(
                    candidate_table.get(
                        "viability_fail_due_to_freshness", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_fail_due_to_score_threshold": int(
                pd.to_numeric(
                    candidate_table.get(
                        "viability_fail_due_to_score_threshold", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_fail_multiple_reasons": int(
                pd.to_numeric(
                    candidate_table.get(
                        "viability_fail_multiple_reasons", pd.Series(dtype=float)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .sum()
            ),
        }
        promotion_funnel = {
            "raw_candidate_count": int(
                pd.to_numeric(candidate_table["raw_candidate_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "maturity_pass_count": int(
                pd.to_numeric(candidate_table["maturity_pass_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "viability_pass_count": int(
                pd.to_numeric(candidate_table["viability_pass_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
                .sum()
            ),
            "confirmed_range_count": int(len(detect_rows)),
            "counts_by_candidate_lookback_bars": {
                "raw_candidates": _counts_by_lookback(
                    candidate_table.get("candidate_lookback_bars"),
                    pd.to_numeric(
                        candidate_table.get(
                            "raw_candidate_flag", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1),
                ),
                "maturity_pass": _counts_by_lookback(
                    candidate_table.get("candidate_lookback_bars"),
                    pd.to_numeric(
                        candidate_table.get(
                            "maturity_pass_flag", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1),
                ),
                "viability_pass": _counts_by_lookback(
                    candidate_table.get("candidate_lookback_bars"),
                    pd.to_numeric(
                        candidate_table.get(
                            "viability_pass_flag", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1),
                ),
                "confirmed": _counts_by_lookback(
                    (
                        event_table.get("candidate_lookback_bars")
                        if event_table is not None
                        else None
                    ),
                ),
            },
            "counts_per_calendar_year": {
                "raw_candidates": _year_counts(
                    candidate_table.get(
                        "birth_timestamp", pd.Series(dtype="datetime64[ns, UTC]")
                    )
                ),
                "maturity_pass": _year_counts(
                    candidate_table.get(
                        "maturity_pass_timestamp",
                        pd.Series(dtype="datetime64[ns, UTC]"),
                    )
                ),
                "viability_pass": _year_counts(
                    candidate_table.get(
                        "viability_pass_timestamp",
                        pd.Series(dtype="datetime64[ns, UTC]"),
                    )
                ),
                "confirmed": _year_counts(
                    candidate_table.get(
                        "range_confirm_timestamp",
                        pd.Series(dtype="datetime64[ns, UTC]"),
                    )
                ),
            },
            "maturity_rejection_breakdown": {
                "failed_dwell": int(
                    pd.to_numeric(
                        rejected_before_maturity.get(
                            "failed_dwell", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1)
                    .sum()
                ),
                "failed_boundary_stability": int(
                    pd.to_numeric(
                        rejected_before_maturity.get(
                            "failed_boundary_stability", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1)
                    .sum()
                ),
                "failed_same_lineage_continuation": int(
                    pd.to_numeric(
                        rejected_before_maturity.get(
                            "failed_same_lineage_continuation", pd.Series(dtype=float)
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1)
                    .sum()
                ),
                "failed_candidate_eligibility_before_maturity": int(
                    pd.to_numeric(
                        rejected_before_maturity.get(
                            "failed_candidate_eligibility_before_maturity",
                            pd.Series(dtype=float),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .eq(1)
                    .sum()
                ),
            },
        }

    summary = {
        "window": {
            "start": str(out["timestamp"].min()),
            "end": str(out["timestamp"].max()),
            "rows": int(len(out)),
        },
        "event_counts": {
            "confirmed_ranges": int(len(detect_rows)),
            "active_rows": int((out["range_active"] == 1).sum()),
            "breakout_pending_rows": int(
                (out["range_breakout_pending_flag"] == 1).sum()
            ),
            "accepted_breakout_rows": int(
                (out["range_accepted_breakout_flag"] == 1).sum()
            ),
            "expired_rows": int((out["range_expired_flag"] == 1).sum()),
            "superseded_rows": int((out["range_superseded_flag"] == 1).sum()),
            "invalidated_rows": int((out["range_invalidated_flag"] == 1).sum()),
        },
        "detect_distributions": {
            "width_atr": _continuous_stats(detect_rows["range_width_atr"]),
            "strength": _continuous_stats(detect_rows["range_strength"]),
            "close_inside_frac": _continuous_stats(
                detect_rows["range_close_inside_frac"]
            ),
            "touch_count_upper": _continuous_stats(
                detect_rows["range_touch_count_upper"]
            ),
            "touch_count_lower": _continuous_stats(
                detect_rows["range_touch_count_lower"]
            ),
            "boundary_stability_score": _continuous_stats(
                detect_rows["range_boundary_stability_score"]
            ),
        },
        "lifecycle": {
            "terminal_state_counts": terminal_state_counts,
            "duration_bars": _continuous_stats(durations),
            "bars_to_first_breach": _continuous_stats(
                event_table["bars_to_first_breach"]
                if event_table is not None
                and "bars_to_first_breach" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "bars_to_breakout_accept": _continuous_stats(
                event_table["bars_to_breakout_accept"]
                if event_table is not None
                and "bars_to_breakout_accept" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "breakout_side_first": _value_counts(
                event_table["breakout_side_first"]
                if event_table is not None
                and "breakout_side_first" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "overlap_rate": _overlap_rate(event_table),
        },
        "confirmation_timing": {
            "confirm_latency_bars": _continuous_stats(
                event_table["confirm_latency_bars"]
                if event_table is not None
                and "confirm_latency_bars" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "candidate_boundary_drift_abs": _continuous_stats(
                event_table["candidate_boundary_drift_abs"]
                if event_table is not None
                and "candidate_boundary_drift_abs" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "candidate_boundary_drift_atr": _continuous_stats(
                event_table["candidate_boundary_drift_atr"]
                if event_table is not None
                and "candidate_boundary_drift_atr" in event_table.columns
                else pd.Series(dtype=float)
            ),
        },
        "viability_diagnostics": viability_diagnostics,
        "promotion_funnel": promotion_funnel,
        "reclaim_behavior": reclaim_stats,
        "confirm_edge_position": {
            "close_position_in_range": _continuous_stats(
                event_table["confirm_close_position_in_range"]
                if event_table is not None
                and "confirm_close_position_in_range" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "distance_to_upper_atr": _continuous_stats(
                event_table["confirm_dist_to_upper_atr"]
                if event_table is not None
                and "confirm_dist_to_upper_atr" in event_table.columns
                else pd.Series(dtype=float)
            ),
            "distance_to_lower_atr": _continuous_stats(
                event_table["confirm_dist_to_lower_atr"]
                if event_table is not None
                and "confirm_dist_to_lower_atr" in event_table.columns
                else pd.Series(dtype=float)
            ),
        },
        "active_context": {
            "active_count_over_time": _continuous_stats(active_counts),
            "selected_age_bars": _continuous_stats(out["range_age_bars"]),
            "selected_strength": _continuous_stats(out["range_strength"]),
        },
        "regime_conditional_confirm_counts": regime_confirm_counts,
        "regime_conditional_quality": _regime_quality_table(event_table),
        "checks": checks,
    }
    return summary


def plot_range_boundaries_validation(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Range Boundary Validation",
) -> Path:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "range_detect_flag",
        "range_active",
        "range_high",
        "range_low",
        "range_strength",
        "range_state",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing range boundary plot columns: {sorted(missing)}")

    out = _ensure_datetime(df)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
        subplot_titles=("Price + Active Range Boundaries", "Range Strength / State"),
    )

    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
            increasing_line_color="#15803d",
            decreasing_line_color="#b45309",
        ),
        row=1,
        col=1,
    )

    active = out["range_active"] == 1
    if active.any():
        fig.add_trace(
            go.Scatter(
                x=out.loc[active, "timestamp"],
                y=out.loc[active, "range_high"],
                mode="lines",
                name="Active Range High",
                line=dict(color="#b91c1c", width=1.6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=out.loc[active, "timestamp"],
                y=out.loc[active, "range_low"],
                mode="lines",
                name="Active Range Low",
                line=dict(color="#15803d", width=1.6),
            ),
            row=1,
            col=1,
        )

    detect_rows = out[out["range_detect_flag"] == 1]
    if not detect_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=detect_rows["timestamp"],
                y=detect_rows["close"],
                mode="markers",
                name="Range Confirm",
                marker=dict(
                    symbol="diamond",
                    size=11,
                    color=detect_rows["range_strength"].fillna(0.0),
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    line=dict(color="#111827", width=0.8),
                ),
                customdata=detect_rows[
                    ["range_id", "range_width_atr", "range_strength"]
                ].to_numpy(),
                hovertemplate=(
                    "range_id=%{customdata[0]}<br>"
                    "width_atr=%{customdata[1]:.2f}<br>"
                    "strength=%{customdata[2]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    terminal_rows = out[
        (out["range_accepted_breakout_flag"] == 1)
        | (out["range_expired_flag"] == 1)
        | (out["range_superseded_flag"] == 1)
        | (out["range_invalidated_flag"] == 1)
    ]
    if not terminal_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=terminal_rows["timestamp"],
                y=terminal_rows["close"],
                mode="markers",
                name="Terminal",
                marker=dict(symbol="x", size=10, color="#7c2d12"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=out["timestamp"],
            y=out["range_strength"],
            mode="lines",
            name="Range Strength",
            line=dict(color="#1d4ed8", width=1.6),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=out["timestamp"],
            y=out["range_state"],
            mode="lines",
            name="Range State",
            line=dict(color="#7c3aed", width=1.2, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Strength / State", row=2, col=1)

    output = Path(outpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure_html(fig, output)
    return output


def validate_range_boundaries(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Range Boundary Validation",
    summary_df: pd.DataFrame | None = None,
    event_table: pd.DataFrame | None = None,
    candidate_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    html_path = plot_range_boundaries_validation(df, outpath=outpath, title=title)
    summary_source = summary_df if summary_df is not None else df
    summary = summarize_range_boundaries(
        summary_source,
        event_table=event_table,
        candidate_table=candidate_table,
    )
    return {"html_path": html_path, "summary": summary}
