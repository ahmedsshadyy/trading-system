"""
validation/indicators/sr_levels.py

Summary statistics and interactive HTML chart for the S/R level engine.
"""

from __future__ import annotations

from pathlib import Path

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
    SR_STATE_INVALIDATED,
    SR_STATE_RETIRED,
    SR_SIDE_RESISTANCE,
    SR_SIDE_SUPPORT,
    SRLevel,
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
    }
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_sr_levels(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
    *,
    live_df: pd.DataFrame | None = None,
) -> dict:
    """Compute validation summary statistics.

    Parameters
    ----------
    df : DataFrame
        Output of ``add_sr_levels(raw, include_research_only=True)``.
    registry : dict
        Registry returned by ``build_sr_level_registry`` after lifecycle run.
    live_df : DataFrame, optional
        Output of ``add_sr_levels(raw, include_research_only=False)``.
        Used for live/research parity check.
    """
    n = len(df)
    all_levels = list(registry.values())

    # --- source event counts by family ---
    by_family: dict[str, int] = {f: 0 for f in ALL_FAMILIES}
    for lev in all_levels:
        fam = lev.source_family
        by_family[fam] = by_family.get(fam, 0) + 1

    # --- state counts ---
    state_counts = {
        "active": sum(1 for l in all_levels if l.state == SR_STATE_ACTIVE),
        "invalidated": sum(1 for l in all_levels if l.state == SR_STATE_INVALIDATED),
        "retired": sum(1 for l in all_levels if l.state == SR_STATE_RETIRED),
        "inactive_pre_live": sum(1 for l in all_levels if l.state == 0),
    }

    # --- side breakdown ---
    sup_levels = [l for l in all_levels if l.side == SR_SIDE_SUPPORT]
    res_levels = [l for l in all_levels if l.side == SR_SIDE_RESISTANCE]

    # --- lifecycle stats ---
    lifecycle_stats = {
        "total_touches": sum(l.touch_count for l in all_levels),
        "total_weakens": sum(l.weaken_count for l in all_levels),
        "total_invalidations": sum(1 for l in all_levels if l.invalidation_idx >= 0),
        "total_retirements": sum(1 for l in all_levels if l.state == SR_STATE_RETIRED),
        "total_supersessions": sum(
            1 for l in all_levels if l.superseded_by is not None
        ),
    }

    # --- nearest support/resistance availability ---
    ns_avail = float(df["nearest_support_active"].sum()) / n if n else 0.0
    nr_avail = float(df["nearest_resistance_active"].sum()) / n if n else 0.0

    # --- distance distributions (ATR-normalised) ---
    ns_dist = pd.to_numeric(
        df["nearest_support_distance_atr"], errors="coerce"
    ).dropna()
    nr_dist = pd.to_numeric(
        df["nearest_resistance_distance_atr"], errors="coerce"
    ).dropna()

    def _dist_summary(series: pd.Series) -> dict:
        if series.empty:
            return {}
        return {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

    # --- strength distributions by side and family ---
    def _strength_by_family(levels: list[SRLevel]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for fam in ALL_FAMILIES:
            fam_levs = [l for l in levels if l.source_family == fam]
            if not fam_levs:
                continue
            strengths = [l.level_strength for l in fam_levs]
            out[fam] = {
                "count": len(fam_levs),
                "mean_strength": round(float(np.mean(strengths)), 4),
                "min_strength": round(float(np.min(strengths)), 4),
                "max_strength": round(float(np.max(strengths)), 4),
            }
        return out

    # --- parity check ---
    parity_ok: bool | str = "skipped"
    if live_df is not None:
        live_cols = [c for c in df.columns if not c.startswith("r_")]
        shared = [c for c in live_cols if c in live_df.columns]
        try:
            parity_ok = bool(df[shared].equals(live_df[shared]))
        except Exception:
            parity_ok = "error"

    # --- contamination check ---
    bad_cols = [
        c for c in df.columns if c.startswith("label_") or c.startswith("future_")
    ]
    contamination_clean = len(bad_cols) == 0

    return {
        "row_count": n,
        "total_source_events": len(all_levels),
        "source_events_by_family": by_family,
        "state_counts": state_counts,
        "support_level_count": len(sup_levels),
        "resistance_level_count": len(res_levels),
        "lifecycle_stats": lifecycle_stats,
        "nearest_support_availability_rate": round(ns_avail, 4),
        "nearest_resistance_availability_rate": round(nr_avail, 4),
        "support_distance_atr": _dist_summary(ns_dist),
        "resistance_distance_atr": _dist_summary(nr_dist),
        "strength_by_side": {
            "support": _strength_by_family(sup_levels),
            "resistance": _strength_by_family(res_levels),
        },
        "live_research_parity_ok": parity_ok,
        "contamination_clean": contamination_clean,
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_FAMILY_COLORS: dict[str, str] = {
    SR_FAMILY_SWING: "rgba(0, 100, 220, {a})",
    SR_FAMILY_EQHL: "rgba(170, 0, 220, {a})",
    SR_FAMILY_SESSION: "rgba(0, 180, 140, {a})",
    SR_FAMILY_DAY: "rgba(230, 140, 0, {a})",
    SR_FAMILY_WEEK: "rgba(200, 0, 0, {a})",
    SR_FAMILY_VP: "rgba(100, 100, 100, {a})",
}

_FAMILY_DASH: dict[str, str] = {
    SR_FAMILY_SWING: "solid",
    SR_FAMILY_EQHL: "dot",
    SR_FAMILY_SESSION: "dash",
    SR_FAMILY_DAY: "dashdot",
    SR_FAMILY_WEEK: "longdash",
    SR_FAMILY_VP: "longdashdot",
}


def _level_color(lev: SRLevel, alpha: float = 0.85) -> str:
    template = _FAMILY_COLORS.get(lev.source_family, "rgba(120,120,120,{a})")
    return template.format(a=alpha)


def _add_level_line(
    fig: go.Figure,
    df: pd.DataFrame,
    lev: SRLevel,
    *,
    x_col: str = "timestamp",
) -> None:
    """Draw a horizontal level line from its live-from row to retirement/end."""
    n = len(df)
    live_idx = max(0, lev.source_live_from_idx)
    end_idx = n - 1
    if (
        lev.state in (SR_STATE_INVALIDATED, SR_STATE_RETIRED)
        and lev.invalidation_idx >= 0
    ):
        end_idx = lev.invalidation_idx

    x_start = df[x_col].iloc[min(live_idx, n - 1)]
    x_end = df[x_col].iloc[min(end_idx, n - 1)]
    strength = max(0.3, min(1.0, lev.level_strength))
    alpha = 0.35 + 0.55 * strength
    color = _level_color(lev, alpha)
    dash = _FAMILY_DASH.get(lev.source_family, "solid")
    side_label = "SUP" if lev.side == SR_SIDE_SUPPORT else "RES"
    label = f"{side_label} {lev.source_family} lv{lev.level_id} str={lev.level_strength:.2f}"

    fig.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[lev.level_price, lev.level_price],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.2, dash=dash),
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"price={lev.level_price:.5f}<br>"
                f"age={lev.age_bars} bars, touches={lev.touch_count}, "
                f"weakens={lev.weaken_count}<extra></extra>"
            ),
        )
    )


def _add_invalidation_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
    *,
    x_col: str = "timestamp",
) -> None:
    n = len(df)
    for lev in registry.values():
        if lev.invalidation_idx < 0 or lev.invalidation_idx >= n:
            continue
        x_val = df[x_col].iloc[lev.invalidation_idx]
        y_val = lev.level_price
        symbol = "x" if lev.side == SR_SIDE_SUPPORT else "x-open"
        color = "red" if lev.side == SR_SIDE_SUPPORT else "orange"
        fig.add_trace(
            go.Scatter(
                x=[x_val],
                y=[y_val],
                mode="markers",
                name=f"Inv lv{lev.level_id}",
                marker=dict(symbol=symbol, size=8, color=color),
                showlegend=False,
                hovertemplate=(
                    f"<b>Invalidated lv{lev.level_id}</b><br>"
                    f"{lev.source_family} {lev.source_sub}<br>"
                    f"at bar {lev.invalidation_idx}<extra></extra>"
                ),
            )
        )


def _add_nearest_sr_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    x_col: str = "timestamp",
) -> None:
    for col, color, name in (
        ("nearest_support_price", "rgba(0,140,255,0.5)", "Nearest Sup"),
        ("nearest_resistance_price", "rgba(255,80,0,0.5)", "Nearest Res"),
    ):
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=series,
                mode="lines",
                name=name,
                line=dict(color=color, width=2, dash="dot"),
            )
        )


# ---------------------------------------------------------------------------
# Main validate entry-point
# ---------------------------------------------------------------------------


def validate_sr_levels(
    plot_df: pd.DataFrame,
    *,
    registry: dict[int, SRLevel],
    summary_df: pd.DataFrame,
    live_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "S/R Level Validation",
    max_levels_in_chart: int = 120,
) -> dict:
    """Produce summary dict and optional HTML chart.

    Parameters
    ----------
    plot_df : DataFrame
        Slice used for the chart (e.g. ``.tail(300)``).
    registry : dict
        Registry after lifecycle run.
    summary_df : DataFrame
        Full dataset for computing aggregate statistics.
    live_df : DataFrame, optional
        Live-build output for parity check.
    outpath : path, optional
        Save interactive HTML here if provided.
    title : str
        Chart title.
    max_levels_in_chart : int
        Cap on number of level lines to draw (performance guard).

    Returns
    -------
    dict  Summary statistics.
    """
    summary = summarize_sr_levels(summary_df, registry, live_df=live_df)

    if outpath is not None:
        fig = create_candlestick_figure(plot_df, title=title)

        # Draw level lines (capped for perf)
        n_plot = len(plot_df)
        visible_levels = [
            lev
            for lev in registry.values()
            if (lev.source_live_from_idx < n_plot or (lev.state == SR_STATE_ACTIVE))
        ]
        visible_levels.sort(key=lambda l: l.level_strength, reverse=True)
        for lev in visible_levels[:max_levels_in_chart]:
            _add_level_line(fig, plot_df, lev)

        _add_invalidation_markers(fig, plot_df, registry)
        _add_nearest_sr_lines(fig, plot_df)

        # Mark break bars
        for col, color, sym in (
            ("support_broken_this_bar", "red", "triangle-down"),
            ("resistance_broken_this_bar", "orange", "triangle-up"),
        ):
            if col in plot_df.columns:
                mask = plot_df[col] == 1
                sub = plot_df[mask]
                if not sub.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["timestamp"],
                            y=sub["close"],
                            mode="markers",
                            name=col,
                            marker=dict(symbol=sym, size=10, color=color),
                        )
                    )

        save_figure_html(fig, outpath)
        summary["chart_path"] = str(outpath)

    return summary
