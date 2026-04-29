"""
scripts/validate_sr_levels.py

Interactive validation chart and benchmark summary for the S/R zone engine.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dag_runtime import GraphRunContext, execute_graph
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.indicators.foundation.sr_levels import (
    SR_SIDE_SUPPORT,
    SR_STATE_ACTIVE,
    SR_STATE_ACTIVE_WEAKENED,
    SR_STATE_BREAK_PENDING,
)
from src.validation.common import cleanup_validation_artifacts

DATE_FROM = "2026-01-10"
OUT_DIR = Path("notebooks/foundation")
CACHE_ROOT = Path("data/validation_cache")
VALIDATOR_NAME = "validate_sr_levels"
RUNS = (("XAU_USD", "H4"),)

_SUP_LINE = "rgba(80, 145, 230, 0.34)"
_RES_LINE = "rgba(224, 112, 89, 0.34)"
_SUP_FILL = "rgba(80, 145, 230, 0.10)"
_RES_FILL = "rgba(224, 112, 89, 0.10)"
_PRIMARY_SUP_LINE = "rgba(0, 109, 119, 0.96)"
_PRIMARY_SUP_FILL = "rgba(0, 109, 119, 0.18)"
_PRIMARY_RES_LINE = "rgba(188, 108, 37, 0.96)"
_PRIMARY_RES_FILL = "rgba(188, 108, 37, 0.18)"
_NEAREST_SUP_LINE = "rgba(38, 70, 83, 0.95)"
_NEAREST_RES_LINE = "rgba(84, 84, 84, 0.95)"


def _state_label(value: float) -> str:
    if not pd.notna(value):
        return "none"
    state = int(value)
    if state == SR_STATE_ACTIVE:
        return "active"
    if state == SR_STATE_ACTIVE_WEAKENED:
        return "weakened"
    if state == SR_STATE_BREAK_PENDING:
        return "break_pending"
    return f"state_{state}"


# Structural-quality filter — only zones that look like real S/R get rendered.
# A zone passes if it satisfies BOTH:
#   (a) level_strength >= _STRUCTURAL_SCORE_MIN, and
#   (b) any one of: anchor_count >= 2 (confluence), family_count >= 2
#       (cross-family confluence), high-info family (eqhl/swing), or
#       touch_count >= _STRUCTURAL_MIN_TOUCHES (proven by interaction).
_STRUCTURAL_SCORE_MIN: float = 0.45
_STRUCTURAL_MIN_TOUCHES: int = 2
_STRUCTURAL_HIGH_INFO_FAMILIES: frozenset[str] = frozenset({"eqhl", "swing"})

# Per-side render budget on the chart. Was 250 before structural filtering,
# now ~50 because the filter throws out single-anchor wick zones.
_ZONE_RENDER_BUDGET_PER_SIDE = 60


def _is_structural_zone(level) -> bool:
    if not getattr(level, "emitted_zone_flag", False):
        return False
    if float(getattr(level, "level_strength", 0.0)) < _STRUCTURAL_SCORE_MIN:
        return False
    family = getattr(level, "best_source_family", "") or getattr(
        level, "source_family", ""
    )
    if int(getattr(level, "anchor_count", 1)) >= 2:
        return True
    if int(getattr(level, "family_count", 1)) >= 2:
        return True
    if family in _STRUCTURAL_HIGH_INFO_FAMILIES:
        return True
    if int(getattr(level, "touch_count", 0)) >= _STRUCTURAL_MIN_TOUCHES:
        return True
    return False


def _zone_rect_color(side: int, strength: float) -> tuple[str, str]:
    """Return (line_color, fill_color) — opacity scales with level_strength."""
    s = float(np.clip(strength, 0.0, 1.0))
    fill_alpha = 0.08 + 0.22 * s
    line_alpha = 0.45 + 0.45 * s
    if side == SR_SIDE_SUPPORT:
        return (
            f"rgba(33, 113, 181, {line_alpha:.3f})",
            f"rgba(33, 113, 181, {fill_alpha:.3f})",
        )
    return (
        f"rgba(203, 24, 29, {line_alpha:.3f})",
        f"rgba(203, 24, 29, {fill_alpha:.3f})",
    )


def _terminal_label(reason: str | None, state: int) -> str:
    if reason:
        return reason
    if state == SR_STATE_ACTIVE:
        return "active"
    if state == SR_STATE_ACTIVE_WEAKENED:
        return "weakened"
    if state == SR_STATE_BREAK_PENDING:
        return "break_pending"
    return "unknown"


def _zone_lifecycle_bounds(
    level, x_values: pd.Series, plot_start, plot_end
) -> tuple[object, object] | None:
    live_from = getattr(level, "source_live_from_idx", -1)
    if live_from < 0 or live_from >= len(x_values):
        return None
    x0 = x_values.iloc[live_from]
    end_candidates = [
        idx
        for idx in (
            getattr(level, "invalidation_idx", -1),
            getattr(level, "expiry_idx", -1),
            getattr(level, "retirement_idx", -1),
        )
        if isinstance(idx, int) and 0 <= idx < len(x_values)
    ]
    x1 = x_values.iloc[min(end_candidates)] if end_candidates else plot_end
    if x1 < plot_start or x0 > plot_end:
        return None
    return (max(x0, plot_start), min(x1, plot_end))


def _select_visible_zones(
    registry: dict, x_values: pd.Series, plot_start, plot_end
) -> tuple[list, list]:
    """Return (support_zones, resistance_zones), each a list of (level, x0, x1).

    Selection: zones whose [live_from, terminal_or_end] intersects [plot_start, plot_end]
    AND pass _is_structural_zone (multi-anchor / multi-family / high-info family /
    touched at least N times). Ranked by level_strength descending, clipped per
    side to the render budget.
    """
    sup, res = [], []
    for level in registry.values():
        if not _is_structural_zone(level):
            continue
        bounds = _zone_lifecycle_bounds(level, x_values, plot_start, plot_end)
        if bounds is None:
            continue
        x0, x1 = bounds
        target = sup if level.side == SR_SIDE_SUPPORT else res
        target.append((level, x0, x1))
    sup.sort(key=lambda item: float(item[0].level_strength), reverse=True)
    res.sort(key=lambda item: float(item[0].level_strength), reverse=True)
    return sup[:_ZONE_RENDER_BUDGET_PER_SIDE], res[:_ZONE_RENDER_BUDGET_PER_SIDE]


def _add_zone_lifecycle_shapes(
    fig: go.Figure,
    zones: list,
    *,
    side_label: str,
    row: int,
    col: int,
) -> int:
    rendered = 0
    legend_emitted = False
    for level, x0, x1 in zones:
        line_color, fill_color = _zone_rect_color(level.side, level.level_strength)
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=float(level.zone_low),
            y1=float(level.zone_high),
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            row=row,
            col=col,
            layer="below",
        )
        family = getattr(level, "best_source_family", None) or getattr(
            level, "source_family", ""
        )
        terminal = _terminal_label(
            getattr(level, "terminal_reason", "") or None,
            int(getattr(level, "state", 0)),
        )
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[float(level.level_price)],
                mode="markers",
                marker=dict(
                    size=5,
                    color=line_color,
                    symbol="line-ns-open",
                    line=dict(width=0),
                ),
                name=f"{side_label} zone",
                showlegend=not legend_emitted,
                legendgroup=f"{side_label.lower()}_zone",
                hovertemplate=(
                    f"<b>{side_label} zone {int(level.level_id)}</b><br>"
                    f"family={family}<br>"
                    f"score={float(level.level_strength):.3f}<br>"
                    f"anchors={int(level.anchor_count)}<br>"
                    f"width_atr={float(level.zone_width_atr):.3f}<br>"
                    f"state={terminal}<br>"
                    f"price=[{float(level.zone_low):.4f}, {float(level.zone_high):.4f}]"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        legend_emitted = True
        rendered += 1
    return rendered


def _add_zone_terminal_markers(
    fig: go.Figure,
    zones: list,
    x_values: pd.Series,
    *,
    plot_start,
    plot_end,
    row: int,
    col: int,
) -> None:
    inval_x: list = []
    inval_y: list = []
    inval_text: list = []
    expiry_x: list = []
    expiry_y: list = []
    expiry_text: list = []
    for level, _x0, _x1 in zones:
        inv = getattr(level, "invalidation_idx", -1)
        if isinstance(inv, int) and 0 <= inv < len(x_values):
            ts = x_values.iloc[inv]
            if plot_start <= ts <= plot_end:
                inval_x.append(ts)
                inval_y.append(float(level.level_price))
                inval_text.append(f"invalidated z{int(level.level_id)}")
        exp = getattr(level, "expiry_idx", -1)
        if isinstance(exp, int) and 0 <= exp < len(x_values):
            ts = x_values.iloc[exp]
            if plot_start <= ts <= plot_end:
                expiry_x.append(ts)
                expiry_y.append(float(level.level_price))
                expiry_text.append(f"expired z{int(level.level_id)}")
    if inval_x:
        fig.add_trace(
            go.Scatter(
                x=inval_x,
                y=inval_y,
                mode="markers",
                marker=dict(symbol="x", size=9, color="rgba(120, 30, 30, 0.95)"),
                name="zone invalidation",
                hovertext=inval_text,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )
    if expiry_x:
        fig.add_trace(
            go.Scatter(
                x=expiry_x,
                y=expiry_y,
                mode="markers",
                marker=dict(symbol="square-x", size=9, color="rgba(60, 60, 60, 0.85)"),
                name="zone hard expiry",
                hovertext=expiry_text,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )


def _add_zone_touch_markers(
    fig: go.Figure,
    audit: pd.DataFrame | None,
    visible_zone_ids: set[int],
    full_ts: pd.Series,
    plot_start,
    plot_end,
    *,
    row: int,
    col: int,
) -> None:
    if audit is None or audit.empty or "row" not in audit.columns:
        return
    aud = audit.copy()
    if "zone_id" in aud.columns and visible_zone_ids:
        aud = aud[aud["zone_id"].astype(int).isin(visible_zone_ids)]
    aud = aud.dropna(subset=["row", "touch_price"])
    if aud.empty:
        return
    aud["row"] = aud["row"].astype(int)
    aud = aud[(aud["row"] >= 0) & (aud["row"] < len(full_ts))]
    aud = aud.assign(
        ts=pd.to_datetime(full_ts.to_numpy()[aud["row"].to_numpy()], utc=True)
    )
    aud = aud[(aud["ts"] >= plot_start) & (aud["ts"] <= plot_end)]
    if aud.empty:
        return
    style_map = {
        "clean touch": ("circle", "rgba(20, 90, 50, 0.85)"),
        "weak pierce": ("triangle-down", "rgba(180, 130, 0, 0.85)"),
        "reclaim-after-break-pending": ("star", "rgba(0, 130, 130, 0.95)"),
    }
    for touch_type, sub in aud.groupby("touch_type"):
        symbol, color = style_map.get(touch_type, ("x", "rgba(60, 60, 60, 0.6)"))
        fig.add_trace(
            go.Scatter(
                x=sub["ts"],
                y=pd.to_numeric(sub["touch_price"], errors="coerce"),
                mode="markers",
                name=f"touch: {touch_type}",
                marker=dict(symbol=symbol, size=6, color=color),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "zone=%{customdata[1]}<br>"
                    "score=%{customdata[2]:.3f}<br>"
                    "price=%{y:.4f}<extra></extra>"
                ),
                customdata=sub[["touch_type", "zone_id", "score"]].to_numpy(),
            ),
            row=row,
            col=col,
        )


def _add_primary_zone_band(
    fig: go.Figure,
    plot: pd.DataFrame,
    *,
    side: str,
    row: int,
    col: int,
) -> None:
    """Render the per-bar primary support/resistance band.

    The primary zone is the engine's "main S/R right now" pick — chosen from
    the active stack in update_sr_lifecycle. Drawn as a thick distinguished
    band sitting on top of the contextual zone rectangles.
    """
    low_col = f"primary_{side}_zone_low"
    high_col = f"primary_{side}_zone_high"
    score_col = f"primary_{side}_zone_score"
    if low_col not in plot.columns or high_col not in plot.columns:
        return
    low = pd.to_numeric(plot[low_col], errors="coerce")
    high = pd.to_numeric(plot[high_col], errors="coerce")
    if not bool((low.notna() & high.notna()).any()):
        return
    score = (
        pd.to_numeric(plot[score_col], errors="coerce")
        if score_col in plot.columns
        else None
    )
    if side == "support":
        line_color = "rgba(0, 109, 119, 0.96)"
        fill_color = "rgba(0, 109, 119, 0.22)"
    else:
        line_color = "rgba(188, 70, 37, 0.96)"
        fill_color = "rgba(188, 70, 37, 0.22)"
    fig.add_trace(
        go.Scatter(
            x=plot["timestamp"],
            y=high,
            mode="lines",
            line=dict(color=line_color, width=2.6),
            name=f"Primary {side} (top)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=plot["timestamp"],
            y=low,
            mode="lines",
            line=dict(color=line_color, width=2.6),
            fill="tonexty",
            fillcolor=fill_color,
            name=f"Primary {side.title()}",
            customdata=score if score is not None else None,
            hovertemplate=(
                f"<b>Primary {side}</b><br>"
                "low=%{y:.4f}<br>"
                + (
                    "score=%{customdata:.3f}<extra></extra>"
                    if score is not None
                    else "<extra></extra>"
                )
            ),
        ),
        row=row,
        col=col,
    )


def _build_sr_chart(
    enriched: pd.DataFrame,
    full_out: pd.DataFrame,
    registry: dict,
    *,
    title: str,
    date_from: str = DATE_FROM,
    audit: pd.DataFrame | None = None,
) -> go.Figure:
    all_ts = pd.to_datetime(
        full_out["timestamp"] if "timestamp" in full_out.columns else full_out.index,
        utc=True,
        errors="coerce",
    ).reset_index(drop=True)
    start_ts = pd.Timestamp(date_from, tz="UTC")
    plot = full_out.loc[all_ts >= start_ts].copy()
    if plot.empty:
        raise ValueError(f"No data on or after {date_from}")
    plot_ts = pd.to_datetime(plot["timestamp"], utc=True, errors="coerce").reset_index(
        drop=True
    )
    plot_start = plot_ts.iloc[0]
    plot_end = plot_ts.iloc[-1]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.46, 0.135, 0.135, 0.135, 0.135],
        subplot_titles=[
            "Price | TEAL/ORANGE band = primary S/R | thin rectangles = structural deeper levels",
            "Distance to nearest zone midpoint (ATR)",
            "Nearest zone strength",
            "Active zone count",
            "Primary zone width (ATR)",
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=plot["timestamp"],
            open=plot["open"],
            high=plot["high"],
            low=plot["low"],
            close=plot["close"],
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Layer order on the price subplot:
    #   below: zone rectangles (structural deeper levels)
    #   above: primary support / resistance band (the "main S/R right now")
    #   on top: candles (rendered first, but plotly draws candles above shapes)
    sup_zones, res_zones = _select_visible_zones(registry, all_ts, plot_start, plot_end)
    sup_rendered = _add_zone_lifecycle_shapes(
        fig, sup_zones, side_label="Support", row=1, col=1
    )
    res_rendered = _add_zone_lifecycle_shapes(
        fig, res_zones, side_label="Resistance", row=1, col=1
    )
    _add_primary_zone_band(fig, plot, side="support", row=1, col=1)
    _add_primary_zone_band(fig, plot, side="resistance", row=1, col=1)
    visible_zone_ids = {int(level.level_id) for level, _, _ in sup_zones + res_zones}
    _add_zone_terminal_markers(
        fig,
        sup_zones + res_zones,
        all_ts,
        plot_start=plot_start,
        plot_end=plot_end,
        row=1,
        col=1,
    )
    _add_zone_touch_markers(
        fig,
        audit,
        visible_zone_ids,
        all_ts,
        plot_start,
        plot_end,
        row=1,
        col=1,
    )

    for col, color, symbol, name in (
        (
            "support_broken_this_bar",
            "rgba(33, 113, 181, 0.95)",
            "triangle-down",
            "Support Break",
        ),
        (
            "resistance_broken_this_bar",
            "rgba(203, 24, 29, 0.95)",
            "triangle-up",
            "Resistance Break",
        ),
        ("sr_reclaim_this_bar_flag", "rgba(20,140,20,0.9)", "diamond", "Reclaim"),
    ):
        if col not in plot.columns:
            continue
        mask = pd.to_numeric(plot[col], errors="coerce").fillna(0).eq(1)
        if not bool(mask.any()):
            continue
        fig.add_trace(
            go.Scatter(
                x=plot.loc[mask, "timestamp"],
                y=plot.loc[mask, "close"],
                mode="markers",
                name=name,
                marker=dict(symbol=symbol, size=10, color=color),
            ),
            row=1,
            col=1,
        )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.005,
        y=0.995,
        text=(
            "Teal/Orange band = PRIMARY S/R per bar | "
            f"{sup_rendered} support / {res_rendered} resistance structural zones "
            f"(score≥{_STRUCTURAL_SCORE_MIN}, multi-anchor or eqhl/swing or touched, "
            f"capped at {_ZONE_RENDER_BUDGET_PER_SIDE}/side). "
            "Hover for family/score/anchors/state."
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(90,90,90,0.25)",
        borderwidth=1,
        font=dict(size=11, color="rgba(45,55,72,0.95)"),
    )

    for row, series in (
        (
            2,
            (
                ("nearest_support_distance_atr", _SUP_LINE, "Support Distance"),
                ("nearest_resistance_distance_atr", _RES_LINE, "Resistance Distance"),
            ),
        ),
        (
            3,
            (
                ("nearest_support_strength", _SUP_LINE, "Support Strength"),
                ("nearest_resistance_strength", _RES_LINE, "Resistance Strength"),
            ),
        ),
        (
            4,
            (
                ("active_support_count", _SUP_LINE, "Active Support Zones"),
                ("active_resistance_count", _RES_LINE, "Active Resistance Zones"),
            ),
        ),
        (
            5,
            (
                ("primary_support_zone_width_atr", _SUP_LINE, "Primary Support Width"),
                (
                    "primary_resistance_zone_width_atr",
                    _RES_LINE,
                    "Primary Resistance Width",
                ),
            ),
        ),
    ):
        for col, color, name in series:
            if col not in plot.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=plot["timestamp"],
                    y=pd.to_numeric(plot[col], errors="coerce"),
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.8),
                ),
                row=row,
                col=1,
            )

    fig.add_hline(
        y=0.15,
        row=2,
        col=1,
        line_dash="dot",
        line_color="rgba(100,100,100,0.45)",
        annotation_text="touch zone",
    )
    fig.add_hline(
        y=0.5,
        row=3,
        col=1,
        line_dash="dot",
        line_color="rgba(100,100,100,0.45)",
        annotation_text="mid score",
    )

    fig.update_layout(
        title=f"{title} | lifecycle view",
        template="plotly_white",
        hovermode="x unified",
        height=1650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Score", row=3, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Count", row=4, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Width ATR", row=5, col=1, rangemode="tozero")
    return fig


def _print_summary(d: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print_summary(v, indent=indent + 2)
            else:
                print(f"{prefix}{k}: {v}")
    else:
        print(f"{prefix}{d}")


def _print_benchmark(summary: dict) -> None:
    funnel = summary.get("source_funnel", {})
    calibration = summary.get("score_calibration", {})
    structure_status = summary.get("structure_status", {})
    score_status = summary.get("score_status", {})
    print("\nbenchmark:")
    print(
        "  headline:"
        f" structure={structure_status.get('label')}"
        f" score={score_status.get('label')}"
        f" (high_info={score_status.get('predictive_high_info')},"
        f" pooled={score_status.get('predictive_pooled')})"
    )
    print(f"  raw_source_count: {funnel.get('raw_source_count')}")
    print(f"  absorbed_source_count: {funnel.get('absorbed_source_count')}")
    print(f"  emitted_zone_count: {funnel.get('emitted_zone_count')}")
    print(f"  active_zone_count: {funnel.get('active_zone_count')}")
    print(
        "  nearest_availability:"
        f" support={summary.get('nearest_support_availability_rate')}"
        f" resistance={summary.get('nearest_resistance_availability_rate')}"
    )
    print("  pooled_drift_monotonicity:" f" {calibration.get('monotonicity', {})}")
    print(
        "  pooled_drift_top_vs_bottom_delta:"
        f" {calibration.get('top_vs_bottom_delta', {})}"
    )

    # Per-family signed-outcome deltas (the gold-standard metric)
    by_family = calibration.get("by_source_family", {})
    print("  per_family_signed_outcome_top_vs_bottom_delta_h8:")
    for family in ("eqhl", "swing", "session", "day", "week", "vp"):
        block = by_family.get(family, {}) if isinstance(by_family, dict) else {}
        signed = (block.get("top_vs_bottom_delta_by_metric") or {}).get("signed", {})
        held = (block.get("top_vs_bottom_delta_by_metric") or {}).get("held", {})
        delta_signed = signed.get("8") if isinstance(signed, dict) else None
        delta_held = held.get("8") if isinstance(held, dict) else None
        n = block.get("touch_count", 0) if isinstance(block, dict) else 0
        signed_str = (
            f"{delta_signed:+.3f}" if isinstance(delta_signed, (int, float)) else "n/a"
        )
        held_str = (
            f"{delta_held:+.3f}" if isinstance(delta_held, (int, float)) else "n/a"
        )
        print(f"    {family:8s} n={n:>5}  signed={signed_str}  held={held_str}")

    # High-info gate detail
    high_info = score_status.get("high_info", {}) or {}
    family_results = (
        high_info.get("by_family", {}) if isinstance(high_info, dict) else {}
    )
    print(
        f"  high_info_gate (signed_delta_h8 OR held_delta_h8 >= "
        f"{high_info.get('min_delta_8')} on n>={high_info.get('min_touch_count')}):"
    )
    for fam, item in family_results.items():
        signed = item.get("signed_delta_8")
        held = item.get("held_delta_8")
        ss = f"{signed:+.3f}" if isinstance(signed, (int, float)) else "  n/a"
        hs = f"{held:+.3f}" if isinstance(held, (int, float)) else "  n/a"
        passes = "PASS" if item.get("passes") else "fail"
        print(
            f"    {fam:8s} n={item.get('touch_count', 0):>5}"
            f"  signed={ss}  held={hs} -> {passes}"
        )

    # Component audit — which subscore carries signal?
    component_audit = calibration.get("component_audit", {})
    if component_audit:
        print("  component_audit (rank_corr to outcome_8 for raw drift):")
        for component, payload in component_audit.items():
            rank_corr = (
                payload.get("rank_corr", {}) if isinstance(payload, dict) else {}
            )
            r8 = rank_corr.get("8")
            print(
                f"    {component:24s} rank_corr_8=" f"{r8:+.3f}"
                if isinstance(r8, (int, float))
                else f"    {component:24s} rank_corr_8=n/a"
            )


def _print_audit(summary: dict) -> None:
    calibration = summary.get("score_calibration", {})
    diagnostics = summary.get("diagnostics", {})
    primary = diagnostics.get("primary_selection", {})
    width_bucket = calibration.get("by_width_bucket", {})
    print("\naudit:")
    print(f"  score_monotonicity: {calibration.get('monotonicity', {})}")
    print(f"  top_vs_bottom_delta: {calibration.get('top_vs_bottom_delta', {})}")
    print("  family_preservation:")
    _print_summary(diagnostics.get("absorption_diagnostics", {}), indent=4)
    print("  width_bucket_calibration:")
    _print_summary(width_bucket, indent=4)
    print("  primary_vs_nearest:")
    _print_summary(primary, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate S/R zones and optionally produce an HTML chart."
    )
    parser.add_argument("--html", action="store_true", help="Generate the HTML chart.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--date-from", type=str, default=DATE_FROM)
    parser.add_argument("--last-days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--cleanup-stale", action="store_true")
    parser.add_argument("--max-artifact-age-days", type=int, default=30)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cleanup_stale:
        removed = cleanup_validation_artifacts(
            cache_root=CACHE_ROOT,
            max_age_days=args.max_artifact_age_days,
            report_roots=[OUT_DIR],
        )
        print(f"cleanup_removed: {len(removed)}")

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        if not data_file.exists():
            print(f"[SKIP] {instrument} {timeframe}: {data_file} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  {instrument} {timeframe}")
        print(f"{'=' * 60}")

        raw = pd.read_parquet(data_file)
        plot_suffix = ""
        plot_label = f"{args.date_from} -> end"
        date_from = args.date_from
        if args.last_days is not None:
            last_ts = (
                pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                .dropna()
                .max()
            )
            if pd.isna(last_ts):
                raise ValueError(
                    "Cannot derive --last-days window: timestamp column invalid"
                )
            start_ts = (last_ts - timedelta(days=int(args.last_days))).normalize()
            date_from = start_ts.date().isoformat()
            plot_suffix = f"_last_{int(args.last_days)}d"
            plot_label = f"last {int(args.last_days)} days ({date_from} -> end)"

        graph = get_builtin_graph(
            "validate_sr_levels",
            instrument=instrument,
            timeframe=timeframe,
        )
        context = GraphRunContext(
            graph_name=graph.graph_name,
            symbol=instrument,
            timeframe=timeframe,
            inputs={"raw_input": raw},
            config={
                "html": args.html,
                "out_dir": str(OUT_DIR),
                "date_from": date_from,
                "plot_suffix": plot_suffix,
                "plot_label": plot_label,
            },
            cache_root=CACHE_ROOT,
            force=args.force,
            invalidate_cache=args.invalidate_cache,
        )
        graph_result = execute_graph(
            graph, context=context, target="sr_validation_bundle"
        )
        result = graph_result.output().payload
        summary = result["summary"]

        print(f"  Total bars: {result['row_count']}")
        print()
        if args.audit_only:
            _print_audit(summary)
        else:
            _print_summary(summary)
        if args.benchmark:
            _print_benchmark(summary)

        if args.html and result["html_path"] is not None:
            print(f"\n  Chart saved -> {result['html_path']}")
        else:
            print("\n  [chart skipped -- pass --html to generate HTML]")

        profile_path = (
            CACHE_ROOT / VALIDATOR_NAME / instrument / timeframe / "run-summary.json"
        )
        graph_result.profiler.write_json(profile_path)
        print(f"  Profiler summary -> {profile_path}")


if __name__ == "__main__":
    main()
