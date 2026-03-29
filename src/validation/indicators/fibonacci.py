"""
validation/indicators/fibonacci.py

Chart builder and validation logic for the Fibonacci level engine.

Produces a 4-panel Plotly HTML chart:
  Row 1: Candlestick price + fib level lines (bull=blue, bear=red, golden=gold)
  Row 2: ATR-normalized distance to nearest fib level (bull + bear)
  Row 3: Active structure count (bull + bear)
  Row 4: Quality score of selected active structures

Plus research summary tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.foundation.fibonacci import (
    FIB_RATIOS,
    FIB_RATIO_LABELS,
    FIB_DIR_BULL,
    FIB_DIR_BEAR,
)
from src.indicators.research.fibonacci_research import (
    build_fibonacci_research_table,
    summarize_fibonacci_research,
    FIB_RESEARCH_COLUMNS,
)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
_BULL_SOLID = "rgba(30, 110, 255, 0.85)"
_BEAR_SOLID = "rgba(210, 30, 30, 0.85)"
_GOLD_SOLID = "rgba(230, 175, 0, 0.95)"
_BULL_FILL = "rgba(30, 110, 255, 0.10)"
_BEAR_FILL = "rgba(210, 30, 30, 0.10)"

# Colors per ratio — special gold for 0.618
_RATIO_COLORS_BULL = {
    "0000": "rgba(30, 110, 255, 0.30)",
    "0236": "rgba(30, 110, 255, 0.45)",
    "0382": "rgba(30, 110, 255, 0.60)",
    "0500": "rgba(30, 110, 255, 0.70)",
    "0618": "rgba(230, 175, 0, 0.95)",  # golden
    "0786": "rgba(30, 110, 255, 0.80)",
    "1000": "rgba(30, 110, 255, 0.30)",
    "1272": "rgba(30, 110, 255, 0.25)",
    "1618": "rgba(30, 110, 255, 0.20)",
}
_RATIO_COLORS_BEAR = {
    "0000": "rgba(210, 30, 30, 0.30)",
    "0236": "rgba(210, 30, 30, 0.45)",
    "0382": "rgba(210, 30, 30, 0.60)",
    "0500": "rgba(210, 30, 30, 0.70)",
    "0618": "rgba(230, 175, 0, 0.95)",  # golden
    "0786": "rgba(210, 30, 30, 0.80)",
    "1000": "rgba(210, 30, 30, 0.30)",
    "1272": "rgba(210, 30, 30, 0.25)",
    "1618": "rgba(210, 30, 30, 0.20)",
}

MAX_LEVEL_LINES = 200  # cap to keep browser fast


def _add_candlestick(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1) -> None:
    x = df["timestamp"] if "timestamp" in df.columns else df.index
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _draw_fib_levels(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    direction: str,
    max_lines: int = MAX_LEVEL_LINES,
) -> int:
    """Draw horizontal Fibonacci level lines for one direction.

    Returns the number of lines drawn.
    """
    prefix = direction  # "bull" or "bear"
    colors = _RATIO_COLORS_BULL if direction == "bull" else _RATIO_COLORS_BEAR
    detect_col = f"fib_{prefix}_detect_flag"
    if detect_col not in df.columns:
        return 0

    x_col = df["timestamp"] if "timestamp" in df.columns else df.index
    detect_positions = np.flatnonzero(
        pd.to_numeric(df[detect_col], errors="coerce").fillna(0).to_numpy() == 1
    )

    n_drawn = 0
    # Draw most recent structures first (cap by max_lines)
    for pos in reversed(detect_positions):
        if n_drawn >= max_lines:
            break
        row_data = df.iloc[int(pos)]

        # Check expiry: approximate by checking if quality_score_on_detect is present
        event_id_val = row_data.get(f"fib_{prefix}_event_id", np.nan)
        if pd.isna(event_id_val):
            continue

        x_start = x_col.iloc[int(pos)]

        for lbl, ratio in zip(FIB_RATIO_LABELS, FIB_RATIOS):
            col_name = f"fib_{prefix}_l{lbl}"
            if col_name not in df.columns:
                continue
            lp = row_data.get(col_name, np.nan)
            if not np.isfinite(float(lp)):
                continue

            color = colors.get(lbl, _BULL_SOLID if direction == "bull" else _BEAR_SOLID)
            label_str = f"{ratio:.3f}"
            is_golden = lbl == "0618"
            width = 2.0 if is_golden else 1.0
            dash = "solid" if is_golden else "dot"

            fig.add_shape(
                type="line",
                x0=x_start,
                x1=x_col.iloc[-1],
                y0=float(lp),
                y1=float(lp),
                line=dict(color=color, width=width, dash=dash),
                row=row,
                col=1,
            )
            n_drawn += 1
            if n_drawn >= max_lines:
                break

    return n_drawn


def validate_fibonacci(
    df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    outpath: Path | None = None,
    title: str = "Fibonacci Levels Validation",
    research_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build validation chart and return summary tables.

    Parameters
    ----------
    df:
        Filtered DataFrame for the chart window (with fib columns already present).
    full_df:
        Full dataset used for research table. If None, uses df.
    outpath:
        If provided, save HTML chart here.
    title:
        Chart title.
    research_table:
        Pre-built research table. If None, will be built from full_df.

    Returns
    -------
    dict with keys: ``chart``, ``research_summary``, ``tables``
    """
    source_df = full_df if full_df is not None else df

    if research_table is None:
        research_table = build_fibonacci_research_table(source_df)

    summary = summarize_fibonacci_research(research_table)

    x = df["timestamp"] if "timestamp" in df.columns else df.index

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        vertical_spacing=0.03,
        subplot_titles=[
            "Price + Fibonacci Levels",
            "Distance to Nearest Fib Level (ATR)",
            "Active Structure Count",
            "Quality Score (Selected Structure)",
        ],
    )

    # Row 1: candlestick + fib levels
    _add_candlestick(fig, df, row=1)
    _draw_fib_levels(fig, df, row=1, direction="bull")
    _draw_fib_levels(fig, df, row=1, direction="bear")

    # Row 2: distance to nearest fib level
    if "fib_nearest_bull_distance_atr" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_nearest_bull_distance_atr"],
                name="Bull Fib Distance (ATR)",
                mode="lines",
                line=dict(color=_BULL_SOLID, width=1.5),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
    if "fib_nearest_bear_distance_atr" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_nearest_bear_distance_atr"],
                name="Bear Fib Distance (ATR)",
                mode="lines",
                line=dict(color=_BEAR_SOLID, width=1.5),
                showlegend=True,
            ),
            row=2,
            col=1,
        )
    # Golden ratio distance
    if "fib_golden_bull_distance_atr" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_golden_bull_distance_atr"],
                name="Bull 0.618 Distance (ATR)",
                mode="lines",
                line=dict(color=_GOLD_SOLID, width=1.5, dash="dot"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Row 3: active count
    if "fib_active_bull_count" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_active_bull_count"],
                name="Active Bull Structures",
                mode="lines",
                line=dict(color=_BULL_SOLID, width=1.5),
                fill="tozeroy",
                fillcolor=_BULL_FILL,
            ),
            row=3,
            col=1,
        )
    if "fib_active_bear_count" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_active_bear_count"],
                name="Active Bear Structures",
                mode="lines",
                line=dict(color=_BEAR_SOLID, width=1.5),
                fill="tozeroy",
                fillcolor=_BEAR_FILL,
            ),
            row=3,
            col=1,
        )

    # Row 4: quality score
    if "fib_bull_quality_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_bull_quality_score"],
                name="Bull Quality Score",
                mode="lines",
                line=dict(color=_BULL_SOLID, width=1.5),
            ),
            row=4,
            col=1,
        )
    if "fib_bear_quality_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fib_bear_quality_score"],
                name="Bear Quality Score",
                mode="lines",
                line=dict(color=_BEAR_SOLID, width=1.5),
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title=title,
        height=1200,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(outpath))

    # Build summary tables
    bull_events = research_table[research_table["fib_r_direction"] == 1]
    bear_events = research_table[research_table["fib_r_direction"] == -1]

    strongest_bull = (
        bull_events.nlargest(50, "fib_r_quality_score")
        if not bull_events.empty
        else bull_events
    )
    strongest_bear = (
        bear_events.nlargest(50, "fib_r_quality_score")
        if not bear_events.empty
        else bear_events
    )

    return {
        "chart": fig,
        "research_summary": summary,
        "tables": {
            "all_events": research_table,
            "strongest_bull": strongest_bull,
            "strongest_bear": strongest_bear,
        },
    }
