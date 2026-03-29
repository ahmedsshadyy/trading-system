from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def slice_view(
    df: pd.DataFrame,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Return a sliced copy for plotting/inspection."""
    return (
        df.iloc[start:end].copy()
        if (start is not None or end is not None)
        else df.copy()
    )


def create_candlestick_figure(
    df: pd.DataFrame,
    *,
    title: str = "Validation Chart",
) -> go.Figure:
    """Create a base candlestick figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=850,
    )
    return fig


def add_scatter_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    x_col: str = "timestamp",
    y_col: str,
    mask_col: str,
    name: str,
    symbol: str,
    size: int = 10,
) -> None:
    """Add marker layer for rows where mask_col == 1."""
    if mask_col not in df.columns or y_col not in df.columns:
        return

    sub = df[df[mask_col] == 1]
    if sub.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers",
            name=name,
            marker=dict(symbol=symbol, size=size),
        )
    )


def add_line_series(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    x_col: str = "timestamp",
    y_col: str,
    name: str,
    dash: str = "solid",
) -> None:
    """Add line series if column exists."""
    if y_col not in df.columns:
        return

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=name,
            line=dict(dash=dash),
        )
    )


def add_state_background(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    state_col: str = "trend_state",
    x_col: str = "timestamp",
    bull_value: int = 1,
    bear_value: int = -1,
    bull_fill: str = "rgba(0, 200, 0, 0.08)",
    bear_fill: str = "rgba(200, 0, 0, 0.08)",
) -> None:
    """Add background shading for state regimes."""
    if state_col not in df.columns:
        return
    if df.empty:
        return

    x = df[x_col].reset_index(drop=True)
    state = df[state_col].to_numpy()

    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or state[i] != state[start]:
            s = state[start]
            if s == bull_value:
                fill = bull_fill
            elif s == bear_value:
                fill = bear_fill
            else:
                fill = None

            if fill is not None:
                fig.add_vrect(
                    x0=x.iloc[start],
                    x1=x.iloc[i - 1],
                    fillcolor=fill,
                    line_width=0,
                    layer="below",
                )
            start = i


def add_transition_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    x_col: str = "timestamp",
    y_col: str = "close",
    changed_col: str = "trend_state_changed",
    name: str = "State Change",
    symbol: str = "x",
    size: int = 11,
) -> None:
    """Add markers on rows where changed_col == 1."""
    if changed_col not in df.columns or y_col not in df.columns:
        return

    sub = df[df[changed_col] == 1]
    if sub.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers",
            name=name,
            marker=dict(symbol=symbol, size=size),
        )
    )


def save_figure_html(fig: go.Figure, outpath: str | Path) -> Path:
    """Persist Plotly figure to HTML using atomic rename semantics."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = outpath.with_suffix(outpath.suffix + ".tmp")
    fig.write_html(str(tmp_path))
    os.replace(tmp_path, outpath)
    return outpath
