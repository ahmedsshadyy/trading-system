"""Plotly chart helper for final-sweeps audits.

Renders OHLC candles, the unified-source ladder (top-K above/below),
and sweep markers (same-bar / delayed / accepted breakout / unresolved /
failed-breakout-reclaim) onto a single HTML page suitable for human review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators.smc.sweeps.final_sweeps import (
    SWEEP_CLASS_ACCEPTED_BREAKOUT,
    SWEEP_CLASS_DELAYED_REJECTION,
    SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM,
    SWEEP_CLASS_PROBED,
    SWEEP_CLASS_SAME_BAR,
    SWEEP_CLASS_UNRESOLVED,
)

_CLASS_STYLE: dict[int, tuple[str, str, str]] = {
    SWEEP_CLASS_SAME_BAR: ("same-bar sweep", "#22d3ee", "triangle-down"),
    SWEEP_CLASS_DELAYED_REJECTION: (
        "delayed-rejection sweep",
        "#a855f7",
        "triangle-down-open",
    ),
    SWEEP_CLASS_ACCEPTED_BREAKOUT: ("accepted breakout", "#10b981", "x"),
    SWEEP_CLASS_UNRESOLVED: ("unresolved breach", "#f59e0b", "circle-open"),
    SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM: (
        "failed-breakout reclaim",
        "#ef4444",
        "diamond",
    ),
    SWEEP_CLASS_PROBED: ("probed", "#6b7280", "circle"),
}


def render_final_sweeps_chart(
    df: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    ladder_depth: int = 5,
) -> Path:
    """Render the chart for the supplied (already-sliced) frame."""

    if "timestamp" not in df.columns:
        raise ValueError("render_final_sweeps_chart: df must contain 'timestamp'")

    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Plotly's candlestick + tz-aware Timestamp in pandas 2.x can produce a
    # blank chart when serialized with the default JSON encoder. Strip the
    # timezone so plotly receives plain datetime64[ns] values.
    df["timestamp"] = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    df = df.sort_values("timestamp").reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"].astype(float),
            high=df["high"].astype(float),
            low=df["low"].astype(float),
            close=df["close"].astype(float),
            name="XAU/USD",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            line={"width": 1},
        )
    )

    # Ladder L1 above and below — render as faded scatter so the user can
    # see the most-important nearest source per side without overplotting
    # the deeper ranks.
    for side, color in (
        ("above", "rgba(220,38,38,0.45)"),
        ("below", "rgba(22,163,74,0.45)"),
    ):
        col = f"liq_{side}_l1_level"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[col],
                    mode="lines",
                    line={"color": color, "width": 1, "dash": "dot"},
                    name=f"liq L1 {side}",
                    hovertemplate=("%{y:.2f}<extra>L1 " + side + "</extra>"),
                )
            )

    # Sweep markers, grouped by class for clean legend.
    if "sweep_class" in df.columns:
        for cls, (label, color, symbol) in _CLASS_STYLE.items():
            mask = df["sweep_class"].fillna(-1).astype(int) == cls
            if not mask.any():
                continue
            sub = df.loc[mask]
            # Use mid as marker y when the source level is known; otherwise
            # use the bar high.
            ys = sub["sweep_source_level"].where(
                sub["sweep_source_level"].notna(), sub["high"]
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=ys,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 9,
                        "symbol": symbol,
                        "line": {"width": 1, "color": "#111"},
                    },
                    name=label,
                    hovertext=[
                        f"<br>fam: {fam}"
                        f"<br>side: {int(sd) if pd.notna(sd) else 'NA'}"
                        f"<br>pen_atr: {p:.2f}"
                        f"<br>quality: {q:.2f}"
                        for fam, sd, p, q in zip(
                            sub["sweep_primary_family"].astype(str),
                            sub["sweep_source_side"],
                            sub["penetration_atr"].fillna(np.nan),
                            sub["sweep_quality_score"].fillna(np.nan),
                        )
                    ],
                    hovertemplate="%{x}%{hovertext}<extra></extra>",
                )
            )

    # Force the x-range to the visible slice so plotly doesn't auto-pan to
    # an empty area (the symptom that produces the "blank chart" report).
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        xaxis={
            "type": "date",
            "range": [
                df["timestamp"].iloc[0],
                df["timestamp"].iloc[-1],
            ],
            "showspikes": True,
            "spikemode": "across",
            "spikethickness": 1,
            "rangebreaks": [
                # Skip weekends so the H4 candles sit shoulder-to-shoulder.
                {"bounds": ["sat", "mon"]},
            ],
        },
        yaxis={
            "autorange": True,
            "fixedrange": False,
            "showspikes": True,
            "spikemode": "across",
            "spikethickness": 1,
        },
        height=820,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 80, "b": 40, "l": 40, "r": 20},
        template="plotly_white",
        hovermode="x unified",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


__all__ = ["render_final_sweeps_chart"]
