# src/validation/indicators/wedges.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def summarize_wedges(df: pd.DataFrame) -> dict[str, object]:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "wedge_rising",
        "wedge_falling",
        "wedge_active",
        "wedge_kind",
        "wedge_upper_bound",
        "wedge_lower_bound",
        "wedge_width_atr",
        "wedge_compression_ratio",
        "wedge_quality",
        "wedge_confirm_count",
        "wedge_breakout_up",
        "wedge_breakout_down",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing wedge columns: {sorted(missing)}")

    active = df["wedge_active"].fillna(0).astype(int)
    rising = df["wedge_rising"].fillna(0).astype(int)
    falling = df["wedge_falling"].fillna(0).astype(int)
    breakout_up = df["wedge_breakout_up"].fillna(0).astype(int)
    breakout_down = df["wedge_breakout_down"].fillna(0).astype(int)

    width_atr = pd.to_numeric(df["wedge_width_atr"], errors="coerce")
    compression = pd.to_numeric(df["wedge_compression_ratio"], errors="coerce")
    quality = pd.to_numeric(df["wedge_quality"], errors="coerce")

    return {
        "n_rows": int(len(df)),
        "wedge_active_rows": int(active.sum()),
        "wedge_active_pct": float(active.mean()) if len(df) else np.nan,
        "wedge_rising_rows": int(rising.sum()),
        "wedge_falling_rows": int(falling.sum()),
        "wedge_breakout_up_count": int(breakout_up.sum()),
        "wedge_breakout_down_count": int(breakout_down.sum()),
        "avg_wedge_width_atr": (
            float(width_atr.dropna().mean()) if width_atr.notna().any() else np.nan
        ),
        "median_wedge_width_atr": (
            float(width_atr.dropna().median()) if width_atr.notna().any() else np.nan
        ),
        "avg_wedge_compression_ratio": (
            float(compression.dropna().mean()) if compression.notna().any() else np.nan
        ),
        "median_wedge_compression_ratio": (
            float(compression.dropna().median())
            if compression.notna().any()
            else np.nan
        ),
        "avg_wedge_quality": (
            float(quality.dropna().mean()) if quality.notna().any() else np.nan
        ),
        "median_wedge_quality": (
            float(quality.dropna().median()) if quality.notna().any() else np.nan
        ),
    }


def wedge_event_windows(
    df: pd.DataFrame,
    *,
    event_col: str = "wedge_breakout_up",
    bars_before: int = 12,
    bars_after: int = 12,
    limit: int = 5,
) -> list[pd.DataFrame]:
    if event_col not in df.columns:
        return []

    event_pos = np.flatnonzero(df[event_col].fillna(0).astype(int).to_numpy())[:limit]
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "wedge_active",
        "wedge_rising",
        "wedge_falling",
        "wedge_upper_bound",
        "wedge_lower_bound",
        "wedge_width_atr",
        "wedge_compression_ratio",
        "wedge_quality",
        "wedge_target_50_price",
        "wedge_target_75_price",
        "wedge_target_100_price",
        "wedge_breakout_up",
        "wedge_breakout_down",
    ]
    cols = [c for c in cols if c in df.columns]

    out = []
    for pos in event_pos:
        lo = max(0, pos - bars_before)
        hi = min(len(df), pos + bars_after + 1)
        win = df.iloc[lo:hi][cols].copy()
        win["event_row"] = 0
        win.iloc[pos - lo, win.columns.get_loc("event_row")] = 1
        out.append(win)

    return out


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    out = []
    if len(values) == 0:
        return out
    s = 0
    cur = int(values[0])
    for i in range(1, len(values)):
        if int(values[i]) != cur:
            out.append((s, i - 1, cur))
            s = i
            cur = int(values[i])
    out.append((s, len(values) - 1, cur))
    return out


def _active_segments(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    return [
        (a, b, k)
        for a, b, k in _runs(df["wedge_kind"].fillna(0).astype(int).to_numpy())
        if k != 0
    ]


def plot_wedges_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Wedges Validation",
) -> Path:
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
            increasing_line_color="#00cc96",
            increasing_fillcolor="#00cc96",
            decreasing_line_color="#ef553b",
            decreasing_fillcolor="#ef553b",
        )
    )

    for start, end, kind in _active_segments(out):
        seg = out.iloc[start : end + 1]
        color = "#4cd964" if kind == -1 else "#ff5c5c"
        fill = "rgba(76,217,100,0.12)" if kind == -1 else "rgba(255,92,92,0.12)"

        upper = seg["wedge_upper_bound"].to_numpy(dtype=float)
        lower = seg["wedge_lower_bound"].to_numpy(dtype=float)
        ts = seg["timestamp"]

        fig.add_trace(
            go.Scatter(
                x=ts,
                y=upper,
                mode="lines",
                name="Falling Wedge Upper" if kind == -1 else "Rising Wedge Upper",
                line=dict(color=color, width=3),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=lower,
                mode="lines",
                line=dict(color=color, width=3),
                fill="tonexty",
                fillcolor=fill,
                name="Falling Wedge" if kind == -1 else "Rising Wedge",
            )
        )

        x0 = ts.iloc[0]
        y_mid = float(np.nanmean([upper[0], lower[0]]))

        fig.add_annotation(
            x=x0,
            y=y_mid,
            text="Falling Wedge" if kind == -1 else "Rising Wedge",
            showarrow=False,
            bgcolor=color,
            font=dict(color="white", size=12),
            borderpad=4,
            xanchor="left",
        )

    if {
        "wedge_breakout_up",
        "wedge_target_50_price",
        "wedge_target_75_price",
        "wedge_target_100_price",
    }.issubset(out.columns):
        bu = out[out["wedge_breakout_up"] == 1]
        for idx, row in bu.iterrows():
            x = row["timestamp"]
            y = row["close"]
            t50 = row["wedge_target_50_price"]
            t75 = row["wedge_target_75_price"]
            t100 = row["wedge_target_100_price"]

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="Breakout Up",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#19d3f3",
                        line=dict(width=1, color="white"),
                    ),
                    showlegend=bool(idx == bu.index[0]),
                )
            )

            if np.isfinite(t50):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t50,
                    line=dict(color="#4cd964", width=4, dash="solid"),
                )
                fig.add_annotation(
                    x=x,
                    y=t50,
                    text="T1 (50%)",
                    showarrow=False,
                    bgcolor="#4cd964",
                    font=dict(color="white", size=11),
                    yanchor="bottom",
                )

            if np.isfinite(t75):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t75,
                    line=dict(color="#fecb52", width=3, dash="dot"),
                )

            if np.isfinite(t100):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t100,
                    line=dict(color="#ff5c5c", width=2, dash="dash"),
                )

    if {
        "wedge_breakout_down",
        "wedge_target_50_price",
        "wedge_target_75_price",
        "wedge_target_100_price",
    }.issubset(out.columns):
        bd = out[out["wedge_breakout_down"] == 1]
        for idx, row in bd.iterrows():
            x = row["timestamp"]
            y = row["close"]
            t50 = row["wedge_target_50_price"]
            t75 = row["wedge_target_75_price"]
            t100 = row["wedge_target_100_price"]

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name="Breakout Down",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#ffffff",
                        line=dict(width=1, color="#ff5c5c"),
                    ),
                    showlegend=bool(idx == bd.index[0]),
                )
            )

            if np.isfinite(t50):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t50,
                    line=dict(color="#ff5c5c", width=4, dash="solid"),
                )
                fig.add_annotation(
                    x=x,
                    y=t50,
                    text="T1 (50%)",
                    showarrow=False,
                    bgcolor="#ff5c5c",
                    font=dict(color="white", size=11),
                    yanchor="top",
                )

            if np.isfinite(t75):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t75,
                    line=dict(color="#fecb52", width=3, dash="dot"),
                )

            if np.isfinite(t100):
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=y,
                    y1=t100,
                    line=dict(color="#19d3f3", width=2, dash="dash"),
                )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=950,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", side="right")

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def validate_wedges(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Wedges Validation",
    n_windows: int = 5,
) -> dict[str, object]:
    summary = summarize_wedges(df)
    breakout_up_windows = wedge_event_windows(
        df, event_col="wedge_breakout_up", limit=n_windows
    )
    breakout_down_windows = wedge_event_windows(
        df, event_col="wedge_breakout_down", limit=n_windows
    )
    html_path = plot_wedges_validation(df, outpath=outpath, title=title)

    return {
        "summary": summary,
        "breakout_up_windows": breakout_up_windows,
        "breakout_down_windows": breakout_down_windows,
        "html_path": html_path,
    }
