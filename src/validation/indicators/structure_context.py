from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def _add_trend_background(
    fig: go.Figure, df: pd.DataFrame, *, row: int, col: int
) -> None:
    if "trend_state" not in df.columns or df.empty:
        return

    x = df["timestamp"].reset_index(drop=True)
    values = df["trend_state"].fillna(0).astype(int).to_numpy()

    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            state = values[start]
            if state == 1:
                color = "rgba(21, 128, 61, 0.07)"
            elif state == -1:
                color = "rgba(185, 28, 28, 0.07)"
            else:
                color = None

            if color is not None:
                fig.add_vrect(
                    x0=x.iloc[start],
                    x1=x.iloc[i - 1],
                    fillcolor=color,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=col,
                )
            start = i


def _score_cap(series_list: list[pd.Series]) -> float:
    caps: list[float] = []
    for series in series_list:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if not clean.empty:
            caps.append(float(clean.quantile(0.95)))
    return max(max(caps, default=0.0), 1e-6)


def summarize_structure_context(df: pd.DataFrame) -> dict[str, object]:
    required = {
        "timestamp",
        "trend_state",
        "trend_bias_state",
        "bos_bull",
        "bos_bear",
        "choch_bull",
        "choch_bear",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing structure context columns: {sorted(missing)}")

    out = _ensure_datetime(df)
    bos_mask = (out["bos_bull"] == 1) | (out["bos_bear"] == 1)
    choch_mask = (out["choch_bull"] == 1) | (out["choch_bear"] == 1)

    return {
        "window": {
            "start": str(out["timestamp"].min()),
            "end": str(out["timestamp"].max()),
            "rows": int(len(out)),
        },
        "trend_state_distribution": (
            out["trend_state"]
            .fillna(0)
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "trend_bias_distribution": (
            out["trend_bias_state"]
            .fillna(0)
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "event_counts": {
            "bos_count": int(bos_mask.sum()),
            "bos_bull_count": int((out["bos_bull"] == 1).sum()),
            "bos_bear_count": int((out["bos_bear"] == 1).sum()),
            "choch_count": int(choch_mask.sum()),
            "choch_bull_count": int((out["choch_bull"] == 1).sum()),
            "choch_bear_count": int((out["choch_bear"] == 1).sum()),
            "same_bar_overlap": int((bos_mask & choch_mask).sum()),
        },
        "transition_pairs": (
            out.loc[choch_mask, ["choch_trend_state_from", "choch_trend_state_to"]]
            .value_counts()
            .to_dict()
            if {"choch_trend_state_from", "choch_trend_state_to"}.issubset(out.columns)
            else {}
        ),
    }


def plot_structure_context_validation(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Structure Context Validation",
) -> Path:
    out = _ensure_datetime(df)
    bos_mask = (out["bos_bull"] == 1) | (out["bos_bear"] == 1)
    choch_mask = (out["choch_bull"] == 1) | (out["choch_bear"] == 1)
    score_cap = _score_cap(
        [
            (
                out.loc[bos_mask, "bos_displacement_score"]
                if "bos_displacement_score" in out.columns
                else pd.Series(dtype=float)
            ),
            (
                out.loc[choch_mask, "choch_displacement_score"]
                if "choch_displacement_score" in out.columns
                else pd.Series(dtype=float)
            ),
        ]
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.76, 0.24],
        subplot_titles=(
            "Price + Trend State + BOS / CHoCH",
            "Strict Trend State vs Bias State",
        ),
    )

    _add_trend_background(fig, out, row=1, col=1)

    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
            increasing_line_color="#15803d",
            increasing_fillcolor="#15803d",
            decreasing_line_color="#b45309",
            decreasing_fillcolor="#b45309",
        ),
        row=1,
        col=1,
    )

    if "last_swing_high" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_high"],
                mode="lines",
                name="Last Swing High",
                line=dict(color="#b91c1c", width=1, dash="dot"),
                opacity=0.5,
            ),
            row=1,
            col=1,
        )

    if "last_swing_low" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_low"],
                mode="lines",
                name="Last Swing Low",
                line=dict(color="#15803d", width=1, dash="dot"),
                opacity=0.5,
            ),
            row=1,
            col=1,
        )

    for mask_col, score_col, name, symbol in [
        ("bos_bull", "bos_displacement_score", "BOS Bull", "triangle-up"),
        ("bos_bear", "bos_displacement_score", "BOS Bear", "triangle-down"),
        ("choch_bull", "choch_displacement_score", "CHoCH Bull", "diamond"),
        (
            "choch_bear",
            "choch_displacement_score",
            "CHoCH Bear",
            "diamond-wide",
        ),
    ]:
        if mask_col not in out.columns:
            continue

        sub = out[out[mask_col] == 1].copy()
        if sub.empty:
            continue

        hover_cols = [
            score_col,
            "trend_state",
            "trend_bias_state",
            (
                "bos_source_rank"
                if mask_col.startswith("bos")
                else "choch_trend_state_from"
            ),
            (
                "bos_source_age"
                if mask_col.startswith("bos")
                else "choch_against_prev_trend"
            ),
        ]
        customdata = np.column_stack(
            [
                (
                    sub[col].to_numpy(dtype=float)
                    if col in sub.columns
                    else np.full(len(sub), np.nan)
                )
                for col in hover_cols
            ]
        )
        score_values = (
            pd.to_numeric(sub[score_col], errors="coerce")
            if score_col in sub.columns
            else pd.Series(np.nan, index=sub.index)
        )
        is_first_scored_trace = name == "BOS Bull"

        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["close"],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=12,
                    color=score_values.clip(upper=score_cap),
                    colorscale="RdYlGn",
                    cmin=0.0,
                    cmax=score_cap,
                    showscale=is_first_scored_trace,
                    colorbar=(
                        dict(
                            title="Disp. Score",
                            thickness=14,
                            len=0.45,
                            y=0.76,
                        )
                        if is_first_scored_trace
                        else None
                    ),
                    line=dict(width=1, color="#111827"),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Time=%{x}<br>"
                    "Close=%{y:.2f}<br>"
                    "Score=%{customdata[0]:.3f}<br>"
                    "Trend=%{customdata[1]:.0f}<br>"
                    "Bias=%{customdata[2]:.0f}<br>"
                    "Meta 1=%{customdata[3]:.0f}<br>"
                    "Meta 2=%{customdata[4]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=out["timestamp"],
            y=out["trend_state"].astype(float),
            mode="lines+markers",
            name="Trend State",
            line=dict(color="#0f172a", width=2),
            marker=dict(size=4, color="#0f172a"),
        ),
        row=2,
        col=1,
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.99,
        xanchor="right",
        yanchor="top",
        align="right",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(15,23,42,0.15)",
        borderwidth=1,
        text=(
            f"BOS: {int(bos_mask.sum())}<br>"
            f"CHoCH: {int(choch_mask.sum())}<br>"
            f"Overlap: {int((bos_mask & choch_mask).sum())}"
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=out["timestamp"],
            y=out["trend_bias_state"].astype(float),
            mode="lines",
            name="Trend Bias State",
            line=dict(color="#475569", width=1.5, dash="dash"),
        ),
        row=2,
        col=1,
    )

    for y, color in [
        (1, "rgba(21,128,61,0.25)"),
        (0, "rgba(100,116,139,0.2)"),
        (-1, "rgba(185,28,28,0.25)"),
    ]:
        fig.add_hline(y=y, line_width=1, line_color=color, row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=980,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8fafc",
        font=dict(color="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=50, t=90, b=40),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        range=[out["timestamp"].min(), out["timestamp"].max()],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        range=[out["timestamp"].min(), out["timestamp"].max()],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        side="right",
        title_text="Price",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15,23,42,0.06)",
        side="right",
        title_text="State",
        tickmode="array",
        tickvals=[-1, 0, 1],
        row=2,
        col=1,
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def validate_structure_context(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Structure Context Validation",
) -> dict[str, object]:
    summary = summarize_structure_context(df)
    html_path = plot_structure_context_validation(df, outpath=outpath, title=title)
    return {"summary": summary, "html_path": html_path}
