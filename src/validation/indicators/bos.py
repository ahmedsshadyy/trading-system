# src/validation/indicators/bos.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CORE_BOS_COLUMNS = [
    "bos_bull",
    "bos_bear",
    "bos_direction",
    "bos_event_id",
    "bos_source_side",
    "bos_source_idx",
    "bos_source_price",
    "bos_level",
]

RAW_BREAK_COLUMNS = [
    "bos_close_break_bull",
    "bos_close_break_bear",
    "bos_wick_break_bull",
    "bos_wick_break_bear",
    "bos_raw_candidate_bull",
    "bos_raw_candidate_bear",
]

PASS_FLAG_COLUMNS = [
    "bos_pass_source_age_bull",
    "bos_pass_source_age_bear",
    "bos_pass_break_distance_bull",
    "bos_pass_break_distance_bear",
    "bos_pass_body_bull",
    "bos_pass_body_bear",
    "bos_pass_source_strength_bull",
    "bos_pass_source_strength_bear",
    "bos_pass_trend_bull",
    "bos_pass_trend_bear",
]

BREAK_QUALITY_COLUMNS = [
    "bos_break_distance",
    "bos_break_distance_atr",
    "bos_candle_body_atr",
    "bos_candle_range_atr",
    "bos_upper_wick_atr",
    "bos_lower_wick_atr",
    "bos_body_to_range",
    "bos_close_location",
    "bos_gap_from_level_atr",
    "bos_displacement_score",
]

SOURCE_QUALITY_COLUMNS = [
    "bos_source_age",
    "bos_source_strength",
    "bos_source_prominence_atr",
    "bos_source_fresh",
    "bos_source_stale",
    "bos_source_rank",
    "bos_source_in_trend_direction",
]

ALL_BOS_COLUMNS = (
    CORE_BOS_COLUMNS
    + RAW_BREAK_COLUMNS
    + PASS_FLAG_COLUMNS
    + BREAK_QUALITY_COLUMNS
    + SOURCE_QUALITY_COLUMNS
)


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
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


def _add_trend_background(fig: go.Figure, df: pd.DataFrame) -> None:
    if "trend_state" not in df.columns or df.empty:
        return

    ts = pd.to_datetime(df["timestamp"], utc=True).reset_index(drop=True)
    state = df["trend_state"].to_numpy()

    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or state[i] != state[start]:
            fill = None
            if state[start] == 1:
                fill = "rgba(34, 197, 94, 0.06)"
            elif state[start] == -1:
                fill = "rgba(239, 68, 68, 0.06)"

            if fill is not None:
                fig.add_vrect(
                    x0=ts.iloc[start],
                    x1=ts.iloc[i - 1],
                    fillcolor=fill,
                    line_width=0,
                    layer="below",
                )
            start = i


def summarize_bos(df: pd.DataFrame) -> dict[str, object]:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "bos_bull",
        "bos_bear",
        "bos_direction",
        "bos_event_id",
        "bos_source_side",
        "bos_source_idx",
        "bos_source_price",
        "bos_level",
        "bos_close_break_bull",
        "bos_close_break_bear",
        "bos_wick_break_bull",
        "bos_wick_break_bear",
        "bos_raw_candidate_bull",
        "bos_raw_candidate_bear",
        "bos_break_distance",
        "bos_break_distance_atr",
        "bos_candle_body_atr",
        "bos_source_age",
        *ALL_BOS_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing BOS columns: {sorted(missing)}")

    bos_rows = (df["bos_bull"] == 1) | (df["bos_bear"] == 1)
    bull_rows = df["bos_bull"] == 1
    bear_rows = df["bos_bear"] == 1
    event_df = df.loc[bos_rows]

    summary = {
        "window": {
            "start": str(pd.to_datetime(df["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(df["timestamp"], utc=True).max()),
            "rows": int(len(df)),
        },
        "schema": {
            "expected_bos_columns": len(ALL_BOS_COLUMNS),
            "present_bos_columns": int(
                sum(col in df.columns for col in ALL_BOS_COLUMNS)
            ),
            "missing_bos_columns": sorted(set(ALL_BOS_COLUMNS) - set(df.columns)),
        },
        "event_counts": {
            "bos_count": int(bos_rows.sum()),
            "bos_bull_count": int(bull_rows.sum()),
            "bos_bear_count": int(bear_rows.sum()),
            "bos_rate": float(bos_rows.mean()) if len(df) else np.nan,
            "wick_break_bull_count": int((df["bos_wick_break_bull"] == 1).sum()),
            "wick_break_bear_count": int((df["bos_wick_break_bear"] == 1).sum()),
            "close_break_bull_count": int((df["bos_close_break_bull"] == 1).sum()),
            "close_break_bear_count": int((df["bos_close_break_bear"] == 1).sum()),
            "raw_candidate_bull_count": int((df["bos_raw_candidate_bull"] == 1).sum()),
            "raw_candidate_bear_count": int((df["bos_raw_candidate_bear"] == 1).sum()),
        },
        "direction_distribution": _value_counts(event_df["bos_direction"]),
        "source_side_distribution": _value_counts(event_df["bos_source_side"]),
        "continuous_event_stats": {
            col: _continuous_stats(event_df[col])
            for col in BREAK_QUALITY_COLUMNS + SOURCE_QUALITY_COLUMNS
            if col
            not in {
                "bos_source_fresh",
                "bos_source_stale",
                "bos_source_in_trend_direction",
            }
        },
        "flag_value_counts": {
            col: _value_counts(df[col]) for col in RAW_BREAK_COLUMNS + PASS_FLAG_COLUMNS
        },
        "event_boolean_value_counts": {
            col: _value_counts(event_df[col])
            for col in [
                "bos_source_fresh",
                "bos_source_stale",
                "bos_source_in_trend_direction",
            ]
        },
        "sanity_checks": {
            "body_to_range_in_unit_interval": bool(
                event_df["bos_body_to_range"].between(0.0, 1.0).all()
                if not event_df.empty
                else True
            ),
            "close_location_in_unit_interval": bool(
                event_df["bos_close_location"].between(0.0, 1.0).all()
                if not event_df.empty
                else True
            ),
            "displacement_score_positive": bool(
                (event_df["bos_displacement_score"] > 0).all()
                if not event_df.empty
                else True
            ),
            "source_rank_between_1_and_10": bool(
                event_df["bos_source_rank"].between(1.0, 10.0).all()
                if not event_df.empty
                else True
            ),
            "source_fresh_and_stale_overlap_count": int(
                (
                    event_df["bos_source_fresh"].astype(bool)
                    & event_df["bos_source_stale"].astype(bool)
                ).sum()
            ),
        },
    }

    return summary


def bos_event_windows(
    df: pd.DataFrame,
    *,
    side: str = "all",
    bars_before: int = 8,
    bars_after: int = 8,
    limit: int = 5,
) -> list[pd.DataFrame]:
    if side not in {"all", "bull", "bear"}:
        raise ValueError("side must be one of {'all', 'bull', 'bear'}")

    if side == "bull":
        event_pos = np.flatnonzero(df["bos_bull"].fillna(0).astype(int).to_numpy())[
            :limit
        ]
    elif side == "bear":
        event_pos = np.flatnonzero(df["bos_bear"].fillna(0).astype(int).to_numpy())[
            :limit
        ]
    else:
        event_pos = np.flatnonzero(
            ((df["bos_bull"] == 1) | (df["bos_bear"] == 1)).to_numpy()
        )[:limit]

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "last_swing_high",
        "last_swing_low",
        "bos_bull",
        "bos_bear",
        "bos_direction",
        "bos_source_idx",
        "bos_source_price",
        "bos_level",
        "bos_close_break_bull",
        "bos_close_break_bear",
        "bos_wick_break_bull",
        "bos_wick_break_bear",
        "bos_raw_candidate_bull",
        "bos_raw_candidate_bear",
        "bos_break_distance_atr",
        "bos_candle_body_atr",
        "bos_candle_range_atr",
        "bos_body_to_range",
        "bos_close_location",
        "bos_gap_from_level_atr",
        "bos_displacement_score",
        "bos_source_age",
        "bos_source_strength",
        "bos_source_prominence_atr",
        "bos_source_rank",
        "bos_source_fresh",
        "bos_source_stale",
        "bos_source_in_trend_direction",
    ]
    cols = [c for c in cols if c in df.columns]

    out: list[pd.DataFrame] = []
    for pos in event_pos:
        lo = max(0, pos - bars_before)
        hi = min(len(df), pos + bars_after + 1)
        win = df.iloc[lo:hi][cols].copy()
        win["event_row"] = 0
        win.iloc[pos - lo, win.columns.get_loc("event_row")] = 1
        out.append(win)

    return out


def plot_bos_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "BOS Validation",
) -> Path:
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    event_mask = (out["bos_bull"] == 1) | (out["bos_bear"] == 1)
    event_df = out.loc[event_mask].copy()
    score_cap = 1.0
    if not event_df.empty and event_df["bos_displacement_score"].notna().any():
        score_cap = float(max(event_df["bos_displacement_score"].quantile(0.95), 1e-6))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
    )

    _add_trend_background(fig, out)

    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
            increasing_line_color="#159947",
            increasing_fillcolor="#159947",
            decreasing_line_color="#c2410c",
            decreasing_fillcolor="#c2410c",
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
                name="Last Confirmed Swing High",
                line=dict(color="#b91c1c", width=1, dash="dot"),
                opacity=0.55,
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
                name="Last Confirmed Swing Low",
                line=dict(color="#15803d", width=1, dash="dot"),
                opacity=0.55,
            ),
            row=1,
            col=1,
        )

    if "bos_wick_break_bull" in out.columns:
        wb = out[out["bos_wick_break_bull"] == 1]
        if not wb.empty:
            fig.add_trace(
                go.Scatter(
                    x=wb["timestamp"],
                    y=wb["high"],
                    mode="markers",
                    name="Wick Break Bull",
                    marker=dict(
                        symbol="triangle-up-open",
                        size=7,
                        color="#0ea5e9",
                        opacity=0.45,
                    ),
                    visible="legendonly",
                ),
                row=1,
                col=1,
            )

    if "bos_wick_break_bear" in out.columns:
        wb = out[out["bos_wick_break_bear"] == 1]
        if not wb.empty:
            fig.add_trace(
                go.Scatter(
                    x=wb["timestamp"],
                    y=wb["low"],
                    mode="markers",
                    name="Wick Break Bear",
                    marker=dict(
                        symbol="triangle-down-open",
                        size=7,
                        color="#7c3aed",
                        opacity=0.45,
                    ),
                    visible="legendonly",
                ),
                row=1,
                col=1,
            )

    if "bos_bull" in out.columns:
        bb = out[out["bos_bull"] == 1]
        if not bb.empty:
            bull_score = bb["bos_displacement_score"].clip(upper=score_cap)
            bull_customdata = np.column_stack(
                [
                    bb["bos_displacement_score"].to_numpy(dtype=float),
                    bb["bos_break_distance_atr"].to_numpy(dtype=float),
                    bb["bos_body_to_range"].to_numpy(dtype=float),
                    bb["bos_close_location"].to_numpy(dtype=float),
                    bb["bos_source_age"].to_numpy(dtype=float),
                    bb["bos_source_rank"].to_numpy(dtype=float),
                    bb["bos_source_prominence_atr"].to_numpy(dtype=float),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=bb["timestamp"],
                    y=bb["close"],
                    mode="markers",
                    name="BOS Bull",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=bull_score,
                        colorscale="RdYlGn",
                        cmin=0.0,
                        cmax=score_cap,
                        showscale=True,
                        colorbar=dict(
                            title="Displacement",
                            thickness=14,
                            len=0.5,
                            y=0.78,
                        ),
                        line=dict(width=1, color="#111827"),
                    ),
                    customdata=bull_customdata,
                    hovertemplate=(
                        "<b>BOS Bull</b><br>"
                        "Time=%{x}<br>"
                        "Close=%{y:.2f}<br>"
                        "Displacement=%{customdata[0]:.3f}<br>"
                        "Break ATR=%{customdata[1]:.3f}<br>"
                        "Body/Range=%{customdata[2]:.3f}<br>"
                        "Close Location=%{customdata[3]:.3f}<br>"
                        "Source Age=%{customdata[4]:.0f}<br>"
                        "Source Rank=%{customdata[5]:.0f}<br>"
                        "Prominence ATR=%{customdata[6]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if "bos_bear" in out.columns:
        bb = out[out["bos_bear"] == 1]
        if not bb.empty:
            bear_score = bb["bos_displacement_score"].clip(upper=score_cap)
            bear_customdata = np.column_stack(
                [
                    bb["bos_displacement_score"].to_numpy(dtype=float),
                    bb["bos_break_distance_atr"].to_numpy(dtype=float),
                    bb["bos_body_to_range"].to_numpy(dtype=float),
                    bb["bos_close_location"].to_numpy(dtype=float),
                    bb["bos_source_age"].to_numpy(dtype=float),
                    bb["bos_source_rank"].to_numpy(dtype=float),
                    bb["bos_source_prominence_atr"].to_numpy(dtype=float),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=bb["timestamp"],
                    y=bb["close"],
                    mode="markers",
                    name="BOS Bear",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=bear_score,
                        colorscale="RdYlGn",
                        cmin=0.0,
                        cmax=score_cap,
                        showscale=False,
                        line=dict(width=1, color="#111827"),
                    ),
                    customdata=bear_customdata,
                    hovertemplate=(
                        "<b>BOS Bear</b><br>"
                        "Time=%{x}<br>"
                        "Close=%{y:.2f}<br>"
                        "Displacement=%{customdata[0]:.3f}<br>"
                        "Break ATR=%{customdata[1]:.3f}<br>"
                        "Body/Range=%{customdata[2]:.3f}<br>"
                        "Close Location=%{customdata[3]:.3f}<br>"
                        "Source Age=%{customdata[4]:.0f}<br>"
                        "Source Rank=%{customdata[5]:.0f}<br>"
                        "Prominence ATR=%{customdata[6]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if not event_df.empty:
        signed_score = event_df["bos_displacement_score"] * event_df["bos_direction"]
        fig.add_trace(
            go.Bar(
                x=event_df["timestamp"],
                y=signed_score,
                name="Signed Displacement",
                marker=dict(
                    color=event_df["bos_displacement_score"].clip(upper=score_cap),
                    colorscale="RdYlGn",
                    cmin=0.0,
                    cmax=score_cap,
                    showscale=False,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "Time=%{x}<br>"
                    "Signed Score=%{y:.3f}<br>"
                    "Displacement=%{marker.color:.3f}<extra></extra>"
                ),
                opacity=0.9,
            ),
            row=2,
            col=1,
        )

        fig.add_hline(
            y=0.0,
            line_width=1,
            line_color="rgba(15,23,42,0.35)",
            row=2,
            col=1,
        )

    median_score = (
        float(event_df["bos_displacement_score"].median())
        if not event_df.empty
        else np.nan
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
            f"BOS events: {int(event_mask.sum())}<br>"
            f"Median displacement: {median_score:.3f}"
            if not np.isnan(median_score)
            else f"BOS events: {int(event_mask.sum())}"
        ),
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=980,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8fafc",
        font=dict(color="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=70, t=90, b=40),
        bargap=0.3,
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
        zeroline=False,
        side="right",
        title_text="Signed Disp.",
        row=2,
        col=1,
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def validate_bos(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "BOS Validation",
    n_windows: int = 5,
) -> dict[str, object]:
    summary = summarize_bos(df)
    bull_windows = bos_event_windows(df, side="bull", limit=n_windows)
    bear_windows = bos_event_windows(df, side="bear", limit=n_windows)
    html_path = plot_bos_validation(df, outpath=outpath, title=title)

    return {
        "summary": summary,
        "bull_windows": bull_windows,
        "bear_windows": bear_windows,
        "html_path": html_path,
    }
