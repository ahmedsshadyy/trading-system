from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CORE_CHOCH_COLUMNS = [
    "choch_bull",
    "choch_bear",
    "choch_direction",
    "choch_event_id",
    "choch_source_side",
    "choch_source_idx",
    "choch_source_price",
    "choch_level",
]

RAW_BREAK_COLUMNS = [
    "choch_close_break_bull",
    "choch_close_break_bear",
    "choch_wick_break_bull",
    "choch_wick_break_bear",
    "choch_raw_candidate_bull",
    "choch_raw_candidate_bear",
]

PASS_FLAG_COLUMNS = [
    "choch_pass_source_age_bull",
    "choch_pass_source_age_bear",
    "choch_pass_break_distance_bull",
    "choch_pass_break_distance_bear",
    "choch_pass_body_bull",
    "choch_pass_body_bear",
    "choch_pass_source_strength_bull",
    "choch_pass_source_strength_bear",
    "choch_pass_trend_bull",
    "choch_pass_trend_bear",
]

BREAK_QUALITY_COLUMNS = [
    "choch_break_distance",
    "choch_break_distance_atr",
    "choch_candle_body_atr",
    "choch_candle_range_atr",
    "choch_upper_wick_atr",
    "choch_lower_wick_atr",
    "choch_body_to_range",
    "choch_close_location",
    "choch_gap_from_level_atr",
    "choch_displacement_score",
]

REGIME_TRANSITION_COLUMNS = [
    "choch_trend_state_from",
    "choch_trend_state_to",
    "choch_bias_state_from",
    "choch_bias_state_to",
    "choch_against_prev_trend",
    "choch_after_structure_loss",
]

ALL_CHOCH_COLUMNS = (
    CORE_CHOCH_COLUMNS
    + RAW_BREAK_COLUMNS
    + PASS_FLAG_COLUMNS
    + BREAK_QUALITY_COLUMNS
    + REGIME_TRANSITION_COLUMNS
)


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


def summarize_choch(df: pd.DataFrame) -> dict[str, object]:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        *ALL_CHOCH_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CHoCH columns: {sorted(missing)}")

    choch_rows = (df["choch_bull"] == 1) | (df["choch_bear"] == 1)
    bull_rows = df["choch_bull"] == 1
    bear_rows = df["choch_bear"] == 1
    event_df = df.loc[choch_rows]

    return {
        "window": {
            "start": str(pd.to_datetime(df["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(df["timestamp"], utc=True).max()),
            "rows": int(len(df)),
        },
        "schema": {
            "expected_choch_columns": len(ALL_CHOCH_COLUMNS),
            "present_choch_columns": int(
                sum(col in df.columns for col in ALL_CHOCH_COLUMNS)
            ),
            "missing_choch_columns": sorted(set(ALL_CHOCH_COLUMNS) - set(df.columns)),
        },
        "event_counts": {
            "choch_count": int(choch_rows.sum()),
            "choch_bull_count": int(bull_rows.sum()),
            "choch_bear_count": int(bear_rows.sum()),
            "choch_rate": float(choch_rows.mean()) if len(df) else np.nan,
            "wick_break_bull_count": int((df["choch_wick_break_bull"] == 1).sum()),
            "wick_break_bear_count": int((df["choch_wick_break_bear"] == 1).sum()),
            "close_break_bull_count": int((df["choch_close_break_bull"] == 1).sum()),
            "close_break_bear_count": int((df["choch_close_break_bear"] == 1).sum()),
            "raw_candidate_bull_count": int(
                (df["choch_raw_candidate_bull"] == 1).sum()
            ),
            "raw_candidate_bear_count": int(
                (df["choch_raw_candidate_bear"] == 1).sum()
            ),
        },
        "direction_distribution": _value_counts(event_df["choch_direction"]),
        "source_side_distribution": _value_counts(event_df["choch_source_side"]),
        "transition_pairs": (
            event_df[["choch_trend_state_from", "choch_trend_state_to"]]
            .value_counts()
            .to_dict()
        ),
        "bias_pairs": (
            event_df[["choch_bias_state_from", "choch_bias_state_to"]]
            .value_counts()
            .to_dict()
        ),
        "continuous_event_stats": {
            col: _continuous_stats(event_df[col])
            for col in BREAK_QUALITY_COLUMNS + REGIME_TRANSITION_COLUMNS
            if col not in {"choch_against_prev_trend", "choch_after_structure_loss"}
        },
        "flag_value_counts": {
            col: _value_counts(df[col]) for col in RAW_BREAK_COLUMNS + PASS_FLAG_COLUMNS
        },
        "event_boolean_value_counts": {
            col: _value_counts(event_df[col])
            for col in ["choch_against_prev_trend", "choch_after_structure_loss"]
        },
        "sanity_checks": {
            "body_to_range_in_unit_interval": bool(
                event_df["choch_body_to_range"].between(0.0, 1.0).all()
                if not event_df.empty
                else True
            ),
            "close_location_in_unit_interval": bool(
                event_df["choch_close_location"].between(0.0, 1.0).all()
                if not event_df.empty
                else True
            ),
            "displacement_score_positive": bool(
                (event_df["choch_displacement_score"] > 0).all()
                if not event_df.empty
                else True
            ),
        },
    }


def choch_event_windows(
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
        event_pos = np.flatnonzero(df["choch_bull"].fillna(0).astype(int).to_numpy())[
            :limit
        ]
    elif side == "bear":
        event_pos = np.flatnonzero(df["choch_bear"].fillna(0).astype(int).to_numpy())[
            :limit
        ]
    else:
        event_pos = np.flatnonzero(
            ((df["choch_bull"] == 1) | (df["choch_bear"] == 1)).to_numpy()
        )[:limit]

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "trend_state",
        "trend_bias_state",
        "last_swing_high",
        "last_swing_low",
        "choch_bull",
        "choch_bear",
        "choch_direction",
        "choch_source_idx",
        "choch_source_price",
        "choch_level",
        "choch_break_distance_atr",
        "choch_candle_body_atr",
        "choch_candle_range_atr",
        "choch_body_to_range",
        "choch_close_location",
        "choch_displacement_score",
        "choch_trend_state_from",
        "choch_trend_state_to",
        "choch_bias_state_from",
        "choch_bias_state_to",
        "choch_against_prev_trend",
        "choch_after_structure_loss",
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


def plot_choch_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "CHoCH Validation",
) -> Path:
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    event_mask = (out["choch_bull"] == 1) | (out["choch_bear"] == 1)
    event_df = out.loc[event_mask].copy()
    score_cap = 1.0
    if not event_df.empty and event_df["choch_displacement_score"].notna().any():
        score_cap = float(
            max(event_df["choch_displacement_score"].quantile(0.95), 1e-6)
        )

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

    if "choch_bull" in out.columns:
        chb = out[out["choch_bull"] == 1]
        if not chb.empty:
            customdata = np.column_stack(
                [
                    chb["choch_displacement_score"].to_numpy(dtype=float),
                    chb["choch_break_distance_atr"].to_numpy(dtype=float),
                    chb["choch_body_to_range"].to_numpy(dtype=float),
                    chb["choch_close_location"].to_numpy(dtype=float),
                    chb["choch_trend_state_from"].to_numpy(dtype=float),
                    chb["choch_trend_state_to"].to_numpy(dtype=float),
                    chb["choch_after_structure_loss"].to_numpy(dtype=float),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=chb["timestamp"],
                    y=chb["close"],
                    mode="markers",
                    name="CHoCH Bull",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=chb["choch_displacement_score"].clip(upper=score_cap),
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
                    customdata=customdata,
                    hovertemplate=(
                        "<b>CHoCH Bull</b><br>"
                        "Time=%{x}<br>"
                        "Close=%{y:.2f}<br>"
                        "Displacement=%{customdata[0]:.3f}<br>"
                        "Break ATR=%{customdata[1]:.3f}<br>"
                        "Body/Range=%{customdata[2]:.3f}<br>"
                        "Close Location=%{customdata[3]:.3f}<br>"
                        "Trend From=%{customdata[4]:.0f}<br>"
                        "Trend To=%{customdata[5]:.0f}<br>"
                        "After Structure Loss=%{customdata[6]:.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if "choch_bear" in out.columns:
        chb = out[out["choch_bear"] == 1]
        if not chb.empty:
            customdata = np.column_stack(
                [
                    chb["choch_displacement_score"].to_numpy(dtype=float),
                    chb["choch_break_distance_atr"].to_numpy(dtype=float),
                    chb["choch_body_to_range"].to_numpy(dtype=float),
                    chb["choch_close_location"].to_numpy(dtype=float),
                    chb["choch_trend_state_from"].to_numpy(dtype=float),
                    chb["choch_trend_state_to"].to_numpy(dtype=float),
                    chb["choch_after_structure_loss"].to_numpy(dtype=float),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=chb["timestamp"],
                    y=chb["close"],
                    mode="markers",
                    name="CHoCH Bear",
                    marker=dict(
                        symbol="diamond-wide",
                        size=12,
                        color=chb["choch_displacement_score"].clip(upper=score_cap),
                        colorscale="RdYlGn",
                        cmin=0.0,
                        cmax=score_cap,
                        showscale=False,
                        line=dict(width=1, color="#111827"),
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>CHoCH Bear</b><br>"
                        "Time=%{x}<br>"
                        "Close=%{y:.2f}<br>"
                        "Displacement=%{customdata[0]:.3f}<br>"
                        "Break ATR=%{customdata[1]:.3f}<br>"
                        "Body/Range=%{customdata[2]:.3f}<br>"
                        "Close Location=%{customdata[3]:.3f}<br>"
                        "Trend From=%{customdata[4]:.0f}<br>"
                        "Trend To=%{customdata[5]:.0f}<br>"
                        "After Structure Loss=%{customdata[6]:.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if not event_df.empty:
        signed_score = (
            event_df["choch_displacement_score"] * event_df["choch_direction"]
        )
        fig.add_trace(
            go.Bar(
                x=event_df["timestamp"],
                y=signed_score,
                name="Signed Displacement",
                marker=dict(
                    color=event_df["choch_displacement_score"].clip(upper=score_cap),
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


def validate_choch(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "CHoCH Validation",
    n_windows: int = 5,
) -> dict[str, object]:
    summary = summarize_choch(df)
    bull_windows = choch_event_windows(df, side="bull", limit=n_windows)
    bear_windows = choch_event_windows(df, side="bear", limit=n_windows)
    html_path = plot_choch_validation(df, outpath=outpath, title=title)

    return {
        "summary": summary,
        "bull_windows": bull_windows,
        "bear_windows": bear_windows,
        "html_path": html_path,
    }
