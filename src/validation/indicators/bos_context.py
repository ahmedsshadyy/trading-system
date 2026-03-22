from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.features.bos_context import (
    LIVE_BOS_CONTEXT_COLUMNS,
    RESEARCH_BOS_CONTEXT_COLUMNS,
)

QUALITY_BUCKET_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
QUALITY_BUCKET_LABELS = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]

BOOLEAN_CONTEXT_COLUMNS = [
    "bos_against_prev_trend",
    "bos_in_neutral_trend",
    "bos_near_wedge",
    "bos_after_sweep",
    "bos_after_displacement",
    "bos_after_fvg",
    "bos_into_ob",
    "bos_into_fvg",
    "bos_near_eqhl",
    "bos_near_liquidity",
    "bos_hold_1",
    "bos_hold_2",
    "bos_hold_3",
    "bos_hold_5",
    "bos_failed_1",
    "bos_failed_2",
    "bos_failed_3",
    "bos_failed_5",
    "bos_retest_1",
    "bos_retest_3",
    "bos_retest_5",
]

DISCRETE_CONTEXT_COLUMNS = [
    "bos_trend_alignment",
    "bos_trend_state_on_event",
    "bos_trend_bias_state_on_event",
    "bos_wedge_kind",
]

CONTINUOUS_CONTEXT_COLUMNS = [
    "bos_mfe_3_atr",
    "bos_mae_3_atr",
    "bos_mfe_5_atr",
    "bos_mae_5_atr",
    "bos_mfe_10_atr",
    "bos_mae_10_atr",
    "bos_quality_score",
    "bos_tradeable_score",
]


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


def _event_mask(df: pd.DataFrame) -> pd.Series:
    if "bos_direction" in df.columns:
        return df["bos_direction"].fillna(0) != 0
    return (df["bos_bull"].fillna(0) == 1) | (df["bos_bear"].fillna(0) == 1)


def _quality_bucket_frame(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty or "bos_quality_score" not in event_df.columns:
        return pd.DataFrame(
            columns=["quality_bucket", "event_count", "hold_3_rate", "mean_mfe_5_atr"]
        )

    bucketed = event_df.copy()
    bucketed["quality_bucket"] = pd.cut(
        bucketed["bos_quality_score"],
        bins=QUALITY_BUCKET_EDGES,
        labels=QUALITY_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    bucketed = bucketed.dropna(subset=["quality_bucket"])
    if bucketed.empty:
        return pd.DataFrame(
            columns=["quality_bucket", "event_count", "hold_3_rate", "mean_mfe_5_atr"]
        )

    grouped = (
        bucketed.groupby("quality_bucket", observed=False)
        .agg(
            event_count=("bos_quality_score", "size"),
            hold_3_rate=("bos_hold_3", "mean"),
            mean_mfe_5_atr=("bos_mfe_5_atr", "mean"),
        )
        .reset_index()
    )
    grouped["quality_bucket"] = grouped["quality_bucket"].astype(str)
    return grouped


def summarize_bos_context(df: pd.DataFrame) -> dict[str, object]:
    required = {"timestamp", "open", "high", "low", "close", *LIVE_BOS_CONTEXT_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing BOS context columns: {sorted(missing)}")

    out = _ensure_datetime(df)
    bos_mask = _event_mask(out)
    event_df = out.loc[bos_mask].copy()
    present_context_cols = [
        col for col in RESEARCH_BOS_CONTEXT_COLUMNS if col in out.columns
    ]
    forward_cols_present = [
        col
        for col in RESEARCH_BOS_CONTEXT_COLUMNS
        if col not in LIVE_BOS_CONTEXT_COLUMNS and col in out.columns
    ]

    quality_buckets = _quality_bucket_frame(event_df)

    summary: dict[str, object] = {
        "window": {
            "start": str(out["timestamp"].min()),
            "end": str(out["timestamp"].max()),
            "rows": int(len(out)),
        },
        "schema": {
            "expected_bos_context_columns": len(RESEARCH_BOS_CONTEXT_COLUMNS),
            "present_bos_context_columns": int(
                sum(col in out.columns for col in RESEARCH_BOS_CONTEXT_COLUMNS)
            ),
            "missing_bos_context_columns": sorted(
                set(RESEARCH_BOS_CONTEXT_COLUMNS) - set(out.columns)
            ),
            "live_mode_columns_present": int(
                sum(col in out.columns for col in LIVE_BOS_CONTEXT_COLUMNS)
            ),
            "forward_diagnostics_present": len(forward_cols_present) > 0,
        },
        "event_counts": {
            "bos_count": int(bos_mask.sum()),
            "bos_bull_count": (
                int((out["bos_bull"] == 1).sum()) if "bos_bull" in out.columns else 0
            ),
            "bos_bear_count": (
                int((out["bos_bear"] == 1).sum()) if "bos_bear" in out.columns else 0
            ),
            "bos_rate": float(bos_mask.mean()) if len(out) else np.nan,
        },
        "discrete_event_value_counts": {
            col: _value_counts(event_df[col])
            for col in DISCRETE_CONTEXT_COLUMNS
            if col in event_df.columns
        },
        "boolean_event_value_counts": {
            col: _value_counts(event_df[col])
            for col in BOOLEAN_CONTEXT_COLUMNS
            if col in event_df.columns
        },
        "continuous_event_stats": {
            col: _continuous_stats(event_df[col])
            for col in CONTINUOUS_CONTEXT_COLUMNS
            if col in event_df.columns
        },
        "headline_metrics": {
            "after_sweep_rate": (
                float(event_df["bos_after_sweep"].dropna().mean())
                if "bos_after_sweep" in event_df.columns and not event_df.empty
                else np.nan
            ),
            "after_displacement_rate": (
                float(event_df["bos_after_displacement"].dropna().mean())
                if "bos_after_displacement" in event_df.columns and not event_df.empty
                else np.nan
            ),
            "after_fvg_rate": (
                float(event_df["bos_after_fvg"].dropna().mean())
                if "bos_after_fvg" in event_df.columns and not event_df.empty
                else np.nan
            ),
            "hold_3_rate": (
                float(event_df["bos_hold_3"].dropna().mean())
                if "bos_hold_3" in event_df.columns and not event_df.empty
                else np.nan
            ),
            "mean_quality_score": (
                float(event_df["bos_quality_score"].dropna().mean())
                if "bos_quality_score" in event_df.columns and not event_df.empty
                else np.nan
            ),
            "mean_tradeable_score": (
                float(event_df["bos_tradeable_score"].dropna().mean())
                if "bos_tradeable_score" in event_df.columns and not event_df.empty
                else np.nan
            ),
        },
        "quality_bucket_stats": quality_buckets.to_dict(orient="records"),
        "sanity_checks": {
            "non_event_context_rows_all_nan": bool(
                out.loc[~bos_mask, present_context_cols].isna().all().all()
            ),
            "quality_score_in_unit_interval": bool(
                event_df["bos_quality_score"].between(0.0, 1.0).all()
                if "bos_quality_score" in event_df.columns and not event_df.empty
                else True
            ),
            "tradeable_score_in_unit_interval": bool(
                event_df["bos_tradeable_score"].between(0.0, 1.0).all()
                if "bos_tradeable_score" in event_df.columns and not event_df.empty
                else True
            ),
            "hold_3_rate_non_degenerate": bool(
                0.0 < event_df["bos_hold_3"].dropna().mean() < 1.0
                if "bos_hold_3" in event_df.columns
                and not event_df["bos_hold_3"].dropna().empty
                else True
            ),
        },
    }

    return summary


def plot_bos_context_validation(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "BOS Context Validation",
) -> Path:
    out = _ensure_datetime(df)
    bos_mask = _event_mask(out)
    event_df = out.loc[bos_mask].copy()
    score_cap = 1.0
    if "bos_tradeable_score" in event_df.columns and not event_df.empty:
        clean = pd.to_numeric(event_df["bos_tradeable_score"], errors="coerce").dropna()
        if not clean.empty:
            score_cap = float(max(clean.quantile(0.95), 1e-6))

    bucket_df = _quality_bucket_frame(event_df)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Price + BOS Events Colored by Tradeable Score",
            "Quality Score Distribution",
            "Hold_3 Rate by Quality Bucket",
            "Mean MFE_5_ATR by Quality Bucket",
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

    for line_col, name, color in [
        ("last_swing_high", "Last Swing High", "#b91c1c"),
        ("last_swing_low", "Last Swing Low", "#15803d"),
    ]:
        if line_col in out.columns:
            fig.add_trace(
                go.Scatter(
                    x=out["timestamp"],
                    y=out[line_col],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.5,
                ),
                row=1,
                col=1,
            )

    for mask_col, name, symbol in [
        ("bos_bull", "BOS Bull", "triangle-up"),
        ("bos_bear", "BOS Bear", "triangle-down"),
    ]:
        if mask_col not in out.columns:
            continue
        sub = out[out[mask_col] == 1].copy()
        if sub.empty:
            continue
        customdata = np.column_stack(
            [
                (
                    sub[col].to_numpy(dtype=float)
                    if col in sub.columns
                    else np.full(len(sub), np.nan)
                )
                for col in [
                    "bos_quality_score",
                    "bos_tradeable_score",
                    "bos_trend_alignment",
                    "bos_after_sweep",
                    "bos_after_displacement",
                    "bos_after_fvg",
                    "bos_hold_3",
                    "bos_mfe_5_atr",
                    "bos_mae_5_atr",
                ]
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["close"],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=12,
                    color=sub["bos_tradeable_score"].clip(upper=score_cap),
                    colorscale="RdYlGn",
                    cmin=0.0,
                    cmax=score_cap,
                    showscale=(name == "BOS Bull"),
                    colorbar=dict(
                        title="Tradeable",
                        thickness=14,
                        len=0.45,
                        y=0.84,
                    ),
                    line=dict(color="#111827", width=1),
                ),
                customdata=customdata,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Time=%{x}<br>"
                    "Close=%{y:.2f}<br>"
                    "Quality=%{customdata[0]:.3f}<br>"
                    "Tradeable=%{customdata[1]:.3f}<br>"
                    "Trend Align=%{customdata[2]:.0f}<br>"
                    "After Sweep=%{customdata[3]:.0f}<br>"
                    "After Displacement=%{customdata[4]:.0f}<br>"
                    "After FVG=%{customdata[5]:.0f}<br>"
                    "Hold_3=%{customdata[6]:.0f}<br>"
                    "MFE_5_ATR=%{customdata[7]:.3f}<br>"
                    "MAE_5_ATR=%{customdata[8]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if "bos_quality_score" in event_df.columns:
        fig.add_trace(
            go.Histogram(
                x=event_df["bos_quality_score"],
                nbinsx=20,
                name="Quality Score",
                marker=dict(color="#2563eb", line=dict(color="#1e3a8a", width=1)),
                opacity=0.9,
            ),
            row=2,
            col=1,
        )

    if not bucket_df.empty:
        fig.add_trace(
            go.Bar(
                x=bucket_df["quality_bucket"],
                y=bucket_df["hold_3_rate"],
                name="Hold_3 Rate",
                marker_color="#059669",
                text=np.round(bucket_df["hold_3_rate"], 3),
                textposition="outside",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=bucket_df["quality_bucket"],
                y=bucket_df["mean_mfe_5_atr"],
                name="Mean MFE_5_ATR",
                marker_color="#7c3aed",
                text=np.round(bucket_df["mean_mfe_5_atr"], 3),
                textposition="outside",
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Hold_3 Rate", range=[0, 1], row=2, col=2)
    fig.update_yaxes(title_text="Mean MFE_5_ATR", row=3, col=1)
    fig.update_xaxes(title_text="Quality Score", row=2, col=1)
    fig.update_xaxes(title_text="Quality Bucket", row=2, col=2)
    fig.update_xaxes(title_text="Quality Bucket", row=3, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        width=1500,
        height=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=30, t=90, b=50),
    )
    fig.update(layout_xaxis_rangeslider_visible=False)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(outpath)
    return outpath


def validate_bos_context(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "BOS Context Validation",
) -> dict[str, object]:
    summary = summarize_bos_context(df)
    html_path = plot_bos_context_validation(df, outpath=outpath, title=title)
    return {
        "summary": summary,
        "html_path": html_path,
    }
