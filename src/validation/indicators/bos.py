# src/validation/indicators/bos.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing BOS columns: {sorted(missing)}")

    bos_rows = (df["bos_bull"] == 1) | (df["bos_bear"] == 1)
    bull_rows = df["bos_bull"] == 1
    bear_rows = df["bos_bear"] == 1

    summary = {
        "n_rows": int(len(df)),
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
        "avg_break_distance_atr": (
            float(df.loc[bos_rows, "bos_break_distance_atr"].mean())
            if bos_rows.any()
            else np.nan
        ),
        "median_break_distance_atr": (
            float(df.loc[bos_rows, "bos_break_distance_atr"].median())
            if bos_rows.any()
            else np.nan
        ),
        "avg_body_atr": (
            float(df.loc[bos_rows, "bos_candle_body_atr"].mean())
            if bos_rows.any()
            else np.nan
        ),
        "median_body_atr": (
            float(df.loc[bos_rows, "bos_candle_body_atr"].median())
            if bos_rows.any()
            else np.nan
        ),
        "avg_source_age": (
            float(df.loc[bos_rows, "bos_source_age"].mean())
            if bos_rows.any()
            else np.nan
        ),
        "median_source_age": (
            float(df.loc[bos_rows, "bos_source_age"].median())
            if bos_rows.any()
            else np.nan
        ),
        "source_side_distribution": df.loc[bos_rows, "bos_source_side"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "direction_distribution": df.loc[bos_rows, "bos_direction"]
        .value_counts()
        .sort_index()
        .to_dict(),
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
        "bos_source_age",
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

    if "last_swing_high" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_high"],
                mode="lines",
                name="Last Confirmed Swing High",
                line=dict(color="#ef553b", width=1, dash="dot"),
            )
        )

    if "last_swing_low" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_low"],
                mode="lines",
                name="Last Confirmed Swing Low",
                line=dict(color="#00cc96", width=1, dash="dot"),
            )
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
                    marker=dict(symbol="triangle-up-open", size=8, color="#19d3f3"),
                )
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
                    marker=dict(symbol="triangle-down-open", size=8, color="#ab63fa"),
                )
            )

    if "bos_bull" in out.columns:
        bb = out[out["bos_bull"] == 1]
        if not bb.empty:
            fig.add_trace(
                go.Scatter(
                    x=bb["timestamp"],
                    y=bb["close"],
                    mode="markers",
                    name="BOS Bull",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#00cc96",
                        line=dict(width=1, color="white"),
                    ),
                )
            )

    if "bos_bear" in out.columns:
        bb = out[out["bos_bear"] == 1]
        if not bb.empty:
            fig.add_trace(
                go.Scatter(
                    x=bb["timestamp"],
                    y=bb["close"],
                    mode="markers",
                    name="BOS Bear",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#ef553b",
                        line=dict(width=1, color="white"),
                    ),
                )
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
