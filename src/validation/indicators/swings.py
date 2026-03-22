from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.common.chart_core import (
    add_line_series,
    add_scatter_markers,
    create_candlestick_figure,
    save_figure_html,
    slice_view,
)


def summarize_swings(df: pd.DataFrame) -> dict[str, object]:
    """Return numeric summary statistics for retrace-confirmed causal swings."""
    required = {
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "swing_high_idx",
        "swing_low_idx",
        "swing_high_detect_idx",
        "swing_low_detect_idx",
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
        "last_swing_high",
        "last_swing_low",
        "last_swing_high_idx",
        "last_swing_low_idx",
        "swing_high_age",
        "swing_low_age",
        "swing_high_prominence_atr",
        "swing_low_prominence_atr",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing swing columns: {sorted(missing)}")

    n = len(df)

    sh_rows = df[df["swing_high"] == 1]
    sl_rows = df[df["swing_low"] == 1]
    sh_conf_rows = df[df["swing_high_confirm_flag"] == 1]
    sl_conf_rows = df[df["swing_low_confirm_flag"] == 1]

    sh_idx = sh_rows.index.to_numpy()
    sl_idx = sl_rows.index.to_numpy()

    def _avg_gap(idxs: np.ndarray) -> float:
        if len(idxs) < 2:
            return np.nan
        return float(np.diff(idxs).mean())

    def _median_gap(idxs: np.ndarray) -> float:
        if len(idxs) < 2:
            return np.nan
        return float(np.median(np.diff(idxs)))

    high_origin_detect_forward_ok = (
        bool((sh_rows["swing_high_detect_idx"] >= sh_rows.index).all())
        if len(sh_rows)
        else True
    )
    low_origin_detect_forward_ok = (
        bool((sl_rows["swing_low_detect_idx"] >= sl_rows.index).all())
        if len(sl_rows)
        else True
    )

    high_confirm_origin_valid = True
    if len(sh_conf_rows):
        origin_idx = sh_conf_rows["swing_high_confirm_origin_idx"].to_numpy(dtype=int)
        valid_mask = (origin_idx >= 0) & (origin_idx < n)
        high_confirm_origin_valid = bool(valid_mask.all())

    low_confirm_origin_valid = True
    if len(sl_conf_rows):
        origin_idx = sl_conf_rows["swing_low_confirm_origin_idx"].to_numpy(dtype=int)
        valid_mask = (origin_idx >= 0) & (origin_idx < n)
        low_confirm_origin_valid = bool(valid_mask.all())

    high_sync_on_confirm_ok = (
        bool(
            (
                sh_conf_rows["last_swing_high"].to_numpy()
                == sh_conf_rows["swing_high_confirm_price"].to_numpy()
            ).all()
        )
        if len(sh_conf_rows)
        else True
    )
    low_sync_on_confirm_ok = (
        bool(
            (
                sl_conf_rows["last_swing_low"].to_numpy()
                == sl_conf_rows["swing_low_confirm_price"].to_numpy()
            ).all()
        )
        if len(sl_conf_rows)
        else True
    )

    high_age_zero_on_confirm_ok = (
        bool((sh_conf_rows["swing_high_age"] == 0).all()) if len(sh_conf_rows) else True
    )
    low_age_zero_on_confirm_ok = (
        bool((sl_conf_rows["swing_low_age"] == 0).all()) if len(sl_conf_rows) else True
    )

    high_origin_confirm_count_match = int(len(sh_rows)) == int(len(sh_conf_rows))
    low_origin_confirm_count_match = int(len(sl_rows)) == int(len(sl_conf_rows))

    return {
        "n_rows": n,
        "swing_high_count": int(len(sh_rows)),
        "swing_low_count": int(len(sl_rows)),
        "swing_high_confirm_count": int(len(sh_conf_rows)),
        "swing_low_confirm_count": int(len(sl_conf_rows)),
        "swing_high_rate": float(len(sh_rows) / n) if n > 0 else np.nan,
        "swing_low_rate": float(len(sl_rows) / n) if n > 0 else np.nan,
        "avg_bars_between_swing_highs": _avg_gap(sh_idx),
        "avg_bars_between_swing_lows": _avg_gap(sl_idx),
        "median_bars_between_swing_highs": _median_gap(sh_idx),
        "median_bars_between_swing_lows": _median_gap(sl_idx),
        "avg_swing_high_age": float(df["swing_high_age"].dropna().mean()),
        "avg_swing_low_age": float(df["swing_low_age"].dropna().mean()),
        "avg_swing_high_prom_atr": float(
            df["swing_high_prominence_atr"].dropna().mean()
        ),
        "avg_swing_low_prom_atr": float(df["swing_low_prominence_atr"].dropna().mean()),
        "median_swing_high_prom_atr": float(
            df["swing_high_prominence_atr"].dropna().median()
        ),
        "median_swing_low_prom_atr": float(
            df["swing_low_prominence_atr"].dropna().median()
        ),
        "high_origin_detect_forward_ok": high_origin_detect_forward_ok,
        "low_origin_detect_forward_ok": low_origin_detect_forward_ok,
        "high_confirm_origin_valid": high_confirm_origin_valid,
        "low_confirm_origin_valid": low_confirm_origin_valid,
        "last_swing_high_updates_on_confirm_bar": high_sync_on_confirm_ok,
        "last_swing_low_updates_on_confirm_bar": low_sync_on_confirm_ok,
        "swing_high_age_zero_on_confirm_bar": high_age_zero_on_confirm_ok,
        "swing_low_age_zero_on_confirm_bar": low_age_zero_on_confirm_ok,
        "high_origin_confirm_count_match": high_origin_confirm_count_match,
        "low_origin_confirm_count_match": low_origin_confirm_count_match,
    }


def swing_event_windows(
    df: pd.DataFrame,
    side: str = "high",
    bars_before: int = 4,
    bars_after: int = 4,
    limit: int = 10,
) -> list[pd.DataFrame]:
    """Return small windows around swing origin bars for manual inspection."""
    if side not in {"high", "low"}:
        raise ValueError("side must be 'high' or 'low'")

    flag_col = "swing_high" if side == "high" else "swing_low"
    event_idx = df.index[df[flag_col] == 1].tolist()[:limit]

    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "swing_high_detect_idx",
        "swing_low_detect_idx",
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
        "last_swing_high",
        "last_swing_low",
        "swing_high_age",
        "swing_low_age",
        "swing_high_prominence_atr",
        "swing_low_prominence_atr",
    ]
    cols = [c for c in cols if c in df.columns]

    out = []
    for idx in event_idx:
        lo = max(0, idx - bars_before)
        hi = min(len(df), idx + bars_after + 1)
        win = df.iloc[lo:hi][cols].copy()
        win["event_row"] = 0
        win.loc[df.index[idx], "event_row"] = 1
        out.append(win)
    return out


def _apply_compare_style(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        side="right",
    )

    for trace in fig.data:
        if getattr(trace, "type", None) == "candlestick":
            trace.increasing.line.color = "#00cc96"
            trace.increasing.fillcolor = "#00cc96"
            trace.decreasing.line.color = "#ef553b"
            trace.decreasing.fillcolor = "#ef553b"
    """
    Make validation chart visually match compare_swing_configs.py
    as closely as possible.
    """
    fig.update_layout(
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        side="right",
    )

    # Candlestick colors to match your compare chart feel
    for trace in fig.data:
        if getattr(trace, "type", None) == "candlestick":
            trace.increasing.line.color = "#00cc96"
            trace.increasing.fillcolor = "#00cc96"
            trace.decreasing.line.color = "#ef553b"
            trace.decreasing.fillcolor = "#ef553b"


def _coerce_ts_like_column(
    value: str | pd.Timestamp | None,
    ts: pd.Series,
) -> pd.Timestamp | None:
    if value is None:
        return None

    out = pd.Timestamp(value)

    # If column is tz-aware, localize/convert boundary to same tz.
    col_tz = getattr(ts.dt, "tz", None)
    if col_tz is not None:
        if out.tzinfo is None:
            out = out.tz_localize(col_tz)
        else:
            out = out.tz_convert(col_tz)
    else:
        # If column is tz-naive, strip timezone from boundary if needed.
        if out.tzinfo is not None:
            out = out.tz_localize(None)

    return out


def _filter_timeframe(
    df: pd.DataFrame,
    *,
    start_ts: str | pd.Timestamp | None = None,
    end_ts: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = df.copy()

    start_bound = _coerce_ts_like_column(start_ts, out["timestamp"])
    end_bound = _coerce_ts_like_column(end_ts, out["timestamp"])

    if start_bound is not None:
        out = out[out["timestamp"] >= start_bound]
    if end_bound is not None:
        out = out[out["timestamp"] <= end_bound]

    return out


def plot_swings_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Swings Validation",
    start: int | None = None,
    end: int | None = None,
    start_ts: str | pd.Timestamp | None = None,
    end_ts: str | pd.Timestamp | None = None,
) -> Path:
    """
    Create a visual validation chart for retrace-confirmed swings.

    Notes
    -----
    - Origin markers show where the confirmed pivot actually occurred.
    - Confirm markers show when that same pivot became knowable.
    - These are NOT extra swings; they are two timestamps for the same swing.
    """
    view = slice_view(df, start=start, end=end)
    view = _filter_timeframe(view, start_ts=start_ts, end_ts=end_ts)

    fig = create_candlestick_figure(view, title=title)

    # Origin markers: same confirmed swings, plotted at pivot bar
    add_scatter_markers(
        fig,
        view,
        y_col="high",
        mask_col="swing_high",
        name="Swing High (Origin)",
        symbol="triangle-down",
        size=11,
    )
    add_scatter_markers(
        fig,
        view,
        y_col="low",
        mask_col="swing_low",
        name="Swing Low (Origin)",
        symbol="triangle-up",
        size=11,
    )

    # Confirm markers: same confirmed swings, plotted at confirmation bar
    add_scatter_markers(
        fig,
        view,
        y_col="swing_high_confirm_price",
        mask_col="swing_high_confirm_flag",
        name="Swing High (Confirm)",
        symbol="x",
        size=9,
    )
    add_scatter_markers(
        fig,
        view,
        y_col="swing_low_confirm_price",
        mask_col="swing_low_confirm_flag",
        name="Swing Low (Confirm)",
        symbol="x",
        size=9,
    )

    # Running confirmed levels
    add_line_series(
        fig,
        view,
        y_col="last_swing_high",
        name="Last Confirmed Swing High",
        dash="dot",
    )
    add_line_series(
        fig,
        view,
        y_col="last_swing_low",
        name="Last Confirmed Swing Low",
        dash="dot",
    )

    _apply_compare_style(fig)
    return save_figure_html(fig, outpath)


def validate_swings(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Swings Validation",
    start: int | None = None,
    end: int | None = None,
    start_ts: str | pd.Timestamp | None = None,
    end_ts: str | pd.Timestamp | None = None,
    n_windows: int = 3,
) -> dict[str, object]:
    """Run both numeric and visual validation for swings."""
    summary = summarize_swings(df)

    high_windows = swing_event_windows(df, side="high", limit=n_windows)
    low_windows = swing_event_windows(df, side="low", limit=n_windows)

    html_path = plot_swings_validation(
        df,
        outpath=outpath,
        title=title,
        start=start,
        end=end,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    return {
        "summary": summary,
        "high_windows": high_windows,
        "low_windows": low_windows,
        "html_path": html_path,
    }
