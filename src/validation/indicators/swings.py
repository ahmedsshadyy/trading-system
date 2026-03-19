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
    """Return numeric summary statistics for canonical causal swings."""
    required = {
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "last_swing_high",
        "last_swing_low",
        "last_swing_high_idx",
        "last_swing_low_idx",
        "swing_high_age",
        "swing_low_age",
        "swing_high_prominence_atr",
        "swing_low_prominence_atr",
        "swing_high_detect_flag",
        "swing_low_detect_flag",
        "swing_high_detect_idx",
        "swing_low_detect_idx",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing swing columns: {sorted(missing)}")

    n = len(df)
    sh_rows = df[df["swing_high"] == 1]
    sl_rows = df[df["swing_low"] == 1]

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

    # Mechanical consistency checks
    high_sync_ok = (
        bool(
            (
                sh_rows["last_swing_high"].to_numpy()
                == sh_rows["swing_high_price"].to_numpy()
            ).all()
        )
        if len(sh_rows)
        else True
    )

    low_sync_ok = (
        bool(
            (
                sl_rows["last_swing_low"].to_numpy()
                == sl_rows["swing_low_price"].to_numpy()
            ).all()
        )
        if len(sl_rows)
        else True
    )

    high_age_zero_ok = (
        bool((sh_rows["swing_high_age"] == 0).all()) if len(sh_rows) else True
    )
    low_age_zero_ok = (
        bool((sl_rows["swing_low_age"] == 0).all()) if len(sl_rows) else True
    )

    high_detect_same_bar_ok = (
        bool(
            (sh_rows["swing_high_detect_flag"] == 1).all()
            and (
                sh_rows["swing_high_detect_idx"].to_numpy() == sh_rows.index.to_numpy()
            ).all()
        )
        if len(sh_rows)
        else True
    )

    low_detect_same_bar_ok = (
        bool(
            (sl_rows["swing_low_detect_flag"] == 1).all()
            and (
                sl_rows["swing_low_detect_idx"].to_numpy() == sl_rows.index.to_numpy()
            ).all()
        )
        if len(sl_rows)
        else True
    )

    return {
        "n_rows": n,
        "swing_high_count": int(len(sh_rows)),
        "swing_low_count": int(len(sl_rows)),
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
        "last_swing_high_updates_on_swing_bar": high_sync_ok,
        "last_swing_low_updates_on_swing_bar": low_sync_ok,
        "swing_high_age_zero_on_swing_bar": high_age_zero_ok,
        "swing_low_age_zero_on_swing_bar": low_age_zero_ok,
        "swing_high_detect_same_bar": high_detect_same_bar_ok,
        "swing_low_detect_same_bar": low_detect_same_bar_ok,
    }


def swing_event_windows(
    df: pd.DataFrame,
    side: str = "high",
    bars_before: int = 4,
    bars_after: int = 4,
    limit: int = 10,
) -> list[pd.DataFrame]:
    """Return small windows around swing events for manual inspection."""
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


def plot_swings_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Swings Validation",
    start: int | None = None,
    end: int | None = None,
) -> Path:
    """Create a visual validation chart for swings."""
    view = slice_view(df, start=start, end=end)

    fig = create_candlestick_figure(view, title=title)

    add_scatter_markers(
        fig,
        view,
        y_col="high",
        mask_col="swing_high",
        name="Swing High",
        symbol="triangle-down",
        size=10,
    )
    add_scatter_markers(
        fig,
        view,
        y_col="low",
        mask_col="swing_low",
        name="Swing Low",
        symbol="triangle-up",
        size=10,
    )

    add_line_series(
        fig,
        view,
        y_col="last_swing_high",
        name="Last Swing High",
        dash="dot",
    )
    add_line_series(
        fig,
        view,
        y_col="last_swing_low",
        name="Last Swing Low",
        dash="dot",
    )

    return save_figure_html(fig, outpath)


def validate_swings(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "Swings Validation",
    start: int | None = None,
    end: int | None = None,
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
    )

    return {
        "summary": summary,
        "high_windows": high_windows,
        "low_windows": low_windows,
        "html_path": html_path,
    }
