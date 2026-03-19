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


def _state_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if len(values) == 0:
        return runs

    start = 0
    cur = int(values[0])

    for i in range(1, len(values)):
        val = int(values[i])
        if val != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = val

    runs.append((start, len(values) - 1, cur))
    return runs


def _add_state_background(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    state_col: str,
    opacity: float,
    max_runs: int = 250,
) -> None:
    """
    Add lightweight regime shading for strict trend state only.

    Important:
    - Only intended for low-frequency regime states like trend_state.
    - Do NOT use for rapidly flipping bias states.
    """
    if state_col != "trend_state":
        return
    if state_col not in df.columns:
        return

    values = df[state_col].fillna(0).astype(int).to_numpy()
    x = df["timestamp"].reset_index(drop=True)

    runs = _state_runs(values)
    non_zero_runs = [(s, e, v) for s, e, v in runs if v != 0]

    if len(non_zero_runs) > max_runs:
        step = int(np.ceil(len(non_zero_runs) / max_runs))
        non_zero_runs = non_zero_runs[::step]

    for start, end, state in non_zero_runs:
        color = (
            f"rgba(0, 180, 0, {opacity})"
            if state == 1
            else f"rgba(200, 0, 0, {opacity})"
        )

        fig.add_vrect(
            x0=x.iloc[start],
            x1=x.iloc[end],
            fillcolor=color,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )


def _transition_table(series: pd.Series, from_name: str, to_name: str) -> pd.DataFrame:
    prev = series.shift(1).fillna(0).astype(int)
    cur = series.fillna(0).astype(int)
    changed = prev != cur

    tbl = (
        pd.DataFrame({from_name: prev[changed], to_name: cur[changed]})
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return tbl


def _duration_stats(series: pd.Series) -> pd.DataFrame:
    vals = series.fillna(0).astype(int).to_numpy()
    runs = _state_runs(vals)

    rows: list[dict[str, float]] = []
    for start, end, state in runs:
        rows.append({"state": state, "duration": end - start + 1})

    if not rows:
        return pd.DataFrame(columns=["count", "mean", "median", "max"])

    runs_df = pd.DataFrame(rows)
    return (
        runs_df.groupby("state")["duration"]
        .agg(["count", "mean", "median", "max"])
        .sort_index()
    )


def _sample_transition_windows(
    df: pd.DataFrame,
    *,
    transition_col: str,
    n_windows: int,
    pad: int = 6,
) -> list[pd.DataFrame]:
    if transition_col not in df.columns:
        return []

    idxs = df.index[df[transition_col].fillna(0).astype(int) == 1].tolist()
    windows: list[pd.DataFrame] = []

    for idx in idxs[:n_windows]:
        start = max(0, idx - pad)
        end = min(len(df), idx + pad + 1)
        win = df.iloc[start:end].copy()
        win["event_row"] = 0
        win.loc[win.index == idx, "event_row"] = 1
        windows.append(win)

    return windows


def validate_trend_state(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Trend State Validation",
    n_windows: int = 5,
) -> dict:
    out = _ensure_datetime(df)

    strict = out["trend_state"].fillna(0).astype(int)
    bias = (
        out["trend_bias_state"].fillna(0).astype(int)
        if "trend_bias_state" in out.columns
        else pd.Series(0, index=out.index)
    )
    conf = (
        out["trend_confidence"].fillna(-1).astype(int)
        if "trend_confidence" in out.columns
        else pd.Series(-1, index=out.index)
    )

    strict_state_counts = strict.value_counts().sort_index().to_dict()
    strict_state_pct = strict.value_counts(normalize=True).sort_index().to_dict()

    bias_state_counts = bias.value_counts().sort_index().to_dict()
    bias_state_pct = bias.value_counts(normalize=True).sort_index().to_dict()

    transition_count = int(
        out.get("trend_state_changed", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        .sum()
    )

    avg_bias_score_by_strict_state = (
        out.groupby(strict)["trend_bias_score"].mean().to_dict()
        if "trend_bias_score" in out.columns
        else {}
    )

    avg_age_by_strict_state = (
        out.groupby(strict)["trend_state_age"].mean().to_dict()
        if "trend_state_age" in out.columns
        else {}
    )

    avg_strength_raw_by_strict_state = (
        out.groupby(strict)["trend_strength_raw"].mean().to_dict()
        if "trend_strength_raw" in out.columns
        else {}
    )

    avg_strength_ema_by_strict_state = (
        out.groupby(strict)["trend_strength_ema"].mean().to_dict()
        if "trend_strength_ema" in out.columns
        else {}
    )

    strict_neutral_rows = int((strict == 0).sum())
    bias_carry_rows = int(((strict == 0) & (bias != 0)).sum())

    strict_bull_not_ready_rows = (
        int(((strict == 1) & (out["trend_bull_ready"] != 1)).sum())
        if "trend_bull_ready" in out.columns
        else 0
    )
    strict_bear_not_ready_rows = (
        int(((strict == -1) & (out["trend_bear_ready"] != 1)).sum())
        if "trend_bear_ready" in out.columns
        else 0
    )

    strict_bull_fresh_rows = int(((strict == 1) & (conf == 2)).sum())
    strict_bull_intact_rows = int(((strict == 1) & (conf == 1)).sum())
    strict_bear_fresh_rows = int(((strict == -1) & (conf == 2)).sum())
    strict_bear_intact_rows = int(((strict == -1) & (conf == 1)).sum())

    confidence_distribution = (
        pd.DataFrame({"trend_state": strict, "trend_confidence": conf})
        .value_counts()
        .reset_index(name="count")
        .sort_values(["trend_state", "trend_confidence"])
        .reset_index(drop=True)
    )

    transitions_table = _transition_table(strict, "trend_state_from", "trend_state_to")
    bias_transitions_table = _transition_table(bias, "trend_bias_from", "trend_bias_to")
    duration_stats = _duration_stats(strict)

    structure_loss_bull_rows = int(
        out.get("trend_structure_loss_bull", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    structure_loss_bear_rows = int(
        out.get("trend_structure_loss_bear", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    emerging_bull_rows = int(
        out.get("trend_emerging_bull", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    emerging_bear_rows = int(
        out.get("trend_emerging_bear", pd.Series(0, index=out.index))
        .fillna(0)
        .astype(int)
        .sum()
    )

    regime_phase_counts = (
        out["trend_regime_phase"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if "trend_regime_phase" in out.columns
        else {}
    )

    transition_windows = _sample_transition_windows(
        out,
        transition_col="trend_state_changed",
        n_windows=n_windows,
    )

    html_path = plot_trend_state_validation(
        out,
        outpath=outpath,
        title=title,
    )

    summary = {
        "strict_state_counts": strict_state_counts,
        "strict_state_pct": strict_state_pct,
        "bias_state_counts": bias_state_counts,
        "bias_state_pct": bias_state_pct,
        "transition_count": transition_count,
        "avg_bias_score_by_strict_state": avg_bias_score_by_strict_state,
        "avg_age_by_strict_state": avg_age_by_strict_state,
        "avg_strength_raw_by_strict_state": avg_strength_raw_by_strict_state,
        "avg_strength_ema_by_strict_state": avg_strength_ema_by_strict_state,
        "strict_neutral_rows": strict_neutral_rows,
        "bias_carry_rows": bias_carry_rows,
        "strict_bull_not_ready_rows": strict_bull_not_ready_rows,
        "strict_bear_not_ready_rows": strict_bear_not_ready_rows,
        "strict_bull_fresh_rows": strict_bull_fresh_rows,
        "strict_bull_intact_rows": strict_bull_intact_rows,
        "strict_bear_fresh_rows": strict_bear_fresh_rows,
        "strict_bear_intact_rows": strict_bear_intact_rows,
        "confidence_distribution": confidence_distribution,
        "transitions_table": transitions_table,
        "bias_transitions_table": bias_transitions_table,
        "duration_stats": duration_stats,
        "structure_loss_bull_rows": structure_loss_bull_rows,
        "structure_loss_bear_rows": structure_loss_bear_rows,
        "emerging_bull_rows": emerging_bull_rows,
        "emerging_bear_rows": emerging_bear_rows,
        "regime_phase_counts": regime_phase_counts,
    }

    return {
        "summary": summary,
        "transition_windows": transition_windows,
        "html_path": html_path,
    }


def plot_trend_state_validation(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Trend State Validation",
) -> Path:
    out = _ensure_datetime(df).copy()

    if "trend_bias_score_live" in out.columns:
        out["trend_bias_score_live"] = pd.to_numeric(
            out["trend_bias_score_live"], errors="coerce"
        )
    if "trend_strength_raw" in out.columns:
        out["trend_strength_raw"] = pd.to_numeric(
            out["trend_strength_raw"], errors="coerce"
        )
    if "trend_strength_ema" in out.columns:
        out["trend_strength_ema"] = pd.to_numeric(
            out["trend_strength_ema"], errors="coerce"
        )

    outpath = Path(outpath)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.56, 0.14, 0.12, 0.18],
        subplot_titles=(
            "Price + Swings + Strict Regime",
            "Strict State vs Bias State",
            "Trend Confidence",
            "Bias / Strength Scores",
        ),
    )

    # Panel 1: price
    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    _add_state_background(
        fig,
        out,
        row=1,
        col=1,
        state_col="trend_state",
        opacity=0.10,
        max_runs=180,
    )

    if "swing_high" in out.columns:
        sh = out[out["swing_high"] == 1]
        fig.add_trace(
            go.Scatter(
                x=sh["timestamp"],
                y=sh["high"],
                mode="markers",
                name="Swing High",
                marker=dict(symbol="triangle-down", size=8),
            ),
            row=1,
            col=1,
        )

    if "swing_low" in out.columns:
        sl = out[out["swing_low"] == 1]
        fig.add_trace(
            go.Scatter(
                x=sl["timestamp"],
                y=sl["low"],
                mode="markers",
                name="Swing Low",
                marker=dict(symbol="triangle-up", size=8),
            ),
            row=1,
            col=1,
        )

    if "trend_state_changed" in out.columns:
        tc = out[out["trend_state_changed"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tc["timestamp"],
                y=tc["close"],
                mode="markers",
                name="Strict Transition",
                marker=dict(symbol="x", size=10),
            ),
            row=1,
            col=1,
        )

    if "trend_bias_changed" in out.columns:
        bc = out[out["trend_bias_changed"] == 1]
        fig.add_trace(
            go.Scatter(
                x=bc["timestamp"],
                y=bc["close"],
                mode="markers",
                name="Bias Transition",
                marker=dict(symbol="diamond", size=7),
            ),
            row=1,
            col=1,
        )

    if "trend_structure_loss_bull" in out.columns:
        tlb = out[out["trend_structure_loss_bull"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tlb["timestamp"],
                y=tlb["close"],
                mode="markers",
                name="Bull Structure Loss",
                marker=dict(symbol="circle-open", size=9),
            ),
            row=1,
            col=1,
        )

    if "trend_structure_loss_bear" in out.columns:
        tlbr = out[out["trend_structure_loss_bear"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tlbr["timestamp"],
                y=tlbr["close"],
                mode="markers",
                name="Bear Structure Loss",
                marker=dict(symbol="circle-open", size=9),
            ),
            row=1,
            col=1,
        )

    if "trend_emerging_bull" in out.columns:
        eb = out[out["trend_emerging_bull"] == 1]
        fig.add_trace(
            go.Scatter(
                x=eb["timestamp"],
                y=eb["close"],
                mode="markers",
                name="Emerging Bull",
                marker=dict(symbol="star", size=9),
            ),
            row=1,
            col=1,
        )

    if "trend_emerging_bear" in out.columns:
        er = out[out["trend_emerging_bear"] == 1]
        fig.add_trace(
            go.Scatter(
                x=er["timestamp"],
                y=er["close"],
                mode="markers",
                name="Emerging Bear",
                marker=dict(symbol="star", size=9),
            ),
            row=1,
            col=1,
        )

    # Panel 2: strict + bias state
    if "trend_state" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_state"].astype(float),
                mode="lines",
                name="Strict State",
                line=dict(width=2),
            ),
            row=2,
            col=1,
        )

    if "trend_bias_state" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_bias_state"].astype(float),
                mode="lines",
                name="Bias State",
                line=dict(width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0.0, line_dash="dot", row=2, col=1)

    # Panel 3: confidence
    if "trend_confidence" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_confidence"].astype(float),
                mode="lines",
                name="Confidence",
                line=dict(width=2),
            ),
            row=3,
            col=1,
        )

    fig.add_hline(y=0.0, line_dash="dot", row=3, col=1)
    fig.add_hline(y=1.0, line_dash="dot", row=3, col=1)
    fig.add_hline(y=2.0, line_dash="dot", row=3, col=1)

    # Panel 4: bias / strength scores
    if "trend_bias_score_live" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_bias_score_live"],
                mode="lines",
                name="Bias Score Live",
                line=dict(width=2),
            ),
            row=4,
            col=1,
        )

    if "trend_strength_raw" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_strength_raw"],
                mode="lines",
                name="Trend Strength Raw",
                line=dict(width=1),
            ),
            row=4,
            col=1,
        )

    if "trend_strength_ema" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_strength_ema"],
                mode="lines",
                name="Trend Strength EMA",
                line=dict(width=3),
            ),
            row=4,
            col=1,
        )

    fig.add_hline(y=0.0, line_dash="dot", row=4, col=1)

    # Axes / layout
    fig.update_yaxes(title_text="Price", row=1, col=1)

    fig.update_yaxes(
        title_text="State",
        row=2,
        col=1,
        tickmode="array",
        tickvals=[-1, 0, 1],
        range=[-1.25, 1.25],
    )

    fig.update_yaxes(
        title_text="Conf",
        row=3,
        col=1,
        tickmode="array",
        tickvals=[-1, 0, 1, 2],
        range=[-1.25, 2.25],
    )

    fig.update_yaxes(
        title_text="Bias / Strength",
        row=4,
        col=1,
        range=[-1.05, 1.05],
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath
