from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators.research.cross_asset_research import (
    build_cross_asset_correlation_audit,
    summarize_cross_asset_correlation_audit,
)


def _metric_matrix(
    matrix_long: pd.DataFrame,
    *,
    context_type: str,
    context_value: str,
    window: int,
    return_mode: str,
    matrix_family: str,
    metric: str,
    lag: int | None = None,
) -> pd.DataFrame:
    subset = matrix_long[
        (matrix_long["context_type"].astype(str) == context_type)
        & (matrix_long["context_value"].astype(str) == str(context_value))
        & (pd.to_numeric(matrix_long["window"], errors="coerce") == int(window))
        & (matrix_long["return_mode"].astype(str) == return_mode)
        & (matrix_long["matrix_family"].astype(str) == matrix_family)
    ].copy()
    if metric != "selected_lag":
        subset = subset[subset["metric"].astype(str) == metric]
    if lag is not None:
        subset = subset[pd.to_numeric(subset["lag"], errors="coerce") == int(lag)]
    if subset.empty:
        return pd.DataFrame()
    if metric == "selected_lag":
        subset = subset.drop_duplicates(subset=["left", "right", "lag"])
    matrix = subset.pivot(index="left", columns="right", values="value")
    labels = sorted(set(matrix.index).union(matrix.columns))
    square = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for _, row in subset.iterrows():
        left = str(row["left"])
        right = str(row["right"])
        value = (
            float(row["lag"])
            if metric == "selected_lag"
            else (float(row["value"]) if pd.notna(row["value"]) else np.nan)
        )
        square.loc[left, right] = value
        square.loc[right, left] = value
    diagonal_value = 1.0 if metric.endswith("correlation") else np.nan
    for label in labels:
        square.loc[label, label] = diagonal_value
    return square


def _heatmap_figure(
    matrix: pd.DataFrame,
    *,
    title: str,
    zmid: float | None = None,
    colorscale: str = "RdBu",
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix.to_numpy(dtype=float),
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale=colorscale,
                zmid=zmid,
                colorbar=dict(title=title),
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
        width=850,
    )
    return fig


def summarize_cross_asset_research(
    audit_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    summary = summarize_cross_asset_correlation_audit(audit_tables)
    ranking = audit_tables.get("lead_lag_candidate_ranking", pd.DataFrame())
    diagnostics = audit_tables.get("matrix_diagnostics", pd.DataFrame())
    summary["top_candidates"] = (
        ranking[
            [
                "pair",
                "window",
                "return_mode",
                "best_lag_effect",
                "best_lag",
                "candidate_classification",
            ]
        ]
        .head(10)
        .to_dict(orient="records")
        if not ranking.empty
        else []
    )
    summary["diagnostic_metrics"] = (
        diagnostics["metric"].dropna().astype(str).unique().tolist()
        if not diagnostics.empty
        else []
    )
    return summary


def build_cross_asset_matrix_views(
    audit_tables: dict[str, pd.DataFrame],
    *,
    return_mode: str = "vol_norm",
) -> dict[str, go.Figure]:
    matrix_long = audit_tables.get("matrix_long", pd.DataFrame())
    if matrix_long.empty:
        return {}
    global_rows = matrix_long[matrix_long["context_type"].astype(str) == "global"]
    if global_rows.empty:
        return {}
    latest = global_rows.sort_values("timestamp").iloc[-1]
    context_value = str(latest["context_value"])
    window = int(latest["window"])
    figures: dict[str, go.Figure] = {}

    contemporaneous = _metric_matrix(
        matrix_long,
        context_type="global",
        context_value=context_value,
        window=window,
        return_mode=return_mode,
        matrix_family="contemporaneous",
        metric="pearson_correlation",
        lag=0,
    )
    if not contemporaneous.empty:
        figures["contemporaneous"] = _heatmap_figure(
            contemporaneous,
            title=f"Contemporaneous Correlation Matrix ({return_mode}, w={window})",
            zmid=0.0,
        )

    lag1 = _metric_matrix(
        matrix_long,
        context_type="global",
        context_value=context_value,
        window=window,
        return_mode=return_mode,
        matrix_family="lag_1",
        metric="pearson_correlation",
        lag=1,
    )
    if not lag1.empty:
        figures["lag_1"] = _heatmap_figure(
            lag1,
            title=f"Lag-1 Lead-Lag Matrix ({return_mode}, w={window})",
            zmid=0.0,
        )

    best_lag = _metric_matrix(
        matrix_long,
        context_type="global",
        context_value=context_value,
        window=window,
        return_mode=return_mode,
        matrix_family="best_lag",
        metric="selected_lag",
    )
    if not best_lag.empty:
        figures["best_lag"] = _heatmap_figure(
            best_lag,
            title=f"Best-Lag Matrix ({return_mode}, w={window})",
            zmid=0.0,
            colorscale="Viridis",
        )

    significance = _metric_matrix(
        matrix_long,
        context_type="global",
        context_value=context_value,
        window=window,
        return_mode=return_mode,
        matrix_family="best_lag",
        metric="hac_t_stat",
    )
    if not significance.empty:
        figures["hac_significance"] = _heatmap_figure(
            significance,
            title=f"HAC t-stat Matrix ({return_mode}, w={window})",
            zmid=0.0,
        )

    stability = _metric_matrix(
        matrix_long,
        context_type="global",
        context_value=context_value,
        window=window,
        return_mode=return_mode,
        matrix_family="stability_zscore",
        metric="stability_zscore",
    )
    if not stability.empty:
        figures["stability"] = _heatmap_figure(
            stability,
            title=f"Stability Z-score Matrix ({return_mode}, w={window})",
            zmid=0.0,
        )
    return figures


def validate_cross_asset(
    frame: pd.DataFrame,
    *,
    market_context: pd.DataFrame,
    instrument: str,
    timeframe: str,
    audit_tables: dict[str, pd.DataFrame] | None = None,
    outpath: str | Path | None = None,
    title: str = "Cross-Asset Validation",
) -> dict[str, object]:
    tables = (
        audit_tables
        if audit_tables is not None
        else build_cross_asset_correlation_audit(
            frame,
            market_context,
            instrument=instrument,
            timeframe=timeframe,
        )
    )
    summary = summarize_cross_asset_research(tables)
    figures = build_cross_asset_matrix_views(tables, return_mode="vol_norm")
    html_path: Path | None = None
    if outpath is not None:
        html_path = Path(outpath)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        ranking = tables.get("lead_lag_candidate_ranking", pd.DataFrame()).head(25)
        diagnostics = tables.get("matrix_diagnostics", pd.DataFrame())
        sections = [
            "<html><head><meta charset='utf-8'><title>"
            + title
            + "</title></head><body>",
            f"<h1>{title}</h1>",
            f"<h2>{instrument} {timeframe}</h2>",
            "<h3>Summary</h3>",
            pd.DataFrame([summary]).to_html(index=False),
        ]
        for name, fig in figures.items():
            sections.append(f"<h3>{name.replace('_', ' ').title()}</h3>")
            sections.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        if not ranking.empty:
            sections.append("<h3>Lead-Lag Candidate Ranking</h3>")
            sections.append(ranking.to_html(index=False))
        if not diagnostics.empty:
            sections.append("<h3>Matrix Diagnostics</h3>")
            sections.append(diagnostics.head(50).to_html(index=False))
        sections.append("</body></html>")
        html_path.write_text("\n".join(sections), encoding="utf-8")
    return {
        "summary": summary,
        "audit_tables": tables,
        "figures": figures,
        "html_path": html_path,
    }
