from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CORR_RE = re.compile(r"^corr_(.+)__(.+)__w(\d+)$")
_CORR_Z_RE = re.compile(r"^corr_z_(.+)__(.+)__w(\d+)$")
_LAG_RE = re.compile(r"^lagcorr_best_(.+)__(.+)__w(\d+)$")
_LAG_LAG_RE = re.compile(r"^lagcorr_best_lag_(.+)__(.+)__w(\d+)$")


def _parse(pattern: re.Pattern[str], column: str) -> tuple[str, str, int] | None:
    match = pattern.match(column)
    if match is None:
        return None
    left, right, window = match.groups()
    return left, right, int(window)


def build_cross_asset_correlation_audit(
    frame: pd.DataFrame,
    market_context: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if market_context.empty:
        return {
            "pairwise_snapshot": pd.DataFrame(),
            "stability_summary": pd.DataFrame(),
            "lag_summary": pd.DataFrame(),
            "session_stratification": pd.DataFrame(),
            "regime_stratification": pd.DataFrame(),
        }

    context = market_context.copy()
    context["timestamp"] = pd.to_datetime(
        context["timestamp"], utc=True, errors="coerce"
    )

    corr_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    latest_context = context.sort_values("timestamp").iloc[-1]
    for column in context.columns:
        parsed = _parse(_CORR_RE, column)
        if parsed is None:
            continue
        left, right, window = parsed
        z_column = f"corr_z_{left}__{right}__w{window}"
        series = pd.to_numeric(context[column], errors="coerce")
        latest_valid = series.dropna()
        latest_value = latest_valid.iloc[-1] if not latest_valid.empty else np.nan
        latest_z = (
            pd.to_numeric(context[z_column], errors="coerce").dropna().iloc[-1]
            if z_column in context.columns
            and not pd.to_numeric(context[z_column], errors="coerce").dropna().empty
            else np.nan
        )
        corr_rows.append(
            {
                "timestamp": latest_context["timestamp"],
                "pair": f"{left}__{right}",
                "left": left,
                "right": right,
                "window": window,
                "correlation": latest_value,
                "correlation_zscore": latest_z,
            }
        )
        stability_rows.append(
            {
                "pair": f"{left}__{right}",
                "left": left,
                "right": right,
                "window": window,
                "observations": int(series.notna().sum()),
                "mean_correlation": float(series.mean()),
                "std_correlation": float(series.std()),
                "min_correlation": float(series.min()),
                "max_correlation": float(series.max()),
                "positive_rate": float((series > 0).mean()),
            }
        )

    lag_rows: list[dict[str, object]] = []
    for column in context.columns:
        parsed = _parse(_LAG_RE, column)
        if parsed is None:
            continue
        left, right, window = parsed
        lag_column = f"lagcorr_best_lag_{left}__{right}__w{window}"
        score_series = pd.to_numeric(context[column], errors="coerce")
        lag_series = (
            pd.to_numeric(context[lag_column], errors="coerce")
            if lag_column in context.columns
            else pd.Series(np.nan, index=context.index, dtype=float)
        )
        latest_score = (
            score_series.dropna().iloc[-1]
            if not score_series.dropna().empty
            else np.nan
        )
        latest_lag = (
            lag_series.dropna().iloc[-1] if not lag_series.dropna().empty else np.nan
        )
        lag_rows.append(
            {
                "pair": f"{left}__{right}",
                "left": left,
                "right": right,
                "window": window,
                "observations": int(score_series.notna().sum()),
                "latest_best_corr": latest_score,
                "latest_best_lag": latest_lag,
                "mean_abs_best_corr": float(score_series.abs().mean()),
                "mean_abs_best_lag": float(lag_series.abs().mean()),
            }
        )

    xasset_metric_columns = [
        column
        for column in frame.columns
        if column.startswith("xasset_corr_")
        or column.startswith("xasset_corr_z_")
        or column.startswith("xasset_lagcorr_")
    ]
    session_stratification = pd.DataFrame()
    if "session_name" in frame.columns and xasset_metric_columns:
        session_stratification = (
            frame[["session_name", *xasset_metric_columns]]
            .groupby("session_name", dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )

    regime_stratification = pd.DataFrame()
    if "regime" in frame.columns and xasset_metric_columns:
        regime_stratification = (
            frame[["regime", *xasset_metric_columns]]
            .groupby("regime", dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )

    return {
        "pairwise_snapshot": pd.DataFrame(corr_rows),
        "stability_summary": pd.DataFrame(stability_rows),
        "lag_summary": pd.DataFrame(lag_rows),
        "session_stratification": session_stratification,
        "regime_stratification": regime_stratification,
    }


def summarize_cross_asset_correlation_audit(
    audit_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    pairwise_snapshot = audit_tables.get("pairwise_snapshot", pd.DataFrame())
    stability_summary = audit_tables.get("stability_summary", pd.DataFrame())
    lag_summary = audit_tables.get("lag_summary", pd.DataFrame())
    return {
        "pairwise_snapshot_rows": int(len(pairwise_snapshot)),
        "stability_summary_rows": int(len(stability_summary)),
        "lag_summary_rows": int(len(lag_summary)),
        "pairs": sorted(
            pairwise_snapshot.get("pair", pd.Series(dtype=object))
            .dropna()
            .unique()
            .tolist()
        ),
        "lag_pairs": sorted(
            lag_summary.get("pair", pd.Series(dtype=object)).dropna().unique().tolist()
        ),
    }
