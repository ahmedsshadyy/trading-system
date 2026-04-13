from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators.research.cross_asset_research import (
    build_cross_asset_correlation_audit,
    summarize_cross_asset_correlation_audit,
)
from src.indicators.research.smt_research import (
    build_smt_research_table,
    summarize_smt_research,
)

_SMT_DIR_RE = re.compile(r"^xasset_smt_(.+)_dir$")
_CORR_RE = re.compile(r"^xasset_corr_(.+)_w(\d+)$")
_LAG_RE = re.compile(r"^xasset_lagcorr_(.+)_best_w(\d+)$")

REQUIRED_BASE_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "xasset_smt_any_flag",
    "xasset_smt_best_partner",
    "xasset_smt_best_score",
}

PARTNER_COLORS = {
    "dxy": "#b91c1c",
    "usd_jpy": "#0369a1",
    "usd_cad": "#7c3aed",
    "xau_usd": "#a16207",
    "usoil": "#15803d",
}


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


def _partner_tokens(df: pd.DataFrame) -> list[str]:
    partners: list[str] = []
    for column in df.columns:
        match = _SMT_DIR_RE.match(column)
        if match is None:
            continue
        partners.append(match.group(1))
    return sorted(set(partners))


def _corr_windows_by_partner(df: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for column in df.columns:
        match = _CORR_RE.match(column)
        if match is None:
            continue
        partner, window = match.groups()
        out.setdefault(partner, []).append(int(window))
    return {partner: sorted(set(windows)) for partner, windows in out.items()}


def _lag_windows(df: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for column in df.columns:
        match = _LAG_RE.match(column)
        if match is None:
            continue
        pair_alias, window = match.groups()
        out.setdefault(pair_alias, []).append(int(window))
    return {pair: sorted(set(windows)) for pair, windows in out.items()}


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing SMT validation columns: {sorted(missing)}")


def summarize_smt(
    df: pd.DataFrame,
    *,
    market_context: pd.DataFrame | None = None,
    research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    _validate_required_columns(df)

    partners = _partner_tokens(df)
    corr_windows = _corr_windows_by_partner(df)
    lag_windows = _lag_windows(df)
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    best_score = pd.to_numeric(df["xasset_smt_best_score"], errors="coerce")
    any_flag = pd.to_numeric(df["xasset_smt_any_flag"], errors="coerce").fillna(0)

    by_partner: dict[str, object] = {}
    for partner in partners:
        bull_col = f"xasset_smt_{partner}_bull_flag"
        bear_col = f"xasset_smt_{partner}_bear_flag"
        dir_col = f"xasset_smt_{partner}_dir"
        score_col = f"xasset_smt_{partner}_score"
        relation_col = f"xasset_smt_{partner}_expected_relation"
        bull = (
            pd.to_numeric(df[bull_col], errors="coerce").fillna(0)
            if bull_col in df.columns
            else pd.Series(0, index=df.index, dtype=float)
        )
        bear = (
            pd.to_numeric(df[bear_col], errors="coerce").fillna(0)
            if bear_col in df.columns
            else pd.Series(0, index=df.index, dtype=float)
        )
        direction = (
            pd.to_numeric(df[dir_col], errors="coerce").fillna(0)
            if dir_col in df.columns
            else pd.Series(0, index=df.index, dtype=float)
        )
        score = (
            pd.to_numeric(df[score_col], errors="coerce")
            if score_col in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        relation = (
            pd.to_numeric(df[relation_col], errors="coerce")
            if relation_col in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        by_partner[partner] = {
            "bull_count": int((bull == 1).sum()),
            "bear_count": int((bear == 1).sum()),
            "event_count": int((direction != 0).sum()),
            "direction_distribution": _value_counts(direction[direction != 0]),
            "expected_relation_distribution": _value_counts(relation.dropna()),
            "score_stats": _continuous_stats(score),
            "correlation_windows": corr_windows.get(partner, []),
        }

    corr_columns = sorted(column for column in df.columns if _CORR_RE.match(column))
    lag_columns = sorted(column for column in df.columns if _LAG_RE.match(column))
    corr_availability = {
        column: int(pd.to_numeric(df[column], errors="coerce").notna().sum())
        for column in corr_columns
    }
    lag_availability = {
        column: int(pd.to_numeric(df[column], errors="coerce").notna().sum())
        for column in lag_columns
    }

    smt_research = (
        research_table.copy()
        if research_table is not None
        else build_smt_research_table(df)
    )
    correlation_audit = (
        build_cross_asset_correlation_audit(df, market_context)
        if market_context is not None and not market_context.empty
        else None
    )

    summary = {
        "window": {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
            "rows": int(len(df)),
        },
        "schema": {
            "smt_partner_count": int(len(partners)),
            "smt_partners": partners,
            "corr_partner_windows": corr_windows,
            "lag_pair_windows": lag_windows,
        },
        "event_counts": {
            "smt_any_count": int((any_flag == 1).sum()),
            "best_partner_distribution": _value_counts(df["xasset_smt_best_partner"]),
        },
        "best_score_stats": _continuous_stats(best_score),
        "by_partner": by_partner,
        "feature_availability": {
            "corr_columns": corr_availability,
            "lag_columns": lag_availability,
        },
        "sanity_checks": {
            "best_score_in_unit_interval": bool(
                best_score.dropna().between(0.0, 1.0).all()
                if best_score.notna().any()
                else True
            ),
            "any_flag_matches_partner_dirs": bool(
                (
                    any_flag.to_numpy(dtype=int)
                    == np.where(
                        (
                            np.any(
                                np.column_stack(
                                    [
                                        pd.to_numeric(
                                            df[f"xasset_smt_{partner}_dir"],
                                            errors="coerce",
                                        )
                                        .fillna(0)
                                        .to_numpy(dtype=int)
                                        != 0
                                        for partner in partners
                                    ]
                                ),
                                axis=1,
                            )
                            if partners
                            else np.zeros(len(df), dtype=bool)
                        ),
                        1,
                        0,
                    )
                ).all()
            ),
            "best_partner_present_when_flagged": bool(
                df.loc[any_flag == 1, "xasset_smt_best_partner"].notna().all()
                if (any_flag == 1).any()
                else True
            ),
        },
        "research_summary": summarize_smt_research(smt_research),
        "correlation_audit_summary": (
            summarize_cross_asset_correlation_audit(correlation_audit)
            if correlation_audit is not None
            else None
        ),
        "audit_classification": {
            "annotation_safe": bool((any_flag >= 0).all()),
            "detect_safe": bool((any_flag >= 0).all()),
            "confirm_safe": True,
            "active_safe": True,
            "model_safe": bool(
                best_score.dropna().between(0.0, 1.0).all()
                if best_score.notna().any()
                else True
            ),
            "research_only_model_safe": False,
        },
    }
    return summary


def smt_event_windows(
    df: pd.DataFrame,
    *,
    side: str = "all",
    bars_before: int = 8,
    bars_after: int = 8,
    limit: int = 5,
) -> list[pd.DataFrame]:
    if side not in {"all", "bull", "bear"}:
        raise ValueError("side must be one of {'all', 'bull', 'bear'}")

    partners = _partner_tokens(df)
    if side == "bull":
        positions = np.flatnonzero(
            np.any(
                np.column_stack(
                    [
                        pd.to_numeric(
                            df.get(f"xasset_smt_{partner}_bull_flag", 0),
                            errors="coerce",
                        )
                        .fillna(0)
                        .to_numpy(dtype=int)
                        == 1
                        for partner in partners
                    ]
                ),
                axis=1,
            )
            if partners
            else np.zeros(len(df), dtype=bool)
        )[:limit]
    elif side == "bear":
        positions = np.flatnonzero(
            np.any(
                np.column_stack(
                    [
                        pd.to_numeric(
                            df.get(f"xasset_smt_{partner}_bear_flag", 0),
                            errors="coerce",
                        )
                        .fillna(0)
                        .to_numpy(dtype=int)
                        == 1
                        for partner in partners
                    ]
                ),
                axis=1,
            )
            if partners
            else np.zeros(len(df), dtype=bool)
        )[:limit]
    else:
        positions = np.flatnonzero(
            pd.to_numeric(df["xasset_smt_any_flag"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=int)
            == 1
        )[:limit]

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "xasset_smt_any_flag",
        "xasset_smt_best_partner",
        "xasset_smt_best_score",
    ]
    for partner in partners:
        columns.extend(
            [
                f"xasset_smt_{partner}_bull_flag",
                f"xasset_smt_{partner}_bear_flag",
                f"xasset_smt_{partner}_dir",
                f"xasset_smt_{partner}_score",
                f"xasset_smt_{partner}_expected_relation",
            ]
        )
    columns.extend(sorted(column for column in df.columns if _CORR_RE.match(column)))
    columns.extend(sorted(column for column in df.columns if _LAG_RE.match(column)))
    columns = [column for column in columns if column in df.columns]

    windows: list[pd.DataFrame] = []
    for pos in positions:
        lo = max(0, pos - bars_before)
        hi = min(len(df), pos + bars_after + 1)
        win = df.iloc[lo:hi][columns].copy()
        win["event_row"] = 0
        win.iloc[pos - lo, win.columns.get_loc("event_row")] = 1
        windows.append(win)
    return windows


def _selected_corr_columns(df: pd.DataFrame, max_columns: int = 4) -> list[str]:
    windows_by_partner = _corr_windows_by_partner(df)
    selected: list[str] = []
    for partner in sorted(windows_by_partner):
        window = windows_by_partner[partner][0]
        column = f"xasset_corr_{partner}_w{window}"
        if column in df.columns:
            selected.append(column)
    return selected[:max_columns]


def plot_smt_validation(
    df: pd.DataFrame,
    outpath: str | Path,
    *,
    title: str = "SMT Validation",
) -> Path:
    _validate_required_columns(df)
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    partners = _partner_tokens(out)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
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

    for partner in partners:
        color = PARTNER_COLORS.get(partner, "#334155")
        bull_col = f"xasset_smt_{partner}_bull_flag"
        bear_col = f"xasset_smt_{partner}_bear_flag"
        score_col = f"xasset_smt_{partner}_score"
        if bull_col in out.columns:
            bull = out[pd.to_numeric(out[bull_col], errors="coerce").fillna(0) == 1]
            if not bull.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bull["timestamp"],
                        y=bull["low"],
                        mode="markers",
                        name=f"SMT Bull {partner}",
                        marker=dict(
                            symbol="triangle-up",
                            size=11,
                            color=color,
                            line=dict(width=1, color="#111827"),
                        ),
                        customdata=(
                            bull[[score_col]].to_numpy()
                            if score_col in bull.columns
                            else None
                        ),
                        hovertemplate=(
                            "<b>SMT Bull</b><br>"
                            f"Partner={partner}<br>"
                            "Time=%{x}<br>"
                            "Price=%{y:.4f}<br>"
                            "Score=%{customdata[0]:.3f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
        if bear_col in out.columns:
            bear = out[pd.to_numeric(out[bear_col], errors="coerce").fillna(0) == 1]
            if not bear.empty:
                fig.add_trace(
                    go.Scatter(
                        x=bear["timestamp"],
                        y=bear["high"],
                        mode="markers",
                        name=f"SMT Bear {partner}",
                        marker=dict(
                            symbol="triangle-down",
                            size=11,
                            color=color,
                            line=dict(width=1, color="#111827"),
                        ),
                        customdata=(
                            bear[[score_col]].to_numpy()
                            if score_col in bear.columns
                            else None
                        ),
                        hovertemplate=(
                            "<b>SMT Bear</b><br>"
                            f"Partner={partner}<br>"
                            "Time=%{x}<br>"
                            "Price=%{y:.4f}<br>"
                            "Score=%{customdata[0]:.3f}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

    fig.add_trace(
        go.Scatter(
            x=out["timestamp"],
            y=pd.to_numeric(out["xasset_smt_best_score"], errors="coerce"),
            mode="lines",
            name="Best SMT Score",
            line=dict(color="#0f172a", width=2),
        ),
        row=2,
        col=1,
    )
    selected_corr_columns = _selected_corr_columns(out)
    for column in selected_corr_columns:
        partner = column.split("_")[2]
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=pd.to_numeric(out[column], errors="coerce"),
                mode="lines",
                name=column,
                line=dict(
                    color=PARTNER_COLORS.get(partner, "#64748b"),
                    width=1,
                    dash="dot",
                ),
                opacity=0.75,
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Score / Corr", row=2, col=1, range=[-1.05, 1.05])
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=950,
        legend=dict(orientation="h"),
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def validate_smt(
    plot_df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    market_context: pd.DataFrame | None = None,
    research_table: pd.DataFrame | None = None,
    outpath: str | Path,
    title: str = "SMT Validation",
    n_windows: int = 5,
) -> dict[str, object]:
    summary_source = full_df if full_df is not None else plot_df
    summary = summarize_smt(
        summary_source,
        market_context=market_context,
        research_table=research_table,
    )
    bull_windows = smt_event_windows(summary_source, side="bull", limit=n_windows)
    bear_windows = smt_event_windows(summary_source, side="bear", limit=n_windows)
    html_path = plot_smt_validation(plot_df, outpath=outpath, title=title)
    return {
        "summary": summary,
        "bull_windows": bull_windows,
        "bear_windows": bear_windows,
        "html_path": html_path,
    }
