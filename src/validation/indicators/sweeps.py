from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

AGE_BUCKET_BINS = [-np.inf, 5, 10, 20, 40, np.inf]
AGE_BUCKET_LABELS = ["0_5", "6_10", "11_20", "21_40", "41_plus"]

CORE_SWEEP_COLUMNS = [
    "sweep_event_id",
    "sweep_direction",
    "sweep_detect_flag",
    "sweep_confirm_flag",
    "sweep_detect_idx",
    "sweep_confirm_idx",
    "sweep_source_idx",
    "sweep_level",
    "sweep_level_type",
    "sweep_level_age",
    "sweep_extension_atr",
    "sweep_reclaim_atr",
    "sweep_wick_frac",
    "sweep_body_frac",
    "sweep_score",
    "sweep_high_detect_flag",
    "sweep_low_detect_flag",
    "sweep_high_confirm_flag",
    "sweep_low_confirm_flag",
]

PASS_COLUMNS = [
    "sweep_high_pass_valid_inputs",
    "sweep_high_pass_source_exists",
    "sweep_high_pass_source_age",
    "sweep_high_pass_wick_through",
    "sweep_high_pass_close_back_inside",
    "sweep_high_pass_extension",
    "sweep_high_pass_reclaim",
    "sweep_high_pass_wick_frac",
    "sweep_high_pass_body_frac",
    "sweep_low_pass_valid_inputs",
    "sweep_low_pass_source_exists",
    "sweep_low_pass_source_age",
    "sweep_low_pass_wick_through",
    "sweep_low_pass_close_back_inside",
    "sweep_low_pass_extension",
    "sweep_low_pass_reclaim",
    "sweep_low_pass_wick_frac",
    "sweep_low_pass_body_frac",
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


def _age_bucket_counts(series: pd.Series) -> dict[str, int]:
    clean = pd.to_numeric(series, errors="coerce")
    bucketed = pd.cut(
        clean,
        bins=AGE_BUCKET_BINS,
        labels=AGE_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    counts = bucketed.value_counts(dropna=False).sort_index()
    out = {str(label): int(counts.get(label, 0)) for label in AGE_BUCKET_LABELS}
    if counts.isna().any():
        out["NaN"] = int(counts[counts.index.isna()].sum())
    return out


def _confirmation_lineage_stats(df: pd.DataFrame) -> dict[str, int | bool]:
    high_confirm_mask = df["sweep_high_confirm_flag"] == 1
    low_confirm_mask = df["sweep_low_confirm_flag"] == 1

    prev_high_detect = (
        pd.Series(df["sweep_high_detect_flag"], copy=False).shift(1, fill_value=0) == 1
    )
    prev_low_detect = (
        pd.Series(df["sweep_low_detect_flag"], copy=False).shift(1, fill_value=0) == 1
    )

    high_confirm_has_ancestor = (~high_confirm_mask) | prev_high_detect
    low_confirm_has_ancestor = (~low_confirm_mask) | prev_low_detect

    collision_mask = (df["sweep_detect_flag"] == 1) & (
        high_confirm_mask | low_confirm_mask
    )

    return {
        "high_confirm_rows_have_prev_detect": bool(high_confirm_has_ancestor.all()),
        "low_confirm_rows_have_prev_detect": bool(low_confirm_has_ancestor.all()),
        "all_confirm_rows_have_prev_detect": bool(
            high_confirm_has_ancestor.all() and low_confirm_has_ancestor.all()
        ),
        "confirm_delay_is_one_bar": bool(
            high_confirm_has_ancestor[high_confirm_mask].all()
            and low_confirm_has_ancestor[low_confirm_mask].all()
        ),
        "shared_detect_confirm_collision_count": int(collision_mask.sum()),
    }


def summarize_sweeps(df: pd.DataFrame) -> dict[str, object]:
    required = {"timestamp", "open", "high", "low", "close", *CORE_SWEEP_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing sweep columns: {sorted(missing)}")

    detect_mask = df["sweep_detect_flag"] == 1
    confirm_mask = df["sweep_confirm_flag"] == 1
    detect_df = df.loc[detect_mask].copy()
    confirm_df = df.loc[confirm_mask].copy()
    confirm_lineage = _confirmation_lineage_stats(df)

    source_before_detect = (
        detect_df["sweep_source_idx"] <= detect_df["sweep_detect_idx"]
        if not detect_df.empty
        else pd.Series(dtype=bool)
    )

    return {
        "window": {
            "start": str(pd.to_datetime(df["timestamp"], utc=True).min()),
            "end": str(pd.to_datetime(df["timestamp"], utc=True).max()),
            "rows": int(len(df)),
        },
        "event_counts": {
            "detect_count": int(detect_mask.sum()),
            "confirm_count": int(confirm_mask.sum()),
            "high_detect_count": int((df["sweep_high_detect_flag"] == 1).sum()),
            "low_detect_count": int((df["sweep_low_detect_flag"] == 1).sum()),
            "high_confirm_count": int((df["sweep_high_confirm_flag"] == 1).sum()),
            "low_confirm_count": int((df["sweep_low_confirm_flag"] == 1).sum()),
            "confirm_rate_of_detects": (
                float(
                    confirm_df["sweep_event_id"].nunique()
                    / detect_df["sweep_event_id"].nunique()
                )
                if not detect_df.empty
                else np.nan
            ),
        },
        "direction_distribution": _value_counts(detect_df["sweep_direction"]),
        "level_type_distribution": _value_counts(detect_df["sweep_level_type"]),
        "age_bucket_distribution": _age_bucket_counts(detect_df["sweep_level_age"]),
        "continuous_event_stats": {
            "extension_atr": _continuous_stats(detect_df["sweep_extension_atr"]),
            "reclaim_atr": _continuous_stats(detect_df["sweep_reclaim_atr"]),
            "wick_frac": _continuous_stats(detect_df["sweep_wick_frac"]),
            "body_frac": _continuous_stats(detect_df["sweep_body_frac"]),
            "score": _continuous_stats(detect_df["sweep_score"]),
        },
        "pass_flag_counts": {
            col: int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            for col in PASS_COLUMNS
            if col in df.columns
        },
        "timing_invariants": {
            "source_idx_le_detect_idx": (
                bool(source_before_detect.all()) if not detect_df.empty else True
            ),
            "detect_idx_lt_confirm_idx": bool(
                confirm_lineage["confirm_delay_is_one_bar"]
            ),
            "confirm_rows_have_detect_ancestor": bool(
                confirm_lineage["all_confirm_rows_have_prev_detect"]
            ),
            "unique_detect_event_ids": bool(
                detect_df["sweep_event_id"].dropna().is_unique
            ),
            "high_confirm_rows_have_prev_detect": bool(
                confirm_lineage["high_confirm_rows_have_prev_detect"]
            ),
            "low_confirm_rows_have_prev_detect": bool(
                confirm_lineage["low_confirm_rows_have_prev_detect"]
            ),
            "shared_detect_confirm_collision_count": int(
                confirm_lineage["shared_detect_confirm_collision_count"]
            ),
        },
    }


def extract_sweep_audit_tables(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    detect_mask = df["sweep_detect_flag"] == 1
    confirm_mask = df["sweep_confirm_flag"] == 1

    preferred = [
        "timestamp",
        "sweep_event_id",
        "sweep_direction",
        "sweep_detect_idx",
        "sweep_confirm_idx",
        "sweep_source_idx",
        "sweep_level",
        "sweep_level_type",
        "sweep_level_age",
        "sweep_extension_atr",
        "sweep_reclaim_atr",
        "sweep_wick_frac",
        "sweep_body_frac",
        "sweep_score",
        "sweep_high_detect_flag",
        "sweep_low_detect_flag",
        "sweep_high_confirm_flag",
        "sweep_low_confirm_flag",
    ]
    cols = [col for col in preferred if col in df.columns]

    strongest = (
        df.loc[detect_mask, cols]
        .sort_values(["sweep_score", "sweep_extension_atr"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    confirmed = (
        df.loc[confirm_mask, cols]
        .sort_values(["timestamp", "sweep_event_id"])
        .head(top_n)
        .reset_index(drop=True)
    )

    high_miss = (
        np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_high_pass_extension", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_high_pass_reclaim", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_high_pass_wick_frac", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_high_pass_body_frac", 0), errors="coerce"
            ).fillna(0),
        )
    )
    low_miss = (
        np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_low_pass_extension", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_low_pass_reclaim", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_low_pass_wick_frac", 0), errors="coerce"
            ).fillna(0),
        )
        + np.maximum(
            0.0,
            1.0
            - pd.to_numeric(
                df.get("sweep_low_pass_body_frac", 0), errors="coerce"
            ).fillna(0),
        )
    )
    near_miss_distance = np.minimum(high_miss, low_miss)
    near_miss_mask = (df["sweep_detect_flag"] == 0) & (
        (
            pd.to_numeric(
                df.get("sweep_high_pass_source_exists", 0), errors="coerce"
            ).fillna(0)
            == 1
        )
        | (
            pd.to_numeric(
                df.get("sweep_low_pass_source_exists", 0), errors="coerce"
            ).fillna(0)
            == 1
        )
    )
    near_miss = df.loc[near_miss_mask, cols].copy()
    near_miss["near_miss_distance"] = near_miss_distance[near_miss_mask]
    near_miss = (
        near_miss.sort_values(
            ["near_miss_distance", "timestamp"], ascending=[True, True]
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "strongest": strongest,
        "confirmed": confirmed,
        "near_miss": near_miss,
    }


def validate_sweeps(
    df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Liquidity Sweeps Validation",
    top_n: int = 20,
) -> dict[str, object]:
    analysis_df = full_df if full_df is not None else df
    summary = summarize_sweeps(analysis_df)
    tables = extract_sweep_audit_tables(analysis_df, top_n=top_n)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=ts,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#0f9d58",
            decreasing_line_color="#c5221f",
        )
    )

    high_detect = df["sweep_high_detect_flag"] == 1
    low_detect = df["sweep_low_detect_flag"] == 1
    high_confirm = df["sweep_high_confirm_flag"] == 1
    low_confirm = df["sweep_low_confirm_flag"] == 1

    if high_detect.any():
        fig.add_trace(
            go.Scatter(
                x=ts[high_detect],
                y=df.loc[high_detect, "high"],
                mode="markers",
                name="High Sweep Detect",
                marker=dict(symbol="triangle-down", color="#b91c1c", size=10),
            )
        )
    if low_detect.any():
        fig.add_trace(
            go.Scatter(
                x=ts[low_detect],
                y=df.loc[low_detect, "low"],
                mode="markers",
                name="Low Sweep Detect",
                marker=dict(symbol="triangle-up", color="#15803d", size=10),
            )
        )
    if high_confirm.any():
        fig.add_trace(
            go.Scatter(
                x=ts[high_confirm],
                y=df.loc[high_confirm, "high"],
                mode="markers",
                name="High Sweep Confirm",
                marker=dict(symbol="x", color="#7f1d1d", size=9),
            )
        )
    if low_confirm.any():
        fig.add_trace(
            go.Scatter(
                x=ts[low_confirm],
                y=df.loc[low_confirm, "low"],
                mode="markers",
                name="Low Sweep Confirm",
                marker=dict(symbol="x", color="#14532d", size=9),
            )
        )

    fig.update_layout(
        title=title,
        height=760,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )

    html_path: str | None = None
    if outpath is not None:
        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_file)
        html_path = str(out_file)

    return {
        "summary": summary,
        "tables": tables,
        "figure": fig,
        "html_path": html_path,
    }
