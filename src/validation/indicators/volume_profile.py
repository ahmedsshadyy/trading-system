from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators.foundation.volume_profile import (
    CANONICAL_VP_MODE,
    compute_volume_profile,
)
from src.validation.common.chart_core import create_candlestick_figure, save_figure_html

REQUIRED_VP_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vp_poc",
    "vp_vah",
    "vp_val",
    "vp_poc_distance_atr",
    "vp_value_width",
    "vp_value_width_atr",
    "vp_inside_value_area",
    "vp_above_vah",
    "vp_below_val",
    "vp_distance_to_vah_atr",
    "vp_distance_to_val_atr",
}


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _value_area_coverage(
    profile: np.ndarray, vah: float, val: float, bin_edges: np.ndarray
) -> float:
    if profile.size == 0 or bin_edges.size == 0:
        return np.nan
    included = 0.0
    total = float(profile.sum())
    if total <= 0:
        return np.nan
    for idx in range(len(profile)):
        left = float(bin_edges[idx])
        right = float(bin_edges[idx + 1])
        if left >= val and right <= vah + 1e-12:
            included += float(profile[idx])
    return included / total


def _full_vp_window_audit(
    df: pd.DataFrame,
    *,
    lookback: int,
    n_bins: int,
) -> dict[str, object]:
    valid_positions = np.flatnonzero(
        pd.to_numeric(df["vp_poc"], errors="coerce").notna().to_numpy()
    )
    all_rows_ok = {
        "current_bar_exclusion_ok": True,
        "allocated_volume_ok": True,
        "value_area_coverage_ok": True,
    }
    first_failures = {
        "current_bar_exclusion_row": None,
        "allocated_volume_row": None,
        "value_area_coverage_row": None,
    }
    degenerate = {
        "zero_volume_windows": 0,
        "flat_price_windows": 0,
        "single_bin_windows": 0,
        "nan_contaminated_windows": 0,
    }

    audited_rows = 0
    for idx in valid_positions:
        if idx < lookback:
            continue
        audited_rows += 1
        history = df.iloc[idx - lookback : idx]
        high = pd.to_numeric(history["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(history["low"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(history["close"], errors="coerce").to_numpy(dtype=float)
        volume = pd.to_numeric(history["volume"], errors="coerce").to_numpy(dtype=float)

        if not np.isfinite(np.r_[high, low, close, volume]).all():
            degenerate["nan_contaminated_windows"] += 1
        if np.isclose(np.nansum(volume), 0.0):
            degenerate["zero_volume_windows"] += 1
        if np.isclose(np.nanmax(high), np.nanmin(low)):
            degenerate["flat_price_windows"] += 1

        vp = compute_volume_profile(history, lookback=lookback, n_bins=n_bins)
        profile = vp["profile"]
        bin_edges = vp["bin_edges"]
        if np.count_nonzero(profile > 0) <= 1:
            degenerate["single_bin_windows"] += 1

        row = df.iloc[idx]
        exclusion_ok = (
            bool(np.isclose(float(row["vp_poc"]), vp["poc"], equal_nan=True))
            and bool(np.isclose(float(row["vp_vah"]), vp["vah"], equal_nan=True))
            and bool(np.isclose(float(row["vp_val"]), vp["val"], equal_nan=True))
        )
        if not exclusion_ok:
            all_rows_ok["current_bar_exclusion_ok"] = False
            if first_failures["current_bar_exclusion_row"] is None:
                first_failures["current_bar_exclusion_row"] = int(idx)

        allocated_ok = bool(
            np.isclose(
                float(np.nansum(profile)), float(np.nansum(volume)), equal_nan=True
            )
        )
        if not allocated_ok:
            all_rows_ok["allocated_volume_ok"] = False
            if first_failures["allocated_volume_row"] is None:
                first_failures["allocated_volume_row"] = int(idx)

        coverage = _value_area_coverage(profile, vp["vah"], vp["val"], bin_edges)
        coverage_ok = bool(np.isnan(coverage) or coverage >= 0.70 - 1e-12)
        if not coverage_ok:
            all_rows_ok["value_area_coverage_ok"] = False
            if first_failures["value_area_coverage_row"] is None:
                first_failures["value_area_coverage_row"] = int(idx)

    return {
        "audited_row_count": int(audited_rows),
        "all_rows_ok": all_rows_ok,
        "first_failures": first_failures,
        "degenerate_window_counts": degenerate,
    }


def _approximation_audit(
    exact_df: pd.DataFrame,
    stepped_df: pd.DataFrame | None,
) -> dict[str, object]:
    audit = {
        "mode": CANONICAL_VP_MODE,
        "canonical_mode": CANONICAL_VP_MODE,
        "stepped_mode_role": "approximation_only",
        "stepped_mode_parity_equivalent": False,
        "stepped_mode_supported": True,
        "comparison_performed": stepped_df is not None,
    }
    if stepped_df is None:
        return audit

    stats: dict[str, dict[str, float | int | None]] = {}
    for col in ("vp_poc", "vp_vah", "vp_val"):
        exact = pd.to_numeric(exact_df[col], errors="coerce")
        stepped = pd.to_numeric(stepped_df[col], errors="coerce")
        valid = exact.notna() & stepped.notna()
        if not bool(valid.any()):
            stats[col] = {"valid_rows": 0, "mean_abs_diff": None, "max_abs_diff": None}
            continue
        diffs = (exact.loc[valid] - stepped.loc[valid]).abs()
        stats[col] = {
            "valid_rows": int(valid.sum()),
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff": float(diffs.max()),
        }
    audit["exact_vs_stepped"] = stats
    return audit


def _distribution_summary(df: pd.DataFrame) -> dict[str, object]:
    valid = pd.to_numeric(df["vp_poc"], errors="coerce").notna()
    valid_df = df.loc[valid]
    if valid_df.empty:
        return {
            "value_area_percentages": {
                "inside": None,
                "above_vah": None,
                "below_val": None,
            },
            "summary_stats": {},
        }
    return {
        "value_area_percentages": {
            "inside": float(
                pd.to_numeric(valid_df["vp_inside_value_area"], errors="coerce").mean()
                * 100.0
            ),
            "above_vah": float(
                pd.to_numeric(valid_df["vp_above_vah"], errors="coerce").mean() * 100.0
            ),
            "below_val": float(
                pd.to_numeric(valid_df["vp_below_val"], errors="coerce").mean() * 100.0
            ),
        },
        "summary_stats": {
            "vp_value_width_atr": _continuous_stats(valid_df["vp_value_width_atr"]),
            "vp_poc_distance_atr": _continuous_stats(valid_df["vp_poc_distance_atr"]),
            "vp_distance_to_vah_atr": _continuous_stats(
                valid_df["vp_distance_to_vah_atr"]
            ),
            "vp_distance_to_val_atr": _continuous_stats(
                valid_df["vp_distance_to_val_atr"]
            ),
        },
    }


def _build_vp_figure(df: pd.DataFrame, *, title: str) -> go.Figure:
    fig = create_candlestick_figure(df, title=title)
    for col, dash in (("vp_poc", "solid"), ("vp_vah", "dash"), ("vp_val", "dash")):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(dash=dash),
            )
        )
    inside = df.loc[
        pd.to_numeric(df["vp_inside_value_area"], errors="coerce").fillna(0).eq(1)
    ]
    if not inside.empty:
        fig.add_trace(
            go.Scatter(
                x=inside["timestamp"],
                y=inside["close"],
                mode="markers",
                name="inside_value_area",
                marker=dict(symbol="circle", size=7),
            )
        )
    return fig


def validate_volume_profile(
    df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame | None = None,
    stepped_df: pd.DataFrame | None = None,
    lookback: int = 80,
    n_bins: int = 50,
    outpath: str | Path | None = None,
    title: str = "Volume Profile Validation",
    source_parity_ok: bool | None = None,
) -> dict[str, object]:
    audit_df = summary_df if summary_df is not None else df

    missing = sorted(REQUIRED_VP_COLUMNS - set(audit_df.columns))
    if missing:
        raise ValueError(
            f"validate_volume_profile: missing required columns: {missing}"
        )

    valid_rows = pd.to_numeric(audit_df["vp_poc"], errors="coerce").notna().to_numpy()
    valid_positions = np.flatnonzero(valid_rows)
    expected_first_valid_row = lookback if len(audit_df) > lookback else None
    observed_first_valid_row = int(valid_positions[0]) if len(valid_positions) else None
    no_premature_values = bool(
        pd.to_numeric(audit_df.iloc[:lookback]["vp_poc"], errors="coerce").isna().all()
    )
    expected_valid_count = max(len(audit_df) - lookback, 0)

    poc = pd.to_numeric(audit_df["vp_poc"], errors="coerce")
    vah = pd.to_numeric(audit_df["vp_vah"], errors="coerce")
    val = pd.to_numeric(audit_df["vp_val"], errors="coerce")
    poc_order_ok = bool(((val <= poc) & (poc <= vah) | poc.isna()).all())
    width_ok = bool(
        (
            (pd.to_numeric(audit_df["vp_value_width"], errors="coerce") >= 0)
            | poc.isna()
        ).all()
    )
    vp_columns = [
        "vp_poc",
        "vp_vah",
        "vp_val",
        "vp_poc_distance_atr",
        "vp_value_width",
        "vp_value_width_atr",
        "vp_distance_to_vah_atr",
        "vp_distance_to_val_atr",
    ]
    vp_values = (
        audit_df[vp_columns].select_dtypes(include=[np.number]).to_numpy(dtype=float)
    )
    no_inf_in_vp_columns = bool(np.isfinite(vp_values[~np.isnan(vp_values)]).all())
    full_window_audit = _full_vp_window_audit(
        audit_df, lookback=lookback, n_bins=n_bins
    )

    checks = {
        "required_columns_present": True,
        "poc_vah_val_order_ok": poc_order_ok,
        "value_width_non_negative": width_ok,
        "source_parity_ok": source_parity_ok,
        "warmup_count_ok": int(valid_rows.sum()) == expected_valid_count,
        "first_valid_row_matches_expected": observed_first_valid_row
        == expected_first_valid_row,
        "no_premature_vp_values_before_lookback": no_premature_values,
        "full_current_bar_exclusion_ok": bool(
            full_window_audit["all_rows_ok"]["current_bar_exclusion_ok"]
        ),
        "full_allocated_volume_ok": bool(
            full_window_audit["all_rows_ok"]["allocated_volume_ok"]
        ),
        "full_value_area_coverage_ok": bool(
            full_window_audit["all_rows_ok"]["value_area_coverage_ok"]
        ),
        "no_inf_in_vp_columns": no_inf_in_vp_columns,
    }

    summary = {
        "mode": CANONICAL_VP_MODE,
        "lookback": int(lookback),
        "n_bins": int(n_bins),
        "row_count": int(len(audit_df)),
        "valid_vp_row_count": int(valid_rows.sum()),
        "warmup": {
            "expected_first_valid_row": expected_first_valid_row,
            "observed_first_valid_row": observed_first_valid_row,
            "expected_valid_row_count": int(expected_valid_count),
            "warmup_row_count": int(min(len(audit_df), lookback)),
        },
        "checks": checks,
        "full_window_audit": full_window_audit,
        "approximation_audit": _approximation_audit(audit_df, stepped_df),
        "distribution_summary": _distribution_summary(audit_df),
    }

    html_path = None
    if outpath is not None:
        fig = _build_vp_figure(df, title=title)
        html_path = save_figure_html(fig, outpath)

    return {"summary": summary, "html_path": html_path}
