from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.validation.common.chart_core import create_candlestick_figure, save_figure_html

REQUIRED_AVWAP_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "avwap",
    "avwap_std",
    "avwap_upper_1",
    "avwap_lower_1",
    "avwap_upper_2",
    "avwap_lower_2",
    "avwap_dev_sigma",
    "avwap_distance",
    "avwap_distance_atr",
    "avwap_distance_pct",
    "avwap_above",
    "avwap_below",
    "avwap_inside_1sigma",
    "avwap_outside_2sigma",
    "avwap_cross_up",
    "avwap_cross_down",
    "avwap_slope_5",
    "avwap_slope_20",
    "avwap_trend_state",
    "bars_since_anchor",
    "avwap_anchor_class",
    "avwap_anchor_label",
    "avwap_anchor_idx",
    "avwap_anchor_origin_idx",
    "avwap_anchor_confirm_idx",
    "avwap_anchor_live_from_idx",
}

SUMMARY_COLUMNS = [
    "avwap_dev_sigma",
    "avwap_distance_atr",
    "avwap_distance_pct",
    "avwap_slope_5",
    "avwap_slope_20",
]

AVWAP_FAMILY_SAMPLE_LIMIT = 5


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


def _no_inf_ok(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    values = numeric.to_numpy(dtype=float)
    return bool(np.isfinite(values[~np.isnan(values)]).all())


def _active_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["avwap"], errors="coerce").notna()


def _live_activation_checks(df: pd.DataFrame) -> dict[str, object]:
    active = _active_mask(df).to_numpy()
    active_positions = np.flatnonzero(active)
    live_from_series = pd.to_numeric(df["avwap_anchor_live_from_idx"], errors="coerce")
    live_from_values = live_from_series.dropna().unique()
    live_from_idx = int(live_from_values[0]) if len(live_from_values) else None
    first_active_row = int(active_positions[0]) if len(active_positions) else None

    no_values_before_live = True
    if live_from_idx is not None and live_from_idx > 0:
        no_values_before_live = bool(
            pd.to_numeric(df.iloc[:live_from_idx]["avwap"], errors="coerce")
            .isna()
            .all()
        )

    first_active_matches = (
        live_from_idx is None and first_active_row is None
    ) or first_active_row == live_from_idx

    return {
        "live_from_idx": live_from_idx,
        "first_active_row": first_active_row,
        "no_values_before_live_activation": bool(no_values_before_live),
        "first_active_row_matches_live_from": bool(first_active_matches),
    }


def _band_order_ok(df: pd.DataFrame) -> bool:
    active = _active_mask(df)
    if not bool(active.any()):
        return True
    scoped = df.loc[active]
    lower2 = pd.to_numeric(scoped["avwap_lower_2"], errors="coerce")
    lower1 = pd.to_numeric(scoped["avwap_lower_1"], errors="coerce")
    avwap = pd.to_numeric(scoped["avwap"], errors="coerce")
    upper1 = pd.to_numeric(scoped["avwap_upper_1"], errors="coerce")
    upper2 = pd.to_numeric(scoped["avwap_upper_2"], errors="coerce")
    return bool(
        (
            (lower2 <= lower1)
            & (lower1 <= avwap)
            & (avwap <= upper1)
            & (upper1 <= upper2)
        ).all()
    )


def _std_contract_ok(df: pd.DataFrame) -> bool:
    active = _active_mask(df)
    if not bool(active.any()):
        return True
    std = pd.to_numeric(df.loc[active, "avwap_std"], errors="coerce")
    dev = pd.to_numeric(df.loc[active, "avwap_dev_sigma"], errors="coerce")
    std_non_negative = bool((std >= 0).all())
    zero_std_dev_zero = bool(dev.loc[std.eq(0)].fillna(0).eq(0).all())
    return bool(std_non_negative and zero_std_dev_zero)


def _bars_since_anchor_ok(df: pd.DataFrame) -> bool:
    active = _active_mask(df).to_numpy()
    active_positions = np.flatnonzero(active)
    if len(active_positions) == 0:
        return True
    anchor_idx_series = pd.to_numeric(df["avwap_anchor_idx"], errors="coerce")
    anchor_values = anchor_idx_series.dropna().unique()
    if len(anchor_values) != 1:
        return False
    anchor_idx = int(anchor_values[0])
    observed = pd.to_numeric(
        df.iloc[active_positions]["bars_since_anchor"], errors="coerce"
    ).to_numpy(dtype=float)
    expected = active_positions.astype(float) - float(anchor_idx)
    return bool(np.allclose(observed, expected, atol=1e-12, equal_nan=False))


def _trend_state_values_ok(df: pd.DataFrame) -> bool:
    values = (
        pd.to_numeric(df["avwap_trend_state"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    return bool(set(values).issubset({-1, 0, 1}))


def _cross_events_ok(df: pd.DataFrame) -> bool:
    cross_up = pd.to_numeric(df["avwap_cross_up"], errors="coerce").fillna(0)
    cross_down = pd.to_numeric(df["avwap_cross_down"], errors="coerce").fillna(0)
    return bool(((cross_up + cross_down) <= 1).all())


def _build_avwap_figure(df: pd.DataFrame, *, title: str) -> go.Figure:
    fig = create_candlestick_figure(df, title=title)
    for col, dash in (
        ("avwap", "solid"),
        ("avwap_upper_1", "dot"),
        ("avwap_lower_1", "dot"),
        ("avwap_upper_2", "dash"),
        ("avwap_lower_2", "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(dash=dash),
            )
        )

    cross_up = df.loc[
        pd.to_numeric(df["avwap_cross_up"], errors="coerce").fillna(0).eq(1)
    ]
    if not cross_up.empty:
        fig.add_trace(
            go.Scatter(
                x=cross_up["timestamp"],
                y=cross_up["close"],
                mode="markers",
                name="avwap_cross_up",
                marker=dict(symbol="triangle-up", size=9),
            )
        )

    cross_down = df.loc[
        pd.to_numeric(df["avwap_cross_down"], errors="coerce").fillna(0).eq(1)
    ]
    if not cross_down.empty:
        fig.add_trace(
            go.Scatter(
                x=cross_down["timestamp"],
                y=cross_down["close"],
                mode="markers",
                name="avwap_cross_down",
                marker=dict(symbol="triangle-down", size=9),
            )
        )

    active_positions = np.flatnonzero(_active_mask(df).to_numpy())
    if len(active_positions):
        anchor_row = df.iloc[[active_positions[0]]]
        fig.add_trace(
            go.Scatter(
                x=anchor_row["timestamp"],
                y=anchor_row["close"],
                mode="markers",
                name="avwap_live_from",
                marker=dict(symbol="x", size=11),
            )
        )

    return fig


def _single_family_audit(df: pd.DataFrame) -> dict[str, object]:
    active = _active_mask(df)
    activation = _live_activation_checks(df)
    anchor_labels = pd.Series(df["avwap_anchor_label"]).dropna().unique().tolist()
    anchor_classes = pd.Series(df["avwap_anchor_class"]).dropna().unique().tolist()
    return {
        "row_count": int(len(df)),
        "active_row_count": int(active.sum()),
        "anchor_labels": anchor_labels,
        "anchor_classes": anchor_classes,
        "activation": activation,
        "checks": {
            "band_order_ok": _band_order_ok(df),
            "std_contract_ok": _std_contract_ok(df),
            "bars_since_anchor_ok": _bars_since_anchor_ok(df),
            "trend_state_values_ok": _trend_state_values_ok(df),
            "cross_events_non_overlapping": _cross_events_ok(df),
            "no_values_before_live_activation": activation[
                "no_values_before_live_activation"
            ],
            "first_active_row_matches_live_from": activation[
                "first_active_row_matches_live_from"
            ],
        },
    }


def _family_audit(
    frames: pd.DataFrame | list[pd.DataFrame] | None,
) -> dict[str, object]:
    if frames is None:
        return {"available": False}
    family_frames = (
        [frames]
        if isinstance(frames, pd.DataFrame)
        else [frame for frame in frames if frame is not None]
    )
    if not family_frames:
        return {"available": False}
    samples = [_single_family_audit(frame) for frame in family_frames]
    all_checks = sorted(samples[0]["checks"].keys())
    active_counts = np.array(
        [int(sample["active_row_count"]) for sample in samples], dtype=float
    )
    row_counts = np.array([int(sample["row_count"]) for sample in samples], dtype=float)
    anchor_labels = sorted(
        {label for sample in samples for label in sample["anchor_labels"]}
    )
    anchor_classes = sorted(
        {label for sample in samples for label in sample["anchor_classes"]}
    )
    return {
        "available": True,
        "sample_count": int(len(samples)),
        "row_count": int(row_counts.sum()),
        "active_row_count": int(active_counts.sum()),
        "active_row_count_stats": {
            "min": int(active_counts.min()),
            "mean": float(active_counts.mean()),
            "max": int(active_counts.max()),
        },
        "anchor_labels": anchor_labels,
        "anchor_classes": anchor_classes,
        "checks": {
            name: all(bool(sample["checks"][name]) for sample in samples)
            for name in all_checks
        },
        "sample_previews": samples[:AVWAP_FAMILY_SAMPLE_LIMIT],
    }


def validate_avwap(
    df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame | None = None,
    live_df: pd.DataFrame | None = None,
    family_frames: dict[str, pd.DataFrame | None] | None = None,
    outpath: str | Path | None = None,
    title: str = "AVWAP Validation",
    source_parity_ok: bool | None = None,
) -> dict[str, object]:
    audit_df = summary_df if summary_df is not None else df
    missing = sorted(REQUIRED_AVWAP_COLUMNS - set(audit_df.columns))
    if missing:
        raise ValueError(f"validate_avwap: missing required columns: {missing}")

    live_research_parity_ok = None
    no_research_in_live_ok = None
    if live_df is not None:
        no_research_in_live_ok = not any(
            col.startswith("r_") for col in live_df.columns
        )
    if live_df is not None and summary_df is not None:
        non_research_cols = [
            col for col in summary_df.columns if not col.startswith("r_")
        ]
        live_research_parity_ok = bool(
            live_df[non_research_cols].equals(summary_df[non_research_cols])
        )

    active = _active_mask(audit_df)
    anchor_labels = pd.Series(audit_df["avwap_anchor_label"]).dropna().unique().tolist()
    anchor_classes = (
        pd.Series(audit_df["avwap_anchor_class"]).dropna().unique().tolist()
    )
    activation = _live_activation_checks(audit_df)

    summary = {
        "row_count": int(len(audit_df)),
        "active_row_count": int(active.sum()),
        "anchor_metadata": {
            "anchor_labels": anchor_labels,
            "anchor_classes": anchor_classes,
            "anchor_idx_values": pd.to_numeric(
                audit_df["avwap_anchor_idx"], errors="coerce"
            )
            .dropna()
            .unique()
            .tolist(),
            "anchor_confirm_idx_values": pd.to_numeric(
                audit_df["avwap_anchor_confirm_idx"], errors="coerce"
            )
            .dropna()
            .unique()
            .tolist(),
            "anchor_live_from_idx_values": pd.to_numeric(
                audit_df["avwap_anchor_live_from_idx"], errors="coerce"
            )
            .dropna()
            .unique()
            .tolist(),
        },
        "activation": activation,
        "checks": {
            "required_columns_present": True,
            "no_inf_values": _no_inf_ok(audit_df),
            "band_order_ok": _band_order_ok(audit_df),
            "std_contract_ok": _std_contract_ok(audit_df),
            "bars_since_anchor_ok": _bars_since_anchor_ok(audit_df),
            "trend_state_values_ok": _trend_state_values_ok(audit_df),
            "cross_events_non_overlapping": _cross_events_ok(audit_df),
            "no_values_before_live_activation": activation[
                "no_values_before_live_activation"
            ],
            "first_active_row_matches_live_from": activation[
                "first_active_row_matches_live_from"
            ],
            "source_parity_ok": source_parity_ok,
            "live_research_parity_ok": live_research_parity_ok,
            "no_research_columns_in_live_ok": no_research_in_live_ok,
        },
        "summary_stats": {
            col: _continuous_stats(audit_df[col]) for col in SUMMARY_COLUMNS
        },
    }

    if family_frames:
        summary["anchor_family_audits"] = {
            name: _family_audit(frame) for name, frame in family_frames.items()
        }
        sweep_audit = summary["anchor_family_audits"].get(
            "sweep_detect_to_confirm_hybrid"
        )
        if isinstance(sweep_audit, dict) and sweep_audit.get("available") is False:
            summary["pending_items"] = {
                "sweep_detect_to_confirm_hybrid": (
                    "Unavailable in the current validation sample. "
                    "This should be revisited and validated with real sweep events "
                    "when the sweep family is finalized."
                )
            }

    html_path = None
    if outpath is not None:
        fig = _build_avwap_figure(df, title=title)
        html_path = save_figure_html(fig, outpath)

    return {"summary": summary, "html_path": html_path}
