from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.validation.common.chart_core import save_figure_html

REQUIRED_VOLUME_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vol_sma_20",
    "vol_med_20",
    "vol_std_20",
    "vol_zscore_20",
    "vol_ratio",
    "vol_ratio_med_20",
    "vol_pct_rank_100",
    "vol_above_avg",
    "vol_below_avg",
    "vol_above_1_5x",
    "vol_above_2_0x",
    "vol_below_0_8x",
    "vol_extreme_pct90",
    "vol_extreme_pct95",
    "vol_slope_5",
    "vol_slope_5_norm",
    "vol_ema_5",
    "vol_ema_20",
    "vol_ema_ratio_5_20",
    "bar_range",
    "bar_range_atr",
    "bar_body",
    "bar_body_frac",
    "close_pos_in_range",
    "body_direction",
    "true_spread_ratio",
    "effort_vs_result",
    "effort_vs_body",
    "result_vs_effort",
    "body_result_vs_effort",
    "close_strength",
    "body_strength",
    "wick_bias",
    "signed_tick_pressure_raw",
    "signed_tick_pressure_norm",
    "signed_tick_pressure_body",
    "signed_tick_pressure_wick",
    "signed_tick_pressure_blend",
    "signed_tick_pressure_z",
    "pressure_agrees_with_body",
    "pressure_agrees_with_close_location",
    "pressure_divergence_flag",
    "pressure_extreme_pos",
    "pressure_extreme_neg",
    "delta_proxy_raw",
    "delta_proxy_norm",
    "delta_proxy_body",
    "candle_delta_proxy",
    "vsa_absorption",
    "vsa_directional",
    "vsa_no_demand",
    "vsa_no_supply",
    "vsa_climactic_up",
    "vsa_climactic_down",
    "vsa_churn",
    "vsa_effort_failure",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "dominant_wick_ratio",
    "wick_imbalance",
    "upper_rejection_effort",
    "lower_rejection_effort",
    "wick_effort_imbalance",
}

FLAG_COLUMNS = [
    "vol_above_avg",
    "vol_below_avg",
    "vol_above_1_5x",
    "vol_above_2_0x",
    "vol_below_0_8x",
    "vol_extreme_pct90",
    "vol_extreme_pct95",
    "vsa_absorption",
    "vsa_directional",
    "vsa_no_demand",
    "vsa_no_supply",
    "vsa_climactic_up",
    "vsa_climactic_down",
    "vsa_churn",
    "vsa_effort_failure",
    "pressure_agrees_with_body",
    "pressure_agrees_with_close_location",
    "pressure_divergence_flag",
    "pressure_extreme_pos",
    "pressure_extreme_neg",
]

SUMMARY_COLUMNS = [
    "vol_ratio",
    "vol_pct_rank_100",
    "candle_delta_proxy",
    "signed_tick_pressure_blend",
    "signed_tick_pressure_z",
    "wick_bias",
    "effort_vs_result",
    "upper_rejection_effort",
    "lower_rejection_effort",
]

PRESSURE_RELATIONSHIP_DENOM_FLOOR = 0.10


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


def _zero_range_ok(df: pd.DataFrame) -> bool:
    zero_range = pd.to_numeric(df["bar_range"], errors="coerce").fillna(np.nan).eq(0)
    if not bool(zero_range.any()):
        return True
    scoped = df.loc[zero_range]
    close_pos_ok = scoped["close_pos_in_range"].isna().all()
    wick_ok = bool(
        (scoped["upper_wick_ratio"] == 0).all()
        and (scoped["lower_wick_ratio"] == 0).all()
    )
    strength_ok = bool((scoped["close_strength"] == 0).all())
    delta_ok = bool(
        scoped["delta_proxy_raw"].fillna(0).eq(0).all()
        and scoped["delta_proxy_norm"].fillna(0).eq(0).all()
        and scoped["delta_proxy_body"].fillna(0).eq(0).all()
        and scoped["signed_tick_pressure_wick"].fillna(0).eq(0).all()
        and scoped["signed_tick_pressure_blend"].fillna(0).eq(0).all()
    )
    wick_bias_ok = bool(scoped["wick_bias"].fillna(0).eq(0).all())
    return bool(close_pos_ok and wick_ok and strength_ok and delta_ok and wick_bias_ok)


def _zero_volume_ok(df: pd.DataFrame) -> bool:
    zero_volume = pd.to_numeric(df["volume"], errors="coerce").fillna(np.nan).eq(0)
    if not bool(zero_volume.any()):
        return True
    scoped = df.loc[zero_volume, ["vol_ratio", "vol_ratio_med_20", "vol_zscore_20"]]
    values = scoped.to_numpy(dtype=float)
    return bool(np.isfinite(values[~np.isnan(values)]).all())


def _pressure_proxy_consistency_ok(df: pd.DataFrame) -> bool:
    vol_ratio = pd.to_numeric(df["vol_ratio"], errors="coerce")
    upper_wick_ratio = pd.to_numeric(df["upper_wick_ratio"], errors="coerce")
    lower_wick_ratio = pd.to_numeric(df["lower_wick_ratio"], errors="coerce")
    dominant_wick_ratio = pd.to_numeric(df["dominant_wick_ratio"], errors="coerce")
    close_strength = pd.to_numeric(df["close_strength"], errors="coerce")
    body_strength = pd.to_numeric(df["body_strength"], errors="coerce")

    wick_bias_expected = lower_wick_ratio - upper_wick_ratio
    wick_bias = pd.to_numeric(df["wick_bias"], errors="coerce")
    wick_ok = np.allclose(
        wick_bias.fillna(0.0).to_numpy(dtype=float),
        wick_bias_expected.fillna(0.0).to_numpy(dtype=float),
        atol=1e-12,
    )
    norm_ok = np.allclose(
        pd.to_numeric(df["signed_tick_pressure_norm"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        (close_strength * vol_ratio).fillna(0.0).to_numpy(dtype=float),
        atol=1e-12,
    )
    body_ok = np.allclose(
        pd.to_numeric(df["signed_tick_pressure_body"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        (body_strength * vol_ratio).fillna(0.0).to_numpy(dtype=float),
        atol=1e-12,
    )
    wick_pressure_expected = wick_bias_expected * vol_ratio
    wick_pressure_ok = np.allclose(
        pd.to_numeric(df["signed_tick_pressure_wick"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        wick_pressure_expected.fillna(0.0).to_numpy(dtype=float),
        atol=1e-12,
    )
    close_sign = pd.Series(np.sign(close_strength), index=df.index, dtype=float)
    wick_sign = pd.Series(np.sign(wick_bias_expected), index=df.index, dtype=float)
    body_sign = pd.Series(np.sign(body_strength), index=df.index, dtype=float)
    wick_opposes_close = close_sign.ne(0) & wick_sign.ne(0) & wick_sign.ne(close_sign)
    body_opposes_close = close_sign.ne(0) & body_sign.ne(0) & body_sign.ne(close_sign)
    rejection_regime = dominant_wick_ratio.ge(0.35)
    adjusted_body = (body_strength * vol_ratio) * np.where(body_opposes_close, 1.2, 1.0)
    adjusted_wick = wick_pressure_expected * np.where(wick_opposes_close, 1.5, 1.0)
    norm_weight = pd.Series(
        np.where(rejection_regime, 0.15, 0.35), index=df.index, dtype=float
    )
    body_weight = pd.Series(
        np.where(rejection_regime, 0.20, 0.30), index=df.index, dtype=float
    )
    wick_weight = pd.Series(
        np.where(rejection_regime, 0.65, 0.35), index=df.index, dtype=float
    )
    blend_expected = (
        norm_weight * (close_strength * vol_ratio)
        + body_weight * adjusted_body
        + wick_weight * adjusted_wick
    )
    blend_ok = np.allclose(
        pd.to_numeric(df["signed_tick_pressure_blend"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        blend_expected.fillna(0.0).to_numpy(dtype=float),
        atol=1e-12,
    )
    return bool(wick_ok and norm_ok and body_ok and wick_pressure_ok and blend_ok)


def _warmup_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "vol_20_family_nan_rows": int(
            pd.to_numeric(df["vol_ratio"], errors="coerce").isna().sum()
        ),
        "vol_pct_rank_100_nan_rows": int(
            pd.to_numeric(df["vol_pct_rank_100"], errors="coerce").isna().sum()
        ),
        "vol_slope_5_nan_rows": int(
            pd.to_numeric(df["vol_slope_5"], errors="coerce").isna().sum()
        ),
    }


def _first_valid_row(series: pd.Series) -> int | None:
    valid = pd.to_numeric(series, errors="coerce").notna().to_numpy()
    positions = np.flatnonzero(valid)
    return int(positions[0]) if len(positions) else None


def _flag_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        col: {
            "zeros": int(pd.to_numeric(df[col], errors="coerce").fillna(0).eq(0).sum()),
            "ones": int(pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1).sum()),
        }
        for col in FLAG_COLUMNS
        if col in df.columns
    }


def _pressure_relationship_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    blend = pd.to_numeric(df["signed_tick_pressure_blend"], errors="coerce")
    norm = pd.to_numeric(df["candle_delta_proxy"], errors="coerce")
    body = pd.to_numeric(df["signed_tick_pressure_body"], errors="coerce")
    wick = pd.to_numeric(df["signed_tick_pressure_wick"], errors="coerce")
    valid = blend.notna() & norm.notna()
    if not bool(valid.any()):
        return {
            "valid_rows": 0,
            "blend_vs_norm_correlation": None,
            "mean_abs_blend_norm_gap": None,
            "sign_disagreement_rate_pct": None,
            "blend_near_zero_rate_pct": None,
            "mean_abs_body_component_share": None,
            "mean_abs_wick_component_share": None,
        }

    blend_v = blend.loc[valid]
    norm_v = norm.loc[valid]
    body_v = body.loc[valid].fillna(0.0)
    wick_v = wick.loc[valid].fillna(0.0)
    denom = blend_v.abs().clip(lower=PRESSURE_RELATIONSHIP_DENOM_FLOOR)
    sign_disagreement = pd.Series(
        np.sign(blend_v), index=blend_v.index, dtype=float
    ) != pd.Series(np.sign(norm_v), index=norm_v.index, dtype=float)
    near_zero_rate = blend_v.abs().lt(PRESSURE_RELATIONSHIP_DENOM_FLOOR).mean()
    corr = blend_v.corr(norm_v)
    return {
        "valid_rows": int(valid.sum()),
        "blend_vs_norm_correlation": float(corr) if pd.notna(corr) else None,
        "mean_abs_blend_norm_gap": float((blend_v - norm_v).abs().mean()),
        "sign_disagreement_rate_pct": float(sign_disagreement.mean() * 100.0),
        "blend_near_zero_rate_pct": float(near_zero_rate * 100.0),
        "mean_abs_body_component_share": float(
            (0.3 * body_v).abs().div(denom).dropna().mean()
        ),
        "mean_abs_wick_component_share": float(
            (0.2 * wick_v).abs().div(denom).dropna().mean()
        ),
    }


def _build_volume_figure(df: pd.DataFrame, *, title: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.42, 0.18, 0.20, 0.20],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    pressure = pd.to_numeric(df["signed_tick_pressure_blend"], errors="coerce")
    volume_colors = np.where(
        pressure.isna(),
        "rgba(128, 128, 128, 0.55)",
        np.where(
            pressure >= 0,
            "rgba(34, 139, 34, 0.75)",
            "rgba(200, 0, 0, 0.75)",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="volume_signed_proxy_colored",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )
    for col in ("vol_ratio", "vol_pct_rank_100"):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col),
                row=2,
                col=1,
            )
    for col in (
        "candle_delta_proxy",
        "signed_tick_pressure_blend",
        "signed_tick_pressure_z",
        "effort_vs_result",
        "upper_rejection_effort",
        "lower_rejection_effort",
    ):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col),
                row=3,
                col=1,
            )

    vsa_mask = pd.Series(False, index=df.index)
    for col in (
        "vsa_absorption",
        "vsa_directional",
        "vsa_no_demand",
        "vsa_no_supply",
        "vsa_churn",
        "vsa_effort_failure",
    ):
        if col in df.columns:
            vsa_mask |= pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)
    marker_df = df.loc[vsa_mask]
    if not marker_df.empty:
        fig.add_trace(
            go.Scatter(
                x=marker_df["timestamp"],
                y=marker_df["close"],
                mode="markers",
                name="VSA flags",
                marker=dict(symbol="diamond", size=8),
            ),
            row=4,
            col=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1100,
        xaxis_rangeslider_visible=False,
    )
    return fig


def validate_volume(
    df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame | None = None,
    live_df: pd.DataFrame | None = None,
    research_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Volume Validation",
    source_parity_ok: bool | None = None,
) -> dict[str, object]:
    audit_df = (
        summary_df
        if summary_df is not None
        else (research_df if research_df is not None else df)
    )

    missing = sorted(REQUIRED_VOLUME_COLUMNS - set(audit_df.columns))
    if missing:
        raise ValueError(f"validate_volume: missing required columns: {missing}")

    live_research_parity_ok = None
    if live_df is not None and research_df is not None:
        non_research_cols = [
            col for col in research_df.columns if not col.startswith("r_")
        ]
        live_research_parity_ok = bool(
            live_df[non_research_cols].equals(research_df[non_research_cols])
        )

    no_label_contamination_ok = True
    if live_df is not None:
        no_label_contamination_ok = not any(
            col.startswith(("r_", "label_", "future_")) for col in live_df.columns
        )

    display_start_row = (
        max(len(audit_df) - len(df), 0) if len(audit_df) >= len(df) else 0
    )
    display_end_row = display_start_row + len(df) - 1 if len(df) else None
    first_valid_rows = {
        "vol_20_family_first_valid_row": _first_valid_row(audit_df["vol_ratio"]),
        "vol_pct_rank_100_first_valid_row": _first_valid_row(
            audit_df["vol_pct_rank_100"]
        ),
        "vol_slope_5_first_valid_row": _first_valid_row(audit_df["vol_slope_5"]),
    }
    post_warmup_only = all(
        value is None or display_start_row >= value
        for value in first_valid_rows.values()
    )

    summary = {
        "row_count": int(len(audit_df)),
        "display_window": {
            "displayed_row_count": int(len(df)),
            "displayed_slice_start_row": int(display_start_row),
            "displayed_slice_end_row": (
                int(display_end_row) if display_end_row is not None else None
            ),
            "warmup_reference_frame_row_count": int(len(audit_df)),
            "reported_slice_is_post_warmup_only": bool(post_warmup_only),
        },
        "warmup_counts": _warmup_counts(audit_df),
        "warmup_first_valid_rows": first_valid_rows,
        "checks": {
            "required_columns_present": True,
            "no_inf_values": _no_inf_ok(audit_df),
            "zero_range_handling_ok": _zero_range_ok(audit_df),
            "zero_volume_handling_ok": _zero_volume_ok(audit_df),
            "pressure_proxy_consistency_ok": _pressure_proxy_consistency_ok(audit_df),
            "source_parity_ok": source_parity_ok,
            "live_research_parity_ok": live_research_parity_ok,
            "no_label_contamination_ok": no_label_contamination_ok,
        },
        "flag_value_counts": _flag_counts(df),
        "summary_stats": {
            col: _continuous_stats(audit_df[col]) for col in SUMMARY_COLUMNS
        },
        "pressure_relationship": _pressure_relationship_summary(audit_df),
    }

    html_path = None
    if outpath is not None:
        fig = _build_volume_figure(df, title=title)
        html_path = save_figure_html(fig, outpath)

    return {"summary": summary, "html_path": html_path}
