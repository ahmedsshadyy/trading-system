"""
Volume context indicators.

Canonical internal volume features built on schema-normalized ``volume``
with optional research-only extras behind an explicit flag.

Signed pressure features here are tick-volume-backed pressure proxies,
not true bid/ask delta, net volume, or real signed traded volume.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import ta_core as ta
from src.indicators._helpers.schema import normalize_candle_schema

EPS = 1e-12
BASELINE_PERIOD = 20
PCT_RANK_PERIOD = 100
SLOPE_PERIOD = 5
PRESSURE_DIVERGENCE_THRESHOLD = 0.5
PRESSURE_REJECTION_WICK_THRESHOLD = 0.35
PRESSURE_WICK_OPPOSES_CLOSE_MULTIPLIER = 1.5
PRESSURE_BODY_OPPOSES_CLOSE_MULTIPLIER = 1.2
EFFORT_RESULT_RANGE_ATR_FLOOR = 0.05
EFFORT_BODY_FRAC_FLOOR = 0.05
RESULT_EFFORT_VOL_RATIO_FLOOR = 0.10


def _normalize_volume_input(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_candle_schema(df, require_volume=True)


def _ensure_atr(df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
    out = df.copy()
    if atr_col not in out.columns:
        out[atr_col] = ta.atr(out["high"], out["low"], out["close"], length=14)
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & (denominator != 0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _rolling_percent_rank(window: np.ndarray) -> float:
    current = window[-1]
    return float(np.count_nonzero(window <= current) / len(window))


def _ols_slope_5(window: np.ndarray) -> float:
    x = np.arange(len(window), dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    y_centered = window - window.mean()
    return float(np.dot(x_centered, y_centered) / denom)


def _add_research_only_volume_columns(out: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(out["close"], errors="coerce")
    body_direction = pd.to_numeric(out["body_direction"], errors="coerce")
    spike = (
        pd.to_numeric(out["vol_extreme_pct95"], errors="coerce").fillna(0).astype(int)
    )
    pressure_blend = pd.to_numeric(out["signed_tick_pressure_blend"], errors="coerce")

    out["r_vol_forward_1_return"] = close.shift(-1).div(close).sub(1.0)
    out["r_vol_forward_3_return"] = close.shift(-3).div(close).sub(1.0)
    out["r_vol_forward_5_return"] = close.shift(-5).div(close).sub(1.0)
    out["r_pressure_forward_1_return"] = out["r_vol_forward_1_return"]
    out["r_pressure_forward_3_return"] = out["r_vol_forward_3_return"]
    out["r_pressure_forward_5_return"] = out["r_vol_forward_5_return"]

    forward_sign = pd.Series(
        np.sign(out["r_vol_forward_3_return"]), index=out.index, dtype=float
    )
    pressure_sign = pd.Series(np.sign(pressure_blend), index=out.index, dtype=float)

    followthrough = spike.eq(1) & body_direction.ne(0) & forward_sign.eq(body_direction)
    reversal = spike.eq(1) & body_direction.ne(0) & forward_sign.eq(-body_direction)

    out["r_post_spike_followthrough_label"] = followthrough.astype("int8")
    out["r_post_spike_reversal_label"] = reversal.astype("int8")
    out["r_pressure_followthrough_label"] = (
        pressure_sign.ne(0) & forward_sign.eq(pressure_sign)
    ).astype("int8")
    out["r_pressure_reversal_label"] = (
        pressure_sign.ne(0) & forward_sign.eq(-pressure_sign)
    ).astype("int8")
    range_atr = pd.to_numeric(out["bar_range_atr"], errors="coerce")
    body_frac = pd.to_numeric(out["bar_body_frac"], errors="coerce")
    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce")
    out["r_effort_vs_result_raw"] = vol_ratio / np.maximum(range_atr, EPS)
    out["r_effort_vs_body_raw"] = vol_ratio / np.maximum(body_frac, EPS)
    out["r_result_vs_effort_raw"] = range_atr / np.maximum(vol_ratio, EPS)
    out["r_body_result_vs_effort_raw"] = body_frac / np.maximum(vol_ratio, EPS)
    return out


def add_volume_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical volume baseline and trend columns."""
    out = _normalize_volume_input(df)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)

    out["vol_sma_20"] = volume.rolling(
        BASELINE_PERIOD, min_periods=BASELINE_PERIOD
    ).mean()
    out["vol_med_20"] = volume.rolling(
        BASELINE_PERIOD, min_periods=BASELINE_PERIOD
    ).median()
    out["vol_std_20"] = volume.rolling(
        BASELINE_PERIOD, min_periods=BASELINE_PERIOD
    ).std(ddof=0)
    out["vol_zscore_20"] = pd.Series(np.nan, index=out.index, dtype=float)
    std_mask = out["vol_std_20"] > 0
    out.loc[std_mask, "vol_zscore_20"] = (
        volume.loc[std_mask] - out.loc[std_mask, "vol_sma_20"]
    ) / out.loc[std_mask, "vol_std_20"]
    out["vol_ratio"] = _safe_divide(volume, out["vol_sma_20"])
    out["vol_ratio_med_20"] = _safe_divide(volume, out["vol_med_20"])
    out["vol_pct_rank_100"] = volume.rolling(
        PCT_RANK_PERIOD,
        min_periods=PCT_RANK_PERIOD,
    ).apply(_rolling_percent_rank, raw=True)

    out["vol_slope_5"] = volume.rolling(
        SLOPE_PERIOD,
        min_periods=SLOPE_PERIOD,
    ).apply(_ols_slope_5, raw=True)
    out["vol_slope_5_norm"] = _safe_divide(out["vol_slope_5"], out["vol_sma_20"])
    out["vol_ema_5"] = volume.ewm(span=5, adjust=False, min_periods=5).mean()
    out["vol_ema_20"] = volume.ewm(span=20, adjust=False, min_periods=20).mean()
    out["vol_ema_ratio_5_20"] = _safe_divide(out["vol_ema_5"], out["vol_ema_20"])
    return out


def add_volume_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized binary participation flags."""
    required = {"vol_sma_20", "vol_ratio", "vol_pct_rank_100"}
    out = (
        add_volume_baselines(df)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )

    out["vol_above_avg"] = (
        pd.to_numeric(out["volume"], errors="coerce")
        > pd.to_numeric(out["vol_sma_20"], errors="coerce")
    ).astype("int8")
    out["vol_below_avg"] = (
        pd.to_numeric(out["volume"], errors="coerce")
        < pd.to_numeric(out["vol_sma_20"], errors="coerce")
    ).astype("int8")
    out["vol_above_1_5x"] = (
        pd.to_numeric(out["vol_ratio"], errors="coerce") > 1.5
    ).astype("int8")
    out["vol_above_2_0x"] = (
        pd.to_numeric(out["vol_ratio"], errors="coerce") > 2.0
    ).astype("int8")
    out["vol_below_0_8x"] = (
        pd.to_numeric(out["vol_ratio"], errors="coerce") < 0.8
    ).astype("int8")
    out["vol_extreme_pct90"] = (
        pd.to_numeric(out["vol_pct_rank_100"], errors="coerce") >= 0.90
    ).astype("int8")
    out["vol_extreme_pct95"] = (
        pd.to_numeric(out["vol_pct_rank_100"], errors="coerce") >= 0.95
    ).astype("int8")
    return out


def add_effort_result_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add candle spread, result, and effort/result proxy columns."""
    required = {"vol_ratio", "vol_above_avg", "vol_pct_rank_100"}
    out = (
        add_volume_flags(df)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )
    out = _ensure_atr(out, atr_col)

    open_ = pd.to_numeric(out["open"], errors="coerce").astype(float)
    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    atr = pd.to_numeric(out[atr_col], errors="coerce").astype(float)
    bar_range = high - low
    bar_body = (close - open_).abs()

    out["bar_range"] = bar_range
    out["bar_range_atr"] = _safe_divide(bar_range, atr)
    out["bar_body"] = bar_body
    out["bar_body_frac"] = np.where(bar_range > 0, bar_body / bar_range, 0.0)

    close_pos = pd.Series(np.nan, index=out.index, dtype=float)
    valid_range = bar_range > 0
    close_pos.loc[valid_range] = ((close - low) / bar_range).loc[valid_range]
    out["close_pos_in_range"] = close_pos.clip(lower=0.0, upper=1.0)

    direction = np.zeros(len(out), dtype=np.int8)
    direction[(close > open_).to_numpy()] = 1
    direction[(close < open_).to_numpy()] = -1
    out["body_direction"] = direction

    range_sma_20 = bar_range.rolling(
        BASELINE_PERIOD, min_periods=BASELINE_PERIOD
    ).mean()
    out["true_spread_ratio"] = _safe_divide(bar_range, range_sma_20)

    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce").astype(float)
    body_frac = pd.to_numeric(out["bar_body_frac"], errors="coerce").astype(float)
    range_atr = pd.to_numeric(out["bar_range_atr"], errors="coerce").astype(float)

    effective_range_atr = pd.Series(np.nan, index=out.index, dtype=float)
    mask = range_atr.notna()
    effective_range_atr.loc[mask] = np.maximum(
        range_atr.loc[mask],
        EFFORT_RESULT_RANGE_ATR_FLOOR,
    )
    out["effective_range_atr_floor"] = effective_range_atr
    out["effort_vs_result"] = pd.Series(np.nan, index=out.index, dtype=float)
    mask = vol_ratio.notna() & effective_range_atr.notna()
    out.loc[mask, "effort_vs_result"] = (
        vol_ratio.loc[mask] / effective_range_atr.loc[mask]
    )

    effective_body_frac = pd.Series(np.nan, index=out.index, dtype=float)
    mask = body_frac.notna()
    effective_body_frac.loc[mask] = np.maximum(
        body_frac.loc[mask],
        EFFORT_BODY_FRAC_FLOOR,
    )
    out["effective_body_frac_floor"] = effective_body_frac
    out["effort_vs_body"] = pd.Series(np.nan, index=out.index, dtype=float)
    mask = vol_ratio.notna() & effective_body_frac.notna()
    out.loc[mask, "effort_vs_body"] = (
        vol_ratio.loc[mask] / effective_body_frac.loc[mask]
    )

    out["result_vs_effort"] = pd.Series(np.nan, index=out.index, dtype=float)
    effective_vol_ratio = pd.Series(np.nan, index=out.index, dtype=float)
    mask = vol_ratio.notna()
    effective_vol_ratio.loc[mask] = np.maximum(
        vol_ratio.loc[mask],
        RESULT_EFFORT_VOL_RATIO_FLOOR,
    )
    out["effective_vol_ratio_floor"] = effective_vol_ratio
    mask = range_atr.notna() & effective_vol_ratio.notna()
    out.loc[mask, "result_vs_effort"] = (
        range_atr.loc[mask] / effective_vol_ratio.loc[mask]
    )

    out["body_result_vs_effort"] = pd.Series(np.nan, index=out.index, dtype=float)
    mask = body_frac.notna() & effective_vol_ratio.notna()
    out.loc[mask, "body_result_vs_effort"] = (
        body_frac.loc[mask] / effective_vol_ratio.loc[mask]
    )

    return out


def add_delta_proxy_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add the base signed tick-pressure proxy components."""
    required = {
        "effort_vs_result",
        "close_pos_in_range",
        "body_direction",
        "bar_body_frac",
    }
    out = (
        add_effort_result_features(df, atr_col=atr_col)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )

    close_pos = pd.to_numeric(out["close_pos_in_range"], errors="coerce").astype(float)
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float)
    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce").astype(float)
    body_direction = pd.to_numeric(out["body_direction"], errors="coerce").astype(float)
    body_frac = pd.to_numeric(out["bar_body_frac"], errors="coerce").astype(float)

    close_strength = pd.Series(0.0, index=out.index, dtype=float)
    finite_close = close_pos.notna()
    close_strength.loc[finite_close] = 2.0 * close_pos.loc[finite_close] - 1.0
    out["close_strength"] = close_strength
    out["body_strength"] = body_direction * body_frac
    out["signed_tick_pressure_raw"] = close_strength * volume
    out["signed_tick_pressure_norm"] = close_strength * vol_ratio
    out.loc[vol_ratio.isna(), "signed_tick_pressure_norm"] = np.nan
    out["signed_tick_pressure_body"] = out["body_strength"] * vol_ratio
    out.loc[vol_ratio.isna(), "signed_tick_pressure_body"] = np.nan
    out["delta_proxy_raw"] = out["signed_tick_pressure_raw"]
    out["delta_proxy_norm"] = out["signed_tick_pressure_norm"]
    out["delta_proxy_body"] = out["signed_tick_pressure_body"]
    out["candle_delta_proxy"] = out["signed_tick_pressure_norm"]
    return out


def add_vsa_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add standardized VSA-style proxy flags."""
    required = {
        "candle_delta_proxy",
        "bar_range_atr",
        "true_spread_ratio",
        "close_strength",
        "vol_pct_rank_100",
    }
    out = (
        add_delta_proxy_features(df, atr_col=atr_col)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )

    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce").astype(float)
    body_frac = pd.to_numeric(out["bar_body_frac"], errors="coerce").astype(float)
    bar_range_atr = pd.to_numeric(out["bar_range_atr"], errors="coerce").astype(float)
    true_spread_ratio = pd.to_numeric(out["true_spread_ratio"], errors="coerce").astype(
        float
    )
    body_direction = pd.to_numeric(out["body_direction"], errors="coerce").astype(float)
    bar_range = pd.to_numeric(out["bar_range"], errors="coerce").astype(float)
    close_pos = pd.to_numeric(out["close_pos_in_range"], errors="coerce").astype(float)
    vol_pct_rank = pd.to_numeric(out["vol_pct_rank_100"], errors="coerce").astype(float)
    close_strength = pd.to_numeric(out["close_strength"], errors="coerce").astype(float)
    range_sma_20 = bar_range.rolling(
        BASELINE_PERIOD, min_periods=BASELINE_PERIOD
    ).mean()

    out["vsa_absorption"] = (
        (vol_ratio > 1.5) & (body_frac < 0.35) & (bar_range_atr < true_spread_ratio)
    ).astype("int8")
    out["vsa_directional"] = (
        (vol_ratio > 1.5) & (body_frac > 0.60) & (bar_range_atr > 1.0)
    ).astype("int8")
    out["vsa_no_demand"] = (
        (body_direction == 1)
        & (vol_ratio < 0.8)
        & (bar_range < range_sma_20)
        & (close_pos < 0.75)
    ).astype("int8")
    out["vsa_no_supply"] = (
        (body_direction == -1)
        & (vol_ratio < 0.8)
        & (bar_range < range_sma_20)
        & (close_pos > 0.25)
    ).astype("int8")
    out["vsa_climactic_up"] = (
        (vol_pct_rank >= 0.95)
        & (body_direction == 1)
        & (close_pos >= 0.70)
        & (bar_range_atr > 1.25)
    ).astype("int8")
    out["vsa_climactic_down"] = (
        (vol_pct_rank >= 0.95)
        & (body_direction == -1)
        & (close_pos <= 0.30)
        & (bar_range_atr > 1.25)
    ).astype("int8")
    out["vsa_churn"] = (
        (vol_ratio > 1.5) & (body_frac < 0.30) & (bar_range_atr > 0.75)
    ).astype("int8")
    out["vsa_effort_failure"] = (
        (vol_ratio > 1.5) & (close_strength.abs() < 0.20)
    ).astype("int8")
    return out


def add_wick_effort_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add wick ratios and wick-effort rejection proxies."""
    required = {"vsa_absorption", "bar_range", "vol_ratio"}
    out = (
        add_vsa_features(df, atr_col=atr_col)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )

    open_ = pd.to_numeric(out["open"], errors="coerce").astype(float)
    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    bar_range = pd.to_numeric(out["bar_range"], errors="coerce").astype(float)
    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce").astype(float)

    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    upper_wick = high - body_top
    lower_wick = body_bottom - low

    out["upper_wick_ratio"] = np.where(bar_range > 0, upper_wick / bar_range, 0.0)
    out["lower_wick_ratio"] = np.where(bar_range > 0, lower_wick / bar_range, 0.0)
    out["dominant_wick_ratio"] = np.maximum(
        out["upper_wick_ratio"], out["lower_wick_ratio"]
    )
    out["wick_imbalance"] = out["upper_wick_ratio"] - out["lower_wick_ratio"]
    out["upper_rejection_effort"] = out["upper_wick_ratio"] * vol_ratio
    out["lower_rejection_effort"] = out["lower_wick_ratio"] * vol_ratio
    out.loc[vol_ratio.isna(), ["upper_rejection_effort", "lower_rejection_effort"]] = (
        np.nan
    )
    out["wick_effort_imbalance"] = (
        out["upper_rejection_effort"] - out["lower_rejection_effort"]
    )
    return out


def add_signed_tick_pressure_features(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add the canonical signed tick-pressure proxy family."""
    required = {
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_strength",
        "body_strength",
        "vol_ratio",
    }
    out = (
        add_wick_effort_features(df, atr_col=atr_col)
        if not required.issubset(df.columns)
        else _normalize_volume_input(df)
    )

    vol_ratio = pd.to_numeric(out["vol_ratio"], errors="coerce").astype(float)
    close_strength = pd.to_numeric(out["close_strength"], errors="coerce").astype(float)
    body_strength = pd.to_numeric(out["body_strength"], errors="coerce").astype(float)
    upper_wick_ratio = pd.to_numeric(out["upper_wick_ratio"], errors="coerce").astype(
        float
    )
    lower_wick_ratio = pd.to_numeric(out["lower_wick_ratio"], errors="coerce").astype(
        float
    )
    dominant_wick_ratio = pd.to_numeric(
        out["dominant_wick_ratio"], errors="coerce"
    ).astype(float)
    body_direction = pd.to_numeric(out["body_direction"], errors="coerce").astype(float)

    out["wick_bias"] = lower_wick_ratio - upper_wick_ratio
    out["signed_tick_pressure_wick"] = out["wick_bias"] * vol_ratio
    out.loc[vol_ratio.isna(), "signed_tick_pressure_wick"] = np.nan
    norm_component = pd.to_numeric(
        out["signed_tick_pressure_norm"], errors="coerce"
    ).astype(float)
    body_component = pd.to_numeric(
        out["signed_tick_pressure_body"], errors="coerce"
    ).astype(float)
    wick_component = pd.to_numeric(
        out["signed_tick_pressure_wick"], errors="coerce"
    ).astype(float)
    close_sign = pd.Series(np.sign(close_strength), index=out.index, dtype=float)
    wick_sign = pd.Series(np.sign(out["wick_bias"]), index=out.index, dtype=float)
    body_sign = pd.Series(np.sign(body_strength), index=out.index, dtype=float)

    wick_opposes_close = close_sign.ne(0) & wick_sign.ne(0) & wick_sign.ne(close_sign)
    body_opposes_close = close_sign.ne(0) & body_sign.ne(0) & body_sign.ne(close_sign)
    rejection_regime = dominant_wick_ratio.ge(PRESSURE_REJECTION_WICK_THRESHOLD)

    adjusted_body_component = body_component * np.where(
        body_opposes_close,
        PRESSURE_BODY_OPPOSES_CLOSE_MULTIPLIER,
        1.0,
    )
    adjusted_wick_component = wick_component * np.where(
        wick_opposes_close,
        PRESSURE_WICK_OPPOSES_CLOSE_MULTIPLIER,
        1.0,
    )

    norm_weight = pd.Series(
        np.where(rejection_regime, 0.15, 0.35),
        index=out.index,
        dtype=float,
    )
    body_weight = pd.Series(
        np.where(rejection_regime, 0.20, 0.30),
        index=out.index,
        dtype=float,
    )
    wick_weight = pd.Series(
        np.where(rejection_regime, 0.65, 0.35),
        index=out.index,
        dtype=float,
    )

    out["signed_tick_pressure_blend"] = (
        norm_weight * norm_component
        + body_weight * adjusted_body_component
        + wick_weight * adjusted_wick_component
    )
    out.loc[vol_ratio.isna(), "signed_tick_pressure_blend"] = np.nan

    blend = pd.to_numeric(out["signed_tick_pressure_blend"], errors="coerce").astype(
        float
    )
    blend_mean = blend.rolling(BASELINE_PERIOD, min_periods=BASELINE_PERIOD).mean()
    blend_std = blend.rolling(BASELINE_PERIOD, min_periods=BASELINE_PERIOD).std(ddof=0)
    out["signed_tick_pressure_z"] = pd.Series(np.nan, index=out.index, dtype=float)
    valid_z = blend.notna() & blend_std.gt(0)
    out.loc[valid_z, "signed_tick_pressure_z"] = (
        blend.loc[valid_z] - blend_mean.loc[valid_z]
    ) / blend_std.loc[valid_z]

    blend_sign = pd.Series(np.sign(blend), index=out.index, dtype=float)
    close_sign = pd.Series(np.sign(close_strength), index=out.index, dtype=float)
    out["pressure_agrees_with_body"] = (
        blend_sign.eq(body_direction) & body_direction.ne(0)
    ).astype("int8")
    out["pressure_agrees_with_close_location"] = (
        blend_sign.eq(close_sign) & close_sign.ne(0)
    ).astype("int8")
    out["pressure_divergence_flag"] = (
        body_direction.ne(0)
        & blend_sign.ne(0)
        & blend_sign.ne(body_direction)
        & blend.abs().ge(PRESSURE_DIVERGENCE_THRESHOLD)
    ).astype("int8")
    out["pressure_extreme_pos"] = (
        pd.to_numeric(out["signed_tick_pressure_z"], errors="coerce") >= 2.0
    ).astype("int8")
    out["pressure_extreme_neg"] = (
        pd.to_numeric(out["signed_tick_pressure_z"], errors="coerce") <= -2.0
    ).astype("int8")
    return out


def add_volume_features(
    df: pd.DataFrame,
    *,
    include_research_only: bool = False,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Add the full canonical volume context layer."""
    out = _normalize_volume_input(df)
    out = _ensure_atr(out, atr_col)
    out = add_volume_baselines(out)
    out = add_volume_flags(out)
    out = add_effort_result_features(out, atr_col=atr_col)
    out = add_delta_proxy_features(out, atr_col=atr_col)
    out = add_vsa_features(out, atr_col=atr_col)
    out = add_wick_effort_features(out, atr_col=atr_col)
    out = add_signed_tick_pressure_features(out, atr_col=atr_col)
    if include_research_only:
        out = _add_research_only_volume_columns(out)
    return out


def add_volume_ratio(df: pd.DataFrame, period: int = BASELINE_PERIOD) -> pd.DataFrame:
    """Backward-compatible wrapper for the canonical volume baseline family."""
    if period != BASELINE_PERIOD:
        raise ValueError(
            f"Only period={BASELINE_PERIOD} is supported in the frozen contract"
        )
    out = add_volume_baselines(df)
    out = add_volume_flags(out)
    return out


def add_key_volume_flags(
    df: pd.DataFrame, period: int = BASELINE_PERIOD
) -> pd.DataFrame:
    """Backward-compatible wrapper for standardized participation flags."""
    if period != BASELINE_PERIOD:
        raise ValueError(
            f"Only period={BASELINE_PERIOD} is supported in the frozen contract"
        )
    out = add_volume_baselines(df)
    return add_volume_flags(out)


def add_candle_delta_proxy(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Backward-compatible wrapper for the signed tick-pressure family."""
    return add_signed_tick_pressure_features(df, atr_col=atr_col)


def add_vsa(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Backward-compatible wrapper for VSA-style proxy features."""
    return add_vsa_features(df, atr_col=atr_col)


def add_wick_ratio(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Backward-compatible wrapper for wick and rejection-effort features."""
    return add_wick_effort_features(df, atr_col=atr_col)
