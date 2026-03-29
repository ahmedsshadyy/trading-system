"""
Value & Reference Level indicators.

Anchored VWAP is implemented here as a tick-volume-weighted anchored
fair-value proxy built on canonical internal ``volume``. It is not true
exchange VWAP.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import ta_core as ta
from src.indicators._helpers.schema import normalize_candle_schema

AVWAP_SLOPE_SHORT = 5
AVWAP_SLOPE_LONG = 20
AVWAP_TREND_EPS = 1e-6
AVWAP_ALLOWED_CLASSES = {"live_safe", "retrospective", "hybrid"}


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


def _ols_slope(window: np.ndarray) -> float:
    x = np.arange(len(window), dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    y_centered = window - window.mean()
    return float(np.dot(x_centered, y_centered) / denom)


def _rolling_ols_slope(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).apply(_ols_slope, raw=True)


def _validate_anchor_index(anchor_idx: int, n: int, *, name: str) -> int:
    idx = int(anchor_idx)
    if idx < 0 or idx >= n:
        raise ValueError(f"{name} must be within [0, {n - 1}]")
    return idx


def compute_anchored_vwap(
    df: pd.DataFrame,
    anchor_idx: int,
    *,
    atr_col: str = "atr_14",
) -> pd.DataFrame:
    """Compute anchored VWAP core math from an explicit anchor index.

    This is the low-level builder. It uses canonical internal ``volume``,
    which is typically backed by raw provider ``tickVolume``.
    """
    out = normalize_candle_schema(df, require_volume=True)
    out = _ensure_atr(out, atr_col)
    n = len(out)
    anchor_idx = _validate_anchor_index(anchor_idx, n, name="anchor_idx")

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(out["volume"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(out[atr_col], errors="coerce").astype(float)
    typical_price = (high + low + close) / 3.0

    avwap = np.full(n, np.nan, dtype=float)
    avwap_std = np.full(n, np.nan, dtype=float)
    upper1 = np.full(n, np.nan, dtype=float)
    lower1 = np.full(n, np.nan, dtype=float)
    upper2 = np.full(n, np.nan, dtype=float)
    lower2 = np.full(n, np.nan, dtype=float)
    dev_sigma = np.full(n, np.nan, dtype=float)
    bars_since_anchor = np.full(n, np.nan, dtype=float)

    cum_tpv = 0.0
    cum_v = 0.0
    cum_tp2v = 0.0

    for i in range(anchor_idx, n):
        tp_i = typical_price[i]
        v_i = volume[i]
        if np.isfinite(tp_i) and np.isfinite(v_i):
            cum_tpv += tp_i * v_i
            cum_v += v_i
            cum_tp2v += (tp_i**2) * v_i

        if cum_v > 0:
            vwap_val = cum_tpv / cum_v
            variance = max((cum_tp2v / cum_v) - (vwap_val**2), 0.0)
            std = float(np.sqrt(variance))

            avwap[i] = vwap_val
            avwap_std[i] = std
            upper1[i] = vwap_val + std
            lower1[i] = vwap_val - std
            upper2[i] = vwap_val + 2.0 * std
            lower2[i] = vwap_val - 2.0 * std
            dev_sigma[i] = (close[i] - vwap_val) / std if std > 0 else 0.0
            bars_since_anchor[i] = float(i - anchor_idx)

    out["avwap"] = avwap
    out["avwap_std"] = avwap_std
    out["avwap_upper_1"] = upper1
    out["avwap_lower_1"] = lower1
    out["avwap_upper_2"] = upper2
    out["avwap_lower_2"] = lower2
    out["avwap_dev_sigma"] = dev_sigma

    avwap_series = pd.Series(avwap, index=out.index, dtype=float)
    close_series = pd.Series(close, index=out.index, dtype=float)

    out["avwap_distance"] = close_series - avwap_series
    out["avwap_distance_atr"] = _safe_divide(out["avwap_distance"], atr)
    out["avwap_distance_pct"] = _safe_divide(out["avwap_distance"], avwap_series)
    out["avwap_above"] = (close_series > avwap_series).astype("int8")
    out["avwap_below"] = (close_series < avwap_series).astype("int8")
    out["avwap_inside_1sigma"] = (
        close_series.ge(pd.Series(lower1, index=out.index))
        & close_series.le(pd.Series(upper1, index=out.index))
        & avwap_series.notna()
    ).astype("int8")
    out["avwap_outside_2sigma"] = (
        (
            close_series.gt(pd.Series(upper2, index=out.index))
            | close_series.lt(pd.Series(lower2, index=out.index))
        )
        & avwap_series.notna()
    ).astype("int8")
    out["avwap_cross_up"] = (
        close_series.gt(avwap_series)
        & close_series.shift(1).le(avwap_series.shift(1))
        & avwap_series.notna()
        & avwap_series.shift(1).notna()
    ).astype("int8")
    out["avwap_cross_down"] = (
        close_series.lt(avwap_series)
        & close_series.shift(1).ge(avwap_series.shift(1))
        & avwap_series.notna()
        & avwap_series.shift(1).notna()
    ).astype("int8")
    out["avwap_slope_5"] = _rolling_ols_slope(avwap_series, AVWAP_SLOPE_SHORT)
    out["avwap_slope_20"] = _rolling_ols_slope(avwap_series, AVWAP_SLOPE_LONG)
    slope_20 = pd.to_numeric(out["avwap_slope_20"], errors="coerce")
    trend_state = np.zeros(n, dtype=np.int8)
    trend_state[slope_20 > AVWAP_TREND_EPS] = 1
    trend_state[slope_20 < -AVWAP_TREND_EPS] = -1
    out["avwap_trend_state"] = trend_state
    out["bars_since_anchor"] = bars_since_anchor
    return out


def _mask_avwap_pre_live(out: pd.DataFrame, *, live_from_idx: int) -> pd.DataFrame:
    masked = out.copy()
    continuous_cols = [
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
        "avwap_slope_5",
        "avwap_slope_20",
        "bars_since_anchor",
    ]
    flag_cols = [
        "avwap_above",
        "avwap_below",
        "avwap_inside_1sigma",
        "avwap_outside_2sigma",
        "avwap_cross_up",
        "avwap_cross_down",
        "avwap_trend_state",
    ]
    if live_from_idx > 0:
        pre_live_index = masked.index[:live_from_idx]
        masked.loc[pre_live_index, continuous_cols] = np.nan
        masked.loc[pre_live_index, flag_cols] = 0
    return masked


def _add_avwap_research_columns(out: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(out["close"], errors="coerce")
    avwap_distance = pd.to_numeric(out["avwap_distance"], errors="coerce")
    avwap_dev_sigma = pd.to_numeric(out["avwap_dev_sigma"], errors="coerce")
    anchor_active = pd.to_numeric(out["avwap"], errors="coerce").notna()

    out["r_avwap_forward_1_return"] = (
        close.shift(-1).div(close).sub(1.0).where(anchor_active)
    )
    out["r_avwap_forward_3_return"] = (
        close.shift(-3).div(close).sub(1.0).where(anchor_active)
    )
    out["r_avwap_forward_5_return"] = (
        close.shift(-5).div(close).sub(1.0).where(anchor_active)
    )

    distance_sign = pd.Series(np.sign(avwap_distance), index=out.index, dtype=float)
    forward_sign = pd.Series(
        np.sign(out["r_avwap_forward_3_return"]), index=out.index, dtype=float
    )
    out["r_avwap_reversion_label"] = (
        anchor_active & distance_sign.ne(0) & forward_sign.eq(-distance_sign)
    ).astype("int8")
    out["r_avwap_breakout_followthrough_label"] = (
        anchor_active & distance_sign.ne(0) & forward_sign.eq(distance_sign)
    ).astype("int8")

    touch_mask = anchor_active & pd.to_numeric(
        out["avwap_distance_atr"], errors="coerce"
    ).abs().le(0.1)
    out["r_avwap_touch_count_since_anchor"] = (
        touch_mask.astype(int).cumsum().where(anchor_active)
    )
    out["r_avwap_max_dev_sigma_since_anchor"] = avwap_dev_sigma.where(
        anchor_active
    ).cummax()
    out["r_avwap_min_dev_sigma_since_anchor"] = avwap_dev_sigma.where(
        anchor_active
    ).cummin()
    return out


def add_anchored_vwap(
    df: pd.DataFrame,
    *,
    anchor_idx: int,
    anchor_label: str,
    anchor_class: str,
    anchor_origin_idx: int | None = None,
    anchor_confirm_idx: int | None = None,
    anchor_live_from_idx: int | None = None,
    atr_col: str = "atr_14",
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add canonical anchored AVWAP context for one explicit anchor."""
    out = normalize_candle_schema(df, require_volume=True)
    n = len(out)
    anchor_idx = _validate_anchor_index(anchor_idx, n, name="anchor_idx")
    if anchor_class not in AVWAP_ALLOWED_CLASSES:
        raise ValueError(f"anchor_class must be one of {sorted(AVWAP_ALLOWED_CLASSES)}")
    if not anchor_label:
        raise ValueError("anchor_label must be non-empty")

    anchor_origin_idx = (
        _validate_anchor_index(anchor_origin_idx, n, name="anchor_origin_idx")
        if anchor_origin_idx is not None
        else anchor_idx
    )
    anchor_confirm_idx = (
        _validate_anchor_index(anchor_confirm_idx, n, name="anchor_confirm_idx")
        if anchor_confirm_idx is not None
        else anchor_idx
    )
    anchor_live_from_idx = (
        _validate_anchor_index(anchor_live_from_idx, n, name="anchor_live_from_idx")
        if anchor_live_from_idx is not None
        else anchor_confirm_idx
    )

    if anchor_origin_idx != anchor_idx:
        raise ValueError(
            "anchor_origin_idx must match anchor_idx for canonical AVWAP math"
        )
    if anchor_confirm_idx < anchor_origin_idx:
        raise ValueError("anchor_confirm_idx must be >= anchor_origin_idx")
    if anchor_live_from_idx < anchor_confirm_idx:
        raise ValueError("anchor_live_from_idx must be >= anchor_confirm_idx")

    out = compute_anchored_vwap(out, anchor_idx=anchor_idx, atr_col=atr_col)
    out = _mask_avwap_pre_live(out, live_from_idx=anchor_live_from_idx)

    meta_mask = np.arange(n) >= anchor_live_from_idx
    out["avwap_anchor_class"] = np.where(meta_mask, anchor_class, None)
    out["avwap_anchor_label"] = np.where(meta_mask, anchor_label, None)
    out["avwap_anchor_idx"] = np.where(meta_mask, float(anchor_idx), np.nan)
    out["avwap_anchor_origin_idx"] = np.where(
        meta_mask, float(anchor_origin_idx), np.nan
    )
    out["avwap_anchor_confirm_idx"] = np.where(
        meta_mask, float(anchor_confirm_idx), np.nan
    )
    out["avwap_anchor_live_from_idx"] = np.where(
        meta_mask, float(anchor_live_from_idx), np.nan
    )

    if include_research_only:
        out = _add_avwap_research_columns(out)
    return out


def add_avwap_from_last_swing(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Optional helper: build AVWAP from the latest swing anchor semantics.

    Preferred semantics:
    - use the latest swing confirm bar as activation timing
    - anchor the AVWAP math to the corresponding swing origin bar
    - classify this helper as ``hybrid`` because the anchor origin is known
      only once the confirm bar occurs

    Fallback semantics when only origin-bar swing annotations exist:
    - use the latest origin-bar swing as a retrospective helper
    """
    out = df.copy()

    if {
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
    }.issubset(out.columns):
        last_confirm_idx = None
        last_label = None
        last_origin_idx = None
        for i in range(len(out) - 1, -1, -1):
            sh_flag = pd.to_numeric(
                out.iloc[i]["swing_high_confirm_flag"], errors="coerce"
            )
            sl_flag = pd.to_numeric(
                out.iloc[i]["swing_low_confirm_flag"], errors="coerce"
            )
            if np.isfinite(sh_flag) and int(sh_flag) == 1:
                last_confirm_idx = i
                last_origin_idx = int(out.iloc[i]["swing_high_confirm_origin_idx"])
                last_label = "last_confirmed_swing_high"
                break
            if np.isfinite(sl_flag) and int(sl_flag) == 1:
                last_confirm_idx = i
                last_origin_idx = int(out.iloc[i]["swing_low_confirm_origin_idx"])
                last_label = "last_confirmed_swing_low"
                break
        if last_confirm_idx is not None and last_origin_idx is not None:
            return add_anchored_vwap(
                out,
                anchor_idx=last_origin_idx,
                anchor_label=str(last_label),
                anchor_class="hybrid",
                anchor_origin_idx=last_origin_idx,
                anchor_confirm_idx=last_confirm_idx,
                anchor_live_from_idx=last_confirm_idx,
                atr_col=atr_col,
                include_research_only=include_research_only,
            )

    sh = (
        pd.to_numeric(out["swing_high"], errors="coerce")
        if "swing_high" in out.columns
        else pd.Series(0, index=out.index)
    )
    sl = (
        pd.to_numeric(out["swing_low"], errors="coerce")
        if "swing_low" in out.columns
        else pd.Series(0, index=out.index)
    )
    last_swing_idx = None
    last_label = None
    for i in range(len(out) - 1, -1, -1):
        if sh.iloc[i] == 1:
            last_swing_idx = i
            last_label = "last_swing_high_origin"
            break
        if sl.iloc[i] == 1:
            last_swing_idx = i
            last_label = "last_swing_low_origin"
            break
    if last_swing_idx is None:
        last_swing_idx = 0
        last_label = "fallback_start_bar"

    return add_anchored_vwap(
        out,
        anchor_idx=last_swing_idx,
        anchor_label=str(last_label),
        anchor_class="retrospective",
        anchor_origin_idx=last_swing_idx,
        anchor_confirm_idx=last_swing_idx,
        anchor_live_from_idx=last_swing_idx,
        atr_col=atr_col,
        include_research_only=include_research_only,
    )


# ---------------------------------------------------------------------------
# Asian Session High / Low
# ---------------------------------------------------------------------------


def add_asian_session_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Asian session (00:00–08:00 UTC) high and low."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    hour = ts.dt.hour
    date = ts.dt.date

    is_asian = hour < 8
    out["_date"] = date

    asian_hl = (
        out[is_asian]
        .groupby("_date")
        .agg(asian_high=("high", "max"), asian_low=("low", "min"))
    )

    out = out.merge(asian_hl, left_on="_date", right_index=True, how="left")
    out["asian_high"] = out.groupby("_date")["asian_high"].ffill()
    out["asian_low"] = out.groupby("_date")["asian_low"].ffill()
    out["asian_range"] = out["asian_high"] - out["asian_low"]

    daily_ranges = asian_hl["asian_high"] - asian_hl["asian_low"]
    avg_range = daily_ranges.rolling(10, min_periods=1).mean()
    avg_range.name = "_avg_asian_range"
    out = out.merge(avg_range, left_on="_date", right_index=True, how="left")
    out["asian_range_ratio"] = out["asian_range"] / out["_avg_asian_range"]

    out.drop(columns=["_date", "_avg_asian_range"], inplace=True)
    return out


# ---------------------------------------------------------------------------
# Previous Day / Week High-Low
# ---------------------------------------------------------------------------


def add_prev_day_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Previous day high/low and position flags."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["_date"] = ts.dt.date

    daily_hl = out.groupby("_date").agg(_dh=("high", "max"), _dl=("low", "min"))
    daily_hl["prev_day_high"] = daily_hl["_dh"].shift(1)
    daily_hl["prev_day_low"] = daily_hl["_dl"].shift(1)

    out = out.merge(
        daily_hl[["prev_day_high", "prev_day_low"]],
        left_on="_date",
        right_index=True,
        how="left",
    )
    out["above_prev_day_high"] = (out["close"] > out["prev_day_high"]).astype(int)
    out["below_prev_day_low"] = (out["close"] < out["prev_day_low"]).astype(int)
    out.drop(columns=["_date"], inplace=True)
    return out


def add_prev_week_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Previous week high/low."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["_yw"] = (
        ts.dt.isocalendar().year.astype(str)
        + "_"
        + ts.dt.isocalendar().week.astype(str).str.zfill(2)
    )

    weekly_hl = out.groupby("_yw").agg(_wh=("high", "max"), _wl=("low", "min"))
    weekly_hl["prev_week_high"] = weekly_hl["_wh"].shift(1)
    weekly_hl["prev_week_low"] = weekly_hl["_wl"].shift(1)

    out = out.merge(
        weekly_hl[["prev_week_high", "prev_week_low"]],
        left_on="_yw",
        right_index=True,
        how="left",
    )
    out["above_prev_week_high"] = (out["close"] > out["prev_week_high"]).astype(int)
    out["below_prev_week_low"] = (out["close"] < out["prev_week_low"]).astype(int)
    out.drop(columns=["_yw"], inplace=True)
    return out


# ---------------------------------------------------------------------------
# Round Number Proximity
# ---------------------------------------------------------------------------


def add_round_number_flag(
    df: pd.DataFrame,
    instrument: str,
) -> pd.DataFrame:
    """Flag candles within 0.3 ATR of a round number."""
    out = df.copy()
    if "atr_14" not in out.columns:
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    increments = {"XAU_USD": 50.0, "USOIL": 5.0}
    inc = increments.get(instrument)
    if inc is None:
        out["near_round_number"] = 0
        return out

    price = out["close"].astype(float)
    atr = out["atr_14"].astype(float)
    dist = (price % inc).clip(upper=inc / 2)
    dist = np.minimum(dist, inc - dist)
    out["near_round_number"] = (dist < 0.3 * atr).astype(int)
    return out
