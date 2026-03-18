"""
Value & Reference Level indicators.

Anchored VWAP + σ bands, Asian session high/low, previous day/week
high/low, round number proximity.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Anchored VWAP
# ---------------------------------------------------------------------------


def compute_anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.DataFrame:
    """Compute VWAP anchored to a specific bar index.

    Parameters
    ----------
    df : DataFrame
        Must have ``high, low, close, volume, timestamp`` columns.
    anchor_idx : int
        Row index (iloc position) to anchor the VWAP from.

    Returns
    -------
    DataFrame with columns added from ``anchor_idx`` onward:
        ``avwap``           – anchored VWAP
        ``avwap_upper_1``   – +1σ band
        ``avwap_lower_1``   – −1σ band
        ``avwap_upper_2``   – +2σ band
        ``avwap_lower_2``   – −2σ band
        ``avwap_dev_sigma`` – current deviation in σ units
    """
    out = df.copy()
    h = out["high"].values.astype(float)
    l = out["low"].values.astype(float)
    c = out["close"].values.astype(float)
    v = out["volume"].values.astype(float)
    n = len(out)

    tp = (h + l + c) / 3.0

    avwap = np.full(n, np.nan)
    upper1 = np.full(n, np.nan)
    lower1 = np.full(n, np.nan)
    upper2 = np.full(n, np.nan)
    lower2 = np.full(n, np.nan)
    dev_sigma = np.full(n, np.nan)

    cum_tpv = 0.0
    cum_v = 0.0
    cum_tp2v = 0.0  # for variance

    for i in range(anchor_idx, n):
        cum_tpv += tp[i] * v[i]
        cum_v += v[i]
        cum_tp2v += (tp[i] ** 2) * v[i]

        if cum_v > 0:
            vwap_val = cum_tpv / cum_v
            variance = (cum_tp2v / cum_v) - (vwap_val**2)
            std = np.sqrt(max(variance, 0))

            avwap[i] = vwap_val
            upper1[i] = vwap_val + std
            lower1[i] = vwap_val - std
            upper2[i] = vwap_val + 2 * std
            lower2[i] = vwap_val - 2 * std

            if std > 0:
                dev_sigma[i] = (c[i] - vwap_val) / std
            else:
                dev_sigma[i] = 0.0

    out["avwap"] = avwap
    out["avwap_upper_1"] = upper1
    out["avwap_lower_1"] = lower1
    out["avwap_upper_2"] = upper2
    out["avwap_lower_2"] = lower2
    out["avwap_dev_sigma"] = dev_sigma

    return out


def add_avwap_from_last_swing(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: anchor VWAP to the most recent significant swing.

    Requires ``swing_high`` and ``swing_low`` columns from ``add_swings()``.
    Picks the most recent swing (high or low) and anchors from there.

    For Strategy 9 the scanner will call ``compute_anchored_vwap()``
    directly with its own anchor logic.
    """
    out = df.copy()

    # Find most recent swing (either H or L)
    sh = out["swing_high"].values if "swing_high" in out.columns else np.zeros(len(out))
    sl = out["swing_low"].values if "swing_low" in out.columns else np.zeros(len(out))

    last_swing_idx = 0
    for i in range(len(out) - 1, -1, -1):
        if sh[i] == 1 or sl[i] == 1:
            last_swing_idx = i
            break

    return compute_anchored_vwap(out, last_swing_idx)


# ---------------------------------------------------------------------------
# Asian Session High / Low
# ---------------------------------------------------------------------------


def add_asian_session_hl(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Asian session (00:00–08:00 UTC) high and low.

    Works on intraday timeframes (H4, H1, M15).  For each trading day,
    computes the high and low of candles within the Asian window, then
    forward-fills for the rest of the day.

    Columns
    ~~~~~~~
    * ``asian_high``          – Asian session high (forward-filled)
    * ``asian_low``           – Asian session low (forward-filled)
    * ``asian_range``         – high − low
    * ``asian_range_ratio``   – current range / 10-session avg range
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    hour = ts.dt.hour
    date = ts.dt.date

    # Mark Asian candles
    is_asian = hour < 8  # 00:00–07:59 UTC
    out["_date"] = date

    # Group by date and compute Asian H/L
    asian_hl = (
        out[is_asian]
        .groupby("_date")
        .agg(asian_high=("high", "max"), asian_low=("low", "min"))
    )

    out = out.merge(asian_hl, left_on="_date", right_index=True, how="left")

    # Forward-fill within each day
    out["asian_high"] = out.groupby("_date")["asian_high"].ffill()
    out["asian_low"] = out.groupby("_date")["asian_low"].ffill()
    out["asian_range"] = out["asian_high"] - out["asian_low"]

    # 10-session average range
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
    """Previous day high/low and position flags.

    Columns
    ~~~~~~~
    * ``prev_day_high``     – previous day's high
    * ``prev_day_low``      – previous day's low
    * ``above_prev_day_high`` – binary
    * ``below_prev_day_low``  – binary
    """
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
    """Previous week high/low.

    Columns: ``prev_week_high``, ``prev_week_low``,
    ``above_prev_week_high``, ``below_prev_week_low``.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    # ISO week number + year for grouping
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
    """Flag candles within 0.3 ATR of a round number.

    Round numbers: XAU_USD every $50, USOIL every $5.

    Requires ``atr_14`` column.

    Columns: ``near_round_number`` (binary).
    """
    out = df.copy()
    if "atr_14" not in out.columns:
        from src.indicators import ta_core as ta

        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    increments = {"XAU_USD": 50.0, "USOIL": 5.0}
    inc = increments.get(instrument)
    if inc is None:
        out["near_round_number"] = 0
        return out

    price = out["close"].astype(float)
    atr = out["atr_14"].astype(float)
    dist = (price % inc).clip(upper=inc / 2)
    dist = np.minimum(dist, inc - dist)  # distance to nearest round number
    out["near_round_number"] = (dist < 0.3 * atr).astype(int)
    return out
