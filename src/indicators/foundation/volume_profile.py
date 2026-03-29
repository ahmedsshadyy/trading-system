"""
Volume Profile calculator.

POC (Point of Control), VAH (Value Area High), and VAL (Value Area Low)
computed from canonical internal ``volume`` after schema normalization.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import ta_core as ta
from src.indicators._helpers.schema import normalize_candle_schema

CANONICAL_VP_MODE = "exact"


def _ensure_atr(df: pd.DataFrame, atr_col: str) -> pd.DataFrame:
    out = df.copy()
    if atr_col not in out.columns:
        out[atr_col] = ta.atr(out["high"], out["low"], out["close"], length=14)
    return out


def compute_volume_profile(
    df: pd.DataFrame,
    lookback: int = 80,
    n_bins: int = 50,
) -> dict:
    """Compute VP levels over the last ``lookback`` rows of ``df``."""
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")

    segment = normalize_candle_schema(df, require_volume=True).tail(lookback).copy()
    if segment.empty:
        return {
            "poc": np.nan,
            "vah": np.nan,
            "val": np.nan,
            "profile": np.zeros(n_bins, dtype=float),
            "bin_edges": np.array([], dtype=float),
        }

    h = pd.to_numeric(segment["high"], errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(segment["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(segment["close"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(segment["volume"], errors="coerce").to_numpy(dtype=float)
    tp = (h + l + c) / 3.0

    price_min = float(np.nanmin(l))
    price_max = float(np.nanmax(h))
    if not np.isfinite(price_min) or not np.isfinite(price_max):
        return {
            "poc": np.nan,
            "vah": np.nan,
            "val": np.nan,
            "profile": np.zeros(n_bins, dtype=float),
            "bin_edges": np.array([], dtype=float),
        }
    if price_max == price_min:
        return {
            "poc": price_min,
            "vah": price_max,
            "val": price_min,
            "profile": np.zeros(n_bins, dtype=float),
            "bin_edges": np.array([], dtype=float),
        }

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=float)

    bin_indices = np.digitize(tp, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    for idx, vol in zip(bin_indices, v, strict=False):
        if np.isfinite(vol):
            profile[int(idx)] += float(vol)

    poc_bin = int(np.argmax(profile))
    poc = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0)

    total_vol = float(profile.sum())
    target = total_vol * 0.7

    va_low_bin = poc_bin
    va_high_bin = poc_bin
    va_vol = float(profile[poc_bin])

    while va_vol < target and (va_low_bin > 0 or va_high_bin < n_bins - 1):
        vol_below = float(profile[va_low_bin - 1]) if va_low_bin > 0 else -np.inf
        vol_above = (
            float(profile[va_high_bin + 1]) if va_high_bin < n_bins - 1 else -np.inf
        )
        if vol_below >= vol_above and va_low_bin > 0:
            va_low_bin -= 1
            va_vol += float(profile[va_low_bin])
        elif va_high_bin < n_bins - 1:
            va_high_bin += 1
            va_vol += float(profile[va_high_bin])
        else:
            break

    vah = float(bin_edges[va_high_bin + 1])
    val = float(bin_edges[va_low_bin])
    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "profile": profile,
        "bin_edges": bin_edges,
    }


def add_volume_profile(
    df: pd.DataFrame,
    lookback: int = 80,
    n_bins: int = 50,
    *,
    atr_col: str = "atr_14",
    mode: str = "exact",
) -> pd.DataFrame:
    """Add causal rolling VP levels and context columns.

    ``vp_*[t]`` uses the trailing ``lookback`` rows ending at ``t-1``.
    The current bar is excluded by contract.

    Canonical doctrine:
    - ``mode="exact"`` is the frozen canonical mode.
    - ``mode="stepped"`` exists only as an approximation / audit mode and
      is not parity-equivalent to exact.
    """
    if mode not in {"exact", "stepped"}:
        raise ValueError("mode must be 'exact' or 'stepped'")

    out = normalize_candle_schema(df, require_volume=True)
    out = _ensure_atr(out, atr_col)
    n = len(out)

    poc = np.full(n, np.nan, dtype=float)
    vah = np.full(n, np.nan, dtype=float)
    val = np.full(n, np.nan, dtype=float)

    if mode == CANONICAL_VP_MODE:
        for i in range(lookback, n):
            vp = compute_volume_profile(
                out.iloc[i - lookback : i], lookback=lookback, n_bins=n_bins
            )
            poc[i] = vp["poc"]
            vah[i] = vp["vah"]
            val[i] = vp["val"]
    else:
        step = max(lookback // 4, 1)
        for i in range(lookback, n, step):
            vp = compute_volume_profile(
                out.iloc[i - lookback : i], lookback=lookback, n_bins=n_bins
            )
            end_idx = min(i + step, n)
            poc[i:end_idx] = vp["poc"]
            vah[i:end_idx] = vp["vah"]
            val[i:end_idx] = vp["val"]

    out["vp_poc"] = poc
    out["vp_vah"] = vah
    out["vp_val"] = val

    atr = pd.to_numeric(out[atr_col], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    out["vp_poc_distance_atr"] = np.where(
        atr > 0,
        np.abs(close - out["vp_poc"]) / atr,
        np.nan,
    )
    out["vp_value_width"] = out["vp_vah"] - out["vp_val"]
    out["vp_value_width_atr"] = np.where(
        atr > 0,
        out["vp_value_width"] / atr,
        np.nan,
    )
    vp_ready = out["vp_val"].notna() & out["vp_vah"].notna()
    out["vp_inside_value_area"] = (
        vp_ready & close.ge(out["vp_val"]) & close.le(out["vp_vah"])
    ).astype("int8")
    out["vp_above_vah"] = (vp_ready & close.gt(out["vp_vah"])).astype("int8")
    out["vp_below_val"] = (vp_ready & close.lt(out["vp_val"])).astype("int8")
    out["vp_distance_to_vah_atr"] = np.where(
        atr > 0,
        (close - out["vp_vah"]) / atr,
        np.nan,
    )
    out["vp_distance_to_val_atr"] = np.where(
        atr > 0,
        (close - out["vp_val"]) / atr,
        np.nan,
    )
    return out
