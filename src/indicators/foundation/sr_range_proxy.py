"""
foundation/sr_range_proxy.py

Causal range-like proxy built from live support/resistance geometry.

This layer is intentionally not a replacement for ``range_boundaries``.
It formalizes the narrower question:

"Do the currently active S/R bands behave enough like a bounded range that
downstream consumers can treat them as a range-style proxy?"

The proxy is fully causal and derived only from live-safe S/R projections plus
trailing stability / containment checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators.foundation.sr_levels import add_sr_levels

__all__ = ["add_sr_range_proxy"]

RANGE_PROXY_STABILITY_LOOKBACK = 6
RANGE_PROXY_CONTAINMENT_LOOKBACK = 8
RANGE_PROXY_MIN_BAND_WIDTH_ATR = 0.40
RANGE_PROXY_MAX_BAND_WIDTH_ATR = 4.50
RANGE_PROXY_MAX_OVERLAP_ATR = 0.20
RANGE_PROXY_MIN_QUALITY = 0.58


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))


def _band_width_score(width_atr: float) -> float:
    if not np.isfinite(width_atr) or width_atr <= 0:
        return np.nan
    if width_atr < RANGE_PROXY_MIN_BAND_WIDTH_ATR:
        return float(
            np.clip((width_atr / RANGE_PROXY_MIN_BAND_WIDTH_ATR) * 0.70, 0.0, 1.0)
        )
    if width_atr <= 2.80:
        return 1.0
    if width_atr <= RANGE_PROXY_MAX_BAND_WIDTH_ATR:
        fade = (width_atr - 2.80) / max(RANGE_PROXY_MAX_BAND_WIDTH_ATR - 2.80, 1e-9)
        return float(np.clip(1.0 - 0.70 * fade, 0.0, 1.0))
    return 0.0


def _gap_score(gap_atr: float) -> float:
    if not np.isfinite(gap_atr):
        return np.nan
    if gap_atr < -RANGE_PROXY_MAX_OVERLAP_ATR:
        return 0.0
    if gap_atr <= 0.10:
        return 1.0
    if gap_atr <= 0.75:
        fade = (gap_atr - 0.10) / 0.65
        return float(np.clip(1.0 - 0.55 * fade, 0.0, 1.0))
    return 0.35


def _id_persistence(ids: np.ndarray, current: float) -> float:
    if not np.isfinite(current):
        return np.nan
    valid = np.isfinite(ids)
    if not valid.any():
        return np.nan
    return float(np.mean(ids[valid] == current))


def add_sr_range_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Add a causal range-style proxy derived from current S/R geometry."""
    out = df.copy()
    if out.empty:
        return out

    required_sr_cols = {
        "primary_support_zone_low",
        "primary_support_zone_high",
        "primary_support_zone_mid",
        "primary_support_zone_id",
        "primary_support_zone_score",
        "primary_resistance_zone_low",
        "primary_resistance_zone_high",
        "primary_resistance_zone_mid",
        "primary_resistance_zone_id",
        "primary_resistance_zone_score",
        "active_support_count",
        "active_resistance_count",
        "support_cluster_density_atr",
        "resistance_cluster_density_atr",
    }
    if not required_sr_cols.issubset(out.columns):
        out = add_sr_levels(out, include_research_only=False)

    atr = get_atr_array(out)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)

    sup_low = pd.to_numeric(out["primary_support_zone_low"], errors="coerce").to_numpy(
        dtype=float
    )
    sup_high = pd.to_numeric(
        out["primary_support_zone_high"], errors="coerce"
    ).to_numpy(dtype=float)
    sup_mid = pd.to_numeric(out["primary_support_zone_mid"], errors="coerce").to_numpy(
        dtype=float
    )
    sup_id = pd.to_numeric(out["primary_support_zone_id"], errors="coerce").to_numpy(
        dtype=float
    )
    sup_score = pd.to_numeric(
        out["primary_support_zone_score"], errors="coerce"
    ).to_numpy(dtype=float)

    res_low = pd.to_numeric(
        out["primary_resistance_zone_low"], errors="coerce"
    ).to_numpy(dtype=float)
    res_high = pd.to_numeric(
        out["primary_resistance_zone_high"], errors="coerce"
    ).to_numpy(dtype=float)
    res_mid = pd.to_numeric(
        out["primary_resistance_zone_mid"], errors="coerce"
    ).to_numpy(dtype=float)
    res_id = pd.to_numeric(out["primary_resistance_zone_id"], errors="coerce").to_numpy(
        dtype=float
    )
    res_score = pd.to_numeric(
        out["primary_resistance_zone_score"], errors="coerce"
    ).to_numpy(dtype=float)

    support_count = (
        pd.to_numeric(out["active_support_count"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    resistance_count = (
        pd.to_numeric(out["active_resistance_count"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    support_density = pd.to_numeric(
        out["support_cluster_density_atr"], errors="coerce"
    ).to_numpy(dtype=float)
    resistance_density = pd.to_numeric(
        out["resistance_cluster_density_atr"], errors="coerce"
    ).to_numpy(dtype=float)

    n = len(out)
    active = np.zeros(n, dtype=np.int8)
    band_low = np.full(n, np.nan)
    band_high = np.full(n, np.nan)
    band_mid = np.full(n, np.nan)
    outer_low = np.full(n, np.nan)
    outer_high = np.full(n, np.nan)
    band_width_atr = np.full(n, np.nan)
    outer_width_atr = np.full(n, np.nan)
    gap_atr = np.full(n, np.nan)
    compression_score = np.full(n, np.nan)
    balance_score = np.full(n, np.nan)
    stability_score = np.full(n, np.nan)
    containment_score = np.full(n, np.nan)
    quality_score = np.full(n, np.nan)

    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else np.nan
        if not np.isfinite(atr_i):
            continue
        if not (
            np.isfinite(sup_mid[i])
            and np.isfinite(res_mid[i])
            and np.isfinite(sup_low[i])
            and np.isfinite(sup_high[i])
            and np.isfinite(res_low[i])
            and np.isfinite(res_high[i])
        ):
            continue
        if res_mid[i] <= sup_mid[i]:
            continue

        band_low[i] = sup_mid[i]
        band_high[i] = res_mid[i]
        band_mid[i] = (sup_mid[i] + res_mid[i]) / 2.0
        outer_low[i] = sup_low[i]
        outer_high[i] = res_high[i]
        band_width_atr[i] = (res_mid[i] - sup_mid[i]) / atr_i
        outer_width_atr[i] = (res_high[i] - sup_low[i]) / atr_i
        gap_atr[i] = (res_low[i] - sup_high[i]) / atr_i

        width_score = _band_width_score(band_width_atr[i])
        gap_score = _gap_score(gap_atr[i])
        mean_zone_width_atr = (
            (sup_high[i] - sup_low[i]) + (res_high[i] - res_low[i])
        ) / (2.0 * atr_i)
        width_clarity = _clip01(
            1.0 - (mean_zone_width_atr / max(band_width_atr[i], 1e-9))
        )
        compression_score[i] = _clip01(
            0.50 * width_score + 0.30 * gap_score + 0.20 * width_clarity
        )

        score_balance = _clip01(1.0 - abs(sup_score[i] - res_score[i]))
        count_balance = _clip01(
            1.0
            - abs(support_count[i] - resistance_count[i])
            / max(support_count[i] + resistance_count[i], 1.0)
        )
        density_values = [
            v for v in (support_density[i], resistance_density[i]) if np.isfinite(v)
        ]
        density_balance = (
            _clip01(1.0 - float(np.mean(density_values)) / 1.25)
            if density_values
            else 0.55
        )
        balance_score[i] = _clip01(
            0.40 * score_balance + 0.30 * count_balance + 0.30 * density_balance
        )

        lo = max(0, i - RANGE_PROXY_STABILITY_LOOKBACK + 1)
        sup_id_persist = _id_persistence(sup_id[lo : i + 1], sup_id[i])
        res_id_persist = _id_persistence(res_id[lo : i + 1], res_id[i])
        if np.isfinite(sup_id_persist) and np.isfinite(res_id_persist):
            id_stability = float((sup_id_persist + res_id_persist) / 2.0)
        elif np.isfinite(sup_id_persist):
            id_stability = sup_id_persist
        elif np.isfinite(res_id_persist):
            id_stability = res_id_persist
        else:
            id_stability = np.nan

        hist_band = band_width_atr[lo : i + 1]
        hist_band = hist_band[np.isfinite(hist_band)]
        if len(hist_band) >= 2 and float(np.nanmean(hist_band)) > 0:
            band_cv = float(np.nanstd(hist_band) / np.nanmean(hist_band))
            band_consistency = _clip01(1.0 - band_cv / 0.75)
        else:
            band_consistency = 0.55

        hist_sup = sup_mid[lo : i + 1]
        hist_res = res_mid[lo : i + 1]
        mid_drift_atr = 0.0
        if np.isfinite(hist_sup).sum() >= 2:
            mid_drift_atr += float(np.nanstd(hist_sup) / atr_i)
        if np.isfinite(hist_res).sum() >= 2:
            mid_drift_atr += float(np.nanstd(hist_res) / atr_i)
        drift_stability = _clip01(1.0 - (mid_drift_atr / 0.90))
        stability_score[i] = _clip01(
            0.45 * id_stability + 0.30 * band_consistency + 0.25 * drift_stability
        )

        contain_lo = max(0, i - RANGE_PROXY_CONTAINMENT_LOOKBACK + 1)
        trail_close = close[contain_lo : i + 1]
        valid_close = np.isfinite(trail_close)
        if valid_close.any():
            inside = (trail_close[valid_close] >= sup_low[i]) & (
                trail_close[valid_close] <= res_high[i]
            )
            containment_score[i] = float(np.mean(inside))

        mean_strength = float(np.nanmean([sup_score[i], res_score[i]]))
        quality_score[i] = _clip01(
            0.30 * compression_score[i]
            + 0.22 * balance_score[i]
            + 0.23 * stability_score[i]
            + 0.15 * containment_score[i]
            + 0.10 * mean_strength
        )

        if (
            quality_score[i] >= RANGE_PROXY_MIN_QUALITY
            and RANGE_PROXY_MIN_BAND_WIDTH_ATR
            <= band_width_atr[i]
            <= RANGE_PROXY_MAX_BAND_WIDTH_ATR
            and gap_atr[i] >= -RANGE_PROXY_MAX_OVERLAP_ATR
            and np.isfinite(close[i])
            and sup_low[i] <= close[i] <= res_high[i]
        ):
            active[i] = 1

    out["sr_range_proxy_active"] = active
    out["sr_range_proxy_low"] = band_low
    out["sr_range_proxy_high"] = band_high
    out["sr_range_proxy_mid"] = band_mid
    out["sr_range_proxy_outer_low"] = outer_low
    out["sr_range_proxy_outer_high"] = outer_high
    out["sr_range_proxy_support_zone_id"] = sup_id
    out["sr_range_proxy_resistance_zone_id"] = res_id
    out["sr_range_proxy_width_atr"] = band_width_atr
    out["sr_range_proxy_outer_width_atr"] = outer_width_atr
    out["sr_range_proxy_gap_atr"] = gap_atr
    out["sr_range_proxy_compression_score"] = compression_score
    out["sr_range_proxy_balance_score"] = balance_score
    out["sr_range_proxy_stability_score"] = stability_score
    out["sr_range_proxy_containment_score"] = containment_score
    out["sr_range_proxy_quality_score"] = quality_score
    return out
