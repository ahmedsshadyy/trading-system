"""
smc/fvg_fill.py

FVG fill tracking — tracks multiple FVGs per side.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_fvg_fill(
    df: pd.DataFrame,
    *,
    track_last_n_bull: int = 5,
    track_last_n_bear: int = 5,
    fill_measure: str = "extreme",
    invalidate_on_close_through: bool = False,
    remove_fully_filled: bool = True,
) -> pd.DataFrame:
    """Enhanced FVG fill tracking — tracks multiple FVGs per side.

    Backward-compatible columns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``fvg_active_bull / fvg_active_bear``
    * ``fvg_fill_pct``
    * ``fvg_age``

    Extra columns
    ~~~~~~~~~~~~~
    * ``fvg_fill_pct_bull / fvg_fill_pct_bear``
    * ``fvg_age_bull / fvg_age_bear``
    * ``fvg_touched_bull / fvg_touched_bear``
    * ``fvg_partial_bull / fvg_partial_bear``
    * ``fvg_full_bull / fvg_full_bear``
    """
    out = df.copy()
    n = len(out)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    fb = out.get("fvg_bull", pd.Series(np.zeros(n, dtype=np.int8))).to_numpy(
        dtype=np.int8
    )
    fr = out.get("fvg_bear", pd.Series(np.zeros(n, dtype=np.int8))).to_numpy(
        dtype=np.int8
    )
    fb_lo = out.get("fvg_bull_low", pd.Series(np.full(n, np.nan))).to_numpy(dtype=float)
    fb_hi = out.get("fvg_bull_high", pd.Series(np.full(n, np.nan))).to_numpy(
        dtype=float
    )
    fr_hi = out.get("fvg_bear_high", pd.Series(np.full(n, np.nan))).to_numpy(
        dtype=float
    )
    fr_lo = out.get("fvg_bear_low", pd.Series(np.full(n, np.nan))).to_numpy(dtype=float)
    bull_ci = out.get(
        "fvg_bull_confirm_idx", pd.Series(np.full(n, -1, dtype=int))
    ).to_numpy(dtype=int)
    bear_ci = out.get(
        "fvg_bear_confirm_idx", pd.Series(np.full(n, -1, dtype=int))
    ).to_numpy(dtype=int)

    active_bull = np.zeros(n, dtype=np.int8)
    active_bear = np.zeros(n, dtype=np.int8)
    fill_pct = np.full(n, np.nan)
    age_arr = np.full(n, np.nan)
    fill_pct_bull = np.full(n, np.nan)
    fill_pct_bear = np.full(n, np.nan)
    age_bull = np.full(n, np.nan)
    age_bear = np.full(n, np.nan)
    touched_bull = np.zeros(n, dtype=np.int8)
    touched_bear = np.zeros(n, dtype=np.int8)
    partial_bull = np.zeros(n, dtype=np.int8)
    partial_bear = np.zeros(n, dtype=np.int8)
    full_bull = np.zeros(n, dtype=np.int8)
    full_bear = np.zeros(n, dtype=np.int8)
    invalid_bull = np.zeros(n, dtype=np.int8)
    invalid_bear = np.zeros(n, dtype=np.int8)

    bull_zones = []
    bear_zones = []

    def _bull_fill(zone, i_):
        size = zone["hi"] - zone["lo"]
        if size <= 0:
            return 0.0
        if fill_measure == "close":
            pen = max(0.0, zone["hi"] - c[i_])
        elif fill_measure == "mid":
            pen = max(0.0, zone["hi"] - (h[i_] + lo[i_]) / 2.0)
        else:
            pen = max(0.0, zone["hi"] - lo[i_])
        return float(np.clip(pen / size, 0.0, 1.0))

    def _bear_fill(zone, i_):
        size = zone["hi"] - zone["lo"]
        if size <= 0:
            return 0.0
        if fill_measure == "close":
            pen = max(0.0, c[i_] - zone["lo"])
        elif fill_measure == "mid":
            pen = max(0.0, (h[i_] + lo[i_]) / 2.0 - zone["lo"])
        else:
            pen = max(0.0, h[i_] - zone["lo"])
        return float(np.clip(pen / size, 0.0, 1.0))

    for i in range(n):
        # Register new zones
        if fb[i] == 1 and np.isfinite(fb_lo[i]):
            ci = bull_ci[i] if bull_ci[i] >= 0 else i + 1
            bull_zones.append(
                {
                    "ci": ci,
                    "lo": fb_lo[i],
                    "hi": fb_hi[i],
                    "max_fill": 0.0,
                    "active": True,
                }
            )
            bull_zones = bull_zones[-track_last_n_bull:]

        if fr[i] == 1 and np.isfinite(fr_lo[i]):
            ci = bear_ci[i] if bear_ci[i] >= 0 else i + 1
            bear_zones.append(
                {
                    "ci": ci,
                    "lo": fr_lo[i],
                    "hi": fr_hi[i],
                    "max_fill": 0.0,
                    "active": True,
                }
            )
            bear_zones = bear_zones[-track_last_n_bear:]

        best_bf = np.nan
        best_ba = np.nan
        for z in bull_zones:
            if not z["active"] or i <= z["ci"]:
                continue
            fp = _bull_fill(z, i)
            z["max_fill"] = max(z["max_fill"], fp)
            if fp > 0:
                touched_bull[i] = 1
            if 0 < fp < 1:
                partial_bull[i] = 1
            if fp >= 1:
                full_bull[i] = 1
                if remove_fully_filled:
                    z["active"] = False
            if invalidate_on_close_through and c[i] < z["lo"]:
                invalid_bull[i] = 1
                z["active"] = False
            if z["active"]:
                active_bull[i] = 1
                if np.isnan(best_bf) or z["max_fill"] > best_bf:
                    best_bf = z["max_fill"]
                    best_ba = float(i - z["ci"])

        best_sf = np.nan
        best_sa = np.nan
        for z in bear_zones:
            if not z["active"] or i <= z["ci"]:
                continue
            fp = _bear_fill(z, i)
            z["max_fill"] = max(z["max_fill"], fp)
            if fp > 0:
                touched_bear[i] = 1
            if 0 < fp < 1:
                partial_bear[i] = 1
            if fp >= 1:
                full_bear[i] = 1
                if remove_fully_filled:
                    z["active"] = False
            if invalidate_on_close_through and c[i] > z["hi"]:
                invalid_bear[i] = 1
                z["active"] = False
            if z["active"]:
                active_bear[i] = 1
                if np.isnan(best_sf) or z["max_fill"] > best_sf:
                    best_sf = z["max_fill"]
                    best_sa = float(i - z["ci"])

        fill_pct_bull[i] = best_bf
        fill_pct_bear[i] = best_sf
        age_bull[i] = best_ba
        age_bear[i] = best_sa

        # Backward-compatible shared fields
        cands = []
        if np.isfinite(best_bf):
            cands.append((best_bf, best_ba))
        if np.isfinite(best_sf):
            cands.append((best_sf, best_sa))
        if cands:
            best = max(cands, key=lambda x: x[0])
            fill_pct[i] = best[0]
            age_arr[i] = best[1]

        # Prune dead zones
        bull_zones = [z for z in bull_zones if z["active"]]
        bear_zones = [z for z in bear_zones if z["active"]]

    out["fvg_active_bull"] = active_bull
    out["fvg_active_bear"] = active_bear
    out["fvg_fill_pct"] = fill_pct
    out["fvg_age"] = age_arr
    out["fvg_fill_pct_bull"] = fill_pct_bull
    out["fvg_fill_pct_bear"] = fill_pct_bear
    out["fvg_age_bull"] = age_bull
    out["fvg_age_bear"] = age_bear
    out["fvg_touched_bull"] = touched_bull
    out["fvg_touched_bear"] = touched_bear
    out["fvg_partial_bull"] = partial_bull
    out["fvg_partial_bear"] = partial_bear
    out["fvg_full_bull"] = full_bull
    out["fvg_full_bear"] = full_bear
    out["fvg_invalid_bull"] = invalid_bull
    out["fvg_invalid_bear"] = invalid_bear

    return out
