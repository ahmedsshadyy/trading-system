"""
smc/fvg.py

Enhanced Fair Value Gap (FVG) detector.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.pivots import last_confirmed_swing_levels
from src.indicators._helpers.zones import (
    zones_overlap,
    merge_zone_bounds,
    body_high,
    body_low,
)


def add_fvg(
    df: pd.DataFrame,
    min_atr_mult: float = 0.3,
    *,
    atr_length: int = 14,
    mode: str = "wick",
    require_displacement: bool = True,
    displacement_body_atr_mult: float = 0.8,
    require_bos: bool = False,
    swing_left: int = 2,
    swing_right: int = 2,
    merge_overlaps: bool = True,
    merge_max_gap_bars: int = 2,
    keep_strongest_only_within: int = 0,
) -> pd.DataFrame:
    """Enhanced Fair Value Gap detector.

    Columns (backward-compatible)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``fvg_bull / fvg_bear``             – binary flags
    * ``fvg_bull_low / fvg_bull_high``    – zone boundaries
    * ``fvg_bear_low / fvg_bear_high``    – zone boundaries
    * ``fvg_size_atr``                    – gap size / ATR

    Extra columns
    ~~~~~~~~~~~~~
    * ``fvg_bull_confirm_idx / fvg_bear_confirm_idx``
    * ``fvg_confirm_delay``
    * ``fvg_mid_body_atr``  – middle candle body / ATR
    * ``fvg_bos_bull / fvg_bos_bear``
    """
    out = df.copy()
    n = len(out)
    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)
    bh = body_high(o, c)
    bl = body_low(o, c)
    mid_body = np.abs(c - o)

    last_ph, last_pl = last_confirmed_swing_levels(
        h, lo, left=swing_left, right=swing_right
    )

    # Collect candidates
    candidates_bull = []
    candidates_bear = []

    def _gap(i_):
        if mode == "body":
            return (
                bl[i_ + 1] - bh[i_ - 1],
                bl[i_ - 1] - bh[i_ + 1],
                bh[i_ - 1],
                bl[i_ + 1],
                bl[i_ - 1],
                bh[i_ + 1],
            )
        elif mode == "hybrid":
            return (
                bl[i_ + 1] - h[i_ - 1],
                lo[i_ - 1] - bh[i_ + 1],
                h[i_ - 1],
                bl[i_ + 1],
                lo[i_ - 1],
                bh[i_ + 1],
            )
        else:  # wick (default)
            return (
                lo[i_ + 1] - h[i_ - 1],
                lo[i_ - 1] - h[i_ + 1],
                h[i_ - 1],
                lo[i_ + 1],
                lo[i_ - 1],
                h[i_ + 1],
            )

    for i in range(1, n - 1):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        gap_bull, gap_bear, bull_lo, bull_hi, bear_hi, bear_lo = _gap(i)
        body_atr = mid_body[i] / atr_i

        bull_bos = np.isfinite(last_ph[i + 1]) and c[i + 1] > last_ph[i + 1]
        bear_bos = np.isfinite(last_pl[i + 1]) and c[i + 1] < last_pl[i + 1]

        # Bullish FVG
        bull_ok = gap_bull > 0 and gap_bull >= min_atr_mult * atr_i
        if require_displacement:
            bull_ok = bull_ok and body_atr >= displacement_body_atr_mult
        if require_bos:
            bull_ok = bull_ok and bull_bos
        if bull_ok:
            candidates_bull.append(
                {
                    "idx": i,
                    "confirm_idx": i + 1,
                    "lo": bull_lo,
                    "hi": bull_hi,
                    "size_atr": gap_bull / atr_i,
                    "body_atr": body_atr,
                    "bos": int(bull_bos),
                }
            )

        # Bearish FVG
        bear_ok = gap_bear > 0 and gap_bear >= min_atr_mult * atr_i
        if require_displacement:
            bear_ok = bear_ok and body_atr >= displacement_body_atr_mult
        if require_bos:
            bear_ok = bear_ok and bear_bos
        if bear_ok:
            candidates_bear.append(
                {
                    "idx": i,
                    "confirm_idx": i + 1,
                    "lo": bear_lo,
                    "hi": bear_hi,
                    "size_atr": gap_bear / atr_i,
                    "body_atr": body_atr,
                    "bos": int(bear_bos),
                }
            )

    # Prune strongest in local window
    def _prune(cands):
        if keep_strongest_only_within <= 0 or not cands:
            return cands
        kept = {}
        for cand in cands:
            t = cand["idx"]
            local = [
                x for x in cands if abs(x["idx"] - t) <= keep_strongest_only_within
            ]
            best = max(local, key=lambda z: z["size_atr"])
            if cand["idx"] == best["idx"]:
                kept[cand["idx"]] = cand
        return [kept[k] for k in sorted(kept)]

    candidates_bull = _prune(candidates_bull)
    candidates_bear = _prune(candidates_bear)

    # Merge overlapping zones
    def _merge(cands):
        if not merge_overlaps or not cands:
            return cands
        cands = sorted(cands, key=lambda z: z["idx"])
        merged = [cands[0].copy()]
        for cand in cands[1:]:
            prev = merged[-1]
            if cand["idx"] - prev["idx"] <= merge_max_gap_bars and zones_overlap(
                prev["lo"], prev["hi"], cand["lo"], cand["hi"]
            ):
                prev["lo"], prev["hi"] = merge_zone_bounds(
                    prev["lo"], prev["hi"], cand["lo"], cand["hi"]
                )
                prev["size_atr"] = max(prev["size_atr"], cand["size_atr"])
                prev["body_atr"] = max(prev["body_atr"], cand["body_atr"])
                prev["confirm_idx"] = max(prev["confirm_idx"], cand["confirm_idx"])
                prev["bos"] = max(prev["bos"], cand["bos"])
            else:
                merged.append(cand.copy())
        return merged

    candidates_bull = _merge(candidates_bull)
    candidates_bear = _merge(candidates_bear)

    # Stamp output arrays
    fb = np.zeros(n, dtype=np.int8)
    fr = np.zeros(n, dtype=np.int8)
    fb_lo = np.full(n, np.nan)
    fb_hi = np.full(n, np.nan)
    fr_hi = np.full(n, np.nan)
    fr_lo = np.full(n, np.nan)
    fvg_size = np.full(n, np.nan)
    bull_ci = np.full(n, -1, dtype=int)
    bear_ci = np.full(n, -1, dtype=int)
    fvg_delay = np.full(n, np.nan)
    fvg_mba = np.full(n, np.nan)
    fvg_bos_b = np.zeros(n, dtype=np.int8)
    fvg_bos_r = np.zeros(n, dtype=np.int8)

    for z in candidates_bull:
        i = z["idx"]
        fb[i] = 1
        fb_lo[i] = z["lo"]
        fb_hi[i] = z["hi"]
        fvg_size[i] = z["size_atr"]
        bull_ci[i] = z["confirm_idx"]
        fvg_delay[i] = z["confirm_idx"] - i
        fvg_mba[i] = z["body_atr"]
        fvg_bos_b[i] = z["bos"]

    for z in candidates_bear:
        i = z["idx"]
        fr[i] = 1
        fr_hi[i] = z["hi"]
        fr_lo[i] = z["lo"]
        if np.isnan(fvg_size[i]):
            fvg_size[i] = z["size_atr"]
        bear_ci[i] = z["confirm_idx"]
        if np.isnan(fvg_delay[i]):
            fvg_delay[i] = z["confirm_idx"] - i
        if np.isnan(fvg_mba[i]):
            fvg_mba[i] = z["body_atr"]
        fvg_bos_r[i] = z["bos"]

    out["fvg_bull"] = fb
    out["fvg_bear"] = fr
    out["fvg_bull_low"] = fb_lo
    out["fvg_bull_high"] = fb_hi
    out["fvg_bear_high"] = fr_hi
    out["fvg_bear_low"] = fr_lo
    out["fvg_size_atr"] = fvg_size
    out["fvg_bull_confirm_idx"] = bull_ci
    out["fvg_bear_confirm_idx"] = bear_ci
    out["fvg_confirm_delay"] = fvg_delay
    out["fvg_mid_body_atr"] = fvg_mba
    out["fvg_bos_bull"] = fvg_bos_b
    out["fvg_bos_bear"] = fvg_bos_r

    return out
