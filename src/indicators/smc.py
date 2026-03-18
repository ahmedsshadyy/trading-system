"""
SMC (Smart Money Concepts) structure detectors.

FVG detector (enhanced), IFVG classifier (state-machine), FVG fill tracker,
Order Block detector (enhanced), OB mitigation tracker, liquidity sweep
detector, equal highs/lows detector, displacement candle detector,
AMD phase classifier.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators.ta_core import ensure_atr as _ensure_atr

# ============================================================================
# Shared Internal Helpers (structure-specific, not in ta_core)
# ============================================================================


def _pivot_high(high: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    n = len(high)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(high[i] > high[i - left : i]) and np.all(
            high[i] >= high[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def _pivot_low(low: np.ndarray, left: int = 2, right: int = 2) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=np.int8)
    for i in range(left, n - right):
        if np.all(low[i] < low[i - left : i]) and np.all(
            low[i] <= low[i + 1 : i + right + 1]
        ):
            out[i] = 1
    return out


def _last_confirmed_swing_levels(
    high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2
):
    """For each bar i, most recent confirmed pivot (knowable by bar i)."""
    n = len(high)
    ph = _pivot_high(high, left, right)
    pl = _pivot_low(low, left, right)
    last_ph = np.full(n, np.nan)
    last_pl = np.full(n, np.nan)
    cur_ph = np.nan
    cur_pl = np.nan
    for i in range(n):
        j = i - right
        if j >= 0:
            if ph[j] == 1:
                cur_ph = high[j]
            if pl[j] == 1:
                cur_pl = low[j]
        last_ph[i] = cur_ph
        last_pl[i] = cur_pl
    return last_ph, last_pl


def _body_high(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return np.maximum(open_, close)


def _body_low(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    return np.minimum(open_, close)


def _zones_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


def _merge_zone_bounds(a_lo: float, a_hi: float, b_lo: float, b_hi: float):
    return min(a_lo, b_lo), max(a_hi, b_hi)


# ============================================================================
# FVG Detector (Enhanced)
# ============================================================================


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

    atr = _ensure_atr(out, atr_length)
    bh = _body_high(o, c)
    bl = _body_low(o, c)
    mid_body = np.abs(c - o)

    last_ph, last_pl = _last_confirmed_swing_levels(
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
            if cand["idx"] - prev["idx"] <= merge_max_gap_bars and _zones_overlap(
                prev["lo"], prev["hi"], cand["lo"], cand["hi"]
            ):
                prev["lo"], prev["hi"] = _merge_zone_bounds(
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


# ============================================================================
# FVG Fill Tracker (Enhanced)
# ============================================================================


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


# ============================================================================
# IFVG Classifier (State Machine)
# ============================================================================


def add_ifvg(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    require_retest: bool = True,
    retest_mode: str = "any",
    track_last_n_bull: int = 5,
    track_last_n_bear: int = 5,
    remove_confirmed: bool = False,
) -> pd.DataFrame:
    """Enhanced IFVG classifier with state-machine logic.

    States: active_fvg(1) → broken/candidate(3) → confirmed(4)
    Note: state 2 is unused — break goes directly to candidate(3).

    Backward-compatible columns
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``ifvg_bull / ifvg_bear``    – 1 on confirmation
    * ``ifvg_width_atr``           – zone width / ATR

    Extra columns
    ~~~~~~~~~~~~~
    * ``ifvg_bull_break / ifvg_bear_break``   – 1 when FVG is broken through
    * ``ifvg_bull_retest / ifvg_bear_retest`` – 1 when IFVG retest confirmed
    * ``ifvg_state_bull / ifvg_state_bear``   – current highest state (0/1/3/4)
    * ``ifvg_bull_candidate / ifvg_bear_candidate`` – 1 on break (candidate stage)
    * ``ifvg_origin_idx_bull / ifvg_origin_idx_bear`` – origin FVG index
    """
    out = df.copy()
    n = len(out)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    atr = _ensure_atr(out, atr_length)

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

    ifvg_bull = np.zeros(n, dtype=np.int8)
    ifvg_bear = np.zeros(n, dtype=np.int8)
    ifvg_width = np.full(n, np.nan)
    ifvg_bull_break = np.zeros(n, dtype=np.int8)
    ifvg_bear_break = np.zeros(n, dtype=np.int8)
    ifvg_bull_candidate = np.zeros(n, dtype=np.int8)
    ifvg_bear_candidate = np.zeros(n, dtype=np.int8)
    ifvg_bull_retest = np.zeros(n, dtype=np.int8)
    ifvg_bear_retest = np.zeros(n, dtype=np.int8)
    ifvg_state_bull = np.zeros(n, dtype=np.int8)
    ifvg_state_bear = np.zeros(n, dtype=np.int8)
    ifvg_origin_idx_bull = np.full(n, -1, dtype=int)
    ifvg_origin_idx_bear = np.full(n, -1, dtype=int)

    active_bull_fvgs = []  # can invert to bearish IFVG
    active_bear_fvgs = []  # can invert to bullish IFVG

    def _retest_bear_ifvg(zone, i_):
        """Bullish FVG broken below → retest from below."""
        if retest_mode == "deep":
            return h[i_] >= zone["hi"]
        if retest_mode == "mid":
            return h[i_] >= (zone["lo"] + zone["hi"]) / 2.0
        return h[i_] >= zone["lo"]

    def _retest_bull_ifvg(zone, i_):
        """Bearish FVG broken above → retest from above."""
        if retest_mode == "deep":
            return lo[i_] <= zone["lo"]
        if retest_mode == "mid":
            return lo[i_] <= (zone["lo"] + zone["hi"]) / 2.0
        return lo[i_] <= zone["hi"]

    for i in range(n):
        # Register new FVGs
        if fb[i] == 1 and np.isfinite(fb_lo[i]):
            ci = bull_ci[i] if bull_ci[i] >= 0 else i + 1
            active_bull_fvgs.append(
                {
                    "lo": fb_lo[i],
                    "hi": fb_hi[i],
                    "ci": ci,
                    "state": 1,
                    "broken_idx": -1,
                    "confirmed": False,
                }
            )
            active_bull_fvgs = active_bull_fvgs[-track_last_n_bull:]

        if fr[i] == 1 and np.isfinite(fr_lo[i]):
            ci = bear_ci[i] if bear_ci[i] >= 0 else i + 1
            active_bear_fvgs.append(
                {
                    "lo": fr_lo[i],
                    "hi": fr_hi[i],
                    "ci": ci,
                    "state": 1,
                    "broken_idx": -1,
                    "confirmed": False,
                }
            )
            active_bear_fvgs = active_bear_fvgs[-track_last_n_bear:]

        # Bullish FVGs → can become bearish IFVG
        for z in active_bull_fvgs:
            if z["confirmed"] and remove_confirmed:
                continue
            if i <= z["ci"]:
                continue

            if z["state"] == 1 and c[i] < z["lo"]:
                z["state"] = 3 if require_retest else 4
                z["broken_idx"] = i
                ifvg_bear_break[i] = 1
                ifvg_bear_candidate[i] = 1
                if not require_retest:
                    ifvg_bear[i] = 1
                    ifvg_width[i] = (
                        (z["hi"] - z["lo"]) / atr[i] if atr[i] > 0 else np.nan
                    )
                    z["confirmed"] = True

            elif z["state"] == 3 and i > z["broken_idx"]:
                if _retest_bear_ifvg(z, i):
                    z["state"] = 4
                    z["confirmed"] = True
                    ifvg_bear_retest[i] = 1
                    ifvg_bear[i] = 1
                    ifvg_width[i] = (
                        (z["hi"] - z["lo"]) / atr[i] if atr[i] > 0 else np.nan
                    )

        # Bearish FVGs → can become bullish IFVG
        for z in active_bear_fvgs:
            if z["confirmed"] and remove_confirmed:
                continue
            if i <= z["ci"]:
                continue

            if z["state"] == 1 and c[i] > z["hi"]:
                z["state"] = 3 if require_retest else 4
                z["broken_idx"] = i
                ifvg_bull_break[i] = 1
                ifvg_bull_candidate[i] = 1
                if not require_retest:
                    ifvg_bull[i] = 1
                    ifvg_width[i] = (
                        (z["hi"] - z["lo"]) / atr[i] if atr[i] > 0 else np.nan
                    )
                    z["confirmed"] = True

            elif z["state"] == 3 and i > z["broken_idx"]:
                if _retest_bull_ifvg(z, i):
                    z["state"] = 4
                    z["confirmed"] = True
                    ifvg_bull_retest[i] = 1
                    ifvg_bull[i] = 1
                    ifvg_width[i] = (
                        (z["hi"] - z["lo"]) / atr[i] if atr[i] > 0 else np.nan
                    )

        # Stamp current best state per side + origin index
        bull_states = [z for z in active_bear_fvgs if z["state"] > 0]
        bear_states = [z for z in active_bull_fvgs if z["state"] > 0]
        if bull_states:
            best = max(bull_states, key=lambda z: (z["state"], z["ci"]))
            ifvg_state_bull[i] = best["state"]
            ifvg_origin_idx_bull[i] = best["ci"]
        if bear_states:
            best = max(bear_states, key=lambda z: (z["state"], z["ci"]))
            ifvg_state_bear[i] = best["state"]
            ifvg_origin_idx_bear[i] = best["ci"]

    out["ifvg_bull"] = ifvg_bull
    out["ifvg_bear"] = ifvg_bear
    out["ifvg_width_atr"] = ifvg_width
    out["ifvg_bull_break"] = ifvg_bull_break
    out["ifvg_bear_break"] = ifvg_bear_break
    out["ifvg_bull_candidate"] = ifvg_bull_candidate
    out["ifvg_bear_candidate"] = ifvg_bear_candidate
    out["ifvg_bull_retest"] = ifvg_bull_retest
    out["ifvg_bear_retest"] = ifvg_bear_retest
    out["ifvg_state_bull"] = ifvg_state_bull
    out["ifvg_state_bear"] = ifvg_state_bear
    out["ifvg_origin_idx_bull"] = ifvg_origin_idx_bull
    out["ifvg_origin_idx_bear"] = ifvg_origin_idx_bear

    return out


# ============================================================================
# Order Block Detector (Enhanced)
# ============================================================================


def _ob_zone_from_candle(
    open_: float,
    high: float,
    low: float,
    close: float,
    side: str,
    mode: str = "openwick",
):
    """Compute OB zone boundaries.

    mode='openwick' (default): bull OB=[low, max(open,close)], bear OB=[min(open,close), high]
    mode='full': full wick range
    mode='body': body only
    """
    if mode == "body":
        return max(open_, close), min(open_, close)
    if mode == "openwick":
        if side == "bull":
            return max(open_, close), low
        return high, min(open_, close)
    return high, low


def add_ob(
    df: pd.DataFrame,
    impulse_candles: int = 4,
    impulse_atr_mult: float = 1.5,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    searchback: int = 10,
    min_same_dir_frac: float = 0.67,
    require_bos: bool = True,
    zone_mode: str = "openwick",
    max_ob_width_atr: float = 3.0,
) -> pd.DataFrame:
    """Enhanced Order Block detector.

    Columns: ``ob_bull``, ``ob_bear``, ``ob_bull_high/low``,
    ``ob_bear_high/low``, ``ob_width_atr``.
    """
    out = df.copy()
    n = len(out)
    o = out["open"].values.astype(float)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    atr = _ensure_atr(out, atr_length)
    last_ph, last_pl = _last_confirmed_swing_levels(h, lo, swing_left, swing_right)

    ob_bull = np.zeros(n, dtype=np.int8)
    ob_bear = np.zeros(n, dtype=np.int8)
    ob_bull_h = np.full(n, np.nan)
    ob_bull_l = np.full(n, np.nan)
    ob_bear_h = np.full(n, np.nan)
    ob_bear_l = np.full(n, np.nan)
    ob_w = np.full(n, np.nan)

    for i in range(max(impulse_candles - 1, 1), n):
        start = i - impulse_candles + 1
        if start < 0:
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        o_win = o[start : i + 1]
        c_win = c[start : i + 1]
        h_win = h[start : i + 1]
        l_win = lo[start : i + 1]

        bull_frac = np.sum(c_win > o_win) / impulse_candles
        bear_frac = np.sum(c_win < o_win) / impulse_candles

        bullish_impulse = (
            bull_frac >= min_same_dir_frac
            and c[i] - np.min(l_win) >= impulse_atr_mult * atr_i
            and np.sum(np.maximum(c_win - o_win, 0.0)) >= 0.8 * atr_i
            and c[i] > c[start]
        )
        bearish_impulse = (
            bear_frac >= min_same_dir_frac
            and np.max(h_win) - c[i] >= impulse_atr_mult * atr_i
            and np.sum(np.maximum(o_win - c_win, 0.0)) >= 0.8 * atr_i
            and c[i] < c[start]
        )

        if require_bos:
            bull_bos = (
                np.isfinite(last_ph[i]) and h[i] > last_ph[i] and c[i] > last_ph[i]
            )
            bear_bos = (
                np.isfinite(last_pl[i]) and lo[i] < last_pl[i] and c[i] < last_pl[i]
            )
        else:
            bull_bos = bullish_impulse
            bear_bos = bearish_impulse

        if bullish_impulse and bull_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] < o[k]:
                    z_hi, z_lo = _ob_zone_from_candle(
                        o[k], h[k], lo[k], c[k], "bull", zone_mode
                    )
                    w = (z_hi - z_lo) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(w) and w <= max_ob_width_atr:
                        ob_bull[k] = 1
                        ob_bull_h[k] = z_hi
                        ob_bull_l[k] = z_lo
                        if np.isnan(ob_w[k]):
                            ob_w[k] = w
                        break

        if bearish_impulse and bear_bos:
            for k in range(start - 1, max(-1, start - searchback - 1), -1):
                if c[k] > o[k]:
                    z_hi, z_lo = _ob_zone_from_candle(
                        o[k], h[k], lo[k], c[k], "bear", zone_mode
                    )
                    w = (z_hi - z_lo) / atr_i if atr_i > 0 else np.nan
                    if np.isfinite(w) and w <= max_ob_width_atr:
                        ob_bear[k] = 1
                        ob_bear_h[k] = z_hi
                        ob_bear_l[k] = z_lo
                        if np.isnan(ob_w[k]):
                            ob_w[k] = w
                        break

    out["ob_bull"] = ob_bull
    out["ob_bear"] = ob_bear
    out["ob_bull_high"] = ob_bull_h
    out["ob_bull_low"] = ob_bull_l
    out["ob_bear_high"] = ob_bear_h
    out["ob_bear_low"] = ob_bear_l
    out["ob_width_atr"] = ob_w
    return out


# ============================================================================
# OB Mitigation Tracker (Enhanced — tracks multiple OBs)
# ============================================================================


def add_ob_mitigation(df: pd.DataFrame, *, keep_last_n: int = 3) -> pd.DataFrame:
    """Track whether recent OBs are still unmitigated.

    Columns: ``ob_unmitigated_bull``, ``ob_unmitigated_bear``, ``ob_first_retest``.
    """
    out = df.copy()
    n = len(out)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)

    ob_b = out.get("ob_bull", pd.Series(np.zeros(n))).values.astype(np.int8)
    ob_r = out.get("ob_bear", pd.Series(np.zeros(n))).values.astype(np.int8)
    ob_bh = out.get("ob_bull_high", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_bl = out.get("ob_bull_low", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_rh = out.get("ob_bear_high", pd.Series(np.full(n, np.nan))).values.astype(float)
    ob_rl = out.get("ob_bear_low", pd.Series(np.full(n, np.nan))).values.astype(float)

    unmit_bull = np.zeros(n, dtype=np.int8)
    unmit_bear = np.zeros(n, dtype=np.int8)
    first_retest = np.zeros(n, dtype=np.int8)

    bull_zones = []
    bear_zones = []

    for i in range(n):
        if ob_b[i] == 1 and np.isfinite(ob_bh[i]):
            bull_zones.append(
                {"idx": i, "hi": ob_bh[i], "lo": ob_bl[i], "retested": False}
            )
            bull_zones = bull_zones[-keep_last_n:]

        if ob_r[i] == 1 and np.isfinite(ob_rh[i]):
            bear_zones.append(
                {"idx": i, "hi": ob_rh[i], "lo": ob_rl[i], "retested": False}
            )
            bear_zones = bear_zones[-keep_last_n:]

        any_bull = False
        for z in bull_zones:
            if i <= z["idx"]:
                any_bull = True
                continue
            if lo[i] <= z["hi"]:
                if not z["retested"]:
                    first_retest[i] = 1
                    z["retested"] = True
                if lo[i] < z["lo"]:
                    z["idx"] = -999
                else:
                    any_bull = True
            else:
                any_bull = True

        any_bear = False
        for z in bear_zones:
            if i <= z["idx"]:
                any_bear = True
                continue
            if h[i] >= z["lo"]:
                if not z["retested"]:
                    first_retest[i] = 1
                    z["retested"] = True
                if h[i] > z["hi"]:
                    z["idx"] = -999
                else:
                    any_bear = True
            else:
                any_bear = True

        bull_zones = [z for z in bull_zones if z["idx"] != -999]
        bear_zones = [z for z in bear_zones if z["idx"] != -999]
        unmit_bull[i] = 1 if any_bull else 0
        unmit_bear[i] = 1 if any_bear else 0

    out["ob_unmitigated_bull"] = unmit_bull
    out["ob_unmitigated_bear"] = unmit_bear
    out["ob_first_retest"] = first_retest
    return out


# ============================================================================
# Enhanced Liquidity Sweep
# ============================================================================


def add_liquidity_sweep(
    df: pd.DataFrame,
    atr_threshold: float = 0.2,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    level_tolerance_atr: float = 0.0,
    min_reclaim_atr: float = 0.0,
    min_wick_frac: float = 0.25,
    max_body_frac: float = 0.8,
    max_level_age: int | None = 80,
    require_next_bar_confirmation: bool = False,
    confirmation_mode: str = "close",
    use_provided_swings: bool = True,
) -> pd.DataFrame:
    """
    Enhanced liquidity sweep detector.

    A sweep is defined as:
    - price trades beyond a reference liquidity level (swing high/low)
    - then closes back inside the level on the same candle
    - optionally, the next bar confirms reversal

    Backward-compatible columns
    ---------------------------
    * sweep_high
    * sweep_low
    * sweep_magnitude

    Added columns
    -------------
    * sweep_high_magnitude
    * sweep_low_magnitude
    * sweep_high_reclaim_atr
    * sweep_low_reclaim_atr
    * sweep_high_wick_frac
    * sweep_low_wick_frac
    * sweep_high_body_frac
    * sweep_low_body_frac
    * sweep_level_high
    * sweep_level_low
    * sweep_level_high_age
    * sweep_level_low_age
    * sweep_confirmed_high
    * sweep_confirmed_low
    * sweep_side
    """
    out = df.copy()
    n = len(out)

    req = {"open", "high", "low", "close"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(
            f"add_liquidity_sweep: missing required columns: {sorted(missing)}"
        )

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)

    # ---------------------------------------------------------------------
    # ATR
    # ---------------------------------------------------------------------
    atr = _ensure_atr(out, atr_length)

    # ---------------------------------------------------------------------
    # Reference swing levels
    # ---------------------------------------------------------------------
    if (
        use_provided_swings
        and "last_swing_high" in out.columns
        and "last_swing_low" in out.columns
    ):
        last_sh = out["last_swing_high"].to_numpy(dtype=float)
        last_sl = out["last_swing_low"].to_numpy(dtype=float)

        # optional age columns if already present
        if "swing_high_age" in out.columns:
            sh_age = out["swing_high_age"].to_numpy(dtype=float)
        else:
            sh_age = np.full(n, np.nan)

        if "swing_low_age" in out.columns:
            sl_age = out["swing_low_age"].to_numpy(dtype=float)
        else:
            sl_age = np.full(n, np.nan)

    else:
        # causal fallback from confirmed pivots
        ph = _pivot_high(h, left=swing_left, right=swing_right)
        pl = _pivot_low(lo, left=swing_left, right=swing_right)

        last_sh = np.full(n, np.nan)
        last_sl = np.full(n, np.nan)
        sh_age = np.full(n, np.nan)
        sl_age = np.full(n, np.nan)

        cur_sh = np.nan
        cur_sl = np.nan
        cur_sh_idx = -1
        cur_sl_idx = -1

        for i in range(n):
            j = i - swing_right
            if j >= 0:
                if ph[j] == 1:
                    cur_sh = h[j]
                    cur_sh_idx = j
                if pl[j] == 1:
                    cur_sl = lo[j]
                    cur_sl_idx = j

            last_sh[i] = cur_sh
            last_sl[i] = cur_sl
            sh_age[i] = (i - cur_sh_idx) if cur_sh_idx >= 0 else np.nan
            sl_age[i] = (i - cur_sl_idx) if cur_sl_idx >= 0 else np.nan

    # ---------------------------------------------------------------------
    # Candle geometry
    # ---------------------------------------------------------------------
    rng = h - lo
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo

    with np.errstate(invalid="ignore", divide="ignore"):
        upper_wick_frac = np.where(rng > 0, upper_wick / rng, 0.0)
        lower_wick_frac = np.where(rng > 0, lower_wick / rng, 0.0)
        body_frac = np.where(rng > 0, body / rng, 0.0)

    # ---------------------------------------------------------------------
    # Output arrays
    # ---------------------------------------------------------------------
    sw_h = np.zeros(n, dtype=np.int8)
    sw_l = np.zeros(n, dtype=np.int8)
    sw_mag = np.full(n, np.nan)

    sw_h_mag = np.full(n, np.nan)
    sw_l_mag = np.full(n, np.nan)

    sw_h_reclaim = np.full(n, np.nan)
    sw_l_reclaim = np.full(n, np.nan)

    sw_h_wick_frac = np.full(n, np.nan)
    sw_l_wick_frac = np.full(n, np.nan)

    sw_h_body_frac = np.full(n, np.nan)
    sw_l_body_frac = np.full(n, np.nan)

    sw_level_high = np.full(n, np.nan)
    sw_level_low = np.full(n, np.nan)

    sw_level_high_age = np.full(n, np.nan)
    sw_level_low_age = np.full(n, np.nan)

    sw_conf_h = np.zeros(n, dtype=np.int8)
    sw_conf_l = np.zeros(n, dtype=np.int8)

    #  1 = high sweep, -1 = low sweep, 0 = none
    sw_side = np.zeros(n, dtype=np.int8)

    # ---------------------------------------------------------------------
    # Main detection
    # ---------------------------------------------------------------------
    for i in range(1, n):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        tol = level_tolerance_atr * atr_i

        # ================================================================
        # High sweep
        # ================================================================
        if np.isfinite(last_sh[i]):
            level = last_sh[i]
            age_ok = (
                True
                if max_level_age is None
                else (np.isfinite(sh_age[i]) and sh_age[i] <= max_level_age)
            )

            wick_through = h[i] > (level + tol)
            close_back_inside = c[i] < (level - tol)
            excursion = h[i] - level
            excursion_atr = excursion / atr_i
            reclaim_atr = (level - c[i]) / atr_i
            wick_ok = upper_wick_frac[i] >= min_wick_frac
            body_ok = body_frac[i] <= max_body_frac

            high_sweep_now = (
                age_ok
                and wick_through
                and close_back_inside
                and excursion_atr >= atr_threshold
                and reclaim_atr >= min_reclaim_atr
                and wick_ok
                and body_ok
            )

            if high_sweep_now:
                confirmed = True
                if require_next_bar_confirmation and i + 1 < n:
                    if confirmation_mode == "mid":
                        confirmed = c[i + 1] < (o[i] + c[i]) / 2.0
                    elif confirmation_mode == "low":
                        confirmed = c[i + 1] < lo[i]
                    else:  # close
                        confirmed = c[i + 1] < c[i]
                elif require_next_bar_confirmation and i + 1 >= n:
                    confirmed = False

                if confirmed:
                    sw_h[i] = 1
                    sw_h_mag[i] = excursion_atr
                    sw_h_reclaim[i] = reclaim_atr
                    sw_h_wick_frac[i] = upper_wick_frac[i]
                    sw_h_body_frac[i] = body_frac[i]
                    sw_level_high[i] = level
                    sw_level_high_age[i] = sh_age[i]
                    sw_conf_h[i] = 1
                    sw_side[i] = 1

        # ================================================================
        # Low sweep
        # ================================================================
        if np.isfinite(last_sl[i]):
            level = last_sl[i]
            age_ok = (
                True
                if max_level_age is None
                else (np.isfinite(sl_age[i]) and sl_age[i] <= max_level_age)
            )

            wick_through = lo[i] < (level - tol)
            close_back_inside = c[i] > (level + tol)
            excursion = level - lo[i]
            excursion_atr = excursion / atr_i
            reclaim_atr = (c[i] - level) / atr_i
            wick_ok = lower_wick_frac[i] >= min_wick_frac
            body_ok = body_frac[i] <= max_body_frac

            low_sweep_now = (
                age_ok
                and wick_through
                and close_back_inside
                and excursion_atr >= atr_threshold
                and reclaim_atr >= min_reclaim_atr
                and wick_ok
                and body_ok
            )

            if low_sweep_now:
                confirmed = True
                if require_next_bar_confirmation and i + 1 < n:
                    if confirmation_mode == "mid":
                        confirmed = c[i + 1] > (o[i] + c[i]) / 2.0
                    elif confirmation_mode == "high":
                        confirmed = c[i + 1] > h[i]
                    else:  # close
                        confirmed = c[i + 1] > c[i]
                elif require_next_bar_confirmation and i + 1 >= n:
                    confirmed = False

                if confirmed:
                    sw_l[i] = 1
                    sw_l_mag[i] = excursion_atr
                    sw_l_reclaim[i] = reclaim_atr
                    sw_l_wick_frac[i] = lower_wick_frac[i]
                    sw_l_body_frac[i] = body_frac[i]
                    sw_level_low[i] = level
                    sw_level_low_age[i] = sl_age[i]
                    sw_conf_l[i] = 1
                    # if both happen on same bar, keep 0 as ambiguous
                    sw_side[i] = -1 if sw_side[i] == 0 else 0

        # backward-compatible shared magnitude
        mags = []
        if np.isfinite(sw_h_mag[i]):
            mags.append(sw_h_mag[i])
        if np.isfinite(sw_l_mag[i]):
            mags.append(sw_l_mag[i])
        if mags:
            sw_mag[i] = np.nanmax(mags)

    out["sweep_high"] = sw_h
    out["sweep_low"] = sw_l
    out["sweep_magnitude"] = sw_mag

    out["sweep_high_magnitude"] = sw_h_mag
    out["sweep_low_magnitude"] = sw_l_mag

    out["sweep_high_reclaim_atr"] = sw_h_reclaim
    out["sweep_low_reclaim_atr"] = sw_l_reclaim

    out["sweep_high_wick_frac"] = sw_h_wick_frac
    out["sweep_low_wick_frac"] = sw_l_wick_frac

    out["sweep_high_body_frac"] = sw_h_body_frac
    out["sweep_low_body_frac"] = sw_l_body_frac

    out["sweep_level_high"] = sw_level_high
    out["sweep_level_low"] = sw_level_low

    out["sweep_level_high_age"] = sw_level_high_age
    out["sweep_level_low_age"] = sw_level_low_age

    out["sweep_confirmed_high"] = sw_conf_h
    out["sweep_confirmed_low"] = sw_conf_l
    out["sweep_side"] = sw_side

    return out


# ============================================================================
# Enhanced Equal Highs / Lows Detector
# ============================================================================


def add_equal_hl(
    df: pd.DataFrame,
    atr_tolerance: float = 0.1,
    *,
    atr_length: int = 14,
    swing_left: int = 2,
    swing_right: int = 2,
    use_provided_swings: bool = True,
    lookback_swings: int = 50,
    min_touches: int = 2,
    level_mode: str = "median",  # "median", "mean", "first", "last"
    max_cluster_width_atr: float | None = None,
    max_cluster_span: int | None = None,
    invalidate_on_sweep: bool = False,
    sweep_tolerance_atr: float = 0.0,
    keep_last_n_clusters: int = 20,
) -> pd.DataFrame:
    """
    Enhanced Equal Highs / Equal Lows detector.

    Backward-compatible columns
    ---------------------------
    * equal_highs
    * equal_lows
    * equal_highs_count
    * equal_lows_count

    Added columns
    -------------
    * equal_highs_level / equal_lows_level
    * equal_highs_width / equal_lows_width
    * equal_highs_width_atr / equal_lows_width_atr
    * equal_highs_age / equal_lows_age
    * equal_highs_span / equal_lows_span
    * equal_highs_active / equal_lows_active
    * equal_highs_score / equal_lows_score
    * equal_highs_cluster_id / equal_lows_cluster_id
    """

    out = df.copy()
    n = len(out)

    req = {"high", "low", "close"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"add_equal_hl: missing required columns: {sorted(missing)}")

    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    atr = _ensure_atr(out, atr_length)

    # ---------------------------------------------------------------------
    # Swing source
    # ---------------------------------------------------------------------
    if (
        use_provided_swings
        and "swing_high_price" in out.columns
        and "swing_low_price" in out.columns
    ):
        sh_price = out["swing_high_price"].to_numpy(dtype=float)
        sl_price = out["swing_low_price"].to_numpy(dtype=float)

        if "swing_high" in out.columns:
            sh_flag = out["swing_high"].to_numpy(dtype=np.int8)
        else:
            sh_flag = np.where(np.isfinite(sh_price), 1, 0).astype(np.int8)

        if "swing_low" in out.columns:
            sl_flag = out["swing_low"].to_numpy(dtype=np.int8)
        else:
            sl_flag = np.where(np.isfinite(sl_price), 1, 0).astype(np.int8)

    else:
        # causal fallback pivots
        ph = _pivot_high(h, left=swing_left, right=swing_right)
        pl = _pivot_low(lo, left=swing_left, right=swing_right)

        sh_flag = ph.astype(np.int8)
        sl_flag = pl.astype(np.int8)

        sh_price = np.where(sh_flag == 1, h, np.nan)
        sl_price = np.where(sl_flag == 1, lo, np.nan)

    # ---------------------------------------------------------------------
    # Outputs
    # ---------------------------------------------------------------------
    eq_h = np.zeros(n, dtype=np.int8)
    eq_l = np.zeros(n, dtype=np.int8)
    eq_h_cnt = np.zeros(n, dtype=np.int16)
    eq_l_cnt = np.zeros(n, dtype=np.int16)

    eq_h_level = np.full(n, np.nan)
    eq_l_level = np.full(n, np.nan)

    eq_h_width = np.full(n, np.nan)
    eq_l_width = np.full(n, np.nan)
    eq_h_width_atr = np.full(n, np.nan)
    eq_l_width_atr = np.full(n, np.nan)

    eq_h_age = np.full(n, np.nan)
    eq_l_age = np.full(n, np.nan)
    eq_h_span = np.full(n, np.nan)
    eq_l_span = np.full(n, np.nan)

    eq_h_active = np.zeros(n, dtype=np.int8)
    eq_l_active = np.zeros(n, dtype=np.int8)

    eq_h_score = np.full(n, np.nan)
    eq_l_score = np.full(n, np.nan)

    eq_h_cluster_id = np.full(n, -1, dtype=int)
    eq_l_cluster_id = np.full(n, -1, dtype=int)

    # ---------------------------------------------------------------------
    # Cluster helpers
    # ---------------------------------------------------------------------
    next_cluster_id_h = 0
    next_cluster_id_l = 0

    high_clusters = []
    low_clusters = []

    def _cluster_level(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return np.nan
        if level_mode == "mean":
            return float(np.mean(arr))
        if level_mode == "first":
            return float(arr[0])
        if level_mode == "last":
            return float(arr[-1])
        return float(np.median(arr))  # default median

    def _cluster_width(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return np.nan
        return float(np.max(arr) - np.min(arr))

    def _cluster_score(count, width_atr, span):
        # higher count = better
        # tighter cluster = better
        # some span is useful, but too huge span should not dominate
        width_term = 1.0 / (1.0 + max(width_atr, 0.0))
        span_term = np.log1p(max(span, 0.0))
        return float(count * width_term * (1.0 + 0.1 * span_term))

    def _match_cluster(price, atr_i, clusters, side):
        tol = atr_tolerance * atr_i if np.isfinite(atr_i) and atr_i > 0 else 0.0
        candidates = []

        for idx, cl in enumerate(clusters):
            if not cl["active"]:
                continue
            level = cl["level"]
            if np.isfinite(level) and abs(price - level) <= tol:
                candidates.append((idx, abs(price - level)))

        if not candidates:
            return None
        # nearest cluster
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _register_touch(cl, price, idx, atr_i):
        cl["prices"].append(float(price))
        cl["indices"].append(int(idx))
        cl["last_idx"] = int(idx)
        cl["count"] = len(cl["prices"])
        cl["level"] = _cluster_level(cl["prices"])
        cl["width"] = _cluster_width(cl["prices"])
        cl["width_atr"] = (
            cl["width"] / atr_i if np.isfinite(atr_i) and atr_i > 0 else np.nan
        )
        cl["span"] = cl["indices"][-1] - cl["indices"][0]
        cl["age"] = idx - cl["indices"][-1]  # usually 0 at update bar
        cl["score"] = _cluster_score(
            cl["count"],
            0.0 if np.isnan(cl["width_atr"]) else cl["width_atr"],
            cl["span"],
        )

    def _prune_clusters(clusters):
        # keep only recent active/relevant clusters
        if len(clusters) <= keep_last_n_clusters:
            return clusters
        # prefer active and recent
        clusters = sorted(
            clusters,
            key=lambda z: (z["active"], z["last_idx"]),
            reverse=True,
        )
        kept = clusters[:keep_last_n_clusters]
        # restore chronological-ish order
        kept = sorted(kept, key=lambda z: z["id"])
        return kept

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    for i in range(n):
        atr_i = atr[i] if np.isfinite(atr[i]) else np.nan
        sweep_tol = (
            sweep_tolerance_atr * atr_i if np.isfinite(atr_i) and atr_i > 0 else 0.0
        )

        # -------------------------------------------------------------
        # Invalidate active EQH/EQL clusters if swept
        # -------------------------------------------------------------
        if invalidate_on_sweep:
            for cl in high_clusters:
                if not cl["active"] or not np.isfinite(cl["level"]):
                    continue
                # high-side liquidity pool swept if current high takes it
                if h[i] > cl["level"] + sweep_tol:
                    cl["active"] = False
                    cl["swept_idx"] = i

            for cl in low_clusters:
                if not cl["active"] or not np.isfinite(cl["level"]):
                    continue
                # low-side liquidity pool swept if current low takes it
                if lo[i] < cl["level"] - sweep_tol:
                    cl["active"] = False
                    cl["swept_idx"] = i

        # -------------------------------------------------------------
        # Process new swing high
        # -------------------------------------------------------------
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            price = sh_price[i]

            # prune clusters by recency/lookback
            high_clusters = [
                cl for cl in high_clusters if (i - cl["last_idx"] <= lookback_swings)
            ]

            match_idx = _match_cluster(price, atr_i, high_clusters, side="high")

            if match_idx is None:
                # start new cluster
                cl = {
                    "id": next_cluster_id_h,
                    "prices": [float(price)],
                    "indices": [int(i)],
                    "first_idx": int(i),
                    "last_idx": int(i),
                    "count": 1,
                    "level": float(price),
                    "width": 0.0,
                    "width_atr": 0.0,
                    "span": 0,
                    "age": 0,
                    "score": 1.0,
                    "active": True,
                    "swept_idx": -1,
                }
                high_clusters.append(cl)
                next_cluster_id_h += 1
            else:
                cl = high_clusters[match_idx]
                _register_touch(cl, price, i, atr_i)

            # optional filters on the updated cluster
            width_ok = True
            if max_cluster_width_atr is not None and np.isfinite(cl["width_atr"]):
                width_ok = cl["width_atr"] <= max_cluster_width_atr

            span_ok = True
            if max_cluster_span is not None:
                span_ok = cl["span"] <= max_cluster_span

            if cl["count"] >= min_touches and width_ok and span_ok:
                eq_h[i] = 1
                eq_h_cnt[i] = cl["count"]
                eq_h_level[i] = cl["level"]
                eq_h_width[i] = cl["width"]
                eq_h_width_atr[i] = cl["width_atr"]
                eq_h_age[i] = i - cl["last_idx"]
                eq_h_span[i] = cl["span"]
                eq_h_active[i] = 1 if cl["active"] else 0
                eq_h_score[i] = cl["score"]
                eq_h_cluster_id[i] = cl["id"]

        # -------------------------------------------------------------
        # Process new swing low
        # -------------------------------------------------------------
        if sl_flag[i] == 1 and np.isfinite(sl_price[i]):
            price = sl_price[i]

            low_clusters = [
                cl for cl in low_clusters if (i - cl["last_idx"] <= lookback_swings)
            ]

            match_idx = _match_cluster(price, atr_i, low_clusters, side="low")

            if match_idx is None:
                cl = {
                    "id": next_cluster_id_l,
                    "prices": [float(price)],
                    "indices": [int(i)],
                    "first_idx": int(i),
                    "last_idx": int(i),
                    "count": 1,
                    "level": float(price),
                    "width": 0.0,
                    "width_atr": 0.0,
                    "span": 0,
                    "age": 0,
                    "score": 1.0,
                    "active": True,
                    "swept_idx": -1,
                }
                low_clusters.append(cl)
                next_cluster_id_l += 1
            else:
                cl = low_clusters[match_idx]
                _register_touch(cl, price, i, atr_i)

            width_ok = True
            if max_cluster_width_atr is not None and np.isfinite(cl["width_atr"]):
                width_ok = cl["width_atr"] <= max_cluster_width_atr

            span_ok = True
            if max_cluster_span is not None:
                span_ok = cl["span"] <= max_cluster_span

            if cl["count"] >= min_touches and width_ok and span_ok:
                eq_l[i] = 1
                eq_l_cnt[i] = cl["count"]
                eq_l_level[i] = cl["level"]
                eq_l_width[i] = cl["width"]
                eq_l_width_atr[i] = cl["width_atr"]
                eq_l_age[i] = i - cl["last_idx"]
                eq_l_span[i] = cl["span"]
                eq_l_active[i] = 1 if cl["active"] else 0
                eq_l_score[i] = cl["score"]
                eq_l_cluster_id[i] = cl["id"]

        high_clusters = _prune_clusters(high_clusters)
        low_clusters = _prune_clusters(low_clusters)

    # ---------------------------------------------------------------------
    # Backfill current active cluster state onto non-swing bars
    # This makes downstream use easier.
    # ---------------------------------------------------------------------
    latest_active_high = None
    latest_active_low = None

    # also include clusters that may have been pruned out of current lists but had outputs already;
    # non-critical, so we keep this part simple and only carry latest seen outputs forward.
    for i in range(n):
        if eq_h_cluster_id[i] >= 0:
            latest_active_high = (
                eq_h_cluster_id[i],
                eq_h_level[i],
                eq_h_cnt[i],
                eq_h_width[i],
                eq_h_width_atr[i],
                eq_h_span[i],
                eq_h_score[i],
                eq_h_active[i],
            )

        if latest_active_high is not None and eq_h_cluster_id[i] < 0:
            cid, lvl, cnt, wid, wid_atr, span, score, active = latest_active_high
            if active == 1:
                eq_h_level[i] = lvl if np.isnan(eq_h_level[i]) else eq_h_level[i]
                eq_h_cnt[i] = cnt if eq_h_cnt[i] == 0 else eq_h_cnt[i]
                eq_h_width[i] = wid if np.isnan(eq_h_width[i]) else eq_h_width[i]
                eq_h_width_atr[i] = (
                    wid_atr if np.isnan(eq_h_width_atr[i]) else eq_h_width_atr[i]
                )
                eq_h_span[i] = span if np.isnan(eq_h_span[i]) else eq_h_span[i]
                eq_h_score[i] = score if np.isnan(eq_h_score[i]) else eq_h_score[i]
                eq_h_active[i] = 1 if eq_h_active[i] == 0 else eq_h_active[i]

        if eq_l_cluster_id[i] >= 0:
            latest_active_low = (
                eq_l_cluster_id[i],
                eq_l_level[i],
                eq_l_cnt[i],
                eq_l_width[i],
                eq_l_width_atr[i],
                eq_l_span[i],
                eq_l_score[i],
                eq_l_active[i],
            )

        if latest_active_low is not None and eq_l_cluster_id[i] < 0:
            cid, lvl, cnt, wid, wid_atr, span, score, active = latest_active_low
            if active == 1:
                eq_l_level[i] = lvl if np.isnan(eq_l_level[i]) else eq_l_level[i]
                eq_l_cnt[i] = cnt if eq_l_cnt[i] == 0 else eq_l_cnt[i]
                eq_l_width[i] = wid if np.isnan(eq_l_width[i]) else eq_l_width[i]
                eq_l_width_atr[i] = (
                    wid_atr if np.isnan(eq_l_width_atr[i]) else eq_l_width_atr[i]
                )
                eq_l_span[i] = span if np.isnan(eq_l_span[i]) else eq_l_span[i]
                eq_l_score[i] = score if np.isnan(eq_l_score[i]) else eq_l_score[i]
                eq_l_active[i] = 1 if eq_l_active[i] == 0 else eq_l_active[i]

    out["equal_highs"] = eq_h
    out["equal_lows"] = eq_l
    out["equal_highs_count"] = eq_h_cnt
    out["equal_lows_count"] = eq_l_cnt

    out["equal_highs_level"] = eq_h_level
    out["equal_lows_level"] = eq_l_level

    out["equal_highs_width"] = eq_h_width
    out["equal_lows_width"] = eq_l_width
    out["equal_highs_width_atr"] = eq_h_width_atr
    out["equal_lows_width_atr"] = eq_l_width_atr

    out["equal_highs_age"] = eq_h_age
    out["equal_lows_age"] = eq_l_age
    out["equal_highs_span"] = eq_h_span
    out["equal_lows_span"] = eq_l_span

    out["equal_highs_active"] = eq_h_active
    out["equal_lows_active"] = eq_l_active

    out["equal_highs_score"] = eq_h_score
    out["equal_lows_score"] = eq_l_score

    out["equal_highs_cluster_id"] = eq_h_cluster_id
    out["equal_lows_cluster_id"] = eq_l_cluster_id

    return out


# ============================================================================
# Enhanced Displacement Candle Detector
# ===========================================================================


def add_displacement_candle(
    df: pd.DataFrame,
    body_atr_mult: float = 1.5,
    *,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
) -> pd.DataFrame:
    """Enhanced displacement candle detector.

    A displacement candle has a large body relative to ATR, closes near its
    directional extreme, and has controlled wick structure.

    Parameters
    ----------
    body_atr_mult : float
        Minimum body / ATR ratio.
    close_extreme_frac : float
        Maximum distance from close to directional extreme as fraction of range.
    min_body_frac : float
        Minimum body / range ratio (body dominance).
    max_opposite_wick_frac : float
        Maximum opposite-side wick / range (bull: lower wick, bear: upper wick).

    Columns
    ~~~~~~~
    * ``displacement_candle``        – 1 if all conditions met
    * ``displacement_body_atr``      – body / ATR (continuous)
    * ``displacement_close_extreme`` – 1 if close near extreme
    * ``displacement_direction``     – +1 bull, -1 bear, 0 doji
    * ``displacement_bull_candle``   – 1 if bullish displacement
    * ``displacement_bear_candle``   – 1 if bearish displacement
    """
    out = df.copy()
    atr = _ensure_atr(out)
    atr_s = pd.Series(atr, index=out.index)

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    lo = out["low"].astype(float)
    c = out["close"].astype(float)

    body = (c - o).abs()
    rng = h - lo

    # Direction
    direction = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)
    bull = direction == 1
    bear = direction == -1

    # Body / ATR ratio
    with np.errstate(invalid="ignore", divide="ignore"):
        body_atr = np.where(atr_s > 0, body / atr_s, 0.0)
        body_frac = np.where(rng > 0, body / rng, 0.0)

    # Close extreme: distance from close to directional extreme / range
    dist_to_extreme = np.where(bull, h - c, np.where(bear, c - lo, np.nan)).astype(
        float
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        close_ext_frac = np.where(rng > 0, dist_to_extreme / rng, np.nan)
    close_extreme_ok = np.where(
        np.isfinite(close_ext_frac), close_ext_frac <= close_extreme_frac, False
    )

    # Opposite wick: bull → lower wick, bear → upper wick
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo
    opp_wick = np.where(bull, lower_wick, np.where(bear, upper_wick, 0.0)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        opp_wick_frac = np.where(rng > 0, opp_wick / rng, 0.0)
    opp_wick_ok = opp_wick_frac <= max_opposite_wick_frac

    # Body dominance
    body_dom_ok = body_frac >= min_body_frac

    # Final flag: all conditions
    valid_dir = direction != 0
    final = (
        valid_dir
        & (body_atr >= body_atr_mult)
        & close_extreme_ok
        & body_dom_ok
        & opp_wick_ok
    )

    out["displacement_candle"] = final.astype(int)
    out["displacement_body_atr"] = body_atr
    out["displacement_close_extreme"] = close_extreme_ok.astype(int)
    out["displacement_direction"] = direction
    out["displacement_bull_candle"] = (final & bull).astype(int)
    out["displacement_bear_candle"] = (final & bear).astype(int)

    return out


# ============================================================================
# Enhanced AMD Engine (Accumulation → Manipulation → Distribution)
# ============================================================================

# AMD state constants
AMD_UNKNOWN = -1
AMD_ACCUMULATION = 0
AMD_MANIPULATION = 1
AMD_DISTRIBUTION = 2


def _amd_rolling_rank_pct(arr: np.ndarray) -> float:
    """Percentile rank of last element in trailing window, [0, 100]."""
    if arr.size == 0 or np.isnan(arr[-1]):
        return np.nan
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan
    return float((valid <= valid[-1]).mean() * 100.0)


def _amd_safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division, NaN where denom is 0 or NaN."""
    out = pd.Series(np.nan, index=numer.index, dtype=float)
    valid = denom.notna() & (denom != 0)
    out.loc[valid] = numer.loc[valid] / denom.loc[valid]
    return out


def _amd_rolling_overlap(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Average consecutive-candle overlap fraction over trailing window."""
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    overlap = (np.minimum(high, prev_h) - np.maximum(low, prev_l)).clip(lower=0.0)
    union = np.maximum(high, prev_h) - np.minimum(low, prev_l)
    return (
        _amd_safe_div(overlap, union).rolling(window=window, min_periods=window).mean()
    )


def _amd_rolling_efficiency(close: pd.Series, window: int) -> pd.Series:
    """Directional efficiency: net move / gross path. Higher = trendier."""
    net = (close - close.shift(window - 1)).abs()
    gross = close.diff().abs().rolling(window=window, min_periods=window).sum()
    return _amd_safe_div(net, gross)


def add_amd_features(
    df: pd.DataFrame,
    *,
    atr_pct_window: int = 50,
    accumulation_window: int = 20,
    overlap_window: int = 10,
    accumulation_min_streak: int = 8,
    atr_pct_low_threshold: float = 45.0,
    box_width_atr_max: float = 12.0,
    box_width_pct_max: float = 0.040,
    overlap_min: float = 0.40,
    efficiency_max: float = 0.35,
    min_touch_count_each_side: int = 1,
    sweep_tolerance_atr: float = 0.15,
    reclaim_min_frac_of_box: float = 0.10,
    displacement_mode: str = "break_only",
    min_distribution_followthrough_bars: int = 2,
    min_distribution_move_atr: float = 0.35,
    min_distribution_move_box_frac: float = 0.30,
    max_reentry_frac_of_box: float = 0.20,
) -> pd.DataFrame:
    """
    Compute causal AMD feature columns. Live-safe (trailing-only windows).

    Parameters
    ----------
    displacement_mode : str
        'all' — require displacement for both reclaim and break manipulation
        'break_only' — require displacement only for break-style, not reclaim
        'none' — never require displacement
          Columns
    ~~~~~~~
    * ``amd_box_high / amd_box_low / amd_box_mid / amd_box_width``
    * ``amd_compression_score``       – 0–5, quality of accumulation
    * ``amd_overlap_score``           – candle overlap in range
    * ``amd_efficiency``              – directional efficiency (low = choppy)
    * ``amd_accumulation_active``     – 1 when compression streak met
    * ``amd_manipulation_candidate``  – 1 on manipulation trigger bar
    * ``amd_manipulation_direction``  – +1 bull / -1 bear / 0 none
    * ``amd_distribution_bull_candidate / amd_distribution_bear_candidate``
    * ``amd_reentry_strict``          – 1 when close is inside box
    * ``amd_reentry_buffered``        – 1 when close is near box (with tolerance)
    """
    out = df.copy()
    h = out["high"].astype(float)
    lo = out["low"].astype(float)
    c = out["close"].astype(float)
    o = out["open"].astype(float)

    atr = pd.Series(_ensure_atr(out), index=out.index, dtype=float)

    # ATR percentile (causal)
    atr_pct = atr.rolling(atr_pct_window, min_periods=atr_pct_window).apply(
        _amd_rolling_rank_pct, raw=True
    )

    # Accumulation box
    box_high = h.rolling(accumulation_window, min_periods=accumulation_window).max()
    box_low = lo.rolling(accumulation_window, min_periods=accumulation_window).min()
    box_mid = (box_high + box_low) / 2.0
    box_width = (box_high - box_low).astype(float)
    box_width_atr = _amd_safe_div(box_width, atr)
    box_width_pct = _amd_safe_div(box_width, c.abs())

    # Compression metrics
    overlap_score = _amd_rolling_overlap(h, lo, overlap_window)
    efficiency = _amd_rolling_efficiency(c, accumulation_window)

    # Box touch counts
    tol = box_width * 0.10
    touch_high = (
        ((box_high - h).abs() <= tol)
        .rolling(accumulation_window, min_periods=accumulation_window)
        .sum()
    )
    touch_low = (
        ((lo - box_low).abs() <= tol)
        .rolling(accumulation_window, min_periods=accumulation_window)
        .sum()
    )

    # Compression flags
    low_atr_flag = atr_pct <= atr_pct_low_threshold
    narrow_box_flag = (box_width_atr <= box_width_atr_max) & (
        box_width_pct <= box_width_pct_max
    )
    overlap_flag = overlap_score >= overlap_min
    efficiency_flag = efficiency <= efficiency_max
    touch_flag = (touch_high >= min_touch_count_each_side) & (
        touch_low >= min_touch_count_each_side
    )

    compression_score = (
        low_atr_flag.astype(int)
        + narrow_box_flag.astype(int)
        + overlap_flag.astype(int)
        + efficiency_flag.astype(int)
        + touch_flag.astype(int)
    ).astype(np.int8)

    accumulation_candidate = (
        low_atr_flag & narrow_box_flag & overlap_flag & efficiency_flag & touch_flag
    )
    streak = pd.Series(
        np.where(accumulation_candidate, 1, 0), index=out.index, dtype="int64"
    )
    streak = streak.groupby((streak == 0).cumsum()).cumsum()
    accumulation_active = streak >= accumulation_min_streak

    # Displacement integration
    disp_flag = pd.Series(False, index=out.index)
    disp_dir = pd.Series(0, index=out.index, dtype="int8")
    if "displacement_candle" in out.columns:
        disp_flag = out["displacement_candle"].fillna(0).astype(int) == 1
    if "displacement_direction" in out.columns:
        disp_dir = out["displacement_direction"].fillna(0).astype("int8")
    else:
        disp_dir = pd.Series(
            np.where(c > o, 1, np.where(c < o, -1, 0)),
            index=out.index,
            dtype="int8",
        )

    # Box breaks and sweeps
    prior_bh = box_high.shift(1)
    prior_bl = box_low.shift(1)
    prior_bw = (prior_bh - prior_bl).astype(float)
    sweep_tol = atr * sweep_tolerance_atr

    break_up = h > prior_bh
    break_down = lo < prior_bl
    sweep_up = break_up & (h <= (prior_bh + sweep_tol))
    sweep_down = break_down & (lo >= (prior_bl - sweep_tol))

    reclaim_thresh = prior_bw * reclaim_min_frac_of_box
    reclaim_bull = sweep_down & (c >= (prior_bl + reclaim_thresh))
    reclaim_bear = sweep_up & (c <= (prior_bh - reclaim_thresh))

    # Manipulation candidates — two styles, gated by displacement_mode
    acc_prev = accumulation_active.shift(1).fillna(False)

    if displacement_mode == "all":
        reclaim_manip_bull = acc_prev & reclaim_bull & disp_flag & (disp_dir == 1)
        reclaim_manip_bear = acc_prev & reclaim_bear & disp_flag & (disp_dir == -1)
        break_manip_bull = acc_prev & break_up & disp_flag & (disp_dir == 1)
        break_manip_bear = acc_prev & break_down & disp_flag & (disp_dir == -1)
    elif displacement_mode == "break_only":
        reclaim_manip_bull = acc_prev & reclaim_bull
        reclaim_manip_bear = acc_prev & reclaim_bear
        break_manip_bull = acc_prev & break_up & disp_flag & (disp_dir == 1)
        break_manip_bear = acc_prev & break_down & disp_flag & (disp_dir == -1)
    elif displacement_mode == "none":
        reclaim_manip_bull = acc_prev & reclaim_bull
        reclaim_manip_bear = acc_prev & reclaim_bear
        break_manip_bull = acc_prev & break_up
        break_manip_bear = acc_prev & break_down
    else:
        raise ValueError(
            "displacement_mode must be 'all', 'break_only', or 'none', "
            f"got '{displacement_mode}'"
        )

    manip_bull = reclaim_manip_bull | break_manip_bull
    manip_bear = reclaim_manip_bear | break_manip_bear

    manip_candidate = manip_bull | manip_bear
    manip_direction = pd.Series(
        np.where(manip_bull, 1, np.where(manip_bear, -1, 0)),
        index=out.index,
        dtype="int8",
    )

    # Rolling-box distribution candidates kept as diagnostics only
    prior_box_mid = box_mid.shift(1)
    move_from_mid = (c - prior_box_mid).abs()
    move_from_mid_atr = _amd_safe_div(move_from_mid, atr)
    move_from_mid_box = _amd_safe_div(move_from_mid, prior_bw)

    outside_up = c > prior_bh
    outside_down = c < prior_bl

    reentry_strict = (c >= prior_bl) & (c <= prior_bh)
    reentry_buffered = (c >= (prior_bl - prior_bw * max_reentry_frac_of_box)) & (
        c <= (prior_bh + prior_bw * max_reentry_frac_of_box)
    )

    dist_bull_pre = (
        outside_up
        & (move_from_mid_atr >= min_distribution_move_atr)
        & (move_from_mid_box >= min_distribution_move_box_frac)
    )
    dist_bear_pre = (
        outside_down
        & (move_from_mid_atr >= min_distribution_move_atr)
        & (move_from_mid_box >= min_distribution_move_box_frac)
    )

    bull_follow = pd.Series(
        np.where(dist_bull_pre, 1, 0), index=out.index, dtype="int64"
    )
    bull_follow = bull_follow.groupby((~dist_bull_pre).cumsum()).cumsum()

    bear_follow = pd.Series(
        np.where(dist_bear_pre, 1, 0), index=out.index, dtype="int64"
    )
    bear_follow = bear_follow.groupby((~dist_bear_pre).cumsum()).cumsum()

    dist_bull = (bull_follow >= min_distribution_followthrough_bars) & (~reentry_strict)
    dist_bear = (bear_follow >= min_distribution_followthrough_bars) & (~reentry_strict)

    # Output columns
    out["amd_box_high"] = box_high
    out["amd_box_low"] = box_low
    out["amd_box_mid"] = box_mid
    out["amd_box_width"] = box_width
    out["amd_compression_score"] = compression_score
    out["amd_overlap_score"] = overlap_score
    out["amd_efficiency"] = efficiency
    out["amd_accumulation_active"] = accumulation_active.astype(np.int8)
    out["amd_manipulation_candidate"] = manip_candidate.astype(np.int8)
    out["amd_manipulation_direction"] = manip_direction
    out["amd_distribution_bull_candidate"] = dist_bull.astype(np.int8)
    out["amd_distribution_bear_candidate"] = dist_bear.astype(np.int8)
    out["amd_reentry_strict"] = reentry_strict.astype(np.int8)
    out["amd_reentry_buffered"] = reentry_buffered.astype(np.int8)

    return out


def add_amd_state(
    df: pd.DataFrame,
    *,
    manipulation_timeout_bars: int = 8,
    allow_unknown_state: bool = True,
    reset_to_accumulation_on_new_box: bool = True,
    accumulation_grace_bars: int = 2,
    max_distribution_stall: int = 4,
    min_distribution_move_atr: float = 0.75,
    min_distribution_move_box_frac: float = 0.60,
    min_distribution_followthrough_bars: int = 4,
    min_distribution_extension_atr: float = 0.10,
) -> pd.DataFrame:
    """
    Causal AMD state machine. Assigns phase per bar.

    Distribution confirmation and stall logic are based on the FROZEN
    originating accumulation box, not the rolling box.
    """
    needed = [
        "amd_box_high",
        "amd_box_low",
        "amd_box_mid",
        "amd_accumulation_active",
        "amd_manipulation_candidate",
        "amd_manipulation_direction",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Run add_amd_features first. Missing: {', '.join(missing)}")

    out = df.copy()
    n = len(out)

    acc_active = out["amd_accumulation_active"].values.astype(int)
    manip_bull = (out["amd_manipulation_direction"].values == 1).astype(int)
    manip_bear = (out["amd_manipulation_direction"].values == -1).astype(int)
    close_v = out["close"].values.astype(float)
    high_v = out["high"].values.astype(float)
    low_v = out["low"].values.astype(float)
    bx_h = out["amd_box_high"].values.astype(float)
    bx_l = out["amd_box_low"].values.astype(float)
    bx_m = out["amd_box_mid"].values.astype(float)
    atr_v = np.asarray(_ensure_atr(out), dtype=float)

    phase = np.full(n, AMD_UNKNOWN, dtype=np.int8)
    direction = np.zeros(n, dtype=np.int8)
    seq_id = np.full(n, -1, dtype=np.int32)
    bars_in = np.zeros(n, dtype=np.int32)
    active_box_high = np.full(n, np.nan, dtype=float)
    active_box_low = np.full(n, np.nan, dtype=float)
    active_box_mid = np.full(n, np.nan, dtype=float)

    cur_phase = AMD_UNKNOWN
    cur_dir = 0
    cur_seq = -1
    phase_age = 0
    manip_age = 0
    grace_count = 0
    dist_stall = 0
    dist_follow = 0
    dist_best_high = np.nan
    dist_best_low = np.nan

    # Frozen box from the accumulation that spawned the current cycle
    frozen_bh = np.nan
    frozen_bl = np.nan
    frozen_bm = np.nan

    for i in range(n):
        acc = acc_active[i] == 1
        mb = manip_bull[i] == 1
        ms = manip_bear[i] == 1

        # Reentry against frozen box
        re = (
            np.isfinite(frozen_bl)
            and np.isfinite(frozen_bh)
            and frozen_bl <= close_v[i] <= frozen_bh
        )

        # Distribution conditions based on frozen box
        bull_dist_entry = False
        bear_dist_entry = False
        bull_dist_active = False
        bear_dist_active = False

        if (
            np.isfinite(frozen_bh)
            and np.isfinite(frozen_bl)
            and np.isfinite(frozen_bm)
            and np.isfinite(atr_v[i])
            and atr_v[i] > 0
        ):
            frozen_bw = frozen_bh - frozen_bl
            if np.isfinite(frozen_bw) and frozen_bw > 0:
                move_from_frozen_mid = abs(close_v[i] - frozen_bm)
                move_from_frozen_mid_atr = move_from_frozen_mid / atr_v[i]
                move_from_frozen_mid_box = move_from_frozen_mid / frozen_bw

                # Entry into distribution: confirmed escape + delivery away from frozen box
                bull_dist_entry = (
                    (close_v[i] > frozen_bh)
                    and (move_from_frozen_mid_atr >= min_distribution_move_atr)
                    and (move_from_frozen_mid_box >= min_distribution_move_box_frac)
                )
                bear_dist_entry = (
                    (close_v[i] < frozen_bl)
                    and (move_from_frozen_mid_atr >= min_distribution_move_atr)
                    and (move_from_frozen_mid_box >= min_distribution_move_box_frac)
                )

                # Active distribution: must make fresh extreme beyond best seen in distribution
                ext_thresh = min_distribution_extension_atr * atr_v[i]

                if cur_phase == AMD_DISTRIBUTION:
                    if cur_dir == 1 and np.isfinite(dist_best_high):
                        bull_dist_active = (close_v[i] > frozen_bh) and (
                            high_v[i] >= dist_best_high + ext_thresh
                        )
                    elif cur_dir == -1 and np.isfinite(dist_best_low):
                        bear_dist_active = (close_v[i] < frozen_bl) and (
                            low_v[i] <= dist_best_low - ext_thresh
                        )

        if cur_phase == AMD_UNKNOWN:
            if acc:
                cur_phase = AMD_ACCUMULATION
                cur_dir = 0
                cur_seq += 1
                frozen_bh = bx_h[i]
                frozen_bl = bx_l[i]
                frozen_bm = bx_m[i]
                phase_age = 1
                grace_count = 0
                manip_age = 0
                dist_stall = 0
                dist_follow = 0
                dist_best_high = np.nan
                dist_best_low = np.nan
            else:
                phase_age = 0

        elif cur_phase == AMD_ACCUMULATION:
            if acc:
                # Update frozen box while still accumulating
                frozen_bh = bx_h[i]
                frozen_bl = bx_l[i]
                frozen_bm = bx_m[i]
                phase_age += 1
                grace_count = 0

            elif mb:
                cur_phase = AMD_MANIPULATION
                cur_dir = 1
                manip_age = 1
                phase_age = 1
                grace_count = 0
                dist_stall = 0
                dist_follow = 0
                dist_best_high = np.nan
                dist_best_low = np.nan

            elif ms:
                cur_phase = AMD_MANIPULATION
                cur_dir = -1
                manip_age = 1
                phase_age = 1
                grace_count = 0
                dist_stall = 0
                dist_follow = 0
                dist_best_high = np.nan
                dist_best_low = np.nan

            else:
                grace_count += 1
                phase_age += 1
                if grace_count > accumulation_grace_bars and allow_unknown_state:
                    cur_phase = AMD_UNKNOWN
                    cur_dir = 0
                    phase_age = 0
                    grace_count = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                    frozen_bh = np.nan
                    frozen_bl = np.nan
                    frozen_bm = np.nan

        elif cur_phase == AMD_MANIPULATION:
            manip_age += 1
            phase_age += 1

            if cur_dir == 1 and bull_dist_entry:
                dist_follow += 1
                if dist_follow >= min_distribution_followthrough_bars:
                    cur_phase = AMD_DISTRIBUTION
                    phase_age = 1
                    dist_stall = 0
                    dist_best_high = high_v[i]
                    dist_best_low = low_v[i]
            elif cur_dir == -1 and bear_dist_entry:
                dist_follow += 1
                if dist_follow >= min_distribution_followthrough_bars:
                    cur_phase = AMD_DISTRIBUTION
                    phase_age = 1
                    dist_stall = 0
                    dist_best_high = high_v[i]
                    dist_best_low = low_v[i]
            else:
                dist_follow = 0

            if cur_phase == AMD_MANIPULATION and manip_age > manipulation_timeout_bars:
                if acc and reset_to_accumulation_on_new_box:
                    cur_phase = AMD_ACCUMULATION
                    cur_dir = 0
                    cur_seq += 1
                    frozen_bh = bx_h[i]
                    frozen_bl = bx_l[i]
                    frozen_bm = bx_m[i]
                    phase_age = 1
                    grace_count = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                elif allow_unknown_state:
                    cur_phase = AMD_UNKNOWN
                    cur_dir = 0
                    phase_age = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                    frozen_bh = np.nan
                    frozen_bl = np.nan
                    frozen_bm = np.nan

        elif cur_phase == AMD_DISTRIBUTION:
            phase_age += 1

            if cur_dir == 1 and bull_dist_active:
                dist_stall = 0
                dist_best_high = high_v[i]
            elif cur_dir == -1 and bear_dist_active:
                dist_stall = 0
                dist_best_low = low_v[i]
            else:
                dist_stall += 1

            if re:
                if acc and reset_to_accumulation_on_new_box:
                    cur_phase = AMD_ACCUMULATION
                    cur_dir = 0
                    cur_seq += 1
                    frozen_bh = bx_h[i]
                    frozen_bl = bx_l[i]
                    frozen_bm = bx_m[i]
                    phase_age = 1
                    grace_count = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                elif allow_unknown_state:
                    cur_phase = AMD_UNKNOWN
                    cur_dir = 0
                    phase_age = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                    frozen_bh = np.nan
                    frozen_bl = np.nan
                    frozen_bm = np.nan

            elif dist_stall > max_distribution_stall:
                if acc and reset_to_accumulation_on_new_box:
                    cur_phase = AMD_ACCUMULATION
                    cur_dir = 0
                    cur_seq += 1
                    frozen_bh = bx_h[i]
                    frozen_bl = bx_l[i]
                    frozen_bm = bx_m[i]
                    phase_age = 1
                    grace_count = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                elif allow_unknown_state:
                    cur_phase = AMD_UNKNOWN
                    cur_dir = 0
                    phase_age = 0
                    manip_age = 0
                    dist_stall = 0
                    dist_follow = 0
                    dist_best_high = np.nan
                    dist_best_low = np.nan
                    frozen_bh = np.nan
                    frozen_bl = np.nan
                    frozen_bm = np.nan

            elif acc and reset_to_accumulation_on_new_box:
                cur_phase = AMD_ACCUMULATION
                cur_dir = 0
                cur_seq += 1
                frozen_bh = bx_h[i]
                frozen_bl = bx_l[i]
                frozen_bm = bx_m[i]
                phase_age = 1
                grace_count = 0
                manip_age = 0
                dist_stall = 0
                dist_follow = 0
                dist_best_high = np.nan
                dist_best_low = np.nan

        phase[i] = cur_phase
        direction[i] = cur_dir
        seq_id[i] = cur_seq
        bars_in[i] = phase_age
        active_box_high[i] = frozen_bh
        active_box_low[i] = frozen_bl
        active_box_mid[i] = frozen_bm

    out["amd_phase"] = phase
    out["amd_direction"] = direction
    out["amd_sequence_id"] = seq_id
    out["amd_bars_in_phase"] = bars_in
    out["amd_active_box_high"] = active_box_high
    out["amd_active_box_low"] = active_box_low
    out["amd_active_box_mid"] = active_box_mid
    out["amd_is_accumulation"] = (phase == AMD_ACCUMULATION).astype(np.int8)
    out["amd_is_manipulation"] = (phase == AMD_MANIPULATION).astype(np.int8)
    out["amd_is_distribution"] = (phase == AMD_DISTRIBUTION).astype(np.int8)

    return out


def add_amd_labels(
    df: pd.DataFrame,
    *,
    label_lookahead: int = 10,
    label_target_atr: float = 1.5,
    label_stop_box_frac: float = 0.50,
) -> pd.DataFrame:
    """
    Retrospective AMD outcome labels for supervised learning.

    NOT live-safe — uses future bars. For offline labeling only.
    Uses the frozen active box for stop sizing.
    """
    needed = [
        "amd_phase",
        "amd_direction",
        "amd_active_box_high",
        "amd_active_box_low",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            "Run add_amd_features + add_amd_state first. Missing: " + ", ".join(missing)
        )

    out = df.copy()
    n = len(out)
    atr = np.asarray(_ensure_atr(out), dtype=float)

    phase_v = out["amd_phase"].values.astype(np.int8)
    dir_v = out["amd_direction"].values.astype(np.int8)
    bx_h = out["amd_active_box_high"].values.astype(float)
    bx_l = out["amd_active_box_low"].values.astype(float)
    h = out["high"].values.astype(float)
    lo = out["low"].values.astype(float)
    c = out["close"].values.astype(float)

    outcome = np.zeros(n, dtype=np.int8)
    fwd_ret = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if phase_v[i] != AMD_MANIPULATION or dir_v[i] == 0:
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        bw = bx_h[i] - bx_l[i]
        if not np.isfinite(bw) or bw <= 0:
            continue

        end = min(n, i + 1 + label_lookahead)
        if end <= i + 1:
            continue

        entry = c[i]
        target_d = label_target_atr * atr[i]
        stop_d = label_stop_box_frac * bw

        if dir_v[i] == 1:
            tgt = entry + target_d
            stp = entry - stop_d
            hit_tgt = np.where(h[i + 1 : end] >= tgt)[0]
            hit_stp = np.where(lo[i + 1 : end] <= stp)[0]
            max_fwd = h[i + 1 : end].max() - entry
        else:
            tgt = entry - target_d
            stp = entry + stop_d
            hit_tgt = np.where(lo[i + 1 : end] <= tgt)[0]
            hit_stp = np.where(h[i + 1 : end] >= stp)[0]
            max_fwd = entry - lo[i + 1 : end].min()

        fwd_ret[i] = max_fwd / atr[i]
        ft = hit_tgt[0] if hit_tgt.size > 0 else None
        fs = hit_stp[0] if hit_stp.size > 0 else None

        if ft is not None and (fs is None or ft < fs):
            outcome[i] = 1
        elif fs is not None and (ft is None or fs < ft):
            outcome[i] = -1

    out["amd_label_outcome"] = outcome
    out["amd_label_forward_return_atr"] = fwd_ret
    return out


def add_amd_engine(
    df: pd.DataFrame,
    *,
    add_labels: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Full AMD pipeline: features → state machine → optional labels."""
    feature_keys = {
        "atr_pct_window",
        "accumulation_window",
        "overlap_window",
        "accumulation_min_streak",
        "atr_pct_low_threshold",
        "box_width_atr_max",
        "box_width_pct_max",
        "overlap_min",
        "efficiency_max",
        "min_touch_count_each_side",
        "sweep_tolerance_atr",
        "reclaim_min_frac_of_box",
        "displacement_mode",
        "min_distribution_followthrough_bars",
        "min_distribution_move_atr",
        "min_distribution_move_box_frac",
        "max_reentry_frac_of_box",
    }
    state_keys = {
        "manipulation_timeout_bars",
        "allow_unknown_state",
        "reset_to_accumulation_on_new_box",
        "accumulation_grace_bars",
        "max_distribution_stall",
        "min_distribution_move_atr",
        "min_distribution_move_box_frac",
        "min_distribution_followthrough_bars",
        "min_distribution_extension_atr",
    }
    label_keys = {"label_lookahead", "label_target_atr", "label_stop_box_frac"}

    feat_kw = {k: v for k, v in kwargs.items() if k in feature_keys}
    state_kw = {k: v for k, v in kwargs.items() if k in state_keys}
    label_kw = {k: v for k, v in kwargs.items() if k in label_keys}

    out = add_amd_features(df, **feat_kw)
    out = add_amd_state(out, **state_kw)
    if add_labels:
        out = add_amd_labels(out, **label_kw)
    return out
