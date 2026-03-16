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

# ============================================================================
# Shared Internal Helpers
# ============================================================================


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )


def _atr_rma(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14
) -> np.ndarray:
    """Wilder-style ATR (self-reliant, no external deps)."""
    tr = _true_range(high, low, close).astype(float)
    out = np.full(len(tr), np.nan, dtype=float)
    if len(tr) < length:
        csum = np.cumsum(tr)
        out[:] = csum / np.arange(1, len(tr) + 1)
        return out
    out[length - 1] = np.nanmean(tr[:length])
    alpha = 1.0 / length
    for i in range(length, len(tr)):
        out[i] = out[i - 1] + alpha * (tr[i] - out[i - 1])
    if length > 1:
        csum = np.cumsum(tr[: length - 1])
        out[: length - 1] = csum / np.arange(1, length)
    return out


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


def _ensure_atr(out: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Return ATR array, computing if not present."""
    if "atr_14" in out.columns:
        return out["atr_14"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    atr = _atr_rma(h, lo, c, length=length)
    out["atr_14"] = atr
    return atr


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
        if "last_swing_high_age" in out.columns:
            sh_age = out["last_swing_high_age"].to_numpy(dtype=float)
        else:
            sh_age = np.full(n, np.nan)

        if "last_swing_low_age" in out.columns:
            sl_age = out["last_swing_low_age"].to_numpy(dtype=float)
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
# Equal Highs / Lows Detector
# ============================================================================


def add_equal_hl(df: pd.DataFrame, atr_tolerance: float = 0.1) -> pd.DataFrame:
    """Detect clusters of equal highs or equal lows (liquidity pools).

    Columns: ``equal_highs``, ``equal_lows``, ``equal_highs_count``, ``equal_lows_count``.
    """
    out = df.copy()
    n = len(out)
    sh_price = out.get("swing_high_price", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )
    sl_price = out.get("swing_low_price", pd.Series(np.full(n, np.nan))).values.astype(
        float
    )

    atr = _ensure_atr(out)

    eq_h = np.zeros(n, dtype=np.int8)
    eq_l = np.zeros(n, dtype=np.int8)
    eq_h_cnt = np.zeros(n, dtype=np.int16)
    eq_l_cnt = np.zeros(n, dtype=np.int16)

    recent_sh = []
    recent_sl = []

    for i in range(n):
        tol = atr_tolerance * atr[i] if not np.isnan(atr[i]) else 0

        if not np.isnan(sh_price[i]):
            cnt = sum(1 for p, _ in recent_sh if abs(p - sh_price[i]) <= tol)
            if cnt > 0:
                eq_h[i] = 1
                eq_h_cnt[i] = cnt + 1
            recent_sh.append((sh_price[i], i))
            if len(recent_sh) > 20:
                recent_sh.pop(0)

        if not np.isnan(sl_price[i]):
            cnt = sum(1 for p, _ in recent_sl if abs(p - sl_price[i]) <= tol)
            if cnt > 0:
                eq_l[i] = 1
                eq_l_cnt[i] = cnt + 1
            recent_sl.append((sl_price[i], i))
            if len(recent_sl) > 20:
                recent_sl.pop(0)

    out["equal_highs"] = eq_h
    out["equal_lows"] = eq_l
    out["equal_highs_count"] = eq_h_cnt
    out["equal_lows_count"] = eq_l_cnt
    return out


# ============================================================================
# Displacement Candle Detector
# ============================================================================


def add_displacement_candle(
    df: pd.DataFrame, body_atr_mult: float = 1.5
) -> pd.DataFrame:
    """Flag candles with body ≥ ``body_atr_mult × ATR``.

    Columns: ``displacement_candle``, ``displacement_body_atr``, ``displacement_close_extreme``.
    """
    out = df.copy()
    atr = _ensure_atr(out)

    body = (out["close"] - out["open"]).abs().astype(float)
    atr_s = pd.Series(atr, index=out.index)
    rng = (out["high"] - out["low"]).astype(float)

    ratio = np.where(atr_s > 0, body / atr_s, 0.0)
    out["displacement_candle"] = (ratio >= body_atr_mult).astype(int)
    out["displacement_body_atr"] = ratio

    bull = out["close"].values >= out["open"].values
    dist = np.where(
        bull,
        out["high"].values - out["close"].values,
        out["close"].values - out["low"].values,
    ).astype(float)
    out["displacement_close_extreme"] = np.where(
        rng > 0, (dist / rng < 0.2).astype(int), 0
    )
    return out


# ============================================================================
# AMD Phase Classifier
# ============================================================================


def add_amd_phase(df: pd.DataFrame, accumulation_candles: int = 10) -> pd.DataFrame:
    """Classify market phase: Accumulation(0) / Manipulation(1) / Distribution(2).

    Columns: ``amd_phase``.
    """
    out = df.copy()
    n = len(out)

    atr = _ensure_atr(out)
    if "atr_pct_50" not in out.columns:
        atr_s = pd.Series(atr, index=out.index)
        out["atr_pct_50"] = atr_s.rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

    atr_pct = out["atr_pct_50"].values.astype(float)
    disp = out.get("displacement_candle", pd.Series(np.zeros(n))).values

    phase = np.zeros(n, dtype=np.int8)
    low_atr_streak = 0

    for i in range(n):
        if np.isnan(atr_pct[i]):
            continue
        if atr_pct[i] < 50:
            low_atr_streak += 1
        else:
            low_atr_streak = 0

        if low_atr_streak >= accumulation_candles:
            phase[i] = 0
        elif disp[i] == 1 and low_atr_streak > 0:
            phase[i] = 1
            low_atr_streak = 0
        elif atr_pct[i] >= 50:
            phase[i] = 2
        else:
            phase[i] = 0

    out["amd_phase"] = phase
    return out
