"""
smc/equal_hl.py

Enhanced Equal Highs / Equal Lows detector with clustering.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.pivots import pivot_high, pivot_low


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
    level_mode: str = "median",
    max_cluster_width_atr: float | None = None,
    max_cluster_span: int | None = None,
    invalidate_on_sweep: bool = False,
    sweep_tolerance_atr: float = 0.0,
    keep_last_n_clusters: int = 20,
) -> pd.DataFrame:
    """Enhanced Equal Highs / Equal Lows detector.

    Backward-compatible columns: equal_highs, equal_lows, equal_highs_count, equal_lows_count
    Added columns: equal_highs_level, equal_lows_level, etc.
    """

    out = df.copy()
    n = len(out)

    req = {"high", "low", "close"}
    missing = req - set(out.columns)
    if missing:
        raise ValueError(f"add_equal_hl: missing required columns: {sorted(missing)}")

    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    atr = get_atr_array(out, atr_length)

    # Swing source
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
        ph = pivot_high(h, left=swing_left, right=swing_right)
        pl = pivot_low(lo, left=swing_left, right=swing_right)

        sh_flag = ph.astype(np.int8)
        sl_flag = pl.astype(np.int8)

        sh_price = np.where(sh_flag == 1, h, np.nan)
        sl_price = np.where(sl_flag == 1, lo, np.nan)

    # Outputs
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
        return float(np.median(arr))

    def _cluster_width(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) == 0:
            return np.nan
        return float(np.max(arr) - np.min(arr))

    def _cluster_score(count, width_atr, span):
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
        cl["age"] = idx - cl["indices"][-1]
        cl["score"] = _cluster_score(
            cl["count"],
            0.0 if np.isnan(cl["width_atr"]) else cl["width_atr"],
            cl["span"],
        )

    def _prune_clusters(clusters):
        if len(clusters) <= keep_last_n_clusters:
            return clusters
        clusters = sorted(
            clusters,
            key=lambda z: (z["active"], z["last_idx"]),
            reverse=True,
        )
        kept = clusters[:keep_last_n_clusters]
        kept = sorted(kept, key=lambda z: z["id"])
        return kept

    for i in range(n):
        atr_i = atr[i] if np.isfinite(atr[i]) else np.nan
        sweep_tol = (
            sweep_tolerance_atr * atr_i if np.isfinite(atr_i) and atr_i > 0 else 0.0
        )

        if invalidate_on_sweep:
            for cl in high_clusters:
                if not cl["active"] or not np.isfinite(cl["level"]):
                    continue
                if h[i] > cl["level"] + sweep_tol:
                    cl["active"] = False
                    cl["swept_idx"] = i

            for cl in low_clusters:
                if not cl["active"] or not np.isfinite(cl["level"]):
                    continue
                if lo[i] < cl["level"] - sweep_tol:
                    cl["active"] = False
                    cl["swept_idx"] = i

        # Process new swing high
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            price = sh_price[i]
            high_clusters = [
                cl for cl in high_clusters if (i - cl["last_idx"] <= lookback_swings)
            ]
            match_idx = _match_cluster(price, atr_i, high_clusters, side="high")

            if match_idx is None:
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

        # Process new swing low
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

    # Backfill current active cluster state onto non-swing bars
    latest_active_high = None
    latest_active_low = None

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
