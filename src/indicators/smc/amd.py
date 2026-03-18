"""
smc/amd.py

AMD (Accumulation → Manipulation → Distribution) engine.

Includes feature computation, causal state machine, retrospective labels,
and the unified pipeline.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array

# AMD state constants
AMD_UNKNOWN = -1
AMD_ACCUMULATION = 0
AMD_MANIPULATION = 1
AMD_DISTRIBUTION = 2


def _amd_rolling_rank_pct(arr: np.ndarray) -> float:
    if arr.size == 0 or np.isnan(arr[-1]):
        return np.nan
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan
    return float((valid <= valid[-1]).mean() * 100.0)


def _amd_safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numer.index, dtype=float)
    valid = denom.notna() & (denom != 0)
    out.loc[valid] = numer.loc[valid] / denom.loc[valid]
    return out


def _amd_rolling_overlap(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    overlap = (np.minimum(high, prev_h) - np.maximum(low, prev_l)).clip(lower=0.0)
    union = np.maximum(high, prev_h) - np.minimum(low, prev_l)
    return (
        _amd_safe_div(overlap, union).rolling(window=window, min_periods=window).mean()
    )


def _amd_rolling_efficiency(close: pd.Series, window: int) -> pd.Series:
    net = (close - close.shift(window - 1)).abs()
    gross = close.diff().abs().rolling(window=window, min_periods=window).sum()
    return _amd_safe_div(net, gross)


def add_amd_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Compute causal AMD feature columns. Live-safe (trailing-only windows)."""
    # Extract kwargs with defaults
    atr_pct_window = kwargs.get("atr_pct_window", 50)
    accumulation_window = kwargs.get("accumulation_window", 20)
    overlap_window = kwargs.get("overlap_window", 10)
    accumulation_min_streak = kwargs.get("accumulation_min_streak", 8)
    atr_pct_low_threshold = kwargs.get("atr_pct_low_threshold", 45.0)
    box_width_atr_max = kwargs.get("box_width_atr_max", 12.0)
    box_width_pct_max = kwargs.get("box_width_pct_max", 0.040)
    overlap_min = kwargs.get("overlap_min", 0.40)
    efficiency_max = kwargs.get("efficiency_max", 0.35)
    min_touch_count_each_side = kwargs.get("min_touch_count_each_side", 1)
    sweep_tolerance_atr = kwargs.get("sweep_tolerance_atr", 0.15)
    reclaim_min_frac_of_box = kwargs.get("reclaim_min_frac_of_box", 0.10)
    displacement_mode = kwargs.get("displacement_mode", "break_only")
    min_distribution_followthrough_bars = kwargs.get(
        "min_distribution_followthrough_bars", 2
    )
    min_distribution_move_atr = kwargs.get("min_distribution_move_atr", 0.35)
    min_distribution_move_box_frac = kwargs.get("min_distribution_move_box_frac", 0.30)
    max_reentry_frac_of_box = kwargs.get("max_reentry_frac_of_box", 0.20)

    out = df.copy()
    h = out["high"].astype(float)
    lo = out["low"].astype(float)
    c = out["close"].astype(float)
    o = out["open"].astype(float)

    atr = pd.Series(get_atr_array(out), index=out.index, dtype=float)

    atr_pct = atr.rolling(atr_pct_window, min_periods=atr_pct_window).apply(
        _amd_rolling_rank_pct, raw=True
    )

    box_high = h.rolling(accumulation_window, min_periods=accumulation_window).max()
    box_low = lo.rolling(accumulation_window, min_periods=accumulation_window).min()
    box_mid = (box_high + box_low) / 2.0
    box_width = (box_high - box_low).astype(float)
    box_width_atr_val = _amd_safe_div(box_width, atr)
    box_width_pct = _amd_safe_div(box_width, c.abs())

    overlap_score = _amd_rolling_overlap(h, lo, overlap_window)
    efficiency = _amd_rolling_efficiency(c, accumulation_window)

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

    low_atr_flag = atr_pct <= atr_pct_low_threshold
    narrow_box_flag = (box_width_atr_val <= box_width_atr_max) & (
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

    prior_bh = box_high.shift(1)
    prior_bl = box_low.shift(1)
    prior_bw = (prior_bh - prior_bl).astype(float)
    sweep_tol_val = atr * sweep_tolerance_atr

    break_up = h > prior_bh
    break_down = lo < prior_bl
    sweep_up = break_up & (h <= (prior_bh + sweep_tol_val))
    sweep_down = break_down & (lo >= (prior_bl - sweep_tol_val))

    reclaim_thresh = prior_bw * reclaim_min_frac_of_box
    reclaim_bull = sweep_down & (c >= (prior_bl + reclaim_thresh))
    reclaim_bear = sweep_up & (c <= (prior_bh - reclaim_thresh))

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


def add_amd_state(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Causal AMD state machine. Assigns phase per bar."""
    manipulation_timeout_bars = kwargs.get("manipulation_timeout_bars", 8)
    allow_unknown_state = kwargs.get("allow_unknown_state", True)
    reset_to_accumulation_on_new_box = kwargs.get(
        "reset_to_accumulation_on_new_box", True
    )
    accumulation_grace_bars = kwargs.get("accumulation_grace_bars", 2)
    max_distribution_stall = kwargs.get("max_distribution_stall", 4)
    min_distribution_move_atr = kwargs.get("min_distribution_move_atr", 0.75)
    min_distribution_move_box_frac = kwargs.get("min_distribution_move_box_frac", 0.60)
    min_distribution_followthrough_bars = kwargs.get(
        "min_distribution_followthrough_bars", 4
    )
    min_distribution_extension_atr = kwargs.get("min_distribution_extension_atr", 0.10)

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
    atr_v = np.asarray(get_atr_array(out), dtype=float)

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
    frozen_bh = np.nan
    frozen_bl = np.nan
    frozen_bm = np.nan

    for i in range(n):
        acc = acc_active[i] == 1
        mb = manip_bull[i] == 1
        ms = manip_bear[i] == 1

        re = (
            np.isfinite(frozen_bl)
            and np.isfinite(frozen_bh)
            and frozen_bl <= close_v[i] <= frozen_bh
        )

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

            if re or dist_stall > max_distribution_stall or acc:
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


def add_amd_labels(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Retrospective AMD outcome labels. NOT live-safe — uses future bars."""
    label_lookahead = kwargs.get("label_lookahead", 10)
    label_target_atr = kwargs.get("label_target_atr", 1.5)
    label_stop_box_frac = kwargs.get("label_stop_box_frac", 0.50)

    needed = ["amd_phase", "amd_direction", "amd_active_box_high", "amd_active_box_low"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            "Run add_amd_features + add_amd_state first. Missing: " + ", ".join(missing)
        )

    out = df.copy()
    n = len(out)
    atr = np.asarray(get_atr_array(out), dtype=float)

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
    df: pd.DataFrame, *, add_labels: bool = False, **kwargs
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
