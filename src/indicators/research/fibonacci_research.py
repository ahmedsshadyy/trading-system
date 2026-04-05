"""
research/fibonacci_research.py

Forward-looking research table for Fibonacci level structures.

CONTRACT: This module uses future bar data and MUST NEVER be imported in live
or production pipelines. Research only.

For each detected Fibonacci structure (bull + bear), this module computes:
- Per-level hit/hold/break rates at multiple horizons
- MFE/MAE from detect bar
- Final directional outcome
- Context features at detect time

The main outputs:
- ``build_fibonacci_research_table(df)`` → event-level research DataFrame
- ``summarize_fibonacci_research(table)`` → aggregate statistics dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc

EPS = 1e-12
# Minimum MAE denominator for excursion ratio (prevents astronomic values when
# price barely moves against the trade — 0.05 ATR floor is economically meaningful)
_EXCURSION_MAE_FLOOR_ATR = 0.05
_EXCURSION_RATIO_CAP = 20.0  # cap at 20:1 MFE/MAE

# Fib ratios tracked for hit statistics (core retracement levels only)
_TRACKED_RATIOS = (
    ("0236", 0.236),
    ("0382", 0.382),
    ("0500", 0.500),
    ("0618", 0.618),
    ("0786", 0.786),
)
_HORIZONS = (5, 10, 20, 40)
_MFE_MAE_HORIZONS = (5, 10, 20)
_OUTCOME_HORIZONS = (5, 10, 20)
_TOUCH_TOL_ATR = 0.15
_ATR_LENGTH = 14

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

FIB_RESEARCH_EVENT_COLUMNS = [
    "fib_r_event_id",
    "fib_r_detect_idx",
    "fib_r_detect_ts",
    "fib_r_direction",
    "fib_r_range_atr",
    "fib_r_detect_delay",
    "fib_r_quality_score",
]

FIB_RESEARCH_CONTEXT_COLUMNS = [
    "fib_r_trend_state_on_event",
    "fib_r_session_on_event",
    "fib_r_regime_on_event",
    "fib_r_adx_on_event",
    "fib_r_rsi_on_event",
    "fib_r_anchor_low",
    "fib_r_anchor_high",
    "fib_r_close_on_detect",
    "fib_r_atr_on_detect",
]

FIB_RESEARCH_LEVEL_COLUMNS = (
    [f"fib_r_l{lbl}_reached_h{h}" for lbl, _ in _TRACKED_RATIOS for h in _HORIZONS]
    + [f"fib_r_l{lbl}_held_h{h}" for lbl, _ in _TRACKED_RATIOS for h in _HORIZONS]
    + [f"fib_r_l{lbl}_broken_h{h}" for lbl, _ in _TRACKED_RATIOS for h in _HORIZONS]
)

FIB_RESEARCH_EXCURSION_COLUMNS = [
    *(f"fib_r_mfe_{h}_atr" for h in _MFE_MAE_HORIZONS),
    *(f"fib_r_mae_{h}_atr" for h in _MFE_MAE_HORIZONS),
    *(f"fib_r_excursion_ratio_{h}" for h in _MFE_MAE_HORIZONS),
]

FIB_RESEARCH_OUTCOME_COLUMNS = [f"fib_r_final_outcome_{h}" for h in _OUTCOME_HORIZONS]

FIB_RESEARCH_COLUMNS = (
    FIB_RESEARCH_EVENT_COLUMNS
    + FIB_RESEARCH_CONTEXT_COLUMNS
    + FIB_RESEARCH_LEVEL_COLUMNS
    + FIB_RESEARCH_EXCURSION_COLUMNS
    + FIB_RESEARCH_OUTCOME_COLUMNS
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _get_optional_col(df: pd.DataFrame, col: str, i: int) -> object:
    if col in df.columns:
        val = df.iloc[i][col]
        return None if pd.isna(val) else val
    return None


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _extract_fib_events(df: pd.DataFrame) -> list[dict]:
    """Extract all bull + bear fib structure events from df."""
    required = {"fib_bull_detect_flag", "fib_bear_detect_flag"}
    if not required.issubset(df.columns):
        return []

    bull_flag = (
        pd.to_numeric(df["fib_bull_detect_flag"], errors="coerce").fillna(0).to_numpy()
    )
    bear_flag = (
        pd.to_numeric(df["fib_bear_detect_flag"], errors="coerce").fillna(0).to_numpy()
    )

    events = []
    # flatnonzero avoids scanning all n rows in Python
    for direction, flag_arr, prefix in (
        (1, bull_flag, "bull"),
        (-1, bear_flag, "bear"),
    ):
        for i in np.flatnonzero(flag_arr == 1).tolist():
            row = df.iloc[i]
            event_id_val = row.get(f"fib_{prefix}_event_id", np.nan)
            if pd.isna(event_id_val):
                continue
            events.append(
                {
                    "detect_pos": i,
                    "direction": direction,
                    "prefix": prefix,
                    "event_id": float(event_id_val),
                    "detect_idx": i,
                    "detect_ts": row.get("timestamp", None),
                    "range_atr": float(row.get(f"fib_{prefix}_range_atr", np.nan)),
                    "detect_delay": float(
                        row.get(f"fib_{prefix}_detect_delay", np.nan)
                    ),
                    "quality_score": float(
                        row.get(f"fib_{prefix}_quality_score_on_detect", np.nan)
                    ),
                    "anchor_low": float(
                        row.get(
                            (
                                f"fib_{prefix}_anchor_a_price"
                                if direction == 1
                                else f"fib_{prefix}_anchor_a_price"
                            ),
                            np.nan,
                        )
                    ),
                    "anchor_high": float(
                        row.get(
                            (
                                f"fib_{prefix}_anchor_b_price"
                                if direction == 1
                                else f"fib_{prefix}_anchor_b_price"
                            ),
                            np.nan,
                        )
                    ),
                    # correct: anchor_low/high are geometry
                    "anc_low": (
                        float(row.get(f"fib_{prefix}_anchor_a_price", np.nan))
                        if direction == 1
                        else float(row.get(f"fib_{prefix}_anchor_b_price", np.nan))
                    ),
                    "anc_high": (
                        float(row.get(f"fib_{prefix}_anchor_b_price", np.nan))
                        if direction == 1
                        else float(row.get(f"fib_{prefix}_anchor_a_price", np.nan))
                    ),
                    "level_prices": {
                        lbl: float(row.get(f"fib_{prefix}_l{lbl}", np.nan))
                        for lbl, _ in _TRACKED_RATIOS
                    },
                }
            )
    return events


# ---------------------------------------------------------------------------
# Main research table builder
# ---------------------------------------------------------------------------


def build_fibonacci_research_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the event-level research table with forward-looking outcomes.

    Parameters
    ----------
    df:
        Full DataFrame with ``add_fibonacci_levels`` already applied.
        OHLC must be present.

    Returns
    -------
    pd.DataFrame
        One row per detected fib structure event, with research columns.
        Rows near the end of ``df`` will have partial/NaN forward data.
    """
    require_ohlc(df, caller="build_fibonacci_research_table")

    n = len(df)
    atr = get_atr_array(df, length=_ATR_LENGTH)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    events = _extract_fib_events(df)
    if not events:
        return pd.DataFrame(columns=FIB_RESEARCH_COLUMNS)

    # Pre-extract optional context columns as arrays — avoids df.iloc[i][col] per event
    _ctx_cols = {
        "fib_r_trend_state_on_event": "trend_state",
        "fib_r_session_on_event": "session",
        "fib_r_regime_on_event": "regime",
        "fib_r_adx_on_event": "adx_14",
        "fib_r_rsi_on_event": "rsi_14",
    }
    _ctx_arrays: dict[str, object] = {}
    for out_col, src_col in _ctx_cols.items():
        if src_col in df.columns:
            _ctx_arrays[out_col] = df[src_col].to_numpy()
        else:
            _ctx_arrays[out_col] = None

    _max_h = max(_HORIZONS)  # 40 — forward window cap

    rows = []
    for ev in events:
        pos = ev["detect_pos"]
        direction = ev["direction"]
        atr_on_detect = (
            float(atr[pos]) if np.isfinite(atr[pos]) and atr[pos] > 0 else 1.0
        )
        close_on_detect = close[pos]

        record: dict[str, object] = {
            "fib_r_event_id": ev["event_id"],
            "fib_r_detect_idx": pos,
            "fib_r_detect_ts": ev.get("detect_ts"),
            "fib_r_direction": direction,
            "fib_r_range_atr": ev["range_atr"],
            "fib_r_detect_delay": ev["detect_delay"],
            "fib_r_quality_score": ev["quality_score"],
        }

        # Context (causal — array lookup, no iloc)
        for out_col, arr in _ctx_arrays.items():
            if arr is not None:
                val = arr[pos]
                record[out_col] = (
                    None if (isinstance(val, float) and np.isnan(val)) else val
                )
            else:
                record[out_col] = None
        record["fib_r_anchor_low"] = ev["anc_low"]
        record["fib_r_anchor_high"] = ev["anc_high"]
        record["fib_r_close_on_detect"] = close_on_detect
        record["fib_r_atr_on_detect"] = atr_on_detect

        # Slice forward window ONCE per event (vectorized touch detection)
        fw_start = pos + 1
        fw_end = min(fw_start + _max_h, n)
        fw_len = fw_end - fw_start
        tol = _TOUCH_TOL_ATR * atr_on_detect

        if fw_len > 0:
            fw_high = high[fw_start:fw_end]
            fw_low = low[fw_start:fw_end]
            fw_close = close[fw_start:fw_end]
        else:
            fw_high = fw_low = fw_close = np.empty(0)

        for lbl, _ in _TRACKED_RATIOS:
            lp = ev["level_prices"].get(lbl, np.nan)

            if not np.isfinite(lp) or fw_len == 0:
                for h in _HORIZONS:
                    record[f"fib_r_l{lbl}_reached_h{h}"] = 0
                    record[f"fib_r_l{lbl}_held_h{h}"] = 0
                    record[f"fib_r_l{lbl}_broken_h{h}"] = 0
                continue

            # Vectorized mask — no Python loop over bars
            touched_mask = (fw_low <= lp + tol) & (fw_high >= lp - tol)
            any_touched = bool(touched_mask.any())

            if any_touched:
                first_touch_rel = int(np.argmax(touched_mask))
                touch_close_val = float(fw_close[first_touch_rel])
                if direction == 1:
                    first_held = 1 if touch_close_val >= lp else 0
                else:
                    first_held = 1 if touch_close_val <= lp else 0
                first_broken = 1 - first_held
            else:
                first_touch_rel = fw_len  # sentinel: outside any window

            for h in _HORIZONS:
                window = min(h, fw_len)
                if any_touched and first_touch_rel < window:
                    record[f"fib_r_l{lbl}_reached_h{h}"] = 1
                    record[f"fib_r_l{lbl}_held_h{h}"] = first_held
                    record[f"fib_r_l{lbl}_broken_h{h}"] = first_broken
                else:
                    record[f"fib_r_l{lbl}_reached_h{h}"] = 0
                    record[f"fib_r_l{lbl}_held_h{h}"] = 0
                    record[f"fib_r_l{lbl}_broken_h{h}"] = 0

        # MFE / MAE (already slicing numpy arrays — just cap per horizon)
        for h in _MFE_MAE_HORIZONS:
            end_idx = min(h, fw_len)
            if end_idx > 0:
                if direction == 1:
                    mfe = float(fw_high[:end_idx].max() - close_on_detect)
                    mae = float(close_on_detect - fw_low[:end_idx].min())
                else:
                    mfe = float(close_on_detect - fw_low[:end_idx].min())
                    mae = float(fw_high[:end_idx].max() - close_on_detect)
                mfe_atr = max(mfe, 0.0) / atr_on_detect
                mae_atr = max(mae, 0.0) / atr_on_detect
                # Floor MAE to prevent astronomic ratios when price barely moves back
                mae_floor = max(mae_atr, _EXCURSION_MAE_FLOOR_ATR)
                exc_ratio = min(mfe_atr / mae_floor, _EXCURSION_RATIO_CAP)
            else:
                mfe_atr = mae_atr = exc_ratio = np.nan

            record[f"fib_r_mfe_{h}_atr"] = mfe_atr
            record[f"fib_r_mae_{h}_atr"] = mae_atr
            record[f"fib_r_excursion_ratio_{h}"] = exc_ratio

        # Final outcome
        for h in _OUTCOME_HORIZONS:
            future_pos = pos + h
            if future_pos < n and np.isfinite(close[future_pos]):
                record[f"fib_r_final_outcome_{h}"] = float(
                    (close[future_pos] - close_on_detect) * direction / atr_on_detect
                )
            else:
                record[f"fib_r_final_outcome_{h}"] = np.nan

        rows.append(record)

    result = pd.DataFrame(rows, columns=FIB_RESEARCH_COLUMNS)
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_fibonacci_research(table: pd.DataFrame) -> dict:
    """Compute aggregate statistics from the fib research table.

    Returns a nested dict with per-direction, per-ratio, per-horizon hit rates,
    conditional hold rates, IC analysis, and quality decile breakdown.
    """
    if table.empty:
        return {}

    summary: dict = {}

    for direction, label in ((1, "bull"), (-1, "bear")):
        sub = table[table["fib_r_direction"] == direction]
        if sub.empty:
            summary[label] = {"count": 0}
            continue

        count = len(sub)
        dir_stats: dict = {"count": count}

        # Per-ratio hit/hold/break rates + conditional hold rate
        level_stats: dict = {}
        for lbl, ratio in _TRACKED_RATIOS:
            ratio_stats: dict = {}
            for h in _HORIZONS:
                reach_col = f"fib_r_l{lbl}_reached_h{h}"
                held_col = f"fib_r_l{lbl}_held_h{h}"
                broken_col = f"fib_r_l{lbl}_broken_h{h}"
                if reach_col in sub.columns:
                    n_reach = int(sub[reach_col].sum())
                    n_held = int(sub[held_col].sum())
                    n_broken = int(sub[broken_col].sum())
                    # Conditional hold rate: held / reached (the real Fib signal metric)
                    cond_hold = round(n_held / n_reach, 4) if n_reach > 0 else np.nan
                    ratio_stats[f"h{h}"] = {
                        "reached_count": n_reach,
                        "reached_pct": (
                            round(n_reach / count, 4) if count > 0 else np.nan
                        ),
                        "held_count": n_held,
                        "held_pct": round(n_held / count, 4) if count > 0 else np.nan,
                        "broken_count": n_broken,
                        "broken_pct": (
                            round(n_broken / count, 4) if count > 0 else np.nan
                        ),
                        "conditional_hold_rate": cond_hold,
                    }
            level_stats[f"l{lbl}"] = ratio_stats
        dir_stats["levels"] = level_stats

        # MFE/MAE
        excursion_stats: dict = {}
        for h in _MFE_MAE_HORIZONS:
            mfe_col = f"fib_r_mfe_{h}_atr"
            mae_col = f"fib_r_mae_{h}_atr"
            exc_col = f"fib_r_excursion_ratio_{h}"
            if mfe_col in sub.columns:
                excursion_stats[f"h{h}"] = {
                    "mean_mfe_atr": round(float(sub[mfe_col].mean(skipna=True)), 4),
                    "mean_mae_atr": round(float(sub[mae_col].mean(skipna=True)), 4),
                    "mean_excursion_ratio": round(
                        float(sub[exc_col].mean(skipna=True)), 4
                    ),
                    "median_mfe_atr": round(float(sub[mfe_col].median()), 4),
                    "median_mae_atr": round(float(sub[mae_col].median()), 4),
                }
        dir_stats["excursion"] = excursion_stats

        # Final outcome
        outcome_stats: dict = {}
        for h in _OUTCOME_HORIZONS:
            oc = f"fib_r_final_outcome_{h}"
            if oc in sub.columns:
                clean = sub[oc].dropna()
                if not clean.empty:
                    outcome_stats[f"h{h}"] = {
                        "mean": round(float(clean.mean()), 4),
                        "median": round(float(clean.median()), 4),
                        "win_rate": round(float((clean > 0).mean()), 4),
                        "count": int(len(clean)),
                    }
        dir_stats["outcome"] = outcome_stats

        # IC analysis: Spearman rank correlation of quality_score vs outcome
        # IC > 0.05 is economically meaningful; > 0.10 is strong for price signals
        ic_stats: dict = {}
        qs_col = "fib_r_quality_score"
        if qs_col in sub.columns:
            sub[qs_col]
            for h in _OUTCOME_HORIZONS:
                oc = f"fib_r_final_outcome_{h}"
                if oc in sub.columns:
                    valid = sub[[qs_col, oc]].dropna()
                    if len(valid) >= 10:
                        rho, pval = _scipy_stats.spearmanr(valid[qs_col], valid[oc])
                        ic_stats[f"h{h}"] = {
                            "spearman_ic": round(float(rho), 4),
                            "p_value": round(float(pval), 4),
                            "n": int(len(valid)),
                        }
        dir_stats["ic"] = ic_stats

        # Quality decile breakdown: does higher quality_score predict better outcomes?
        quality_decile_stats: dict = {}
        if qs_col in sub.columns and not sub[qs_col].dropna().empty:
            try:
                sub_copy = sub.copy()
                sub_copy["_qdecile"] = pd.qcut(
                    sub_copy[qs_col], q=10, labels=False, duplicates="drop"
                )
                for h in _OUTCOME_HORIZONS:
                    oc = f"fib_r_final_outcome_{h}"
                    if oc in sub_copy.columns:
                        grp = (
                            sub_copy.groupby("_qdecile")[oc]
                            .agg(["mean", "count"])
                            .dropna()
                        )
                        quality_decile_stats[f"h{h}"] = {
                            int(d): {
                                "mean_outcome": round(float(row["mean"]), 4),
                                "count": int(row["count"]),
                            }
                            for d, row in grp.iterrows()
                        }
            except Exception:
                pass
        dir_stats["quality_deciles"] = quality_decile_stats

        summary[label] = dir_stats

    return summary
