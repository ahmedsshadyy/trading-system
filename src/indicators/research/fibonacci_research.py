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

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_ohlc

EPS = 1e-12

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

    n = len(df)
    bull_flag = (
        pd.to_numeric(df["fib_bull_detect_flag"], errors="coerce").fillna(0).to_numpy()
    )
    bear_flag = (
        pd.to_numeric(df["fib_bear_detect_flag"], errors="coerce").fillna(0).to_numpy()
    )

    events = []
    for i in range(n):
        for direction, flag_arr, prefix in (
            (1, bull_flag, "bull"),
            (-1, bear_flag, "bear"),
        ):
            if flag_arr[i] != 1:
                continue
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

    rows = []
    for ev in events:
        pos = ev["detect_pos"]
        direction = ev["direction"]
        prefix = ev["prefix"]
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

        # Context (causal — captured at detect bar)
        record["fib_r_trend_state_on_event"] = _get_optional_col(df, "trend_state", pos)
        record["fib_r_session_on_event"] = _get_optional_col(df, "session", pos)
        record["fib_r_regime_on_event"] = _get_optional_col(df, "regime", pos)
        record["fib_r_adx_on_event"] = _get_optional_col(df, "adx_14", pos)
        record["fib_r_rsi_on_event"] = _get_optional_col(df, "rsi_14", pos)
        record["fib_r_anchor_low"] = ev["anc_low"]
        record["fib_r_anchor_high"] = ev["anc_high"]
        record["fib_r_close_on_detect"] = close_on_detect
        record["fib_r_atr_on_detect"] = atr_on_detect

        # Level hit/hold/break at each horizon
        for lbl, ratio in _TRACKED_RATIOS:
            lp = ev["level_prices"].get(lbl, np.nan)
            tol = _TOUCH_TOL_ATR * atr_on_detect

            for h in _HORIZONS:
                end = min(pos + h + 1, n)
                future_slice = range(pos + 1, end)

                reached = 0
                held = 0
                broken = 0

                if np.isfinite(lp) and len(future_slice) > 0:
                    for j in future_slice:
                        # Touch: wick reaches within tolerance of level
                        if low[j] <= lp + tol and high[j] >= lp - tol:
                            reached = 1
                            # Held: close stays on "correct" side
                            if direction == 1:
                                # Bull: level is support; correct side = close above
                                if close[j] >= lp:
                                    held = 1
                                else:
                                    broken = 1
                            else:
                                # Bear: level is resistance; correct side = close below
                                if close[j] <= lp:
                                    held = 1
                                else:
                                    broken = 1
                            break  # first touch only

                record[f"fib_r_l{lbl}_reached_h{h}"] = reached
                record[f"fib_r_l{lbl}_held_h{h}"] = held
                record[f"fib_r_l{lbl}_broken_h{h}"] = broken

        # MFE / MAE
        for h in _MFE_MAE_HORIZONS:
            end = min(pos + h + 1, n)
            future_slice = list(range(pos + 1, end))
            if future_slice:
                if direction == 1:  # bull: favorable = higher
                    mfe = float(np.max(high[future_slice]) - close_on_detect)
                    mae = float(close_on_detect - np.min(low[future_slice]))
                else:
                    mfe = float(close_on_detect - np.min(low[future_slice]))
                    mae = float(np.max(high[future_slice]) - close_on_detect)
                mfe_atr = mfe / atr_on_detect
                mae_atr = mae / atr_on_detect
                exc_ratio = _safe_ratio(mfe_atr, mae_atr + EPS)
            else:
                mfe_atr = np.nan
                mae_atr = np.nan
                exc_ratio = np.nan

            record[f"fib_r_mfe_{h}_atr"] = mfe_atr
            record[f"fib_r_mae_{h}_atr"] = mae_atr
            record[f"fib_r_excursion_ratio_{h}"] = exc_ratio

        # Final outcome
        for h in _OUTCOME_HORIZONS:
            future_pos = pos + h
            if future_pos < n and np.isfinite(close[future_pos]):
                outcome = (
                    (close[future_pos] - close_on_detect) * direction / atr_on_detect
                )
                record[f"fib_r_final_outcome_{h}"] = float(outcome)
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

    Returns a nested dict with per-direction, per-ratio, per-horizon hit rates
    and mean excursion stats. Useful for calibration and reporting.
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

        # Per-ratio hit/hold/break rates
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

        summary[label] = dir_stats

    return summary
