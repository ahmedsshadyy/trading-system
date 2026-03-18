"""
smc/ifvg.py

Enhanced IFVG (Inverse FVG) classifier with state-machine logic.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.indicators._helpers.arrays import get_atr_array


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

    atr = get_atr_array(out, atr_length)

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
