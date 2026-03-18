"""
smc/ob_mitigation.py

OB mitigation tracker — tracks whether recent OBs are still unmitigated.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
