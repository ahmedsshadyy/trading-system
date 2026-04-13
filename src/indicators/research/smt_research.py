from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SMT_RESEARCH_HORIZONS = (1, 2, 3, 5, 10, 20)

SMT_RESEARCH_COLUMNS = [
    "smt_event_id",
    "smt_pair_id",
    "smt_partner",
    "smt_detect_idx",
    "smt_detect_ts",
    "smt_direction",
    "smt_divergence_type",
    "smt_score",
    "smt_expected_relation",
    "smt_close_on_detect",
    "smt_atr_on_detect",
    *(f"smt_hold_{h}" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
    *(f"smt_failed_{h}" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
    *(f"smt_reversal_{h}" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
    *(f"smt_mfe_{h}_atr" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
    *(f"smt_mae_{h}_atr" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
    *(f"smt_final_outcome_{h}" for h in DEFAULT_SMT_RESEARCH_HORIZONS),
]


def _partner_columns(df: pd.DataFrame) -> list[str]:
    prefixes = []
    for column in df.columns:
        if not column.startswith("xasset_smt_") or not column.endswith("_dir"):
            continue
        prefixes.append(column[len("xasset_smt_") : -len("_dir")])
    return sorted(set(prefixes))


def build_smt_research_table(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_SMT_RESEARCH_HORIZONS,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SMT_RESEARCH_COLUMNS)

    partner_tokens = _partner_columns(df)
    rows: list[dict[str, object]] = []
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    atr = (
        pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
        if "atr_14" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )

    event_id = 1
    for partner in partner_tokens:
        dir_col = f"xasset_smt_{partner}_dir"
        score_col = f"xasset_smt_{partner}_score"
        relation_col = f"xasset_smt_{partner}_expected_relation"
        if dir_col not in df.columns or score_col not in df.columns:
            continue
        directions = (
            pd.to_numeric(df[dir_col], errors="coerce").fillna(0).to_numpy(dtype=int)
        )
        scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
        relations = (
            pd.to_numeric(df[relation_col], errors="coerce").to_numpy(dtype=float)
            if relation_col in df.columns
            else np.full(len(df), np.nan, dtype=float)
        )
        for idx in np.flatnonzero(directions != 0):
            direction = int(directions[idx])
            detect_close = close[idx]
            detect_atr = atr[idx]
            row: dict[str, object] = {
                "smt_event_id": event_id,
                "smt_pair_id": f"primary__{partner}",
                "smt_partner": partner,
                "smt_detect_idx": int(idx),
                "smt_detect_ts": timestamps.iloc[idx],
                "smt_direction": direction,
                "smt_divergence_type": "bullish" if direction > 0 else "bearish",
                "smt_score": scores[idx],
                "smt_expected_relation": relations[idx],
                "smt_close_on_detect": detect_close,
                "smt_atr_on_detect": detect_atr,
            }
            for horizon in horizons:
                end = min(idx + horizon, len(df) - 1)
                window_high = high[idx + 1 : end + 1]
                window_low = low[idx + 1 : end + 1]
                if len(window_high) == 0:
                    row[f"smt_hold_{horizon}"] = False
                    row[f"smt_failed_{horizon}"] = False
                    row[f"smt_reversal_{horizon}"] = False
                    row[f"smt_mfe_{horizon}_atr"] = np.nan
                    row[f"smt_mae_{horizon}_atr"] = np.nan
                    row[f"smt_final_outcome_{horizon}"] = "insufficient"
                    continue

                if direction > 0:
                    mfe = np.nanmax(window_high) - detect_close
                    mae = detect_close - np.nanmin(window_low)
                    hold_flag = close[end] > detect_close
                    reversal_flag = np.nanmin(window_low) < detect_close
                else:
                    mfe = detect_close - np.nanmin(window_low)
                    mae = np.nanmax(window_high) - detect_close
                    hold_flag = close[end] < detect_close
                    reversal_flag = np.nanmax(window_high) > detect_close
                failed_flag = bool(reversal_flag and not hold_flag)
                row[f"smt_hold_{horizon}"] = bool(hold_flag)
                row[f"smt_failed_{horizon}"] = bool(failed_flag)
                row[f"smt_reversal_{horizon}"] = bool(reversal_flag)
                if np.isfinite(detect_atr) and detect_atr > 0:
                    row[f"smt_mfe_{horizon}_atr"] = float(max(mfe, 0.0) / detect_atr)
                    row[f"smt_mae_{horizon}_atr"] = float(max(mae, 0.0) / detect_atr)
                else:
                    row[f"smt_mfe_{horizon}_atr"] = np.nan
                    row[f"smt_mae_{horizon}_atr"] = np.nan
                if hold_flag and not reversal_flag:
                    outcome = "held"
                elif reversal_flag and not hold_flag:
                    outcome = "reversed"
                elif failed_flag:
                    outcome = "failed"
                else:
                    outcome = "mixed"
                row[f"smt_final_outcome_{horizon}"] = outcome
            rows.append(row)
            event_id += 1

    return pd.DataFrame(rows, columns=SMT_RESEARCH_COLUMNS)


def summarize_smt_research(research: pd.DataFrame) -> dict[str, object]:
    if research.empty:
        return {
            "event_count": 0,
            "pairs": {},
        }

    summary: dict[str, object] = {
        "event_count": int(len(research)),
        "pairs": {},
    }
    for pair_id, group in research.groupby("smt_pair_id"):
        summary["pairs"][pair_id] = {
            "events": int(len(group)),
            "mean_score": float(group["smt_score"].mean()),
            "bullish_pct": float((group["smt_direction"] > 0).mean()),
        }
    return summary
