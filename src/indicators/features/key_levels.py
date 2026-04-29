"""
features/key_levels.py

Structure-backed key level layer built on top of causal S/R zones.

This layer is intentionally distinct from base ``sr_levels``:
- ``sr_levels`` answers: where are the causal support/resistance zones?
- ``key_levels`` answers: which current S/R zones look important enough that
  a break or reclaim should matter structurally?

The layer stays live-safe. It uses only current / trailing S/R geometry and any
already-computed structure columns that exist on the input frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.features.bos_context import _series_or_default
from src.indicators.foundation.sr_levels import (
    SR_FAMILY_DAY,
    SR_FAMILY_EQHL,
    SR_FAMILY_SESSION,
    SR_FAMILY_SWING,
    SR_FAMILY_VP,
    SR_FAMILY_WEEK,
    SR_SIDE_RESISTANCE,
    SR_SIDE_SUPPORT,
    SRLevel,
    build_sr_level_registry,
    project_sr_context,
)
from src.indicators.foundation.sr_range_proxy import add_sr_range_proxy

__all__ = ["add_key_levels"]

KEY_LEVEL_STRUCTURE_LOOKBACK = 6

FAMILY_BONUS = {
    SR_FAMILY_EQHL: 0.14,
    SR_FAMILY_SWING: 0.10,
    SR_FAMILY_VP: 0.04,
    SR_FAMILY_SESSION: 0.00,
    SR_FAMILY_DAY: -0.02,
    SR_FAMILY_WEEK: -0.05,
}

WIDTH_BUCKET_BONUS = {
    "<=0.30": 0.05,
    "0.30-0.45": -0.10,
    ">0.45": 0.03,
}

AGE_BUCKET_PENALTY = {
    "0-10": 0.00,
    "11-30": 0.01,
    "31-60": 0.03,
    "61+": 0.06,
}


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))


def _width_bucket(value: float) -> str:
    if not np.isfinite(value) or value <= 0.30:
        return "<=0.30"
    if value <= 0.45:
        return "0.30-0.45"
    return ">0.45"


def _age_bucket(value: int) -> str:
    if value <= 10:
        return "0-10"
    if value <= 30:
        return "11-30"
    if value <= 60:
        return "31-60"
    return "61+"


def _recent_flag(values: np.ndarray, i: int, lookback: int) -> float:
    lo = max(0, i - lookback + 1)
    window = values[lo : i + 1]
    if len(window) == 0:
        return 0.0
    return float(np.nanmax(window)) if np.isfinite(window).any() else 0.0


def _trend_alignment(side: int, effective_trend: float, trend_bias: float) -> float:
    trend = int(effective_trend) if np.isfinite(effective_trend) else 0
    bias = int(trend_bias) if np.isfinite(trend_bias) else 0
    desired = 1 if side == SR_SIDE_SUPPORT else -1
    if trend == desired:
        return 1.0
    if bias == desired:
        return 0.65
    if trend == 0 and bias == 0:
        return 0.50
    return 0.0


def _zone_payload(
    level: SRLevel,
    *,
    selection_source: str,
    row_mid: float,
    row_score: float,
) -> dict[str, object]:
    return {
        "zone_id": int(level.level_id),
        "selection_source": selection_source,
        "mid": float(row_mid if np.isfinite(row_mid) else level.level_price),
        "low": float(level.zone_low),
        "high": float(level.zone_high),
        "base_score": float(
            row_score if np.isfinite(row_score) else level.level_strength
        ),
        "width_atr": float(level.zone_width_atr),
        "best_source_family": str(level.best_source_family or level.source_family),
        "age_bars": int(level.age_bars),
        "reclaim_count": int(level.reclaim_count),
        "clean_touch_count": int(level.clean_touch_count),
        "weak_touch_count": int(level.weak_touch_count),
        "anchor_count": int(level.anchor_count),
        "family_count": int(level.family_count),
    }


def _candidate_map_for_row(
    registry: dict[int, SRLevel],
    row: pd.Series,
    *,
    side: int,
) -> dict[int, dict[str, object]]:
    candidates: dict[int, dict[str, object]] = {}

    if side == SR_SIDE_SUPPORT:
        primary_id = row.get("primary_support_zone_id")
        nearest_id = row.get("nearest_support_zone_id")
        primary_mid = row.get("primary_support_zone_mid")
        nearest_mid = row.get("nearest_support_price")
        primary_score = row.get("primary_support_zone_score")
        nearest_score = row.get("nearest_support_strength")
    else:
        primary_id = row.get("primary_resistance_zone_id")
        nearest_id = row.get("nearest_resistance_zone_id")
        primary_mid = row.get("primary_resistance_zone_mid")
        nearest_mid = row.get("nearest_resistance_price")
        primary_score = row.get("primary_resistance_zone_score")
        nearest_score = row.get("nearest_resistance_strength")

    for zone_id, source, mid, score in (
        (primary_id, "primary", primary_mid, primary_score),
        (nearest_id, "nearest", nearest_mid, nearest_score),
    ):
        if not np.isfinite(zone_id):
            continue
        level = registry.get(int(zone_id))
        if level is None:
            continue
        payload = _zone_payload(
            level, selection_source=source, row_mid=mid, row_score=score
        )
        if (
            int(zone_id) not in candidates
            or payload["base_score"] > candidates[int(zone_id)]["base_score"]
        ):
            candidates[int(zone_id)] = payload
    return candidates


def add_key_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add structure-backed key level projections."""
    if df is None:
        raise ValueError("add_key_levels: df must not be None")
    out = df.copy()
    if out.empty:
        return out
    for col in ("high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"add_key_levels: missing required column '{col}'")

    registry = build_sr_level_registry(out)
    out = project_sr_context(out, registry)
    out = add_sr_range_proxy(out)

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)

    effective_trend = _series_or_default(
        out, "effective_trend_state", "trend_state", dtype=float
    )
    trend_bias = _series_or_default(out, "trend_bias_state", dtype=float)
    bos_bull = _series_or_default(out, "bos_bull", default=0, dtype=float)
    bos_bear = _series_or_default(out, "bos_bear", default=0, dtype=float)
    choch_bull = _series_or_default(out, "choch_bull", default=0, dtype=float)
    choch_bear = _series_or_default(out, "choch_bear", default=0, dtype=float)
    sweep_high = _series_or_default(
        out, "sweep_high_confirm_flag", "sweep_high_detect_flag", default=0, dtype=float
    )
    sweep_low = _series_or_default(
        out, "sweep_low_confirm_flag", "sweep_low_detect_flag", default=0, dtype=float
    )
    eqh_active = _series_or_default(
        out, "eqh_active", "equal_highs_active", default=0, dtype=float
    )
    eql_active = _series_or_default(
        out, "eql_active", "equal_lows_active", default=0, dtype=float
    )
    range_active = _series_or_default(
        out, "sr_range_proxy_active", default=0, dtype=float
    )
    range_sup_id = _series_or_default(
        out, "sr_range_proxy_support_zone_id", dtype=float
    )
    range_res_id = _series_or_default(
        out, "sr_range_proxy_resistance_zone_id", dtype=float
    )

    n = len(out)
    obj_nan = np.full(n, None, dtype=object)
    for prefix in ("key_support", "key_resistance"):
        out[f"{prefix}_zone_id"] = np.nan
        out[f"{prefix}_zone_low"] = np.nan
        out[f"{prefix}_zone_high"] = np.nan
        out[f"{prefix}_zone_mid"] = np.nan
        out[f"{prefix}_score"] = np.nan
        out[f"{prefix}_break_importance_score"] = np.nan
        out[f"{prefix}_best_source_family"] = obj_nan.copy()
        out[f"{prefix}_selection_source"] = obj_nan.copy()
    out["inside_key_support_zone_flag"] = np.zeros(n, dtype=np.int8)
    out["inside_key_resistance_zone_flag"] = np.zeros(n, dtype=np.int8)

    for i in range(n):
        row = out.iloc[i]
        for side, prefix in (
            (SR_SIDE_SUPPORT, "key_support"),
            (SR_SIDE_RESISTANCE, "key_resistance"),
        ):
            candidates = _candidate_map_for_row(registry, row, side=side)
            if not candidates:
                continue

            chosen_payload: dict[str, object] | None = None
            chosen_score = -np.inf
            chosen_break_importance = -np.inf

            for payload in candidates.values():
                family = str(payload["best_source_family"])
                width_bonus = WIDTH_BUCKET_BONUS[
                    _width_bucket(float(payload["width_atr"]))
                ]
                age_penalty = AGE_BUCKET_PENALTY[_age_bucket(int(payload["age_bars"]))]

                clean_touch_count = int(payload["clean_touch_count"])
                weak_touch_count = int(payload["weak_touch_count"])
                touch_total = max(clean_touch_count + weak_touch_count, 1)
                touch_quality = 0.05 * (clean_touch_count / touch_total) - 0.06 * (
                    weak_touch_count / touch_total
                )
                reclaim_bonus = 0.04 * min(int(payload["reclaim_count"]), 2)

                alignment = _trend_alignment(side, effective_trend[i], trend_bias[i])
                trend_bonus = (
                    0.05
                    if alignment >= 0.95
                    else (0.02 if alignment >= 0.60 else -0.03)
                )

                if side == SR_SIDE_SUPPORT:
                    structure_bonus = (
                        0.05 * _recent_flag(sweep_low, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.04
                        * _recent_flag(choch_bull, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.03 * _recent_flag(bos_bull, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.04 * float(eql_active[i] > 0)
                    )
                    range_bonus = (
                        0.04
                        if int(range_active[i]) == 1
                        and np.isfinite(range_sup_id[i])
                        and int(range_sup_id[i]) == int(payload["zone_id"])
                        else 0.0
                    )
                else:
                    structure_bonus = (
                        0.05 * _recent_flag(sweep_high, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.04
                        * _recent_flag(choch_bear, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.03 * _recent_flag(bos_bear, i, KEY_LEVEL_STRUCTURE_LOOKBACK)
                        + 0.04 * float(eqh_active[i] > 0)
                    )
                    range_bonus = (
                        0.04
                        if int(range_active[i]) == 1
                        and np.isfinite(range_res_id[i])
                        and int(range_res_id[i]) == int(payload["zone_id"])
                        else 0.0
                    )

                base_score = float(payload["base_score"])
                candidate_score = _clip01(
                    base_score
                    + FAMILY_BONUS.get(family, 0.0)
                    + width_bonus
                    + touch_quality
                    + reclaim_bonus
                    + trend_bonus
                    + structure_bonus
                    + range_bonus
                    - age_penalty
                )

                anchor_bonus = (
                    _clip01(min(int(payload["anchor_count"]), 6) / 6.0) * 0.08
                )
                family_mix_bonus = (
                    _clip01(min(int(payload["family_count"]), 3) / 3.0) * 0.04
                )
                break_importance = _clip01(0.62 * candidate_score + 0.12 * alignment)
                if np.isfinite(break_importance):
                    break_importance = _clip01(
                        break_importance
                        + (0.10 if reclaim_bonus > 0 else 0.0)
                        + (0.08 if range_bonus > 0 else 0.0)
                        + anchor_bonus
                        + family_mix_bonus
                    )

                if (
                    candidate_score > chosen_score
                    or (
                        np.isclose(candidate_score, chosen_score)
                        and break_importance > chosen_break_importance
                    )
                    or (
                        np.isclose(candidate_score, chosen_score)
                        and np.isclose(break_importance, chosen_break_importance)
                        and np.isfinite(close[i])
                        and chosen_payload is not None
                        and abs(float(payload["mid"]) - close[i])
                        < abs(float(chosen_payload["mid"]) - close[i])
                    )
                ):
                    chosen_payload = payload
                    chosen_score = candidate_score
                    chosen_break_importance = break_importance

            if chosen_payload is None:
                continue

            out.at[i, f"{prefix}_zone_id"] = float(chosen_payload["zone_id"])
            out.at[i, f"{prefix}_zone_low"] = float(chosen_payload["low"])
            out.at[i, f"{prefix}_zone_high"] = float(chosen_payload["high"])
            out.at[i, f"{prefix}_zone_mid"] = float(chosen_payload["mid"])
            out.at[i, f"{prefix}_score"] = float(chosen_score)
            out.at[i, f"{prefix}_break_importance_score"] = float(
                chosen_break_importance
            )
            out.at[i, f"{prefix}_best_source_family"] = str(
                chosen_payload["best_source_family"]
            )
            out.at[i, f"{prefix}_selection_source"] = str(
                chosen_payload["selection_source"]
            )

            if (
                np.isfinite(close[i])
                and np.isfinite(float(chosen_payload["low"]))
                and np.isfinite(float(chosen_payload["high"]))
                and float(chosen_payload["low"])
                <= close[i]
                <= float(chosen_payload["high"])
            ):
                out.at[i, f"inside_{prefix}_zone_flag"] = 1

    return out
