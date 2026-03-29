"""
smc/displacement.py

Immediate, same-bar displacement detector.

Contract
--------
- Pure DataFrame-in / DataFrame-out function: input is never mutated.
- Same-bar only: ``origin_idx == detect_idx == current row``.
- Fully causal: no future bars, no active state, no lifecycle state.
- Standalone enrichment detector: depends only on OHLC and ATR.
- Final classification is threshold-gated; continuous metrics and score are
  emitted for downstream filtering, ranking, and validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array

DISPLACEMENT_FLAG_COLUMNS = [
    "displacement_flag",
    "displacement_bull",
    "displacement_bear",
    "displacement_direction",
    "displacement_atr_ready",
    "displacement_pass_valid_inputs",
    "displacement_pass_body_atr",
    "displacement_pass_body_frac",
    "displacement_pass_close_to_extreme",
    "displacement_pass_opposite_wick",
]

DISPLACEMENT_FLOAT_COLUMNS = [
    "displacement_body",
    "displacement_range",
    "displacement_upper_wick",
    "displacement_lower_wick",
    "displacement_body_atr",
    "displacement_range_atr",
    "displacement_upper_wick_atr",
    "displacement_lower_wick_atr",
    "displacement_signed_body_atr",
    "displacement_body_frac",
    "displacement_upper_wick_frac",
    "displacement_lower_wick_frac",
    "displacement_opposite_wick_frac",
    "displacement_close_to_extreme_frac",
    "displacement_score",
]

DISPLACEMENT_CANONICAL_COLUMNS = [
    "displacement_flag",
    "displacement_bull",
    "displacement_bear",
    "displacement_direction",
    "displacement_body",
    "displacement_range",
    "displacement_upper_wick",
    "displacement_lower_wick",
    "displacement_body_atr",
    "displacement_range_atr",
    "displacement_upper_wick_atr",
    "displacement_lower_wick_atr",
    "displacement_signed_body_atr",
    "displacement_body_frac",
    "displacement_upper_wick_frac",
    "displacement_lower_wick_frac",
    "displacement_opposite_wick_frac",
    "displacement_close_to_extreme_frac",
    "displacement_atr_ready",
    "displacement_pass_valid_inputs",
    "displacement_pass_body_atr",
    "displacement_pass_body_frac",
    "displacement_pass_close_to_extreme",
    "displacement_pass_opposite_wick",
    "displacement_score",
]

DISPLACEMENT_LEGACY_COLUMNS = [
    "displacement_candle",
    "displacement_bull_candle",
    "displacement_bear_candle",
    "displacement_close_extreme",
]

DISPLACEMENT_LIVE_SAFE_COLUMNS = (
    DISPLACEMENT_CANONICAL_COLUMNS + DISPLACEMENT_LEGACY_COLUMNS
)


def _validate_parameters(
    *,
    atr_length: int,
    body_atr_mult: float,
    close_extreme_frac: float,
    min_body_frac: float,
    max_opposite_wick_frac: float,
    zero_range_policy: str,
) -> None:
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if body_atr_mult < 0:
        raise ValueError("body_atr_mult must be >= 0")
    if not 0 < close_extreme_frac <= 1:
        raise ValueError("close_extreme_frac must be in (0, 1]")
    if not 0 <= min_body_frac <= 1:
        raise ValueError("min_body_frac must be in [0, 1]")
    if not 0 < max_opposite_wick_frac <= 1:
        raise ValueError("max_opposite_wick_frac must be in (0, 1]")
    if zero_range_policy != "reject":
        raise ValueError("zero_range_policy must be 'reject'")


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _clip_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def add_displacement_candle(
    df: pd.DataFrame,
    atr_length: int = 14,
    body_atr_mult: float = 1.35,
    *,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
    strict_atr_required: bool = True,
    zero_range_policy: str = "reject",
) -> pd.DataFrame:
    """Append canonical displacement columns and legacy compatibility aliases."""
    _validate_parameters(
        atr_length=atr_length,
        body_atr_mult=body_atr_mult,
        close_extreme_frac=close_extreme_frac,
        min_body_frac=min_body_frac,
        max_opposite_wick_frac=max_opposite_wick_frac,
        zero_range_policy=zero_range_policy,
    )

    out = df.copy()

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    atr = get_atr_array(out, length=atr_length).astype(float, copy=False)

    signed_body = c - o
    body = np.abs(signed_body)
    rng = h - lo
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo

    direction = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)
    bull = direction == 1
    bear = direction == -1

    valid_ohlc = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    atr_valid = np.isfinite(atr) & (atr > 0)
    range_valid = np.isfinite(rng) & (rng > 0)
    direction_valid = direction != 0
    if strict_atr_required:
        valid_inputs = valid_ohlc & direction_valid & range_valid & atr_valid
    else:
        valid_inputs = valid_ohlc & direction_valid & range_valid

    atr_ready = (np.arange(len(out)) >= max(atr_length - 1, 0)).astype(np.int8)

    body_atr = _safe_fraction(body, atr)
    range_atr = _safe_fraction(rng, atr)
    upper_wick_atr = _safe_fraction(upper_wick, atr)
    lower_wick_atr = _safe_fraction(lower_wick, atr)
    signed_body_atr = _safe_fraction(signed_body, atr)

    body_frac = _safe_fraction(body, rng)
    upper_wick_frac = _safe_fraction(upper_wick, rng)
    lower_wick_frac = _safe_fraction(lower_wick, rng)

    opposite_wick = np.full(len(out), np.nan, dtype=float)
    opposite_wick[bull] = lower_wick[bull]
    opposite_wick[bear] = upper_wick[bear]
    opposite_wick_frac = _safe_fraction(opposite_wick, rng)

    close_to_extreme = np.full(len(out), np.nan, dtype=float)
    close_to_extreme[bull] = h[bull] - c[bull]
    close_to_extreme[bear] = c[bear] - lo[bear]
    close_to_extreme_frac_arr = _safe_fraction(close_to_extreme, rng)

    pass_body_atr = atr_valid & np.isfinite(body_atr) & (body_atr >= body_atr_mult)
    pass_body_frac = range_valid & np.isfinite(body_frac) & (body_frac >= min_body_frac)
    pass_close = (
        range_valid
        & direction_valid
        & np.isfinite(close_to_extreme_frac_arr)
        & (close_to_extreme_frac_arr <= close_extreme_frac)
    )
    pass_opp = (
        range_valid
        & direction_valid
        & np.isfinite(opposite_wick_frac)
        & (opposite_wick_frac <= max_opposite_wick_frac)
    )

    final = valid_inputs & pass_body_atr & pass_body_frac & pass_close & pass_opp
    bull_final = final & bull
    bear_final = final & bear

    target_body_atr = max(body_atr_mult, 2.0)
    s_body_atr = _clip_unit(body_atr / target_body_atr)

    body_frac_den = 1.0 - min_body_frac
    if body_frac_den > 0:
        s_body_frac = _clip_unit((body_frac - min_body_frac) / body_frac_den)
    else:
        s_body_frac = _clip_unit(np.where(body_frac >= min_body_frac, 1.0, 0.0))

    s_close = _clip_unit(1.0 - (close_to_extreme_frac_arr / close_extreme_frac))
    s_opp = _clip_unit(1.0 - (opposite_wick_frac / max_opposite_wick_frac))

    score = (
        (0.40 * s_body_atr) + (0.25 * s_body_frac) + (0.20 * s_close) + (0.15 * s_opp)
    )
    score[~valid_inputs] = 0.0
    score = score.astype(float, copy=False)

    out["displacement_flag"] = final.astype(np.int8)
    out["displacement_bull"] = bull_final.astype(np.int8)
    out["displacement_bear"] = bear_final.astype(np.int8)
    out["displacement_direction"] = direction.astype(np.int8)

    out["displacement_body"] = body.astype(float, copy=False)
    out["displacement_range"] = rng.astype(float, copy=False)
    out["displacement_upper_wick"] = upper_wick.astype(float, copy=False)
    out["displacement_lower_wick"] = lower_wick.astype(float, copy=False)

    out["displacement_body_atr"] = body_atr.astype(float, copy=False)
    out["displacement_range_atr"] = range_atr.astype(float, copy=False)
    out["displacement_upper_wick_atr"] = upper_wick_atr.astype(float, copy=False)
    out["displacement_lower_wick_atr"] = lower_wick_atr.astype(float, copy=False)
    out["displacement_signed_body_atr"] = signed_body_atr.astype(float, copy=False)

    out["displacement_body_frac"] = body_frac.astype(float, copy=False)
    out["displacement_upper_wick_frac"] = upper_wick_frac.astype(float, copy=False)
    out["displacement_lower_wick_frac"] = lower_wick_frac.astype(float, copy=False)
    out["displacement_opposite_wick_frac"] = opposite_wick_frac.astype(
        float, copy=False
    )
    out["displacement_close_to_extreme_frac"] = close_to_extreme_frac_arr.astype(
        float, copy=False
    )

    out["displacement_atr_ready"] = atr_ready
    out["displacement_pass_valid_inputs"] = valid_inputs.astype(np.int8)
    out["displacement_pass_body_atr"] = pass_body_atr.astype(np.int8)
    out["displacement_pass_body_frac"] = pass_body_frac.astype(np.int8)
    out["displacement_pass_close_to_extreme"] = pass_close.astype(np.int8)
    out["displacement_pass_opposite_wick"] = pass_opp.astype(np.int8)

    out["displacement_score"] = score

    out["displacement_candle"] = out["displacement_flag"]
    out["displacement_bull_candle"] = out["displacement_bull"]
    out["displacement_bear_candle"] = out["displacement_bear"]
    out["displacement_close_extreme"] = out["displacement_pass_close_to_extreme"]

    return out
