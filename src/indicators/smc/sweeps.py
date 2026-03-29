"""
smc/sweeps.py

Canonical causal liquidity sweep detector.

Contract
--------
- Sweep is a sparse structural event detector.
- Raw detect and next-bar confirmation are separate families.
- Raw detect is canonical and live-safe on the sweep bar close.
- Confirmation is optional follow-through metadata on the immediate next bar.
- Source timing is causal: source_idx is the first live-safe row where the
  swept level became available, not the visually interesting origin bar.
- This detector is not an active-state zone, a context builder, or a labeler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.pivots import pivot_high, pivot_low

LEVEL_TYPE_SWING_HIGH = 1
LEVEL_TYPE_SWING_LOW = 2
LEVEL_TYPE_EQUAL_HIGH = 3
LEVEL_TYPE_EQUAL_LOW = 4
LEVEL_TYPE_SESSION_HIGH = 5
LEVEL_TYPE_SESSION_LOW = 6
LEVEL_TYPE_PREV_DAY_HIGH = 7
LEVEL_TYPE_PREV_DAY_LOW = 8
LEVEL_TYPE_PREV_WEEK_HIGH = 9
LEVEL_TYPE_PREV_WEEK_LOW = 10

SWEEP_SHARED_COLUMNS = [
    "sweep_event_id",
    "sweep_direction",
    "sweep_detect_flag",
    "sweep_confirm_flag",
    "sweep_detect_idx",
    "sweep_confirm_idx",
    "sweep_source_idx",
    "sweep_level",
    "sweep_level_type",
    "sweep_level_age",
    "sweep_extension_atr",
    "sweep_reclaim_atr",
    "sweep_wick_frac",
    "sweep_body_frac",
    "sweep_score",
]

SWEEP_HIGH_COLUMNS = [
    "sweep_high_detect_flag",
    "sweep_high_detect_idx",
    "sweep_high_source_idx",
    "sweep_high_level",
    "sweep_high_level_type",
    "sweep_high_level_age",
    "sweep_high_extension_atr",
    "sweep_high_reclaim_atr",
    "sweep_high_wick_frac",
    "sweep_high_body_frac",
]

SWEEP_LOW_COLUMNS = [
    "sweep_low_detect_flag",
    "sweep_low_detect_idx",
    "sweep_low_source_idx",
    "sweep_low_level",
    "sweep_low_level_type",
    "sweep_low_level_age",
    "sweep_low_extension_atr",
    "sweep_low_reclaim_atr",
    "sweep_low_wick_frac",
    "sweep_low_body_frac",
]

SWEEP_CONFIRM_COLUMNS = [
    "sweep_high_confirm_flag",
    "sweep_high_confirm_idx",
    "sweep_low_confirm_flag",
    "sweep_low_confirm_idx",
    "sweep_confirmation_mode",
]

SWEEP_DIAGNOSTIC_COLUMNS = [
    "sweep_atr_ready",
    "sweep_source_ready_high",
    "sweep_source_ready_low",
    "sweep_high_pass_valid_inputs",
    "sweep_high_pass_source_exists",
    "sweep_high_pass_source_age",
    "sweep_high_pass_wick_through",
    "sweep_high_pass_close_back_inside",
    "sweep_high_pass_extension",
    "sweep_high_pass_reclaim",
    "sweep_high_pass_wick_frac",
    "sweep_high_pass_body_frac",
    "sweep_low_pass_valid_inputs",
    "sweep_low_pass_source_exists",
    "sweep_low_pass_source_age",
    "sweep_low_pass_wick_through",
    "sweep_low_pass_close_back_inside",
    "sweep_low_pass_extension",
    "sweep_low_pass_reclaim",
    "sweep_low_pass_wick_frac",
    "sweep_low_pass_body_frac",
]

SWEEP_LEGACY_COLUMNS = [
    "sweep_high",
    "sweep_low",
    "sweep_magnitude",
    "sweep_high_magnitude",
    "sweep_low_magnitude",
    "sweep_level_high",
    "sweep_level_low",
    "sweep_level_high_age",
    "sweep_level_low_age",
    "sweep_confirmed_high",
    "sweep_confirmed_low",
    "sweep_high_wick_frac",
    "sweep_low_wick_frac",
    "sweep_high_body_frac",
    "sweep_low_body_frac",
    "sweep_side",
]

EPS = 1e-12


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _validate_parameters(
    *,
    min_extension_atr: float,
    atr_length: int,
    swing_left: int,
    swing_right: int,
    level_tolerance_atr: float,
    min_reclaim_atr: float,
    min_wick_frac: float,
    max_body_frac: float,
    max_level_age: int | None,
    confirmation_mode: str,
) -> None:
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if swing_left < 1 or swing_right < 1:
        raise ValueError("swing_left and swing_right must be >= 1")
    if min_extension_atr < 0:
        raise ValueError("min_extension_atr must be >= 0")
    if level_tolerance_atr < 0:
        raise ValueError("level_tolerance_atr must be >= 0")
    if min_reclaim_atr < 0:
        raise ValueError("min_reclaim_atr must be >= 0")
    if not 0 <= min_wick_frac <= 1:
        raise ValueError("min_wick_frac must be in [0, 1]")
    if not 0 <= max_body_frac <= 1:
        raise ValueError("max_body_frac must be in [0, 1]")
    if max_level_age is not None and max_level_age < 0:
        raise ValueError("max_level_age must be >= 0 or None")
    if confirmation_mode not in {"close", "mid", "low", "high"}:
        raise ValueError("confirmation_mode must be one of close, mid, low, high")


def _resolve_min_extension_atr(
    *,
    atr_threshold: float | None,
    min_extension_atr: float | None,
) -> float:
    if min_extension_atr is None and atr_threshold is None:
        return 0.2
    if min_extension_atr is None:
        return float(atr_threshold)
    if atr_threshold is None:
        return float(min_extension_atr)
    if not np.isclose(float(atr_threshold), float(min_extension_atr)):
        raise ValueError(
            "atr_threshold and min_extension_atr disagree; pass only one value or "
            "use matching values"
        )
    return float(min_extension_atr)


def _build_fallback_sources(
    h: np.ndarray,
    lo: np.ndarray,
    *,
    swing_left: int,
    swing_right: int,
) -> dict[str, np.ndarray]:
    n = len(h)
    ph = pivot_high(h, left=swing_left, right=swing_right)
    pl = pivot_low(lo, left=swing_left, right=swing_right)

    high_level = np.full(n, np.nan, dtype=float)
    low_level = np.full(n, np.nan, dtype=float)
    high_source_idx = np.full(n, np.nan, dtype=float)
    low_source_idx = np.full(n, np.nan, dtype=float)
    high_age = np.full(n, np.nan, dtype=float)
    low_age = np.full(n, np.nan, dtype=float)
    high_level_type = np.zeros(n, dtype=np.int8)
    low_level_type = np.zeros(n, dtype=np.int8)

    cur_high_level = np.nan
    cur_low_level = np.nan
    cur_high_source_idx = np.nan
    cur_low_source_idx = np.nan

    for i in range(n):
        j = i - swing_right
        if j >= 0:
            if ph[j] == 1:
                cur_high_level = float(h[j])
                cur_high_source_idx = float(j + swing_right)
            if pl[j] == 1:
                cur_low_level = float(lo[j])
                cur_low_source_idx = float(j + swing_right)

        high_level[i] = cur_high_level
        low_level[i] = cur_low_level
        high_source_idx[i] = cur_high_source_idx
        low_source_idx[i] = cur_low_source_idx
        if np.isfinite(cur_high_source_idx):
            high_age[i] = float(i - cur_high_source_idx)
            high_level_type[i] = LEVEL_TYPE_SWING_HIGH
        if np.isfinite(cur_low_source_idx):
            low_age[i] = float(i - cur_low_source_idx)
            low_level_type[i] = LEVEL_TYPE_SWING_LOW

    return {
        "high_level": high_level,
        "low_level": low_level,
        "high_source_idx": high_source_idx,
        "low_source_idx": low_source_idx,
        "high_age": high_age,
        "low_age": low_age,
        "high_level_type": high_level_type,
        "low_level_type": low_level_type,
    }


def _build_source_arrays(
    out: pd.DataFrame,
    h: np.ndarray,
    lo: np.ndarray,
    *,
    use_provided_swings: bool,
    swing_left: int,
    swing_right: int,
) -> dict[str, np.ndarray]:
    required = {
        "last_swing_high",
        "last_swing_low",
        "swing_high_age",
        "swing_low_age",
    }
    if use_provided_swings and required.issubset(out.columns):
        idx_arr = np.arange(len(out), dtype=float)
        high_level = out["last_swing_high"].to_numpy(dtype=float)
        low_level = out["last_swing_low"].to_numpy(dtype=float)
        high_age = out["swing_high_age"].to_numpy(dtype=float)
        low_age = out["swing_low_age"].to_numpy(dtype=float)
        high_source_idx = idx_arr - high_age
        low_source_idx = idx_arr - low_age
        high_level_type = np.where(
            np.isfinite(high_level), LEVEL_TYPE_SWING_HIGH, 0
        ).astype(np.int8)
        low_level_type = np.where(
            np.isfinite(low_level), LEVEL_TYPE_SWING_LOW, 0
        ).astype(np.int8)
        return {
            "high_level": high_level,
            "low_level": low_level,
            "high_source_idx": high_source_idx,
            "low_source_idx": low_source_idx,
            "high_age": high_age,
            "low_age": low_age,
            "high_level_type": high_level_type,
            "low_level_type": low_level_type,
        }

    return _build_fallback_sources(
        h,
        lo,
        swing_left=swing_left,
        swing_right=swing_right,
    )


def _score_components(
    *,
    extension_atr: np.ndarray,
    reclaim_atr: np.ndarray,
    wick_frac: np.ndarray,
    body_frac: np.ndarray,
    level_age: np.ndarray,
    min_extension_atr: float,
    min_reclaim_atr: float,
    min_wick_frac: float,
    max_body_frac: float,
    max_level_age: int | None,
) -> np.ndarray:
    s_ext = _clip01(extension_atr / max(min_extension_atr, 0.50))
    s_reclaim = _clip01(reclaim_atr / max(min_reclaim_atr, 0.25))
    s_wick = _clip01(wick_frac / max(min_wick_frac, 0.33))
    s_body = _clip01(
        1.0
        - _safe_fraction(body_frac, np.full(len(body_frac), max(max_body_frac, EPS)))
    )

    score = np.full(len(extension_atr), np.nan, dtype=float)
    if max_level_age is None:
        finite = (
            np.isfinite(extension_atr)
            & np.isfinite(reclaim_atr)
            & np.isfinite(wick_frac)
            & np.isfinite(body_frac)
        )
        score[finite] = (
            0.30 * s_ext[finite]
            + 0.30 * s_reclaim[finite]
            + 0.20 * s_wick[finite]
            + 0.20 * s_body[finite]
        )
        return score

    s_fresh = _clip01(
        1.0 - _safe_fraction(level_age, np.full(len(level_age), float(max_level_age)))
    )
    finite = (
        np.isfinite(extension_atr)
        & np.isfinite(reclaim_atr)
        & np.isfinite(wick_frac)
        & np.isfinite(body_frac)
        & np.isfinite(level_age)
    )
    score[finite] = (
        0.30 * s_ext[finite]
        + 0.30 * s_reclaim[finite]
        + 0.20 * s_wick[finite]
        + 0.10 * s_body[finite]
        + 0.10 * s_fresh[finite]
    )
    return score


def _confirmation_passes(
    *,
    side: int,
    i: int,
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    mode: str,
) -> bool:
    if side == 1:
        if mode == "mid":
            return bool(c[i + 1] < (o[i] + c[i]) / 2.0)
        if mode == "low":
            return bool(c[i + 1] < lo[i])
        return bool(c[i + 1] < c[i])

    if mode == "mid":
        return bool(c[i + 1] > (o[i] + c[i]) / 2.0)
    if mode == "high":
        return bool(c[i + 1] > h[i])
    return bool(c[i + 1] > c[i])


def _choose_direction(
    *,
    high_detect: bool,
    low_detect: bool,
    high_score: float,
    low_score: float,
    high_extension: float,
    low_extension: float,
    high_reclaim: float,
    low_reclaim: float,
    high_age: float,
    low_age: float,
) -> int:
    if high_detect and not low_detect:
        return 1
    if low_detect and not high_detect:
        return -1
    if not high_detect and not low_detect:
        return 0

    if (
        np.isfinite(high_score)
        and np.isfinite(low_score)
        and not np.isclose(high_score, low_score)
    ):
        return 1 if high_score > low_score else -1
    if (
        np.isfinite(high_extension)
        and np.isfinite(low_extension)
        and not np.isclose(high_extension, low_extension)
    ):
        return 1 if high_extension > low_extension else -1
    if (
        np.isfinite(high_reclaim)
        and np.isfinite(low_reclaim)
        and not np.isclose(high_reclaim, low_reclaim)
    ):
        return 1 if high_reclaim > low_reclaim else -1
    if (
        np.isfinite(high_age)
        and np.isfinite(low_age)
        and not np.isclose(high_age, low_age)
    ):
        return 1 if high_age < low_age else -1
    return 1


def add_liquidity_sweep(
    df: pd.DataFrame,
    atr_threshold: float | None = None,
    *,
    min_extension_atr: float | None = None,
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
    """Append canonical liquidity sweep columns and legacy compatibility aliases."""
    min_extension_atr = _resolve_min_extension_atr(
        atr_threshold=atr_threshold,
        min_extension_atr=min_extension_atr,
    )
    _validate_parameters(
        min_extension_atr=min_extension_atr,
        atr_length=atr_length,
        swing_left=swing_left,
        swing_right=swing_right,
        level_tolerance_atr=level_tolerance_atr,
        min_reclaim_atr=min_reclaim_atr,
        min_wick_frac=min_wick_frac,
        max_body_frac=max_body_frac,
        max_level_age=max_level_age,
        confirmation_mode=confirmation_mode,
    )

    out = df.copy()
    n = len(out)
    if n == 0:
        return out

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
    atr = get_atr_array(out, atr_length).astype(float, copy=False)

    sources = _build_source_arrays(
        out,
        h,
        lo,
        use_provided_swings=use_provided_swings,
        swing_left=swing_left,
        swing_right=swing_right,
    )
    high_level = sources["high_level"]
    low_level = sources["low_level"]
    high_source_idx = sources["high_source_idx"]
    low_source_idx = sources["low_source_idx"]
    high_age = sources["high_age"]
    low_age = sources["low_age"]
    high_level_type = sources["high_level_type"]
    low_level_type = sources["low_level_type"]

    rng = h - lo
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - lo

    upper_wick_frac = _safe_fraction(upper_wick, rng)
    lower_wick_frac = _safe_fraction(lower_wick, rng)
    body_frac = _safe_fraction(body, rng)

    idx_arr = np.arange(n, dtype=float)
    atr_valid = np.isfinite(atr) & (atr > 0)
    valid_ohlc = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    range_valid = np.isfinite(rng) & (rng > 0)
    valid_inputs = valid_ohlc & atr_valid & range_valid
    atr_ready = (idx_arr >= max(atr_length - 1, 0)).astype(np.int8)

    high_source_exists = np.isfinite(high_level) & np.isfinite(high_source_idx)
    low_source_exists = np.isfinite(low_level) & np.isfinite(low_source_idx)
    high_source_ready = (
        high_source_exists
        & np.isfinite(high_age)
        & (high_age >= 0)
        & (high_source_idx <= idx_arr)
    )
    low_source_ready = (
        low_source_exists
        & np.isfinite(low_age)
        & (low_age >= 0)
        & (low_source_idx <= idx_arr)
    )

    tol = level_tolerance_atr * atr

    high_wick_through = high_source_exists & np.isfinite(tol) & (h > (high_level + tol))
    low_wick_through = low_source_exists & np.isfinite(tol) & (lo < (low_level - tol))

    high_close_back_inside = high_source_exists & (c < high_level)
    low_close_back_inside = low_source_exists & (c > low_level)

    high_extension_atr = _safe_fraction(h - high_level, atr)
    low_extension_atr = _safe_fraction(low_level - lo, atr)
    high_reclaim_atr = _safe_fraction(high_level - c, atr)
    low_reclaim_atr = _safe_fraction(c - low_level, atr)

    high_pass_source_age = high_source_ready.copy()
    low_pass_source_age = low_source_ready.copy()
    if max_level_age is not None:
        high_pass_source_age &= high_age <= max_level_age
        low_pass_source_age &= low_age <= max_level_age

    high_pass_valid_inputs = valid_inputs
    low_pass_valid_inputs = valid_inputs
    high_pass_extension = np.isfinite(high_extension_atr) & (
        high_extension_atr >= min_extension_atr
    )
    low_pass_extension = np.isfinite(low_extension_atr) & (
        low_extension_atr >= min_extension_atr
    )
    high_pass_reclaim = np.isfinite(high_reclaim_atr) & (
        high_reclaim_atr >= min_reclaim_atr
    )
    low_pass_reclaim = np.isfinite(low_reclaim_atr) & (
        low_reclaim_atr >= min_reclaim_atr
    )
    high_pass_wick_frac = np.isfinite(upper_wick_frac) & (
        upper_wick_frac >= min_wick_frac
    )
    low_pass_wick_frac = np.isfinite(lower_wick_frac) & (
        lower_wick_frac >= min_wick_frac
    )
    high_pass_body_frac = np.isfinite(body_frac) & (body_frac <= max_body_frac)
    low_pass_body_frac = np.isfinite(body_frac) & (body_frac <= max_body_frac)

    high_score = _score_components(
        extension_atr=high_extension_atr,
        reclaim_atr=high_reclaim_atr,
        wick_frac=upper_wick_frac,
        body_frac=body_frac,
        level_age=high_age,
        min_extension_atr=min_extension_atr,
        min_reclaim_atr=min_reclaim_atr,
        min_wick_frac=min_wick_frac,
        max_body_frac=max_body_frac,
        max_level_age=max_level_age,
    )
    low_score = _score_components(
        extension_atr=low_extension_atr,
        reclaim_atr=low_reclaim_atr,
        wick_frac=lower_wick_frac,
        body_frac=body_frac,
        level_age=low_age,
        min_extension_atr=min_extension_atr,
        min_reclaim_atr=min_reclaim_atr,
        min_wick_frac=min_wick_frac,
        max_body_frac=max_body_frac,
        max_level_age=max_level_age,
    )

    high_raw = (
        high_pass_valid_inputs
        & high_source_exists
        & high_pass_source_age
        & high_wick_through
        & high_close_back_inside
        & high_pass_extension
        & high_pass_reclaim
        & high_pass_wick_frac
        & high_pass_body_frac
    )
    low_raw = (
        low_pass_valid_inputs
        & low_source_exists
        & low_pass_source_age
        & low_wick_through
        & low_close_back_inside
        & low_pass_extension
        & low_pass_reclaim
        & low_pass_wick_frac
        & low_pass_body_frac
    )

    sweep_event_id = np.full(n, np.nan, dtype=float)
    sweep_direction = np.zeros(n, dtype=np.int8)
    sweep_detect_flag = np.zeros(n, dtype=np.int8)
    sweep_confirm_flag = np.zeros(n, dtype=np.int8)
    sweep_detect_idx = np.full(n, np.nan, dtype=float)
    sweep_confirm_idx = np.full(n, np.nan, dtype=float)
    sweep_source_idx = np.full(n, np.nan, dtype=float)
    sweep_level = np.full(n, np.nan, dtype=float)
    sweep_level_type = np.zeros(n, dtype=np.int8)
    sweep_level_age = np.full(n, np.nan, dtype=float)
    sweep_extension = np.full(n, np.nan, dtype=float)
    sweep_reclaim = np.full(n, np.nan, dtype=float)
    sweep_wick = np.full(n, np.nan, dtype=float)
    sweep_body = np.full(n, np.nan, dtype=float)
    sweep_score = np.full(n, np.nan, dtype=float)

    high_detect_flag = np.zeros(n, dtype=np.int8)
    low_detect_flag = np.zeros(n, dtype=np.int8)
    high_detect_idx = np.full(n, np.nan, dtype=float)
    low_detect_idx = np.full(n, np.nan, dtype=float)
    high_source_idx_out = np.full(n, np.nan, dtype=float)
    low_source_idx_out = np.full(n, np.nan, dtype=float)
    high_level_out = np.full(n, np.nan, dtype=float)
    low_level_out = np.full(n, np.nan, dtype=float)
    high_level_type_out = np.zeros(n, dtype=np.int8)
    low_level_type_out = np.zeros(n, dtype=np.int8)
    high_age_out = np.full(n, np.nan, dtype=float)
    low_age_out = np.full(n, np.nan, dtype=float)
    high_extension_out = np.full(n, np.nan, dtype=float)
    low_extension_out = np.full(n, np.nan, dtype=float)
    high_reclaim_out = np.full(n, np.nan, dtype=float)
    low_reclaim_out = np.full(n, np.nan, dtype=float)
    high_wick_out = np.full(n, np.nan, dtype=float)
    low_wick_out = np.full(n, np.nan, dtype=float)
    high_body_out = np.full(n, np.nan, dtype=float)
    low_body_out = np.full(n, np.nan, dtype=float)

    high_confirm_flag = np.zeros(n, dtype=np.int8)
    low_confirm_flag = np.zeros(n, dtype=np.int8)
    high_confirm_idx = np.full(n, np.nan, dtype=float)
    low_confirm_idx = np.full(n, np.nan, dtype=float)
    confirmation_mode_out = np.full(n, None, dtype=object)

    next_event_id = 1
    for i in range(n):
        side = _choose_direction(
            high_detect=bool(high_raw[i]),
            low_detect=bool(low_raw[i]),
            high_score=float(high_score[i]),
            low_score=float(low_score[i]),
            high_extension=float(high_extension_atr[i]),
            low_extension=float(low_extension_atr[i]),
            high_reclaim=float(high_reclaim_atr[i]),
            low_reclaim=float(low_reclaim_atr[i]),
            high_age=float(high_age[i]),
            low_age=float(low_age[i]),
        )
        if side == 0:
            continue

        event_id = float(next_event_id)
        next_event_id += 1

        sweep_event_id[i] = event_id
        sweep_direction[i] = side
        sweep_detect_flag[i] = 1
        sweep_detect_idx[i] = float(i)

        if side == 1:
            high_detect_flag[i] = 1
            high_detect_idx[i] = float(i)
            high_source_idx_out[i] = high_source_idx[i]
            high_level_out[i] = high_level[i]
            high_level_type_out[i] = high_level_type[i]
            high_age_out[i] = high_age[i]
            high_extension_out[i] = high_extension_atr[i]
            high_reclaim_out[i] = high_reclaim_atr[i]
            high_wick_out[i] = upper_wick_frac[i]
            high_body_out[i] = body_frac[i]

            sweep_source_idx[i] = high_source_idx[i]
            sweep_level[i] = high_level[i]
            sweep_level_type[i] = high_level_type[i]
            sweep_level_age[i] = high_age[i]
            sweep_extension[i] = high_extension_atr[i]
            sweep_reclaim[i] = high_reclaim_atr[i]
            sweep_wick[i] = upper_wick_frac[i]
            sweep_body[i] = body_frac[i]
            sweep_score[i] = high_score[i]
        else:
            low_detect_flag[i] = 1
            low_detect_idx[i] = float(i)
            low_source_idx_out[i] = low_source_idx[i]
            low_level_out[i] = low_level[i]
            low_level_type_out[i] = low_level_type[i]
            low_age_out[i] = low_age[i]
            low_extension_out[i] = low_extension_atr[i]
            low_reclaim_out[i] = low_reclaim_atr[i]
            low_wick_out[i] = lower_wick_frac[i]
            low_body_out[i] = body_frac[i]

            sweep_source_idx[i] = low_source_idx[i]
            sweep_level[i] = low_level[i]
            sweep_level_type[i] = low_level_type[i]
            sweep_level_age[i] = low_age[i]
            sweep_extension[i] = low_extension_atr[i]
            sweep_reclaim[i] = low_reclaim_atr[i]
            sweep_wick[i] = lower_wick_frac[i]
            sweep_body[i] = body_frac[i]
            sweep_score[i] = low_score[i]

        if not require_next_bar_confirmation or i + 1 >= n:
            continue
        if not _confirmation_passes(
            side=side,
            i=i,
            o=o,
            h=h,
            lo=lo,
            c=c,
            mode=confirmation_mode,
        ):
            continue

        j = i + 1
        if side == 1:
            high_confirm_flag[j] = 1
            high_confirm_idx[j] = float(j)
        else:
            low_confirm_flag[j] = 1
            low_confirm_idx[j] = float(j)
        confirmation_mode_out[j] = confirmation_mode

        if sweep_detect_flag[j] == 0:
            sweep_event_id[j] = event_id
            sweep_direction[j] = side
            sweep_detect_idx[j] = float(i)
            sweep_source_idx[j] = sweep_source_idx[i]
            sweep_level[j] = sweep_level[i]
            sweep_level_type[j] = sweep_level_type[i]
            sweep_level_age[j] = sweep_level_age[i]
            sweep_extension[j] = sweep_extension[i]
            sweep_reclaim[j] = sweep_reclaim[i]
            sweep_wick[j] = sweep_wick[i]
            sweep_body[j] = sweep_body[i]
            sweep_score[j] = sweep_score[i]
        sweep_confirm_flag[j] = 1
        sweep_confirm_idx[j] = float(j)

    sweep_high = high_detect_flag.copy()
    sweep_low = low_detect_flag.copy()
    sweep_high_magnitude = high_extension_out.copy()
    sweep_low_magnitude = low_extension_out.copy()
    sweep_magnitude = np.where(
        np.isfinite(sweep_high_magnitude) & np.isfinite(sweep_low_magnitude),
        np.maximum(sweep_high_magnitude, sweep_low_magnitude),
        np.where(
            np.isfinite(sweep_high_magnitude),
            sweep_high_magnitude,
            np.where(np.isfinite(sweep_low_magnitude), sweep_low_magnitude, np.nan),
        ),
    )
    sweep_side = sweep_direction.copy()

    arrays: dict[str, np.ndarray] = {
        "sweep_event_id": sweep_event_id,
        "sweep_direction": sweep_direction,
        "sweep_detect_flag": sweep_detect_flag,
        "sweep_confirm_flag": sweep_confirm_flag,
        "sweep_detect_idx": sweep_detect_idx,
        "sweep_confirm_idx": sweep_confirm_idx,
        "sweep_source_idx": sweep_source_idx,
        "sweep_level": sweep_level,
        "sweep_level_type": sweep_level_type,
        "sweep_level_age": sweep_level_age,
        "sweep_extension_atr": sweep_extension,
        "sweep_reclaim_atr": sweep_reclaim,
        "sweep_wick_frac": sweep_wick,
        "sweep_body_frac": sweep_body,
        "sweep_score": sweep_score,
        "sweep_high_detect_flag": high_detect_flag,
        "sweep_high_detect_idx": high_detect_idx,
        "sweep_high_source_idx": high_source_idx_out,
        "sweep_high_level": high_level_out,
        "sweep_high_level_type": high_level_type_out,
        "sweep_high_level_age": high_age_out,
        "sweep_high_extension_atr": high_extension_out,
        "sweep_high_reclaim_atr": high_reclaim_out,
        "sweep_high_wick_frac": high_wick_out,
        "sweep_high_body_frac": high_body_out,
        "sweep_low_detect_flag": low_detect_flag,
        "sweep_low_detect_idx": low_detect_idx,
        "sweep_low_source_idx": low_source_idx_out,
        "sweep_low_level": low_level_out,
        "sweep_low_level_type": low_level_type_out,
        "sweep_low_level_age": low_age_out,
        "sweep_low_extension_atr": low_extension_out,
        "sweep_low_reclaim_atr": low_reclaim_out,
        "sweep_low_wick_frac": low_wick_out,
        "sweep_low_body_frac": low_body_out,
        "sweep_high_confirm_flag": high_confirm_flag,
        "sweep_high_confirm_idx": high_confirm_idx,
        "sweep_low_confirm_flag": low_confirm_flag,
        "sweep_low_confirm_idx": low_confirm_idx,
        "sweep_confirmation_mode": confirmation_mode_out,
        "sweep_atr_ready": atr_ready,
        "sweep_source_ready_high": high_source_ready.astype(np.int8),
        "sweep_source_ready_low": low_source_ready.astype(np.int8),
        "sweep_high_pass_valid_inputs": high_pass_valid_inputs.astype(np.int8),
        "sweep_high_pass_source_exists": high_source_exists.astype(np.int8),
        "sweep_high_pass_source_age": high_pass_source_age.astype(np.int8),
        "sweep_high_pass_wick_through": high_wick_through.astype(np.int8),
        "sweep_high_pass_close_back_inside": high_close_back_inside.astype(np.int8),
        "sweep_high_pass_extension": high_pass_extension.astype(np.int8),
        "sweep_high_pass_reclaim": high_pass_reclaim.astype(np.int8),
        "sweep_high_pass_wick_frac": high_pass_wick_frac.astype(np.int8),
        "sweep_high_pass_body_frac": high_pass_body_frac.astype(np.int8),
        "sweep_low_pass_valid_inputs": low_pass_valid_inputs.astype(np.int8),
        "sweep_low_pass_source_exists": low_source_exists.astype(np.int8),
        "sweep_low_pass_source_age": low_pass_source_age.astype(np.int8),
        "sweep_low_pass_wick_through": low_wick_through.astype(np.int8),
        "sweep_low_pass_close_back_inside": low_close_back_inside.astype(np.int8),
        "sweep_low_pass_extension": low_pass_extension.astype(np.int8),
        "sweep_low_pass_reclaim": low_pass_reclaim.astype(np.int8),
        "sweep_low_pass_wick_frac": low_pass_wick_frac.astype(np.int8),
        "sweep_low_pass_body_frac": low_pass_body_frac.astype(np.int8),
        "sweep_high": sweep_high,
        "sweep_low": sweep_low,
        "sweep_magnitude": sweep_magnitude,
        "sweep_high_magnitude": sweep_high_magnitude,
        "sweep_low_magnitude": sweep_low_magnitude,
        "sweep_level_high": high_level_out,
        "sweep_level_low": low_level_out,
        "sweep_level_high_age": high_age_out,
        "sweep_level_low_age": low_age_out,
        "sweep_confirmed_high": high_confirm_flag,
        "sweep_confirmed_low": low_confirm_flag,
        "sweep_side": sweep_side,
    }

    for column, values in arrays.items():
        out[column] = values

    return out
