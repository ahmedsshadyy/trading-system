"""
Canonical causal range boundary detector.

Doctrine
--------
- Purpose: detect bounded consolidation envelopes and expose their upper/lower
  boundaries as sweepable level sources plus dense context features.
- Causality: no range is active before its confirm bar closes.
- Detect/use split: a range confirmed on row ``t`` becomes active on row ``t``,
  but breakout / breach / reclaim interaction is prohibited on that same row.
- Geometry immutability: confirmed geometry is frozen. Materially different
  boxes create new range IDs rather than mutating prior confirmed ranges.
- Boundary ontology: range boundaries are level sources, not zones.
- Lifecycle separation: breach, breakout acceptance, expiry, invalidation, and
  supersession are distinct outcomes with deterministic precedence.
- Live-safe discipline: canonical outputs depend only on information available
  through the current row. No future labels are mixed into live-safe columns.

Range semantics
---------------
A range is a two-sided bounded consolidation recognized over a rolling window.
Confirmation requires:

- bounded span versus ATR
- repeated support of both upper/lower edges
- close containment inside a shrunken interior band
- suppressed directional drift
- stable candidate boundaries across a confirmation dwell
- no dominant directional structure expansion inside the candidate window

Timing law
----------
For every confirmed range:

- ``range_birth_idx <= range_confirm_idx``
- ``range_active_start_idx = range_confirm_idx``
- ``range_end_idx >= range_active_start_idx`` when terminated
- no boundary may be exported before confirmation
- touch / breach / breakout evaluation begins from ``confirm_idx + 1``

Lifecycle state machine
-----------------------
- ``0 = none``
- ``1 = active_intact``
- ``2 = active_weakened``
- ``3 = broken_unaccepted``
- ``4 = accepted_breakout``      terminal
- ``5 = invalidated``            terminal
- ``6 = expired``                terminal
- ``7 = superseded``             terminal

Terminal precedence on the same bar:
- superseded
- expired
- invalidated
- accepted_breakout
- broken_unaccepted
- active_weakened
- active_intact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns

EPS = 1e-12

RANGE_STATE_NONE = 0
RANGE_STATE_ACTIVE_INTACT = 1
RANGE_STATE_ACTIVE_WEAKENED = 2
RANGE_STATE_BROKEN_UNACCEPTED = 3
RANGE_STATE_ACCEPTED_BREAKOUT = 4
RANGE_STATE_INVALIDATED = 5
RANGE_STATE_EXPIRED = 6
RANGE_STATE_SUPERSEDED = 7

DEFAULT_RANGE_LOOKBACK_BARS = 8
DEFAULT_CANDIDATE_LOOKBACK_BARS = (5, 8, 12, 16)
DEFAULT_MAX_WIDTH_ATR = 3.5
DEFAULT_EDGE_TOLERANCE_ATR = 0.20
DEFAULT_MIN_UPPER_TOUCHES = 2
DEFAULT_MIN_LOWER_TOUCHES = 2
DEFAULT_MIN_CLOSE_INSIDE_FRAC = 0.50
DEFAULT_ALLOWED_WICK_OVERSHOOT_ATR = 1.25
DEFAULT_MAX_DRIFT_FRAC = 0.85
DEFAULT_MIN_CONFIRM_BARS = 2
DEFAULT_MIN_CANDIDATE_DWELL_BARS = 2
DEFAULT_BOUNDARY_STABILITY_TOLERANCE_ATR = 0.35
DEFAULT_MAX_ACTIVE_AGE_BARS = 48
DEFAULT_ACCEPT_MARGIN_ATR = 0.15
DEFAULT_ACCEPT_WINDOW_BARS = 2
DEFAULT_SUPERSESSION_OVERLAP_FRAC = 0.60
DEFAULT_WEAKEN_AGE_FRAC = 0.50
DEFAULT_WEAKEN_PROBE_COUNT = 4
DEFAULT_VIABILITY_LOOKBACK_BARS = 3
DEFAULT_LINEAGE_GRACE_BARS = 1
DEFAULT_DUPLICATE_CONFIRM_LOOKBACK_BARS = 4
DEFAULT_DUPLICATE_INTERVAL_OVERLAP_FRAC = 0.85
DEFAULT_DUPLICATE_MID_DISTANCE_ATR = 0.35
DEFAULT_DUPLICATE_WIDTH_RATIO = 0.75
DEFAULT_MAX_PRESSURE_IMBALANCE = 0.55
DEFAULT_MIN_EQUILIBRIUM_SCORE = 0.45
DEFAULT_MIN_TWO_SIDED_FRESHNESS = 0.25
DEFAULT_MIN_VIABILITY_SCORE = 0.58
DEFAULT_VIABILITY_PRESSURE_WEIGHT = 0.28
DEFAULT_VIABILITY_EQUILIBRIUM_WEIGHT = 0.24
DEFAULT_VIABILITY_FRESHNESS_WEIGHT = 0.20
DEFAULT_VIABILITY_EXPANSION_PRESSURE_WEIGHT = 0.18
DEFAULT_VIABILITY_EXPANSION_VETO_WEIGHT = 0.10
DEFAULT_FINAL_STRENGTH_FORMATION_BASE = 0.35
DEFAULT_FINAL_STRENGTH_VIABILITY_SCALE = 0.65

EVENT_COLUMNS = [
    "range_detect_flag",
    "range_id",
    "range_birth_idx",
    "range_birth_timestamp",
    "range_confirm_idx",
    "range_confirm_timestamp",
    "range_active_start_idx",
    "range_end_idx",
    "range_end_timestamp",
]

GEOMETRY_COLUMNS = [
    "range_low",
    "range_high",
    "range_mid",
    "range_width_abs",
    "range_width_atr",
]

QUALITY_COLUMNS = [
    "range_touch_count_upper",
    "range_touch_count_lower",
    "range_close_inside_frac",
    "range_max_wick_overshoot_atr",
    "range_drift_frac",
    "range_boundary_stability_score",
    "range_compression_score",
    "range_strength_raw",
    "range_strength",
    "range_strength_raw_legacy",
    "range_strength_legacy",
    "range_recent_pressure_imbalance",
    "range_recent_upper_pressure_count",
    "range_recent_lower_pressure_count",
    "range_recent_pressure_dominant_side",
    "range_recent_equilibrium_score",
    "range_recent_expansion_veto_flag",
    "range_viability_gate_pass",
    "range_recent_upper_touch_freshness",
    "range_recent_lower_touch_freshness",
    "range_recent_two_sided_freshness_score",
    "range_strength_structure",
    "range_strength_monitorability",
    "range_strength_semantic",
    "range_strength_formation",
    "range_strength_viability",
    "range_strength_formation_legacy",
    "range_strength_viability_legacy",
    "range_touch_quality_score",
    "range_containment_quality_score",
    "range_stability_quality_score",
]

ACTIVE_STATE_COLUMNS = [
    "range_active",
    "range_state",
    "range_age_bars",
    "range_bars_since_confirm",
    "range_weakened_flag",
    "range_expired_flag",
    "range_superseded_flag",
    "range_invalidated_flag",
    "range_breakout_pending_flag",
    "range_accepted_breakout_flag",
]

BOUNDARY_SOURCE_COLUMNS = [
    "range_upper_active",
    "range_upper_level",
    "range_upper_source_idx",
    "range_upper_source_timestamp",
    "range_upper_age_bars",
    "range_upper_strength",
    "range_lower_active",
    "range_lower_level",
    "range_lower_source_idx",
    "range_lower_source_timestamp",
    "range_lower_age_bars",
    "range_lower_strength",
]

INTERACTION_COLUMNS = [
    "close_position_in_range",
    "dist_to_range_high_abs",
    "dist_to_range_high_atr",
    "dist_to_range_low_abs",
    "dist_to_range_low_atr",
    "inside_active_range_flag",
    "outside_active_range_flag",
]

ALL_RANGE_BOUNDARY_COLUMNS = (
    EVENT_COLUMNS
    + GEOMETRY_COLUMNS
    + QUALITY_COLUMNS
    + ACTIVE_STATE_COLUMNS
    + BOUNDARY_SOURCE_COLUMNS
    + INTERACTION_COLUMNS
)

RANGE_COLUMN_SPECS: dict[str, dict[str, object]] = {}
for col in EVENT_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": (
            "float64"
            if col.endswith("_idx")
            else (
                "int32"
                if col == "range_id"
                else ("int8" if col == "range_detect_flag" else "datetime64[ns, UTC]")
            )
        ),
        "default": (
            0
            if col in {"range_detect_flag", "range_id"}
            else (pd.NaT if "timestamp" in col else np.nan)
        ),
        "family": "detect_event",
    }
for col in GEOMETRY_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": "float64",
        "default": np.nan,
        "family": "geometry",
    }
for col in QUALITY_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": (
            "int8"
            if col in {"range_recent_expansion_veto_flag", "range_viability_gate_pass"}
            else ("int8" if col == "range_recent_pressure_dominant_side" else "float64")
        ),
        "default": (
            0
            if col
            in {
                "range_recent_expansion_veto_flag",
                "range_viability_gate_pass",
                "range_recent_pressure_dominant_side",
            }
            else np.nan
        ),
        "family": "quality",
    }
for col in ACTIVE_STATE_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": (
            "int8"
            if col.endswith("_flag") or col == "range_active"
            else ("int8" if col == "range_state" else "float64")
        ),
        "default": (
            0
            if col != "range_age_bars" and col != "range_bars_since_confirm"
            else np.nan
        ),
        "family": "active_state",
    }
for col in BOUNDARY_SOURCE_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": (
            "int8"
            if col.endswith("_active")
            else ("datetime64[ns, UTC]" if col.endswith("_timestamp") else "float64")
        ),
        "default": (
            0
            if col.endswith("_active")
            else (pd.NaT if col.endswith("_timestamp") else np.nan)
        ),
        "family": "boundary_source",
    }
for col in INTERACTION_COLUMNS:
    RANGE_COLUMN_SPECS[col] = {
        "dtype": "int8" if col.endswith("_flag") else "float64",
        "default": 0 if col.endswith("_flag") else np.nan,
        "family": "annotation",
    }


@dataclass
class _CandidateMetrics:
    eligible: bool
    birth_idx: int | None
    high: float
    low: float
    width_abs: float
    width_atr: float
    upper_touches: int
    lower_touches: int
    close_inside_frac: float
    wick_overshoot_atr: float
    drift_frac: float
    compression_score: float
    structure_penalty: float


TimestampLike = pd.Timestamp | NaTType


@dataclass
class _RangeEvent:
    range_id: int
    candidate_lookback_bars: int
    birth_idx: int
    birth_ts: TimestampLike
    confirm_idx: int
    confirm_ts: TimestampLike
    low: float
    high: float
    mid: float
    width_abs: float
    width_atr: float
    upper_touches: int
    lower_touches: int
    close_inside_frac: float
    wick_overshoot_atr: float
    drift_frac: float
    boundary_stability_score: float
    boundary_drift_abs: float
    boundary_drift_atr: float
    compression_score: float
    strength_raw: float
    base_strength: float
    strength_raw_legacy: float
    base_strength_legacy: float
    confirm_close_position_in_range: float
    confirm_dist_to_upper_atr: float
    confirm_dist_to_lower_atr: float
    confirm_regime: int | None = None
    recent_pressure_imbalance: float = np.nan
    recent_upper_pressure_count: int = 0
    recent_lower_pressure_count: int = 0
    recent_pressure_dominant_side: int = 0
    recent_equilibrium_score: float = np.nan
    recent_expansion_veto_flag: int = 0
    viability_gate_pass: int = 0
    recent_upper_touch_freshness: float = np.nan
    recent_lower_touch_freshness: float = np.nan
    recent_two_sided_freshness_score: float = np.nan
    strength_structure: float = np.nan
    strength_monitorability: float = np.nan
    strength_semantic: float = np.nan
    strength_formation: float = np.nan
    strength_viability: float = np.nan
    strength_formation_legacy: float = np.nan
    strength_viability_legacy: float = np.nan
    touch_quality_score: float = np.nan
    containment_quality_score: float = np.nan
    stability_quality_score: float = np.nan
    current_state: int = RANGE_STATE_ACTIVE_INTACT
    end_idx: int | None = None
    end_ts: TimestampLike = pd.NaT
    pending_side: int = 0
    pending_since_idx: int | None = None
    break_pending_count: int = 0
    total_pending_bars: int = 0
    reclaimed_count: int = 0
    upper_probe_count: int = 0
    lower_probe_count: int = 0
    first_breach_idx: int | None = None
    first_breach_side: int = 0
    first_upper_breach_idx: int | None = None
    first_lower_breach_idx: int | None = None
    accepted_breakout_side: int = 0
    duplicate_suppressed: bool = False


def _utc_timestamp(ts: object) -> TimestampLike:
    if ts is None:
        return pd.NaT
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _series_from_default(index: pd.Index, *, dtype: str, default: object) -> pd.Series:
    if dtype == "datetime64[ns, UTC]":
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    if dtype == "int8":
        return pd.Series(np.full(len(index), int(default), dtype=np.int8), index=index)
    if dtype == "int32":
        return pd.Series(np.full(len(index), int(default), dtype=np.int32), index=index)
    return pd.Series(
        np.full(len(index), default, dtype=float), index=index, dtype=float
    )


def _initialize_columns(out: pd.DataFrame) -> None:
    for col, spec in RANGE_COLUMN_SPECS.items():
        out[col] = _series_from_default(
            out.index,
            dtype=str(spec["dtype"]),
            default=spec["default"],
        )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _compute_candidate_metrics(
    *,
    row_idx: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    regime: np.ndarray | None,
    bos_bull: np.ndarray | None,
    bos_bear: np.ndarray | None,
    choch_bull: np.ndarray | None,
    choch_bear: np.ndarray | None,
    range_lookback_bars: int,
    max_width_atr: float,
    edge_tolerance_atr: float,
    min_upper_touches: int,
    min_lower_touches: int,
    min_close_inside_frac: float,
    allowed_wick_overshoot_atr: float,
    max_drift_frac: float,
) -> _CandidateMetrics:
    if row_idx + 1 < range_lookback_bars:
        return _CandidateMetrics(
            False,
            None,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    atr_now = atr[row_idx]
    if not np.isfinite(atr_now) or atr_now <= EPS:
        return _CandidateMetrics(
            False,
            None,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    start = row_idx - range_lookback_bars + 1
    hi = high[start : row_idx + 1]
    lo = low[start : row_idx + 1]
    cl = close[start : row_idx + 1]
    candidate_high = float(np.nanmax(hi))
    candidate_low = float(np.nanmin(lo))
    width_abs = candidate_high - candidate_low
    if not np.isfinite(width_abs) or width_abs <= EPS:
        return _CandidateMetrics(
            False,
            None,
            candidate_high,
            candidate_low,
            width_abs,
            np.nan,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    width_atr = width_abs / atr_now
    tol_abs = max(edge_tolerance_atr * atr_now, EPS)
    upper_touches = int(np.count_nonzero(hi >= candidate_high - tol_abs))
    lower_touches = int(np.count_nonzero(lo <= candidate_low + tol_abs))

    interior_pad = min(tol_abs, width_abs * 0.20)
    interior_low = candidate_low + interior_pad
    interior_high = candidate_high - interior_pad
    if interior_low >= interior_high:
        close_inside_frac = 0.0
    else:
        close_inside_frac = float(np.mean((cl >= interior_low) & (cl <= interior_high)))

    close_high = float(np.nanmax(cl))
    close_low = float(np.nanmin(cl))
    wick_overshoot_abs = float(
        max(
            0.0,
            np.nanmax(hi - close_high),
            np.nanmax(close_low - lo),
        )
    )
    wick_overshoot_atr = wick_overshoot_abs / atr_now
    drift_frac = abs(close[row_idx] - close[start]) / width_abs
    compression_score = _clip01(1.0 - (width_atr / max(max_width_atr, EPS)))

    structure_penalty = 0.0
    if bos_bull is not None and bos_bear is not None:
        bos_imbalance = abs(
            int(np.nansum(bos_bull[start : row_idx + 1]))
            - int(np.nansum(bos_bear[start : row_idx + 1]))
        )
        structure_penalty += min(1.0, bos_imbalance / 3.0)
    if choch_bull is not None and choch_bear is not None:
        choch_imbalance = abs(
            int(np.nansum(choch_bull[start : row_idx + 1]))
            - int(np.nansum(choch_bear[start : row_idx + 1]))
        )
        structure_penalty += min(1.0, choch_imbalance / 3.0) * 0.5
    if regime is not None:
        recent_regime = regime[start : row_idx + 1]
        structure_penalty += float(np.mean(recent_regime == 2)) * 0.75
    structure_penalty = min(structure_penalty, 1.0)

    eligible = (
        width_atr <= max_width_atr
        and upper_touches >= min_upper_touches
        and lower_touches >= min_lower_touches
        and close_inside_frac >= min_close_inside_frac
        and wick_overshoot_atr <= allowed_wick_overshoot_atr
        and drift_frac <= max_drift_frac
        and structure_penalty <= 0.95
    )
    return _CandidateMetrics(
        eligible=bool(eligible),
        birth_idx=start,
        high=candidate_high,
        low=candidate_low,
        width_abs=width_abs,
        width_atr=width_atr,
        upper_touches=upper_touches,
        lower_touches=lower_touches,
        close_inside_frac=close_inside_frac,
        wick_overshoot_atr=wick_overshoot_atr,
        drift_frac=drift_frac,
        compression_score=compression_score,
        structure_penalty=structure_penalty,
    )


def _boundary_stability_score(
    highs: list[float], lows: list[float], tol_abs: float
) -> float:
    if not highs or not lows or tol_abs <= EPS:
        return np.nan
    high_move = max(highs) - min(highs)
    low_move = max(lows) - min(lows)
    move = max(high_move, low_move)
    return _clip01(1.0 - (move / max(2.0 * tol_abs, EPS)))


def _boundary_drift(
    highs: list[float], lows: list[float], atr_value: float
) -> tuple[float, float]:
    if not highs or not lows:
        return np.nan, np.nan
    high_move = max(highs) - min(highs)
    low_move = max(lows) - min(lows)
    drift_abs = max(high_move, low_move)
    if not np.isfinite(atr_value) or atr_value <= EPS:
        return drift_abs, np.nan
    return drift_abs, drift_abs / atr_value


def _compute_strength(
    *,
    upper_touches: int,
    lower_touches: int,
    close_inside_frac: float,
    width_atr: float,
    max_width_atr: float,
    compression_score: float,
    boundary_stability_score: float,
    drift_frac: float,
    structure_penalty: float,
) -> tuple[float, float, dict[str, float]]:
    min_touches = min(upper_touches, lower_touches)
    touch_symmetry = 1.0 - (
        abs(upper_touches - lower_touches) / max(upper_touches + lower_touches, 1)
    )
    # Do not let minimal extra touches dominate formation quality. The
    # detector only needs sufficient two-sided structure here; viability and
    # later ranking should decide whether the box is worth promoting.
    touch_quality = _clip01((min_touches - 1.0) / 4.0)
    containment_quality = _clip01(close_inside_frac)
    anti_trend_quality = _clip01(1.0 - drift_frac)
    stability_quality = _clip01(boundary_stability_score)
    preferred_width_atr = max(max_width_atr * 0.8, EPS)
    width_quality = _clip01(
        1.0 - (abs(width_atr - preferred_width_atr) / preferred_width_atr)
    )
    formation = (
        0.14 * touch_symmetry
        + 0.08 * touch_quality
        + 0.18 * containment_quality
        + 0.12 * compression_score
        + 0.18 * stability_quality
        + 0.12 * anti_trend_quality
        + 0.18 * width_quality
    )
    formation *= 0.92 + 0.08 * touch_quality
    formation = _clip01(formation)
    return (
        formation,
        formation,
        {
            "touch_quality_score": touch_quality,
            "containment_quality_score": containment_quality,
            "stability_quality_score": stability_quality,
        },
    )


def _compute_rebased_ranking(
    *,
    birth_idx: int,
    confirm_idx: int,
    upper_touches: int,
    lower_touches: int,
    width_atr: float,
    touch_quality_score: float,
    containment_quality_score: float,
    stability_quality_score: float,
    pressure_imbalance: float,
    equilibrium_score: float,
    two_sided_freshness: float,
    expansion_veto_flag: int,
    legacy_viability: float,
    legacy_structure: float,
) -> dict[str, float]:
    confirm_latency = max(confirm_idx - birth_idx, 0)
    min_touches = min(upper_touches, lower_touches)
    touch_balance = _clip01(
        1.0
        - (abs(upper_touches - lower_touches) / max(upper_touches + lower_touches, 1))
    )
    touch_richness = _clip01((min_touches - 2.0) / 3.0)
    width_reasonableness = _clip01(1.0 - (abs(width_atr - 2.2) / 1.3))
    latency_quality = _clip01((confirm_latency - 1.0) / 3.0)
    latency_risk = 1.0 - latency_quality
    early_freshness_risk = _clip01(two_sided_freshness * latency_risk)
    early_equilibrium_risk = _clip01(equilibrium_score * latency_risk)
    touch_minimality_risk = 1.0 - touch_richness
    width_risk = 1.0 - width_reasonableness

    micro_box_risk = _clip01(
        (
            latency_risk
            + touch_minimality_risk
            + width_risk
            + early_freshness_risk
            + early_equilibrium_risk
        )
        / 5.0
    )
    boundary_relevance = _clip01(
        (
            touch_balance
            + touch_richness
            + width_reasonableness
            + (1.0 - pressure_imbalance)
        )
        / 4.0
    )
    monitorability = _clip01(
        (
            0.32 * boundary_relevance
            + 0.28 * (1.0 - micro_box_risk)
            + 0.18 * legacy_viability
            + 0.12 * stability_quality_score
            + 0.10 * (1.0 - float(expansion_veto_flag))
        )
    )
    semantic = _clip01(
        (
            0.36 * monitorability
            + 0.22 * boundary_relevance
            + 0.30 * (1.0 - micro_box_risk)
            + 0.12 * legacy_viability
        )
    )
    rebased_viability = _clip01(
        (0.20 * legacy_viability + 0.40 * monitorability + 0.40 * semantic)
    )
    rebased_viability = min(
        rebased_viability,
        _clip01(0.35 + 0.55 * monitorability),
        _clip01(0.35 + 0.55 * (1.0 - micro_box_risk)),
    )
    final_strength = _clip01(
        (0.08 * legacy_structure + 0.42 * monitorability + 0.50 * semantic)
    )
    final_strength = min(
        final_strength,
        _clip01(0.45 + 0.45 * monitorability),
        _clip01(0.40 + 0.50 * (1.0 - micro_box_risk)),
    )
    return {
        "strength_structure": _clip01(legacy_structure),
        "strength_monitorability": monitorability,
        "strength_semantic": semantic,
        "strength_viability_rebased": rebased_viability,
        "strength_final_rebased": final_strength,
        "micro_box_risk_proxy": micro_box_risk,
        "boundary_relevance_proxy": boundary_relevance,
        "confirm_latency_quality": latency_quality,
        "touch_balance_proxy": touch_balance,
        "width_reasonableness_proxy": width_reasonableness,
        "touch_quality_proxy": _clip01(
            0.50 * touch_quality_score + 0.30 * touch_richness + 0.20 * touch_balance
        ),
        "containment_proxy": _clip01(containment_quality_score),
        "stability_proxy": _clip01(stability_quality_score),
    }


def _compute_viability_metrics(
    *,
    row_idx: int,
    candidate_high: float,
    candidate_low: float,
    width_abs: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    range_lookback_bars: int,
    viability_lookback_bars: int,
    edge_tolerance_atr: float,
    min_viability_score: float,
    pressure_weight: float,
    equilibrium_weight: float,
    freshness_weight: float,
    expansion_pressure_weight: float,
    expansion_veto_weight: float,
) -> dict[str, float | int]:
    atr_now = atr[row_idx]
    if not np.isfinite(atr_now) or atr_now <= EPS or width_abs <= EPS:
        return {
            "recent_pressure_imbalance": np.nan,
            "recent_upper_pressure_count": 0,
            "recent_lower_pressure_count": 0,
            "recent_pressure_dominant_side": 0,
            "recent_equilibrium_score": np.nan,
            "recent_expansion_veto_flag": 0,
            "recent_upper_touch_freshness": np.nan,
            "recent_lower_touch_freshness": np.nan,
            "recent_two_sided_freshness_score": np.nan,
            "pressure_threshold_fail": 0,
            "equilibrium_threshold_fail": 0,
            "freshness_threshold_fail": 0,
            "score_threshold_fail": 0,
            "multiple_reasons_fail": 0,
            "viability_gate_pass": 0,
            "strength_viability": np.nan,
        }

    recent_start = max(0, row_idx - viability_lookback_bars + 1)
    recent_close = close[recent_start : row_idx + 1]
    recent_high = high[recent_start : row_idx + 1]
    recent_low = low[recent_start : row_idx + 1]
    close_pos = np.clip((recent_close - candidate_low) / max(width_abs, EPS), 0.0, 1.0)
    tol_abs = max(edge_tolerance_atr * atr_now, EPS)

    upper_pressure = (recent_high >= candidate_high - tol_abs) | (close_pos >= 0.72)
    lower_pressure = (recent_low <= candidate_low + tol_abs) | (close_pos <= 0.28)
    upper_pressure_count = int(np.count_nonzero(upper_pressure))
    lower_pressure_count = int(np.count_nonzero(lower_pressure))
    close_span = float(np.max(close_pos) - np.min(close_pos)) if len(close_pos) else 0.0
    mean_edge_bias = abs(float(close_pos.mean()) - 0.5) * 2.0
    last_edge_bias = abs(float(close_pos[-1]) - 0.5) * 2.0 if len(close_pos) else 0.0
    pressure_imbalance = _clip01(
        (1.0 - close_span) * (0.5 + 0.5 * max(mean_edge_bias, last_edge_bias))
    )
    dominant_side = 0
    if upper_pressure_count - lower_pressure_count >= 1:
        dominant_side = 1
    elif lower_pressure_count - upper_pressure_count >= 1:
        dominant_side = -1

    equilibrium_score = _clip01(1.0 - mean_edge_bias)
    recent_range_frac = (recent_high - recent_low) / max(width_abs, EPS)
    max_recent_range_frac = (
        float(np.max(recent_range_frac)) if len(recent_range_frac) else 0.0
    )
    expansion_pressure = _clip01((max_recent_range_frac - 0.30) / 0.45)

    candidate_start = max(0, row_idx - range_lookback_bars + 1)
    candidate_high_window = high[candidate_start : row_idx + 1]
    candidate_low_window = low[candidate_start : row_idx + 1]
    upper_touch_idx = np.flatnonzero(candidate_high_window >= candidate_high - tol_abs)
    lower_touch_idx = np.flatnonzero(candidate_low_window <= candidate_low + tol_abs)
    upper_touch_freshness = (
        _clip01(
            1.0
            - (
                (row_idx - (candidate_start + int(upper_touch_idx[-1])))
                / max(range_lookback_bars, 1)
            )
        )
        if len(upper_touch_idx)
        else 0.0
    )
    lower_touch_freshness = (
        _clip01(
            1.0
            - (
                (row_idx - (candidate_start + int(lower_touch_idx[-1])))
                / max(range_lookback_bars, 1)
            )
        )
        if len(lower_touch_idx)
        else 0.0
    )
    two_sided_freshness = min(upper_touch_freshness, lower_touch_freshness)

    recent_move_frac = abs(close[row_idx] - close[recent_start]) / max(width_abs, EPS)
    body_atr = abs(close[row_idx] - open_[row_idx]) / atr_now
    close_pos_last = float(close_pos[-1]) if len(close_pos) else 0.5
    expansion_veto = int(
        (
            dominant_side == 1
            and close_pos_last >= 0.82
            and recent_move_frac >= 0.30
            and body_atr >= 0.35
        )
        or (
            dominant_side == -1
            and close_pos_last <= 0.18
            and recent_move_frac >= 0.30
            and body_atr >= 0.35
        )
    )

    weight_total = max(
        pressure_weight
        + equilibrium_weight
        + freshness_weight
        + expansion_pressure_weight
        + expansion_veto_weight,
        EPS,
    )
    viability = _clip01(
        (
            pressure_weight * (1.0 - pressure_imbalance)
            + equilibrium_weight * equilibrium_score
            + freshness_weight * two_sided_freshness
            + expansion_pressure_weight * (1.0 - expansion_pressure)
            + expansion_veto_weight * (1.0 - expansion_veto)
        )
        / weight_total
    )
    pressure_threshold_fail = int(pressure_imbalance > DEFAULT_MAX_PRESSURE_IMBALANCE)
    equilibrium_threshold_fail = int(equilibrium_score < DEFAULT_MIN_EQUILIBRIUM_SCORE)
    freshness_threshold_fail = int(
        two_sided_freshness < DEFAULT_MIN_TWO_SIDED_FRESHNESS
    )
    score_threshold_fail = int(viability < min_viability_score)
    multiple_reasons_fail = int(
        pressure_threshold_fail + equilibrium_threshold_fail + freshness_threshold_fail
        > 1
    )
    gate_pass = int(expansion_veto == 0 and score_threshold_fail == 0)
    return {
        "recent_pressure_imbalance": pressure_imbalance,
        "recent_upper_pressure_count": upper_pressure_count,
        "recent_lower_pressure_count": lower_pressure_count,
        "recent_pressure_dominant_side": dominant_side,
        "recent_equilibrium_score": equilibrium_score,
        "recent_expansion_veto_flag": expansion_veto,
        "recent_upper_touch_freshness": upper_touch_freshness,
        "recent_lower_touch_freshness": lower_touch_freshness,
        "recent_two_sided_freshness_score": two_sided_freshness,
        "pressure_threshold_fail": pressure_threshold_fail,
        "equilibrium_threshold_fail": equilibrium_threshold_fail,
        "freshness_threshold_fail": freshness_threshold_fail,
        "score_threshold_fail": score_threshold_fail,
        "multiple_reasons_fail": multiple_reasons_fail,
        "viability_gate_pass": gate_pass,
        "strength_viability": viability,
    }


def _event_effective_strength(
    event: _RangeEvent, *, row_idx: int, max_active_age_bars: int
) -> float:
    age = max(row_idx - event.confirm_idx, 0)
    age_decay = 1.0 - min(age / max(max_active_age_bars, 1), 1.0) * 0.50
    state_modifier = 1.0
    if event.current_state == RANGE_STATE_ACTIVE_WEAKENED:
        state_modifier = 0.85
    elif event.current_state == RANGE_STATE_BROKEN_UNACCEPTED:
        state_modifier = 0.60
    return _clip01(event.base_strength * age_decay * state_modifier)


def _interval_overlap_frac(
    a_low: float, a_high: float, b_low: float, b_high: float
) -> float:
    inter = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    if inter <= 0.0:
        return 0.0
    denom = max(min(a_high - a_low, b_high - b_low), EPS)
    return inter / denom


def _finalize_candidate_lineage(
    candidate_lineages: dict[int, dict[str, object]],
    lineage_id: int | None,
    *,
    min_candidate_dwell_bars: int,
    eligibility_failure: bool = False,
    same_lineage_continuation_fail: bool = False,
) -> None:
    if lineage_id is None:
        return
    lineage = candidate_lineages.get(lineage_id)
    if lineage is None:
        return

    if same_lineage_continuation_fail:
        lineage["failed_same_lineage_continuation"] = 1

    maturity_pass = int(lineage.get("maturity_pass_flag", 0)) == 1
    if maturity_pass:
        return

    birth_idx = int(lineage.get("birth_idx", 0))
    last_idx = int(lineage.get("last_idx", birth_idx))
    lifetime_bars = max(last_idx - birth_idx + 1, 0)

    if eligibility_failure:
        lineage["failed_candidate_eligibility_before_maturity"] = 1
    if lifetime_bars < min_candidate_dwell_bars:
        lineage["failed_dwell"] = 1


def _new_candidate_lineage_record(
    *,
    lineage_id: int,
    row_idx: int,
    timestamp: TimestampLike,
    candidate_lookback_bars: int,
) -> dict[str, object]:
    return {
        "candidate_lineage_id": lineage_id,
        "candidate_lookback_bars": candidate_lookback_bars,
        "birth_idx": row_idx,
        "birth_timestamp": timestamp,
        "last_idx": row_idx,
        "last_timestamp": timestamp,
        "raw_candidate_flag": 1,
        "maturity_pass_flag": 0,
        "maturity_pass_idx": np.nan,
        "maturity_pass_timestamp": pd.NaT,
        "viability_pass_flag": 0,
        "viability_pass_idx": np.nan,
        "viability_pass_timestamp": pd.NaT,
        "confirmed_flag": 0,
        "duplicate_suppressed_flag": 0,
        "range_confirm_idx": np.nan,
        "range_confirm_timestamp": pd.NaT,
        "failed_dwell": 0,
        "failed_boundary_stability": 0,
        "failed_same_lineage_continuation": 0,
        "failed_candidate_eligibility_before_maturity": 0,
        "range_recent_expansion_veto_flag": 0,
        "range_viability_gate_pass": 0,
        "range_recent_pressure_imbalance": np.nan,
        "range_recent_equilibrium_score": np.nan,
        "range_recent_two_sided_freshness_score": np.nan,
        "range_recent_upper_pressure_count": np.nan,
        "range_recent_lower_pressure_count": np.nan,
        "range_recent_pressure_dominant_side": 0,
        "range_strength_viability": np.nan,
        "viability_fail_due_to_pressure": 0,
        "viability_fail_due_to_equilibrium": 0,
        "viability_fail_due_to_freshness": 0,
        "viability_fail_due_to_score_threshold": 0,
        "viability_fail_multiple_reasons": 0,
        "expansion_veto_seen_flag": 0,
        "low_viability_reject_seen_flag": 0,
    }


def _reset_range_row(out: pd.DataFrame, row_idx: int) -> None:
    for col, spec in RANGE_COLUMN_SPECS.items():
        dtype = str(spec["dtype"])
        default = spec["default"]
        out.at[row_idx, col] = pd.NaT if dtype == "datetime64[ns, UTC]" else default


def _event_duplicate_key(
    event: _RangeEvent,
) -> tuple[float, float, float, float, float]:
    return (
        event.strength_viability_legacy,
        event.strength_formation_legacy,
        -event.width_atr,
        -float(event.candidate_lookback_bars),
        float(event.confirm_idx),
    )


def _find_duplicate_event(
    events: list[_RangeEvent],
    candidate_event: _RangeEvent,
    *,
    atr_value: float,
) -> _RangeEvent | None:
    for older in events:
        if older.duplicate_suppressed:
            continue
        if (
            older.confirm_idx
            < candidate_event.confirm_idx - DEFAULT_DUPLICATE_CONFIRM_LOOKBACK_BARS
        ):
            continue
        overlap_frac = _interval_overlap_frac(
            older.low,
            older.high,
            candidate_event.low,
            candidate_event.high,
        )
        mid_distance_atr = abs(older.mid - candidate_event.mid) / max(atr_value, EPS)
        width_ratio = min(older.width_abs, candidate_event.width_abs) / max(
            max(older.width_abs, candidate_event.width_abs),
            EPS,
        )
        if (
            overlap_frac >= DEFAULT_DUPLICATE_INTERVAL_OVERLAP_FRAC
            and mid_distance_atr <= DEFAULT_DUPLICATE_MID_DISTANCE_ATR
            and width_ratio >= DEFAULT_DUPLICATE_WIDTH_RATIO
        ):
            return older
    return None


def _select_dense_active(
    events: list[_RangeEvent], *, row_idx: int, max_active_age_bars: int
) -> _RangeEvent | None:
    active = [
        event
        for event in events
        if not event.duplicate_suppressed
        and event.current_state
        in {
            RANGE_STATE_ACTIVE_INTACT,
            RANGE_STATE_ACTIVE_WEAKENED,
            RANGE_STATE_BROKEN_UNACCEPTED,
        }
    ]
    if not active:
        return None
    active.sort(
        key=lambda event: (
            _event_effective_strength(
                event, row_idx=row_idx, max_active_age_bars=max_active_age_bars
            ),
            -event.width_atr,
            event.confirm_idx,
            -event.range_id,
        ),
        reverse=True,
    )
    return active[0]


def _stamp_detect_row(out: pd.DataFrame, event: _RangeEvent) -> None:
    idx = event.confirm_idx
    out.at[idx, "range_detect_flag"] = np.int8(1)
    out.at[idx, "range_id"] = np.int32(event.range_id)
    out.at[idx, "range_birth_idx"] = float(event.birth_idx)
    out.at[idx, "range_birth_timestamp"] = event.birth_ts
    out.at[idx, "range_confirm_idx"] = float(event.confirm_idx)
    out.at[idx, "range_confirm_timestamp"] = event.confirm_ts
    out.at[idx, "range_active_start_idx"] = float(event.confirm_idx)
    out.at[idx, "range_low"] = event.low
    out.at[idx, "range_high"] = event.high
    out.at[idx, "range_mid"] = event.mid
    out.at[idx, "range_width_abs"] = event.width_abs
    out.at[idx, "range_width_atr"] = event.width_atr
    out.at[idx, "range_touch_count_upper"] = float(event.upper_touches)
    out.at[idx, "range_touch_count_lower"] = float(event.lower_touches)
    out.at[idx, "range_close_inside_frac"] = event.close_inside_frac
    out.at[idx, "range_max_wick_overshoot_atr"] = event.wick_overshoot_atr
    out.at[idx, "range_drift_frac"] = event.drift_frac
    out.at[idx, "range_boundary_stability_score"] = event.boundary_stability_score
    out.at[idx, "range_compression_score"] = event.compression_score
    out.at[idx, "range_strength_raw"] = event.strength_raw
    out.at[idx, "range_strength"] = event.base_strength
    out.at[idx, "range_strength_raw_legacy"] = event.strength_raw_legacy
    out.at[idx, "range_strength_legacy"] = event.base_strength_legacy
    out.at[idx, "range_recent_pressure_imbalance"] = event.recent_pressure_imbalance
    out.at[idx, "range_recent_upper_pressure_count"] = float(
        event.recent_upper_pressure_count
    )
    out.at[idx, "range_recent_lower_pressure_count"] = float(
        event.recent_lower_pressure_count
    )
    out.at[idx, "range_recent_pressure_dominant_side"] = np.int8(
        event.recent_pressure_dominant_side
    )
    out.at[idx, "range_recent_equilibrium_score"] = event.recent_equilibrium_score
    out.at[idx, "range_recent_expansion_veto_flag"] = np.int8(
        event.recent_expansion_veto_flag
    )
    out.at[idx, "range_viability_gate_pass"] = np.int8(event.viability_gate_pass)
    out.at[idx, "range_recent_upper_touch_freshness"] = (
        event.recent_upper_touch_freshness
    )
    out.at[idx, "range_recent_lower_touch_freshness"] = (
        event.recent_lower_touch_freshness
    )
    out.at[idx, "range_recent_two_sided_freshness_score"] = (
        event.recent_two_sided_freshness_score
    )
    out.at[idx, "range_strength_structure"] = event.strength_structure
    out.at[idx, "range_strength_monitorability"] = event.strength_monitorability
    out.at[idx, "range_strength_semantic"] = event.strength_semantic
    out.at[idx, "range_strength_formation"] = event.strength_structure
    out.at[idx, "range_strength_viability"] = event.strength_viability
    out.at[idx, "range_strength_formation_legacy"] = event.strength_formation_legacy
    out.at[idx, "range_strength_viability_legacy"] = event.strength_viability_legacy
    out.at[idx, "range_touch_quality_score"] = event.touch_quality_score
    out.at[idx, "range_containment_quality_score"] = event.containment_quality_score
    out.at[idx, "range_stability_quality_score"] = event.stability_quality_score


def _stamp_terminal_on_confirm_row(out: pd.DataFrame, event: _RangeEvent) -> None:
    if event.end_idx is None:
        return
    idx = event.confirm_idx
    out.at[idx, "range_end_idx"] = float(event.end_idx)
    out.at[idx, "range_end_timestamp"] = event.end_ts


def _project_dense_active(
    out: pd.DataFrame,
    *,
    row_idx: int,
    event: _RangeEvent | None,
    close_value: float,
    atr_value: float,
    max_active_age_bars: int,
) -> None:
    if event is None:
        return

    age = max(row_idx - event.confirm_idx, 0)
    strength = _event_effective_strength(
        event, row_idx=row_idx, max_active_age_bars=max_active_age_bars
    )
    out.at[row_idx, "range_id"] = np.int32(event.range_id)
    out.at[row_idx, "range_birth_idx"] = float(event.birth_idx)
    out.at[row_idx, "range_birth_timestamp"] = event.birth_ts
    out.at[row_idx, "range_confirm_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_confirm_timestamp"] = event.confirm_ts
    out.at[row_idx, "range_active_start_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_end_idx"] = (
        float(event.end_idx) if event.end_idx is not None else np.nan
    )
    out.at[row_idx, "range_end_timestamp"] = (
        event.end_ts if event.end_idx is not None else pd.NaT
    )
    out.at[row_idx, "range_low"] = event.low
    out.at[row_idx, "range_high"] = event.high
    out.at[row_idx, "range_mid"] = event.mid
    out.at[row_idx, "range_width_abs"] = event.width_abs
    out.at[row_idx, "range_width_atr"] = event.width_atr
    out.at[row_idx, "range_touch_count_upper"] = float(event.upper_touches)
    out.at[row_idx, "range_touch_count_lower"] = float(event.lower_touches)
    out.at[row_idx, "range_close_inside_frac"] = event.close_inside_frac
    out.at[row_idx, "range_max_wick_overshoot_atr"] = event.wick_overshoot_atr
    out.at[row_idx, "range_drift_frac"] = event.drift_frac
    out.at[row_idx, "range_boundary_stability_score"] = event.boundary_stability_score
    out.at[row_idx, "range_compression_score"] = event.compression_score
    out.at[row_idx, "range_strength_raw"] = event.strength_raw
    out.at[row_idx, "range_strength"] = strength
    out.at[row_idx, "range_strength_raw_legacy"] = event.strength_raw_legacy
    out.at[row_idx, "range_strength_legacy"] = (
        _clip01(event.base_strength_legacy * (strength / max(event.base_strength, EPS)))
        if np.isfinite(event.base_strength) and event.base_strength > EPS
        else event.base_strength_legacy
    )
    out.at[row_idx, "range_recent_pressure_imbalance"] = event.recent_pressure_imbalance
    out.at[row_idx, "range_recent_upper_pressure_count"] = float(
        event.recent_upper_pressure_count
    )
    out.at[row_idx, "range_recent_lower_pressure_count"] = float(
        event.recent_lower_pressure_count
    )
    out.at[row_idx, "range_recent_pressure_dominant_side"] = np.int8(
        event.recent_pressure_dominant_side
    )
    out.at[row_idx, "range_recent_equilibrium_score"] = event.recent_equilibrium_score
    out.at[row_idx, "range_recent_expansion_veto_flag"] = np.int8(
        event.recent_expansion_veto_flag
    )
    out.at[row_idx, "range_viability_gate_pass"] = np.int8(event.viability_gate_pass)
    out.at[row_idx, "range_recent_upper_touch_freshness"] = (
        event.recent_upper_touch_freshness
    )
    out.at[row_idx, "range_recent_lower_touch_freshness"] = (
        event.recent_lower_touch_freshness
    )
    out.at[row_idx, "range_recent_two_sided_freshness_score"] = (
        event.recent_two_sided_freshness_score
    )
    out.at[row_idx, "range_strength_structure"] = event.strength_structure
    out.at[row_idx, "range_strength_monitorability"] = event.strength_monitorability
    out.at[row_idx, "range_strength_semantic"] = event.strength_semantic
    out.at[row_idx, "range_strength_formation"] = event.strength_structure
    out.at[row_idx, "range_strength_viability"] = event.strength_viability
    out.at[row_idx, "range_strength_formation_legacy"] = event.strength_formation_legacy
    out.at[row_idx, "range_strength_viability_legacy"] = event.strength_viability_legacy
    out.at[row_idx, "range_touch_quality_score"] = event.touch_quality_score
    out.at[row_idx, "range_containment_quality_score"] = event.containment_quality_score
    out.at[row_idx, "range_stability_quality_score"] = event.stability_quality_score
    out.at[row_idx, "range_active"] = np.int8(
        event.current_state
        in {
            RANGE_STATE_ACTIVE_INTACT,
            RANGE_STATE_ACTIVE_WEAKENED,
            RANGE_STATE_BROKEN_UNACCEPTED,
        }
    )
    out.at[row_idx, "range_state"] = np.int8(event.current_state)
    out.at[row_idx, "range_age_bars"] = float(age)
    out.at[row_idx, "range_bars_since_confirm"] = float(age)
    out.at[row_idx, "range_weakened_flag"] = np.int8(
        event.current_state == RANGE_STATE_ACTIVE_WEAKENED
    )
    out.at[row_idx, "range_breakout_pending_flag"] = np.int8(
        event.current_state == RANGE_STATE_BROKEN_UNACCEPTED
    )
    out.at[row_idx, "range_accepted_breakout_flag"] = np.int8(
        event.current_state == RANGE_STATE_ACCEPTED_BREAKOUT
    )
    out.at[row_idx, "range_expired_flag"] = np.int8(
        event.current_state == RANGE_STATE_EXPIRED
    )
    out.at[row_idx, "range_superseded_flag"] = np.int8(
        event.current_state == RANGE_STATE_SUPERSEDED
    )
    out.at[row_idx, "range_invalidated_flag"] = np.int8(
        event.current_state == RANGE_STATE_INVALIDATED
    )

    boundary_active = np.int8(
        event.current_state in {RANGE_STATE_ACTIVE_INTACT, RANGE_STATE_ACTIVE_WEAKENED}
    )
    out.at[row_idx, "range_upper_active"] = boundary_active
    out.at[row_idx, "range_upper_level"] = event.high
    out.at[row_idx, "range_upper_source_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_upper_source_timestamp"] = event.confirm_ts
    out.at[row_idx, "range_upper_age_bars"] = float(age)
    out.at[row_idx, "range_upper_strength"] = strength
    out.at[row_idx, "range_lower_active"] = boundary_active
    out.at[row_idx, "range_lower_level"] = event.low
    out.at[row_idx, "range_lower_source_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_lower_source_timestamp"] = event.confirm_ts
    out.at[row_idx, "range_lower_age_bars"] = float(age)
    out.at[row_idx, "range_lower_strength"] = strength

    width = max(event.width_abs, EPS)
    out.at[row_idx, "close_position_in_range"] = (close_value - event.low) / width
    out.at[row_idx, "dist_to_range_high_abs"] = event.high - close_value
    out.at[row_idx, "dist_to_range_high_atr"] = (
        (event.high - close_value) / atr_value
        if np.isfinite(atr_value) and atr_value > EPS
        else np.nan
    )
    out.at[row_idx, "dist_to_range_low_abs"] = close_value - event.low
    out.at[row_idx, "dist_to_range_low_atr"] = (
        (close_value - event.low) / atr_value
        if np.isfinite(atr_value) and atr_value > EPS
        else np.nan
    )
    inside = event.low <= close_value <= event.high
    out.at[row_idx, "inside_active_range_flag"] = np.int8(inside)
    out.at[row_idx, "outside_active_range_flag"] = np.int8(not inside)


def _project_terminal_row(
    out: pd.DataFrame,
    *,
    row_idx: int,
    event: _RangeEvent,
) -> None:
    out.at[row_idx, "range_id"] = np.int32(event.range_id)
    out.at[row_idx, "range_birth_idx"] = float(event.birth_idx)
    out.at[row_idx, "range_birth_timestamp"] = event.birth_ts
    out.at[row_idx, "range_confirm_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_confirm_timestamp"] = event.confirm_ts
    out.at[row_idx, "range_active_start_idx"] = float(event.confirm_idx)
    out.at[row_idx, "range_end_idx"] = (
        float(event.end_idx) if event.end_idx is not None else np.nan
    )
    out.at[row_idx, "range_end_timestamp"] = (
        event.end_ts if event.end_idx is not None else pd.NaT
    )
    out.at[row_idx, "range_low"] = event.low
    out.at[row_idx, "range_high"] = event.high
    out.at[row_idx, "range_mid"] = event.mid
    out.at[row_idx, "range_width_abs"] = event.width_abs
    out.at[row_idx, "range_width_atr"] = event.width_atr
    out.at[row_idx, "range_strength_raw"] = event.strength_raw
    out.at[row_idx, "range_strength"] = event.base_strength
    out.at[row_idx, "range_strength_raw_legacy"] = event.strength_raw_legacy
    out.at[row_idx, "range_strength_legacy"] = event.base_strength_legacy
    out.at[row_idx, "range_recent_pressure_imbalance"] = event.recent_pressure_imbalance
    out.at[row_idx, "range_recent_upper_pressure_count"] = float(
        event.recent_upper_pressure_count
    )
    out.at[row_idx, "range_recent_lower_pressure_count"] = float(
        event.recent_lower_pressure_count
    )
    out.at[row_idx, "range_recent_pressure_dominant_side"] = np.int8(
        event.recent_pressure_dominant_side
    )
    out.at[row_idx, "range_recent_equilibrium_score"] = event.recent_equilibrium_score
    out.at[row_idx, "range_recent_expansion_veto_flag"] = np.int8(
        event.recent_expansion_veto_flag
    )
    out.at[row_idx, "range_viability_gate_pass"] = np.int8(event.viability_gate_pass)
    out.at[row_idx, "range_recent_upper_touch_freshness"] = (
        event.recent_upper_touch_freshness
    )
    out.at[row_idx, "range_recent_lower_touch_freshness"] = (
        event.recent_lower_touch_freshness
    )
    out.at[row_idx, "range_recent_two_sided_freshness_score"] = (
        event.recent_two_sided_freshness_score
    )
    out.at[row_idx, "range_strength_structure"] = event.strength_structure
    out.at[row_idx, "range_strength_monitorability"] = event.strength_monitorability
    out.at[row_idx, "range_strength_semantic"] = event.strength_semantic
    out.at[row_idx, "range_strength_formation"] = event.strength_structure
    out.at[row_idx, "range_strength_viability"] = event.strength_viability
    out.at[row_idx, "range_strength_formation_legacy"] = event.strength_formation_legacy
    out.at[row_idx, "range_strength_viability_legacy"] = event.strength_viability_legacy
    out.at[row_idx, "range_touch_quality_score"] = event.touch_quality_score
    out.at[row_idx, "range_containment_quality_score"] = event.containment_quality_score
    out.at[row_idx, "range_stability_quality_score"] = event.stability_quality_score
    out.at[row_idx, "range_state"] = np.int8(event.current_state)
    out.at[row_idx, "range_active"] = np.int8(0)
    out.at[row_idx, "range_accepted_breakout_flag"] = np.int8(
        event.current_state == RANGE_STATE_ACCEPTED_BREAKOUT
    )
    out.at[row_idx, "range_expired_flag"] = np.int8(
        event.current_state == RANGE_STATE_EXPIRED
    )
    out.at[row_idx, "range_superseded_flag"] = np.int8(
        event.current_state == RANGE_STATE_SUPERSEDED
    )
    out.at[row_idx, "range_invalidated_flag"] = np.int8(
        event.current_state == RANGE_STATE_INVALIDATED
    )


def _build_event_table(events: list[_RangeEvent]) -> pd.DataFrame:
    records = []
    for event in events:
        if event.duplicate_suppressed:
            continue
        bars_to_confirm = event.confirm_idx - event.birth_idx
        bars_to_first_breach = (
            event.first_breach_idx - event.confirm_idx
            if event.first_breach_idx is not None
            else np.nan
        )
        bars_to_breakout_accept = (
            event.end_idx - event.confirm_idx
            if event.current_state == RANGE_STATE_ACCEPTED_BREAKOUT
            and event.end_idx is not None
            else np.nan
        )
        records.append(
            {
                "range_id": event.range_id,
                "candidate_lookback_bars": event.candidate_lookback_bars,
                "birth_idx": event.birth_idx,
                "confirm_idx": event.confirm_idx,
                "end_idx": event.end_idx,
                "state": event.current_state,
                "confirm_latency_bars": bars_to_confirm,
                "low": event.low,
                "high": event.high,
                "mid": event.mid,
                "width_abs": event.width_abs,
                "width_atr": event.width_atr,
                "strength_raw": event.strength_raw,
                "strength": event.base_strength,
                "strength_raw_legacy": event.strength_raw_legacy,
                "strength_legacy": event.base_strength_legacy,
                "upper_touches": event.upper_touches,
                "lower_touches": event.lower_touches,
                "close_inside_frac": event.close_inside_frac,
                "boundary_stability_score": event.boundary_stability_score,
                "candidate_boundary_drift_abs": event.boundary_drift_abs,
                "candidate_boundary_drift_atr": event.boundary_drift_atr,
                "confirm_close_position_in_range": event.confirm_close_position_in_range,
                "confirm_dist_to_upper_atr": event.confirm_dist_to_upper_atr,
                "confirm_dist_to_lower_atr": event.confirm_dist_to_lower_atr,
                "confirm_regime": event.confirm_regime,
                "range_recent_pressure_imbalance": event.recent_pressure_imbalance,
                "range_recent_upper_pressure_count": event.recent_upper_pressure_count,
                "range_recent_lower_pressure_count": event.recent_lower_pressure_count,
                "range_recent_pressure_dominant_side": event.recent_pressure_dominant_side,
                "range_recent_equilibrium_score": event.recent_equilibrium_score,
                "range_recent_expansion_veto_flag": event.recent_expansion_veto_flag,
                "range_viability_gate_pass": event.viability_gate_pass,
                "range_recent_upper_touch_freshness": event.recent_upper_touch_freshness,
                "range_recent_lower_touch_freshness": event.recent_lower_touch_freshness,
                "range_recent_two_sided_freshness_score": event.recent_two_sided_freshness_score,
                "range_strength_structure": event.strength_structure,
                "range_strength_monitorability": event.strength_monitorability,
                "range_strength_semantic": event.strength_semantic,
                "range_strength_formation": event.strength_structure,
                "range_strength_viability": event.strength_viability,
                "range_strength_formation_legacy": event.strength_formation_legacy,
                "range_strength_viability_legacy": event.strength_viability_legacy,
                "range_touch_quality_score": event.touch_quality_score,
                "range_containment_quality_score": event.containment_quality_score,
                "range_stability_quality_score": event.stability_quality_score,
                "break_pending_count": event.break_pending_count,
                "reclaimed_count": event.reclaimed_count,
                "total_pending_bars": event.total_pending_bars,
                "mean_pending_duration_bars": (
                    event.total_pending_bars / event.break_pending_count
                    if event.break_pending_count > 0
                    else np.nan
                ),
                "first_breach_idx": event.first_breach_idx,
                "first_breach_side": event.first_breach_side,
                "first_upper_breach_idx": event.first_upper_breach_idx,
                "first_lower_breach_idx": event.first_lower_breach_idx,
                "bars_to_first_breach": bars_to_first_breach,
                "breakout_side_first": event.first_breach_side,
                "accepted_breakout_side": event.accepted_breakout_side,
                "bars_to_breakout_accept": bars_to_breakout_accept,
            }
        )
    return pd.DataFrame.from_records(records)


def _run_range_boundaries(
    df: pd.DataFrame,
    *,
    atr_length: int,
    range_lookback_bars: int,
    candidate_lookback_bars: tuple[int, ...] | None,
    max_width_atr: float,
    edge_tolerance_atr: float,
    min_upper_touches: int,
    min_lower_touches: int,
    min_close_inside_frac: float,
    allowed_wick_overshoot_atr: float,
    max_drift_frac: float,
    min_confirm_bars: int,
    min_candidate_dwell_bars: int,
    boundary_stability_tolerance_atr: float,
    max_active_age_bars: int | None,
    accept_margin_atr: float,
    accept_window_bars: int,
    supersession_overlap_frac: float,
    viability_lookback_bars: int,
    lineage_grace_bars: int,
    min_viability_score: float,
    viability_pressure_weight: float,
    viability_equilibrium_weight: float,
    viability_freshness_weight: float,
    viability_expansion_pressure_weight: float,
    viability_expansion_veto_weight: float,
    final_strength_formation_base: float,
    final_strength_viability_scale: float,
    collect_debug: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    require_columns(
        df,
        {"open", "high", "low", "close"},
        caller="add_range_boundaries",
    )
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    _initialize_columns(out)

    open_ = out["open"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    atr = get_atr_array(out, length=atr_length)
    timestamps = (
        out["timestamp"].apply(_utc_timestamp).to_list()
        if "timestamp" in out.columns
        else [pd.NaT] * len(out)
    )
    regime = (
        pd.to_numeric(out["regime"], errors="coerce").to_numpy()
        if "regime" in out.columns
        else None
    )
    bos_bull = (
        pd.to_numeric(out["bos_bull"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "bos_bull" in out.columns
        else None
    )
    bos_bear = (
        pd.to_numeric(out["bos_bear"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "bos_bear" in out.columns
        else None
    )
    choch_bull = (
        pd.to_numeric(out["choch_bull"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        if "choch_bull" in out.columns
        else None
    )
    choch_bear = (
        pd.to_numeric(out["choch_bear"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        if "choch_bear" in out.columns
        else None
    )

    candidate_lookbacks = tuple(
        sorted(
            {
                int(lookback)
                for lookback in (candidate_lookback_bars or (range_lookback_bars,))
                if int(lookback) >= 2
            }
        )
    )
    candidate_histories: dict[int, list[_CandidateMetrics]] = {
        lookback: [] for lookback in candidate_lookbacks
    }
    events: list[_RangeEvent] = []
    next_range_id = 1
    next_lineage_id = 1
    candidate_lineages: dict[int, dict[str, object]] = {}
    lineage_states: dict[int, dict[str, object]] = {
        lookback: {
            "lineage_start_idx": None,
            "eligible_streak_start": None,
            "streak_confirmed": False,
            "current_lineage_id": None,
            "grace_remaining": 0,
            "pending_grace": False,
            "last_eligible_snapshot": None,
        }
        for lookback in candidate_lookbacks
    }
    weaken_after_age = (
        int(max_active_age_bars * DEFAULT_WEAKEN_AGE_FRAC)
        if max_active_age_bars is not None
        else 24
    )

    for row_idx in range(len(out)):
        for lookback in candidate_lookbacks:
            lineage_state = lineage_states[lookback]
            candidate = _compute_candidate_metrics(
                row_idx=row_idx,
                high=high,
                low=low,
                close=close,
                atr=atr,
                regime=regime,
                bos_bull=bos_bull,
                bos_bear=bos_bear,
                choch_bull=choch_bull,
                choch_bear=choch_bear,
                range_lookback_bars=lookback,
                max_width_atr=max_width_atr,
                edge_tolerance_atr=edge_tolerance_atr,
                min_upper_touches=min_upper_touches,
                min_lower_touches=min_lower_touches,
                min_close_inside_frac=min_close_inside_frac,
                allowed_wick_overshoot_atr=allowed_wick_overshoot_atr,
                max_drift_frac=max_drift_frac,
            )
            candidate_histories[lookback].append(candidate)

            current_lineage_id = lineage_state["current_lineage_id"]
            lineage_start_idx = lineage_state["lineage_start_idx"]
            eligible_streak_start = lineage_state["eligible_streak_start"]
            streak_confirmed = bool(lineage_state["streak_confirmed"])
            pending_grace = bool(lineage_state["pending_grace"])
            grace_remaining = int(lineage_state["grace_remaining"])
            last_eligible_snapshot = lineage_state["last_eligible_snapshot"]

            if candidate.eligible:
                if current_lineage_id is None:
                    lineage_start_idx = row_idx
                    eligible_streak_start = row_idx
                    streak_confirmed = False
                    current_lineage_id = next_lineage_id
                    candidate_lineages[current_lineage_id] = (
                        _new_candidate_lineage_record(
                            lineage_id=current_lineage_id,
                            row_idx=row_idx,
                            timestamp=timestamps[row_idx],
                            candidate_lookback_bars=lookback,
                        )
                    )
                    next_lineage_id += 1
                elif pending_grace:
                    continuation_ok = False
                    if last_eligible_snapshot is not None and np.isfinite(atr[row_idx]):
                        overlap_frac = _interval_overlap_frac(
                            float(last_eligible_snapshot["low"]),
                            float(last_eligible_snapshot["high"]),
                            candidate.low,
                            candidate.high,
                        )
                        width_change_atr = abs(
                            candidate.width_abs
                            - float(last_eligible_snapshot["width_abs"])
                        ) / max(atr[row_idx], EPS)
                        continuation_ok = (
                            overlap_frac >= 0.75 and width_change_atr <= 0.35
                        )
                    if not continuation_ok:
                        _finalize_candidate_lineage(
                            candidate_lineages,
                            current_lineage_id,
                            min_candidate_dwell_bars=min_candidate_dwell_bars,
                            eligibility_failure=True,
                            same_lineage_continuation_fail=True,
                        )
                        lineage_start_idx = row_idx
                        streak_confirmed = False
                        current_lineage_id = next_lineage_id
                        candidate_lineages[current_lineage_id] = (
                            _new_candidate_lineage_record(
                                lineage_id=current_lineage_id,
                                row_idx=row_idx,
                                timestamp=timestamps[row_idx],
                                candidate_lookback_bars=lookback,
                            )
                        )
                        next_lineage_id += 1
                    eligible_streak_start = row_idx
                    pending_grace = False
                    grace_remaining = 0
                if current_lineage_id is not None:
                    candidate_lineages[current_lineage_id]["last_idx"] = row_idx
                    candidate_lineages[current_lineage_id]["last_timestamp"] = (
                        timestamps[row_idx]
                    )
                last_eligible_snapshot = {
                    "high": candidate.high,
                    "low": candidate.low,
                    "width_abs": candidate.width_abs,
                }
            else:
                if (
                    current_lineage_id is not None
                    and not streak_confirmed
                    and grace_remaining < lineage_grace_bars
                ):
                    pending_grace = True
                    grace_remaining += 1
                    eligible_streak_start = None
                else:
                    _finalize_candidate_lineage(
                        candidate_lineages,
                        current_lineage_id,
                        min_candidate_dwell_bars=min_candidate_dwell_bars,
                        eligibility_failure=current_lineage_id is not None
                        and not streak_confirmed,
                    )
                    lineage_start_idx = None
                    eligible_streak_start = None
                    streak_confirmed = False
                    current_lineage_id = None
                    pending_grace = False
                    grace_remaining = 0
                    last_eligible_snapshot = None

            if (
                candidate.eligible
                and lineage_start_idx is not None
                and eligible_streak_start is not None
                and not streak_confirmed
                and row_idx - eligible_streak_start + 1 >= min_confirm_bars
                and row_idx - lineage_start_idx + 1 >= min_candidate_dwell_bars
            ):
                recent = candidate_histories[lookback][
                    row_idx - min_confirm_bars + 1 : row_idx + 1
                ]
                if all(item.eligible for item in recent):
                    atr_now = atr[row_idx]
                    tol_abs = max(boundary_stability_tolerance_atr * atr_now, EPS)
                    highs = [item.high for item in recent]
                    lows = [item.low for item in recent]
                    stable = (
                        max(highs) - min(highs) <= tol_abs
                        and max(lows) - min(lows) <= tol_abs
                    )
                    if current_lineage_id is not None and not stable:
                        candidate_lineages[current_lineage_id][
                            "failed_boundary_stability"
                        ] = 1
                    if stable and candidate.high > candidate.low:
                        if current_lineage_id is not None:
                            candidate_lineages[current_lineage_id][
                                "maturity_pass_flag"
                            ] = 1
                            candidate_lineages[current_lineage_id][
                                "maturity_pass_idx"
                            ] = row_idx
                            candidate_lineages[current_lineage_id][
                                "maturity_pass_timestamp"
                            ] = timestamps[row_idx]
                        stability_score = _boundary_stability_score(
                            highs, lows, tol_abs
                        )
                        boundary_drift_abs, boundary_drift_atr = _boundary_drift(
                            highs,
                            lows,
                            atr_now,
                        )
                        formation_raw, formation_strength, formation_components = (
                            _compute_strength(
                                upper_touches=candidate.upper_touches,
                                lower_touches=candidate.lower_touches,
                                close_inside_frac=candidate.close_inside_frac,
                                width_atr=candidate.width_atr,
                                max_width_atr=max_width_atr,
                                compression_score=candidate.compression_score,
                                boundary_stability_score=stability_score,
                                drift_frac=candidate.drift_frac,
                                structure_penalty=candidate.structure_penalty,
                            )
                        )
                        viability_metrics = _compute_viability_metrics(
                            row_idx=row_idx,
                            candidate_high=candidate.high,
                            candidate_low=candidate.low,
                            width_abs=candidate.width_abs,
                            high=high,
                            low=low,
                            close=close,
                            open_=open_,
                            atr=atr,
                            range_lookback_bars=lookback,
                            viability_lookback_bars=viability_lookback_bars,
                            edge_tolerance_atr=edge_tolerance_atr,
                            min_viability_score=min_viability_score,
                            pressure_weight=viability_pressure_weight,
                            equilibrium_weight=viability_equilibrium_weight,
                            freshness_weight=viability_freshness_weight,
                            expansion_pressure_weight=viability_expansion_pressure_weight,
                            expansion_veto_weight=viability_expansion_veto_weight,
                        )
                        if current_lineage_id is not None:
                            candidate_lineages[current_lineage_id].update(
                                {
                                    "range_recent_expansion_veto_flag": viability_metrics[
                                        "recent_expansion_veto_flag"
                                    ],
                                    "range_viability_gate_pass": viability_metrics[
                                        "viability_gate_pass"
                                    ],
                                    "range_recent_pressure_imbalance": viability_metrics[
                                        "recent_pressure_imbalance"
                                    ],
                                    "range_recent_equilibrium_score": viability_metrics[
                                        "recent_equilibrium_score"
                                    ],
                                    "range_recent_two_sided_freshness_score": viability_metrics[
                                        "recent_two_sided_freshness_score"
                                    ],
                                    "range_recent_upper_pressure_count": viability_metrics[
                                        "recent_upper_pressure_count"
                                    ],
                                    "range_recent_lower_pressure_count": viability_metrics[
                                        "recent_lower_pressure_count"
                                    ],
                                    "range_recent_pressure_dominant_side": viability_metrics[
                                        "recent_pressure_dominant_side"
                                    ],
                                    "range_strength_viability": viability_metrics[
                                        "strength_viability"
                                    ],
                                    "viability_fail_due_to_pressure": viability_metrics[
                                        "pressure_threshold_fail"
                                    ],
                                    "viability_fail_due_to_equilibrium": viability_metrics[
                                        "equilibrium_threshold_fail"
                                    ],
                                    "viability_fail_due_to_freshness": viability_metrics[
                                        "freshness_threshold_fail"
                                    ],
                                    "viability_fail_due_to_score_threshold": viability_metrics[
                                        "score_threshold_fail"
                                    ],
                                    "viability_fail_multiple_reasons": viability_metrics[
                                        "multiple_reasons_fail"
                                    ],
                                    "expansion_veto_seen_flag": max(
                                        int(
                                            candidate_lineages[current_lineage_id][
                                                "expansion_veto_seen_flag"
                                            ]
                                        ),
                                        int(
                                            viability_metrics[
                                                "recent_expansion_veto_flag"
                                            ]
                                        ),
                                    ),
                                    "low_viability_reject_seen_flag": max(
                                        int(
                                            candidate_lineages[current_lineage_id][
                                                "low_viability_reject_seen_flag"
                                            ]
                                        ),
                                        int(
                                            not bool(
                                                viability_metrics["viability_gate_pass"]
                                            )
                                        ),
                                    ),
                                }
                            )
                        if bool(viability_metrics["viability_gate_pass"]):
                            if current_lineage_id is not None:
                                candidate_lineages[current_lineage_id][
                                    "viability_pass_flag"
                                ] = 1
                                candidate_lineages[current_lineage_id][
                                    "viability_pass_idx"
                                ] = row_idx
                                candidate_lineages[current_lineage_id][
                                    "viability_pass_timestamp"
                                ] = timestamps[row_idx]

                            strength_viability_legacy = float(
                                viability_metrics["strength_viability"]
                            )
                            strength_raw_legacy = formation_raw * (
                                final_strength_formation_base
                                + final_strength_viability_scale
                                * strength_viability_legacy
                            )
                            strength_legacy = _clip01(strength_raw_legacy)
                            rebased_strengths = _compute_rebased_ranking(
                                birth_idx=int(lineage_start_idx),
                                confirm_idx=row_idx,
                                upper_touches=candidate.upper_touches,
                                lower_touches=candidate.lower_touches,
                                width_atr=candidate.width_atr,
                                touch_quality_score=float(
                                    formation_components["touch_quality_score"]
                                ),
                                containment_quality_score=float(
                                    formation_components["containment_quality_score"]
                                ),
                                stability_quality_score=float(
                                    formation_components["stability_quality_score"]
                                ),
                                pressure_imbalance=float(
                                    viability_metrics["recent_pressure_imbalance"]
                                ),
                                equilibrium_score=float(
                                    viability_metrics["recent_equilibrium_score"]
                                ),
                                two_sided_freshness=float(
                                    viability_metrics[
                                        "recent_two_sided_freshness_score"
                                    ]
                                ),
                                expansion_veto_flag=int(
                                    viability_metrics["recent_expansion_veto_flag"]
                                ),
                                legacy_viability=float(strength_viability_legacy),
                                legacy_structure=float(formation_strength),
                            )
                            strength_raw = float(
                                rebased_strengths["strength_final_rebased"]
                            )
                            strength = _clip01(strength_raw)
                            event = _RangeEvent(
                                range_id=next_range_id,
                                candidate_lookback_bars=lookback,
                                birth_idx=int(lineage_start_idx),
                                birth_ts=timestamps[int(lineage_start_idx)],
                                confirm_idx=row_idx,
                                confirm_ts=timestamps[row_idx],
                                low=candidate.low,
                                high=candidate.high,
                                mid=(candidate.low + candidate.high) / 2.0,
                                width_abs=candidate.width_abs,
                                width_atr=candidate.width_atr,
                                upper_touches=candidate.upper_touches,
                                lower_touches=candidate.lower_touches,
                                close_inside_frac=candidate.close_inside_frac,
                                wick_overshoot_atr=candidate.wick_overshoot_atr,
                                drift_frac=candidate.drift_frac,
                                boundary_stability_score=stability_score,
                                boundary_drift_abs=boundary_drift_abs,
                                boundary_drift_atr=boundary_drift_atr,
                                compression_score=candidate.compression_score,
                                strength_raw=strength_raw,
                                base_strength=strength,
                                strength_raw_legacy=float(strength_raw_legacy),
                                base_strength_legacy=float(strength_legacy),
                                confirm_close_position_in_range=(
                                    (close[row_idx] - candidate.low)
                                    / max(candidate.width_abs, EPS)
                                ),
                                confirm_dist_to_upper_atr=(
                                    (candidate.high - close[row_idx]) / atr_now
                                    if atr_now > EPS
                                    else np.nan
                                ),
                                confirm_dist_to_lower_atr=(
                                    (close[row_idx] - candidate.low) / atr_now
                                    if atr_now > EPS
                                    else np.nan
                                ),
                                confirm_regime=(
                                    int(regime[row_idx])
                                    if regime is not None
                                    and np.isfinite(regime[row_idx])
                                    else None
                                ),
                                recent_pressure_imbalance=float(
                                    viability_metrics["recent_pressure_imbalance"]
                                ),
                                recent_upper_pressure_count=int(
                                    viability_metrics["recent_upper_pressure_count"]
                                ),
                                recent_lower_pressure_count=int(
                                    viability_metrics["recent_lower_pressure_count"]
                                ),
                                recent_pressure_dominant_side=int(
                                    viability_metrics["recent_pressure_dominant_side"]
                                ),
                                recent_equilibrium_score=float(
                                    viability_metrics["recent_equilibrium_score"]
                                ),
                                recent_expansion_veto_flag=int(
                                    viability_metrics["recent_expansion_veto_flag"]
                                ),
                                viability_gate_pass=int(
                                    viability_metrics["viability_gate_pass"]
                                ),
                                recent_upper_touch_freshness=float(
                                    viability_metrics["recent_upper_touch_freshness"]
                                ),
                                recent_lower_touch_freshness=float(
                                    viability_metrics["recent_lower_touch_freshness"]
                                ),
                                recent_two_sided_freshness_score=float(
                                    viability_metrics[
                                        "recent_two_sided_freshness_score"
                                    ]
                                ),
                                strength_structure=float(
                                    rebased_strengths["strength_structure"]
                                ),
                                strength_monitorability=float(
                                    rebased_strengths["strength_monitorability"]
                                ),
                                strength_semantic=float(
                                    rebased_strengths["strength_semantic"]
                                ),
                                strength_formation=float(
                                    rebased_strengths["strength_structure"]
                                ),
                                strength_viability=float(
                                    rebased_strengths["strength_viability_rebased"]
                                ),
                                strength_formation_legacy=float(formation_strength),
                                strength_viability_legacy=float(
                                    strength_viability_legacy
                                ),
                                touch_quality_score=float(
                                    formation_components["touch_quality_score"]
                                ),
                                containment_quality_score=float(
                                    formation_components["containment_quality_score"]
                                ),
                                stability_quality_score=float(
                                    formation_components["stability_quality_score"]
                                ),
                            )

                            duplicate_event = _find_duplicate_event(
                                events,
                                event,
                                atr_value=atr_now,
                            )
                            if duplicate_event is not None:
                                if _event_duplicate_key(event) > _event_duplicate_key(
                                    duplicate_event
                                ):
                                    duplicate_event.duplicate_suppressed = True
                                    _reset_range_row(out, duplicate_event.confirm_idx)
                                else:
                                    if current_lineage_id is not None:
                                        candidate_lineages[current_lineage_id][
                                            "duplicate_suppressed_flag"
                                        ] = 1
                                    streak_confirmed = True
                                    lineage_state.update(
                                        {
                                            "lineage_start_idx": lineage_start_idx,
                                            "eligible_streak_start": eligible_streak_start,
                                            "streak_confirmed": streak_confirmed,
                                            "current_lineage_id": current_lineage_id,
                                            "grace_remaining": grace_remaining,
                                            "pending_grace": pending_grace,
                                            "last_eligible_snapshot": last_eligible_snapshot,
                                        }
                                    )
                                    continue

                            if current_lineage_id is not None:
                                candidate_lineages[current_lineage_id][
                                    "confirmed_flag"
                                ] = 1
                                candidate_lineages[current_lineage_id][
                                    "range_confirm_idx"
                                ] = row_idx
                                candidate_lineages[current_lineage_id][
                                    "range_confirm_timestamp"
                                ] = timestamps[row_idx]
                            for older in events:
                                if older.duplicate_suppressed:
                                    continue
                                if older.current_state not in {
                                    RANGE_STATE_ACTIVE_INTACT,
                                    RANGE_STATE_ACTIVE_WEAKENED,
                                    RANGE_STATE_BROKEN_UNACCEPTED,
                                }:
                                    continue
                                overlap_frac = _interval_overlap_frac(
                                    older.low,
                                    older.high,
                                    event.low,
                                    event.high,
                                )
                                dominates = (
                                    event.base_strength >= older.base_strength
                                    or (
                                        event.width_atr <= older.width_atr
                                        and event.base_strength + 0.05
                                        >= older.base_strength
                                    )
                                )
                                if (
                                    overlap_frac >= supersession_overlap_frac
                                    and dominates
                                ):
                                    older.current_state = RANGE_STATE_SUPERSEDED
                                    older.end_idx = row_idx
                                    older.end_ts = timestamps[row_idx]
                            events.append(event)
                            _stamp_detect_row(out, event)
                            next_range_id += 1
                            streak_confirmed = True

            lineage_state.update(
                {
                    "lineage_start_idx": lineage_start_idx,
                    "eligible_streak_start": eligible_streak_start,
                    "streak_confirmed": streak_confirmed,
                    "current_lineage_id": current_lineage_id,
                    "grace_remaining": grace_remaining,
                    "pending_grace": pending_grace,
                    "last_eligible_snapshot": last_eligible_snapshot,
                }
            )

        for event in events:
            if event.duplicate_suppressed:
                continue
            if event.current_state in {
                RANGE_STATE_ACCEPTED_BREAKOUT,
                RANGE_STATE_INVALIDATED,
                RANGE_STATE_EXPIRED,
                RANGE_STATE_SUPERSEDED,
            }:
                continue
            if row_idx <= event.confirm_idx:
                continue

            age = row_idx - event.confirm_idx
            if max_active_age_bars is not None and age >= max_active_age_bars:
                event.current_state = RANGE_STATE_EXPIRED
                event.end_idx = row_idx
                event.end_ts = timestamps[row_idx]
                continue

            tol_abs = max(
                edge_tolerance_atr
                * (atr[row_idx] if np.isfinite(atr[row_idx]) else 0.0),
                EPS,
            )
            if high[row_idx] >= event.high - tol_abs:
                event.upper_probe_count += 1
            if low[row_idx] <= event.low + tol_abs:
                event.lower_probe_count += 1

            accepted_up = (
                np.isfinite(atr[row_idx])
                and close[row_idx] >= event.high + accept_margin_atr * atr[row_idx]
            )
            accepted_down = (
                np.isfinite(atr[row_idx])
                and close[row_idx] <= event.low - accept_margin_atr * atr[row_idx]
            )
            if accepted_up and accepted_down:
                if event.first_breach_idx is None:
                    event.first_breach_idx = row_idx
                    event.first_breach_side = 0
                event.current_state = RANGE_STATE_INVALIDATED
                event.end_idx = row_idx
                event.end_ts = timestamps[row_idx]
                continue
            if accepted_up or accepted_down:
                if event.first_breach_idx is None:
                    event.first_breach_idx = row_idx
                    event.first_breach_side = 1 if accepted_up else -1
                if accepted_up and event.first_upper_breach_idx is None:
                    event.first_upper_breach_idx = row_idx
                if accepted_down and event.first_lower_breach_idx is None:
                    event.first_lower_breach_idx = row_idx
                if event.pending_since_idx is not None:
                    event.total_pending_bars += row_idx - event.pending_since_idx + 1
                event.accepted_breakout_side = 1 if accepted_up else -1
                event.current_state = RANGE_STATE_ACCEPTED_BREAKOUT
                event.end_idx = row_idx
                event.end_ts = timestamps[row_idx]
                continue

            breach_up = high[row_idx] > event.high
            breach_down = low[row_idx] < event.low
            if breach_up and breach_down:
                if event.first_breach_idx is None:
                    event.first_breach_idx = row_idx
                    event.first_breach_side = 0
                event.current_state = RANGE_STATE_INVALIDATED
                event.end_idx = row_idx
                event.end_ts = timestamps[row_idx]
                continue

            if event.pending_side != 0:
                if event.low <= close[row_idx] <= event.high:
                    if event.pending_since_idx is not None:
                        event.total_pending_bars += (
                            row_idx - event.pending_since_idx + 1
                        )
                    event.pending_side = 0
                    event.pending_since_idx = None
                    event.reclaimed_count += 1
                    event.current_state = RANGE_STATE_ACTIVE_WEAKENED
                else:
                    pending_age = row_idx - int(event.pending_since_idx or row_idx) + 1
                    if pending_age >= accept_window_bars:
                        event.current_state = RANGE_STATE_ACCEPTED_BREAKOUT
                        event.end_idx = row_idx
                        event.end_ts = timestamps[row_idx]
                    else:
                        event.current_state = RANGE_STATE_BROKEN_UNACCEPTED
                continue

            if breach_up or breach_down:
                if event.first_breach_idx is None:
                    event.first_breach_idx = row_idx
                    event.first_breach_side = (
                        1
                        if breach_up and not breach_down
                        else (-1 if breach_down and not breach_up else 0)
                    )
                if breach_up and event.first_upper_breach_idx is None:
                    event.first_upper_breach_idx = row_idx
                if breach_down and event.first_lower_breach_idx is None:
                    event.first_lower_breach_idx = row_idx
                event.break_pending_count += 1
                event.pending_side = 1 if breach_up else -1
                event.pending_since_idx = row_idx
                event.current_state = RANGE_STATE_BROKEN_UNACCEPTED
                continue

            if (
                event.reclaimed_count > 0
                or age >= weaken_after_age
                or (event.upper_probe_count + event.lower_probe_count)
                >= DEFAULT_WEAKEN_PROBE_COUNT
            ):
                event.current_state = RANGE_STATE_ACTIVE_WEAKENED
            else:
                event.current_state = RANGE_STATE_ACTIVE_INTACT

        for event in events:
            if event.duplicate_suppressed:
                continue
            _stamp_terminal_on_confirm_row(out, event)

        terminal_events = [
            event
            for event in events
            if not event.duplicate_suppressed
            and event.end_idx == row_idx
            and event.current_state
            in {
                RANGE_STATE_ACCEPTED_BREAKOUT,
                RANGE_STATE_INVALIDATED,
                RANGE_STATE_EXPIRED,
                RANGE_STATE_SUPERSEDED,
            }
        ]
        terminal_events.sort(
            key=lambda event: (
                event.current_state == RANGE_STATE_SUPERSEDED,
                event.current_state == RANGE_STATE_EXPIRED,
                event.current_state == RANGE_STATE_INVALIDATED,
                event.current_state == RANGE_STATE_ACCEPTED_BREAKOUT,
                event.range_id,
            ),
            reverse=True,
        )
        selected = _select_dense_active(
            events,
            row_idx=row_idx,
            max_active_age_bars=max_active_age_bars or DEFAULT_MAX_ACTIVE_AGE_BARS,
        )
        _project_dense_active(
            out,
            row_idx=row_idx,
            event=selected,
            close_value=close[row_idx],
            atr_value=atr[row_idx],
            max_active_age_bars=max_active_age_bars or DEFAULT_MAX_ACTIVE_AGE_BARS,
        )
        if selected is None and terminal_events:
            _project_terminal_row(out, row_idx=row_idx, event=terminal_events[0])

    candidate_table = pd.DataFrame.from_records(list(candidate_lineages.values()))
    if not candidate_table.empty:
        candidate_table = candidate_table.sort_values(
            ["candidate_lookback_bars", "candidate_lineage_id"]
        )
    debug = {
        "event_table": _build_event_table(events),
        "candidate_table": candidate_table,
    }
    return out, debug


def add_range_boundaries(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    range_lookback_bars: int = DEFAULT_RANGE_LOOKBACK_BARS,
    candidate_lookback_bars: tuple[int, ...] | None = DEFAULT_CANDIDATE_LOOKBACK_BARS,
    max_width_atr: float = DEFAULT_MAX_WIDTH_ATR,
    edge_tolerance_atr: float = DEFAULT_EDGE_TOLERANCE_ATR,
    min_upper_touches: int = DEFAULT_MIN_UPPER_TOUCHES,
    min_lower_touches: int = DEFAULT_MIN_LOWER_TOUCHES,
    min_close_inside_frac: float = DEFAULT_MIN_CLOSE_INSIDE_FRAC,
    allowed_wick_overshoot_atr: float = DEFAULT_ALLOWED_WICK_OVERSHOOT_ATR,
    max_drift_frac: float = DEFAULT_MAX_DRIFT_FRAC,
    min_confirm_bars: int = DEFAULT_MIN_CONFIRM_BARS,
    min_candidate_dwell_bars: int = DEFAULT_MIN_CANDIDATE_DWELL_BARS,
    boundary_stability_tolerance_atr: float = DEFAULT_BOUNDARY_STABILITY_TOLERANCE_ATR,
    max_active_age_bars: int | None = DEFAULT_MAX_ACTIVE_AGE_BARS,
    accept_margin_atr: float = DEFAULT_ACCEPT_MARGIN_ATR,
    accept_window_bars: int = DEFAULT_ACCEPT_WINDOW_BARS,
    supersession_overlap_frac: float = DEFAULT_SUPERSESSION_OVERLAP_FRAC,
    viability_lookback_bars: int = DEFAULT_VIABILITY_LOOKBACK_BARS,
    lineage_grace_bars: int = DEFAULT_LINEAGE_GRACE_BARS,
    min_viability_score: float = DEFAULT_MIN_VIABILITY_SCORE,
    viability_pressure_weight: float = DEFAULT_VIABILITY_PRESSURE_WEIGHT,
    viability_equilibrium_weight: float = DEFAULT_VIABILITY_EQUILIBRIUM_WEIGHT,
    viability_freshness_weight: float = DEFAULT_VIABILITY_FRESHNESS_WEIGHT,
    viability_expansion_pressure_weight: float = DEFAULT_VIABILITY_EXPANSION_PRESSURE_WEIGHT,
    viability_expansion_veto_weight: float = DEFAULT_VIABILITY_EXPANSION_VETO_WEIGHT,
    final_strength_formation_base: float = DEFAULT_FINAL_STRENGTH_FORMATION_BASE,
    final_strength_viability_scale: float = DEFAULT_FINAL_STRENGTH_VIABILITY_SCALE,
) -> pd.DataFrame:
    """Add the frozen canonical range-boundary source family."""
    result, _debug = _run_range_boundaries(
        df,
        atr_length=atr_length,
        range_lookback_bars=range_lookback_bars,
        candidate_lookback_bars=candidate_lookback_bars,
        max_width_atr=max_width_atr,
        edge_tolerance_atr=edge_tolerance_atr,
        min_upper_touches=min_upper_touches,
        min_lower_touches=min_lower_touches,
        min_close_inside_frac=min_close_inside_frac,
        allowed_wick_overshoot_atr=allowed_wick_overshoot_atr,
        max_drift_frac=max_drift_frac,
        min_confirm_bars=min_confirm_bars,
        min_candidate_dwell_bars=min_candidate_dwell_bars,
        boundary_stability_tolerance_atr=boundary_stability_tolerance_atr,
        max_active_age_bars=max_active_age_bars,
        accept_margin_atr=accept_margin_atr,
        accept_window_bars=accept_window_bars,
        supersession_overlap_frac=supersession_overlap_frac,
        viability_lookback_bars=viability_lookback_bars,
        lineage_grace_bars=lineage_grace_bars,
        min_viability_score=min_viability_score,
        viability_pressure_weight=viability_pressure_weight,
        viability_equilibrium_weight=viability_equilibrium_weight,
        viability_freshness_weight=viability_freshness_weight,
        viability_expansion_pressure_weight=viability_expansion_pressure_weight,
        viability_expansion_veto_weight=viability_expansion_veto_weight,
        final_strength_formation_base=final_strength_formation_base,
        final_strength_viability_scale=final_strength_viability_scale,
        collect_debug=False,
    )
    return result


def collect_range_boundary_debug_tables(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    range_lookback_bars: int = DEFAULT_RANGE_LOOKBACK_BARS,
    candidate_lookback_bars: tuple[int, ...] | None = DEFAULT_CANDIDATE_LOOKBACK_BARS,
    max_width_atr: float = DEFAULT_MAX_WIDTH_ATR,
    edge_tolerance_atr: float = DEFAULT_EDGE_TOLERANCE_ATR,
    min_upper_touches: int = DEFAULT_MIN_UPPER_TOUCHES,
    min_lower_touches: int = DEFAULT_MIN_LOWER_TOUCHES,
    min_close_inside_frac: float = DEFAULT_MIN_CLOSE_INSIDE_FRAC,
    allowed_wick_overshoot_atr: float = DEFAULT_ALLOWED_WICK_OVERSHOOT_ATR,
    max_drift_frac: float = DEFAULT_MAX_DRIFT_FRAC,
    min_confirm_bars: int = DEFAULT_MIN_CONFIRM_BARS,
    min_candidate_dwell_bars: int = DEFAULT_MIN_CANDIDATE_DWELL_BARS,
    boundary_stability_tolerance_atr: float = DEFAULT_BOUNDARY_STABILITY_TOLERANCE_ATR,
    max_active_age_bars: int | None = DEFAULT_MAX_ACTIVE_AGE_BARS,
    accept_margin_atr: float = DEFAULT_ACCEPT_MARGIN_ATR,
    accept_window_bars: int = DEFAULT_ACCEPT_WINDOW_BARS,
    supersession_overlap_frac: float = DEFAULT_SUPERSESSION_OVERLAP_FRAC,
    viability_lookback_bars: int = DEFAULT_VIABILITY_LOOKBACK_BARS,
    lineage_grace_bars: int = DEFAULT_LINEAGE_GRACE_BARS,
    min_viability_score: float = DEFAULT_MIN_VIABILITY_SCORE,
    viability_pressure_weight: float = DEFAULT_VIABILITY_PRESSURE_WEIGHT,
    viability_equilibrium_weight: float = DEFAULT_VIABILITY_EQUILIBRIUM_WEIGHT,
    viability_freshness_weight: float = DEFAULT_VIABILITY_FRESHNESS_WEIGHT,
    viability_expansion_pressure_weight: float = DEFAULT_VIABILITY_EXPANSION_PRESSURE_WEIGHT,
    viability_expansion_veto_weight: float = DEFAULT_VIABILITY_EXPANSION_VETO_WEIGHT,
    final_strength_formation_base: float = DEFAULT_FINAL_STRENGTH_FORMATION_BASE,
    final_strength_viability_scale: float = DEFAULT_FINAL_STRENGTH_VIABILITY_SCALE,
) -> dict[str, pd.DataFrame]:
    result, debug = _run_range_boundaries(
        df,
        atr_length=atr_length,
        range_lookback_bars=range_lookback_bars,
        candidate_lookback_bars=candidate_lookback_bars,
        max_width_atr=max_width_atr,
        edge_tolerance_atr=edge_tolerance_atr,
        min_upper_touches=min_upper_touches,
        min_lower_touches=min_lower_touches,
        min_close_inside_frac=min_close_inside_frac,
        allowed_wick_overshoot_atr=allowed_wick_overshoot_atr,
        max_drift_frac=max_drift_frac,
        min_confirm_bars=min_confirm_bars,
        min_candidate_dwell_bars=min_candidate_dwell_bars,
        boundary_stability_tolerance_atr=boundary_stability_tolerance_atr,
        max_active_age_bars=max_active_age_bars,
        accept_margin_atr=accept_margin_atr,
        accept_window_bars=accept_window_bars,
        supersession_overlap_frac=supersession_overlap_frac,
        viability_lookback_bars=viability_lookback_bars,
        lineage_grace_bars=lineage_grace_bars,
        min_viability_score=min_viability_score,
        viability_pressure_weight=viability_pressure_weight,
        viability_equilibrium_weight=viability_equilibrium_weight,
        viability_freshness_weight=viability_freshness_weight,
        viability_expansion_pressure_weight=viability_expansion_pressure_weight,
        viability_expansion_veto_weight=viability_expansion_veto_weight,
        final_strength_formation_base=final_strength_formation_base,
        final_strength_viability_scale=final_strength_viability_scale,
        collect_debug=True,
    )
    debug["frame"] = result
    return debug
