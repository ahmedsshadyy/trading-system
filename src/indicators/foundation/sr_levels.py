"""
foundation/sr_levels.py

Canonical support/resistance level engine.

Architecture
------------
Support and resistance are unified in one engine.  They differ only by
``side`` (−1 = support, +1 = resistance).  Everything else — source
creation, lifecycle, strength, invalidation, retirement, deduplication —
is shared logic.

Causality doctrine
------------------
A level may be active at row ``t`` only if its source event was already
live-known at ``t``.  No backfilling.

Public API
----------
- ``add_sr_levels(df, *, include_research_only)``   ← main entry point
- ``extract_sr_source_events(df)``
- ``build_sr_level_registry(df)``
- ``update_sr_lifecycle(df, registry)``
- ``project_sr_context(df, registry)``
- ``add_sr_research_columns(df, registry)``

All public functions are pure with respect to the input DataFrame (they
return a new copy and never mutate the caller's frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.event_ids import SequentialEventIdAllocator

__all__ = [
    "add_sr_levels",
    "extract_sr_source_events",
    "build_sr_level_registry",
    "update_sr_lifecycle",
    "project_sr_context",
    "add_sr_research_columns",
    # State constants
    "SR_STATE_INACTIVE_PRE_LIVE",
    "SR_STATE_ACTIVE",
    "SR_STATE_INVALIDATED",
    "SR_STATE_RETIRED",
    # Side constants
    "SR_SIDE_SUPPORT",
    "SR_SIDE_RESISTANCE",
    # Family constants
    "SR_FAMILY_SWING",
    "SR_FAMILY_EQHL",
    "SR_FAMILY_SESSION",
    "SR_FAMILY_DAY",
    "SR_FAMILY_WEEK",
    "SR_FAMILY_VP",
    # Level class
    "SRLevel",
]

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

SR_STATE_INACTIVE_PRE_LIVE = 0
SR_STATE_ACTIVE = 1
SR_STATE_INVALIDATED = 2
SR_STATE_RETIRED = 3

# Side constants: support = −1, resistance = +1
SR_SIDE_SUPPORT = -1
SR_SIDE_RESISTANCE = +1

# Source-family constants
SR_FAMILY_SWING = "swing"
SR_FAMILY_EQHL = "eqhl"
SR_FAMILY_SESSION = "session"
SR_FAMILY_DAY = "day"
SR_FAMILY_WEEK = "week"
SR_FAMILY_VP = "vp"

# ---------------------------------------------------------------------------
# Frozen hyper-parameters (see spec §5 / §8 / §9 / §10)
# ---------------------------------------------------------------------------

# Strength weights
_W1: float = 0.40  # source strength
_W2: float = 0.20  # touch strength
_W3: float = 0.20  # freshness strength
_W4: float = 0.20  # family prior
_WEAKEN_PENALTY_RATE: float = 0.20  # per weaken event

_FRESHNESS_TAU: float = 100.0  # bars time-constant for freshness decay

# Fixed family priors (week > day > eqhl > swing > session/vp)
FAMILY_PRIOR: dict[str, float] = {
    SR_FAMILY_SWING: 0.60,
    SR_FAMILY_EQHL: 0.55,
    SR_FAMILY_SESSION: 0.50,
    SR_FAMILY_DAY: 0.65,
    SR_FAMILY_WEEK: 0.80,
    SR_FAMILY_VP: 0.50,
}

# Touch / invalidation tolerances (in ATR units)
TOUCH_TOL_ATR: float = 0.15
INVALIDATION_BUFFER_ATR: float = 0.10

# Retirement limits
MAX_AGE_BARS: int = 500  # default for swing / eqhl
INVALIDATED_RETIRE_BARS: int = 50
MAX_WEAKEN_COUNT: int = 4

# Per-family hard expiry (bars after activation).  Overrides MAX_AGE_BARS when smaller.
# At H4: session~18 bars=3 sessions, day~30 bars=5 days, week~120 bars=4 weeks.
FAMILY_MAX_AGE: dict[str, int] = {
    SR_FAMILY_SESSION: 30,  # ~5 trading days
    SR_FAMILY_DAY: 60,  # ~10 trading days
    SR_FAMILY_WEEK: 120,  # ~4 weeks
    SR_FAMILY_VP: 200,  # ~33 trading days
    SR_FAMILY_SWING: 500,
    SR_FAMILY_EQHL: 500,
}

# Deduplication merge tolerance
MERGE_TOL_ATR: float = 0.20

# ---------------------------------------------------------------------------
# Prior-period source family map
# ---------------------------------------------------------------------------

# (column_name, side, family, sub_label)
_PRIOR_PERIOD_SOURCES: list[tuple[str, int, str, str]] = [
    ("prev_day_high", SR_SIDE_RESISTANCE, SR_FAMILY_DAY, "prior_day_high"),
    ("prev_day_low", SR_SIDE_SUPPORT, SR_FAMILY_DAY, "prior_day_low"),
    ("prev_week_high", SR_SIDE_RESISTANCE, SR_FAMILY_WEEK, "prior_week_high"),
    ("prev_week_low", SR_SIDE_SUPPORT, SR_FAMILY_WEEK, "prior_week_low"),
    ("prev_asia_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_asia_high"),
    ("prev_asia_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_asia_low"),
    ("prev_london_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_london_high"),
    ("prev_london_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_london_low"),
    ("prev_ny_high", SR_SIDE_RESISTANCE, SR_FAMILY_SESSION, "prior_ny_high"),
    ("prev_ny_low", SR_SIDE_SUPPORT, SR_FAMILY_SESSION, "prior_ny_low"),
]

_VP_SOURCES: list[tuple[str, int, str, str]] = [
    ("vp_vah", SR_SIDE_RESISTANCE, SR_FAMILY_VP, "vp_vah"),
    ("vp_val", SR_SIDE_SUPPORT, SR_FAMILY_VP, "vp_val"),
]

# ---------------------------------------------------------------------------
# Canonical level object
# ---------------------------------------------------------------------------


@dataclass
class SRLevel:
    """One canonical horizontal S/R level with full lifecycle state."""

    # --- Source identity (immutable after construction) ---
    level_id: int
    side: int  # SR_SIDE_SUPPORT or SR_SIDE_RESISTANCE
    level_price: float
    source_family: str
    source_sub: str = ""
    source_origin_idx: int = -1
    source_confirm_idx: int = -1
    source_live_from_idx: int = -1
    source_origin_ts: Any = None
    source_confirm_ts: Any = None
    source_live_from_ts: Any = None
    source_strength_initial: float = 0.50
    source_metadata_norm: dict = field(default_factory=dict)

    # --- Mutable lifecycle state (updated during forward pass) ---
    state: int = field(default=SR_STATE_INACTIVE_PRE_LIVE)
    age_bars: int = field(default=0)
    touch_count: int = field(default=0)
    refresh_count: int = field(default=0)
    weaken_count: int = field(default=0)
    last_touch_idx: int = field(default=-1)
    invalidation_idx: int = field(default=-1)
    invalidation_ts: Any = field(default=None)
    bars_since_invalidation: int = field(default=0)
    superseded_by: Optional[int] = field(default=None)
    retirement_idx: int = field(default=-1)  # bar at which state became RETIRED

    # --- Derived ---
    level_strength: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Strength model  (spec §8)
# ---------------------------------------------------------------------------


def _compute_strength(level: SRLevel) -> float:
    """Compute level_strength ∈ [0,1] from decomposed components."""
    src_s = float(np.clip(level.source_strength_initial, 0.0, 1.0))
    touch_s = float(np.clip(min(level.touch_count, 4) / 4.0, 0.0, 1.0))
    fresh_s = float(math.exp(-level.age_bars / _FRESHNESS_TAU))
    fam_p = FAMILY_PRIOR.get(level.source_family, 0.50)
    weaken_p = float(np.clip(_WEAKEN_PENALTY_RATE * level.weaken_count, 0.0, 1.0))
    raw = _W1 * src_s + _W2 * touch_s + _W3 * fresh_s + _W4 * fam_p - weaken_p
    return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def _get_timestamps(df: pd.DataFrame) -> list:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return ts.tolist()
    return [None] * len(df)


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------


def _extract_swing_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    required = {
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
    }
    if not required.issubset(df.columns):
        return []

    n = len(df)
    sh_flag = df["swing_high_confirm_flag"].to_numpy()
    sl_flag = df["swing_low_confirm_flag"].to_numpy()
    sh_price = df["swing_high_confirm_price"].to_numpy(dtype=float)
    sl_price = df["swing_low_confirm_price"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    has_sh_orig = "swing_high_confirm_origin_idx" in df.columns
    has_sl_orig = "swing_low_confirm_origin_idx" in df.columns
    sh_origin_arr = (
        df["swing_high_confirm_origin_idx"].to_numpy(dtype=float)
        if has_sh_orig
        else np.full(n, np.nan)
    )
    sl_origin_arr = (
        df["swing_low_confirm_origin_idx"].to_numpy(dtype=float)
        if has_sl_orig
        else np.full(n, np.nan)
    )

    levels: list[SRLevel] = []
    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0

        # Resistance from confirmed swing high
        if sh_flag[i] and np.isfinite(sh_price[i]):
            orig_idx = int(sh_origin_arr[i]) if np.isfinite(sh_origin_arr[i]) else i
            orig_idx = max(0, min(orig_idx, n - 1))
            mag_atr = (
                abs(sh_price[i] - close[orig_idx]) / atr_i
                if orig_idx != i and np.isfinite(close[orig_idx])
                else 1.5
            )
            src_strength = float(np.clip(mag_atr / 3.0, 0.0, 1.0))
            levels.append(
                SRLevel(
                    level_id=allocator.allocate(),
                    side=SR_SIDE_RESISTANCE,
                    level_price=float(sh_price[i]),
                    source_family=SR_FAMILY_SWING,
                    source_sub="swing_high",
                    source_origin_idx=orig_idx,
                    source_confirm_idx=i,
                    source_live_from_idx=i,
                    source_origin_ts=timestamps[orig_idx],
                    source_confirm_ts=timestamps[i],
                    source_live_from_ts=timestamps[i],
                    source_strength_initial=src_strength,
                    source_metadata_norm={"magnitude_atr": round(mag_atr, 4)},
                )
            )

        # Support from confirmed swing low
        if sl_flag[i] and np.isfinite(sl_price[i]):
            orig_idx = int(sl_origin_arr[i]) if np.isfinite(sl_origin_arr[i]) else i
            orig_idx = max(0, min(orig_idx, n - 1))
            mag_atr = (
                abs(sl_price[i] - close[orig_idx]) / atr_i
                if orig_idx != i and np.isfinite(close[orig_idx])
                else 1.5
            )
            src_strength = float(np.clip(mag_atr / 3.0, 0.0, 1.0))
            levels.append(
                SRLevel(
                    level_id=allocator.allocate(),
                    side=SR_SIDE_SUPPORT,
                    level_price=float(sl_price[i]),
                    source_family=SR_FAMILY_SWING,
                    source_sub="swing_low",
                    source_origin_idx=orig_idx,
                    source_confirm_idx=i,
                    source_live_from_idx=i,
                    source_origin_ts=timestamps[orig_idx],
                    source_confirm_ts=timestamps[i],
                    source_live_from_ts=timestamps[i],
                    source_strength_initial=src_strength,
                    source_metadata_norm={"magnitude_atr": round(mag_atr, 4)},
                )
            )

    return levels


def _extract_eqhl_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    eqh_ok = {"eqh_detect_flag", "eqh_level_on_detect", "eqh_origin_idx"}.issubset(
        df.columns
    )
    eql_ok = {"eql_detect_flag", "eql_level_on_detect", "eql_origin_idx"}.issubset(
        df.columns
    )
    if not eqh_ok and not eql_ok:
        return []

    n = len(df)
    levels: list[SRLevel] = []

    if eqh_ok:
        eqh_flag = df["eqh_detect_flag"].to_numpy()
        eqh_price = df["eqh_level_on_detect"].to_numpy(dtype=float)
        eqh_orig = df["eqh_origin_idx"].to_numpy(dtype=float)
        eqh_score = (
            df["eqh_score_on_detect"].to_numpy(dtype=float)
            if "eqh_score_on_detect" in df.columns
            else np.full(n, 0.5)
        )
        eqh_count = (
            df["eqh_member_count_on_detect"].to_numpy(dtype=float)
            if "eqh_member_count_on_detect" in df.columns
            else np.full(n, 2.0)
        )
        for i in range(n):
            if eqh_flag[i] and np.isfinite(eqh_price[i]):
                orig_idx = int(eqh_orig[i]) if np.isfinite(eqh_orig[i]) else i
                orig_idx = max(0, min(orig_idx, n - 1))
                src_s = (
                    float(np.clip(eqh_score[i], 0.0, 1.0))
                    if np.isfinite(eqh_score[i])
                    else 0.5
                )
                levels.append(
                    SRLevel(
                        level_id=allocator.allocate(),
                        side=SR_SIDE_RESISTANCE,
                        level_price=float(eqh_price[i]),
                        source_family=SR_FAMILY_EQHL,
                        source_sub="eqh",
                        source_origin_idx=orig_idx,
                        source_confirm_idx=i,
                        source_live_from_idx=i,
                        source_origin_ts=timestamps[orig_idx],
                        source_confirm_ts=timestamps[i],
                        source_live_from_ts=timestamps[i],
                        source_strength_initial=src_s,
                        source_metadata_norm={
                            "score": round(float(eqh_score[i]), 4),
                            "member_count": int(eqh_count[i]),
                        },
                    )
                )

    if eql_ok:
        eql_flag = df["eql_detect_flag"].to_numpy()
        eql_price = df["eql_level_on_detect"].to_numpy(dtype=float)
        eql_orig = df["eql_origin_idx"].to_numpy(dtype=float)
        eql_score = (
            df["eql_score_on_detect"].to_numpy(dtype=float)
            if "eql_score_on_detect" in df.columns
            else np.full(n, 0.5)
        )
        eql_count = (
            df["eql_member_count_on_detect"].to_numpy(dtype=float)
            if "eql_member_count_on_detect" in df.columns
            else np.full(n, 2.0)
        )
        for i in range(n):
            if eql_flag[i] and np.isfinite(eql_price[i]):
                orig_idx = int(eql_orig[i]) if np.isfinite(eql_orig[i]) else i
                orig_idx = max(0, min(orig_idx, n - 1))
                src_s = (
                    float(np.clip(eql_score[i], 0.0, 1.0))
                    if np.isfinite(eql_score[i])
                    else 0.5
                )
                levels.append(
                    SRLevel(
                        level_id=allocator.allocate(),
                        side=SR_SIDE_SUPPORT,
                        level_price=float(eql_price[i]),
                        source_family=SR_FAMILY_EQHL,
                        source_sub="eql",
                        source_origin_idx=orig_idx,
                        source_confirm_idx=i,
                        source_live_from_idx=i,
                        source_origin_ts=timestamps[orig_idx],
                        source_confirm_ts=timestamps[i],
                        source_live_from_ts=timestamps[i],
                        source_strength_initial=src_s,
                        source_metadata_norm={
                            "score": round(float(eql_score[i]), 4),
                            "member_count": int(eql_count[i]),
                        },
                    )
                )

    return levels


def _extract_prior_period_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    """One level per unique prior-period value per source column."""
    n = len(df)
    levels: list[SRLevel] = []

    for col, side, family, sub in _PRIOR_PERIOD_SOURCES:
        if col not in df.columns:
            continue
        series = df[col].to_numpy(dtype=float)
        prev_val = np.nan

        for i in range(n):
            val = series[i]
            if not np.isfinite(val):
                continue
            # Create a level only when the value first appears or changes.
            if np.isnan(prev_val) or val != prev_val:
                src_s = FAMILY_PRIOR.get(family, 0.50)
                levels.append(
                    SRLevel(
                        level_id=allocator.allocate(),
                        side=side,
                        level_price=float(val),
                        source_family=family,
                        source_sub=sub,
                        source_origin_idx=i,
                        source_confirm_idx=i,
                        source_live_from_idx=i,
                        source_origin_ts=timestamps[i],
                        source_confirm_ts=timestamps[i],
                        source_live_from_ts=timestamps[i],
                        source_strength_initial=src_s,
                        source_metadata_norm={"sub": sub},
                    )
                )
            prev_val = val

    return levels


def _extract_vp_levels(
    df: pd.DataFrame,
    atr: np.ndarray,
    timestamps: list,
    allocator: SequentialEventIdAllocator,
) -> list[SRLevel]:
    """VP levels: new level when VAH/VAL shifts by > MERGE_TOL_ATR * atr."""
    n = len(df)
    levels: list[SRLevel] = []

    for col, side, family, sub in _VP_SOURCES:
        if col not in df.columns:
            continue
        series = df[col].to_numpy(dtype=float)
        prev_val = np.nan

        for i in range(n):
            val = series[i]
            if not np.isfinite(val):
                continue
            atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
            is_new = np.isnan(prev_val) or abs(val - prev_val) > MERGE_TOL_ATR * atr_i
            if is_new:
                src_s = FAMILY_PRIOR.get(family, 0.50)
                levels.append(
                    SRLevel(
                        level_id=allocator.allocate(),
                        side=side,
                        level_price=float(val),
                        source_family=family,
                        source_sub=sub,
                        source_origin_idx=i,
                        source_confirm_idx=i,
                        source_live_from_idx=i,
                        source_origin_ts=timestamps[i],
                        source_confirm_ts=timestamps[i],
                        source_live_from_ts=timestamps[i],
                        source_strength_initial=src_s,
                        source_metadata_norm={"sub": sub},
                    )
                )
                prev_val = val

    return levels


# ---------------------------------------------------------------------------
# Public: source-event extraction
# ---------------------------------------------------------------------------


def extract_sr_source_events(df: pd.DataFrame) -> list[SRLevel]:
    """Collect all causal source events from the upstream indicator columns.

    Returns a list of ``SRLevel`` objects in the order they would be
    activated (by ``source_live_from_idx``).  The registry is not built
    yet — all levels start in ``SR_STATE_INACTIVE_PRE_LIVE``.
    """
    if len(df) == 0:
        return []
    atr = get_atr_array(df)
    tss = _get_timestamps(df)
    alloc = SequentialEventIdAllocator()
    out: list[SRLevel] = []
    out.extend(_extract_swing_levels(df, atr, tss, alloc))
    out.extend(_extract_eqhl_levels(df, atr, tss, alloc))
    out.extend(_extract_prior_period_levels(df, atr, tss, alloc))
    out.extend(_extract_vp_levels(df, atr, tss, alloc))
    return out


def build_sr_level_registry(df: pd.DataFrame) -> dict[int, SRLevel]:
    """Build the level registry: ``{level_id: SRLevel}``."""
    return {lev.level_id: lev for lev in extract_sr_source_events(df)}


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def _deduplicate_at_activation(
    registry: dict[int, SRLevel],
    new_lid: int,
    atr_i: float,
    active_set: set[int],
    activation_row: int,
) -> None:
    """Retire the weaker of two nearby same-side active levels."""
    new_lev = registry[new_lid]
    if new_lev.state != SR_STATE_ACTIVE:
        return
    for other_lid in list(active_set):
        if other_lid == new_lid:
            continue
        other = registry[other_lid]
        if other.side != new_lev.side:
            continue
        dist = abs(other.level_price - new_lev.level_price)
        if dist <= MERGE_TOL_ATR * atr_i:
            if new_lev.level_strength >= other.level_strength:
                other.state = SR_STATE_RETIRED
                other.superseded_by = new_lid
                other.retirement_idx = activation_row
                active_set.discard(other_lid)
            else:
                new_lev.state = SR_STATE_RETIRED
                new_lev.superseded_by = other_lid
                new_lev.retirement_idx = activation_row
                return  # new level retired — stop


# ---------------------------------------------------------------------------
# Core lifecycle + projection pass
# ---------------------------------------------------------------------------


def update_sr_lifecycle(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
) -> dict[str, np.ndarray]:
    """Run the forward lifecycle pass.

    Mutates ``registry`` in place (state, touch counts, etc.) and returns
    a dict of row-indexed numpy arrays that form the projection output.
    """
    n = len(df)
    atr = get_atr_array(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    # Build row → [level_ids to activate] map
    activation_map: dict[int, list[int]] = {}
    for lid, lev in registry.items():
        activation_map.setdefault(lev.source_live_from_idx, []).append(lid)

    # Allocate output arrays
    ctx: dict[str, np.ndarray] = {
        "nearest_support_price": np.full(n, np.nan),
        "nearest_support_distance": np.full(n, np.nan),
        "nearest_support_distance_atr": np.full(n, np.nan),
        "nearest_support_age_bars": np.full(n, np.nan),
        "nearest_support_strength": np.full(n, np.nan),
        "nearest_support_source_family": np.full(n, None, dtype=object),
        "nearest_support_source_idx": np.full(n, np.nan),
        "nearest_support_touch_count": np.full(n, np.nan),
        "nearest_support_refresh_count": np.full(n, np.nan),
        "nearest_support_weaken_count": np.full(n, np.nan),
        "nearest_support_active": np.zeros(n, dtype=np.int8),
        "nearest_resistance_price": np.full(n, np.nan),
        "nearest_resistance_distance": np.full(n, np.nan),
        "nearest_resistance_distance_atr": np.full(n, np.nan),
        "nearest_resistance_age_bars": np.full(n, np.nan),
        "nearest_resistance_strength": np.full(n, np.nan),
        "nearest_resistance_source_family": np.full(n, None, dtype=object),
        "nearest_resistance_source_idx": np.full(n, np.nan),
        "nearest_resistance_touch_count": np.full(n, np.nan),
        "nearest_resistance_refresh_count": np.full(n, np.nan),
        "nearest_resistance_weaken_count": np.full(n, np.nan),
        "nearest_resistance_active": np.zeros(n, dtype=np.int8),
        "inside_sr_band_flag": np.zeros(n, dtype=np.int8),
        "between_nearest_sr_flag": np.zeros(n, dtype=np.int8),
        "above_nearest_resistance_flag": np.zeros(n, dtype=np.int8),
        "below_nearest_support_flag": np.zeros(n, dtype=np.int8),
        "support_broken_this_bar": np.zeros(n, dtype=np.int8),
        "resistance_broken_this_bar": np.zeros(n, dtype=np.int8),
        "active_support_count": np.zeros(n, dtype=np.int32),
        "active_resistance_count": np.zeros(n, dtype=np.int32),
        "support_cluster_density_atr": np.full(n, np.nan),
        "resistance_cluster_density_atr": np.full(n, np.nan),
    }

    # Track only currently active / invalidated level IDs — avoids scanning
    # the full registry (which grows monotonically) on every bar.
    active_set: set[int] = set()
    invalidated_set: set[int] = set()

    ts_col = "timestamp" in df.columns

    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        cl = close[i]
        hi = high[i]
        lo = low[i]
        cl_finite = np.isfinite(cl)

        # ── Step 1: activate levels whose live-from row is i ──────────────
        for lid in activation_map.get(i, []):
            lev = registry[lid]
            if lev.state == SR_STATE_INACTIVE_PRE_LIVE:
                lev.state = SR_STATE_ACTIVE
                active_set.add(lid)
                lev.level_strength = _compute_strength(lev)
                _deduplicate_at_activation(registry, lid, atr_i, active_set, i)

        if not cl_finite:
            # Skip lifecycle update for NaN close rows but still project NaN
            continue

        # ── Step 2: process each active level ─────────────────────────────
        newly_invalidated: list[int] = []
        for lid in list(active_set):
            lev = registry[lid]

            lev.age_bars += 1

            # --- invalidation ---
            if lev.side == SR_SIDE_SUPPORT:
                invalidated = cl < lev.level_price - INVALIDATION_BUFFER_ATR * atr_i
            else:
                invalidated = cl > lev.level_price + INVALIDATION_BUFFER_ATR * atr_i

            if invalidated:
                lev.state = SR_STATE_INVALIDATED
                lev.invalidation_idx = i
                if ts_col:
                    lev.invalidation_ts = df["timestamp"].iloc[i]
                if lev.side == SR_SIDE_SUPPORT:
                    ctx["support_broken_this_bar"][i] = 1
                else:
                    ctx["resistance_broken_this_bar"][i] = 1
                newly_invalidated.append(lid)
                continue

            # --- touch / weaken ---
            if lev.side == SR_SIDE_SUPPORT:
                in_touch_zone = (
                    np.isfinite(lo) and lo <= lev.level_price + TOUCH_TOL_ATR * atr_i
                )
                weak_pierce = (
                    np.isfinite(lo) and lo < lev.level_price - TOUCH_TOL_ATR * atr_i
                )
                close_holds = cl >= lev.level_price - INVALIDATION_BUFFER_ATR * atr_i
            else:
                in_touch_zone = (
                    np.isfinite(hi) and hi >= lev.level_price - TOUCH_TOL_ATR * atr_i
                )
                weak_pierce = (
                    np.isfinite(hi) and hi > lev.level_price + TOUCH_TOL_ATR * atr_i
                )
                close_holds = cl <= lev.level_price + INVALIDATION_BUFFER_ATR * atr_i

            if in_touch_zone and not weak_pierce and close_holds:
                lev.touch_count += 1
                lev.refresh_count += 1
                lev.last_touch_idx = i
            elif weak_pierce and close_holds:
                lev.weaken_count += 1

            lev.level_strength = _compute_strength(lev)

        # Flush newly invalidated out of active_set into invalidated_set
        for lid in newly_invalidated:
            active_set.discard(lid)
            invalidated_set.add(lid)

        # ── Step 3: tick bars_since_invalidation ──────────────────────────
        for lid in invalidated_set:
            registry[lid].bars_since_invalidation += 1

        # ── Step 4: retirement ────────────────────────────────────────────
        to_retire: list[int] = []
        for lid in active_set:
            lev = registry[lid]
            max_age = FAMILY_MAX_AGE.get(lev.source_family, MAX_AGE_BARS)
            if lev.age_bars > max_age or lev.weaken_count >= MAX_WEAKEN_COUNT:
                lev.state = SR_STATE_RETIRED
                lev.retirement_idx = i
                to_retire.append(lid)
        for lid in to_retire:
            active_set.discard(lid)

        to_retire_inv: list[int] = []
        for lid in invalidated_set:
            lev = registry[lid]
            if lev.bars_since_invalidation > INVALIDATED_RETIRE_BARS:
                lev.state = SR_STATE_RETIRED
                lev.retirement_idx = i
                to_retire_inv.append(lid)
        for lid in to_retire_inv:
            invalidated_set.discard(lid)

        # ── Step 5: projection ────────────────────────────────────────────
        act_sups = [
            registry[lid] for lid in active_set if registry[lid].side == SR_SIDE_SUPPORT
        ]
        act_res = [
            registry[lid]
            for lid in active_set
            if registry[lid].side == SR_SIDE_RESISTANCE
        ]
        ctx["active_support_count"][i] = len(act_sups)
        ctx["active_resistance_count"][i] = len(act_res)

        # Nearest support: highest active price ≤ close
        valid_sups = [lev for lev in act_sups if lev.level_price <= cl]
        if valid_sups:
            ns = max(valid_sups, key=lambda lv: lv.level_price)
            dist = cl - ns.level_price
            ctx["nearest_support_price"][i] = ns.level_price
            ctx["nearest_support_distance"][i] = dist
            ctx["nearest_support_distance_atr"][i] = dist / atr_i
            ctx["nearest_support_age_bars"][i] = float(ns.age_bars)
            ctx["nearest_support_strength"][i] = ns.level_strength
            ctx["nearest_support_source_family"][i] = ns.source_family
            ctx["nearest_support_source_idx"][i] = float(ns.source_confirm_idx)
            ctx["nearest_support_touch_count"][i] = float(ns.touch_count)
            ctx["nearest_support_refresh_count"][i] = float(ns.refresh_count)
            ctx["nearest_support_weaken_count"][i] = float(ns.weaken_count)
            ctx["nearest_support_active"][i] = 1

        # Nearest resistance: lowest active price ≥ close
        valid_res = [lev for lev in act_res if lev.level_price >= cl]
        if valid_res:
            nr = min(valid_res, key=lambda lv: lv.level_price)
            dist = nr.level_price - cl
            ctx["nearest_resistance_price"][i] = nr.level_price
            ctx["nearest_resistance_distance"][i] = dist
            ctx["nearest_resistance_distance_atr"][i] = dist / atr_i
            ctx["nearest_resistance_age_bars"][i] = float(nr.age_bars)
            ctx["nearest_resistance_strength"][i] = nr.level_strength
            ctx["nearest_resistance_source_family"][i] = nr.source_family
            ctx["nearest_resistance_source_idx"][i] = float(nr.source_confirm_idx)
            ctx["nearest_resistance_touch_count"][i] = float(nr.touch_count)
            ctx["nearest_resistance_refresh_count"][i] = float(nr.refresh_count)
            ctx["nearest_resistance_weaken_count"][i] = float(nr.weaken_count)
            ctx["nearest_resistance_active"][i] = 1

        # Context flags
        ns_active = ctx["nearest_support_active"][i]
        nr_active = ctx["nearest_resistance_active"][i]
        ns_price = ctx["nearest_support_price"][i]
        nr_price = ctx["nearest_resistance_price"][i]

        # between: close is sandwiched (both exist — by selection they bound close)
        if ns_active and nr_active:
            ctx["between_nearest_sr_flag"][i] = 1

        # above all resistance: no active resistance at or above close
        if not nr_active:
            ctx["above_nearest_resistance_flag"][i] = 1

        # below all support: no active support at or below close
        if not ns_active:
            ctx["below_nearest_support_flag"][i] = 1

        # inside band: within touch_tol of nearest support or resistance
        if ns_active and abs(cl - ns_price) <= TOUCH_TOL_ATR * atr_i:
            ctx["inside_sr_band_flag"][i] = 1
        if nr_active and abs(cl - nr_price) <= TOUCH_TOL_ATR * atr_i:
            ctx["inside_sr_band_flag"][i] = 1

        # Cluster density
        if len(act_sups) >= 2:
            sup_prices = [lev.level_price for lev in act_sups]
            ctx["support_cluster_density_atr"][i] = float(np.std(sup_prices)) / atr_i
        if len(act_res) >= 2:
            res_prices = [lev.level_price for lev in act_res]
            ctx["resistance_cluster_density_atr"][i] = float(np.std(res_prices)) / atr_i

    return ctx


# ---------------------------------------------------------------------------
# Public: project_sr_context
# ---------------------------------------------------------------------------


def project_sr_context(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
) -> pd.DataFrame:
    """Run lifecycle pass and return a copy of ``df`` with context columns.

    .. warning::
        This mutates ``registry``.  Call ``build_sr_level_registry`` first
        to get a fresh registry.
    """
    ctx = update_sr_lifecycle(df, registry)
    out = df.copy()
    for col, arr in ctx.items():
        out[col] = arr
    return out


# ---------------------------------------------------------------------------
# Public: research extras
# ---------------------------------------------------------------------------


def add_sr_research_columns(
    df: pd.DataFrame,
    registry: dict[int, SRLevel],
) -> pd.DataFrame:
    """Append ``r_`` prefixed research columns to ``df`` (copy).

    Must be called *after* ``project_sr_context`` so the registry state
    reflects the final end-of-frame lifecycle.
    """
    out = df.copy()

    active_ids = sorted(
        lev.level_id for lev in registry.values() if lev.state == SR_STATE_ACTIVE
    )

    # Most recently invalidated support
    inv_sups = [
        lev
        for lev in registry.values()
        if lev.state in (SR_STATE_INVALIDATED, SR_STATE_RETIRED)
        and lev.side == SR_SIDE_SUPPORT
        and lev.invalidation_idx >= 0
    ]
    recent_inv_sup_id: Any = np.nan
    if inv_sups:
        recent_inv_sup_id = float(
            max(inv_sups, key=lambda lv: lv.invalidation_idx).level_id
        )

    # Most recently invalidated resistance
    inv_res = [
        lev
        for lev in registry.values()
        if lev.state in (SR_STATE_INVALIDATED, SR_STATE_RETIRED)
        and lev.side == SR_SIDE_RESISTANCE
        and lev.invalidation_idx >= 0
    ]
    recent_inv_res_id: Any = np.nan
    if inv_res:
        recent_inv_res_id = float(
            max(inv_res, key=lambda lv: lv.invalidation_idx).level_id
        )

    out["r_all_active_level_ids"] = str(active_ids)
    out["r_recent_invalidated_support_id"] = recent_inv_sup_id
    out["r_recent_invalidated_resistance_id"] = recent_inv_res_id
    out["r_level_registry_snapshot_count"] = len(registry)
    return out


# ---------------------------------------------------------------------------
# Public: master builder
# ---------------------------------------------------------------------------


def add_sr_levels(
    df: pd.DataFrame,
    *,
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add canonical support/resistance context to ``df``.

    Parameters
    ----------
    df : DataFrame
        Must contain ``high``, ``low``, ``close``.
        ``atr_14`` is computed on-the-fly if absent.
        Upstream indicator columns (swings, session, value, VP) are consumed
        when present; missing families are silently skipped.
    include_research_only : bool
        Include ``r_`` prefixed research columns.

    Returns
    -------
    New DataFrame with S/R context columns added.  Input is never mutated.

    Available source families (consumed automatically when columns exist)
    -------------------------------------------------------------------
    * ``swing``   — confirmed swing high/low
    * ``eqhl``    — EQH/EQL cluster detect events
    * ``day``     — prev_day_high / prev_day_low
    * ``week``    — prev_week_high / prev_week_low
    * ``session`` — prev_asia/london/ny high/low
    * ``vp``      — vp_vah / vp_val (rolling volume profile)
    """
    if df is None:
        raise ValueError("add_sr_levels: df must not be None")
    if len(df) == 0:
        return df.copy()
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"add_sr_levels: missing required column '{col}'")

    registry = build_sr_level_registry(df)
    out = project_sr_context(df, registry)

    if include_research_only:
        out = add_sr_research_columns(out, registry)

    return out
