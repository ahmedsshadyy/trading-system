"""Step 10 — Unified liquidity source framework.

Single normalized interface for every sweepable level the production sweeps v1
detector consumes. Source families (frozen for v1):

* swing_high / swing_low                — confirmed swings
* equal_high / equal_low                — equal-highs/lows clusters
* resistance / support                  — S/R registry zones
* session_high / session_low            — Asia / London / NY prev-session H/L
* previous_day_high / previous_day_low  — calendar prev-day H/L
* previous_week_high / previous_week_low — calendar prev-week H/L

Explicitly excluded for v1 (kept in code, marked deprecated): range_boundary,
FVG edges, OB edges. See :mod:`src.indicators.foundation.range_boundaries`
for the deprecation note.

Output contract
---------------
The function emits **dense per-bar ladder columns** projecting the top-K
nearest active liquidity clusters per side. The ladder depth is
:data:`LIQ_LADDER_DEPTH` (5 above + 5 below). Each slot exposes the cluster's
identity (id, primary family, attribution), geometry (level, zone bounds,
width), state, timing, and quality. Sweeps consumes this ladder.

The function also returns a sparse **clusters audit table** as a sidecar (via
:func:`build_unified_liquidity_clusters_audit`) for validation. The audit
table has one row per (bar, cluster) and is suitable for CSV export, chart
overlays, and golden tests.

All sources are stamped with ``source_timeframe = scan_timeframe``. The MTF
policy (Step 9) is enforced at the boundary: see
:func:`src.indicators.sweeps_v2.mtf_policy.assert_same_timeframe_sources`.

Causality contract
------------------
Every source row satisfies
``source_origin_idx <= source_confirm_idx <= source_active_start_idx <= current_idx``
or sets the missing component to ``-1`` (unknown). The detector relies on the
upstream stages (swings, equal_hl, sr_levels, session, prev_day_hl,
prev_week_hl) to provide live-safe values, then attaches one consistent
schema on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.indicators.sweeps_v2.mtf_policy import (
    SWEEP_MTF_POLICY,
    assert_known_timeframe,
    assert_same_timeframe_sources,
)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: Ladder depth per side. Top N nearest active clusters above price + below.
LIQ_LADDER_DEPTH: int = 5

#: Global cap on active clusters retained per bar (across both sides).
LIQ_GLOBAL_CROWDING_CAP: int = 12

#: Dedup tolerance: clusters with same side and level within this fraction of
#: ATR are merged.
LIQ_DEDUP_ATR_FRAC: float = 0.10

#: Per-instrument minimum tick (price step). Used as a floor for the dedup
#: tolerance so clusters at very low ATR still merge sensibly.
LIQ_MIN_TICK: dict[str, float] = {
    "XAU_USD": 0.01,
    "USOIL": 0.001,
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.001,
    "AUD_USD": 0.0001,
    "NZD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "USD_CHF": 0.0001,
    "DXY": 0.01,
}

#: Default min-tick if instrument is unknown.
LIQ_MIN_TICK_DEFAULT: float = 0.0001

#: Multiplier on min-tick that floors the dedup tolerance.
LIQ_MIN_TICK_MULT: int = 5

#: Lifecycle states (frozen integer enum).
LIQ_STATE_UNAVAILABLE: int = 0
LIQ_STATE_BORN: int = 1
LIQ_STATE_ACTIVE: int = 2
LIQ_STATE_WEAKENED: int = 3
LIQ_STATE_CONSUMED_SWEPT: int = 4
LIQ_STATE_INVALIDATED: int = 5
LIQ_STATE_RETIRED: int = 6

#: Source side convention.
LIQ_SIDE_BUY_LIQ_ABOVE: int = +1  # buy-side liquidity above price
LIQ_SIDE_SELL_LIQ_BELOW: int = -1  # sell-side liquidity below price

#: Source families (canonical strings used in attribution).
LIQ_SOURCE_FAMILIES: tuple[str, ...] = (
    "swing_high",
    "swing_low",
    "equal_high",
    "equal_low",
    "resistance",
    "support",
    "session_high",
    "session_low",
    "previous_day_high",
    "previous_day_low",
    "previous_week_high",
    "previous_week_low",
)

#: Family sweep-side (which sweep direction each family seeds).
LIQ_FAMILY_SIDE: dict[str, int] = {
    "swing_high": +1,
    "equal_high": +1,
    "resistance": +1,
    "session_high": +1,
    "previous_day_high": +1,
    "previous_week_high": +1,
    "swing_low": -1,
    "equal_low": -1,
    "support": -1,
    "session_low": -1,
    "previous_day_low": -1,
    "previous_week_low": -1,
}

#: Precedence rank within a cluster (lower number wins). Mirrors the spec:
#: equal_hl → S/R → prev_week → prev_day → session → swing.
LIQ_FAMILY_PRECEDENCE: dict[str, int] = {
    "equal_high": 1,
    "equal_low": 1,
    "resistance": 2,
    "support": 2,
    "previous_week_high": 3,
    "previous_week_low": 3,
    "previous_day_high": 4,
    "previous_day_low": 4,
    "session_high": 5,
    "session_low": 5,
    "swing_high": 6,
    "swing_low": 6,
}

#: Calibrated default strength per family. Used as the source row's
#: ``source_strength`` only when the upstream stage does not provide one.
LIQ_FAMILY_DEFAULT_STRENGTH: dict[str, float] = {
    "swing_high": 0.40,
    "swing_low": 0.40,
    "equal_high": 0.65,
    "equal_low": 0.65,
    "resistance": 0.60,
    "support": 0.60,
    "session_high": 0.45,
    "session_low": 0.45,
    "previous_day_high": 0.50,
    "previous_day_low": 0.50,
    "previous_week_high": 0.55,
    "previous_week_low": 0.55,
}

#: Range boundary is explicitly excluded from production sources. The
#: detector will hard-fail if it ever encounters a row tagged with this
#: family — this is the runtime guard against accidental re-introduction.
LIQ_DEPRECATED_FAMILIES: frozenset[str] = frozenset(
    {
        "range_boundary_high",
        "range_boundary_low",
        "fvg_high",
        "fvg_low",
        "ob_high",
        "ob_low",
    }
)


# ---------------------------------------------------------------------------
# Schema — dense per-bar ladder columns
# ---------------------------------------------------------------------------

_LADDER_FIELDS: tuple[str, ...] = (
    "cluster_id",
    "primary_family",
    "level",
    "zone_low",
    "zone_high",
    "is_zone",
    "width_abs",
    "width_atr",
    "strength",
    "state",
    "age_bars",
    "freshness",
    "touch_count",
    "signed_dist_atr",
    "attribution_families",
    "member_count",
    "origin_idx",
    "active_start_idx",
)

_LADDER_SIDES: tuple[str, ...] = ("above", "below")


def _ladder_columns() -> list[str]:
    cols: list[str] = []
    for side in _LADDER_SIDES:
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            for field_name in _LADDER_FIELDS:
                cols.append(f"liq_{side}_l{rank}_{field_name}")
    return cols


_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "liq_active_total_count",
    "liq_active_above_count",
    "liq_active_below_count",
    "liq_dropped_by_crowding_count",
    "liq_dropped_by_dominance_count",
    "liq_nearest_above_dist_atr",
    "liq_nearest_below_dist_atr",
    "liq_top_above_strength",
    "liq_top_below_strength",
    "liq_source_timeframe",
    "liq_mtf_policy",
)

#: Canonical, ordered list of every column the unified-sources stage emits.
UNIFIED_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    list(_AGGREGATE_COLUMNS) + _ladder_columns()
)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Source:
    family: str
    side: int
    level: float
    zone_low: float
    zone_high: float
    is_zone: bool
    strength: float
    state: int
    origin_idx: int
    active_start_idx: int
    age_bars: int
    touch_count: int


@dataclass(slots=True)
class _Cluster:
    side: int
    primary_family: str
    members: list[_Source] = field(default_factory=list)
    level: float = float("nan")
    zone_low: float = float("nan")
    zone_high: float = float("nan")
    is_zone: bool = False
    width_abs: float = float("nan")
    width_atr: float = float("nan")
    strength: float = 0.0
    state: int = LIQ_STATE_UNAVAILABLE
    origin_idx: int = -1
    active_start_idx: int = -1
    age_bars: int = 0
    touch_count: int = 0
    attribution_families: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Per-family extractors — read upstream live-safe columns
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(f):
        return float("nan")
    return f


def _safe_int(value: object, default: int = -1) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return int(f)


def _resolve_min_tick(instrument: str | None) -> float:
    if instrument is None:
        return LIQ_MIN_TICK_DEFAULT
    return LIQ_MIN_TICK.get(str(instrument).upper(), LIQ_MIN_TICK_DEFAULT)


def _has_columns(df: pd.DataFrame, names: Iterable[str]) -> bool:
    return all(col in df.columns for col in names)


def _extract_swing_sources(
    df: pd.DataFrame,
    *,
    bar_idx: int,
) -> list[_Source]:
    out: list[_Source] = []
    if _has_columns(df, ("last_swing_high", "last_swing_high_idx", "swing_high_age")):
        level = _safe_float(df["last_swing_high"].iat[bar_idx])
        if math.isfinite(level):
            origin = _safe_int(df["last_swing_high_idx"].iat[bar_idx])
            age = _safe_int(df["swing_high_age"].iat[bar_idx], default=0)
            confirm_idx = bar_idx - max(0, age)
            out.append(
                _Source(
                    family="swing_high",
                    side=+1,
                    level=level,
                    zone_low=level,
                    zone_high=level,
                    is_zone=False,
                    strength=LIQ_FAMILY_DEFAULT_STRENGTH["swing_high"],
                    state=LIQ_STATE_ACTIVE,
                    origin_idx=origin,
                    active_start_idx=confirm_idx,
                    age_bars=max(0, age),
                    touch_count=0,
                )
            )
    if _has_columns(df, ("last_swing_low", "last_swing_low_idx", "swing_low_age")):
        level = _safe_float(df["last_swing_low"].iat[bar_idx])
        if math.isfinite(level):
            origin = _safe_int(df["last_swing_low_idx"].iat[bar_idx])
            age = _safe_int(df["swing_low_age"].iat[bar_idx], default=0)
            confirm_idx = bar_idx - max(0, age)
            out.append(
                _Source(
                    family="swing_low",
                    side=-1,
                    level=level,
                    zone_low=level,
                    zone_high=level,
                    is_zone=False,
                    strength=LIQ_FAMILY_DEFAULT_STRENGTH["swing_low"],
                    state=LIQ_STATE_ACTIVE,
                    origin_idx=origin,
                    active_start_idx=confirm_idx,
                    age_bars=max(0, age),
                    touch_count=0,
                )
            )
    return out


def _extract_equal_hl_sources(
    df: pd.DataFrame,
    *,
    bar_idx: int,
) -> list[_Source]:
    out: list[_Source] = []
    # Equal high — sits above price (buy-side liquidity).
    if _has_columns(df, ("eqh_active", "eqh_active_level")):
        active = _safe_int(df["eqh_active"].iat[bar_idx], default=0)
        level = _safe_float(df["eqh_active_level"].iat[bar_idx])
        if active and math.isfinite(level):
            zone_low = (
                _safe_float(df["eqh_active_low"].iat[bar_idx])
                if "eqh_active_low" in df.columns
                else level
            )
            zone_high = (
                _safe_float(df["eqh_active_high"].iat[bar_idx])
                if "eqh_active_high" in df.columns
                else level
            )
            if not math.isfinite(zone_low):
                zone_low = level
            if not math.isfinite(zone_high):
                zone_high = level
            strength = (
                _safe_float(df["eqh_active_score"].iat[bar_idx])
                if "eqh_active_score" in df.columns
                else LIQ_FAMILY_DEFAULT_STRENGTH["equal_high"]
            )
            if not math.isfinite(strength):
                strength = LIQ_FAMILY_DEFAULT_STRENGTH["equal_high"]
            origin = (
                _safe_int(df["eqh_active_id"].iat[bar_idx])
                if "eqh_active_id" in df.columns
                else -1
            )
            since = (
                _safe_int(df["eqh_active_since_idx"].iat[bar_idx])
                if "eqh_active_since_idx" in df.columns
                else -1
            )
            age = (
                _safe_int(df["eqh_active_age"].iat[bar_idx], default=0)
                if "eqh_active_age" in df.columns
                else 0
            )
            touches = (
                _safe_int(df["eqh_active_touch_count"].iat[bar_idx], default=0)
                if "eqh_active_touch_count" in df.columns
                else 0
            )
            out.append(
                _Source(
                    family="equal_high",
                    side=+1,
                    level=level,
                    zone_low=zone_low,
                    zone_high=zone_high,
                    is_zone=zone_high > zone_low,
                    strength=float(np.clip(strength, 0.0, 1.0)),
                    state=LIQ_STATE_ACTIVE,
                    origin_idx=origin if origin > 0 else -1,
                    active_start_idx=since if since >= 0 else bar_idx - max(0, age),
                    age_bars=max(0, age),
                    touch_count=max(0, touches),
                )
            )
    # Equal low — sits below price (sell-side liquidity).
    if _has_columns(df, ("eql_active", "eql_active_level")):
        active = _safe_int(df["eql_active"].iat[bar_idx], default=0)
        level = _safe_float(df["eql_active_level"].iat[bar_idx])
        if active and math.isfinite(level):
            zone_low = (
                _safe_float(df["eql_active_low"].iat[bar_idx])
                if "eql_active_low" in df.columns
                else level
            )
            zone_high = (
                _safe_float(df["eql_active_high"].iat[bar_idx])
                if "eql_active_high" in df.columns
                else level
            )
            if not math.isfinite(zone_low):
                zone_low = level
            if not math.isfinite(zone_high):
                zone_high = level
            strength = (
                _safe_float(df["eql_active_score"].iat[bar_idx])
                if "eql_active_score" in df.columns
                else LIQ_FAMILY_DEFAULT_STRENGTH["equal_low"]
            )
            if not math.isfinite(strength):
                strength = LIQ_FAMILY_DEFAULT_STRENGTH["equal_low"]
            origin = (
                _safe_int(df["eql_active_id"].iat[bar_idx])
                if "eql_active_id" in df.columns
                else -1
            )
            since = (
                _safe_int(df["eql_active_since_idx"].iat[bar_idx])
                if "eql_active_since_idx" in df.columns
                else -1
            )
            age = (
                _safe_int(df["eql_active_age"].iat[bar_idx], default=0)
                if "eql_active_age" in df.columns
                else 0
            )
            touches = (
                _safe_int(df["eql_active_touch_count"].iat[bar_idx], default=0)
                if "eql_active_touch_count" in df.columns
                else 0
            )
            out.append(
                _Source(
                    family="equal_low",
                    side=-1,
                    level=level,
                    zone_low=zone_low,
                    zone_high=zone_high,
                    is_zone=zone_high > zone_low,
                    strength=float(np.clip(strength, 0.0, 1.0)),
                    state=LIQ_STATE_ACTIVE,
                    origin_idx=origin if origin > 0 else -1,
                    active_start_idx=since if since >= 0 else bar_idx - max(0, age),
                    age_bars=max(0, age),
                    touch_count=max(0, touches),
                )
            )
    return out


def _extract_sr_sources(
    df: pd.DataFrame,
    *,
    bar_idx: int,
) -> list[_Source]:
    """Pull S/R ladder zones from the projected per-bar columns."""

    out: list[_Source] = []
    # Resistance ladder — above price.
    for slot in (1, 2, 3):
        prefix = f"sr_resistance_l{slot}"
        if not _has_columns(df, (f"{prefix}_mid", f"{prefix}_low", f"{prefix}_high")):
            continue
        level = _safe_float(df[f"{prefix}_mid"].iat[bar_idx])
        if not math.isfinite(level):
            continue
        zone_low = _safe_float(df[f"{prefix}_low"].iat[bar_idx])
        zone_high = _safe_float(df[f"{prefix}_high"].iat[bar_idx])
        if not math.isfinite(zone_low):
            zone_low = level
        if not math.isfinite(zone_high):
            zone_high = level
        strength = (
            _safe_float(df[f"{prefix}_score"].iat[bar_idx])
            if f"{prefix}_score" in df.columns
            else LIQ_FAMILY_DEFAULT_STRENGTH["resistance"]
        )
        if not math.isfinite(strength):
            strength = LIQ_FAMILY_DEFAULT_STRENGTH["resistance"]
        age = (
            _safe_int(df[f"{prefix}_age_bars"].iat[bar_idx], default=0)
            if f"{prefix}_age_bars" in df.columns
            else 0
        )
        zone_id = (
            _safe_int(df[f"{prefix}_id"].iat[bar_idx])
            if f"{prefix}_id" in df.columns
            else -1
        )
        out.append(
            _Source(
                family="resistance",
                side=+1,
                level=level,
                zone_low=zone_low,
                zone_high=zone_high,
                is_zone=zone_high > zone_low,
                strength=float(np.clip(strength, 0.0, 1.0)),
                state=LIQ_STATE_ACTIVE,
                origin_idx=zone_id if zone_id > 0 else -1,
                active_start_idx=bar_idx - max(0, age),
                age_bars=max(0, age),
                touch_count=0,
            )
        )
    # Support ladder — below price.
    for slot in (1, 2, 3):
        prefix = f"sr_support_l{slot}"
        if not _has_columns(df, (f"{prefix}_mid", f"{prefix}_low", f"{prefix}_high")):
            continue
        level = _safe_float(df[f"{prefix}_mid"].iat[bar_idx])
        if not math.isfinite(level):
            continue
        zone_low = _safe_float(df[f"{prefix}_low"].iat[bar_idx])
        zone_high = _safe_float(df[f"{prefix}_high"].iat[bar_idx])
        if not math.isfinite(zone_low):
            zone_low = level
        if not math.isfinite(zone_high):
            zone_high = level
        strength = (
            _safe_float(df[f"{prefix}_score"].iat[bar_idx])
            if f"{prefix}_score" in df.columns
            else LIQ_FAMILY_DEFAULT_STRENGTH["support"]
        )
        if not math.isfinite(strength):
            strength = LIQ_FAMILY_DEFAULT_STRENGTH["support"]
        age = (
            _safe_int(df[f"{prefix}_age_bars"].iat[bar_idx], default=0)
            if f"{prefix}_age_bars" in df.columns
            else 0
        )
        zone_id = (
            _safe_int(df[f"{prefix}_id"].iat[bar_idx])
            if f"{prefix}_id" in df.columns
            else -1
        )
        out.append(
            _Source(
                family="support",
                side=-1,
                level=level,
                zone_low=zone_low,
                zone_high=zone_high,
                is_zone=zone_high > zone_low,
                strength=float(np.clip(strength, 0.0, 1.0)),
                state=LIQ_STATE_ACTIVE,
                origin_idx=zone_id if zone_id > 0 else -1,
                active_start_idx=bar_idx - max(0, age),
                age_bars=max(0, age),
                touch_count=0,
            )
        )
    return out


def _value_change_active_start(arr: np.ndarray) -> np.ndarray:
    """For a calendar/session source whose level changes step-wise, compute
    the active-start index per bar — the most recent bar where the value
    transitioned from a different value (or NaN) to the current one.

    The result has the same length as ``arr``. Bars with NaN sources get -1.
    """

    n = arr.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    last_start = -1
    last_val = float("nan")
    for i in range(n):
        v = arr[i]
        if not np.isfinite(v):
            last_start = -1
            last_val = float("nan")
            out[i] = -1
            continue
        if not np.isfinite(last_val) or v != last_val:
            last_start = i
            last_val = v
        out[i] = last_start
    return out


def _extract_calendar_sources(
    df: pd.DataFrame,
    *,
    bar_idx: int,
    cache: dict[str, np.ndarray],
) -> list[_Source]:
    out: list[_Source] = []

    def _push(
        family: str,
        side: int,
        col: str,
    ) -> None:
        if col not in df.columns:
            return
        level = _safe_float(df[col].iat[bar_idx])
        if not math.isfinite(level):
            return
        active_start = int(cache[col][bar_idx])
        if active_start < 0:
            return
        out.append(
            _Source(
                family=family,
                side=side,
                level=level,
                zone_low=level,
                zone_high=level,
                is_zone=False,
                strength=LIQ_FAMILY_DEFAULT_STRENGTH[family],
                state=LIQ_STATE_ACTIVE,
                origin_idx=active_start,
                active_start_idx=active_start,
                age_bars=max(0, bar_idx - active_start),
                touch_count=0,
            )
        )

    # Session highs/lows — Asia, London, NY (these all map to "session_high"
    # / "session_low" family but the level differs per session). We add each
    # as a separate candidate; clustering will collapse near-duplicates.
    for col in ("prev_asia_high", "prev_london_high", "prev_ny_high"):
        _push("session_high", +1, col)
    for col in ("prev_asia_low", "prev_london_low", "prev_ny_low"):
        _push("session_low", -1, col)
    _push("previous_day_high", +1, "prev_day_high")
    _push("previous_day_low", -1, "prev_day_low")
    _push("previous_week_high", +1, "prev_week_high")
    _push("previous_week_low", -1, "prev_week_low")
    return out


def _build_active_start_cache(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Precompute active-start indices for every calendar/session column
    used by :func:`_extract_calendar_sources`. One pass, no per-bar cost.
    """

    cache: dict[str, np.ndarray] = {}
    cols = (
        "prev_asia_high",
        "prev_london_high",
        "prev_ny_high",
        "prev_asia_low",
        "prev_london_low",
        "prev_ny_low",
        "prev_day_high",
        "prev_day_low",
        "prev_week_high",
        "prev_week_low",
    )
    for col in cols:
        if col in df.columns:
            cache[col] = _value_change_active_start(
                pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            )
    return cache


# ---------------------------------------------------------------------------
# Per-bar clustering / dedup / crowding
# ---------------------------------------------------------------------------


def _zones_overlap(
    a_low: float,
    a_high: float,
    b_low: float,
    b_high: float,
) -> bool:
    return not (a_high < b_low or b_high < a_low)


def _cluster_sources(
    sources: Sequence[_Source],
    *,
    atr: float,
    min_tick: float,
) -> list[_Cluster]:
    """Group sources of the same side into clusters using a price-tolerance
    rule. For point sources the tolerance is ``max(0.10*ATR, 5*min_tick)``;
    for zone sources we additionally merge if zones overlap.
    """

    if not sources:
        return []

    if not math.isfinite(atr) or atr <= 0:
        atr = 0.0
    tolerance = max(LIQ_DEDUP_ATR_FRAC * atr, LIQ_MIN_TICK_MULT * min_tick)

    # Split by side
    by_side: dict[int, list[_Source]] = {+1: [], -1: []}
    for s in sources:
        if s.family in LIQ_DEPRECATED_FAMILIES:
            raise ValueError(
                f"Deprecated source family {s.family!r} reached the unified "
                "framework; production v1 excludes range_boundary, FVG, OB."
            )
        by_side.setdefault(s.side, []).append(s)

    clusters: list[_Cluster] = []
    for side, side_sources in by_side.items():
        if not side_sources:
            continue
        side_sources = sorted(side_sources, key=lambda s: s.level)
        used = [False] * len(side_sources)
        for i, src in enumerate(side_sources):
            if used[i]:
                continue
            members = [src]
            used[i] = True
            for j in range(i + 1, len(side_sources)):
                if used[j]:
                    continue
                cand = side_sources[j]
                close_levels = abs(cand.level - src.level) <= tolerance
                zone_overlap = (src.is_zone or cand.is_zone) and _zones_overlap(
                    src.zone_low, src.zone_high, cand.zone_low, cand.zone_high
                )
                if close_levels or zone_overlap:
                    members.append(cand)
                    used[j] = True
            cluster = _build_cluster_from_members(members, side=side, atr=atr)
            clusters.append(cluster)
    return clusters


def _build_cluster_from_members(
    members: Sequence[_Source],
    *,
    side: int,
    atr: float,
) -> _Cluster:
    primary = min(
        members,
        key=lambda s: (
            LIQ_FAMILY_PRECEDENCE[s.family],
            -s.strength,
            s.age_bars,
        ),
    )
    zone_lows = [m.zone_low for m in members if math.isfinite(m.zone_low)]
    zone_highs = [m.zone_high for m in members if math.isfinite(m.zone_high)]
    if zone_lows and zone_highs:
        zone_low = float(min(zone_lows))
        zone_high = float(max(zone_highs))
    else:
        zone_low = primary.level
        zone_high = primary.level
    width_abs = max(0.0, zone_high - zone_low)
    width_atr = (width_abs / atr) if math.isfinite(atr) and atr > 0 else float("nan")
    cluster_strength = float(
        np.clip(
            max((m.strength for m in members), default=primary.strength)
            + 0.05 * (len(members) - 1),
            0.0,
            1.0,
        )
    )
    origin_idx = min((m.origin_idx for m in members if m.origin_idx >= 0), default=-1)
    active_start_idx = min(
        (m.active_start_idx for m in members if m.active_start_idx >= 0),
        default=-1,
    )
    touch_count = max((m.touch_count for m in members), default=0)
    families = tuple(sorted({m.family for m in members}))
    is_zone = any(m.is_zone for m in members)
    age_bars = max((m.age_bars for m in members), default=0)
    return _Cluster(
        side=side,
        primary_family=primary.family,
        members=list(members),
        level=primary.level if not is_zone else (zone_low + zone_high) / 2.0,
        zone_low=zone_low,
        zone_high=zone_high,
        is_zone=is_zone,
        width_abs=width_abs,
        width_atr=width_atr,
        strength=cluster_strength,
        state=LIQ_STATE_ACTIVE,
        origin_idx=origin_idx,
        active_start_idx=active_start_idx,
        age_bars=age_bars,
        touch_count=touch_count,
        attribution_families=families,
    )


def _dominance_filter(clusters: Sequence[_Cluster]) -> tuple[list[_Cluster], int]:
    """Drop weaker duplicates: same side, within tolerance, lower strength,
    same/lower precedence, no unique family contribution.

    Returns (kept, dropped_count).
    """

    if len(clusters) < 2:
        return list(clusters), 0
    kept: list[_Cluster] = []
    dropped = 0
    # Sort strongest first so we keep the dominant one.
    ranked = sorted(
        clusters,
        key=lambda c: (
            LIQ_FAMILY_PRECEDENCE[c.primary_family],
            -c.strength,
        ),
    )
    for cand in ranked:
        is_dominated = False
        for keep in kept:
            if keep.side != cand.side:
                continue
            same_area = (
                _zones_overlap(
                    keep.zone_low, keep.zone_high, cand.zone_low, cand.zone_high
                )
                or abs(keep.level - cand.level) < 1e-9
            )
            if not same_area:
                continue
            cand_unique = set(cand.attribution_families) - set(
                keep.attribution_families
            )
            if not cand_unique and cand.strength <= keep.strength + 1e-9:
                is_dominated = True
                break
        if is_dominated:
            dropped += 1
        else:
            kept.append(cand)
    return kept, dropped


def _apply_crowding(
    clusters: Sequence[_Cluster],
    *,
    close: float,
) -> tuple[list[_Cluster], list[_Cluster], int]:
    """Sort clusters by (side, distance to close) and keep the top
    LIQ_LADDER_DEPTH per side, capped globally at LIQ_GLOBAL_CROWDING_CAP.

    Returns (above_kept, below_kept, dropped_by_crowding_count).
    """

    above = sorted(
        (c for c in clusters if c.side == +1),
        key=lambda c: abs(c.level - close),
    )
    below = sorted(
        (c for c in clusters if c.side == -1),
        key=lambda c: abs(c.level - close),
    )
    above_kept = above[:LIQ_LADDER_DEPTH]
    below_kept = below[:LIQ_LADDER_DEPTH]
    # Global cap: if combined > LIQ_GLOBAL_CROWDING_CAP, trim by interleaving.
    combined = above_kept + below_kept
    if len(combined) > LIQ_GLOBAL_CROWDING_CAP:
        above_share = LIQ_GLOBAL_CROWDING_CAP // 2
        below_share = LIQ_GLOBAL_CROWDING_CAP - above_share
        above_kept = above_kept[:above_share]
        below_kept = below_kept[:below_share]
    dropped = (len(above) - len(above_kept)) + (len(below) - len(below_kept))
    return above_kept, below_kept, dropped


# ---------------------------------------------------------------------------
# Output projection
# ---------------------------------------------------------------------------


def _bar_idx_to_cluster_id(bar_idx: int, side: int, slot: int) -> int:
    """Compose a stable per-(bar, side, slot) cluster id. Sweeps refers to
    this id when attributing a same-bar multi-source breach.
    """

    return int(bar_idx) * 1000 + (1 if side == +1 else 2) * 100 + int(slot)


def _project_cluster_to_row(
    out: dict[str, np.ndarray],
    *,
    bar_idx: int,
    cluster: _Cluster,
    side_label: str,
    rank: int,
    close: float,
    atr: float,
) -> None:
    prefix = f"liq_{side_label}_l{rank}"
    out[f"{prefix}_cluster_id"][bar_idx] = float(
        _bar_idx_to_cluster_id(bar_idx, cluster.side, rank)
    )
    out[f"{prefix}_level"][bar_idx] = float(cluster.level)
    out[f"{prefix}_zone_low"][bar_idx] = float(cluster.zone_low)
    out[f"{prefix}_zone_high"][bar_idx] = float(cluster.zone_high)
    out[f"{prefix}_is_zone"][bar_idx] = float(1.0 if cluster.is_zone else 0.0)
    out[f"{prefix}_width_abs"][bar_idx] = float(cluster.width_abs)
    out[f"{prefix}_width_atr"][bar_idx] = float(cluster.width_atr)
    out[f"{prefix}_strength"][bar_idx] = float(cluster.strength)
    out[f"{prefix}_state"][bar_idx] = float(cluster.state)
    out[f"{prefix}_age_bars"][bar_idx] = float(cluster.age_bars)
    # freshness: simple decay, bounded; sweeps quality re-uses this.
    freshness = 1.0 / (1.0 + max(0, cluster.age_bars) / 50.0)
    out[f"{prefix}_freshness"][bar_idx] = float(freshness)
    out[f"{prefix}_touch_count"][bar_idx] = float(cluster.touch_count)
    if math.isfinite(atr) and atr > 0 and math.isfinite(close):
        signed = (cluster.level - close) / atr
    else:
        signed = float("nan")
    out[f"{prefix}_signed_dist_atr"][bar_idx] = float(signed)
    out[f"{prefix}_member_count"][bar_idx] = float(len(cluster.members))
    out[f"{prefix}_origin_idx"][bar_idx] = float(cluster.origin_idx)
    out[f"{prefix}_active_start_idx"][bar_idx] = float(cluster.active_start_idx)
    # Primary family + attribution stored as object-array slots after the loop.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _required_atr_array(df: pd.DataFrame) -> np.ndarray:
    if "atr_14" in df.columns:
        return pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
    if "atr" in df.columns:
        return pd.to_numeric(df["atr"], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def add_unified_liquidity_sources(
    df: pd.DataFrame,
    *,
    scan_timeframe: str,
    instrument: str | None = None,
) -> pd.DataFrame:
    """Project the per-bar unified liquidity ladder onto ``df``.

    Reads the upstream live-safe columns produced by swings, equal_hl,
    sr_levels, session, prev_day_hl, prev_week_hl. Emits the dense ladder
    columns enumerated by :data:`UNIFIED_SOURCE_COLUMNS`.

    Parameters
    ----------
    df
        The pipeline frame after every source-producing stage. Required:
        ``high``, ``low``, ``close``, ``atr_14`` (or ``atr``) and at minimum
        the swings live-safe columns. Other family columns are optional —
        when missing, that family simply contributes zero candidates.
    scan_timeframe
        The timeframe the sweeps are being scanned on (e.g., ``"H4"``). Used
        to stamp every emitted source row and validated by the MTF policy
        guard.
    instrument
        Optional instrument label. Used to look up the per-instrument min-tick
        for the dedup-tolerance floor. Falls back to a 0.0001 default.
    """

    if df is None:
        raise ValueError("add_unified_liquidity_sources: df must not be None")
    if len(df) == 0:
        return df.copy()

    scan_timeframe = assert_known_timeframe(scan_timeframe)
    min_tick = _resolve_min_tick(instrument)

    out = df.copy()
    n = len(out)

    # ── Initialize output arrays ────────────────────────────────────────
    # Family columns are object-dtype strings; everything else is float NaN.
    family_columns = {
        f"liq_{side}_l{rank}_{field}"
        for side in _LADDER_SIDES
        for rank in range(1, LIQ_LADDER_DEPTH + 1)
        for field in ("primary_family", "attribution_families")
    }
    numeric_cols = [
        c
        for c in UNIFIED_SOURCE_COLUMNS
        if c not in family_columns
        and c not in ("liq_source_timeframe", "liq_mtf_policy")
    ]
    arrays: dict[str, np.ndarray] = {
        col: np.full(n, np.nan, dtype=float) for col in numeric_cols
    }
    # Object slots for primary_family + attribution_families per ladder slot
    family_slot_arrays: dict[str, np.ndarray] = {}
    for side in _LADDER_SIDES:
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            family_slot_arrays[f"liq_{side}_l{rank}_primary_family"] = np.full(
                n, "", dtype=object
            )
            family_slot_arrays[f"liq_{side}_l{rank}_attribution_families"] = np.full(
                n, "", dtype=object
            )

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    open_ = (
        pd.to_numeric(out["open"], errors="coerce").to_numpy(dtype=float)
        if "open" in out.columns
        else close.copy()
    )
    atr_arr = _required_atr_array(out)

    cal_cache = _build_active_start_cache(out)

    # Audit accumulator for the sidecar table.
    audit_rows: list[dict[str, object]] = []

    for i in range(n):
        bar_close = close[i]
        bar_atr = atr_arr[i]

        sources = (
            _extract_swing_sources(out, bar_idx=i)
            + _extract_equal_hl_sources(out, bar_idx=i)
            + _extract_sr_sources(out, bar_idx=i)
            + _extract_calendar_sources(out, bar_idx=i, cache=cal_cache)
        )
        # Filter sources whose level was on the wrong side of price *at the
        # start of the bar*. Using ``open_i`` (or the prior bar's close as a
        # fallback) — instead of the bar's own close — keeps levels in the
        # ladder when intra-bar action closes through them. That preserves
        # the "above price" semantic on entry while letting the sweeps
        # detector observe a close-acceptance breakout at the breach bar.
        ref_price = (
            open_[i]
            if math.isfinite(open_[i])
            else (close[i - 1] if i > 0 else bar_close)
        )
        eligible: list[_Source] = []
        for s in sources:
            if not math.isfinite(s.level) or not math.isfinite(ref_price):
                continue
            if s.side == +1 and s.level < ref_price - 1e-9:
                continue
            if s.side == -1 and s.level > ref_price + 1e-9:
                continue
            eligible.append(s)

        clusters = _cluster_sources(eligible, atr=bar_atr, min_tick=min_tick)
        clusters, dom_dropped = _dominance_filter(clusters)
        above_kept, below_kept, crowd_dropped = _apply_crowding(
            clusters, close=bar_close
        )

        arrays["liq_active_total_count"][i] = float(len(above_kept) + len(below_kept))
        arrays["liq_active_above_count"][i] = float(len(above_kept))
        arrays["liq_active_below_count"][i] = float(len(below_kept))
        arrays["liq_dropped_by_crowding_count"][i] = float(crowd_dropped)
        arrays["liq_dropped_by_dominance_count"][i] = float(dom_dropped)
        if above_kept:
            arrays["liq_nearest_above_dist_atr"][i] = (
                (above_kept[0].level - bar_close) / bar_atr
                if math.isfinite(bar_atr) and bar_atr > 0
                else float("nan")
            )
            arrays["liq_top_above_strength"][i] = max(c.strength for c in above_kept)
        if below_kept:
            arrays["liq_nearest_below_dist_atr"][i] = (
                (bar_close - below_kept[0].level) / bar_atr
                if math.isfinite(bar_atr) and bar_atr > 0
                else float("nan")
            )
            arrays["liq_top_below_strength"][i] = max(c.strength for c in below_kept)

        for rank, cluster in enumerate(above_kept, start=1):
            _project_cluster_to_row(
                arrays,
                bar_idx=i,
                cluster=cluster,
                side_label="above",
                rank=rank,
                close=bar_close,
                atr=bar_atr,
            )
            family_slot_arrays[f"liq_above_l{rank}_primary_family"][
                i
            ] = cluster.primary_family
            family_slot_arrays[f"liq_above_l{rank}_attribution_families"][i] = "|".join(
                cluster.attribution_families
            )
            audit_rows.append(
                _audit_row(
                    i, rank, "above", cluster, bar_close, bar_atr, scan_timeframe
                )
            )
        for rank, cluster in enumerate(below_kept, start=1):
            _project_cluster_to_row(
                arrays,
                bar_idx=i,
                cluster=cluster,
                side_label="below",
                rank=rank,
                close=bar_close,
                atr=bar_atr,
            )
            family_slot_arrays[f"liq_below_l{rank}_primary_family"][
                i
            ] = cluster.primary_family
            family_slot_arrays[f"liq_below_l{rank}_attribution_families"][i] = "|".join(
                cluster.attribution_families
            )
            audit_rows.append(
                _audit_row(
                    i, rank, "below", cluster, bar_close, bar_atr, scan_timeframe
                )
            )

    # Combine all new columns into a single frame and concat once to avoid
    # the per-column insert fragmentation cost that pandas warns about when
    # adding 190+ columns sequentially.
    new_cols: dict[str, np.ndarray | str] = dict(arrays)
    new_cols.update(family_slot_arrays)
    new_cols["liq_source_timeframe"] = np.full(n, scan_timeframe, dtype=object)
    new_cols["liq_mtf_policy"] = np.full(n, SWEEP_MTF_POLICY, dtype=object)
    addition = pd.DataFrame(new_cols, index=out.index)
    # Drop any pre-existing duplicates that an older pipeline run may have
    # left in the frame so the concat does not create ambiguous columns.
    duplicates = [c for c in addition.columns if c in out.columns]
    if duplicates:
        out = out.drop(columns=duplicates)
    out = pd.concat([out, addition], axis=1)

    # Causality + MTF policy assertions on the audit table.
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        bad = audit_df[
            (audit_df["source_active_start_idx"] >= 0)
            & (audit_df["source_active_start_idx"] > audit_df["bar_idx"])
        ]
        if len(bad) > 0:
            raise ValueError(
                "Causality violation in unified sources: "
                f"{len(bad)} rows have active_start_idx > bar_idx"
            )
        # Stamp source_timeframe column for the MTF guard.
        audit_df["source_timeframe"] = scan_timeframe
        assert_same_timeframe_sources(audit_df, scan_timeframe=scan_timeframe)

    # The audit table is rebuildable from the dense ladder columns by
    # :func:`build_unified_liquidity_clusters_audit` — we deliberately do not
    # stash it in ``df.attrs`` because that breaks parquet serialization.
    return out


def _audit_row(
    bar_idx: int,
    rank: int,
    side_label: str,
    cluster: _Cluster,
    close: float,
    atr: float,
    scan_timeframe: str,
) -> dict[str, object]:
    if math.isfinite(atr) and atr > 0:
        signed = (
            (cluster.level - close) / atr
            if side_label == "above"
            else (close - cluster.level) / atr
        )
    else:
        signed = float("nan")
    return {
        "bar_idx": int(bar_idx),
        "side": cluster.side,
        "side_label": side_label,
        "rank": int(rank),
        "cluster_id": _bar_idx_to_cluster_id(bar_idx, cluster.side, rank),
        "primary_family": cluster.primary_family,
        "attribution_families": "|".join(cluster.attribution_families),
        "level": float(cluster.level),
        "zone_low": float(cluster.zone_low),
        "zone_high": float(cluster.zone_high),
        "is_zone": bool(cluster.is_zone),
        "width_abs": float(cluster.width_abs),
        "width_atr": float(cluster.width_atr),
        "strength": float(cluster.strength),
        "state": int(cluster.state),
        "age_bars": int(cluster.age_bars),
        "touch_count": int(cluster.touch_count),
        "signed_dist_atr": float(signed),
        "member_count": len(cluster.members),
        "source_origin_idx": int(cluster.origin_idx),
        "source_active_start_idx": int(cluster.active_start_idx),
        "source_timeframe": scan_timeframe,
    }


_AUDIT_COLUMNS: tuple[str, ...] = (
    "bar_idx",
    "side",
    "side_label",
    "rank",
    "cluster_id",
    "primary_family",
    "attribution_families",
    "level",
    "zone_low",
    "zone_high",
    "is_zone",
    "width_abs",
    "width_atr",
    "strength",
    "state",
    "age_bars",
    "touch_count",
    "signed_dist_atr",
    "member_count",
    "source_origin_idx",
    "source_active_start_idx",
    "source_timeframe",
)


def build_unified_liquidity_clusters_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct the per-(bar, cluster) audit table from the dense ladder
    columns produced by :func:`add_unified_liquidity_sources`.

    The audit table is suitable for CSV export, chart overlays, and golden
    tests. If the frame is missing the ladder columns, returns an empty
    DataFrame with the canonical schema.
    """

    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(columns=list(_AUDIT_COLUMNS))

    rows: list[dict[str, object]] = []
    scan_tf = (
        str(df["liq_source_timeframe"].iat[0])
        if "liq_source_timeframe" in df.columns and len(df) > 0
        else ""
    )
    for side_label, side_int in (("above", +1), ("below", -1)):
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            cid_col = f"liq_{side_label}_l{rank}_cluster_id"
            if cid_col not in df.columns:
                continue
            cids = pd.to_numeric(df[cid_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(cids)
            if not mask.any():
                continue
            level = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_level"], errors="coerce"
            ).to_numpy(dtype=float)
            zlo = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_zone_low"], errors="coerce"
            ).to_numpy(dtype=float)
            zhi = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_zone_high"], errors="coerce"
            ).to_numpy(dtype=float)
            is_zone = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_is_zone"], errors="coerce"
            ).to_numpy(dtype=float)
            wabs = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_width_abs"], errors="coerce"
            ).to_numpy(dtype=float)
            watr = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_width_atr"], errors="coerce"
            ).to_numpy(dtype=float)
            strength = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_strength"], errors="coerce"
            ).to_numpy(dtype=float)
            state = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_state"], errors="coerce"
            ).to_numpy(dtype=float)
            age = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_age_bars"], errors="coerce"
            ).to_numpy(dtype=float)
            touches = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_touch_count"], errors="coerce"
            ).to_numpy(dtype=float)
            signed = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_signed_dist_atr"], errors="coerce"
            ).to_numpy(dtype=float)
            member_count = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_member_count"], errors="coerce"
            ).to_numpy(dtype=float)
            origin_idx = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_origin_idx"], errors="coerce"
            ).to_numpy(dtype=float)
            active_start = pd.to_numeric(
                df[f"liq_{side_label}_l{rank}_active_start_idx"], errors="coerce"
            ).to_numpy(dtype=float)
            primary_fam = df[f"liq_{side_label}_l{rank}_primary_family"].astype(object)
            attr_fam = df[f"liq_{side_label}_l{rank}_attribution_families"].astype(
                object
            )
            for i in np.flatnonzero(mask):
                rows.append(
                    {
                        "bar_idx": int(i),
                        "side": side_int,
                        "side_label": side_label,
                        "rank": rank,
                        "cluster_id": float(cids[i]),
                        "primary_family": str(primary_fam.iat[i] or ""),
                        "attribution_families": str(attr_fam.iat[i] or ""),
                        "level": float(level[i]),
                        "zone_low": float(zlo[i]),
                        "zone_high": float(zhi[i]),
                        "is_zone": bool(is_zone[i] > 0),
                        "width_abs": float(wabs[i]),
                        "width_atr": float(watr[i]),
                        "strength": float(strength[i]),
                        "state": int(state[i]) if math.isfinite(state[i]) else 0,
                        "age_bars": int(age[i]) if math.isfinite(age[i]) else 0,
                        "touch_count": (
                            int(touches[i]) if math.isfinite(touches[i]) else 0
                        ),
                        "signed_dist_atr": float(signed[i]),
                        "member_count": (
                            int(member_count[i])
                            if math.isfinite(member_count[i])
                            else 0
                        ),
                        "source_origin_idx": (
                            int(origin_idx[i]) if math.isfinite(origin_idx[i]) else -1
                        ),
                        "source_active_start_idx": (
                            int(active_start[i])
                            if math.isfinite(active_start[i])
                            else -1
                        ),
                        "source_timeframe": scan_tf,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(_AUDIT_COLUMNS))
    return (
        pd.DataFrame(rows, columns=list(_AUDIT_COLUMNS))
        .sort_values(["bar_idx", "side_label", "rank"])
        .reset_index(drop=True)
    )


__all__ = [
    "LIQ_LADDER_DEPTH",
    "LIQ_GLOBAL_CROWDING_CAP",
    "LIQ_DEDUP_ATR_FRAC",
    "LIQ_MIN_TICK",
    "LIQ_MIN_TICK_DEFAULT",
    "LIQ_MIN_TICK_MULT",
    "LIQ_STATE_UNAVAILABLE",
    "LIQ_STATE_BORN",
    "LIQ_STATE_ACTIVE",
    "LIQ_STATE_WEAKENED",
    "LIQ_STATE_CONSUMED_SWEPT",
    "LIQ_STATE_INVALIDATED",
    "LIQ_STATE_RETIRED",
    "LIQ_SIDE_BUY_LIQ_ABOVE",
    "LIQ_SIDE_SELL_LIQ_BELOW",
    "LIQ_SOURCE_FAMILIES",
    "LIQ_FAMILY_SIDE",
    "LIQ_FAMILY_PRECEDENCE",
    "LIQ_FAMILY_DEFAULT_STRENGTH",
    "LIQ_DEPRECATED_FAMILIES",
    "UNIFIED_SOURCE_COLUMNS",
    "add_unified_liquidity_sources",
    "build_unified_liquidity_clusters_audit",
]
