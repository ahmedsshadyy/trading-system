"""Step 11 — Final sweeps detector consuming the unified liquidity ladder.

A sweep is **not** any level breach. It requires four ingredients:

1. an eligible active liquidity source from the unified framework,
2. price penetration beyond the source level/zone,
3. rejection or reclaim evidence within a causal confirmation window,
4. deterministic classification as sweep / breakout / unresolved.

This module consumes only the dense ladder columns produced by
:mod:`src.indicators.sweeps_v2.unified_sources` (``liq_above_l*_*`` and
``liq_below_l*_*``). It does not reach into raw upstream stages — the
unified framework is the single source of truth.

Causality contract
------------------
Sweep signals become available only at ``confirm_idx`` close. A same-bar
sweep (where rejection condition is satisfied at the breach bar's close) is
the only case in which ``confirm_idx == breach_idx``. Live consumers must
gate any tradable use on the ``sweep_is_tradeable_candidate`` flag, which is
``False`` until ``confirm_idx`` close.

Follow-through diagnostics (displacement / BOS / CHoCH within N bars after
confirmation) are flagged as ``sweep_research_followthrough_available`` —
they are causal at the later bar, but NOT known at confirmation time. They
must not enter live features without an explicit research-only label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators.sweeps_v2.unified_sources import (
    LIQ_LADDER_DEPTH,
)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

#: Maximum number of bars after the breach in which a rejection still
#: counts as a sweep (instead of an accepted breakout).
DEFAULT_CONFIRMATION_WINDOW_BARS: int = 3

#: Window for follow-through diagnostics (displacement / BOS / CHoCH).
DEFAULT_FOLLOW_THROUGH_WINDOW_BARS: int = 5

#: Step 11B: a wick-breach that closes back inside is only counted as a
#: same-bar sweep if the wick penetrated the source by at least this much
#: ATR. Tiny noise wicks below this threshold do not register.
DEFAULT_MIN_PENETRATION_ATR_FOR_SAME_BAR: float = 0.15

#: Step 11B: the wick beyond the source must occupy at least this fraction
#: of the bar's full range. Filters out incidental brushes where most of
#: the bar sits well inside the level.
DEFAULT_MIN_WICK_PROMINENCE: float = 0.25

#: Step 11C: bars to suppress a re-fire on the same price cluster after a
#: confirmed sweep or accepted breakout. We validate both 5 and 10 bars, but
#: default to the stricter 10-bar profile for production.
DEFAULT_COOLDOWN_BARS: int = 10

#: Step 11B baseline: per-family minimum source strength.
STEP11B_BASELINE_FAMILY_MIN_STRENGTH: dict[str, float] = {
    "resistance": 0.50,
    "support": 0.50,
    "equal_high": 0.55,
    "equal_low": 0.55,
    "previous_week_high": 0.45,
    "previous_week_low": 0.45,
    "previous_day_high": 0.45,
    "previous_day_low": 0.45,
    "session_high": 0.40,
    "session_low": 0.40,
    "swing_high": 0.30,
    "swing_low": 0.30,
}

#: Step 11C: stricter family-aware minimum strength. S/R is intentionally
#: tighter than calendar/session levels; EQH/EQL remains preserved.
DEFAULT_FAMILY_MIN_STRENGTH: dict[str, float] = {
    "resistance": 0.55,
    "support": 0.55,
    "equal_high": 0.55,
    "equal_low": 0.55,
    "previous_week_high": 0.50,
    "previous_week_low": 0.50,
    "previous_day_high": 0.50,
    "previous_day_low": 0.50,
    "session_high": 0.45,
    "session_low": 0.45,
    "swing_high": 0.35,
    "swing_low": 0.35,
}

#: Step 11B: per-family maximum age (bars). Stale levels lose eligibility
#: even when still tagged as active by the upstream framework. ``None``
#: means "no age cap".
DEFAULT_FAMILY_MAX_AGE_BARS: dict[str, int] = {
    "resistance": 250,
    "support": 250,
    "equal_high": 200,
    "equal_low": 200,
    "previous_week_high": 80,
    "previous_week_low": 80,
    "previous_day_high": 30,
    "previous_day_low": 30,
    "session_high": 24,
    "session_low": 24,
    "swing_high": 200,
    "swing_low": 200,
}

#: Step 11B baseline: no minimum age gate beyond the max-age cap.
STEP11B_BASELINE_FAMILY_MIN_AGE_BARS: dict[str, int] = {
    "resistance": 0,
    "support": 0,
    "equal_high": 0,
    "equal_low": 0,
    "previous_week_high": 0,
    "previous_week_low": 0,
    "previous_day_high": 0,
    "previous_day_low": 0,
    "session_high": 0,
    "session_low": 0,
    "swing_high": 0,
    "swing_low": 0,
}

#: Step 11C: minimum age before a source is sweep-eligible.
DEFAULT_FAMILY_MIN_AGE_BARS: dict[str, int] = {
    "resistance": 3,
    "support": 3,
    "equal_high": 1,
    "equal_low": 1,
    "previous_week_high": 1,
    "previous_week_low": 1,
    "previous_day_high": 1,
    "previous_day_low": 1,
    "session_high": 1,
    "session_low": 1,
    "swing_high": 1,
    "swing_low": 1,
}

#: Step 11B baseline: no family-specific penetration floor before opening a
#: breach tracker.
STEP11B_BASELINE_FAMILY_MIN_PENETRATION_ATR: dict[str, float] = {
    "resistance": 0.0,
    "support": 0.0,
    "equal_high": 0.0,
    "equal_low": 0.0,
    "previous_week_high": 0.0,
    "previous_week_low": 0.0,
    "previous_day_high": 0.0,
    "previous_day_low": 0.0,
    "session_high": 0.0,
    "session_low": 0.0,
    "swing_high": 0.0,
    "swing_low": 0.0,
}

#: Step 11C: minimum penetration required for a breach to become sweep-
#: eligible at all. S/R is stricter than EQH/EQL and prior-period levels.
DEFAULT_FAMILY_MIN_PENETRATION_ATR: dict[str, float] = {
    "resistance": 0.20,
    "support": 0.20,
    "equal_high": 0.10,
    "equal_low": 0.10,
    "previous_week_high": 0.15,
    "previous_week_low": 0.15,
    "previous_day_high": 0.15,
    "previous_day_low": 0.15,
    "session_high": 0.18,
    "session_low": 0.18,
    "swing_high": 0.12,
    "swing_low": 0.12,
}

#: Step 11C baseline: no historical-separation requirement.
STEP11C_BASELINE_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR: dict[str, float] = {
    "resistance": 0.0,
    "support": 0.0,
    "equal_high": 0.0,
    "equal_low": 0.0,
    "previous_week_high": 0.0,
    "previous_week_low": 0.0,
    "previous_day_high": 0.0,
    "previous_day_low": 0.0,
    "session_high": 0.0,
    "session_low": 0.0,
    "swing_high": 0.0,
    "swing_low": 0.0,
}

#: Step 11D: session sources must have stood materially away from price at
#: some point before the breach; otherwise they are "born near price" noise.
DEFAULT_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR: dict[str, float] = {
    **STEP11C_BASELINE_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR,
    "session_high": 0.35,
    "session_low": 0.35,
}

# Sweep classification (frozen integer enum).
SWEEP_CLASS_NO_INTERACTION: int = 0
SWEEP_CLASS_PROBED: int = 1
SWEEP_CLASS_UNRESOLVED: int = 2
SWEEP_CLASS_SAME_BAR: int = 3
SWEEP_CLASS_DELAYED_REJECTION: int = 4
SWEEP_CLASS_ACCEPTED_BREAKOUT: int = 5
SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM: int = 6
SWEEP_CLASS_SWEEP_THEN_BREAK: int = 7

# Interaction phase (frozen integer enum).
SWEEP_INTERACTION_UNTOUCHED: int = 0
SWEEP_INTERACTION_PROBED: int = 1
SWEEP_INTERACTION_PARTIALLY_SWEPT: int = 2
SWEEP_INTERACTION_FULLY_SWEPT: int = 3
SWEEP_INTERACTION_ACCEPTED_BEYOND: int = 4
SWEEP_INTERACTION_RECLAIMED: int = 5

#: Quality weights — sum need not be 1; final score is renormalised.
SWEEP_QUALITY_WEIGHTS: dict[str, float] = {
    "source_strength_component": 0.25,
    "penetration_component": 0.15,
    "rejection_component": 0.20,
    "displacement_followthrough_component": 0.15,
    "regime_context_component": 0.10,
    "volume_confirmation_component": 0.10,
    "crowding_component": 0.05,
}

#: Step 11B baseline tradeable rules.
STEP11B_BASELINE_TRADEABLE_MIN_QUALITY_BY_CLASS: dict[int, float] = {
    SWEEP_CLASS_SAME_BAR: 0.50,
    SWEEP_CLASS_DELAYED_REJECTION: 0.45,
}
STEP11B_BASELINE_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS: dict[int, float] = {
    SWEEP_CLASS_SAME_BAR: 0.30,
    SWEEP_CLASS_DELAYED_REJECTION: 0.20,
}
STEP11B_BASELINE_TRADEABLE_FAMILY_MIN_STRENGTH: dict[str, float] = (
    STEP11B_BASELINE_FAMILY_MIN_STRENGTH.copy()
)
STEP11B_BASELINE_TRADEABLE_MAX_ACTIVE_SOURCES: int = 999

#: Step 11C tradeable hardening: keep the event, tighten the candidate flag.
DEFAULT_TRADEABLE_MIN_QUALITY_BY_CLASS: dict[int, float] = {
    SWEEP_CLASS_SAME_BAR: 0.55,
    SWEEP_CLASS_DELAYED_REJECTION: 0.50,
}
DEFAULT_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS: dict[int, float] = {
    SWEEP_CLASS_SAME_BAR: 0.35,
    SWEEP_CLASS_DELAYED_REJECTION: 0.25,
}
DEFAULT_TRADEABLE_FAMILY_MIN_STRENGTH: dict[str, float] = {
    "resistance": 0.58,
    "support": 0.58,
    "equal_high": 0.58,
    "equal_low": 0.58,
    "previous_week_high": 0.50,
    "previous_week_low": 0.50,
    "previous_day_high": 0.50,
    "previous_day_low": 0.50,
    "session_high": 0.45,
    "session_low": 0.45,
    "swing_high": 0.40,
    "swing_low": 0.40,
}
DEFAULT_TRADEABLE_MAX_ACTIVE_SOURCES: int = 8
DEFAULT_SESSION_TRADEABLE_MIN_REJECTION_COMPONENT: float = 0.60
DEFAULT_SESSION_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT: bool = True
DEFAULT_STANDARD_MIN_PRE_BREACH_DISTANCE_ATR: float = 0.25
DEFAULT_STANDARD_EXCEPTIONAL_PENETRATION_ATR: float = 0.50
DEFAULT_STANDARD_EXCEPTIONAL_REJECTION_COMPONENT: float = 0.65
DEFAULT_TRADEABLE_FAMILY_MIN_PENETRATION_ATR: dict[str, float] = {
    "resistance": 0.30,
    "support": 0.30,
    "equal_high": 0.20,
    "equal_low": 0.20,
    "previous_week_high": 0.25,
    "previous_week_low": 0.25,
    "previous_day_high": 0.25,
    "previous_day_low": 0.25,
    "session_high": 0.30,
    "session_low": 0.30,
    "swing_high": 0.20,
    "swing_low": 0.20,
}
DEFAULT_TRADEABLE_MIN_REJECTION_COMPONENT: float = 0.60
DEFAULT_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT: bool = True

SWEEP_SELECTIVITY_MICRO_INTERACTION: str = "micro_interaction_sweep"
SWEEP_SELECTIVITY_STANDARD_LIQUIDITY: str = "standard_liquidity_sweep"
SWEEP_SELECTIVITY_DISPLACEMENT_CONFIRMED: str = "displacement_confirmed_sweep"
SWEEP_SELECTIVITY_TRADEABLE_CANDIDATE: str = "tradeable_sweep_candidate"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

#: Per-bar live-safe sweep signal columns.
FINAL_SWEEPS_LIVE_COLUMNS: tuple[str, ...] = (
    "sweep_flag",
    "sweep_side",
    "sweep_class",
    "sweep_breach_idx",
    "sweep_confirm_idx",
    "sweep_latency_bars",
    "sweep_event_id",
    "sweep_source_id",
    "sweep_source_cluster_id",
    "sweep_primary_family",
    "sweep_attribution_families",
    "sweep_source_side",
    "sweep_source_level",
    "sweep_source_strength",
    "sweep_quality_score",
    "sweep_is_tradeable_candidate",
    "sweep_invalidated_flag",
    "sweep_research_followthrough_available",
)

#: Breach mechanics columns — emitted at the breach bar.
FINAL_SWEEPS_BREACH_COLUMNS: tuple[str, ...] = (
    "sweep_breach_flag",
    "sweep_breach_side",
    "sweep_breach_event_id",
    "sweep_breach_source_id",
    "sweep_breach_source_cluster_id",
    "sweep_breach_source_family",
    "sweep_breach_source_level",
    "sweep_breach_source_strength",
    "sweep_breach_timestamp",
    "penetration_abs",
    "penetration_atr",
    "penetration_source_width_frac",
    "breach_by_wick",
    "breach_by_close",
    "breach_gap_flag",
)

#: Quality components — emitted at the confirm bar so the score can be audited.
FINAL_SWEEPS_QUALITY_COLUMNS: tuple[str, ...] = (
    "sweep_q_source_strength",
    "sweep_q_penetration",
    "sweep_q_rejection",
    "sweep_q_displacement_followthrough",
    "sweep_q_regime_context",
    "sweep_q_volume_confirmation",
    "sweep_q_crowding",
)

FINAL_SWEEPS_SELECTIVITY_COLUMNS: tuple[str, ...] = (
    "sweep_pre_breach_distance_atr",
    "sweep_history_max_distance_atr",
    "sweep_is_micro_interaction",
    "sweep_is_standard_liquidity",
    "sweep_is_displacement_confirmed",
    "sweep_selectivity_class",
)

#: Follow-through diagnostics — known only N bars after confirm. Causal at
#: the later bar; NOT known at confirm time. Marked as research-only.
FINAL_SWEEPS_FOLLOWTHROUGH_COLUMNS: tuple[str, ...] = (
    "sweep_followed_by_displacement",
    "sweep_displacement_within_bars",
    "sweep_displacement_strength",
    "sweep_followed_by_bos",
    "sweep_bos_within_bars",
    "sweep_bos_strength",
    "sweep_followed_by_choch",
    "sweep_choch_within_bars",
    "sweep_choch_strength",
)

#: Interaction phase column — one per source ladder slot per side.
FINAL_SWEEPS_INTERACTION_COLUMNS: tuple[str, ...] = tuple(
    f"sweep_interaction_phase_{side}_l{rank}"
    for side in ("above", "below")
    for rank in range(1, LIQ_LADDER_DEPTH + 1)
)

#: All columns this stage emits (canonical, ordered).
FINAL_SWEEPS_COLUMNS: tuple[str, ...] = (
    FINAL_SWEEPS_LIVE_COLUMNS
    + FINAL_SWEEPS_BREACH_COLUMNS
    + FINAL_SWEEPS_QUALITY_COLUMNS
    + FINAL_SWEEPS_SELECTIVITY_COLUMNS
    + FINAL_SWEEPS_FOLLOWTHROUGH_COLUMNS
    + FINAL_SWEEPS_INTERACTION_COLUMNS
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _BreachState:
    """Per-(side, slot) tracker for breaches awaiting confirmation."""

    breach_idx: int
    breach_close: float
    source_level: float
    source_zone_low: float
    source_zone_high: float
    source_strength: float
    source_age_bars: int
    source_family: str
    source_attribution: str
    source_cluster_id: float
    source_origin_idx: int
    source_active_start_idx: int
    source_history_max_distance_atr: float
    source_pre_breach_distance_atr: float
    side: int
    breach_by_wick: bool
    breach_by_close: bool
    penetration_abs: float
    penetration_atr: float
    penetration_source_width_frac: float
    confirmed: bool = False
    invalidated: bool = False
    confirm_idx: int = -1
    quality_components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — column reads
# ---------------------------------------------------------------------------


def _ladder_array(
    df: pd.DataFrame, side: str, rank: int, field_name: str
) -> np.ndarray | None:
    col = f"liq_{side}_l{rank}_{field_name}"
    if col not in df.columns:
        return None
    series = df[col]
    # Treat any non-numeric dtype (object, string-extension, "str") as raw
    # values so family names round-trip as strings rather than being coerced
    # to NaN by ``pd.to_numeric``.
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.to_numpy()
    return pd.to_numeric(series, errors="coerce").to_numpy()


def _safe_atr(df: pd.DataFrame) -> np.ndarray:
    if "atr_14" in df.columns:
        return pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)
    if "atr" in df.columns:
        return pd.to_numeric(df["atr"], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


# ---------------------------------------------------------------------------
# Detection core — per-(side, rank) breach scan + confirmation window
# ---------------------------------------------------------------------------


def _detect_breach(
    *,
    side: int,
    bar_idx: int,
    high: float,
    low: float,
    close: float,
    atr: float,
    source_level: float,
    source_zone_low: float,
    source_zone_high: float,
) -> tuple[bool, dict[str, float]]:
    """Detect whether ``bar_idx`` constitutes a breach of the given source.

    Returns (is_breach, mechanics_dict). ``mechanics_dict`` always contains
    the four canonical mechanics fields (``penetration_abs``,
    ``penetration_atr``, ``breach_by_wick``, ``breach_by_close``); they are
    NaN/0 when no breach occurred.
    """

    if not (
        math.isfinite(source_level)
        and math.isfinite(close)
        and math.isfinite(high)
        and math.isfinite(low)
    ):
        return False, {
            "penetration_abs": float("nan"),
            "penetration_atr": float("nan"),
            "breach_by_wick": 0.0,
            "breach_by_close": 0.0,
        }
    if side == +1:
        edge = source_zone_high if math.isfinite(source_zone_high) else source_level
        breach_wick = high > edge + 1e-12
        breach_close = close > edge + 1e-12
        penetration_abs = max(0.0, high - edge) if breach_wick else 0.0
    else:
        edge = source_zone_low if math.isfinite(source_zone_low) else source_level
        breach_wick = low < edge - 1e-12
        breach_close = close < edge - 1e-12
        penetration_abs = max(0.0, edge - low) if breach_wick else 0.0
    if not breach_wick:
        return False, {
            "penetration_abs": float("nan"),
            "penetration_atr": float("nan"),
            "breach_by_wick": 0.0,
            "breach_by_close": 0.0,
        }
    penetration_atr = (
        penetration_abs / atr if math.isfinite(atr) and atr > 0 else float("nan")
    )
    return True, {
        "penetration_abs": float(penetration_abs),
        "penetration_atr": float(penetration_atr),
        "breach_by_wick": 1.0 if breach_wick else 0.0,
        "breach_by_close": 1.0 if breach_close else 0.0,
    }


def _check_rejection(
    *,
    side: int,
    close: float,
    source_level: float,
    source_zone_low: float,
    source_zone_high: float,
) -> bool:
    """True if the bar's close has reclaimed back across the source edge."""

    if not (math.isfinite(close) and math.isfinite(source_level)):
        return False
    if side == +1:
        edge = source_zone_high if math.isfinite(source_zone_high) else source_level
        return close < edge - 1e-12
    else:
        edge = source_zone_low if math.isfinite(source_zone_low) else source_level
        return close > edge + 1e-12


def _classify_breach(
    *,
    breach: _BreachState,
    confirmed_at: int | None,
    confirmation_window_bars: int,
    bar_idx: int,
    is_last_bar: bool,
) -> int:
    """Determine the canonical sweep class given the breach + confirmation."""

    if confirmed_at is None:
        # No reclaim yet.
        if bar_idx - breach.breach_idx >= confirmation_window_bars:
            # Window closed without rejection.
            if breach.breach_by_close:
                return SWEEP_CLASS_ACCEPTED_BREAKOUT
            return SWEEP_CLASS_PROBED
        # Within window, not yet rejected.
        if is_last_bar:
            return SWEEP_CLASS_UNRESOLVED
        return SWEEP_CLASS_UNRESOLVED
    # Confirmed.
    if confirmed_at == breach.breach_idx:
        return SWEEP_CLASS_SAME_BAR
    return SWEEP_CLASS_DELAYED_REJECTION


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def _quality_score(components: dict[str, float]) -> float:
    weights = SWEEP_QUALITY_WEIGHTS
    total_w = sum(weights.values())
    if total_w <= 0:
        return float("nan")
    s = 0.0
    for k, w in weights.items():
        v = components.get(k, 0.0)
        if not math.isfinite(v):
            v = 0.0
        s += w * float(np.clip(v, 0.0, 1.0))
    return float(np.clip(s / total_w, 0.0, 1.0))


def _component_penetration(penetration_atr: float) -> float:
    if not math.isfinite(penetration_atr) or penetration_atr <= 0:
        return 0.0
    # 0.25 ATR penetration scores ~0.5; saturates by 1.0 ATR.
    return float(np.clip(penetration_atr / 0.5, 0.0, 1.0))


def _component_rejection(penetration_atr: float, latency_bars: int) -> float:
    if not math.isfinite(penetration_atr) or penetration_atr <= 0:
        return 0.0
    latency_penalty = max(0.0, 1.0 - 0.2 * max(0, latency_bars))
    depth_bonus = float(np.clip(penetration_atr, 0.0, 1.0))
    return float(np.clip(0.5 * latency_penalty + 0.5 * depth_bonus, 0.0, 1.0))


def _component_volume(df: pd.DataFrame, idx: int) -> float:
    for col in ("vol_ratio", "volume_ratio_20"):
        if col in df.columns:
            v = float(pd.to_numeric(df[col].iat[idx], errors="coerce"))
            if math.isfinite(v):
                return float(np.clip((v - 1.0) / 1.0, 0.0, 1.0))
    return 0.5


def _component_regime(df: pd.DataFrame, idx: int) -> float:
    if "regime" in df.columns:
        v = float(pd.to_numeric(df["regime"].iat[idx], errors="coerce"))
        if math.isfinite(v):
            return 0.5  # regime info present; neutral pending calibration
    return 0.5


def _build_quality_components(
    df: pd.DataFrame,
    *,
    i: int,
    breach: _BreachState,
    latency: int,
) -> dict[str, float]:
    """Pure helper: assemble the 7-component quality vector at the confirm
    bar. Public so the same logic can be re-used by ablation studies."""

    return {
        "source_strength_component": float(np.clip(breach.source_strength, 0.0, 1.0)),
        "penetration_component": _component_penetration(breach.penetration_atr),
        "rejection_component": _component_rejection(breach.penetration_atr, latency),
        "displacement_followthrough_component": 0.5,
        "regime_context_component": _component_regime(df, i),
        "volume_confirmation_component": _component_volume(df, i),
        "crowding_component": _component_crowding(df, i),
    }


def _component_crowding(df: pd.DataFrame, idx: int) -> float:
    if "liq_active_total_count" in df.columns:
        v = float(pd.to_numeric(df["liq_active_total_count"].iat[idx], errors="coerce"))
        if math.isfinite(v):
            # Lower active count → less crowded → higher score.
            return float(np.clip(1.0 - v / 12.0, 0.0, 1.0))
    return 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _family_eligible(
    family: str,
    strength: float,
    age_bars: int,
    *,
    family_min_strength: dict[str, float],
    family_min_age: dict[str, int],
    family_max_age: dict[str, int],
) -> bool:
    """Step 11B: per-family eligibility gate.

    The unified-source framework produces a *complete* ladder; final sweeps
    is responsible for applying the doctrine-level filters that block
    unwanted source over-fire (especially the dense S/R 3-deep ladder).
    """

    if not family:
        return False
    min_s = family_min_strength.get(family, 0.0)
    min_age = family_min_age.get(family, 0)
    if not math.isfinite(strength):
        return False
    if strength < min_s - 1e-12:
        return False
    if age_bars < min_age:
        return False
    max_age = family_max_age.get(family)
    if max_age is not None and age_bars > max_age:
        return False
    return True


def _penetration_eligible(
    family: str,
    penetration_atr: float,
    *,
    family_min_penetration_atr: dict[str, float],
) -> bool:
    min_pen = family_min_penetration_atr.get(family, 0.0)
    if min_pen <= 0:
        return True
    return bool(math.isfinite(penetration_atr) and penetration_atr >= min_pen - 1e-12)


def _historical_max_distance_atr(
    *,
    side: int,
    source_level: float,
    active_start_idx: int,
    breach_idx: int,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
) -> float:
    """Maximum side-correct source-to-price distance observed before breach.

    This is causal by construction: only bars strictly before ``breach_idx``
    are examined.
    """

    if not math.isfinite(source_level):
        return float("nan")
    if active_start_idx < 0 or breach_idx <= active_start_idx:
        return float("nan")
    start = max(0, active_start_idx)
    best = float("nan")
    for j in range(start, breach_idx):
        atr = atr_arr[j]
        close = close_arr[j]
        if not (math.isfinite(atr) and atr > 0 and math.isfinite(close)):
            continue
        dist = (
            (source_level - close) / atr if side == +1 else (close - source_level) / atr
        )
        if not math.isfinite(dist):
            continue
        if not math.isfinite(best) or dist > best:
            best = dist
    return best


def _pre_breach_distance_atr(
    *,
    side: int,
    source_level: float,
    breach_idx: int,
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
) -> float:
    """Distance from the prior close to the source, in ATR."""

    ref_idx = breach_idx - 1
    if ref_idx < 0 or not math.isfinite(source_level):
        return float("nan")
    atr = atr_arr[ref_idx]
    close = close_arr[ref_idx]
    if not (math.isfinite(atr) and atr > 0 and math.isfinite(close)):
        return float("nan")
    return (source_level - close) / atr if side == +1 else (close - source_level) / atr


def _historical_distance_eligible(
    family: str,
    history_max_distance_atr: float,
    *,
    family_min_history_max_distance_atr: dict[str, float],
) -> bool:
    min_required = family_min_history_max_distance_atr.get(family, 0.0)
    if min_required <= 0:
        return True
    return bool(
        math.isfinite(history_max_distance_atr)
        and history_max_distance_atr >= min_required - 1e-12
    )


def _volume_or_displacement_confirmed(df: pd.DataFrame, idx: int) -> bool:
    volume_ok = False
    displacement_ok = False
    for col in ("vol_ratio", "volume_ratio_20"):
        if col in df.columns:
            value = float(pd.to_numeric(df[col].iat[idx], errors="coerce"))
            if math.isfinite(value) and value > 1.0:
                volume_ok = True
                break
    if "displacement_flag" in df.columns:
        disp = float(pd.to_numeric(df["displacement_flag"].iat[idx], errors="coerce"))
        displacement_ok = math.isfinite(disp) and disp > 0
    return bool(volume_ok or displacement_ok)


def _is_standard_liquidity_sweep(
    breach: _BreachState,
    components: dict[str, float],
    *,
    standard_min_pre_breach_distance_atr: float,
    standard_exceptional_penetration_atr: float,
    standard_exceptional_rejection_component: float,
) -> bool:
    pre_breach_distance = breach.source_pre_breach_distance_atr
    if (
        math.isfinite(pre_breach_distance)
        and pre_breach_distance >= standard_min_pre_breach_distance_atr - 1e-12
    ):
        return True
    rejection_component = float(components.get("rejection_component", float("nan")))
    return bool(
        math.isfinite(breach.penetration_atr)
        and breach.penetration_atr >= standard_exceptional_penetration_atr - 1e-12
        and math.isfinite(rejection_component)
        and rejection_component >= standard_exceptional_rejection_component - 1e-12
    )


def _is_tradeable_candidate(
    df: pd.DataFrame,
    *,
    confirm_idx: int,
    breach: _BreachState,
    sweep_class: int,
    quality: float,
    components: dict[str, float],
    is_micro_interaction: bool,
    tradeable_min_quality_by_class: dict[int, float],
    tradeable_min_penetration_atr_by_class: dict[int, float],
    tradeable_family_min_penetration_atr: dict[str, float],
    tradeable_family_min_strength: dict[str, float],
    tradeable_min_rejection_component: float,
    tradeable_max_active_sources: int,
    tradeable_require_volume_or_displacement: bool,
    session_tradeable_min_rejection_component: float,
    session_tradeable_require_volume_or_displacement: bool,
) -> bool:
    if is_micro_interaction:
        return False
    min_q = tradeable_min_quality_by_class.get(sweep_class)
    min_pen = tradeable_min_penetration_atr_by_class.get(sweep_class)
    if min_q is None or min_pen is None:
        return False
    if not math.isfinite(quality) or quality < min_q - 1e-12:
        return False
    min_pen = max(
        min_pen, tradeable_family_min_penetration_atr.get(breach.source_family, 0.0)
    )
    if (
        not math.isfinite(breach.penetration_atr)
        or breach.penetration_atr < min_pen - 1e-12
    ):
        return False
    rejection_component = float(components.get("rejection_component", float("nan")))
    if (
        not math.isfinite(rejection_component)
        or rejection_component < tradeable_min_rejection_component - 1e-12
    ):
        return False
    if breach.source_strength < tradeable_family_min_strength.get(
        breach.source_family, 0.0
    ):
        return False
    if "liq_active_total_count" in df.columns:
        active = float(
            pd.to_numeric(
                df["liq_active_total_count"].iat[confirm_idx], errors="coerce"
            )
        )
        if math.isfinite(active) and active > tradeable_max_active_sources:
            return False
    if (
        tradeable_require_volume_or_displacement
        and not _volume_or_displacement_confirmed(df, confirm_idx)
    ):
        return False
    if breach.source_family in {"session_high", "session_low"}:
        if (
            not math.isfinite(rejection_component)
            or rejection_component < session_tradeable_min_rejection_component - 1e-12
        ):
            return False
        if session_tradeable_require_volume_or_displacement:
            if not _volume_or_displacement_confirmed(df, confirm_idx):
                return False
    return True


def add_final_sweeps(
    df: pd.DataFrame,
    *,
    confirmation_window_bars: int = DEFAULT_CONFIRMATION_WINDOW_BARS,
    follow_through_window_bars: int = DEFAULT_FOLLOW_THROUGH_WINDOW_BARS,
    min_penetration_atr_for_same_bar: float = DEFAULT_MIN_PENETRATION_ATR_FOR_SAME_BAR,
    min_wick_prominence: float = DEFAULT_MIN_WICK_PROMINENCE,
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
    family_min_strength: dict[str, float] | None = None,
    family_min_age_bars: dict[str, int] | None = None,
    family_max_age_bars: dict[str, int] | None = None,
    family_min_penetration_atr: dict[str, float] | None = None,
    family_min_history_max_distance_atr: dict[str, float] | None = None,
    standard_min_pre_breach_distance_atr: float = 0.0,
    standard_exceptional_penetration_atr: float = float("inf"),
    standard_exceptional_rejection_component: float = float("inf"),
    tradeable_min_quality_by_class: dict[int, float] | None = None,
    tradeable_min_penetration_atr_by_class: dict[int, float] | None = None,
    tradeable_family_min_penetration_atr: dict[str, float] | None = None,
    tradeable_family_min_strength: dict[str, float] | None = None,
    tradeable_min_rejection_component: float = 0.0,
    tradeable_max_active_sources: int = DEFAULT_TRADEABLE_MAX_ACTIVE_SOURCES,
    tradeable_require_volume_or_displacement: bool = False,
    session_tradeable_min_rejection_component: float = DEFAULT_SESSION_TRADEABLE_MIN_REJECTION_COMPONENT,
    session_tradeable_require_volume_or_displacement: bool = DEFAULT_SESSION_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT,
    consume_confirmed_source_instances: bool = True,
) -> pd.DataFrame:
    """Detect final sweeps off the unified liquidity ladder.

    The frame must already have the dense ladder columns produced by
    :func:`src.indicators.sweeps_v2.unified_sources.add_unified_liquidity_sources`.

    Step 11B repair (versus the original Step 11 sketch):

    * **Persistent breach tracking by stable cluster key**. The unified
      ladder is by-construction filtered to only sources on the right side
      of close, so once price closes through a level the level falls out
      of the ladder. The original implementation keyed open breaches by
      ``(side, rank)`` which therefore never observed a close-acceptance.
      The repair keys open breaches by ``(side, rounded_level)``: the
      breach state stores the original source geometry at breach time and
      keeps advancing it across bars even after the level disappears from
      the ladder.

    * **Penetration + wick-prominence thresholds** for same-bar sweeps.
      Tiny noise wicks below ``min_penetration_atr_for_same_bar`` ATR or
      ``min_wick_prominence`` of the bar's range no longer count.

    * **Cluster cooldown + consumption**. After a confirmed sweep, the same
      source instance is retired for sweep emission until a genuinely new
      instance appears (detected via ``active_start_idx`` change). A
      cluster-level cooldown still suppresses nearby re-fires.

    * **Family-specific eligibility**. Per-family minimum source age,
      strength, and penetration floors plus a maximum age cap. Especially
      relevant for S/R and session levels, where source density otherwise
      dominates by sheer count.

    * **Followthrough column-name fix**. ``bos_bull|bos_bear``,
      ``choch_bull|choch_bear``, ``regime`` (the actual upstream column
      names) instead of the placeholder ``*_flag`` / ``regime_state``.
    """

    if df is None:
        raise ValueError("add_final_sweeps: df must not be None")
    if len(df) == 0:
        return df.copy()

    family_min_strength = family_min_strength or DEFAULT_FAMILY_MIN_STRENGTH
    family_min_age_bars = family_min_age_bars or DEFAULT_FAMILY_MIN_AGE_BARS
    family_max_age_bars = family_max_age_bars or DEFAULT_FAMILY_MAX_AGE_BARS
    family_min_penetration_atr = (
        family_min_penetration_atr or DEFAULT_FAMILY_MIN_PENETRATION_ATR
    )
    family_min_history_max_distance_atr = (
        family_min_history_max_distance_atr
        or DEFAULT_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR
    )
    tradeable_min_quality_by_class = (
        tradeable_min_quality_by_class or DEFAULT_TRADEABLE_MIN_QUALITY_BY_CLASS
    )
    tradeable_min_penetration_atr_by_class = (
        tradeable_min_penetration_atr_by_class
        or DEFAULT_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS
    )
    tradeable_family_min_penetration_atr = tradeable_family_min_penetration_atr or {
        family: 0.0 for family in family_min_strength.keys()
    }
    tradeable_family_min_strength = (
        tradeable_family_min_strength or DEFAULT_TRADEABLE_FAMILY_MIN_STRENGTH
    )

    out = df.copy()
    n = len(out)

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    atr = _safe_atr(out)
    timestamps = (
        pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        if "timestamp" in out.columns
        else pd.Series([pd.NaT] * n)
    )

    # Initialize all output arrays.
    arr: dict[str, np.ndarray] = {}
    for col in FINAL_SWEEPS_COLUMNS:
        if col in (
            "sweep_primary_family",
            "sweep_attribution_families",
            "sweep_breach_source_family",
            "sweep_selectivity_class",
        ):
            arr[col] = np.full(n, "", dtype=object)
        elif col == "sweep_breach_timestamp":
            arr[col] = np.full(n, pd.NaT, dtype="object")
        else:
            arr[col] = np.full(n, np.nan, dtype=float)

    # Pre-extract every ladder column once. Cuts per-bar Python overhead.
    sides = ((+1, "above"), (-1, "below"))
    ladder: dict[tuple[str, int, str], np.ndarray] = {}
    for _, side_label in sides:
        for rank in range(1, LIQ_LADDER_DEPTH + 1):
            for fname in (
                "cluster_id",
                "level",
                "zone_low",
                "zone_high",
                "strength",
                "age_bars",
                "origin_idx",
                "active_start_idx",
                "primary_family",
                "attribution_families",
            ):
                a = _ladder_array(out, side_label, rank, fname)
                if a is None:
                    a = np.full(
                        n,
                        (
                            ""
                            if fname.endswith("family") or fname.endswith("families")
                            else np.nan
                        ),
                        dtype=(
                            object
                            if fname.endswith("family") or fname.endswith("families")
                            else float
                        ),
                    )
                ladder[(side_label, rank, fname)] = a

    # Breach state keyed by stable cluster identity (side, rounded_level).
    # Storing the level rounded to 4 decimal places gives us cross-bar
    # stability for typical FX/metals tick sizes; the dedup tolerance in
    # unified_sources already merges anything closer than this.
    open_breaches: dict[tuple[int, float], _BreachState] = {}
    cooldown_until: dict[tuple[int, float], int] = {}
    interaction_state: dict[tuple[int, float], int] = {}
    consumed_source_instances: set[tuple[int, float, int, str]] = set()
    next_event_id = 1

    def _round_key(side: int, level: float) -> tuple[int, float]:
        return (side, round(float(level), 4))

    for i in range(n):
        bar_atr = atr[i]
        bar_close = close[i]
        bar_high = high[i]
        bar_low = low[i]
        bar_range = bar_high - bar_low if math.isfinite(bar_high - bar_low) else 0.0

        # ── Step A: try to open new breaches from currently visible slots ──
        # We iterate slots in (above L1..L5, below L1..L5) order so that
        # the per-bar single-event breach columns get stamped by the
        # nearest cluster first (then overwritten by stronger if any).
        for side, side_label in sides:
            for rank in range(1, LIQ_LADDER_DEPTH + 1):
                cid = ladder[(side_label, rank, "cluster_id")][i]
                if not math.isfinite(cid):
                    continue
                source_level = float(ladder[(side_label, rank, "level")][i])
                if not math.isfinite(source_level):
                    continue
                source_zone_low = float(ladder[(side_label, rank, "zone_low")][i])
                source_zone_high = float(ladder[(side_label, rank, "zone_high")][i])
                if not math.isfinite(source_zone_low):
                    source_zone_low = source_level
                if not math.isfinite(source_zone_high):
                    source_zone_high = source_level
                source_strength = float(ladder[(side_label, rank, "strength")][i])
                source_age = ladder[(side_label, rank, "age_bars")][i]
                source_age_int = int(source_age) if math.isfinite(source_age) else 0
                source_origin_idx = (
                    int(ladder[(side_label, rank, "origin_idx")][i])
                    if math.isfinite(ladder[(side_label, rank, "origin_idx")][i])
                    else -1
                )
                source_active_start_idx = (
                    int(ladder[(side_label, rank, "active_start_idx")][i])
                    if math.isfinite(ladder[(side_label, rank, "active_start_idx")][i])
                    else -1
                )
                source_family = str(
                    ladder[(side_label, rank, "primary_family")][i] or ""
                )
                source_attribution = str(
                    ladder[(side_label, rank, "attribution_families")][i]
                    or source_family
                )

                # Family-specific eligibility filter
                if not _family_eligible(
                    source_family,
                    source_strength,
                    source_age_int,
                    family_min_strength=family_min_strength,
                    family_min_age=family_min_age_bars,
                    family_max_age=family_max_age_bars,
                ):
                    continue

                key = _round_key(side, source_level)
                source_instance_key = (
                    side,
                    key[1],
                    source_active_start_idx,
                    source_family,
                )
                if key in open_breaches:
                    continue
                if (
                    consume_confirmed_source_instances
                    and source_instance_key in consumed_source_instances
                ):
                    continue
                if cooldown_until.get(key, -1) >= i:
                    continue

                is_breach, mech = _detect_breach(
                    side=side,
                    bar_idx=i,
                    high=bar_high,
                    low=bar_low,
                    close=bar_close,
                    atr=bar_atr,
                    source_level=source_level,
                    source_zone_low=source_zone_low,
                    source_zone_high=source_zone_high,
                )
                if not is_breach:
                    continue
                if not _penetration_eligible(
                    source_family,
                    mech["penetration_atr"],
                    family_min_penetration_atr=family_min_penetration_atr,
                ):
                    continue
                history_max_distance_atr = _historical_max_distance_atr(
                    side=side,
                    source_level=source_level,
                    active_start_idx=source_active_start_idx,
                    breach_idx=i,
                    close_arr=close,
                    atr_arr=atr,
                )
                pre_breach_distance_atr = _pre_breach_distance_atr(
                    side=side,
                    source_level=source_level,
                    breach_idx=i,
                    close_arr=close,
                    atr_arr=atr,
                )
                if not _historical_distance_eligible(
                    source_family,
                    history_max_distance_atr,
                    family_min_history_max_distance_atr=family_min_history_max_distance_atr,
                ):
                    continue

                width_abs = max(0.0, source_zone_high - source_zone_low)
                pen_width_frac = (
                    mech["penetration_abs"] / width_abs
                    if width_abs > 1e-12
                    else float("nan")
                )
                breach = _BreachState(
                    breach_idx=i,
                    breach_close=bar_close,
                    source_level=source_level,
                    source_zone_low=source_zone_low,
                    source_zone_high=source_zone_high,
                    source_strength=source_strength,
                    source_age_bars=source_age_int,
                    source_family=source_family,
                    source_attribution=source_attribution,
                    source_cluster_id=cid,
                    source_origin_idx=source_origin_idx,
                    source_active_start_idx=source_active_start_idx,
                    source_history_max_distance_atr=history_max_distance_atr,
                    source_pre_breach_distance_atr=pre_breach_distance_atr,
                    side=side,
                    breach_by_wick=mech["breach_by_wick"] > 0,
                    breach_by_close=mech["breach_by_close"] > 0,
                    penetration_abs=mech["penetration_abs"],
                    penetration_atr=mech["penetration_atr"],
                    penetration_source_width_frac=pen_width_frac,
                )
                open_breaches[key] = breach

                # Mark interaction = probed for this cluster
                interaction_state[key] = max(
                    interaction_state.get(key, SWEEP_INTERACTION_UNTOUCHED),
                    (
                        SWEEP_INTERACTION_FULLY_SWEPT
                        if mech["breach_by_close"] > 0
                        else SWEEP_INTERACTION_PARTIALLY_SWEPT
                    ),
                )

                _stamp_breach_row(arr, i, breach, mech, timestamps)

        # ── Step B: advance every open breach (visible or not) ──────────
        for key, breach in list(open_breaches.items()):
            bars_since = i - breach.breach_idx

            # ----- Same-bar handling ----------------------------------------
            if bars_since == 0:
                # Apply same-bar quality filter
                close_back_inside = (
                    bar_close < breach.source_zone_high - 1e-12
                    if breach.side == +1
                    else bar_close > breach.source_zone_low + 1e-12
                )
                prominence = (
                    breach.penetration_abs / max(bar_range, 1e-12)
                    if bar_range > 0
                    else 0.0
                )
                meets_pen = (
                    math.isfinite(breach.penetration_atr)
                    and breach.penetration_atr >= min_penetration_atr_for_same_bar
                )
                meets_prom = prominence >= min_wick_prominence
                if close_back_inside and meets_pen and meets_prom:
                    breach.confirmed = True
                    breach.confirm_idx = i
                    components = _build_quality_components(
                        out, i=i, breach=breach, latency=0
                    )
                    _emit_confirmed_sweep(
                        out,
                        arr,
                        confirm_idx=i,
                        breach=breach,
                        sweep_class=SWEEP_CLASS_SAME_BAR,
                        event_id=next_event_id,
                        components=components,
                        standard_min_pre_breach_distance_atr=standard_min_pre_breach_distance_atr,
                        standard_exceptional_penetration_atr=standard_exceptional_penetration_atr,
                        standard_exceptional_rejection_component=standard_exceptional_rejection_component,
                        timestamps=timestamps,
                        tradeable_min_quality_by_class=tradeable_min_quality_by_class,
                        tradeable_min_penetration_atr_by_class=tradeable_min_penetration_atr_by_class,
                        tradeable_family_min_penetration_atr=tradeable_family_min_penetration_atr,
                        tradeable_family_min_strength=tradeable_family_min_strength,
                        tradeable_min_rejection_component=tradeable_min_rejection_component,
                        tradeable_max_active_sources=tradeable_max_active_sources,
                        tradeable_require_volume_or_displacement=tradeable_require_volume_or_displacement,
                        session_tradeable_min_rejection_component=session_tradeable_min_rejection_component,
                        session_tradeable_require_volume_or_displacement=session_tradeable_require_volume_or_displacement,
                    )
                    next_event_id += 1
                    interaction_state[key] = SWEEP_INTERACTION_FULLY_SWEPT
                    cooldown_until[key] = i + cooldown_bars
                    if consume_confirmed_source_instances:
                        consumed_source_instances.add(
                            (
                                breach.side,
                                key[1],
                                breach.source_active_start_idx,
                                breach.source_family,
                            )
                        )
                    del open_breaches[key]
                # Else: stay open (close beyond → wait for reclaim, or
                # close-back-inside but failed quality filter → wait too)
                continue

            # ----- Subsequent bars -----------------------------------------
            close_back_across = (
                bar_close < breach.source_zone_high - 1e-12
                if breach.side == +1
                else bar_close > breach.source_zone_low + 1e-12
            )

            if close_back_across:
                if bars_since <= confirmation_window_bars:
                    # Delayed-rejection sweep
                    breach.confirmed = True
                    breach.confirm_idx = i
                    components = _build_quality_components(
                        out, i=i, breach=breach, latency=bars_since
                    )
                    _emit_confirmed_sweep(
                        out,
                        arr,
                        confirm_idx=i,
                        breach=breach,
                        sweep_class=SWEEP_CLASS_DELAYED_REJECTION,
                        event_id=next_event_id,
                        components=components,
                        standard_min_pre_breach_distance_atr=standard_min_pre_breach_distance_atr,
                        standard_exceptional_penetration_atr=standard_exceptional_penetration_atr,
                        standard_exceptional_rejection_component=standard_exceptional_rejection_component,
                        timestamps=timestamps,
                        tradeable_min_quality_by_class=tradeable_min_quality_by_class,
                        tradeable_min_penetration_atr_by_class=tradeable_min_penetration_atr_by_class,
                        tradeable_family_min_penetration_atr=tradeable_family_min_penetration_atr,
                        tradeable_family_min_strength=tradeable_family_min_strength,
                        tradeable_min_rejection_component=tradeable_min_rejection_component,
                        tradeable_max_active_sources=tradeable_max_active_sources,
                        tradeable_require_volume_or_displacement=tradeable_require_volume_or_displacement,
                        session_tradeable_min_rejection_component=session_tradeable_min_rejection_component,
                        session_tradeable_require_volume_or_displacement=session_tradeable_require_volume_or_displacement,
                    )
                    next_event_id += 1
                    interaction_state[key] = SWEEP_INTERACTION_FULLY_SWEPT
                    cooldown_until[key] = i + cooldown_bars
                    if consume_confirmed_source_instances:
                        consumed_source_instances.add(
                            (
                                breach.side,
                                key[1],
                                breach.source_active_start_idx,
                                breach.source_family,
                            )
                        )
                    del open_breaches[key]
                    continue
                else:
                    # Late reclaim → failed-breakout-reclaim
                    _emit_failed_breakout_reclaim(
                        arr, i, breach, next_event_id, timestamps
                    )
                    next_event_id += 1
                    interaction_state[key] = SWEEP_INTERACTION_RECLAIMED
                    cooldown_until[key] = i + cooldown_bars
                    del open_breaches[key]
                    continue

            if bars_since > confirmation_window_bars:
                # Window closed without reclaim → accepted breakout
                # (or probed if the close never accepted beyond)
                sweep_class = (
                    SWEEP_CLASS_ACCEPTED_BREAKOUT
                    if breach.breach_by_close
                    else SWEEP_CLASS_PROBED
                )
                _emit_terminal_state(
                    arr,
                    i,
                    breach,
                    sweep_class=sweep_class,
                    event_id=next_event_id,
                    timestamps=timestamps,
                )
                next_event_id += 1
                if sweep_class == SWEEP_CLASS_ACCEPTED_BREAKOUT:
                    interaction_state[key] = SWEEP_INTERACTION_ACCEPTED_BEYOND
                cooldown_until[key] = i + cooldown_bars
                del open_breaches[key]
                continue

            # Still inside the confirmation window with no resolution yet
            if math.isnan(arr["sweep_class"][i]):
                arr["sweep_class"][i] = float(SWEEP_CLASS_UNRESOLVED)
                arr["sweep_breach_idx"][i] = float(breach.breach_idx)
                arr["sweep_source_id"][i] = float(breach.source_cluster_id)
                arr["sweep_source_cluster_id"][i] = float(breach.source_cluster_id)
                arr["sweep_primary_family"][i] = breach.source_family
                arr["sweep_attribution_families"][i] = breach.source_attribution
                arr["sweep_source_side"][i] = float(breach.side)
                arr["sweep_source_level"][i] = float(breach.source_level)
                arr["sweep_source_strength"][i] = float(breach.source_strength)

        # ── Step C: project per-slot interaction phase ─────────────────
        for side, side_label in sides:
            for rank in range(1, LIQ_LADDER_DEPTH + 1):
                cid = ladder[(side_label, rank, "cluster_id")][i]
                if not math.isfinite(cid):
                    continue
                lvl = float(ladder[(side_label, rank, "level")][i])
                if not math.isfinite(lvl):
                    continue
                key = _round_key(side, lvl)
                phase = interaction_state.get(key, SWEEP_INTERACTION_UNTOUCHED)
                arr[f"sweep_interaction_phase_{side_label}_l{rank}"][i] = float(phase)

    # End-of-frame: emit unresolved class for any breach still open
    if open_breaches:
        last = n - 1
        if math.isnan(arr["sweep_class"][last]):
            arr["sweep_class"][last] = float(SWEEP_CLASS_UNRESOLVED)

    # Compute follow-through diagnostics (research-only) for confirmed sweeps.
    _attach_follow_through(
        out, arr, follow_through_window_bars=follow_through_window_bars
    )
    _finalize_selectivity_classes(arr)

    # Single concat to keep the frame de-fragmented.
    addition = pd.DataFrame(arr, index=out.index)
    duplicates = [c for c in addition.columns if c in out.columns]
    if duplicates:
        out = out.drop(columns=duplicates)
    out = pd.concat([out, addition], axis=1)
    return out


def _stamp_breach_row(
    arr: dict[str, np.ndarray],
    idx: int,
    breach: _BreachState,
    mech: dict[str, float],
    timestamps: pd.Series,
) -> None:
    """At the breach bar, stamp the per-bar single-event breach columns.

    If multiple slots breach on the same bar, the first stamp wins for the
    per-bar columns; subsequent slots append into the audit/list. The
    deterministic precedence (family precedence + strength) is enforced by
    the order of slot scanning and the LIQ_FAMILY_PRECEDENCE dict so this is
    stable.
    """

    cur = arr["sweep_breach_flag"][idx]
    # If a higher-precedence breach already stamped this bar, skip.
    if math.isfinite(cur) and cur > 0:
        existing_strength = arr["sweep_breach_source_strength"][idx]
        if (
            math.isfinite(existing_strength)
            and existing_strength >= breach.source_strength
        ):
            return
    arr["sweep_breach_flag"][idx] = 1.0
    arr["sweep_breach_side"][idx] = float(breach.side)
    arr["sweep_breach_source_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_breach_source_cluster_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_breach_source_family"][idx] = breach.source_family
    arr["sweep_breach_source_level"][idx] = float(breach.source_level)
    arr["sweep_breach_source_strength"][idx] = float(breach.source_strength)
    if idx < len(timestamps):
        arr["sweep_breach_timestamp"][idx] = timestamps.iat[idx]
    arr["penetration_abs"][idx] = float(mech["penetration_abs"])
    arr["penetration_atr"][idx] = float(mech["penetration_atr"])
    if (
        math.isfinite(breach.source_zone_high - breach.source_zone_low)
        and (breach.source_zone_high - breach.source_zone_low) > 1e-12
    ):
        arr["penetration_source_width_frac"][idx] = float(
            mech["penetration_abs"] / (breach.source_zone_high - breach.source_zone_low)
        )
    arr["breach_by_wick"][idx] = float(mech["breach_by_wick"])
    arr["breach_by_close"][idx] = float(mech["breach_by_close"])
    # Gap detection: if previous bar's close was beyond the source already.
    arr["breach_gap_flag"][idx] = 0.0


def _emit_confirmed_sweep(
    df: pd.DataFrame,
    arr: dict[str, np.ndarray],
    *,
    confirm_idx: int,
    breach: _BreachState,
    sweep_class: int,
    event_id: int,
    components: dict[str, float],
    standard_min_pre_breach_distance_atr: float,
    standard_exceptional_penetration_atr: float,
    standard_exceptional_rejection_component: float,
    timestamps: pd.Series,
    session_tradeable_min_rejection_component: float,
    session_tradeable_require_volume_or_displacement: bool,
    tradeable_min_quality_by_class: dict[int, float],
    tradeable_min_penetration_atr_by_class: dict[int, float],
    tradeable_family_min_penetration_atr: dict[str, float],
    tradeable_family_min_strength: dict[str, float],
    tradeable_min_rejection_component: float,
    tradeable_max_active_sources: int,
    tradeable_require_volume_or_displacement: bool,
) -> None:
    arr["sweep_flag"][confirm_idx] = 1.0
    arr["sweep_side"][confirm_idx] = float(breach.side)
    arr["sweep_class"][confirm_idx] = float(sweep_class)
    arr["sweep_breach_idx"][confirm_idx] = float(breach.breach_idx)
    arr["sweep_confirm_idx"][confirm_idx] = float(confirm_idx)
    arr["sweep_latency_bars"][confirm_idx] = float(confirm_idx - breach.breach_idx)
    arr["sweep_event_id"][confirm_idx] = float(event_id)
    arr["sweep_source_id"][confirm_idx] = float(breach.source_cluster_id)
    arr["sweep_source_cluster_id"][confirm_idx] = float(breach.source_cluster_id)
    arr["sweep_primary_family"][confirm_idx] = breach.source_family
    arr["sweep_attribution_families"][confirm_idx] = breach.source_attribution
    arr["sweep_source_side"][confirm_idx] = float(breach.side)
    arr["sweep_source_level"][confirm_idx] = float(breach.source_level)
    arr["sweep_source_strength"][confirm_idx] = float(breach.source_strength)
    quality = _quality_score(components)
    arr["sweep_quality_score"][confirm_idx] = float(quality)
    arr["sweep_q_source_strength"][confirm_idx] = float(
        components.get("source_strength_component", float("nan"))
    )
    arr["sweep_q_penetration"][confirm_idx] = float(
        components.get("penetration_component", float("nan"))
    )
    arr["sweep_q_rejection"][confirm_idx] = float(
        components.get("rejection_component", float("nan"))
    )
    arr["sweep_q_displacement_followthrough"][confirm_idx] = float(
        components.get("displacement_followthrough_component", float("nan"))
    )
    arr["sweep_q_regime_context"][confirm_idx] = float(
        components.get("regime_context_component", float("nan"))
    )
    arr["sweep_q_volume_confirmation"][confirm_idx] = float(
        components.get("volume_confirmation_component", float("nan"))
    )
    arr["sweep_q_crowding"][confirm_idx] = float(
        components.get("crowding_component", float("nan"))
    )
    arr["sweep_pre_breach_distance_atr"][confirm_idx] = float(
        breach.source_pre_breach_distance_atr
    )
    arr["sweep_history_max_distance_atr"][confirm_idx] = float(
        breach.source_history_max_distance_atr
    )
    is_standard_liquidity = _is_standard_liquidity_sweep(
        breach,
        components,
        standard_min_pre_breach_distance_atr=standard_min_pre_breach_distance_atr,
        standard_exceptional_penetration_atr=standard_exceptional_penetration_atr,
        standard_exceptional_rejection_component=standard_exceptional_rejection_component,
    )
    is_micro_interaction = not is_standard_liquidity
    arr["sweep_is_micro_interaction"][confirm_idx] = (
        1.0 if is_micro_interaction else 0.0
    )
    arr["sweep_is_standard_liquidity"][confirm_idx] = (
        1.0 if is_standard_liquidity else 0.0
    )
    arr["sweep_is_displacement_confirmed"][confirm_idx] = 0.0
    tradeable = _is_tradeable_candidate(
        df,
        confirm_idx=confirm_idx,
        breach=breach,
        sweep_class=sweep_class,
        quality=quality,
        components=components,
        is_micro_interaction=is_micro_interaction,
        tradeable_min_quality_by_class=tradeable_min_quality_by_class,
        tradeable_min_penetration_atr_by_class=tradeable_min_penetration_atr_by_class,
        tradeable_family_min_penetration_atr=tradeable_family_min_penetration_atr,
        tradeable_family_min_strength=tradeable_family_min_strength,
        tradeable_min_rejection_component=tradeable_min_rejection_component,
        tradeable_max_active_sources=tradeable_max_active_sources,
        tradeable_require_volume_or_displacement=tradeable_require_volume_or_displacement,
        session_tradeable_min_rejection_component=session_tradeable_min_rejection_component,
        session_tradeable_require_volume_or_displacement=session_tradeable_require_volume_or_displacement,
    )
    arr["sweep_is_tradeable_candidate"][confirm_idx] = 1.0 if tradeable else 0.0
    if tradeable:
        arr["sweep_selectivity_class"][
            confirm_idx
        ] = SWEEP_SELECTIVITY_TRADEABLE_CANDIDATE
    elif is_standard_liquidity:
        arr["sweep_selectivity_class"][
            confirm_idx
        ] = SWEEP_SELECTIVITY_STANDARD_LIQUIDITY
    else:
        arr["sweep_selectivity_class"][
            confirm_idx
        ] = SWEEP_SELECTIVITY_MICRO_INTERACTION
    arr["sweep_invalidated_flag"][confirm_idx] = 0.0


def _emit_terminal_state(
    arr: dict[str, np.ndarray],
    idx: int,
    breach: _BreachState,
    *,
    sweep_class: int,
    event_id: int,
    timestamps: pd.Series,
) -> None:
    """Stamp the bar where the breach window closed without confirmation."""

    arr["sweep_class"][idx] = float(sweep_class)
    arr["sweep_breach_idx"][idx] = float(breach.breach_idx)
    arr["sweep_event_id"][idx] = float(event_id)
    arr["sweep_source_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_source_cluster_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_primary_family"][idx] = breach.source_family
    arr["sweep_attribution_families"][idx] = breach.source_attribution
    arr["sweep_source_side"][idx] = float(breach.side)
    arr["sweep_source_level"][idx] = float(breach.source_level)
    arr["sweep_source_strength"][idx] = float(breach.source_strength)
    arr["sweep_invalidated_flag"][idx] = (
        1.0 if sweep_class == SWEEP_CLASS_ACCEPTED_BREAKOUT else 0.0
    )
    arr["sweep_is_tradeable_candidate"][idx] = 0.0


def _emit_failed_breakout_reclaim(
    arr: dict[str, np.ndarray],
    idx: int,
    breach: _BreachState,
    event_id: int,
    timestamps: pd.Series,
) -> None:
    """Late reclaim arriving after the confirmation window closed."""

    arr["sweep_class"][idx] = float(SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM)
    arr["sweep_breach_idx"][idx] = float(breach.breach_idx)
    arr["sweep_event_id"][idx] = float(event_id)
    arr["sweep_source_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_source_cluster_id"][idx] = float(breach.source_cluster_id)
    arr["sweep_primary_family"][idx] = breach.source_family
    arr["sweep_attribution_families"][idx] = breach.source_attribution
    arr["sweep_source_side"][idx] = float(breach.side)
    arr["sweep_source_level"][idx] = float(breach.source_level)
    arr["sweep_source_strength"][idx] = float(breach.source_strength)
    arr["sweep_invalidated_flag"][idx] = 0.0
    arr["sweep_is_tradeable_candidate"][idx] = 0.0


def _attach_follow_through(
    df: pd.DataFrame,
    arr: dict[str, np.ndarray],
    *,
    follow_through_window_bars: int,
) -> None:
    """Compute research-only follow-through flags within N bars after every
    confirmed sweep. These are causal at the later bar; NOT at confirm-time.
    """

    n = len(df)
    sweep_flag = arr["sweep_flag"]
    sweep_side = arr["sweep_side"]
    # Step 11B fix: use the actual upstream column names. ``displacement_flag``
    # exists; ``bos_flag`` / ``choch_flag`` do not — the structure stages emit
    # ``bos_bull`` / ``bos_bear`` (and choch counterparts).
    disp_arr = (
        pd.to_numeric(df["displacement_flag"], errors="coerce").to_numpy(dtype=float)
        if "displacement_flag" in df.columns
        else np.zeros(n, dtype=float)
    )
    bos_bull = (
        pd.to_numeric(df["bos_bull"], errors="coerce").to_numpy(dtype=float)
        if "bos_bull" in df.columns
        else np.zeros(n, dtype=float)
    )
    bos_bear = (
        pd.to_numeric(df["bos_bear"], errors="coerce").to_numpy(dtype=float)
        if "bos_bear" in df.columns
        else np.zeros(n, dtype=float)
    )
    choch_bull = (
        pd.to_numeric(df["choch_bull"], errors="coerce").to_numpy(dtype=float)
        if "choch_bull" in df.columns
        else np.zeros(n, dtype=float)
    )
    choch_bear = (
        pd.to_numeric(df["choch_bear"], errors="coerce").to_numpy(dtype=float)
        if "choch_bear" in df.columns
        else np.zeros(n, dtype=float)
    )
    for i in range(n):
        if not (math.isfinite(sweep_flag[i]) and sweep_flag[i] > 0):
            continue
        end = min(n - 1, i + follow_through_window_bars)
        side = sweep_side[i] if math.isfinite(sweep_side[i]) else 0.0
        # Sweeps to the upside (taking buy-side liquidity above) are
        # *bearish* setups: the expected follow-through is a downward
        # displacement / bos_bear / choch_bear. We flip the polarity by
        # sweep_side to capture only directionally-coherent follow-through.
        if side > 0:
            bos_arr = bos_bear
            choch_arr = choch_bear
        elif side < 0:
            bos_arr = bos_bull
            choch_arr = choch_bull
        else:
            bos_arr = np.maximum(bos_bull, bos_bear)
            choch_arr = np.maximum(choch_bull, choch_bear)

        arr["sweep_followed_by_displacement"][i] = 0.0
        for j in range(i, end + 1):
            if disp_arr[j] > 0:
                arr["sweep_followed_by_displacement"][i] = 1.0
                arr["sweep_displacement_within_bars"][i] = float(j - i)
                arr["sweep_displacement_strength"][i] = 1.0
                break

        arr["sweep_followed_by_bos"][i] = 0.0
        for j in range(i, end + 1):
            if bos_arr[j] > 0:
                arr["sweep_followed_by_bos"][i] = 1.0
                arr["sweep_bos_within_bars"][i] = float(j - i)
                arr["sweep_bos_strength"][i] = 1.0
                break

        arr["sweep_followed_by_choch"][i] = 0.0
        for j in range(i, end + 1):
            if choch_arr[j] > 0:
                arr["sweep_followed_by_choch"][i] = 1.0
                arr["sweep_choch_within_bars"][i] = float(j - i)
                arr["sweep_choch_strength"][i] = 1.0
                break

        # The diagnostic is "available" once the follow-through window has
        # fully elapsed; otherwise the diagnostic is partial.
        arr["sweep_research_followthrough_available"][i] = (
            1.0 if (i + follow_through_window_bars) < n else 0.0
        )


def _finalize_selectivity_classes(arr: dict[str, np.ndarray]) -> None:
    """Upgrade confirmed sweeps into the Step 11E research classes."""

    n = len(arr["sweep_flag"])
    for i in range(n):
        if not (math.isfinite(arr["sweep_flag"][i]) and arr["sweep_flag"][i] > 0):
            continue
        tradeable = math.isfinite(arr["sweep_is_tradeable_candidate"][i]) and (
            arr["sweep_is_tradeable_candidate"][i] > 0
        )
        standard = math.isfinite(arr["sweep_is_standard_liquidity"][i]) and (
            arr["sweep_is_standard_liquidity"][i] > 0
        )
        followthrough_available = math.isfinite(
            arr["sweep_research_followthrough_available"][i]
        ) and (arr["sweep_research_followthrough_available"][i] > 0)
        has_followthrough = any(
            math.isfinite(arr[col][i]) and arr[col][i] > 0
            for col in (
                "sweep_followed_by_displacement",
                "sweep_followed_by_bos",
                "sweep_followed_by_choch",
            )
        )
        displacement_confirmed = (
            standard and followthrough_available and has_followthrough
        )
        arr["sweep_is_displacement_confirmed"][i] = (
            1.0 if displacement_confirmed else 0.0
        )
        if tradeable:
            arr["sweep_selectivity_class"][i] = SWEEP_SELECTIVITY_TRADEABLE_CANDIDATE
        elif displacement_confirmed:
            arr["sweep_selectivity_class"][i] = SWEEP_SELECTIVITY_DISPLACEMENT_CONFIRMED
        elif standard:
            arr["sweep_selectivity_class"][i] = SWEEP_SELECTIVITY_STANDARD_LIQUIDITY
        else:
            arr["sweep_selectivity_class"][i] = SWEEP_SELECTIVITY_MICRO_INTERACTION


def step11b_baseline_kwargs() -> dict[str, object]:
    """Reproduce the Step 11B behavior for validator before/after runs."""

    return {
        "cooldown_bars": 6,
        "family_min_strength": STEP11B_BASELINE_FAMILY_MIN_STRENGTH.copy(),
        "family_min_age_bars": STEP11B_BASELINE_FAMILY_MIN_AGE_BARS.copy(),
        "family_max_age_bars": DEFAULT_FAMILY_MAX_AGE_BARS.copy(),
        "family_min_penetration_atr": STEP11B_BASELINE_FAMILY_MIN_PENETRATION_ATR.copy(),
        "family_min_history_max_distance_atr": STEP11C_BASELINE_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR.copy(),
        "standard_min_pre_breach_distance_atr": 0.0,
        "standard_exceptional_penetration_atr": float("inf"),
        "standard_exceptional_rejection_component": float("inf"),
        "tradeable_min_quality_by_class": STEP11B_BASELINE_TRADEABLE_MIN_QUALITY_BY_CLASS.copy(),
        "tradeable_min_penetration_atr_by_class": STEP11B_BASELINE_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS.copy(),
        "tradeable_family_min_penetration_atr": {
            family: 0.0 for family in STEP11B_BASELINE_FAMILY_MIN_STRENGTH
        },
        "tradeable_family_min_strength": STEP11B_BASELINE_TRADEABLE_FAMILY_MIN_STRENGTH.copy(),
        "tradeable_min_rejection_component": 0.0,
        "tradeable_max_active_sources": STEP11B_BASELINE_TRADEABLE_MAX_ACTIVE_SOURCES,
        "tradeable_require_volume_or_displacement": False,
        "session_tradeable_min_rejection_component": 0.0,
        "session_tradeable_require_volume_or_displacement": False,
        "consume_confirmed_source_instances": False,
    }


def step11c_default_kwargs(
    *, cooldown_bars: int = DEFAULT_COOLDOWN_BARS
) -> dict[str, object]:
    """Return the production Step 11C hardening profile."""

    return {
        "cooldown_bars": cooldown_bars,
        "family_min_strength": DEFAULT_FAMILY_MIN_STRENGTH.copy(),
        "family_min_age_bars": DEFAULT_FAMILY_MIN_AGE_BARS.copy(),
        "family_max_age_bars": DEFAULT_FAMILY_MAX_AGE_BARS.copy(),
        "family_min_penetration_atr": DEFAULT_FAMILY_MIN_PENETRATION_ATR.copy(),
        "family_min_history_max_distance_atr": STEP11C_BASELINE_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR.copy(),
        "standard_min_pre_breach_distance_atr": 0.0,
        "standard_exceptional_penetration_atr": float("inf"),
        "standard_exceptional_rejection_component": float("inf"),
        "tradeable_min_quality_by_class": DEFAULT_TRADEABLE_MIN_QUALITY_BY_CLASS.copy(),
        "tradeable_min_penetration_atr_by_class": DEFAULT_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS.copy(),
        "tradeable_family_min_penetration_atr": {
            family: 0.0 for family in DEFAULT_FAMILY_MIN_STRENGTH
        },
        "tradeable_family_min_strength": DEFAULT_TRADEABLE_FAMILY_MIN_STRENGTH.copy(),
        "tradeable_min_rejection_component": 0.0,
        "tradeable_max_active_sources": DEFAULT_TRADEABLE_MAX_ACTIVE_SOURCES,
        "tradeable_require_volume_or_displacement": False,
        "session_tradeable_min_rejection_component": 0.0,
        "session_tradeable_require_volume_or_displacement": False,
        "consume_confirmed_source_instances": True,
    }


def step11d_profile_kwargs(
    *,
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
    session_min_age_bars: int,
    session_min_history_max_distance_atr: float,
    session_min_penetration_atr: float,
) -> dict[str, object]:
    """Step 11D builds on Step 11C and only tightens session selectivity."""

    kwargs = step11c_default_kwargs(cooldown_bars=cooldown_bars)
    family_min_age = dict(kwargs["family_min_age_bars"])
    family_min_pen = dict(kwargs["family_min_penetration_atr"])
    family_min_hist = dict(kwargs["family_min_history_max_distance_atr"])
    for fam in ("session_high", "session_low"):
        family_min_age[fam] = int(session_min_age_bars)
        family_min_pen[fam] = float(session_min_penetration_atr)
        family_min_hist[fam] = float(session_min_history_max_distance_atr)
    kwargs["family_min_age_bars"] = family_min_age
    kwargs["family_min_penetration_atr"] = family_min_pen
    kwargs["family_min_history_max_distance_atr"] = family_min_hist
    kwargs["session_tradeable_min_rejection_component"] = (
        DEFAULT_SESSION_TRADEABLE_MIN_REJECTION_COMPONENT
    )
    kwargs["session_tradeable_require_volume_or_displacement"] = (
        DEFAULT_SESSION_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT
    )
    return kwargs


def step11d_default_kwargs(
    *, cooldown_bars: int = DEFAULT_COOLDOWN_BARS
) -> dict[str, object]:
    """Chosen Step 11D production profile."""

    return step11d_profile_kwargs(
        cooldown_bars=cooldown_bars,
        session_min_age_bars=3,
        session_min_history_max_distance_atr=0.35,
        session_min_penetration_atr=0.30,
    )


def step11e_profile_kwargs(
    *,
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
    standard_min_pre_breach_distance_atr: float,
    standard_exceptional_penetration_atr: float,
    standard_exceptional_rejection_component: float,
    tradeable_min_rejection_component: float,
    tradeable_max_active_sources: int = DEFAULT_TRADEABLE_MAX_ACTIVE_SOURCES,
) -> dict[str, object]:
    """Step 11E builds on Step 11D and separates micro from standard sweeps."""

    kwargs = step11d_default_kwargs(cooldown_bars=cooldown_bars)
    kwargs["standard_min_pre_breach_distance_atr"] = float(
        standard_min_pre_breach_distance_atr
    )
    kwargs["standard_exceptional_penetration_atr"] = float(
        standard_exceptional_penetration_atr
    )
    kwargs["standard_exceptional_rejection_component"] = float(
        standard_exceptional_rejection_component
    )
    kwargs["tradeable_family_min_penetration_atr"] = (
        DEFAULT_TRADEABLE_FAMILY_MIN_PENETRATION_ATR.copy()
    )
    kwargs["tradeable_min_rejection_component"] = float(
        tradeable_min_rejection_component
    )
    kwargs["tradeable_max_active_sources"] = int(tradeable_max_active_sources)
    kwargs["tradeable_require_volume_or_displacement"] = (
        DEFAULT_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT
    )
    return kwargs


def step11e_default_kwargs(
    *, cooldown_bars: int = DEFAULT_COOLDOWN_BARS
) -> dict[str, object]:
    """Chosen Step 11E production profile."""

    return step11e_profile_kwargs(
        cooldown_bars=cooldown_bars,
        standard_min_pre_breach_distance_atr=DEFAULT_STANDARD_MIN_PRE_BREACH_DISTANCE_ATR,
        standard_exceptional_penetration_atr=DEFAULT_STANDARD_EXCEPTIONAL_PENETRATION_ATR,
        standard_exceptional_rejection_component=DEFAULT_STANDARD_EXCEPTIONAL_REJECTION_COMPONENT,
        tradeable_min_rejection_component=DEFAULT_TRADEABLE_MIN_REJECTION_COMPONENT,
    )


__all__ = [
    "DEFAULT_CONFIRMATION_WINDOW_BARS",
    "DEFAULT_FOLLOW_THROUGH_WINDOW_BARS",
    "DEFAULT_COOLDOWN_BARS",
    "DEFAULT_FAMILY_MIN_STRENGTH",
    "DEFAULT_FAMILY_MIN_AGE_BARS",
    "DEFAULT_FAMILY_MAX_AGE_BARS",
    "DEFAULT_FAMILY_MIN_PENETRATION_ATR",
    "DEFAULT_FAMILY_MIN_HISTORY_MAX_DISTANCE_ATR",
    "DEFAULT_STANDARD_MIN_PRE_BREACH_DISTANCE_ATR",
    "DEFAULT_STANDARD_EXCEPTIONAL_PENETRATION_ATR",
    "DEFAULT_STANDARD_EXCEPTIONAL_REJECTION_COMPONENT",
    "DEFAULT_TRADEABLE_MIN_QUALITY_BY_CLASS",
    "DEFAULT_TRADEABLE_MIN_PENETRATION_ATR_BY_CLASS",
    "DEFAULT_TRADEABLE_FAMILY_MIN_PENETRATION_ATR",
    "DEFAULT_TRADEABLE_FAMILY_MIN_STRENGTH",
    "DEFAULT_TRADEABLE_MIN_REJECTION_COMPONENT",
    "DEFAULT_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT",
    "DEFAULT_TRADEABLE_MAX_ACTIVE_SOURCES",
    "DEFAULT_SESSION_TRADEABLE_MIN_REJECTION_COMPONENT",
    "DEFAULT_SESSION_TRADEABLE_REQUIRE_VOLUME_OR_DISPLACEMENT",
    "SWEEP_CLASS_NO_INTERACTION",
    "SWEEP_CLASS_PROBED",
    "SWEEP_CLASS_UNRESOLVED",
    "SWEEP_CLASS_SAME_BAR",
    "SWEEP_CLASS_DELAYED_REJECTION",
    "SWEEP_CLASS_ACCEPTED_BREAKOUT",
    "SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM",
    "SWEEP_CLASS_SWEEP_THEN_BREAK",
    "SWEEP_INTERACTION_UNTOUCHED",
    "SWEEP_INTERACTION_PROBED",
    "SWEEP_INTERACTION_PARTIALLY_SWEPT",
    "SWEEP_INTERACTION_FULLY_SWEPT",
    "SWEEP_INTERACTION_ACCEPTED_BEYOND",
    "SWEEP_INTERACTION_RECLAIMED",
    "SWEEP_SELECTIVITY_MICRO_INTERACTION",
    "SWEEP_SELECTIVITY_STANDARD_LIQUIDITY",
    "SWEEP_SELECTIVITY_DISPLACEMENT_CONFIRMED",
    "SWEEP_SELECTIVITY_TRADEABLE_CANDIDATE",
    "SWEEP_QUALITY_WEIGHTS",
    "FINAL_SWEEPS_LIVE_COLUMNS",
    "FINAL_SWEEPS_BREACH_COLUMNS",
    "FINAL_SWEEPS_QUALITY_COLUMNS",
    "FINAL_SWEEPS_SELECTIVITY_COLUMNS",
    "FINAL_SWEEPS_FOLLOWTHROUGH_COLUMNS",
    "FINAL_SWEEPS_INTERACTION_COLUMNS",
    "FINAL_SWEEPS_COLUMNS",
    "step11b_baseline_kwargs",
    "step11c_default_kwargs",
    "step11d_profile_kwargs",
    "step11d_default_kwargs",
    "step11e_profile_kwargs",
    "step11e_default_kwargs",
    "add_final_sweeps",
]
