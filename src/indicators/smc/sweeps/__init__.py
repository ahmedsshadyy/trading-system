"""Final sweeps v2 — Steps 9, 10, 11 of the SweepsPlan.

This package implements the production sweep stack:

* :mod:`mtf_policy`        – Step 9, the same-timeframe MTF freeze.
* :mod:`unified_sources`   – Step 10, the unified liquidity-source framework.
* :mod:`final_sweeps`      – Step 11, the final sweeps detector that consumes
                             only the unified source schema.

Production sweep source families (frozen for v1):

    1. confirmed swings (swing_high / swing_low)
    2. equal highs/lows  (equal_high / equal_low)
    3. support / resistance
    4. session highs/lows  (Asia / London / NY)
    5. previous day high/low
    6. previous week high/low

Explicitly excluded for v1:

* range boundaries (kept in code, marked deprecated for sweeps)
* FVG edges (deferred to post-v1 enrichment)
* OB edges (deferred to post-v1 enrichment)

All stages are causal and same-timeframe-only. HTF projection is gated by the
:data:`mtf_policy.HTF_LIQUIDITY_PROJECTION_ENABLED` flag and is disabled in v1.
"""

from __future__ import annotations

from src.indicators.smc.sweeps.final_sweeps import (
    FINAL_SWEEPS_COLUMNS,
    SWEEP_CLASS_ACCEPTED_BREAKOUT,
    SWEEP_CLASS_DELAYED_REJECTION,
    SWEEP_CLASS_FAILED_BREAKOUT_RECLAIM,
    SWEEP_CLASS_NO_INTERACTION,
    SWEEP_CLASS_PROBED,
    SWEEP_CLASS_SAME_BAR,
    SWEEP_CLASS_SWEEP_THEN_BREAK,
    SWEEP_CLASS_UNRESOLVED,
    SWEEP_INTERACTION_ACCEPTED_BEYOND,
    SWEEP_INTERACTION_FULLY_SWEPT,
    SWEEP_INTERACTION_PARTIALLY_SWEPT,
    SWEEP_INTERACTION_PROBED,
    SWEEP_INTERACTION_RECLAIMED,
    SWEEP_INTERACTION_UNTOUCHED,
    add_final_sweeps,
)
from src.indicators.smc.sweeps.mtf_policy import (
    HTF_LIQUIDITY_PROJECTION_ENABLED,
    SWEEP_MTF_POLICY,
    assert_same_timeframe_sources,
    mtf_policy_summary,
)
from src.indicators.smc.sweeps.unified_sources import (
    LIQ_LADDER_DEPTH,
    LIQ_SOURCE_FAMILIES,
    LIQ_STATE_ACTIVE,
    LIQ_STATE_BORN,
    LIQ_STATE_CONSUMED_SWEPT,
    LIQ_STATE_INVALIDATED,
    LIQ_STATE_RETIRED,
    LIQ_STATE_UNAVAILABLE,
    LIQ_STATE_WEAKENED,
    UNIFIED_SOURCE_COLUMNS,
    add_unified_liquidity_sources,
)

__all__ = [
    "HTF_LIQUIDITY_PROJECTION_ENABLED",
    "SWEEP_MTF_POLICY",
    "assert_same_timeframe_sources",
    "mtf_policy_summary",
    "LIQ_LADDER_DEPTH",
    "LIQ_SOURCE_FAMILIES",
    "LIQ_STATE_UNAVAILABLE",
    "LIQ_STATE_BORN",
    "LIQ_STATE_ACTIVE",
    "LIQ_STATE_WEAKENED",
    "LIQ_STATE_CONSUMED_SWEPT",
    "LIQ_STATE_INVALIDATED",
    "LIQ_STATE_RETIRED",
    "UNIFIED_SOURCE_COLUMNS",
    "add_unified_liquidity_sources",
    "FINAL_SWEEPS_COLUMNS",
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
    "add_final_sweeps",
]
