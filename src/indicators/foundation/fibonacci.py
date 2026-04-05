"""
foundation/fibonacci.py

Canonical causal Fibonacci retracement and extension level engine.

Causality Doctrine
------------------
A Fibonacci structure is only knowable at bar ``t`` when *both* anchor swing
points are confirmed live-known at or before bar ``t``.

- **Bull structure**: impulse = last confirmed swing LOW (anchor A) → current
  confirmed swing HIGH (anchor B). Retracement levels project *downward* from
  anchor_high toward anchor_low. Known at the close of the swing-high confirm bar.
- **Bear structure**: last confirmed swing HIGH (anchor A) → current confirmed
  swing LOW (anchor B). Levels project *upward* from anchor_low. Known at the
  close of the swing-low confirm bar.

``origin_idx``   = confirm bar of anchor A (the prior swing)
``detect_idx``   = confirm bar of anchor B (the triggering swing, current bar)
``detect_delay`` = ``detect_idx - origin_idx``

No bar at or before ``detect_idx`` is read in the lifecycle; the first
lifecycle evaluation is at ``detect_idx + 1``.

Hard Expiry
-----------
Every structure expires ``max_age_bars`` bars after activation regardless of
any other condition. This prevents indefinite accretion of stale structures.

Level Definitions
-----------------
Fib ratios: 0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618

For a bull structure (anchor_low = A, anchor_high = B):
  level_price(r) = anchor_high - r * (anchor_high - anchor_low)
  → ratio 0.000 = anchor_high (top of impulse)
  → ratio 0.618 = key retracement
  → ratio 1.000 = anchor_low (bottom of impulse)
  → ratio 1.272, 1.618 = extensions below anchor_low

For a bear structure (anchor_high = A, anchor_low = B):
  level_price(r) = anchor_low + r * (anchor_high - anchor_low)
  → ratio 0.000 = anchor_low (bottom of impulse)
  → ratio 0.618 = key retracement
  → ratio 1.000 = anchor_high (top of impulse)
  → ratio 1.272, 1.618 = extensions above anchor_high

Lifecycle State Machine
-----------------------
Per-structure states:
- 0 = inactive (pre-detect)
- 1 = active
- 2 = expired (terminal)
- 3 = invalidated (terminal)

Transitions:
- inactive → active at detect_idx
- active → expired when age_bars >= max_age_bars
- active → invalidated when close breaks beyond the 1.0 level with buffer:
  - bull: close < anchor_low - invalidation_buffer_atr * ATR[i]
  - bear: close > anchor_high + invalidation_buffer_atr * ATR[i]
- Terminal states are sticky — never leave.

Selection
---------
At each bar, the "selected" active structure for each direction is the one
with the highest quality_score among all active structures. This populates
dense projection columns.

Public API
----------
- ``add_fibonacci_levels(df, ...)`` — main entry point
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.event_ids import SequentialEventIdAllocator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIB_RATIOS = (0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618)
FIB_RATIO_LABELS = (
    "0000",
    "0236",
    "0382",
    "0500",
    "0618",
    "0786",
    "1000",
    "1272",
    "1618",
)

FIB_STATE_INACTIVE = 0
FIB_STATE_ACTIVE = 1
FIB_STATE_EXPIRED = 2
FIB_STATE_INVALIDATED = 3

# Per-level touch outcome states (sticky: first touch determines final state)
FIB_LEVEL_STATE_UNTOUCHED = 1
FIB_LEVEL_STATE_HELD = 2  # price reached, closed on the correct side
FIB_LEVEL_STATE_BROKEN = 3  # price reached, closed through the level

FIB_DIR_BULL = 1
FIB_DIR_BEAR = -1

# Hard expiry default (bars after activation).
# H4: 150 bars ≈ 25 trading days.  Caller may override per timeframe.
DEFAULT_MAX_AGE_BARS = 150
# Maximum simultaneous active structures per direction.
# Default is 1: each new confirmed swing supersedes the previous structure
# ("latest always wins"). Raise only when studying multi-structure overlap.
DEFAULT_MAX_CONCURRENT_PER_DIR = 1
# Minimum impulse size in ATR units.  Structures with a smaller range are
# rejected as noise — they don't represent meaningful price structure.
DEFAULT_MIN_RANGE_ATR = 1.5
# ATR buffer beyond 1.0 level before structure is invalidated
DEFAULT_INVALIDATION_BUFFER_ATR = 0.10
# ATR multiple for touch / zone detection
DEFAULT_TOUCH_TOL_ATR = 0.15
# ATR length
DEFAULT_ATR_LENGTH = 14
# Quality score weights
_W_RANGE = 0.50
_W_ANCHOR_A = 0.20
_W_ANCHOR_B = 0.20
_W_DELAY = 0.10
_RANGE_CAP_ATR = 5.0

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_RATIO_LBLS = FIB_RATIO_LABELS  # short alias

FIB_EVENT_COLUMNS: list[str] = []
for _d in ("bull", "bear"):
    FIB_EVENT_COLUMNS += [
        f"fib_{_d}_detect_flag",
        f"fib_{_d}_event_id",
        f"fib_{_d}_detect_idx",
        f"fib_{_d}_detect_ts",
        f"fib_{_d}_origin_idx",
        f"fib_{_d}_origin_ts",
        f"fib_{_d}_anchor_a_price",
        f"fib_{_d}_anchor_b_price",
        f"fib_{_d}_range",
        f"fib_{_d}_range_atr",
        f"fib_{_d}_detect_delay",
        f"fib_{_d}_quality_score_on_detect",
    ] + [f"fib_{_d}_l{lbl}" for lbl in _RATIO_LBLS]

FIB_DENSE_COLUMNS: list[str] = []
for _d in ("bull", "bear"):
    FIB_DENSE_COLUMNS += [
        f"fib_{_d}_active",
        f"fib_active_{_d}_count",
        f"fib_nearest_{_d}_level",
        f"fib_nearest_{_d}_ratio",
        f"fib_nearest_{_d}_distance_atr",
        f"fib_nearest_{_d}_structure_id",
        f"fib_inside_{_d}_zone",
        f"fib_golden_{_d}_distance_atr",
        f"fib_{_d}_anchor_low",
        f"fib_{_d}_anchor_high",
        f"fib_{_d}_structure_age_bars",
        f"fib_{_d}_quality_score",
    ] + [f"fib_{_d}_l{lbl}_state" for lbl in _RATIO_LBLS]

# Final per-level states stamped at detect_idx after the full forward pass.
# These are RESEARCH-ONLY (forward-looking from detect bar) — do NOT use live.
FIB_LEVEL_FINAL_STATE_COLUMNS: list[str] = []
for _d in ("bull", "bear"):
    FIB_LEVEL_FINAL_STATE_COLUMNS += [
        f"fib_{_d}_l{lbl}_state_final" for lbl in _RATIO_LBLS
    ]

FIB_CANONICAL_COLUMNS: list[str] = FIB_EVENT_COLUMNS + FIB_DENSE_COLUMNS
FIB_LIVE_SAFE_COLUMNS: list[str] = FIB_CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class FibStructure:
    """One canonical Fibonacci structure with lifecycle state."""

    structure_id: int
    direction: int  # FIB_DIR_BULL or FIB_DIR_BEAR
    anchor_low: float
    anchor_high: float
    anchor_a_price: float  # origin swing price
    anchor_b_price: float  # detect swing price
    origin_idx: int
    detect_idx: int
    origin_ts: Any
    detect_ts: Any
    detect_delay: int
    quality_score: float
    range_atr: float
    level_prices: dict[str, float]  # label → price

    # mutable lifecycle
    state: int = field(default=FIB_STATE_INACTIVE)
    age_bars: int = field(default=0)
    # per-level touch outcome (1=UNTOUCHED, 2=HELD, 3=BROKEN) — sticky
    level_states: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_level_prices(
    anchor_low: float,
    anchor_high: float,
    direction: int,
) -> dict[str, float]:
    """Compute fib level prices for a structure."""
    rng = anchor_high - anchor_low
    out: dict[str, float] = {}
    for ratio, lbl in zip(FIB_RATIOS, FIB_RATIO_LABELS):
        if direction == FIB_DIR_BULL:
            # levels project downward from anchor_high
            out[lbl] = anchor_high - ratio * rng
        else:
            # levels project upward from anchor_low
            out[lbl] = anchor_low + ratio * rng
    return out


def _compute_quality_score(
    range_atr: float,
    anchor_a_strength: float,
    anchor_b_strength: float,
    detect_delay: int,
) -> float:
    s_range = float(np.clip(range_atr / _RANGE_CAP_ATR, 0.0, 1.0))
    s_a = (
        float(np.clip(anchor_a_strength, 0.0, 1.0))
        if np.isfinite(anchor_a_strength)
        else 0.5
    )
    s_b = (
        float(np.clip(anchor_b_strength, 0.0, 1.0))
        if np.isfinite(anchor_b_strength)
        else 0.5
    )
    s_delay = 1.0 / (1.0 + max(detect_delay, 0) / 20.0)
    raw = (
        _W_RANGE * s_range + _W_ANCHOR_A * s_a + _W_ANCHOR_B * s_b + _W_DELAY * s_delay
    )
    return float(np.clip(raw, 0.0, 1.0))


def _validate_parameters(
    *,
    atr_length: int,
    max_age_bars: int,
    max_concurrent_per_dir: int,
    min_range_atr: float,
    touch_tol_atr: float,
    invalidation_buffer_atr: float,
) -> None:
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if max_age_bars <= 0:
        raise ValueError("max_age_bars must be > 0")
    if max_concurrent_per_dir < 1:
        raise ValueError("max_concurrent_per_dir must be >= 1")
    if min_range_atr < 0:
        raise ValueError("min_range_atr must be >= 0")
    if touch_tol_atr < 0:
        raise ValueError("touch_tol_atr must be >= 0")
    if invalidation_buffer_atr < 0:
        raise ValueError("invalidation_buffer_atr must be >= 0")


def _get_timestamps(df: pd.DataFrame) -> list:
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return ts.tolist()
    return [None] * len(df)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def add_fibonacci_levels(
    df: pd.DataFrame,
    *,
    atr_length: int = DEFAULT_ATR_LENGTH,
    max_age_bars: int = DEFAULT_MAX_AGE_BARS,
    max_concurrent_per_dir: int = DEFAULT_MAX_CONCURRENT_PER_DIR,
    min_range_atr: float = DEFAULT_MIN_RANGE_ATR,
    touch_tol_atr: float = DEFAULT_TOUCH_TOL_ATR,
    invalidation_buffer_atr: float = DEFAULT_INVALIDATION_BUFFER_ATR,
) -> pd.DataFrame:
    """Append canonical Fibonacci level columns to ``df``.

    Parameters
    ----------
    df:
        OHLC DataFrame. Must contain confirmed swing columns:
        ``swing_high_confirm_flag``, ``swing_high_confirm_price``,
        ``swing_high_confirm_origin_idx``, ``swing_low_confirm_flag``,
        ``swing_low_confirm_price``, ``swing_low_confirm_origin_idx``.
        Optional: ``swing_high_strength``, ``swing_low_strength`` for quality.
    atr_length:
        ATR period for normalization.
    max_age_bars:
        Hard expiry: structures expire this many bars after detection.
        H4 default = 150 bars ≈ 25 trading days.
    max_concurrent_per_dir:
        Maximum simultaneous active structures per direction.  When a new
        structure would exceed this cap the active structure with the lowest
        quality_score is retired before the new one is admitted.  Prevents
        unbounded chart clutter and keeps projection columns meaningful.
    min_range_atr:
        Minimum impulse size (anchor_high - anchor_low) in ATR units.
        Structures below this threshold are rejected as noise.
    touch_tol_atr:
        ATR multiple for fib zone touch detection.
    invalidation_buffer_atr:
        ATR buffer beyond 1.0 level before invalidation triggers.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with all FIB_CANONICAL_COLUMNS appended.
    """
    _validate_parameters(
        atr_length=atr_length,
        max_age_bars=max_age_bars,
        max_concurrent_per_dir=max_concurrent_per_dir,
        min_range_atr=min_range_atr,
        touch_tol_atr=touch_tol_atr,
        invalidation_buffer_atr=invalidation_buffer_atr,
    )

    required_swing_cols = {
        "swing_high_confirm_flag",
        "swing_high_confirm_price",
        "swing_low_confirm_flag",
        "swing_low_confirm_price",
    }
    missing = required_swing_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"add_fibonacci_levels: missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)
    timestamps = _get_timestamps(out)

    # --- extract arrays ---
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    atr = get_atr_array(out, length=atr_length)

    sh_flag = (
        pd.to_numeric(out["swing_high_confirm_flag"], errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    sl_flag = (
        pd.to_numeric(out["swing_low_confirm_flag"], errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    sh_price = out["swing_high_confirm_price"].to_numpy(dtype=float)
    sl_price = out["swing_low_confirm_price"].to_numpy(dtype=float)

    has_sh_origin = "swing_high_confirm_origin_idx" in out.columns
    has_sl_origin = "swing_low_confirm_origin_idx" in out.columns
    sh_origin_arr = (
        out["swing_high_confirm_origin_idx"].to_numpy(dtype=float)
        if has_sh_origin
        else np.full(n, np.nan)
    )
    sl_origin_arr = (
        out["swing_low_confirm_origin_idx"].to_numpy(dtype=float)
        if has_sl_origin
        else np.full(n, np.nan)
    )

    has_sh_strength = "swing_high_strength" in out.columns
    has_sl_strength = "swing_low_strength" in out.columns
    sh_strength_arr = (
        out["swing_high_strength"].to_numpy(dtype=float)
        if has_sh_strength
        else np.full(n, np.nan)
    )
    sl_strength_arr = (
        out["swing_low_strength"].to_numpy(dtype=float)
        if has_sl_strength
        else np.full(n, np.nan)
    )

    # --- allocate output arrays ---
    # Event columns (sparse — NaN default for float/object, 0 for int8)
    bull_detect_flag = np.zeros(n, dtype=np.int8)
    bull_event_id = np.full(n, np.nan)
    bull_detect_idx_arr = np.full(n, np.nan)
    bull_detect_ts_arr = np.full(n, None, dtype=object)
    bull_origin_idx_arr = np.full(n, np.nan)
    bull_origin_ts_arr = np.full(n, None, dtype=object)
    bull_anchor_a_arr = np.full(n, np.nan)
    bull_anchor_b_arr = np.full(n, np.nan)
    bull_range_arr = np.full(n, np.nan)
    bull_range_atr_arr = np.full(n, np.nan)
    bull_delay_arr = np.full(n, np.nan)
    bull_quality_detect_arr = np.full(n, np.nan)
    bull_level_arrs: dict[str, np.ndarray] = {
        lbl: np.full(n, np.nan) for lbl in FIB_RATIO_LABELS
    }

    bear_detect_flag = np.zeros(n, dtype=np.int8)
    bear_event_id = np.full(n, np.nan)
    bear_detect_idx_arr = np.full(n, np.nan)
    bear_detect_ts_arr = np.full(n, None, dtype=object)
    bear_origin_idx_arr = np.full(n, np.nan)
    bear_origin_ts_arr = np.full(n, None, dtype=object)
    bear_anchor_a_arr = np.full(n, np.nan)
    bear_anchor_b_arr = np.full(n, np.nan)
    bear_range_arr = np.full(n, np.nan)
    bear_range_atr_arr = np.full(n, np.nan)
    bear_delay_arr = np.full(n, np.nan)
    bear_quality_detect_arr = np.full(n, np.nan)
    bear_level_arrs: dict[str, np.ndarray] = {
        lbl: np.full(n, np.nan) for lbl in FIB_RATIO_LABELS
    }

    # Dense columns (all bars)
    bull_active_arr = np.zeros(n, dtype=np.int8)
    bear_active_arr = np.zeros(n, dtype=np.int8)
    bull_count_arr = np.zeros(n, dtype=np.int32)
    bear_count_arr = np.zeros(n, dtype=np.int32)
    near_bull_level = np.full(n, np.nan)
    near_bull_ratio = np.full(n, np.nan)
    near_bull_dist_atr = np.full(n, np.nan)
    near_bull_sid = np.full(n, np.nan)
    near_bear_level = np.full(n, np.nan)
    near_bear_ratio = np.full(n, np.nan)
    near_bear_dist_atr = np.full(n, np.nan)
    near_bear_sid = np.full(n, np.nan)
    inside_bull_zone = np.zeros(n, dtype=np.int8)
    inside_bear_zone = np.zeros(n, dtype=np.int8)
    golden_bull_dist = np.full(n, np.nan)
    golden_bear_dist = np.full(n, np.nan)
    bull_anchor_low_arr = np.full(n, np.nan)
    bull_anchor_high_arr = np.full(n, np.nan)
    bear_anchor_low_arr = np.full(n, np.nan)
    bear_anchor_high_arr = np.full(n, np.nan)
    bull_age_arr = np.full(n, np.nan)
    bear_age_arr = np.full(n, np.nan)
    bull_quality_arr = np.full(n, np.nan)
    bear_quality_arr = np.full(n, np.nan)
    # Per-level state arrays: 0=no active structure, 1=UNTOUCHED, 2=HELD, 3=BROKEN
    bull_level_state_arrs: dict[str, np.ndarray] = {
        lbl: np.zeros(n, dtype=np.int8) for lbl in FIB_RATIO_LABELS
    }
    bear_level_state_arrs: dict[str, np.ndarray] = {
        lbl: np.zeros(n, dtype=np.int8) for lbl in FIB_RATIO_LABELS
    }

    # --- tracking state ---
    # Use registry + active_set so the lifecycle loop only iterates ACTIVE
    # structures (O(active) ≤ max_age_bars) rather than the ever-growing list.
    allocator = SequentialEventIdAllocator(start=1)
    bull_registry: dict[int, FibStructure] = {}
    active_bull_ids: set[int] = set()
    bear_registry: dict[int, FibStructure] = {}
    active_bear_ids: set[int] = set()

    # Track last confirmed swing in each direction for anchor detection
    last_sh_confirm_idx: int = -1
    last_sh_confirm_price: float = np.nan
    last_sh_strength: float = np.nan

    last_sl_confirm_idx: int = -1
    last_sl_confirm_price: float = np.nan
    last_sl_strength: float = np.nan

    # ---------------------------------------------------------------------------
    # Main forward pass
    # ---------------------------------------------------------------------------
    for i in range(n):
        atr_i = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 1.0
        cl = close[i]
        hi = high[i]
        lo = low[i]

        # --- Step 1: detect new fib structures on confirmed swings ---

        # Swing high confirm → creates a BULL fib structure
        # (retracement of the bullish leg: swing low → swing high)
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            sh_p = float(sh_price[i])
            # origin idx: use swing_high_confirm_origin_idx if available and valid
            if np.isfinite(sh_origin_arr[i]):
                sh_orig_idx = int(sh_origin_arr[i])
            else:
                sh_orig_idx = i
            sh_orig_idx = max(0, min(sh_orig_idx, n - 1))

            # Anchor A = last confirmed swing low before this bar
            if (
                last_sl_confirm_idx >= 0
                and last_sl_confirm_idx < i
                and np.isfinite(last_sl_confirm_price)
            ):
                anchor_a_price = last_sl_confirm_price
                anchor_a_strength = last_sl_strength
                anchor_b_price = sh_p
                anchor_b_strength = sh_strength_arr[i]

                anchor_low = min(anchor_a_price, anchor_b_price)
                anchor_high = max(anchor_a_price, anchor_b_price)
                rng = anchor_high - anchor_low

                if rng > 0 and np.isfinite(anchor_a_price):
                    rng_atr = rng / atr_i
                    # Reject impulses too small to be meaningful
                    if rng_atr < min_range_atr:
                        pass  # skip — handled by outer if block continuing
                    else:
                        delay = i - last_sl_confirm_idx
                        qscore = _compute_quality_score(
                            rng_atr, anchor_a_strength, anchor_b_strength, delay
                        )
                        level_prices = _compute_level_prices(
                            anchor_low, anchor_high, FIB_DIR_BULL
                        )

                        # Enforce concurrent cap: latest always wins — retire all
                        # existing structures before admitting the new one.
                        while len(active_bull_ids) >= max_concurrent_per_dir:
                            retire_id = next(iter(active_bull_ids))
                            bull_registry[retire_id].state = FIB_STATE_EXPIRED
                            active_bull_ids.discard(retire_id)

                        if len(active_bull_ids) < max_concurrent_per_dir:
                            struct = FibStructure(
                                structure_id=allocator.allocate(),
                                direction=FIB_DIR_BULL,
                                anchor_low=anchor_low,
                                anchor_high=anchor_high,
                                anchor_a_price=anchor_a_price,
                                anchor_b_price=anchor_b_price,
                                origin_idx=last_sl_confirm_idx,
                                detect_idx=i,
                                origin_ts=timestamps[last_sl_confirm_idx],
                                detect_ts=timestamps[i],
                                detect_delay=delay,
                                quality_score=qscore,
                                range_atr=rng_atr,
                                level_prices=level_prices,
                                state=FIB_STATE_ACTIVE,
                                age_bars=0,
                                level_states={
                                    lbl: FIB_LEVEL_STATE_UNTOUCHED
                                    for lbl in FIB_RATIO_LABELS
                                },
                            )
                            bull_registry[struct.structure_id] = struct
                            active_bull_ids.add(struct.structure_id)

                            # Stamp sparse event columns (only for admitted structures)
                            bull_detect_flag[i] = 1
                            bull_event_id[i] = float(struct.structure_id)
                            bull_detect_idx_arr[i] = float(i)
                            bull_detect_ts_arr[i] = timestamps[i]
                            bull_origin_idx_arr[i] = float(last_sl_confirm_idx)
                            bull_origin_ts_arr[i] = timestamps[last_sl_confirm_idx]
                            bull_anchor_a_arr[i] = anchor_a_price
                            bull_anchor_b_arr[i] = anchor_b_price
                            bull_range_arr[i] = rng
                            bull_range_atr_arr[i] = rng_atr
                            bull_delay_arr[i] = float(delay)
                            bull_quality_detect_arr[i] = qscore
                            for lbl, lp in level_prices.items():
                                bull_level_arrs[lbl][i] = lp

        # Swing low confirm → creates a BEAR fib structure
        # (retracement of the bearish leg: swing high → swing low)
        if sl_flag[i] == 1 and np.isfinite(sl_price[i]):
            sl_p = float(sl_price[i])
            if np.isfinite(sl_origin_arr[i]):
                sl_orig_idx = int(sl_origin_arr[i])
            else:
                sl_orig_idx = i
            sl_orig_idx = max(0, min(sl_orig_idx, n - 1))

            # Anchor A = last confirmed swing high before this bar
            if (
                last_sh_confirm_idx >= 0
                and last_sh_confirm_idx < i
                and np.isfinite(last_sh_confirm_price)
            ):
                anchor_a_price = last_sh_confirm_price
                anchor_a_strength = last_sh_strength
                anchor_b_price = sl_p
                anchor_b_strength = sl_strength_arr[i]

                anchor_low = min(anchor_a_price, anchor_b_price)
                anchor_high = max(anchor_a_price, anchor_b_price)
                rng = anchor_high - anchor_low

                if rng > 0 and np.isfinite(anchor_a_price):
                    rng_atr = rng / atr_i
                    if rng_atr >= min_range_atr:
                        delay = i - last_sh_confirm_idx
                        qscore = _compute_quality_score(
                            rng_atr, anchor_a_strength, anchor_b_strength, delay
                        )
                        level_prices = _compute_level_prices(
                            anchor_low, anchor_high, FIB_DIR_BEAR
                        )

                        # Enforce concurrent cap: latest always wins — retire all
                        # existing structures before admitting the new one.
                        while len(active_bear_ids) >= max_concurrent_per_dir:
                            retire_id = next(iter(active_bear_ids))
                            bear_registry[retire_id].state = FIB_STATE_EXPIRED
                            active_bear_ids.discard(retire_id)

                        if len(active_bear_ids) < max_concurrent_per_dir:
                            struct = FibStructure(
                                structure_id=allocator.allocate(),
                                direction=FIB_DIR_BEAR,
                                anchor_low=anchor_low,
                                anchor_high=anchor_high,
                                anchor_a_price=anchor_a_price,
                                anchor_b_price=anchor_b_price,
                                origin_idx=last_sh_confirm_idx,
                                detect_idx=i,
                                origin_ts=timestamps[last_sh_confirm_idx],
                                detect_ts=timestamps[i],
                                detect_delay=delay,
                                quality_score=qscore,
                                range_atr=rng_atr,
                                level_prices=level_prices,
                                state=FIB_STATE_ACTIVE,
                                age_bars=0,
                                level_states={
                                    lbl: FIB_LEVEL_STATE_UNTOUCHED
                                    for lbl in FIB_RATIO_LABELS
                                },
                            )
                            bear_registry[struct.structure_id] = struct
                            active_bear_ids.add(struct.structure_id)

                            bear_detect_flag[i] = 1
                            bear_event_id[i] = float(struct.structure_id)
                            bear_detect_idx_arr[i] = float(i)
                            bear_detect_ts_arr[i] = timestamps[i]
                            bear_origin_idx_arr[i] = float(last_sh_confirm_idx)
                            bear_origin_ts_arr[i] = timestamps[last_sh_confirm_idx]
                            bear_anchor_a_arr[i] = anchor_a_price
                            bear_anchor_b_arr[i] = anchor_b_price
                            bear_range_arr[i] = rng
                            bear_range_atr_arr[i] = rng_atr
                            bear_delay_arr[i] = float(delay)
                            bear_quality_detect_arr[i] = qscore
                            for lbl, lp in level_prices.items():
                                bear_level_arrs[lbl][i] = lp

        # Update last-seen confirmed swings AFTER creating structures for this bar
        # (so same-bar swing high and swing low don't reference each other as anchors)
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            last_sh_confirm_idx = i
            last_sh_confirm_price = float(sh_price[i])
            (int(sh_origin_arr[i]) if np.isfinite(sh_origin_arr[i]) else i)
            last_sh_strength = (
                float(sh_strength_arr[i]) if np.isfinite(sh_strength_arr[i]) else np.nan
            )

        if sl_flag[i] == 1 and np.isfinite(sl_price[i]):
            last_sl_confirm_idx = i
            last_sl_confirm_price = float(sl_price[i])
            (int(sl_origin_arr[i]) if np.isfinite(sl_origin_arr[i]) else i)
            last_sl_strength = (
                float(sl_strength_arr[i]) if np.isfinite(sl_strength_arr[i]) else np.nan
            )

        # --- Step 2: lifecycle update (active set only — O(active) not O(total)) ---
        active_bulls: list[FibStructure] = []
        active_bears: list[FibStructure] = []
        _bull_terminate: list[int] = []
        _bear_terminate: list[int] = []

        tol_i = touch_tol_atr * atr_i

        for sid in active_bull_ids:
            struct = bull_registry[sid]
            if struct.detect_idx >= i:
                active_bulls.append(struct)
                continue
            struct.age_bars += 1
            if struct.age_bars >= max_age_bars:
                struct.state = FIB_STATE_EXPIRED
                _bull_terminate.append(sid)
                continue
            if cl < struct.anchor_low - invalidation_buffer_atr * atr_i:
                struct.state = FIB_STATE_INVALIDATED
                _bull_terminate.append(sid)
                continue
            # Per-level touch state (sticky — first touch wins)
            for lbl, lp in struct.level_prices.items():
                if (
                    struct.level_states.get(lbl, FIB_LEVEL_STATE_UNTOUCHED)
                    == FIB_LEVEL_STATE_UNTOUCHED
                ):
                    if lo <= lp + tol_i and hi >= lp - tol_i:
                        # Level touched this bar — close determines held/broken
                        struct.level_states[lbl] = (
                            FIB_LEVEL_STATE_HELD if cl >= lp else FIB_LEVEL_STATE_BROKEN
                        )
            active_bulls.append(struct)
        for sid in _bull_terminate:
            active_bull_ids.discard(sid)

        for sid in active_bear_ids:
            struct = bear_registry[sid]
            if struct.detect_idx >= i:
                active_bears.append(struct)
                continue
            struct.age_bars += 1
            if struct.age_bars >= max_age_bars:
                struct.state = FIB_STATE_EXPIRED
                _bear_terminate.append(sid)
                continue
            if cl > struct.anchor_high + invalidation_buffer_atr * atr_i:
                struct.state = FIB_STATE_INVALIDATED
                _bear_terminate.append(sid)
                continue
            # Per-level touch state (sticky — first touch wins)
            for lbl, lp in struct.level_prices.items():
                if (
                    struct.level_states.get(lbl, FIB_LEVEL_STATE_UNTOUCHED)
                    == FIB_LEVEL_STATE_UNTOUCHED
                ):
                    if lo <= lp + tol_i and hi >= lp - tol_i:
                        struct.level_states[lbl] = (
                            FIB_LEVEL_STATE_HELD if cl <= lp else FIB_LEVEL_STATE_BROKEN
                        )
            active_bears.append(struct)
        for sid in _bear_terminate:
            active_bear_ids.discard(sid)

        # --- Step 3: project dense columns ---
        bull_count_arr[i] = len(active_bulls)
        bear_count_arr[i] = len(active_bears)
        bull_active_arr[i] = np.int8(1) if active_bulls else np.int8(0)
        bear_active_arr[i] = np.int8(1) if active_bears else np.int8(0)

        # Select best bull structure by quality
        if active_bulls:
            sel_bull = max(active_bulls, key=lambda s: s.quality_score)
            bull_anchor_low_arr[i] = sel_bull.anchor_low
            bull_anchor_high_arr[i] = sel_bull.anchor_high
            bull_age_arr[i] = float(sel_bull.age_bars)
            bull_quality_arr[i] = sel_bull.quality_score

            # Find nearest fib level from selected structure
            best_dist = np.inf
            best_lev_price = np.nan
            best_lev_ratio = np.nan
            in_zone = False
            golden_dist = np.nan

            for ratio, lbl in zip(FIB_RATIOS, FIB_RATIO_LABELS):
                lp = sel_bull.level_prices[lbl]
                dist = abs(cl - lp) / atr_i
                if dist < best_dist:
                    best_dist = dist
                    best_lev_price = lp
                    best_lev_ratio = ratio
                if dist <= touch_tol_atr:
                    in_zone = True
                if lbl == "0618":
                    golden_dist = dist

            near_bull_level[i] = best_lev_price
            near_bull_ratio[i] = best_lev_ratio
            near_bull_dist_atr[i] = best_dist
            near_bull_sid[i] = float(sel_bull.structure_id)
            inside_bull_zone[i] = np.int8(1) if in_zone else np.int8(0)
            golden_bull_dist[i] = golden_dist
            # Stamp per-level state from selected structure
            for lbl in FIB_RATIO_LABELS:
                bull_level_state_arrs[lbl][i] = np.int8(
                    sel_bull.level_states.get(lbl, FIB_LEVEL_STATE_UNTOUCHED)
                )

        if active_bears:
            sel_bear = max(active_bears, key=lambda s: s.quality_score)
            bear_anchor_low_arr[i] = sel_bear.anchor_low
            bear_anchor_high_arr[i] = sel_bear.anchor_high
            bear_age_arr[i] = float(sel_bear.age_bars)
            bear_quality_arr[i] = sel_bear.quality_score

            best_dist = np.inf
            best_lev_price = np.nan
            best_lev_ratio = np.nan
            in_zone = False
            golden_dist = np.nan

            for ratio, lbl in zip(FIB_RATIOS, FIB_RATIO_LABELS):
                lp = sel_bear.level_prices[lbl]
                dist = abs(cl - lp) / atr_i
                if dist < best_dist:
                    best_dist = dist
                    best_lev_price = lp
                    best_lev_ratio = ratio
                if dist <= touch_tol_atr:
                    in_zone = True
                if lbl == "0618":
                    golden_dist = dist

            near_bear_level[i] = best_lev_price
            near_bear_ratio[i] = best_lev_ratio
            near_bear_dist_atr[i] = best_dist
            near_bear_sid[i] = float(sel_bear.structure_id)
            inside_bear_zone[i] = np.int8(1) if in_zone else np.int8(0)
            golden_bear_dist[i] = golden_dist
            # Stamp per-level state from selected structure
            for lbl in FIB_RATIO_LABELS:
                bear_level_state_arrs[lbl][i] = np.int8(
                    sel_bear.level_states.get(lbl, FIB_LEVEL_STATE_UNTOUCHED)
                )

    # ---------------------------------------------------------------------------
    # Write arrays to output DataFrame
    # ---------------------------------------------------------------------------
    out["fib_bull_detect_flag"] = bull_detect_flag
    out["fib_bull_event_id"] = bull_event_id
    out["fib_bull_detect_idx"] = bull_detect_idx_arr
    out["fib_bull_detect_ts"] = bull_detect_ts_arr
    out["fib_bull_origin_idx"] = bull_origin_idx_arr
    out["fib_bull_origin_ts"] = bull_origin_ts_arr
    out["fib_bull_anchor_a_price"] = bull_anchor_a_arr
    out["fib_bull_anchor_b_price"] = bull_anchor_b_arr
    out["fib_bull_range"] = bull_range_arr
    out["fib_bull_range_atr"] = bull_range_atr_arr
    out["fib_bull_detect_delay"] = bull_delay_arr
    out["fib_bull_quality_score_on_detect"] = bull_quality_detect_arr
    for lbl in FIB_RATIO_LABELS:
        out[f"fib_bull_l{lbl}"] = bull_level_arrs[lbl]

    out["fib_bear_detect_flag"] = bear_detect_flag
    out["fib_bear_event_id"] = bear_event_id
    out["fib_bear_detect_idx"] = bear_detect_idx_arr
    out["fib_bear_detect_ts"] = bear_detect_ts_arr
    out["fib_bear_origin_idx"] = bear_origin_idx_arr
    out["fib_bear_origin_ts"] = bear_origin_ts_arr
    out["fib_bear_anchor_a_price"] = bear_anchor_a_arr
    out["fib_bear_anchor_b_price"] = bear_anchor_b_arr
    out["fib_bear_range"] = bear_range_arr
    out["fib_bear_range_atr"] = bear_range_atr_arr
    out["fib_bear_detect_delay"] = bear_delay_arr
    out["fib_bear_quality_score_on_detect"] = bear_quality_detect_arr
    for lbl in FIB_RATIO_LABELS:
        out[f"fib_bear_l{lbl}"] = bear_level_arrs[lbl]

    out["fib_bull_active"] = bull_active_arr
    out["fib_active_bull_count"] = bull_count_arr
    out["fib_nearest_bull_level"] = near_bull_level
    out["fib_nearest_bull_ratio"] = near_bull_ratio
    out["fib_nearest_bull_distance_atr"] = near_bull_dist_atr
    out["fib_nearest_bull_structure_id"] = near_bull_sid
    out["fib_inside_bull_zone"] = inside_bull_zone
    out["fib_golden_bull_distance_atr"] = golden_bull_dist
    out["fib_bull_anchor_low"] = bull_anchor_low_arr
    out["fib_bull_anchor_high"] = bull_anchor_high_arr
    out["fib_bull_structure_age_bars"] = bull_age_arr
    out["fib_bull_quality_score"] = bull_quality_arr

    out["fib_bear_active"] = bear_active_arr
    out["fib_active_bear_count"] = bear_count_arr
    out["fib_nearest_bear_level"] = near_bear_level
    out["fib_nearest_bear_ratio"] = near_bear_ratio
    out["fib_nearest_bear_distance_atr"] = near_bear_dist_atr
    out["fib_nearest_bear_structure_id"] = near_bear_sid
    out["fib_inside_bear_zone"] = inside_bear_zone
    out["fib_golden_bear_distance_atr"] = golden_bear_dist
    out["fib_bear_anchor_low"] = bear_anchor_low_arr
    out["fib_bear_anchor_high"] = bear_anchor_high_arr
    out["fib_bear_structure_age_bars"] = bear_age_arr
    out["fib_bear_quality_score"] = bear_quality_arr

    # Per-level state columns (0=no struct, 1=UNTOUCHED, 2=HELD, 3=BROKEN)
    for lbl in FIB_RATIO_LABELS:
        out[f"fib_bull_l{lbl}_state"] = bull_level_state_arrs[lbl]
        out[f"fib_bear_l{lbl}_state"] = bear_level_state_arrs[lbl]

    return out
