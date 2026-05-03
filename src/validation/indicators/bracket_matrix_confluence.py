"""Step 11U — bracket matrix + confluence edge audit.

Research-only validation layer that combines confirmed sweeps, confirmed
causal swings, and the displacement detector to test whether confluence
between any of these signals improves TP-before-SL win rates under four
ATR-bracket profiles.

No production schema is touched. All forward-looking outcomes are
strictly anchored at the entry bar (sweep_confirm_idx for sweeps, swing
confirm bar for swings) and read only bars in
``confirm_idx + 1 .. confirm_idx + horizon``. Same-bar TP/SL collisions
are explicitly flagged as ``ambiguous`` — never guessed.

Universe definitions:

* ``sweep_all`` — every confirmed sweep
* ``sweep_displacement_confirmed`` — sweeps whose
  ``sweep_is_displacement_confirmed`` flag is set OR a bullish/bearish
  displacement (matching the reversal direction) lands within 1..3 bars
  after the sweep confirm bar
* ``swing_all`` — every confirmed causal swing
* ``swing_displacement_confirmed`` — confirmed swings with a reversal-
  direction displacement landing 1..3 bars after the swing confirm bar
  (bullish displacement for swing_low, bearish for swing_high)
* ``sweep_swing_confluence`` — confirmed sweeps that touch a swing-based
  liquidity source. Direct: ``sweep_primary_family`` ∈ {``swing_high``,
  ``swing_low``}. Proximity: ``|sweep_source_level - nearest confirmed
  same-side swing level| <= 0.25 * ATR`` at the sweep confirm bar. The
  matched swing must be confirmed at or before the sweep confirm bar.
* ``sweep_swing_displacement_confluence`` — sweeps satisfying both
  ``sweep_swing_confluence`` and ``sweep_displacement_confirmed``.

Bracket profiles tested (TP, SL multipliers in ATR units):

* ``tp0p5_sl0p5`` — (0.5, 0.5)
* ``tp1p0_sl1p0`` — (1.0, 1.0)
* ``tp1p0_sl0p5`` — (1.0, 0.5)
* ``tp0p5_sl1p0`` — (0.5, 1.0)

Direction convention:

* ``below_sell_side`` sweep / ``swing_low`` → LONG: TP = entry + tp*ATR,
  SL = entry - sl*ATR; TP fires when ``high >= TP``, SL when ``low <= SL``.
* ``above_buy_side`` sweep / ``swing_high`` → SHORT: TP = entry - tp*ATR,
  SL = entry + sl*ATR; TP fires when ``low <= TP``, SL when ``high >= SL``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Horizons evaluated.
STEP11U_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Bracket profiles: (label, tp_mult, sl_mult).
STEP11U_BRACKET_PROFILES: tuple[tuple[str, float, float], ...] = (
    ("tp0p5_sl0p5", 0.5, 0.5),
    ("tp1p0_sl1p0", 1.0, 1.0),
    ("tp1p0_sl0p5", 1.0, 0.5),
    ("tp0p5_sl1p0", 0.5, 1.0),
)

# Lookahead window (bars) used to attach reversal-direction displacement
# to a confirmed swing or sweep. The window is part of the research lens,
# not an entry trigger — the events table records membership only; entry
# pricing remains anchored at the original signal confirm bar.
STEP11U_DISPLACEMENT_LOOKAHEAD: tuple[int, int] = (1, 3)

# Confluence threshold for sweep_source_level vs confirmed swing level.
STEP11U_SWING_CONFLUENCE_ATR_TOL: float = 0.25

# Universe identifiers in stable order for grouped reporting.
STEP11U_UNIVERSES: tuple[str, ...] = (
    "sweep_all",
    "sweep_displacement_confirmed",
    "swing_all",
    "swing_displacement_confirmed",
    "sweep_swing_confluence",
    "sweep_swing_displacement_confluence",
)

_BULLISH = "bullish_reversal"
_BEARISH = "bearish_reversal"

# Sample-size thresholds applied to grouped tables.
LOW_SAMPLE_COUNT_THRESHOLD: int = 100
LOW_RESOLUTION_THRESHOLD: int = 50


# ---------------------------------------------------------------------------
# Bracket scan / outcome helpers
# ---------------------------------------------------------------------------


def _scan_bracket(
    *,
    side: str,
    entry_close: float,
    atr_ref: float,
    tp_mult: float,
    sl_mult: float,
    high: np.ndarray,
    low: np.ndarray,
    confirm_idx: int,
    max_horizon: int,
) -> tuple[float | None, float | None, float | None, float, float]:
    """Forward-scan up to ``max_horizon`` bars and return
    ``(first_tp_only, first_sl_only, first_ambiguous, tp_price, sl_price)``.
    Stops at the earliest event because the bracket position closes there.
    """

    if side == _BULLISH:
        tp_price = entry_close + tp_mult * atr_ref
        sl_price = entry_close - sl_mult * atr_ref
    else:
        tp_price = entry_close - tp_mult * atr_ref
        sl_price = entry_close + sl_mult * atr_ref

    first_tp_only: float | None = None
    first_sl_only: float | None = None
    first_ambig: float | None = None

    n = len(high)
    end_idx = min(confirm_idx + max_horizon, n - 1)
    for delay in range(1, max_horizon + 1):
        j = confirm_idx + delay
        if j > end_idx:
            break
        bar_high = high[j]
        bar_low = low[j]
        if not (np.isfinite(bar_high) and np.isfinite(bar_low)):
            continue
        if side == _BULLISH:
            tp_hit = bar_high >= tp_price - 1e-12
            sl_hit = bar_low <= sl_price + 1e-12
        else:
            tp_hit = bar_low <= tp_price + 1e-12
            sl_hit = bar_high >= sl_price - 1e-12
        if tp_hit and sl_hit:
            first_ambig = float(delay)
            break
        if tp_hit:
            first_tp_only = float(delay)
            break
        if sl_hit:
            first_sl_only = float(delay)
            break
    return first_tp_only, first_sl_only, first_ambig, tp_price, sl_price


def _outcome_for_horizon(
    *,
    horizon: int,
    first_tp_only: float | None,
    first_sl_only: float | None,
    first_ambig: float | None,
    available_bars: int,
) -> tuple[str, float | None, float | None]:
    candidates: list[tuple[float, str]] = []
    if first_tp_only is not None and first_tp_only <= horizon:
        candidates.append((first_tp_only, "tp_first"))
    if first_sl_only is not None and first_sl_only <= horizon:
        candidates.append((first_sl_only, "sl_first"))
    if first_ambig is not None and first_ambig <= horizon:
        candidates.append((first_ambig, "ambiguous"))

    if candidates:
        candidates.sort(key=lambda x: (x[0],))
        bar, outcome = candidates[0]
        if outcome == "tp_first":
            return outcome, float(bar), None
        if outcome == "sl_first":
            return outcome, None, float(bar)
        return outcome, float(bar), float(bar)

    if available_bars < horizon:
        return "insufficient", None, None
    return "neither", None, None


# ---------------------------------------------------------------------------
# Confluence + displacement membership
# ---------------------------------------------------------------------------


def _swing_direction(side_value: str) -> str:
    return _BULLISH if side_value == "swing_low" else _BEARISH


def _sweep_direction(sweep_side_numeric: float) -> str:
    if not np.isfinite(sweep_side_numeric):
        return ""
    val = int(sweep_side_numeric)
    if val == -1:
        return _BULLISH
    if val == 1:
        return _BEARISH
    return ""


def _sweep_side_label(sweep_side_numeric: float) -> str:
    if not np.isfinite(sweep_side_numeric):
        return ""
    val = int(sweep_side_numeric)
    if val == -1:
        return "below_sell_side"
    if val == 1:
        return "above_buy_side"
    return ""


def _has_reversal_displacement(
    *,
    direction: str,
    confirm_idx: int,
    n: int,
    bull_arr: np.ndarray,
    bear_arr: np.ndarray,
    lookahead: tuple[int, int],
) -> bool:
    lo, hi = lookahead
    end = min(confirm_idx + hi, n - 1)
    arr = bull_arr if direction == _BULLISH else bear_arr
    for delay in range(lo, hi + 1):
        j = confirm_idx + delay
        if j > end:
            break
        if arr[j] > 0:
            return True
    return False


def _nearest_confirmed_swing_level(
    *,
    direction_for_match: str,
    confirmed_high_levels: list[float],
    confirmed_low_levels: list[float],
    target_level: float,
) -> float:
    """Return the nearest already-confirmed swing level on the requested
    side. Returns NaN when there are no confirmed swings of that side yet.
    """

    pool = (
        confirmed_low_levels
        if direction_for_match == "swing_low"
        else confirmed_high_levels
    )
    if not pool or not np.isfinite(target_level):
        return float("nan")
    arr = np.asarray(pool, dtype=float)
    diffs = np.abs(arr - target_level)
    if diffs.size == 0:
        return float("nan")
    return float(arr[int(np.argmin(diffs))])


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------


def _coerce_int_array(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(n, dtype=float)


def _coerce_float_array(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.full(n, np.nan, dtype=float)


def _coerce_str_array(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    if col in df.columns:
        return df[col].astype(str).fillna("").to_numpy(dtype=object)
    return np.array([""] * n, dtype=object)


def build_step11u_events(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = STEP11U_HORIZONS,
    bracket_profiles: Iterable[tuple[str, float, float]] = STEP11U_BRACKET_PROFILES,
    displacement_lookahead: tuple[int, int] = STEP11U_DISPLACEMENT_LOOKAHEAD,
    swing_confluence_atr_tol: float = STEP11U_SWING_CONFLUENCE_ATR_TOL,
) -> pd.DataFrame:
    """Build the per-event Step 11U table for both confirmed swings and
    confirmed sweeps, with confluence/displacement membership flags and
    bracket outcomes for each profile × horizon combination.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    horizons_t = tuple(int(h) for h in horizons)
    profiles_t = tuple(bracket_profiles)
    if not horizons_t or not profiles_t:
        return pd.DataFrame()
    max_horizon = max(horizons_t)

    required = {"high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    if (
        not df.index.is_monotonic_increasing
        or df.index[0] != 0
        or df.index[-1] != len(df) - 1
    ):
        df = df.reset_index(drop=True)

    n = len(df)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df["atr_14"], errors="coerce").to_numpy(dtype=float)

    timestamps = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "timestamp" in df.columns
        else pd.Series([pd.NaT] * n)
    )

    regime_label = _coerce_str_array(df, "regime_label", n)
    session_phase = _coerce_str_array(df, "session_name", n)
    vol_ratio = (
        pd.to_numeric(df["vol_ratio"], errors="coerce").to_numpy(dtype=float)
        if "vol_ratio" in df.columns
        else _coerce_float_array(df, "volume_ratio_20", n)
    )
    bull_arr = _coerce_int_array(df, "displacement_bull", n)
    bear_arr = _coerce_int_array(df, "displacement_bear", n)

    swing_high_flag = _coerce_int_array(df, "swing_high_confirm_flag", n)
    swing_low_flag = _coerce_int_array(df, "swing_low_confirm_flag", n)
    swing_high_price = _coerce_float_array(df, "swing_high_confirm_price", n)
    swing_low_price = _coerce_float_array(df, "swing_low_confirm_price", n)

    sweep_flag = _coerce_int_array(df, "sweep_flag", n)
    sweep_side = _coerce_float_array(df, "sweep_side", n)
    sweep_source_level = _coerce_float_array(df, "sweep_source_level", n)
    sweep_class_arr = _coerce_float_array(df, "sweep_class", n)
    sel_class = _coerce_str_array(df, "sweep_selectivity_class", n)
    primary_family = _coerce_str_array(df, "sweep_primary_family", n)
    is_disp_conf = _coerce_int_array(df, "sweep_is_displacement_confirmed", n)
    is_tradeable = _coerce_int_array(df, "sweep_is_tradeable_candidate", n)

    # Running pools of confirmed swing levels usable up to (and including)
    # the current bar — never future swings, preserving causality.
    confirmed_high_levels: list[float] = []
    confirmed_low_levels: list[float] = []

    rows: list[dict[str, object]] = []
    event_id = 1

    for t in range(n):
        # First emit signal events at this bar using the confirmed-swing
        # pool BEFORE we add today's confirmations — a swing only becomes
        # "available" for confluence on subsequent bars per causality
        # (a swing cannot confluence with itself; the sweep+swing setup
        # must reference a swing already confirmed strictly before, OR at
        # the same bar but contributed by a prior signal flow). The spec
        # allows "before or at sweep_confirm_idx", so we add today's
        # confirmations to the pool BEFORE evaluating sweeps at the same
        # bar. Order within bar: swings first, then sweeps.
        if swing_high_flag[t] > 0 and np.isfinite(swing_high_price[t]):
            confirmed_high_levels.append(float(swing_high_price[t]))
            rows.append(
                _build_swing_event_row(
                    event_id=event_id,
                    confirm_idx=t,
                    confirm_ts=timestamps.iat[t],
                    side="swing_high",
                    entry_close=close[t],
                    atr_ref=atr[t],
                    high=high,
                    low=low,
                    n=n,
                    horizons=horizons_t,
                    profiles=profiles_t,
                    max_horizon=max_horizon,
                    bull_arr=bull_arr,
                    bear_arr=bear_arr,
                    displacement_lookahead=displacement_lookahead,
                    regime_label=str(regime_label[t] or ""),
                    session_phase=str(session_phase[t] or ""),
                    vol_ratio_value=vol_ratio[t],
                )
            )
            event_id += 1
        if swing_low_flag[t] > 0 and np.isfinite(swing_low_price[t]):
            confirmed_low_levels.append(float(swing_low_price[t]))
            rows.append(
                _build_swing_event_row(
                    event_id=event_id,
                    confirm_idx=t,
                    confirm_ts=timestamps.iat[t],
                    side="swing_low",
                    entry_close=close[t],
                    atr_ref=atr[t],
                    high=high,
                    low=low,
                    n=n,
                    horizons=horizons_t,
                    profiles=profiles_t,
                    max_horizon=max_horizon,
                    bull_arr=bull_arr,
                    bear_arr=bear_arr,
                    displacement_lookahead=displacement_lookahead,
                    regime_label=str(regime_label[t] or ""),
                    session_phase=str(session_phase[t] or ""),
                    vol_ratio_value=vol_ratio[t],
                )
            )
            event_id += 1

        if sweep_flag[t] > 0:
            direction = _sweep_direction(sweep_side[t])
            if direction == "":
                continue
            # Same-side swing for confluence: bullish sweep matches against
            # a swing_low pool (sell-side liquidity below); bearish sweep
            # matches against swing_high pool (buy-side liquidity above).
            match_pool = "swing_low" if direction == _BULLISH else "swing_high"
            nearest_level = _nearest_confirmed_swing_level(
                direction_for_match=match_pool,
                confirmed_high_levels=confirmed_high_levels,
                confirmed_low_levels=confirmed_low_levels,
                target_level=sweep_source_level[t],
            )
            atr_ref = atr[t]
            if (
                np.isfinite(nearest_level)
                and np.isfinite(sweep_source_level[t])
                and np.isfinite(atr_ref)
                and atr_ref > 0
            ):
                proximity_dist_atr = (
                    abs(sweep_source_level[t] - nearest_level) / atr_ref
                )
            else:
                proximity_dist_atr = float("nan")
            family_str = str(primary_family[t] or "")
            direct_confluence = family_str in {"swing_high", "swing_low"}
            proximity_confluence = (
                np.isfinite(proximity_dist_atr)
                and proximity_dist_atr <= swing_confluence_atr_tol
            )
            swing_confluent = bool(direct_confluence or proximity_confluence)
            if direct_confluence:
                confluence_kind = "direct_source"
            elif proximity_confluence:
                confluence_kind = "proximity"
            else:
                confluence_kind = "none"

            class_displacement = is_disp_conf[t] > 0
            window_displacement = _has_reversal_displacement(
                direction=direction,
                confirm_idx=t,
                n=n,
                bull_arr=bull_arr,
                bear_arr=bear_arr,
                lookahead=displacement_lookahead,
            )
            displacement_confirmed = bool(class_displacement or window_displacement)

            rows.append(
                _build_sweep_event_row(
                    event_id=event_id,
                    confirm_idx=t,
                    confirm_ts=timestamps.iat[t],
                    direction=direction,
                    entry_close=close[t],
                    atr_ref=atr_ref,
                    high=high,
                    low=low,
                    n=n,
                    horizons=horizons_t,
                    profiles=profiles_t,
                    max_horizon=max_horizon,
                    sweep_side_label=_sweep_side_label(sweep_side[t]),
                    sweep_class_value=sweep_class_arr[t],
                    selectivity_class=str(sel_class[t] or ""),
                    primary_family=family_str,
                    is_tradeable=is_tradeable[t] > 0,
                    displacement_confirmed=displacement_confirmed,
                    swing_confluent=swing_confluent,
                    swing_confluence_type=confluence_kind,
                    swing_confluence_dist_atr=proximity_dist_atr,
                    regime_label=str(regime_label[t] or ""),
                    session_phase=str(session_phase[t] or ""),
                    vol_ratio_value=vol_ratio[t],
                )
            )
            event_id += 1

    if not rows:
        return pd.DataFrame()

    events = (
        pd.DataFrame(rows)
        .sort_values(["signal_idx", "event_id"])
        .reset_index(drop=True)
    )
    return events


def _build_swing_event_row(
    *,
    event_id: int,
    confirm_idx: int,
    confirm_ts: pd.Timestamp,
    side: str,
    entry_close: float,
    atr_ref: float,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
    horizons: tuple[int, ...],
    profiles: tuple[tuple[str, float, float], ...],
    max_horizon: int,
    bull_arr: np.ndarray,
    bear_arr: np.ndarray,
    displacement_lookahead: tuple[int, int],
    regime_label: str,
    session_phase: str,
    vol_ratio_value: float,
) -> dict[str, object]:
    direction = _swing_direction(side)
    displacement_confirmed = _has_reversal_displacement(
        direction=direction,
        confirm_idx=confirm_idx,
        n=n,
        bull_arr=bull_arr,
        bear_arr=bear_arr,
        lookahead=displacement_lookahead,
    )
    confluence_type = (
        "swing_displacement_confirmed" if displacement_confirmed else "swing_all"
    )
    row: dict[str, object] = {
        "event_id": int(event_id),
        "signal_entity_type": "swing",
        "signal_confluence_type": confluence_type,
        "signal_idx": int(confirm_idx),
        "signal_timestamp": confirm_ts,
        "signal_side": side,
        "entry_direction": "LONG" if direction == _BULLISH else "SHORT",
        "entry_price": float(entry_close) if np.isfinite(entry_close) else np.nan,
        "atr_ref": float(atr_ref) if np.isfinite(atr_ref) else np.nan,
        "source_family": side,
        "regime_label": regime_label,
        "session_phase": session_phase,
        "volume_confirmed": bool(
            np.isfinite(vol_ratio_value) and vol_ratio_value > 1.0
        ),
        "displacement_confirmed": bool(displacement_confirmed),
        "swing_confluent": False,
        "swing_confluence_type": "none",
        "swing_confluence_distance_atr": float("nan"),
        # Sweep-specific columns blank for swing events.
        "sweep_side_label": "",
        "sweep_class": float("nan"),
        "sweep_selectivity_class": "",
        "sweep_primary_family": "",
        "sweep_is_tradeable_candidate": False,
        "swing_side": side,
    }
    _attach_bracket_outcomes(
        row=row,
        side=direction,
        entry_close=entry_close,
        atr_ref=atr_ref,
        high=high,
        low=low,
        n=n,
        confirm_idx=confirm_idx,
        horizons=horizons,
        profiles=profiles,
        max_horizon=max_horizon,
    )
    return row


def _build_sweep_event_row(
    *,
    event_id: int,
    confirm_idx: int,
    confirm_ts: pd.Timestamp,
    direction: str,
    entry_close: float,
    atr_ref: float,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
    horizons: tuple[int, ...],
    profiles: tuple[tuple[str, float, float], ...],
    max_horizon: int,
    sweep_side_label: str,
    sweep_class_value: float,
    selectivity_class: str,
    primary_family: str,
    is_tradeable: bool,
    displacement_confirmed: bool,
    swing_confluent: bool,
    swing_confluence_type: str,
    swing_confluence_dist_atr: float,
    regime_label: str,
    session_phase: str,
    vol_ratio_value: float,
) -> dict[str, object]:
    if swing_confluent and displacement_confirmed:
        confluence = "sweep_swing_displacement_confluence"
    elif swing_confluent:
        confluence = "sweep_swing_confluence"
    elif displacement_confirmed:
        confluence = "sweep_displacement_confirmed"
    else:
        confluence = "sweep_all"

    row: dict[str, object] = {
        "event_id": int(event_id),
        "signal_entity_type": "sweep",
        "signal_confluence_type": confluence,
        "signal_idx": int(confirm_idx),
        "signal_timestamp": confirm_ts,
        "signal_side": sweep_side_label,
        "entry_direction": "LONG" if direction == _BULLISH else "SHORT",
        "entry_price": float(entry_close) if np.isfinite(entry_close) else np.nan,
        "atr_ref": float(atr_ref) if np.isfinite(atr_ref) else np.nan,
        "source_family": primary_family,
        "regime_label": regime_label,
        "session_phase": session_phase,
        "volume_confirmed": bool(
            np.isfinite(vol_ratio_value) and vol_ratio_value > 1.0
        ),
        "displacement_confirmed": bool(displacement_confirmed),
        "swing_confluent": bool(swing_confluent),
        "swing_confluence_type": swing_confluence_type,
        "swing_confluence_distance_atr": (
            float(swing_confluence_dist_atr)
            if np.isfinite(swing_confluence_dist_atr)
            else float("nan")
        ),
        "sweep_side_label": sweep_side_label,
        "sweep_class": (
            float(sweep_class_value) if np.isfinite(sweep_class_value) else float("nan")
        ),
        "sweep_selectivity_class": selectivity_class,
        "sweep_primary_family": primary_family,
        "sweep_is_tradeable_candidate": bool(is_tradeable),
        "swing_side": "",
    }
    _attach_bracket_outcomes(
        row=row,
        side=direction,
        entry_close=entry_close,
        atr_ref=atr_ref,
        high=high,
        low=low,
        n=n,
        confirm_idx=confirm_idx,
        horizons=horizons,
        profiles=profiles,
        max_horizon=max_horizon,
    )
    return row


def _attach_bracket_outcomes(
    *,
    row: dict[str, object],
    side: str,
    entry_close: float,
    atr_ref: float,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
    confirm_idx: int,
    horizons: tuple[int, ...],
    profiles: tuple[tuple[str, float, float], ...],
    max_horizon: int,
) -> None:
    available_bars = max(0, n - 1 - confirm_idx)
    valid_inputs = np.isfinite(entry_close) and np.isfinite(atr_ref) and atr_ref > 0
    for label, tp_mult, sl_mult in profiles:
        if valid_inputs:
            first_tp_only, first_sl_only, first_ambig, tp_price, sl_price = (
                _scan_bracket(
                    side=side,
                    entry_close=float(entry_close),
                    atr_ref=float(atr_ref),
                    tp_mult=tp_mult,
                    sl_mult=sl_mult,
                    high=high,
                    low=low,
                    confirm_idx=confirm_idx,
                    max_horizon=max_horizon,
                )
            )
        else:
            first_tp_only = first_sl_only = first_ambig = None
            tp_price = sl_price = float("nan")
        row[f"{label}_tp_price"] = float(tp_price) if np.isfinite(tp_price) else np.nan
        row[f"{label}_sl_price"] = float(sl_price) if np.isfinite(sl_price) else np.nan
        for horizon in horizons:
            if not valid_inputs:
                outcome, bars_to_tp, bars_to_sl = "insufficient", None, None
            else:
                outcome, bars_to_tp, bars_to_sl = _outcome_for_horizon(
                    horizon=horizon,
                    first_tp_only=first_tp_only,
                    first_sl_only=first_sl_only,
                    first_ambig=first_ambig,
                    available_bars=available_bars,
                )
            row[f"{label}_bracket_outcome_{horizon}"] = outcome
            row[f"{label}_bars_to_tp_{horizon}"] = (
                float(bars_to_tp) if bars_to_tp is not None else np.nan
            )
            row[f"{label}_bars_to_sl_{horizon}"] = (
                float(bars_to_sl) if bars_to_sl is not None else np.nan
            )
            row[f"{label}_tp_before_sl_flag_{horizon}"] = (
                1.0 if outcome == "tp_first" else 0.0
            )
            row[f"{label}_sl_before_tp_flag_{horizon}"] = (
                1.0 if outcome == "sl_first" else 0.0
            )
            row[f"{label}_ambiguous_same_bar_flag_{horizon}"] = (
                1.0 if outcome == "ambiguous" else 0.0
            )


# ---------------------------------------------------------------------------
# Group rollups
# ---------------------------------------------------------------------------


def _group_metrics(
    group: pd.DataFrame,
    *,
    horizons: Iterable[int],
    profiles: Iterable[tuple[str, float, float]],
) -> dict[str, object]:
    metrics: dict[str, object] = {"count": int(len(group))}
    for label, _tp, _sl in profiles:
        for horizon in horizons:
            outcome_col = f"{label}_bracket_outcome_{horizon}"
            outcomes = (
                group[outcome_col].astype(str)
                if outcome_col in group.columns
                else pd.Series([], dtype=str)
            )
            tp = int((outcomes == "tp_first").sum())
            sl = int((outcomes == "sl_first").sum())
            amb = int((outcomes == "ambiguous").sum())
            none = int((outcomes == "neither").sum())
            ins = int((outcomes == "insufficient").sum())
            n = tp + sl + amb + none + ins
            resolved = tp + sl
            metrics[f"{label}_tp_first_count_{horizon}"] = tp
            metrics[f"{label}_sl_first_count_{horizon}"] = sl
            metrics[f"{label}_ambiguous_count_{horizon}"] = amb
            metrics[f"{label}_neither_count_{horizon}"] = none
            metrics[f"{label}_insufficient_count_{horizon}"] = ins
            metrics[f"{label}_resolved_count_{horizon}"] = resolved
            metrics[f"{label}_tp_first_rate_{horizon}"] = (
                float(tp / n) if n else float("nan")
            )
            metrics[f"{label}_sl_first_rate_{horizon}"] = (
                float(sl / n) if n else float("nan")
            )
            metrics[f"{label}_ambiguous_rate_{horizon}"] = (
                float(amb / n) if n else float("nan")
            )
            metrics[f"{label}_neither_rate_{horizon}"] = (
                float(none / n) if n else float("nan")
            )
            metrics[f"{label}_win_rate_ex_ambiguous_{horizon}"] = (
                float(tp / resolved) if resolved > 0 else float("nan")
            )
            denom_half = resolved + amb
            metrics[f"{label}_win_rate_with_ambiguous_half_credit_{horizon}"] = (
                float((tp + 0.5 * amb) / denom_half) if denom_half > 0 else float("nan")
            )
    return metrics


def _annotate_low_flags(
    table: pd.DataFrame,
    *,
    profiles: Iterable[tuple[str, float, float]],
    primary_horizon: int = 5,
) -> pd.DataFrame:
    if table.empty or "count" not in table.columns:
        return table
    table = table.copy()
    table["low_sample"] = (
        pd.to_numeric(table["count"], errors="coerce").fillna(0)
        < LOW_SAMPLE_COUNT_THRESHOLD
    )
    for label, _tp, _sl in profiles:
        col = f"{label}_resolved_count_{primary_horizon}"
        if col in table.columns:
            table[f"low_resolution_{label}_h{primary_horizon}"] = (
                pd.to_numeric(table[col], errors="coerce").fillna(0)
                < LOW_RESOLUTION_THRESHOLD
            )
    return table


def _filter_universe(events: pd.DataFrame, universe: str) -> pd.DataFrame:
    if events.empty:
        return events
    if universe == "sweep_all":
        return events[events["signal_entity_type"] == "sweep"]
    if universe == "sweep_displacement_confirmed":
        return events[
            (events["signal_entity_type"] == "sweep")
            & (events["displacement_confirmed"].astype(bool))
        ]
    if universe == "swing_all":
        return events[events["signal_entity_type"] == "swing"]
    if universe == "swing_displacement_confirmed":
        return events[
            (events["signal_entity_type"] == "swing")
            & (events["displacement_confirmed"].astype(bool))
        ]
    if universe == "sweep_swing_confluence":
        return events[
            (events["signal_entity_type"] == "sweep")
            & (events["swing_confluent"].astype(bool))
        ]
    if universe == "sweep_swing_displacement_confluence":
        return events[
            (events["signal_entity_type"] == "sweep")
            & (events["swing_confluent"].astype(bool))
            & (events["displacement_confirmed"].astype(bool))
        ]
    return events.iloc[0:0]


def build_by_universe_table(
    events: pd.DataFrame,
    *,
    horizons: Iterable[int],
    profiles: Iterable[tuple[str, float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for universe in STEP11U_UNIVERSES:
        sub = _filter_universe(events, universe)
        rows.append(
            {
                "signal_confluence_type": universe,
                **_group_metrics(sub, horizons=horizons, profiles=profiles),
            }
        )
    table = pd.DataFrame(rows)
    return _annotate_low_flags(table, profiles=profiles)


def build_group_table(
    events: pd.DataFrame,
    *,
    group_col: str,
    out_col: str,
    horizons: Iterable[int],
    profiles: Iterable[tuple[str, float, float]],
    ordered_values: list[str] | None = None,
    sort_by_count: bool = True,
    bool_column: bool = False,
) -> pd.DataFrame:
    if events.empty or group_col not in events.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if ordered_values is not None:
        for value in ordered_values:
            sub = events[events[group_col].astype(str) == value]
            rows.append(
                {
                    out_col: value,
                    **_group_metrics(sub, horizons=horizons, profiles=profiles),
                }
            )
    elif bool_column:
        for value, sub in events.groupby(events[group_col].astype(bool), dropna=False):
            rows.append(
                {
                    out_col: bool(value),
                    **_group_metrics(sub, horizons=horizons, profiles=profiles),
                }
            )
    else:
        for value, sub in events.groupby(group_col, dropna=False, observed=False):
            label = "" if pd.isna(value) else str(value)
            if label == "" or label == "nan":
                continue
            rows.append(
                {
                    out_col: label,
                    **_group_metrics(sub, horizons=horizons, profiles=profiles),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    if sort_by_count and "count" in table.columns:
        table = table.sort_values(
            ["count", out_col], ascending=[False, True]
        ).reset_index(drop=True)
    else:
        table = table.reset_index(drop=True)
    return _annotate_low_flags(table, profiles=profiles)


def build_cross_group_table(
    events: pd.DataFrame,
    *,
    second_col: str,
    second_out_col: str,
    horizons: Iterable[int],
    profiles: Iterable[tuple[str, float, float]],
    bool_second: bool = False,
) -> pd.DataFrame:
    """Cross-group ``signal_confluence_type`` × another column."""

    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for universe in STEP11U_UNIVERSES:
        sub = _filter_universe(events, universe)
        if sub.empty or second_col not in sub.columns:
            continue
        if bool_second:
            iterator = sub.groupby(sub[second_col].astype(bool), dropna=False)
        else:
            iterator = sub.groupby(second_col, dropna=False, observed=False)
        for value, group in iterator:
            if not bool_second:
                label = "" if pd.isna(value) else str(value)
                if label == "" or label == "nan":
                    continue
            else:
                label = bool(value)
            rows.append(
                {
                    "signal_confluence_type": universe,
                    second_out_col: label,
                    **_group_metrics(group, horizons=horizons, profiles=profiles),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return _annotate_low_flags(table, profiles=profiles)


# ---------------------------------------------------------------------------
# Best-group selection + headline
# ---------------------------------------------------------------------------


def _best_group_label(
    table: pd.DataFrame,
    *,
    group_col: str,
    profile_label: str,
    horizon: int = 5,
    min_count: int = LOW_SAMPLE_COUNT_THRESHOLD,
    min_resolved: int = LOW_RESOLUTION_THRESHOLD,
) -> str:
    if table.empty or group_col not in table.columns:
        return ""
    win_col = f"{profile_label}_win_rate_ex_ambiguous_{horizon}"
    resolved_col = f"{profile_label}_resolved_count_{horizon}"
    if win_col not in table.columns or resolved_col not in table.columns:
        return ""
    cand = table.copy()
    cand = cand[
        (pd.to_numeric(cand["count"], errors="coerce").fillna(0) >= min_count)
        & (pd.to_numeric(cand[resolved_col], errors="coerce").fillna(0) >= min_resolved)
    ]
    cand = cand.dropna(subset=[win_col])
    if cand.empty:
        return ""
    cand = cand.sort_values([win_col, resolved_col], ascending=[False, False])
    return str(cand.iloc[0][group_col])


def _build_headline_summary(
    *,
    by_confluence: pd.DataFrame,
    horizons: tuple[int, ...],
    profiles: tuple[tuple[str, float, float], ...],
    primary_horizon: int = 5,
) -> dict[str, object]:
    summary: dict[str, object] = {"primary_horizon": primary_horizon}
    if by_confluence.empty:
        return summary
    for _, row in by_confluence.iterrows():
        universe = str(row["signal_confluence_type"])
        summary[f"{universe}_count"] = int(row.get("count", 0) or 0)
        for label, _tp, _sl in profiles:
            summary[f"{universe}_{label}_win_rate_ex_ambiguous_{primary_horizon}"] = (
                row.get(
                    f"{label}_win_rate_ex_ambiguous_{primary_horizon}", float("nan")
                )
            )
            summary[f"{universe}_{label}_ambiguous_rate_{primary_horizon}"] = row.get(
                f"{label}_ambiguous_rate_{primary_horizon}", float("nan")
            )
            summary[f"{universe}_{label}_resolved_count_{primary_horizon}"] = int(
                row.get(f"{label}_resolved_count_{primary_horizon}", 0) or 0
            )

    for label, _tp, _sl in profiles:
        summary[f"best_group_by_{label}"] = _best_group_label(
            by_confluence,
            group_col="signal_confluence_type",
            profile_label=label,
            horizon=primary_horizon,
            min_count=LOW_SAMPLE_COUNT_THRESHOLD,
            min_resolved=LOW_RESOLUTION_THRESHOLD,
        )
    summary["best_group_with_min_count_100"] = _best_group_label(
        by_confluence,
        group_col="signal_confluence_type",
        profile_label="tp0p5_sl0p5",
        horizon=primary_horizon,
        min_count=100,
        min_resolved=LOW_RESOLUTION_THRESHOLD,
    )
    summary["best_group_with_min_count_300"] = _best_group_label(
        by_confluence,
        group_col="signal_confluence_type",
        profile_label="tp0p5_sl0p5",
        horizon=primary_horizon,
        min_count=300,
        min_resolved=LOW_RESOLUTION_THRESHOLD,
    )
    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_step11u_diagnostics(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = STEP11U_HORIZONS,
    bracket_profiles: Iterable[tuple[str, float, float]] = STEP11U_BRACKET_PROFILES,
    displacement_lookahead: tuple[int, int] = STEP11U_DISPLACEMENT_LOOKAHEAD,
    swing_confluence_atr_tol: float = STEP11U_SWING_CONFLUENCE_ATR_TOL,
) -> dict[str, object]:
    horizons_t = tuple(int(h) for h in horizons)
    profiles_t = tuple(bracket_profiles)
    events = build_step11u_events(
        df,
        horizons=horizons_t,
        bracket_profiles=profiles_t,
        displacement_lookahead=displacement_lookahead,
        swing_confluence_atr_tol=swing_confluence_atr_tol,
    )
    if events.empty:
        return {
            "events": pd.DataFrame(),
            "by_confluence_type": pd.DataFrame(),
            "by_entity_type": pd.DataFrame(),
            "by_sweep_class": pd.DataFrame(),
            "by_sweep_family": pd.DataFrame(),
            "by_swing_side": pd.DataFrame(),
            "by_regime": pd.DataFrame(),
            "by_session": pd.DataFrame(),
            "by_volume_confirmed": pd.DataFrame(),
            "by_displacement_confirmed": pd.DataFrame(),
            "by_swing_confluent": pd.DataFrame(),
            "by_confluence_regime": pd.DataFrame(),
            "by_confluence_session": pd.DataFrame(),
            "by_confluence_volume": pd.DataFrame(),
            "summary": {"primary_horizon": 5},
        }

    by_confluence = build_by_universe_table(
        events, horizons=horizons_t, profiles=profiles_t
    )
    by_entity = build_group_table(
        events,
        group_col="signal_entity_type",
        out_col="signal_entity_type",
        horizons=horizons_t,
        profiles=profiles_t,
        ordered_values=["sweep", "swing"],
        sort_by_count=False,
    )
    sweep_events = events[events["signal_entity_type"] == "sweep"]
    swing_events = events[events["signal_entity_type"] == "swing"]
    by_sweep_class = build_group_table(
        sweep_events,
        group_col="sweep_selectivity_class",
        out_col="sweep_selectivity_class",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_sweep_family = build_group_table(
        sweep_events,
        group_col="sweep_primary_family",
        out_col="sweep_primary_family",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_swing_side = build_group_table(
        swing_events,
        group_col="swing_side",
        out_col="swing_side",
        horizons=horizons_t,
        profiles=profiles_t,
        ordered_values=["swing_high", "swing_low"],
        sort_by_count=False,
    )
    by_regime = build_group_table(
        events,
        group_col="regime_label",
        out_col="regime_label",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_session = build_group_table(
        events,
        group_col="session_phase",
        out_col="session_phase",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_volume = build_group_table(
        events,
        group_col="volume_confirmed",
        out_col="volume_confirmed",
        horizons=horizons_t,
        profiles=profiles_t,
        bool_column=True,
        sort_by_count=False,
    )
    by_displacement = build_group_table(
        events,
        group_col="displacement_confirmed",
        out_col="displacement_confirmed",
        horizons=horizons_t,
        profiles=profiles_t,
        bool_column=True,
        sort_by_count=False,
    )
    by_swing_confluent = build_group_table(
        sweep_events,
        group_col="swing_confluent",
        out_col="swing_confluent",
        horizons=horizons_t,
        profiles=profiles_t,
        bool_column=True,
        sort_by_count=False,
    )
    by_conf_regime = build_cross_group_table(
        events,
        second_col="regime_label",
        second_out_col="regime_label",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_conf_session = build_cross_group_table(
        events,
        second_col="session_phase",
        second_out_col="session_phase",
        horizons=horizons_t,
        profiles=profiles_t,
    )
    by_conf_volume = build_cross_group_table(
        events,
        second_col="volume_confirmed",
        second_out_col="volume_confirmed",
        horizons=horizons_t,
        profiles=profiles_t,
        bool_second=True,
    )

    summary = _build_headline_summary(
        by_confluence=by_confluence,
        horizons=horizons_t,
        profiles=profiles_t,
    )

    return {
        "events": events,
        "by_confluence_type": by_confluence,
        "by_entity_type": by_entity,
        "by_sweep_class": by_sweep_class,
        "by_sweep_family": by_sweep_family,
        "by_swing_side": by_swing_side,
        "by_regime": by_regime,
        "by_session": by_session,
        "by_volume_confirmed": by_volume,
        "by_displacement_confirmed": by_displacement,
        "by_swing_confluent": by_swing_confluent,
        "by_confluence_regime": by_conf_regime,
        "by_confluence_session": by_conf_session,
        "by_confluence_volume": by_conf_volume,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_step11u_summary(
    summary: dict[str, object],
    *,
    indent: int = 0,
    primary_horizon: int = 5,
    profiles: Iterable[tuple[str, float, float]] = STEP11U_BRACKET_PROFILES,
) -> None:
    prefix = " " * indent
    print(f"{prefix}=== STEP 11U BRACKET MATRIX + CONFLUENCE EDGE AUDIT ===")
    profile_labels = [label for label, _, _ in profiles]
    h = primary_horizon
    for universe in STEP11U_UNIVERSES:
        count = int(summary.get(f"{universe}_count", 0) or 0)
        print(f"{prefix}{universe}: count={count}")
        for label in profile_labels:
            win = summary.get(
                f"{universe}_{label}_win_rate_ex_ambiguous_{h}", float("nan")
            )
            amb = summary.get(f"{universe}_{label}_ambiguous_rate_{h}", float("nan"))
            res = int(summary.get(f"{universe}_{label}_resolved_count_{h}", 0) or 0)
            try:
                win_v = float(win)
            except (TypeError, ValueError):
                win_v = float("nan")
            try:
                amb_v = float(amb)
            except (TypeError, ValueError):
                amb_v = float("nan")
            win_s = "" if not np.isfinite(win_v) else f"{win_v:.4f}"
            amb_s = "" if not np.isfinite(amb_v) else f"{amb_v:.4f}"
            print(
                f"{prefix}  {label}: win_rate_ex_ambiguous_{h}={win_s} "
                f"ambiguous_rate_{h}={amb_s} resolved_count_{h}={res}"
            )
    for label in profile_labels:
        print(
            f"{prefix}best_group_by_{label}: {summary.get(f'best_group_by_{label}', '')}"
        )
    print(
        f"{prefix}best_group_with_min_count_100: {summary.get('best_group_with_min_count_100', '')}"
    )
    print(
        f"{prefix}best_group_with_min_count_300: {summary.get('best_group_with_min_count_300', '')}"
    )


__all__ = [
    "STEP11U_HORIZONS",
    "STEP11U_BRACKET_PROFILES",
    "STEP11U_UNIVERSES",
    "STEP11U_DISPLACEMENT_LOOKAHEAD",
    "STEP11U_SWING_CONFLUENCE_ATR_TOL",
    "build_step11u_events",
    "build_step11u_diagnostics",
    "print_step11u_summary",
]
