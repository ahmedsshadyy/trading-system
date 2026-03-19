"""
structure/trend_state.py

Causal trend state engine built on canonical swing events.

Separates:

1. strict current structure state
   - ``trend_state`` = +1 / -1 / 0
   - built from RECENT HH+HL or LH+LL evidence only

2. restrained directional bias
   - ``trend_bias_state`` = +1 / -1 / 0
   - only inherited from the most recent strict state
   - decays during neutral structure
   - dies fast on contradiction or TTL expiry
   - never gets refreshed by partial events while strict state is neutral

Also exposes:
- directional structural pressure / strength
- transition interpretability helpers

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns


def add_trend_state(
    df: pd.DataFrame,
    *,
    equal_tol: float = 0.0,
    equal_tol_atr_mult: float = 0.0,
    atr_length: int = 14,
    min_confirmations_per_side: int = 1,
    event_freshness_bars: int = 24,
    bias_half_life_bars: int = 2,
    bias_min_score: float = 0.40,
    bias_neutral_ttl_bars: int = 3,
    strength_ema_span: int = 5,
    strength_decay_half_life_bars: int = 12,
    strength_event_cap: float = 2.5,
    strength_norm_cap: float = 4.0,
    hh_weight: float = 1.00,
    hl_weight: float = 0.85,
    lh_weight: float = 1.00,
    ll_weight: float = 0.85,
    emerging_strength_threshold: float = 0.12,
    structure_loss_strength_threshold: float = 0.08,
) -> pd.DataFrame:
    """Build causal structural trend state from swing events."""
    out = df.copy()

    require_columns(
        out,
        {
            "swing_high",
            "swing_low",
            "swing_high_price",
            "swing_low_price",
        },
    )

    n = len(out)

    sh_flag = out["swing_high"].to_numpy(dtype=np.int8)
    sl_flag = out["swing_low"].to_numpy(dtype=np.int8)
    sh_price = out["swing_high_price"].to_numpy(dtype=float)
    sl_price = out["swing_low_price"].to_numpy(dtype=float)
    atr = get_atr_array(out, atr_length)

    # Optional quality inputs from swings.py
    sh_strength_src = (
        out["swing_high_strength"].to_numpy(dtype=float)
        if "swing_high_strength" in out.columns
        else np.full(n, np.nan)
    )
    sl_strength_src = (
        out["swing_low_strength"].to_numpy(dtype=float)
        if "swing_low_strength" in out.columns
        else np.full(n, np.nan)
    )
    sh_prom_atr_src = (
        out["swing_high_prominence_atr"].to_numpy(dtype=float)
        if "swing_high_prominence_atr" in out.columns
        else np.full(n, np.nan)
    )
    sl_prom_atr_src = (
        out["swing_low_prominence_atr"].to_numpy(dtype=float)
        if "swing_low_prominence_atr" in out.columns
        else np.full(n, np.nan)
    )

    trend_state = np.zeros(n, dtype=np.int8)
    trend_state_age = np.zeros(n, dtype=np.int32)

    trend_bias_state = np.zeros(n, dtype=np.int8)
    trend_bias_age = np.zeros(n, dtype=np.int32)
    trend_bias_score_live = np.zeros(n, dtype=float)

    trend_confidence = np.full(n, -1, dtype=np.int8)
    trend_bias_score = np.zeros(n, dtype=np.int16)

    trend_bull_ready = np.zeros(n, dtype=np.int8)
    trend_bear_ready = np.zeros(n, dtype=np.int8)

    hh_count = np.zeros(n, dtype=np.int16)
    hl_count = np.zeros(n, dtype=np.int16)
    lh_count = np.zeros(n, dtype=np.int16)
    ll_count = np.zeros(n, dtype=np.int16)

    trend_event_high = np.zeros(n, dtype=np.int8)  # +1 HH / -1 LH
    trend_event_low = np.zeros(n, dtype=np.int8)  # +1 HL / -1 LL
    trend_event = np.zeros(n, dtype=np.int8)

    trend_state_changed = np.zeros(n, dtype=np.int8)
    trend_state_from = np.zeros(n, dtype=np.int8)
    trend_state_to = np.zeros(n, dtype=np.int8)

    trend_bias_changed = np.zeros(n, dtype=np.int8)
    trend_bias_from = np.zeros(n, dtype=np.int8)
    trend_bias_to = np.zeros(n, dtype=np.int8)

    last_hh_price = np.full(n, np.nan)
    last_hl_price = np.full(n, np.nan)
    last_lh_price = np.full(n, np.nan)
    last_ll_price = np.full(n, np.nan)

    last_hh_idx = np.full(n, np.nan)
    last_hl_idx = np.full(n, np.nan)
    last_lh_idx = np.full(n, np.nan)
    last_ll_idx = np.full(n, np.nan)

    trend_pressure_bull_raw = np.zeros(n, dtype=float)
    trend_pressure_bear_raw = np.zeros(n, dtype=float)
    trend_strength_raw = np.zeros(n, dtype=float)
    trend_strength_event_score = np.zeros(n, dtype=float)
    trend_strength_event_dir = np.zeros(n, dtype=np.int8)

    # Step 2: interpretability helpers
    trend_structure_loss_bull = np.zeros(n, dtype=np.int8)
    trend_structure_loss_bear = np.zeros(n, dtype=np.int8)
    trend_emerging_bull = np.zeros(n, dtype=np.int8)
    trend_emerging_bear = np.zeros(n, dtype=np.int8)
    trend_regime_phase = np.zeros(n, dtype=np.int8)
    # phase:
    # -2 = bear structure loss
    # -1 = emerging bear
    #  0 = neutral / none
    # +1 = emerging bull
    # +2 = bull structure loss
    # +3 = strict bull
    # -3 = strict bear

    prev_swing_high = np.nan
    prev_swing_low = np.nan

    curr_hh = 0
    curr_hl = 0
    curr_lh = 0
    curr_ll = 0

    cur_last_hh_price = np.nan
    cur_last_hl_price = np.nan
    cur_last_lh_price = np.nan
    cur_last_ll_price = np.nan

    cur_last_hh_idx = np.nan
    cur_last_hl_idx = np.nan
    cur_last_lh_idx = np.nan
    cur_last_ll_idx = np.nan

    curr_state = 0
    curr_state_age = 0

    curr_bias = 0
    curr_bias_age = 0
    curr_bias_mag = 0.0  # always non-negative magnitude

    inherited_bias_dir = 0
    inherited_bias_age = 0

    bull_pressure = 0.0
    bear_pressure = 0.0

    prev_strict_state_for_phase = 0

    bias_decay = 0.5 ** (1.0 / max(bias_half_life_bars, 1))
    strength_decay = 0.5 ** (1.0 / max(strength_decay_half_life_bars, 1))

    def _cmp(new: float, old: float, tol: float) -> int:
        if not np.isfinite(new) or not np.isfinite(old):
            return 0
        diff = new - old
        if abs(diff) <= tol:
            return 0
        return 1 if diff > 0 else -1

    def _recent(idx_val: float, i_: int, freshness: int) -> bool:
        return np.isfinite(idx_val) and (i_ - int(idx_val) <= freshness)

    def _signed_bias(dir_: int, mag: float) -> float:
        if dir_ == 0:
            return 0.0
        return float(dir_) * float(mag)

    def _bounded_strength_score(
        base_strength: float,
        fallback_move: float,
        atr_i: float,
    ) -> float:
        """
        Convert event magnitude into a bounded positive quality score.

        Preference order:
        1) swing strength from swings.py
        2) prominence_atr from swings.py
        3) fallback move / ATR from local structure comparison
        """
        score = np.nan

        if np.isfinite(base_strength):
            score = base_strength
        elif np.isfinite(fallback_move) and np.isfinite(atr_i) and atr_i > 0:
            score = fallback_move / atr_i

        if not np.isfinite(score):
            score = 0.0

        score = max(0.0, float(score))
        return min(score, float(strength_event_cap))

    for i in range(n):
        high_evt = 0
        low_evt = 0

        high_event_score = 0.0
        low_event_score = 0.0

        tol_i = equal_tol
        if np.isfinite(atr[i]) and atr[i] > 0:
            tol_i = max(equal_tol, equal_tol_atr_mult * atr[i])

        # ---------------------------------------------------------
        # High-side structure: HH / LH
        # ---------------------------------------------------------
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            rel = _cmp(sh_price[i], prev_swing_high, tol_i)

            fallback_move = (
                abs(sh_price[i] - prev_swing_high)
                if np.isfinite(prev_swing_high)
                else np.nan
            )

            base_strength = sh_strength_src[i]
            if not np.isfinite(base_strength):
                base_strength = sh_prom_atr_src[i]

            event_mag = _bounded_strength_score(
                base_strength=base_strength,
                fallback_move=fallback_move,
                atr_i=atr[i],
            )

            if rel == 1:
                high_evt = 1
                high_event_score = hh_weight * event_mag

                curr_hh += 1
                curr_lh = 0
                cur_last_hh_price = sh_price[i]
                cur_last_hh_idx = float(i)

            elif rel == -1:
                high_evt = -1
                high_event_score = lh_weight * event_mag

                curr_lh += 1
                curr_hh = 0
                cur_last_lh_price = sh_price[i]
                cur_last_lh_idx = float(i)

            prev_swing_high = sh_price[i]

        # ---------------------------------------------------------
        # Low-side structure: HL / LL
        # ---------------------------------------------------------
        if sl_flag[i] == 1 and np.isfinite(sl_price[i]):
            rel = _cmp(sl_price[i], prev_swing_low, tol_i)

            fallback_move = (
                abs(sl_price[i] - prev_swing_low)
                if np.isfinite(prev_swing_low)
                else np.nan
            )

            base_strength = sl_strength_src[i]
            if not np.isfinite(base_strength):
                base_strength = sl_prom_atr_src[i]

            event_mag = _bounded_strength_score(
                base_strength=base_strength,
                fallback_move=fallback_move,
                atr_i=atr[i],
            )

            if rel == 1:
                low_evt = 1
                low_event_score = hl_weight * event_mag

                curr_hl += 1
                curr_ll = 0
                cur_last_hl_price = sl_price[i]
                cur_last_hl_idx = float(i)

            elif rel == -1:
                low_evt = -1
                low_event_score = ll_weight * event_mag

                curr_ll += 1
                curr_hl = 0
                cur_last_ll_price = sl_price[i]
                cur_last_ll_idx = float(i)

            prev_swing_low = sl_price[i]

        # ---------------------------------------------------------
        # Strict current readiness
        # ---------------------------------------------------------
        bull_pair_recent = _recent(
            cur_last_hh_idx, i, event_freshness_bars
        ) and _recent(cur_last_hl_idx, i, event_freshness_bars)
        bear_pair_recent = _recent(
            cur_last_lh_idx, i, event_freshness_bars
        ) and _recent(cur_last_ll_idx, i, event_freshness_bars)

        bull_counts_ok = (
            curr_hh >= min_confirmations_per_side
            and curr_hl >= min_confirmations_per_side
        )
        bear_counts_ok = (
            curr_lh >= min_confirmations_per_side
            and curr_ll >= min_confirmations_per_side
        )

        latest_bull_pair_idx = (
            max(cur_last_hh_idx, cur_last_hl_idx)
            if np.isfinite(cur_last_hh_idx) and np.isfinite(cur_last_hl_idx)
            else np.nan
        )
        latest_bear_pair_idx = (
            max(cur_last_lh_idx, cur_last_ll_idx)
            if np.isfinite(cur_last_lh_idx) and np.isfinite(cur_last_ll_idx)
            else np.nan
        )

        bull_ready = bull_pair_recent and bull_counts_ok
        bear_ready = bear_pair_recent and bear_counts_ok

        if bull_ready and bear_ready:
            if latest_bull_pair_idx > latest_bear_pair_idx:
                bear_ready = False
            elif latest_bear_pair_idx > latest_bull_pair_idx:
                bull_ready = False
            else:
                bull_ready = False
                bear_ready = False

        trend_bull_ready[i] = int(bull_ready)
        trend_bear_ready[i] = int(bear_ready)

        prev_state = curr_state

        new_state = 0
        if bull_ready and not bear_ready:
            new_state = 1
        elif bear_ready and not bull_ready:
            new_state = -1

        if new_state != curr_state:
            trend_state_changed[i] = 1
            trend_state_from[i] = curr_state
            trend_state_to[i] = new_state
            curr_state = new_state
            curr_state_age = 1 if curr_state != 0 else 0
        else:
            curr_state_age = curr_state_age + 1 if curr_state != 0 else 0

        # ---------------------------------------------------------
        # Inherited neutral bias logic
        # ---------------------------------------------------------
        prev_bias = curr_bias

        if prev_state != 0 and curr_state == 0:
            inherited_bias_dir = prev_state
            inherited_bias_age = 0
            curr_bias = prev_state
            curr_bias_mag = 1.0

        elif curr_state == 1:
            inherited_bias_dir = 0
            inherited_bias_age = 0
            curr_bias = 1
            curr_bias_mag = 1.0

        elif curr_state == -1:
            inherited_bias_dir = 0
            inherited_bias_age = 0
            curr_bias = -1
            curr_bias_mag = 1.0

        else:
            # strict neutral
            if inherited_bias_dir == 0:
                curr_bias = 0
                curr_bias_mag = 0.0
            else:
                inherited_bias_age += 1
                curr_bias_mag *= bias_decay

                contradiction = (
                    inherited_bias_dir == 1 and (high_evt == -1 or low_evt == -1)
                ) or (inherited_bias_dir == -1 and (high_evt == 1 or low_evt == 1))

                expired = (
                    inherited_bias_age > bias_neutral_ttl_bars
                    or curr_bias_mag < bias_min_score
                )

                if contradiction or expired:
                    inherited_bias_dir = 0
                    inherited_bias_age = 0
                    curr_bias = 0
                    curr_bias_mag = 0.0
                else:
                    curr_bias = inherited_bias_dir

        if curr_bias != prev_bias:
            trend_bias_changed[i] = 1
            trend_bias_from[i] = prev_bias
            trend_bias_to[i] = curr_bias
            curr_bias_age = 1 if curr_bias != 0 else 0
        else:
            curr_bias_age = curr_bias_age + 1 if curr_bias != 0 else 0

        # ---------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------
        if curr_state == 1:
            conf = 2 if (high_evt == 1 or low_evt == 1) else 1
        elif curr_state == -1:
            conf = 2 if (high_evt == -1 or low_evt == -1) else 1
        elif curr_bias != 0:
            conf = 0
        else:
            conf = -1

        # ---------------------------------------------------------
        # Directional structural pressure / strength
        # ---------------------------------------------------------
        bull_pressure *= strength_decay
        bear_pressure *= strength_decay

        event_dir = 0
        event_score = 0.0

        if high_evt == 1:
            bull_pressure += high_event_score
            event_dir = 1
            event_score += high_event_score
        elif high_evt == -1:
            bear_pressure += high_event_score
            event_dir = -1
            event_score += high_event_score

        if low_evt == 1:
            bull_pressure += low_event_score
            if event_dir == 0:
                event_dir = 1
            event_score += low_event_score
        elif low_evt == -1:
            bear_pressure += low_event_score
            if event_dir == 0:
                event_dir = -1
            event_score += low_event_score

        bull_pressure = min(max(bull_pressure, 0.0), strength_norm_cap)
        bear_pressure = min(max(bear_pressure, 0.0), strength_norm_cap)

        raw_strength = bull_pressure - bear_pressure
        raw_strength = max(-strength_norm_cap, min(strength_norm_cap, raw_strength))
        if strength_norm_cap > 0:
            raw_strength /= strength_norm_cap

        trend_pressure_bull_raw[i] = bull_pressure
        trend_pressure_bear_raw[i] = bear_pressure
        trend_strength_raw[i] = raw_strength
        trend_strength_event_score[i] = event_score
        trend_strength_event_dir[i] = event_dir

        # ---------------------------------------------------------
        # Step 2: transition interpretability helpers
        # ---------------------------------------------------------
        structure_loss_bull = 0
        structure_loss_bear = 0
        emerging_bull = 0
        emerging_bear = 0
        regime_phase = 0

        if curr_state == 1:
            regime_phase = 3
        elif curr_state == -1:
            regime_phase = -3
        else:
            # strict neutral only below
            recent_bull_component = _recent(
                cur_last_hh_idx, i, event_freshness_bars
            ) or _recent(cur_last_hl_idx, i, event_freshness_bars)
            recent_bear_component = _recent(
                cur_last_lh_idx, i, event_freshness_bars
            ) or _recent(cur_last_ll_idx, i, event_freshness_bars)

            # structure loss: immediately after losing strict structure,
            # while bias/strength still leans in the old direction
            if (
                prev_strict_state_for_phase == 1
                and curr_state == 0
                and curr_bias >= 0
                and raw_strength >= structure_loss_strength_threshold
            ):
                structure_loss_bull = 1
                regime_phase = 2

            elif (
                prev_strict_state_for_phase == -1
                and curr_state == 0
                and curr_bias <= 0
                and raw_strength <= -structure_loss_strength_threshold
            ):
                structure_loss_bear = 1
                regime_phase = -2

            # emerging structure: neutral strict state, but directional pressure
            # and recent same-side evidence are building before strict confirmation
            elif (
                not bull_ready
                and not bear_ready
                and raw_strength >= emerging_strength_threshold
                and recent_bull_component
                and curr_bias >= 0
            ):
                emerging_bull = 1
                regime_phase = 1

            elif (
                not bull_ready
                and not bear_ready
                and raw_strength <= -emerging_strength_threshold
                and recent_bear_component
                and curr_bias <= 0
            ):
                emerging_bear = 1
                regime_phase = -1

        trend_structure_loss_bull[i] = structure_loss_bull
        trend_structure_loss_bear[i] = structure_loss_bear
        trend_emerging_bull[i] = emerging_bull
        trend_emerging_bear[i] = emerging_bear
        trend_regime_phase[i] = regime_phase

        # ---------------------------------------------------------
        # Stamp outputs
        # ---------------------------------------------------------
        trend_event_high[i] = high_evt
        trend_event_low[i] = low_evt
        trend_event[i] = high_evt + low_evt

        hh_count[i] = curr_hh
        hl_count[i] = curr_hl
        lh_count[i] = curr_lh
        ll_count[i] = curr_ll

        trend_state[i] = curr_state
        trend_state_age[i] = curr_state_age

        trend_bias_state[i] = curr_bias
        trend_bias_age[i] = curr_bias_age
        trend_bias_score_live[i] = _signed_bias(curr_bias, curr_bias_mag)

        trend_confidence[i] = conf
        trend_bias_score[i] = (curr_hh + curr_hl) - (curr_lh + curr_ll)

        last_hh_price[i] = cur_last_hh_price
        last_hl_price[i] = cur_last_hl_price
        last_lh_price[i] = cur_last_lh_price
        last_ll_price[i] = cur_last_ll_price

        last_hh_idx[i] = cur_last_hh_idx
        last_hl_idx[i] = cur_last_hl_idx
        last_lh_idx[i] = cur_last_lh_idx
        last_ll_idx[i] = cur_last_ll_idx

        prev_strict_state_for_phase = curr_state

    out["trend_state"] = trend_state
    out["trend_state_age"] = trend_state_age

    out["trend_bias_state"] = trend_bias_state
    out["trend_bias_age"] = trend_bias_age
    out["trend_bias_score_live"] = trend_bias_score_live

    out["trend_confidence"] = trend_confidence
    out["trend_bias_score"] = trend_bias_score

    out["trend_bull_ready"] = trend_bull_ready
    out["trend_bear_ready"] = trend_bear_ready

    out["trend_event_high"] = trend_event_high
    out["trend_event_low"] = trend_event_low
    out["trend_event"] = trend_event

    out["hh_count"] = hh_count
    out["hl_count"] = hl_count
    out["lh_count"] = lh_count
    out["ll_count"] = ll_count

    out["last_hh_price"] = last_hh_price
    out["last_hh_idx"] = last_hh_idx
    out["last_hl_price"] = last_hl_price
    out["last_hl_idx"] = last_hl_idx
    out["last_lh_price"] = last_lh_price
    out["last_lh_idx"] = last_lh_idx
    out["last_ll_price"] = last_ll_price
    out["last_ll_idx"] = last_ll_idx

    out["trend_state_changed"] = trend_state_changed
    out["trend_state_from"] = trend_state_from
    out["trend_state_to"] = trend_state_to

    out["trend_bias_changed"] = trend_bias_changed
    out["trend_bias_from"] = trend_bias_from
    out["trend_bias_to"] = trend_bias_to

    out["trend_pressure_bull_raw"] = trend_pressure_bull_raw
    out["trend_pressure_bear_raw"] = trend_pressure_bear_raw
    out["trend_strength_event_score"] = trend_strength_event_score
    out["trend_strength_event_dir"] = trend_strength_event_dir
    out["trend_strength_raw"] = trend_strength_raw
    out["trend_strength_ema"] = (
        pd.Series(trend_strength_raw, index=out.index)
        .ewm(span=max(strength_ema_span, 1), adjust=False)
        .mean()
        .to_numpy()
    )

    out["trend_structure_loss_bull"] = trend_structure_loss_bull
    out["trend_structure_loss_bear"] = trend_structure_loss_bear
    out["trend_emerging_bull"] = trend_emerging_bull
    out["trend_emerging_bear"] = trend_emerging_bear
    out["trend_regime_phase"] = trend_regime_phase

    return out
