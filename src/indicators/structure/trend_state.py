# src/indicators/structure/trend_state.py

"""
Causal trend state engine built on canonical swing events.

Separates:

1. strict current structure state
   - ``trend_state`` = +1 / -1 / 0
   - built from RECENT HH+HL or LH+LL evidence only
   - ``0`` means genuinely structurally unresolved at current bar close

2. restrained directional bias
   - ``trend_bias_state`` = +1 / -1 / 0
   - only inherited from the most recent strict state
   - decays during neutral structure
   - dies fast on contradiction or TTL expiry
   - never gets refreshed by partial events while strict state is neutral

Also exposes:
- directional structural pressure / strength
- decomposed confidence components
- transition interpretability helpers

All functions are pure: input DataFrame is never mutated.

Canonical doctrine:

- ``trend_state`` is strict structure, not soft direction or inherited bias
- ``trend_bias_state`` may disagree with ``trend_state`` during neutral or
  weakening structure, but never replaces it
- ``trend_strength`` measures directional evidence magnitude
- ``trend_confidence`` measures trustworthiness / coherence of the assignment
- neutral rows may retain meaningful confidence in neutrality; confidence is
  no longer a categorical side-effect ladder
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns

TREND_COMMIT_ENTRY_MIN = 0.62
TREND_COMMIT_OPPOSITE_MAX = 0.38
TREND_COMMIT_GAP_MIN = 0.18
TREND_DIRECTIONAL_EVIDENCE_HIGH = 0.65
STALE_NEUTRAL_MIN_BARS = 6
STALE_NEUTRAL_COMMIT_GAP_MIN = 0.18
STALE_NEUTRAL_COMMIT_GAP_PERSIST_MIN = 0.12
STALE_NEUTRAL_STRONG_ENV_ADX_MIN = 0.70
STALE_NEUTRAL_STRONG_ENV_SLOPE_MIN = 0.70
STALE_NEUTRAL_STRONG_ENV_CONTINUITY_MIN = 0.70
STALE_NEUTRAL_STRONG_ENV_COMPRESSION_MAX = 0.30


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _state_share(series: pd.Series, idx: int, window: int, current: float) -> float:
    start = max(0, idx - window + 1)
    scoped = series.iloc[start : idx + 1].dropna()
    if scoped.empty:
        return np.nan
    return float(scoped.eq(current).mean())


def _add_trend_transition_contract(out: pd.DataFrame) -> pd.DataFrame:
    result = out.copy()
    state = pd.to_numeric(result["trend_state"], errors="coerce").astype(float)
    n = len(result)

    trend_prev = pd.Series(np.nan, index=result.index, dtype=float)
    enter_bull = np.zeros(n, dtype=np.int8)
    enter_bear = np.zeros(n, dtype=np.int8)
    enter_neutral = np.zeros(n, dtype=np.int8)
    bars_in_state = np.zeros(n, dtype=np.int32)
    persistence_5 = pd.Series(np.nan, index=result.index, dtype=float)
    persistence_20 = pd.Series(np.nan, index=result.index, dtype=float)
    direct_opposite = np.zeros(n, dtype=np.int8)

    last_valid_state = np.nan
    current_run = 0

    for i in range(n):
        current = state.iloc[i]
        if not np.isfinite(current):
            last_valid_state = np.nan
            current_run = 0
            continue

        if np.isfinite(last_valid_state):
            trend_prev.iloc[i] = last_valid_state
            if current != last_valid_state:
                current_run = 1
                if abs(int(current) - int(last_valid_state)) == 2:
                    direct_opposite[i] = 1
            else:
                current_run += 1
        else:
            current_run = 1

        if not np.isfinite(trend_prev.iloc[i]) or current != trend_prev.iloc[i]:
            current_int = int(current)
            if current_int == 1:
                enter_bull[i] = 1
            elif current_int == -1:
                enter_bear[i] = 1
            else:
                enter_neutral[i] = 1

        bars_in_state[i] = int(current_run)
        persistence_5.iloc[i] = _state_share(state, i, 5, current)
        persistence_20.iloc[i] = _state_share(state, i, 20, current)
        last_valid_state = current

    result["trend_prev"] = trend_prev
    result["trend_enter_bullish"] = enter_bull
    result["trend_enter_bearish"] = enter_bear
    result["trend_enter_neutral"] = enter_neutral
    result["bars_in_trend_state"] = bars_in_state
    result["trend_persistence_5"] = persistence_5
    result["trend_persistence_20"] = persistence_20
    result["trend_direct_opposite_flip"] = direct_opposite
    return result


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
    """Build causal structural trend state from swing confirmation events."""
    out = df.copy()

    require_columns(
        out,
        {
            "swing_high_confirm_flag",
            "swing_low_confirm_flag",
            "swing_high_confirm_origin_idx",
            "swing_low_confirm_origin_idx",
            "swing_high_confirm_price",
            "swing_low_confirm_price",
        },
    )

    n = len(out)

    sh_flag = out["swing_high_confirm_flag"].to_numpy(dtype=np.int8)
    sl_flag = out["swing_low_confirm_flag"].to_numpy(dtype=np.int8)

    sh_price = out["swing_high_confirm_price"].to_numpy(dtype=float)
    sl_price = out["swing_low_confirm_price"].to_numpy(dtype=float)

    sh_origin_idx = out["swing_high_confirm_origin_idx"].to_numpy(dtype=float)
    sl_origin_idx = out["swing_low_confirm_origin_idx"].to_numpy(dtype=float)

    atr = get_atr_array(out, atr_length)

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
    ema20_slope_atr_src = (
        out["ema_20_slope_atr"].to_numpy(dtype=float)
        if "ema_20_slope_atr" in out.columns
        else np.full(n, np.nan)
    )
    ema50_slope_atr_src = (
        out["ema_50_slope_atr"].to_numpy(dtype=float)
        if "ema_50_slope_atr" in out.columns
        else np.full(n, np.nan)
    )

    trend_state = np.zeros(n, dtype=np.int8)
    trend_state_age = np.zeros(n, dtype=np.int32)

    trend_bias_state = np.zeros(n, dtype=np.int8)
    trend_bias_age = np.zeros(n, dtype=np.int32)
    trend_bias_score_live = np.zeros(n, dtype=float)

    trend_conf_structure_continuity = np.zeros(n, dtype=float)
    trend_conf_freshness = np.zeros(n, dtype=float)
    trend_conf_event_quality = np.zeros(n, dtype=float)
    trend_conf_persistence = np.zeros(n, dtype=float)
    trend_conf_contradiction_penalty = np.zeros(n, dtype=float)
    trend_conf_neutral_coherence = np.zeros(n, dtype=float)
    trend_confidence = np.zeros(n, dtype=float)
    trend_bias_score = np.zeros(n, dtype=np.int16)
    trend_bull_commit_score = np.zeros(n, dtype=float)
    trend_bear_commit_score = np.zeros(n, dtype=float)
    trend_directional_evidence_score = np.zeros(n, dtype=float)
    trend_commit_gap = np.zeros(n, dtype=float)
    trend_commit_dominant_side = np.zeros(n, dtype=np.int8)
    trend_commit_gap_persist_3 = np.zeros(n, dtype=float)
    trend_bull_dominant_2_of_3 = np.zeros(n, dtype=np.int8)
    trend_bear_dominant_2_of_3 = np.zeros(n, dtype=np.int8)
    trend_bull_commit_override = np.zeros(n, dtype=np.int8)
    trend_bear_commit_override = np.zeros(n, dtype=np.int8)

    trend_bull_ready = np.zeros(n, dtype=np.int8)
    trend_bear_ready = np.zeros(n, dtype=np.int8)

    hh_count = np.zeros(n, dtype=np.int16)
    hl_count = np.zeros(n, dtype=np.int16)
    lh_count = np.zeros(n, dtype=np.int16)
    ll_count = np.zeros(n, dtype=np.int16)

    trend_event_high = np.zeros(n, dtype=np.int8)
    trend_event_low = np.zeros(n, dtype=np.int8)
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

    trend_structure_loss_bull = np.zeros(n, dtype=np.int8)
    trend_structure_loss_bear = np.zeros(n, dtype=np.int8)
    trend_emerging_bull = np.zeros(n, dtype=np.int8)
    trend_emerging_bear = np.zeros(n, dtype=np.int8)
    trend_regime_phase = np.zeros(n, dtype=np.int8)
    trend_bias_inherited_flag = np.zeros(n, dtype=np.int8)
    trend_bias_expired_flag = np.zeros(n, dtype=np.int8)
    trend_bias_contradicted_flag = np.zeros(n, dtype=np.int8)

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
    curr_state_run_age = 0

    curr_bias = 0
    curr_bias_age = 0
    curr_bias_mag = 0.0

    inherited_bias_dir = 0
    inherited_bias_age = 0

    bull_pressure = 0.0
    bear_pressure = 0.0

    prev_strict_state_for_phase = 0

    bias_decay = 0.5 ** (1.0 / max(bias_half_life_bars, 1))
    strength_decay = 0.5 ** (1.0 / max(strength_decay_half_life_bars, 1))

    adx_strength_src = (
        out["adx_strength"].to_numpy(dtype=float)
        if "adx_strength" in out.columns
        else (
            np.clip((out["adx_14"].to_numpy(dtype=float) - 20.0) / 15.0, 0.0, 1.0)
            if "adx_14" in out.columns
            else np.full(n, np.nan)
        )
    )
    ema_slope_strength_src = (
        out["ema_slope_strength"].to_numpy(dtype=float)
        if "ema_slope_strength" in out.columns
        else np.full(n, np.nan)
    )
    compression_score_src = (
        out["compression_score"].to_numpy(dtype=float)
        if "compression_score" in out.columns
        else (
            np.clip(
                1.0 - out["bb_width_pct_rank_100"].to_numpy(dtype=float),
                0.0,
                1.0,
            )
            if "bb_width_pct_rank_100" in out.columns
            else np.full(n, np.nan)
        )
    )
    structure_continuity_src = (
        out["structure_continuity"].to_numpy(dtype=float)
        if "structure_continuity" in out.columns
        else np.full(n, np.nan)
    )

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

        # High-side structure: HH / LH
        if sh_flag[i] == 1 and np.isfinite(sh_price[i]):
            rel = _cmp(sh_price[i], prev_swing_high, tol_i)

            fallback_move = (
                abs(sh_price[i] - prev_swing_high)
                if np.isfinite(prev_swing_high)
                else np.nan
            )

            origin_idx = int(sh_origin_idx[i]) if np.isfinite(sh_origin_idx[i]) else -1

            base_strength = np.nan
            if 0 <= origin_idx < n:
                base_strength = sh_strength_src[origin_idx]
                if not np.isfinite(base_strength):
                    base_strength = sh_prom_atr_src[origin_idx]

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

        # Low-side structure: HL / LL
        if sl_flag[i] == 1 and np.isfinite(sl_price[i]):
            rel = _cmp(sl_price[i], prev_swing_low, tol_i)

            fallback_move = (
                abs(sl_price[i] - prev_swing_low)
                if np.isfinite(prev_swing_low)
                else np.nan
            )

            origin_idx = int(sl_origin_idx[i]) if np.isfinite(sl_origin_idx[i]) else -1

            base_strength = np.nan
            if 0 <= origin_idx < n:
                base_strength = sl_strength_src[origin_idx]
                if not np.isfinite(base_strength):
                    base_strength = sl_prom_atr_src[origin_idx]

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

        bull_structure_score = _clip01(
            min(curr_hh, curr_hl) / max(min_confirmations_per_side + 1, 2)
        )
        bear_structure_score = _clip01(
            min(curr_lh, curr_ll) / max(min_confirmations_per_side + 1, 2)
        )
        bull_recent_component = _recent(
            cur_last_hh_idx, i, event_freshness_bars
        ) or _recent(cur_last_hl_idx, i, event_freshness_bars)
        bear_recent_component = _recent(
            cur_last_lh_idx, i, event_freshness_bars
        ) or _recent(cur_last_ll_idx, i, event_freshness_bars)

        if np.isfinite(latest_bull_pair_idx):
            bull_pair_age = max(i - int(latest_bull_pair_idx), 0)
            bull_freshness_score = _clip01(
                1.0 - (bull_pair_age / max(event_freshness_bars, 1))
            )
        else:
            bull_freshness_score = 0.0

        if np.isfinite(latest_bear_pair_idx):
            bear_pair_age = max(i - int(latest_bear_pair_idx), 0)
            bear_freshness_score = _clip01(
                1.0 - (bear_pair_age / max(event_freshness_bars, 1))
            )
        else:
            bear_freshness_score = 0.0

        bull_strength_support = _clip01(max(raw_strength, 0.0))
        bear_strength_support = _clip01(max(-raw_strength, 0.0))
        bull_pressure_score = _clip01(bull_pressure / max(strength_norm_cap, 1e-9))
        bear_pressure_score = _clip01(bear_pressure / max(strength_norm_cap, 1e-9))
        bull_ema_context = _clip01(
            0.65 * max(ema20_slope_atr_src[i], 0.0)
            + 0.35 * max(ema50_slope_atr_src[i], 0.0)
        )
        bear_ema_context = _clip01(
            0.65 * max(-ema20_slope_atr_src[i], 0.0)
            + 0.35 * max(-ema50_slope_atr_src[i], 0.0)
        )
        bull_bias_context = (
            _clip01(curr_bias_mag) if curr_bias == 1 or inherited_bias_dir == 1 else 0.0
        )
        bear_bias_context = (
            _clip01(curr_bias_mag)
            if curr_bias == -1 or inherited_bias_dir == -1
            else 0.0
        )
        bull_context_support = _clip01(
            0.70 * bull_ema_context + 0.30 * bull_bias_context
        )
        bear_context_support = _clip01(
            0.70 * bear_ema_context + 0.30 * bear_bias_context
        )
        bull_event_score = (high_event_score if high_evt == 1 else 0.0) + (
            low_event_score if low_evt == 1 else 0.0
        )
        bear_event_score = (high_event_score if high_evt == -1 else 0.0) + (
            low_event_score if low_evt == -1 else 0.0
        )
        bull_event_quality = max(
            bull_pressure_score,
            _clip01(bull_event_score / max(strength_event_cap, 1e-9)),
        )
        bear_event_quality = max(
            bear_pressure_score,
            _clip01(bear_event_score / max(strength_event_cap, 1e-9)),
        )
        bull_contradiction_penalty = _clip01(
            0.55 * float(bear_recent_component)
            + 0.35 * bear_strength_support
            + 0.20 * float(event_dir == -1)
        )
        bear_contradiction_penalty = _clip01(
            0.55 * float(bull_recent_component)
            + 0.35 * bull_strength_support
            + 0.20 * float(event_dir == 1)
        )
        bull_commit_score = _clip01(
            0.25 * bull_structure_score
            + 0.15 * bull_freshness_score
            + 0.15 * bull_strength_support
            + 0.15 * bull_event_quality
            + 0.20 * bull_context_support
            + 0.10 * (1.0 - bull_contradiction_penalty)
        )
        bear_commit_score = _clip01(
            0.25 * bear_structure_score
            + 0.15 * bear_freshness_score
            + 0.15 * bear_strength_support
            + 0.15 * bear_event_quality
            + 0.20 * bear_context_support
            + 0.10 * (1.0 - bear_contradiction_penalty)
        )
        directional_evidence_score = _clip01(
            0.60 * max(bull_commit_score, bear_commit_score)
            + 0.25 * _clip01(abs(raw_strength))
            + 0.15
            * max(
                bull_pressure_score,
                bear_pressure_score,
                bull_context_support,
                bear_context_support,
            )
        )
        commit_gap = abs(bull_commit_score - bear_commit_score)
        if bull_commit_score > bear_commit_score:
            commit_dominant_side = 1
        elif bear_commit_score > bull_commit_score:
            commit_dominant_side = -1
        else:
            commit_dominant_side = 0
        signed_commit_balance = bull_commit_score - bear_commit_score

        window_start = max(0, i - 2)
        prev_balances = (
            trend_bull_commit_score[window_start:i]
            - trend_bear_commit_score[window_start:i]
        )
        balance_window = np.concatenate(
            [prev_balances, np.asarray([signed_commit_balance], dtype=float)]
        )
        prev_dom = trend_commit_dominant_side[window_start:i]
        dom_window = np.concatenate(
            [prev_dom, np.asarray([commit_dominant_side], dtype=np.int8)]
        )
        gap_window = np.abs(balance_window)
        bull_dominant_2_of_3 = int(np.count_nonzero(dom_window == 1) >= 2)
        bear_dominant_2_of_3 = int(np.count_nonzero(dom_window == -1) >= 2)
        bull_gap_persist_3 = float(np.nanmean(np.maximum(balance_window, 0.0)))
        bear_gap_persist_3 = float(np.nanmean(np.maximum(-balance_window, 0.0)))
        commit_gap_persist_3 = float(np.nanmean(gap_window))

        adx_strength_i = (
            float(adx_strength_src[i]) if np.isfinite(adx_strength_src[i]) else 0.0
        )
        ema_slope_strength_i = (
            float(ema_slope_strength_src[i])
            if np.isfinite(ema_slope_strength_src[i])
            else _clip01(
                abs(0.65 * ema20_slope_atr_src[i] + 0.35 * ema50_slope_atr_src[i])
                / 0.25
            )
        )
        structure_continuity_i = (
            float(structure_continuity_src[i])
            if np.isfinite(structure_continuity_src[i])
            else _clip01(max(curr_hh, curr_ll) / 3.0)
        )
        compression_score_i = (
            float(compression_score_src[i])
            if np.isfinite(compression_score_src[i])
            else 1.0
        )
        strong_environment = bool(
            adx_strength_i >= STALE_NEUTRAL_STRONG_ENV_ADX_MIN
            and ema_slope_strength_i >= STALE_NEUTRAL_STRONG_ENV_SLOPE_MIN
            and structure_continuity_i >= STALE_NEUTRAL_STRONG_ENV_CONTINUITY_MIN
            and compression_score_i <= STALE_NEUTRAL_STRONG_ENV_COMPRESSION_MAX
        )

        prev_state = curr_state
        prev_state_run_age = curr_state_run_age

        new_state = 0
        bull_override = False
        bear_override = False
        if bull_ready and not bear_ready:
            new_state = 1
        elif bear_ready and not bull_ready:
            new_state = -1
        else:
            bull_bias_carry = curr_bias == 1 or inherited_bias_dir == 1
            bear_bias_carry = curr_bias == -1 or inherited_bias_dir == -1
            fresh_bearish_contradiction = bool(event_dir == -1 or bear_recent_component)
            fresh_bullish_contradiction = bool(event_dir == 1 or bull_recent_component)
            neutral_run_age_if_stay = prev_state_run_age + 1 if prev_state == 0 else 1

            if (
                bull_commit_score >= TREND_COMMIT_ENTRY_MIN
                and bear_commit_score <= TREND_COMMIT_OPPOSITE_MAX
                and (bull_commit_score - bear_commit_score) >= TREND_COMMIT_GAP_MIN
                and (bull_recent_component or bull_bias_carry)
                and not fresh_bearish_contradiction
                and prev_state != -1
            ):
                new_state = 1
                bull_override = True
            elif (
                bear_commit_score >= TREND_COMMIT_ENTRY_MIN
                and bull_commit_score <= TREND_COMMIT_OPPOSITE_MAX
                and (bear_commit_score - bull_commit_score) >= TREND_COMMIT_GAP_MIN
                and (bear_recent_component or bear_bias_carry)
                and not fresh_bullish_contradiction
                and prev_state != 1
            ):
                new_state = -1
                bear_override = True
            elif (
                prev_state == 0
                and curr_bias == 0
                and neutral_run_age_if_stay >= STALE_NEUTRAL_MIN_BARS
                and strong_environment
                and bull_commit_score > bear_commit_score
                and commit_dominant_side == 1
                and bull_dominant_2_of_3 == 1
                and commit_gap >= STALE_NEUTRAL_COMMIT_GAP_MIN
                and bull_gap_persist_3 >= STALE_NEUTRAL_COMMIT_GAP_PERSIST_MIN
                and event_dir == 1
                and bull_recent_component
                and not fresh_bearish_contradiction
            ):
                new_state = 1
            elif (
                prev_state == 0
                and curr_bias == 0
                and neutral_run_age_if_stay >= STALE_NEUTRAL_MIN_BARS
                and strong_environment
                and bear_commit_score > bull_commit_score
                and commit_dominant_side == -1
                and bear_dominant_2_of_3 == 1
                and commit_gap >= STALE_NEUTRAL_COMMIT_GAP_MIN
                and bear_gap_persist_3 >= STALE_NEUTRAL_COMMIT_GAP_PERSIST_MIN
                and event_dir == -1
                and bear_recent_component
                and not fresh_bullish_contradiction
            ):
                new_state = -1

        if new_state != curr_state:
            trend_state_changed[i] = 1
            trend_state_from[i] = curr_state
            trend_state_to[i] = new_state
            curr_state = new_state
            curr_state_age = 1 if curr_state != 0 else 0
            curr_state_run_age = 1
        else:
            curr_state_age = curr_state_age + 1 if curr_state != 0 else 0
            curr_state_run_age += 1

        prev_bias = curr_bias

        contradiction = False
        expired = False

        if prev_state != 0 and curr_state == 0:
            inherited_bias_dir = prev_state
            inherited_bias_age = 0
            curr_bias = prev_state
            curr_bias_mag = 1.0
            trend_bias_inherited_flag[i] = 1

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
                    trend_bias_contradicted_flag[i] = int(contradiction)
                    trend_bias_expired_flag[i] = int(expired)
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

            elif (
                not bull_ready
                and not bear_ready
                and raw_strength >= emerging_strength_threshold
                and bull_recent_component
                and curr_bias >= 0
            ):
                emerging_bull = 1
                regime_phase = 1

            elif (
                not bull_ready
                and not bear_ready
                and raw_strength <= -emerging_strength_threshold
                and bear_recent_component
                and curr_bias <= 0
            ):
                emerging_bear = 1
                regime_phase = -1

        trend_structure_loss_bull[i] = structure_loss_bull
        trend_structure_loss_bear[i] = structure_loss_bear
        trend_emerging_bull[i] = emerging_bull
        trend_emerging_bear[i] = emerging_bear
        trend_regime_phase[i] = regime_phase

        if curr_state == 1:
            latest_pair_idx = latest_bull_pair_idx
            structure_score = bull_structure_score
            opposite_recent_component = bear_recent_component
            signed_strength_against_state = max(0.0, -raw_strength)
        elif curr_state == -1:
            latest_pair_idx = latest_bear_pair_idx
            structure_score = bear_structure_score
            opposite_recent_component = bull_recent_component
            signed_strength_against_state = max(0.0, raw_strength)
        else:
            latest_pair_idx = np.nan
            structure_score = 0.0
            opposite_recent_component = bull_recent_component and bear_recent_component
            signed_strength_against_state = 0.0

        if np.isfinite(latest_pair_idx):
            pair_age = max(i - int(latest_pair_idx), 0)
            freshness_score = _clip01(1.0 - (pair_age / max(event_freshness_bars, 1)))
        else:
            freshness_score = 0.0

        pressure_score = _clip01(
            max(bull_pressure, bear_pressure) / max(strength_norm_cap, 1e-9)
        )
        event_score_norm = _clip01(event_score / max(strength_event_cap, 1e-9))
        event_quality_score = max(pressure_score, event_score_norm)

        persistence_score = (
            _clip01(curr_state_age / max(min_confirmations_per_side + 3, 4))
            if curr_state != 0
            else 0.0
        )

        contradiction_penalty = _clip01(
            0.5 * float(opposite_recent_component)
            + 0.5 * signed_strength_against_state
            + 0.25 * float(contradiction)
        )

        recent_directional_component = float(
            bull_recent_component or bear_recent_component
        )
        bias_mag_score = _clip01(abs(curr_bias_mag))
        max_commit_score = max(bull_commit_score, bear_commit_score)
        abs_strength_score = _clip01(abs(raw_strength))
        neutral_coherence = (
            _clip01(
                0.25 * (1.0 - abs_strength_score)
                + 0.25 * (1.0 - bias_mag_score)
                + 0.20 * (1.0 - max_commit_score)
                + 0.15 * (1.0 - directional_evidence_score)
                + 0.15 * (1.0 - recent_directional_component)
            )
            if curr_state == 0
            else 0.0
        )

        if curr_state != 0:
            conf = _clip01(
                0.30 * structure_score
                + 0.20 * freshness_score
                + 0.20 * event_quality_score
                + 0.20 * persistence_score
                + 0.10 * (1.0 - contradiction_penalty)
            )
        else:
            neutral_directional_penalty = _clip01(
                0.40 * directional_evidence_score
                + 0.25 * max_commit_score
                + 0.20 * bias_mag_score
                + 0.15 * recent_directional_component
            )
            neutral_conf_raw = (
                0.35 * neutral_coherence
                + 0.20 * (1.0 - contradiction_penalty)
                + 0.15 * (1.0 - abs_strength_score)
                + 0.15 * (1.0 - directional_evidence_score)
                + 0.15 * (1.0 - max_commit_score)
            )
            conf = _clip01(neutral_conf_raw - 0.35 * neutral_directional_penalty)
            conf = min(0.10 + 0.55 * conf, 0.65)

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

        trend_conf_structure_continuity[i] = structure_score
        trend_conf_freshness[i] = freshness_score
        trend_conf_event_quality[i] = event_quality_score
        trend_conf_persistence[i] = persistence_score
        trend_conf_contradiction_penalty[i] = contradiction_penalty
        trend_conf_neutral_coherence[i] = neutral_coherence
        trend_confidence[i] = conf
        trend_bull_commit_score[i] = bull_commit_score
        trend_bear_commit_score[i] = bear_commit_score
        trend_directional_evidence_score[i] = directional_evidence_score
        trend_commit_gap[i] = commit_gap
        trend_commit_dominant_side[i] = commit_dominant_side
        trend_commit_gap_persist_3[i] = commit_gap_persist_3
        trend_bull_dominant_2_of_3[i] = bull_dominant_2_of_3
        trend_bear_dominant_2_of_3[i] = bear_dominant_2_of_3
        trend_bull_commit_override[i] = int(bull_override)
        trend_bear_commit_override[i] = int(bear_override)
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
    out["trend_bias_inherited_flag"] = trend_bias_inherited_flag
    out["trend_bias_expired_flag"] = trend_bias_expired_flag
    out["trend_bias_contradicted_flag"] = trend_bias_contradicted_flag

    out["trend_confidence"] = trend_confidence
    out["trend_conf_structure_continuity"] = trend_conf_structure_continuity
    out["trend_conf_freshness"] = trend_conf_freshness
    out["trend_conf_event_quality"] = trend_conf_event_quality
    out["trend_conf_persistence"] = trend_conf_persistence
    out["trend_conf_contradiction_penalty"] = trend_conf_contradiction_penalty
    out["trend_conf_neutral_coherence"] = trend_conf_neutral_coherence
    out["trend_bull_commit_score"] = trend_bull_commit_score
    out["trend_bear_commit_score"] = trend_bear_commit_score
    out["trend_directional_evidence_score"] = trend_directional_evidence_score
    out["trend_commit_gap"] = trend_commit_gap
    out["trend_commit_dominant_side"] = trend_commit_dominant_side
    out["trend_commit_gap_persist_3"] = trend_commit_gap_persist_3
    out["trend_bull_dominant_2_of_3"] = trend_bull_dominant_2_of_3
    out["trend_bear_dominant_2_of_3"] = trend_bear_dominant_2_of_3
    out["trend_bull_commit_override"] = trend_bull_commit_override
    out["trend_bear_commit_override"] = trend_bear_commit_override
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
    out = _add_trend_transition_contract(out)

    return out
