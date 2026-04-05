"""
Canonical market regime context layer.

Regime is a non-directional environment classifier, distinct from:

* ``trend_state``: strict structural direction
* ``trend_bias_state``: inherited directional bias

It answers whether the market is currently:

* ranging / balanced
* transitional / mixed
* trending / expanding

Regime is context, not hidden market truth. It is designed to be:

* observable from live-safe inputs
* causal at row ``t``
* usable for scanner and sweep conditioning
* distinct from directional structure and bias layers

The live-safe contract is score-based and causal. No future information is
used in canonical outputs, and research-only extras are gated behind
``include_research_only=True``.

Canonical semantics:

* raw score winners are computed first for auditability
* stabilization is then applied as part of the canonical regime definition
* the stabilized ``regime`` output is the only downstream contract
* ``raw_regime`` exists for validation and diagnostics, not scanner use
* the canonical live-safe ontology is frozen to exactly:
  * ``0 = RANGING``
  * ``1 = TRANSITIONAL``
  * ``2 = TRENDING``

Downstream doctrine:

* treat ``regime_boundary_flag == 1`` as degraded context
* treat low ``regime_confidence`` as degraded context
* treat very small ``bars_in_regime`` as degraded context

Scanner and sweep consumers should use those fields as caution inputs rather
than assuming every categorical regime assignment is equally stable.

Step 6A freeze doctrine:

* the canonical regime core is now considered frozen
* the current stabilization thresholds are accepted as the freeze baseline
  after the latest validator pass and should not be retuned unless a later
  downstream integration failure forces it
* richer taxonomies belong in derived layers, not this base regime module
* future work may fix bugs or resolve downstream semantic conflicts, but
  should not expand the canonical ontology beyond the frozen three-state model
* advanced regime-detection methods are explicitly deferred to
  ``docs/REGIME_DETECTION_NOTES.md``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.validators import require_columns

REGIME_RANGING = 0
REGIME_TRANSITIONAL = 1
REGIME_TRENDING = 2

REGIME_LABELS = {
    REGIME_RANGING: "RANGING",
    REGIME_TRANSITIONAL: "TRANSITIONAL",
    REGIME_TRENDING: "TRENDING",
}

REGIME_REQUIRED_COLUMNS = {
    "adx_14",
    "bb_width",
    "bb_width_pct_rank_100",
    "ema_20_slope",
    "ema_20_slope_atr",
    "trend_state",
    "trend_confidence",
    "hh_count",
    "ll_count",
    "atr_14",
}

# Step 6A freeze: these stabilization thresholds are the canonical baseline.
# Any future changes must clear the validator baseline and should be treated as
# bug fixes or downstream-conflict resolution, not open-ended regime redesign.
REGIME_ENTER_RANGE_MARGIN = 0.08
REGIME_ENTER_TREND_MARGIN = 0.10
REGIME_EXIT_EXTREME_MARGIN = 0.12
REGIME_DIRECT_EXTREME_JUMP_MARGIN = 0.20
REGIME_MISALIGNED_EXTREME_ENTER_MARGIN = 0.24
REGIME_MISALIGNED_TRANSITION_EXIT_MARGIN = 0.28
REGIME_MIN_HOLD_EXTREME = 3
REGIME_MIN_HOLD_TRANSITIONAL = 1
REGIME_TRANSITIONAL_EXIT_CONFIRM_BARS = 2
REGIME_BOUNDARY_MARGIN = 0.10
REGIME_CAUTION_CONFIDENCE = 0.60
REGIME_CAUTION_BARS = 2


def _clip(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=lower, upper=upper)


def _normalized_trend_confidence(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if bool(valid.ge(0).all() and valid.le(1).all()):
        normalized = values
    else:
        normalized = values / 2.0
    return normalized.clip(lower=0.0, upper=1.0)


def _build_valid_mask(out: pd.DataFrame) -> pd.Series:
    required_numeric = [
        "adx_14",
        "bb_width",
        "bb_width_pct_rank_100",
        "ema_20_slope",
        "ema_20_slope_atr",
        "trend_state",
        "trend_confidence_norm",
        "hh_count",
        "ll_count",
        "atr_14",
    ]
    valid = pd.Series(True, index=out.index, dtype=bool)
    for col in required_numeric:
        valid &= pd.to_numeric(out[col], errors="coerce").notna()
    return valid


def compute_regime_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Derive bounded live-safe regime helper inputs.

    ``trend_confidence`` is expected to be either already normalized to
    ``[0, 1]`` or emitted by the structural engine on the frozen integer
    scale ``{-1, 0, 1, 2}``. In the latter case it is normalized by ``/ 2``.
    """
    require_columns(df, REGIME_REQUIRED_COLUMNS, caller="compute_regime_inputs")
    out = df.copy()

    adx = pd.to_numeric(out["adx_14"], errors="coerce")
    bb_rank = pd.to_numeric(out["bb_width_pct_rank_100"], errors="coerce")
    ema_slope_atr = pd.to_numeric(out["ema_20_slope_atr"], errors="coerce")
    trend_state = pd.to_numeric(out["trend_state"], errors="coerce")
    trend_confidence_norm = _normalized_trend_confidence(out["trend_confidence"])
    hh_count = pd.to_numeric(out["hh_count"], errors="coerce")
    ll_count = pd.to_numeric(out["ll_count"], errors="coerce")

    out["adx_strength"] = _clip((adx - 20.0) / 15.0)
    out["compression_score"] = _clip(1.0 - bb_rank)
    out["ema_slope_strength"] = _clip(ema_slope_atr.abs() / 0.25)
    out["structure_continuity"] = _clip(np.maximum(hh_count, ll_count) / 3.0)
    out["trend_confidence_norm"] = trend_confidence_norm
    out["neutral_structure_penalty"] = trend_state.eq(0).astype(float)
    out["regime_input_ready"] = _build_valid_mask(out).astype("int8")
    return out


def compute_regime_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute frozen regime scores, confidence, and boundary diagnostics."""
    required = {
        "adx_strength",
        "compression_score",
        "ema_slope_strength",
        "structure_continuity",
        "trend_confidence_norm",
        "neutral_structure_penalty",
        "regime_input_ready",
        "trend_state",
    }
    require_columns(df, required, caller="compute_regime_scores")
    out = df.copy()

    ready = pd.to_numeric(out["regime_input_ready"], errors="coerce").fillna(0).eq(1)
    adx_strength = pd.to_numeric(out["adx_strength"], errors="coerce")
    compression_score = pd.to_numeric(out["compression_score"], errors="coerce")
    ema_slope_strength = pd.to_numeric(out["ema_slope_strength"], errors="coerce")
    structure_continuity = pd.to_numeric(out["structure_continuity"], errors="coerce")
    trend_confidence_norm = pd.to_numeric(out["trend_confidence_norm"], errors="coerce")
    neutral_structure_penalty = pd.to_numeric(
        out["neutral_structure_penalty"], errors="coerce"
    )
    trend_state = pd.to_numeric(out["trend_state"], errors="coerce")

    directional_support = _clip(
        trend_state.ne(0).astype(float) * ((trend_confidence_norm - 0.35) / 0.65)
    )

    trend_score = (
        0.30 * adx_strength
        + 0.16 * ema_slope_strength
        + 0.30 * structure_continuity
        + 0.24 * trend_confidence_norm
        + 0.08 * directional_support
        - 0.08 * compression_score
        - 0.08 * neutral_structure_penalty
    ).clip(lower=0.0, upper=1.0)
    range_score = (
        0.45 * compression_score
        + 0.20 * (1.0 - adx_strength)
        + 0.15 * (1.0 - ema_slope_strength)
        + 0.20 * neutral_structure_penalty
        - 0.12 * directional_support
    ).clip(lower=0.0, upper=1.0)
    transition_score = 1.0 - np.maximum(trend_score, range_score)
    mixed_bonus = (trend_score.sub(range_score).abs() < 0.15).astype(float) * 0.15
    transition_score = (transition_score + mixed_bonus).clip(lower=0.0, upper=1.0)

    trend_score = trend_score.where(ready)
    range_score = range_score.where(ready)
    transition_score = transition_score.where(ready)

    confidence = pd.Series(np.nan, index=out.index, dtype=float)
    margin = pd.Series(np.nan, index=out.index, dtype=float)
    if bool(ready.any()):
        ready_scores = np.column_stack(
            [
                range_score.loc[ready].to_numpy(dtype=float),
                transition_score.loc[ready].to_numpy(dtype=float),
                trend_score.loc[ready].to_numpy(dtype=float),
            ]
        )
        top_score = np.max(ready_scores, axis=1)
        second_score = np.partition(ready_scores, -2, axis=1)[:, -2]
        confidence.loc[ready] = top_score
        margin.loc[ready] = top_score - second_score

    strength_bucket = pd.Series(
        pd.array([pd.NA] * len(out), dtype="Int8"), index=out.index
    )
    valid_conf = confidence.notna()
    strength_bucket.loc[valid_conf & confidence.lt(0.45)] = 0
    strength_bucket.loc[valid_conf & confidence.ge(0.45) & confidence.lt(0.60)] = 1
    strength_bucket.loc[valid_conf & confidence.ge(0.60) & confidence.lt(0.75)] = 2
    strength_bucket.loc[valid_conf & confidence.ge(0.75)] = 3

    out["trend_regime_score"] = trend_score
    out["range_regime_score"] = range_score
    out["transition_regime_score"] = transition_score
    out["regime_confidence"] = confidence
    out["regime_margin"] = margin
    out["regime_boundary_flag"] = (ready & margin.lt(REGIME_BOUNDARY_MARGIN)).astype(
        "int8"
    )
    out["regime_strength_bucket"] = strength_bucket
    return out


def _assign_raw_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ready = pd.to_numeric(out["regime_input_ready"], errors="coerce").fillna(0).eq(1)
    scores = np.column_stack(
        [
            pd.to_numeric(out["range_regime_score"], errors="coerce")
            .fillna(-np.inf)
            .to_numpy(dtype=float),
            pd.to_numeric(out["transition_regime_score"], errors="coerce")
            .fillna(-np.inf)
            .to_numpy(dtype=float),
            pd.to_numeric(out["trend_regime_score"], errors="coerce")
            .fillna(-np.inf)
            .to_numpy(dtype=float),
        ]
    )
    regime_idx = np.argmax(scores, axis=1)
    raw_regime = pd.Series(pd.array([pd.NA] * len(out), dtype="Int8"), index=out.index)
    raw_regime.loc[ready] = regime_idx[ready.to_numpy()]
    raw_label = pd.Series(np.full(len(out), None, dtype=object), index=out.index)
    raw_label.loc[ready] = [
        REGIME_LABELS[int(value)] for value in raw_regime.loc[ready].astype(int)
    ]

    out["raw_regime"] = raw_regime
    out["raw_regime_label"] = raw_label
    out["raw_regime_confidence"] = pd.to_numeric(
        out["regime_confidence"], errors="coerce"
    )
    out["raw_regime_margin"] = pd.to_numeric(out["regime_margin"], errors="coerce")
    return out


def _stabilize_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ready = pd.to_numeric(out["regime_input_ready"], errors="coerce").fillna(0).eq(1)
    raw_regime = pd.to_numeric(out["raw_regime"], errors="coerce")
    raw_margin = pd.to_numeric(out["raw_regime_margin"], errors="coerce")
    if "trend_state" in out.columns:
        trend_state = pd.to_numeric(out["trend_state"], errors="coerce")
    else:
        trend_state = pd.Series(np.nan, index=out.index, dtype=float)

    final_regime = pd.Series(
        pd.array([pd.NA] * len(out), dtype="Int8"), index=out.index
    )
    final_label = pd.Series(np.full(len(out), None, dtype=object), index=out.index)
    stabilized = np.zeros(len(out), dtype=np.int8)
    forced_transitional = np.zeros(len(out), dtype=np.int8)
    direct_extreme_jump = np.zeros(len(out), dtype=np.int8)

    prev_final = np.nan
    prev_run = 0
    prev_raw = np.nan

    def _is_misaligned_extreme(candidate_value: int, trend_state_value: float) -> bool:
        if not np.isfinite(trend_state_value):
            return False
        if candidate_value == REGIME_TRENDING:
            return int(trend_state_value) == 0
        if candidate_value == REGIME_RANGING:
            return int(trend_state_value) != 0
        return False

    for i in range(len(out)):
        if not ready.iloc[i]:
            prev_final = np.nan
            prev_run = 0
            prev_raw = np.nan
            continue

        raw_value = raw_regime.iloc[i]
        raw_margin_i = raw_margin.iloc[i]
        trend_state_i = trend_state.iloc[i]
        if not np.isfinite(raw_value):
            prev_final = np.nan
            prev_run = 0
            prev_raw = np.nan
            continue

        raw_value_int = int(raw_value)
        candidate = raw_value_int

        if np.isfinite(prev_final):
            prev_final_int = int(prev_final)
        else:
            prev_final_int = None

        candidate_is_misaligned = _is_misaligned_extreme(candidate, trend_state_i)

        if (
            candidate == REGIME_TRENDING
            and prev_final_int != REGIME_TRENDING
            and raw_margin_i < REGIME_ENTER_TREND_MARGIN
        ):
            candidate = REGIME_TRANSITIONAL
            forced_transitional[i] = 1
        elif (
            candidate == REGIME_RANGING
            and prev_final_int != REGIME_RANGING
            and raw_margin_i < REGIME_ENTER_RANGE_MARGIN
        ):
            candidate = REGIME_TRANSITIONAL
            forced_transitional[i] = 1

        if (
            candidate in (REGIME_RANGING, REGIME_TRENDING)
            and prev_final_int != candidate
            and candidate_is_misaligned
            and raw_margin_i < REGIME_MISALIGNED_EXTREME_ENTER_MARGIN
        ):
            candidate = REGIME_TRANSITIONAL
            forced_transitional[i] = 1

        if prev_final_int is not None and prev_final_int in (
            REGIME_RANGING,
            REGIME_TRENDING,
        ):
            if (
                candidate == REGIME_TRANSITIONAL
                and raw_margin_i < REGIME_EXIT_EXTREME_MARGIN
            ):
                candidate = prev_final_int
            if (
                candidate in (REGIME_RANGING, REGIME_TRENDING)
                and candidate != prev_final_int
            ):
                jump_margin_threshold = REGIME_DIRECT_EXTREME_JUMP_MARGIN
                if candidate_is_misaligned:
                    jump_margin_threshold = max(
                        jump_margin_threshold,
                        REGIME_MISALIGNED_EXTREME_ENTER_MARGIN,
                    )
                if (
                    prev_run < REGIME_MIN_HOLD_EXTREME
                    or raw_margin_i < jump_margin_threshold
                ):
                    candidate = REGIME_TRANSITIONAL
                    forced_transitional[i] = 1
                else:
                    direct_extreme_jump[i] = 1
        elif prev_final_int == REGIME_TRANSITIONAL and candidate in (
            REGIME_RANGING,
            REGIME_TRENDING,
        ):
            exit_margin_threshold = REGIME_DIRECT_EXTREME_JUMP_MARGIN
            if candidate_is_misaligned:
                exit_margin_threshold = max(
                    exit_margin_threshold,
                    REGIME_MISALIGNED_TRANSITION_EXIT_MARGIN,
                )
            if raw_margin_i < exit_margin_threshold and (
                prev_run < REGIME_TRANSITIONAL_EXIT_CONFIRM_BARS
                or not (np.isfinite(prev_raw) and int(prev_raw) == candidate)
            ):
                candidate = REGIME_TRANSITIONAL
                forced_transitional[i] = 1
        elif (
            prev_final_int == REGIME_TRANSITIONAL
            and candidate == REGIME_TRANSITIONAL
            and prev_run < REGIME_MIN_HOLD_TRANSITIONAL
        ):
            candidate = REGIME_TRANSITIONAL

        final_regime.iloc[i] = candidate
        final_label.iloc[i] = REGIME_LABELS[candidate]
        stabilized[i] = int(candidate != int(raw_value))

        if prev_final_int == candidate:
            prev_run += 1
        else:
            prev_run = 1
        prev_final = float(candidate)
        int(forced_transitional[i])
        prev_raw = float(raw_value_int)
    out["regime"] = final_regime
    out["regime_label"] = final_label
    out["regime_stabilized_from_raw"] = stabilized
    out["regime_forced_transitional"] = forced_transitional
    out["regime_direct_extreme_jump"] = direct_extreme_jump
    out["regime_confidence"] = pd.to_numeric(
        out["raw_regime_confidence"], errors="coerce"
    )
    out["regime_margin"] = pd.to_numeric(out["raw_regime_margin"], errors="coerce")
    out["regime_boundary_flag"] = (
        ready
        & (
            pd.to_numeric(out["regime_margin"], errors="coerce").lt(
                REGIME_BOUNDARY_MARGIN
            )
            | pd.to_numeric(out["regime_forced_transitional"], errors="coerce")
            .fillna(0)
            .eq(1)
        )
    ).astype("int8")
    return out


def compute_regime_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Derive regime transition and persistence fields from final categorical state."""
    require_columns(df, {"regime"}, caller="compute_regime_transitions")
    out = df.copy()

    regime = pd.to_numeric(out["regime"], errors="coerce").astype(float)
    n = len(out)
    prev = pd.Series(np.nan, index=out.index, dtype=float)
    changed = np.zeros(n, dtype=np.int8)
    enter_range = np.zeros(n, dtype=np.int8)
    enter_transition = np.zeros(n, dtype=np.int8)
    enter_trend = np.zeros(n, dtype=np.int8)
    bars_in_regime = np.zeros(n, dtype=np.int32)
    persistence_5 = pd.Series(np.nan, index=out.index, dtype=float)
    persistence_20 = pd.Series(np.nan, index=out.index, dtype=float)

    last_valid_regime = np.nan
    current_run = 0

    for i in range(n):
        current = regime.iloc[i]
        if pd.isna(current):
            last_valid_regime = np.nan
            current_run = 0
            continue

        if np.isfinite(last_valid_regime):
            prev.iloc[i] = last_valid_regime
            if current != last_valid_regime:
                changed[i] = 1
                current_run = 1
            else:
                current_run += 1
        else:
            current_run = 1

        if not np.isfinite(prev.iloc[i]) or changed[i] == 1:
            if int(current) == REGIME_RANGING:
                enter_range[i] = 1
            elif int(current) == REGIME_TRANSITIONAL:
                enter_transition[i] = 1
            elif int(current) == REGIME_TRENDING:
                enter_trend[i] = 1

        bars_in_regime[i] = int(current_run)
        last_valid_regime = current

        for window in (5, 20):
            start = max(0, i - window + 1)
            scoped = regime.iloc[start : i + 1].dropna()
            if scoped.empty:
                continue
            share = float(scoped.eq(current).mean())
            if window == 5:
                persistence_5.iloc[i] = share
            else:
                persistence_20.iloc[i] = share

    out["regime_prev"] = prev
    out["regime_changed"] = changed
    out["regime_enter_ranging"] = enter_range
    out["regime_enter_transitional"] = enter_transition
    out["regime_enter_trending"] = enter_trend
    out["bars_in_regime"] = bars_in_regime
    out["regime_persistence_5"] = persistence_5
    out["regime_persistence_20"] = persistence_20
    return out


def add_regime_research_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add research-only regime outcomes and finalized segment diagnostics."""
    out = df.copy()
    regime = pd.to_numeric(out["regime"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    valid = regime.notna()

    out["r_regime_forward_5_return_abs"] = (
        close.shift(-5).div(close).sub(1.0).abs().where(valid)
    )
    out["r_regime_forward_10_return_abs"] = (
        close.shift(-10).div(close).sub(1.0).abs().where(valid)
    )

    returns = close.pct_change()
    realized_vol = np.full(len(out), np.nan, dtype=float)
    returns_arr = returns.to_numpy(dtype=float)
    for i in range(len(out)):
        end = i + 11
        if end > len(out):
            continue
        window = returns_arr[i + 1 : end]
        if np.isfinite(window).all():
            realized_vol[i] = float(np.std(window, ddof=0))
    out["r_regime_realized_vol_10"] = pd.Series(realized_vol, index=out.index).where(
        valid
    )

    dwell_final = np.full(len(out), np.nan, dtype=float)
    transition_type = np.full(len(out), None, dtype=object)
    regime_arr = regime.to_numpy(dtype=float)
    segment_start = None
    last_valid_idx = None
    for i, value in enumerate(regime_arr):
        if not np.isfinite(value):
            if segment_start is not None and last_valid_idx is not None:
                dwell = last_valid_idx - segment_start + 1
                dwell_final[segment_start : last_valid_idx + 1] = float(dwell)
            segment_start = None
            last_valid_idx = None
            continue
        if segment_start is None:
            segment_start = i
            last_valid_idx = i
            continue
        if value != regime_arr[last_valid_idx]:
            dwell = last_valid_idx - segment_start + 1
            dwell_final[segment_start : last_valid_idx + 1] = float(dwell)
            transition_type[i] = (
                f"{REGIME_LABELS[int(regime_arr[last_valid_idx])]}_TO_{REGIME_LABELS[int(value)]}"
            )
            segment_start = i
        last_valid_idx = i
    if segment_start is not None and last_valid_idx is not None:
        dwell = last_valid_idx - segment_start + 1
        dwell_final[segment_start : last_valid_idx + 1] = float(dwell)

    out["r_regime_dwell_final"] = pd.Series(dwell_final, index=out.index).where(valid)
    out["r_regime_transition_type"] = pd.Series(transition_type, index=out.index).where(
        valid
    )
    return out


def add_regime(
    df: pd.DataFrame,
    *,
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add the frozen canonical non-directional regime context layer.

    Required upstream columns:

    * ``adx_14``
    * ``bb_width``
    * ``bb_width_pct_rank_100``
    * ``ema_20_slope``
    * ``ema_20_slope_atr``
    * ``trend_state``
    * ``trend_confidence``
    * ``hh_count``
    * ``ll_count``
    * ``atr_14``

    Optional interpretation support:

    * ``trend_bias_state``
    * ``trend_strength_ema``

    Freeze doctrine:

    * canonical outputs are limited to ``RANGING / TRANSITIONAL / TRENDING``
    * stabilization is part of the canonical semantics
    * richer directional-volatility taxonomies are deferred to derived layers
    * advanced regime-detection research is documented separately and is not
      part of this live-safe builder
    """
    out = compute_regime_inputs(df)
    out = compute_regime_scores(out)
    out = _assign_raw_regime(out)
    out = _stabilize_regime(out)

    ready = pd.to_numeric(out["regime_input_ready"], errors="coerce").fillna(0).eq(1)
    regime = pd.to_numeric(out["regime"], errors="coerce")
    out["regime_is_ranging"] = ready & regime.eq(REGIME_RANGING)
    out["regime_is_transitional"] = ready & regime.eq(REGIME_TRANSITIONAL)
    out["regime_is_trending"] = ready & regime.eq(REGIME_TRENDING)
    out["regime_is_ranging"] = out["regime_is_ranging"].astype("int8")
    out["regime_is_transitional"] = out["regime_is_transitional"].astype("int8")
    out["regime_is_trending"] = out["regime_is_trending"].astype("int8")

    out = compute_regime_transitions(out)

    out["regime_context_caution"] = (
        ready
        & (
            pd.to_numeric(out["regime_boundary_flag"], errors="coerce").fillna(0).eq(1)
            | pd.to_numeric(out["regime_confidence"], errors="coerce").lt(
                REGIME_CAUTION_CONFIDENCE
            )
            | pd.to_numeric(out["bars_in_regime"], errors="coerce").le(
                REGIME_CAUTION_BARS
            )
        )
    ).astype("int8")

    trend_state = pd.to_numeric(out["trend_state"], errors="coerce")
    regime_numeric = pd.to_numeric(out["regime"], errors="coerce")
    trend_alignment = (regime_numeric.eq(REGIME_TRENDING) & trend_state.ne(0)) | (
        regime_numeric.eq(REGIME_RANGING) & trend_state.eq(0)
    )
    out["regime_trend_alignment"] = (ready & trend_alignment).astype("int8")

    if "trend_bias_state" in out.columns:
        bias = pd.to_numeric(out["trend_bias_state"], errors="coerce")
        bias_alignment = (regime_numeric.eq(REGIME_TRENDING) & bias.ne(0)) | (
            regime_numeric.eq(REGIME_RANGING) & bias.eq(0)
        )
        bias_alignment_out = pd.Series(np.nan, index=out.index, dtype=float)
        bias_alignment_out.loc[ready] = bias_alignment.loc[ready].astype(int)
        out["regime_bias_alignment"] = bias_alignment_out
    else:
        out["regime_bias_alignment"] = np.nan

    if include_research_only:
        out = add_regime_research_columns(out)

    return out
