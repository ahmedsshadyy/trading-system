# Sweeps Indicator Contract (frozen)

Status: **frozen** as the canonical liquidity-sweep detector. The legacy
``add_liquidity_sweep`` detector has been removed; the ``sweeps_v2``
namespace was promoted to ``src/indicators/smc/sweeps`` and is the only
sweep system going forward.

**Source of truth.** Detection: [src/indicators/smc/sweeps/final_sweeps.py](../../src/indicators/smc/sweeps/final_sweeps.py).
Liquidity ladder: [src/indicators/smc/sweeps/unified_sources.py](../../src/indicators/smc/sweeps/unified_sources.py).
Validation: [src/validation/indicators/canonical_sweeps.py](../../src/validation/indicators/canonical_sweeps.py).
Runner: [scripts/validate_sweeps.py](../../scripts/validate_sweeps.py).

## Purpose

A liquidity sweep is a sparse, structural event detector. It is **not**:

- a continuous active-state zone tracker,
- a context builder,
- a labeler / strategy.

The detector marks the bar where price has both breached a known liquidity
reference level intrabar **and** failed to hold beyond it by close. Its
output is the canonical input for downstream context modules (sweep
selectivity classification, displacement confirmation, range / S-R
maintenance) and for research-only audits like Step 11T / 11U.

## Canonical definition

A liquidity sweep occurs when price breaches a known liquidity reference
level intrabar, then fails to hold beyond it by close.

### Bullish sweep
- Price sweeps **below** a below-price liquidity level.
- ``low ≤ swept_level − breach_tolerance``
- ``close > swept_level`` OR close reclaims sufficiently above
  ``swept_level`` by close.
- Directional interpretation: bullish reversal / liquidity grab.

### Bearish sweep
- Price sweeps **above** an above-price liquidity level.
- ``high ≥ swept_level + breach_tolerance``
- ``close < swept_level`` OR close rejects sufficiently below
  ``swept_level`` by close.
- Directional interpretation: bearish reversal / liquidity grab.

## Causal timing (hard contract)

- Only levels that were **live before or at the current bar's open** are
  eligible. A level produced with delayed confirmation is eligible only
  after its confirm-bar close, never from its origin bar.
- Sweep events become available at ``sweep_confirm_idx`` close. A
  same-bar sweep (rejection satisfied at the breach bar's close) is the
  only case where ``sweep_confirm_idx == sweep_breach_idx``.
- Bar-start price uses ``open`` if available; otherwise the prior close.
  The current bar's close is **never** used to decide whether a level was
  above/below price at the start of the bar.
- Forward-looking diagnostics (displacement / BOS / CHoCH within N bars
  after confirmation) are tagged
  ``sweep_research_followthrough_available`` and must NOT be wired into
  live features.
- Indices respect ``swept_source_idx ≤ sweep_breach_idx ≤ sweep_confirm_idx``.

## Accepted source families

Sources flow from the unified liquidity ladder (``add_unified_liquidity_sources``):

1. ``previous_day_high``
2. ``previous_day_low``
3. ``previous_week_high``
4. ``previous_week_low``
5. ``session_high``
6. ``session_low``
7. ``swing_high``
8. ``swing_low``
9. ``equal_high``
10. ``equal_low``
11. ``resistance``
12. ``support``
13. ``range_high``
14. ``range_low``

A source is rejected when it is inactive, invalidated, retired,
``source_level`` is NaN, ``source_idx`` is NaN, the source is on the
wrong side of bar-start price, ``source_age_bars`` is below the
configured minimum, or the source is farther than the configured
``max_source_distance_atr`` from bar-start price.

## Family priority (default)

Used as the primary tiebreaker when multiple eligible levels are swept on
the same bar in the same direction:

1. ``previous_day_high`` / ``previous_day_low``
2. ``previous_week_high`` / ``previous_week_low``
3. ``range_high`` / ``range_low``
4. ``session_high`` / ``session_low``
5. ``swing_high`` / ``swing_low``
6. ``equal_high`` / ``equal_low``
7. ``resistance`` / ``support``

## Required input columns

OHLC: ``open``, ``high``, ``low``, ``close``.
Volatility: ``atr_14``.
Liquidity ladder: every ``liq_above_l*_*`` and ``liq_below_l*_*`` column
emitted by the unified-sources stage. Optional context: ``session_name``,
``regime_label``, ``vol_ratio``, ``displacement_flag`` /
``displacement_bull`` / ``displacement_bear``, ``trend_state``.

## Output schema

The detector emits the canonical schema below. **All columns are
non-mutable contracts**: removing or renaming any of them is a breaking
change. New columns may be added additively.

### Live-safe core

| Column | Type | Description |
|---|---|---|
| ``sweep_flag`` | int 0/1 | Set on the confirm bar of a confirmed sweep. |
| ``sweep_direction`` | str | "bullish" / "bearish" / "" (no sweep). |
| ``bullish_sweep_flag`` | int 0/1 | True iff a bullish sweep confirmed at this bar. |
| ``bearish_sweep_flag`` | int 0/1 | True iff a bearish sweep confirmed at this bar. Mutually exclusive with ``bullish_sweep_flag``. |
| ``sweep_confirm_idx`` | int | Bar index of the confirm bar. |
| ``sweep_breach_idx`` | int | Bar index of the breach bar; ``≤ sweep_confirm_idx``. |
| ``sweep_event_id`` | int | Unique per (bar, side). |
| ``sweep_source_id`` | int | Cluster id for the swept source on the breach bar. |
| ``sweep_source_cluster_id`` | int | Stable cluster id (same as ``sweep_source_id``). |

### Swept-source metadata (canonical aliases)

Surfaced at the confirm bar — they re-expose the underlying source
metadata as observed at the breach bar.

| Column | Description |
|---|---|
| ``swept_level`` | The price level that was swept (alias of ``sweep_source_level``). |
| ``swept_source_family`` | Source family (alias of ``sweep_primary_family``). |
| ``swept_source_side`` | +1 = above-price (bearish sweep), -1 = below-price (bullish sweep). |
| ``swept_source_strength`` | Source strength as scored by the ladder. |
| ``swept_source_idx`` | Bar index of the bar that produced the swept level (origin). NaN if upstream causality was violated; see ``sweep_origin_idx_upstream_invalid``. |
| ``swept_source_age_bars`` | Bars between source live-start and the breach bar. |
| ``swept_source_timestamp`` | Timestamp resolved through ``swept_source_idx``. |

### Sweep mechanics

| Column | Description |
|---|---|
| ``sweep_breach_atr`` | Penetration distance in ATR at the breach bar (alias of ``penetration_atr``). |
| ``sweep_distance_at_start_atr`` | Distance from bar-start price to ``swept_level`` in ATR (alias of ``sweep_pre_breach_distance_atr``). May be negative when the bar opens through the level — documented non-bug. |
| ``sweep_close_reclaim_atr`` | Reclaim distance in ATR at the confirm bar. Bullish: ``(close − swept_level) / atr_14``. Bearish: ``(swept_level − close) / atr_14``. |
| ``sweep_wick_rejection_ratio`` | Confirm-bar wick fraction on the sweep side (lower wick / range for bullish; upper wick / range for bearish). |
| ``sweep_body_reclaim_ratio`` | Confirm-bar body fraction reclaimed back through ``swept_level``, clipped to [0, 1]. |
| ``sweep_quality_score`` | Composite quality score (see below). |

### Selectivity & dedupe

| Column | Description |
|---|---|
| ``sweep_selectivity_class`` | One of ``micro_interaction_sweep``, ``standard_liquidity_sweep``, ``displacement_confirmed_sweep``, ``tradeable_sweep_candidate``. |
| ``sweep_primary_family`` | Same as ``swept_source_family`` (kept for back-compat). |
| ``sweep_level_rank`` | Within (bar, side) cohort. Always 1 today (the detector enforces a single confirmed event per bar/side). |
| ``sweep_duplicate_group_id`` | Group id for events sharing (confirm_idx, side). Singleton per event today. |

### Diagnostic / quality components

| Column | Description |
|---|---|
| ``sweep_q_source_strength`` | Source-strength contribution to the quality score. |
| ``sweep_q_penetration`` | Penetration contribution. |
| ``sweep_q_rejection`` | Wick / close rejection contribution. |
| ``sweep_q_displacement_followthrough`` | Displacement-followthrough contribution. |
| ``sweep_q_regime_context`` | Regime-context contribution. |
| ``sweep_q_volume_confirmation`` | Volume contribution. |
| ``sweep_q_crowding`` | Crowding penalty. |
| ``sweep_origin_idx_upstream_invalid`` | 1 when the upstream ladder reported an ``origin_idx`` that violated causality and had to be NaN-ed in the canonical alias. **Diagnostic — does not invalidate the sweep itself.** |

### Selectivity classes

1. **standard_liquidity_sweep** — basic valid sweep of an eligible level.
2. **micro_interaction_sweep** — valid but small breach/reclaim.
3. **tradeable_sweep_candidate** — stronger sweep based on frozen
   structural quality (still not a trade signal — strategy logic owns
   trade decisions).
4. **displacement_confirmed_sweep** — sweep with same-bar / immediate-
   context displacement confirmation. Uses **existing** displacement
   indicator outputs only — sweeps does not recompute displacement
   logic.

## Threshold registry

The detector is parameterised through the per-family tables in
[final_sweeps.py](../../src/indicators/smc/sweeps/final_sweeps.py).
``SWEEPS_CANONICAL_THRESHOLDS`` is a read-only metadata mapping that
documents every threshold by its **canonical name** (the contract name
used in this document) and points to the internal constant or per-family
table that supplies its current default value.

| Canonical name | Scope | Default source |
|---|---|---|
| ``breach_tolerance_atr`` | global | 0.0 (exact-touch) |
| ``min_breach_atr`` | per_family + global same-bar floor | ``DEFAULT_FAMILY_MIN_PENETRATION_ATR`` + ``DEFAULT_MIN_PENETRATION_ATR_FOR_SAME_BAR`` |
| ``min_close_reclaim_atr`` | global | encoded as rejection-component threshold (currently 0) |
| ``max_source_distance_atr`` | global | None (not currently gated) |
| ``min_source_age_bars`` | per_family | ``DEFAULT_FAMILY_MIN_AGE_BARS`` |
| ``micro_breach_atr_threshold`` | global | ``DEFAULT_MIN_PENETRATION_ATR_FOR_SAME_BAR`` |
| ``strong_breach_atr_threshold`` | global | ``DEFAULT_STANDARD_EXCEPTIONAL_PENETRATION_ATR`` |
| ``strong_reclaim_atr_threshold`` | global | ``DEFAULT_STANDARD_EXCEPTIONAL_REJECTION_COMPONENT`` |
| ``min_wick_rejection_ratio`` | global | ``DEFAULT_MIN_WICK_PROMINENCE`` |
| ``min_quality_score_tradeable`` | per_class | ``DEFAULT_TRADEABLE_MIN_QUALITY_BY_CLASS`` |

These names are stable; the underlying defaults may be tuned without
breaking the contract.

## Quality score (frozen inputs)

The quality score may use only **causal** current-bar information and
pre-existing source metadata:

- breach size in ATR
- close-reclaim strength
- wick rejection ratio
- source strength
- distance to level at bar start
- source-family priority
- volume confirmation if already provided upstream
- displacement confirmation if already provided upstream

No future outcome metric (MFE / MAE / TP / SL) may enter
``sweep_quality_score``. Period.

## Volume / displacement context

Volume confirmation reads existing ``vol_ratio`` /
``volume_ratio_20`` and is exposed as ``sweep_q_volume_confirmation``.
Displacement confirmation consumes the existing displacement detector
outputs (``displacement_bull`` / ``displacement_bear`` /
``displacement_flag``) and is reflected in
``sweep_is_displacement_confirmed``. Sweeps does **not** rewrite either
indicator.

## Dedupe policy

The detector enforces one confirmed sweep per (confirm_idx, direction)
cohort. ``sweep_level_rank`` is always 1 and
``sweep_duplicate_group_id`` is the event id, surfaced for forward
compatibility should the policy ever relax.

When multiple eligible levels exist on the same bar in the same
direction, ranking priority is:

1. nearest eligible level at bar start
2. higher source strength
3. higher family-priority position (see family priority above)
4. newer level

## Validation report (18 points)

[scripts/validate_sweeps.py](../../scripts/validate_sweeps.py) drives
the canonical pipeline and emits the report defined in
[src/validation/indicators/canonical_sweeps.py](../../src/validation/indicators/canonical_sweeps.py).
The 18 points are:

1. Total sweep count
2. Bullish sweep count
3. Bearish sweep count
4. Counts by source family
5. Counts by selectivity class
6. Counts by ``session_name``
7. Counts by ``regime_label``
8. Counts by ``volume_confirmed`` (derived from
   ``sweep_q_volume_confirmation > 0.5``)
9. Counts by ``displacement_confirmed``
10. Distribution of ``sweep_breach_atr``
11. Distribution of ``sweep_close_reclaim_atr``
12. Distribution of ``sweep_distance_at_start_atr``
13. Percentage of sweeps with valid source metadata
14. Source-family share (%)
15. Schema invariants (alias columns present; no legacy columns)
16. Causality invariants (source ≤ breach ≤ confirm; no out-of-range
    indices; no ambiguous direction labels; mutual exclusivity of
    bullish / bearish flags; no same-bar duplicates)
17. ``future_columns_required`` (must always be empty)
18. ``upstream_origin_idx_invalid_count`` (diagnostic only)

A run is **accepted** iff:

- every alias column is present,
- no legacy ``sweep_high`` / ``sweep_low`` columns are produced,
- every causality counter is zero,
- ``future_columns_required`` is empty.

The diagnostic ``upstream_origin_idx_invalid_count`` is informational —
non-zero values indicate an upstream unified-source ladder issue worth
investigating but do not invalidate the canonical sweep contract; the
canonical alias deliberately NaN-s offending values to preserve causality.

## Known non-bugs

- ``sweep_distance_at_start_atr`` may be negative when the bar opens
  through the level. The signed value is preserved by design.
- ``sweep_close_reclaim_atr`` may be small or negative on a same-bar
  sweep where the close just clears the level — the bracket is the
  validator's domain, not the detector's.
- ``swept_source_idx`` may be NaN even though ``sweep_flag = 1`` when
  the upstream ladder reported a non-causal origin idx; the
  ``sweep_origin_idx_upstream_invalid`` flag tracks this.

## Out of scope

The sweeps indicator does not:

- emit trade entries / exits,
- consume Step 11T / 11U TP-SL outcomes (those are research lenses,
  not detector inputs),
- optimise thresholds based on win rate,
- rewrite the displacement, swings, or unified-liquidity-source
  detectors,
- depend on future-bar confirmation,
- attach machine-learned labels,
- compute profit / loss columns.

If a downstream consumer needs any of those, that work belongs in a
strategy / labelling / research module that reads the canonical sweep
schema — never in the detector itself.
