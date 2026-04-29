# Order Blocks Contract

## Repo stance

- BOS and CHoCH are superior to OB as primary structural signals.
- OB remains in the codebase as a live-safe traced execution and research layer.
- There is no requirement to use OB in any strategy when BOS or CHoCH already
  expresses the structural thesis directly.
- Strategy logic should prefer BOS and CHoCH first; OB is optional and
  non-essential.

## Canonical core

- Parent event: every OB is linked to one confirmed BOS row.
- Source doctrine:
  - bullish OB uses the last bearish candle before the parent bullish BOS impulse
  - bearish OB uses the last bullish candle before the parent bearish BOS impulse
- Geometry doctrine:
  - canonical zone is the full source candle range
  - `ob_zone_low = source_low`
  - `ob_zone_high = source_high`
- Activation doctrine:
  - `ob_activate_idx = ob_parent_bos_idx`
  - activation is exactly the parent BOS confirmation close

OB should not be treated as superior to BOS or CHoCH. The main research
question is whether OB retests improve execution relative to entering directly
from BOS or CHoCH, not whether OB should replace them.

## Sparse event-row contract

Canonical OB rows are stamped on activation rows only.

Identity and lineage:
- `ob_id`
- `ob_side`
- `ob_source_idx`
- `ob_source_timestamp`
- `ob_parent_bos_idx`
- `ob_parent_bos_timestamp`
- `ob_activate_idx`
- `ob_activate_timestamp`

Geometry:
- `ob_zone_high`
- `ob_zone_low`
- `ob_zone_mid`
- `ob_zone_height_abs`
- `ob_zone_height_atr`

Source metadata:
- `ob_source_open`
- `ob_source_high`
- `ob_source_low`
- `ob_source_close`
- `ob_source_body_abs`
- `ob_source_body_frac`
- `ob_source_wick_upper_abs`
- `ob_source_wick_lower_abs`

Parent move and quality:
- `ob_parent_bos_side`
- `ob_parent_displacement_score`
- `ob_parent_move_away_atr`
- `ob_parent_bos_quality`
- `ob_strength_raw`
- `ob_strength`
- `ob_quality_tier`

Lifecycle:
- `ob_state`
- `ob_is_active`
- `ob_is_fresh`
- `ob_is_invalidated`
- `ob_is_retired`
- `ob_age_bars`
- `ob_age_since_activation_bars`

## Mitigation contract

Mitigation starts strictly after activation.

Touch and mitigation:
- `ob_has_been_touched`
- `ob_has_partial_mitigation`
- `ob_has_full_mitigation`
- `ob_first_touch_idx`
- `ob_first_partial_mitigation_idx`
- `ob_first_full_mitigation_idx`
- `ob_invalidation_idx`
- `ob_mitigation_penetration_abs`
- `ob_mitigation_penetration_frac`
- `ob_mitigation_penetration_atr`
- `ob_touch_count`
- `ob_mitigation_count`
- `ob_bars_since_first_touch`
- `ob_bars_since_last_touch`
- `ob_midpoint_touch_flag`
- `ob_midpoint_touch_idx`

Invalidation:
- bullish OB invalidates on close below `zone_low - invalidation_buffer_atr * ATR`
- bearish OB invalidates on close above `zone_high + invalidation_buffer_atr * ATR`

## Dense live-safe exports

`add_ob_mitigation()` also emits a dense per-row active-side view so downstream
live features never need to read future-updated milestone stamps:

- `ob_bull_active*`
- `ob_bear_active*`

These dense columns are point-in-time safe. The sparse milestone columns are for
audit, validation, and research.

## Legacy compatibility

The old surface is still emitted:
- `ob_bull`
- `ob_bear`
- `ob_bull_low`
- `ob_bull_high`
- `ob_bear_low`
- `ob_bear_high`
- `ob_width_atr`
- `ob_unmitigated_bull`
- `ob_unmitigated_bear`
- `ob_first_retest`
