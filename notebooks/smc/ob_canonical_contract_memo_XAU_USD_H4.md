# OB Canonical Contract Memo

- Canonical source doctrine: `last_opposing_before_displacement_leg`.
- Parent event type: `bos` for production canonical family.
- Activation doctrine: `ob_activate_idx == ob_parent_bos_idx`.
- Geometry doctrine: full source candle wick-to-wick range.
- Displacement is metadata, not a hard existence gate.

## Coverage
- confirmed_bos_count: `638`
- raw_canonical_ob_count: `605`
- qualified_canonical_ob_count: `605`
- coverage_fraction: `0.9482758620689655`

## Inventory
- fraction_with_top_inventory_within_1atr: `0.1472421559897334`
- fraction_with_top_inventory_within_2atr: `0.2754701168089676`
- median_distance_to_top_inventory_atr: `2.446559352719671`
- endpoint_raw_fresh_count_strict: `0`
- endpoint_display_fresh_count: `0`
- Freshness note: strict fresh means untouched since activation; display fresh means strict fresh and within the 2 ATR context-near band.

## Execution
- BOS expectancy: `0.15009761067260363`
- OB first-touch expectancy: `0.09702053686229824`
- OB mean-threshold expectancy: `0.17842323651452283`
- Redundancy verdict: `OB mostly redundant to BOS`

## Non-live equivalence
- matched_count: `1` / `9`