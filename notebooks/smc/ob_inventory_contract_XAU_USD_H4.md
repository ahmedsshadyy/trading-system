# OB Inventory Contract — XAU_USD H4

## Ranking doctrine
- Downstream should consume ranked inventory, not raw active/fresh counts in isolation.
- Production ranking inputs: monitorability, proximity to price, recency, structural relevance, consumedness, overlap penalty, cross-side conflict penalty.

## Endpoint contract
- raw_active_count: `2`
- raw_fresh_count: `1`
- top_active_distance_atr: `2.693712081915994`
- top_fresh_distance_atr: `2.693712081915994`
- top_inventory_monitorability_score: `0.5099106367716768`
- top_inventory_ids: `396,398`
- top_inventory_side_mix: `bull,bear`
- endpoint_inventory_is_good_bool: `False`

## Anti-misleading warnings
- raw active inventory exists but top inventory is still far from price


## Downstream fields
- nearest_monitorable_bull_ob: `3.007460616966132`
- nearest_monitorable_bear_ob: `2.693712081915994`
- best_bull_ob_score: `0.355946941898365`
- best_bear_ob_score: `0.5099106367716768`
- any_monitorable_ob_within_1atr: `False`
- any_monitorable_ob_within_2atr: `False`

## Score formula
- `score = 0.25*distance + 0.15*recency + 0.10*fresh_remaining + 0.20*structure + 0.10*zone_size + 0.10*unconsumed + 0.05*(1-overlap) + 0.05*(1-conflict)`
- Each component is clipped to `[0,1]` and computed causally from the current bar state only.