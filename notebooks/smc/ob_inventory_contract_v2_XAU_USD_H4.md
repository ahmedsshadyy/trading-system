# OB Inventory Contract V2 — XAU_USD H4

- Primary endpoint ranking uses `monitorability_score_v3_distance_recency_balance`.
- `monitorability_score_v1 = 0.28*distance + 0.20*freshness + 0.16*recency + 0.14*strength + 0.10*remaining + 0.07*confluence + 0.03*side_context + 0.02*(1-overlap)`.
- `monitorability_score_v2_distance_heavy = 0.50*distance + 0.15*freshness + 0.10*recency + 0.10*strength + 0.10*remaining + 0.05*(1-overlap-consumed)`.
- `monitorability_score_v3_distance_recency_balance = 0.35*distance + 0.25*recency + 0.15*freshness + 0.10*strength + 0.10*remaining + 0.05*confluence`.

- Endpoint warning flags: `['inventory_semantics_ambiguous', 'raw_active_inventory_exists_but_low_monitorability', 'side_mix_conflicted']`.
- Endpoint top inventory ids: `429,431`.