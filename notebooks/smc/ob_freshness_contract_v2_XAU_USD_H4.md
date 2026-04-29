# OB Freshness Contract V2 — XAU_USD H4

- `ob_is_fresh_strict`: untouched since activation.
- `ob_is_fresh_display`: still active, not fully consumed, and within 2 ATR of price.

- Baseline endpoint strict fresh count: `0`.
- Baseline endpoint display fresh count: `1`.

- This split resolves the earlier endpoint ambiguity: strict freshness is ontology/lifecycle truth, display freshness is inventory monitorability truth.