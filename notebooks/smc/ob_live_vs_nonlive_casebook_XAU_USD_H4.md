# OB Live vs Nonlive Casebook — XAU_USD H4

Legacy reference is reconstructed from `notebooks/old/detect_05_ob.html`.
Approximation doctrine: the legacy rectangle `x0` is treated as the old source-row proxy and activation proxy because the archived artifact does not expose separate source/activation fields.

## Summary
- reference_nonlive_count: `9`
- live_matched_count: `1`
- unmatched_fraction: `0.889`
- dominant_root_cause_rankings: `['not_representable_under_current_live_ontology', 'source-candle substitution']`

## Matched cases
```text
 legacy_ob_id legacy_side_label legacy_source_timestamp_proxy  matched_live_ob_id                            match_class  source_idx_delta_bars  activation_idx_delta_bars  overlap_fraction  distance_to_price_at_activation_delta
            9              bear     2026-03-12 21:00:00+00:00               395.0 different_source_semantic_substitution                    3.0                        4.0          0.839338                               0.668184
```

## Unmatched cases
```text
 legacy_ob_id legacy_side_label legacy_source_timestamp_proxy                                             match_class                           dominant_root_cause
            5              bear     2026-02-12 02:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            6              bear     2026-02-16 06:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            1              bull     2026-02-19 22:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            2              bull     2026-02-26 10:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            3              bull     2026-02-27 06:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            7              bear     2026-03-02 18:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            8              bear     2026-03-04 22:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
            4              bull     2026-03-09 09:00:00+00:00 unmatched_not_representable_under_current_live_ontology not_representable_under_current_live_ontology
```