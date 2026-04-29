# OB Next Step Decision Tree — XAU_USD H4

```text
IF fresh_count=0 is endpoint-only and fresh-through-time is healthy:
    do not patch freshness semantics
ELSE IF fresh inventory is chronically near-zero because time-to-first-touch is tiny:
    next patch should target birth quality vs touch immediacy
ELSE IF main attrition is pre-acceptance:
    next patch should target the single strictest gate
ELSE IF overlap/redundancy dominates:
    next patch should target inventory pruning / ranking, not candidate count
ELSE:
    preserve ontology and use the mildest audited gate relaxation with the best usability-score delta
```

## Evaluated branch selection
- chronic_fresh_inventory = `True`
- median_fresh_life_bars = `13.500`
- dominant_pre_acceptance_gate = `displacement`
- overlap_major = `False`
- best_audit_only_relaxation_family = `activation`