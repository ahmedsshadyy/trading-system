# OB Jane Street Decision Memo — XAU_USD H4

## 1. Baseline diagnosis
- Legacy reference count in the 2026-02-01 to 2026-03-13 span: `9`.
- Live matched count under the current live-safe family: `1`.
- Median activation lag on matched cases: `4.0` bars.
- Median geometry drift ATR on matched cases: `0.11372340303226541`.
- Fraction of bars with active inventory within 1 ATR: `0.07574249646430255`.
- Fraction of bars with fresh inventory within 1 ATR: `0.032004609501859516`.

## 2. Best repair candidate
- Best overall audited variant: `r1_b`.
- Best ontology-oriented audited variant: `r3_b`.
- Expected equivalence recovery fraction: `0.1111111111111111`.
- Expected top inventory within 1 ATR fraction: `0.07579487716725158`.
- Expected median best active monitorability score: `0.5738796355569883`.

## 3. What merges now
- Production-ready now: ranked inventory / monitorability / anti-misleading endpoint contract.
- Diagnostic-only for now: legacy equivalence audit, repair candidate grid, source-freeze comparison variants.
- Deferred pending explicit promotion: any ontology rewrite that changes OB birth or activation behavior.

## 4. Hard no-go items
- No future-looking source substitution.
- No count-driven threshold loosening without improved monitorability and equivalence recovery.
- No fixed refresh heuristics unless ranking and ontology repairs fail to solve the live usability problem.
- No ranking metric that simply rebrands raw count.

## 5. Final decision tree
```text
IF non-live equivalence failure is mostly source substitution
THEN prioritize source-freeze-earlier repair

IF non-live equivalence failure is mostly activation lag with same source
THEN prioritize activation refactor, not source rewrite

IF distance/usefulness remains poor despite ontology repair
THEN prioritize inventory ranking and monitorability contract

IF displacement rejection explains a large share of lost-good-OB cases
THEN run targeted displacement refactor and re-audit

IF mild relaxations increase count but not monitorability
THEN reject count-driven tuning and keep ranking/ontology as primary path

IF fresh endpoint zero is merely endpoint noise but historical fresh presence is adequate
THEN do not overreact by redefining freshness globally

IF fresh endpoint zero reflects true systemic staleness near price
THEN consider bounded relevance refresh only after proving ontology/activation issues are not primary
```

## 6. Endpoint inventory truth
- endpoint_inventory_is_good_bool: `False`
- top_inventory_ids: `396,398`
- top_inventory_monitorability_score: `0.5099106367716768`