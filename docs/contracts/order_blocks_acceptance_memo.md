# Order Blocks Acceptance Memo

Date: 2026-04-27

## Scope

- Canonical OB core
- Canonical OB mitigation

## Current status

- Core implementation: complete
- Mitigation implementation: complete
- Contract documentation: complete
- Automated validation checks: passing
- Deterministic unit tests: passing
- Historical validation run on `XAU_USD_H4`: completed

## Historical validation snapshot

From `scripts/validate_ob.py` on `XAU_USD_H4`:

- confirmed OB count: 47
- bullish OB count: 38
- bearish OB count: 9
- active count at end of window: 2
- touched fraction: `0.7447`
- partial mitigation fraction: `0.3830`
- full mitigation fraction: `0.6383`
- invalidated fraction: `0.2553`

## Acceptance decision

Not fully frozen yet.

Reason:
- the implementation and automated checks are in place
- the generated validation report is usable
- but the final manual chart-review gate from the spec is still outstanding

## What is already accepted

- BOS-linked ontology
- source / parent BOS / activation separation
- body-to-extreme geometry
- mitigation ordering and timestamp semantics
- dense live-safe active-side exports
- legacy compatibility aliases for existing downstream consumers

## Remaining freeze gate

Manual review of the generated chart report at:

- `notebooks/smc/ob_validation_XAU_USD_H4.html`

Freeze can be upgraded to final once that chart review confirms:

- visually plausible source selection
- visually plausible zone geometry
- sensible strength ordering
- sensible touch / full / invalidation behavior
