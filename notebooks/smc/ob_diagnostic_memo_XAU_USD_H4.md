# OB Diagnostic Memo — XAU_USD H4

## 1. Birth sparsity diagnosis
- Raw BOS-linked candidate space under exact baseline reconstruction: `638` candidates.
- Final accepted OB count: `398`.
- Rejected for missing displacement qualification: `232`.
- Rejected for geometry/cap reasons: `0`.
- Rejected for missing parent BOS linkage: `0`. Under the current ontology this is effectively zero because the production candidate loop starts from BOS rows.

## 2. Freshness and live inventory diagnosis
- Endpoint fresh count is `0`, but the key question is chronicity.
- Proportion of bars with any fresh OB: `0.462`.
- Median fresh count over time: `0.000`.
- Median fresh life bars: `13.500`.
- This indicates fresh inventory is `chronically scarce` rather than being a clean endpoint-only artifact. In practice the family still has fresh supply on many bars, but the median bar has none.

## 3. Consumption vs birth
- Final accepted count is `398`, while touched fraction is `0.804` and full mitigation fraction is `0.734`.
- This means the family does not only suffer from birth sparsity; accepted inventory is also consumed quickly after activation.

## 4. Overlap and redundancy
- Total overlap pairs: `700`.
- Same-side overlap pairs: `362`.
- Fraction of same-side overlap where the younger zone looks redundant to the older one: `0.000`.
- Overlap therefore appears to be a secondary redundancy source.

## 5. Quality vs sparsity
- Median OB strength is `0.688` and median parent displacement score is `0.794`.
- Sparse inventory is therefore not obviously low-quality by default; the question is whether limited inventory is justified by sufficiently high usability.

## 6. Direct answers to the required questions
1. Raw candidates are not the main problem; the larger attrition comes after BOS-row candidate formation.
2. BOS linkage is not the main attrition source because `rejected_no_parent_bos=0` in the exact reconstruction.
3. Displacement is a major pre-acceptance bottleneck.
4. Geometry is a secondary pre-acceptance bottleneck.
5. Fresh inventory dies quickly enough that birth count alone does not explain usability.
6. `fresh_count=0` is not endpoint-only. The better diagnosis is `chronically scarce`: fresh OBs appear on `0.462` of bars, but the median fresh count is `0.000`.
7. Overlap is present but not dominant as a redundancy source.
8. Active and fresh inventory are low both because accepted OBs are limited at birth and because the median fresh life is only `13.500` bars.
9. Current live inventory quality is moderate-to-good on paper, but live availability is constrained by low fresh coverage and high consumption.
10. Top 3 root causes: `cause_freshness_dies_too_fast`, `cause_displacement_overrestriction`, `cause_retirement_or_terminal_semantics_too_aggressive`.

## 7. Approximation notes
- The promotion funnel is reconstructed from the exact production gate order in `add_ob()` as closely as possible.
- `rejected_no_parent_bos` is structurally near-zero because the production detector starts from BOS rows rather than from a broader pre-BOS source-candle universe.
- `fresh_fraction_of_all_live_obs_over_time` is identical to `fresh_fraction_of_active_over_time` under the current contract because there is no additional non-active live inventory layer.