

1. Current file working on : SMC , init (probably persistent throughout)
2. Finished : 0
3. Need to work on : ta_core (make sure its alright), trend, momentum,regime,session,value,volatility,volume profile,volume. 
4. Then finalize with SMT as well, which also requires data fetching for several instruments
5. Enhance current strategies
6. do support & resistance, fibonacci and price action
Sessions and AMD. 
1. need for research: combinations of (up,down) x (Asia,London,NY)
2. need for research: combinations of (A,M,D) x (Asia,London,NY)




Revisit note:

BOS context and CHoCH context are implemented early so the execution plan can continue, but they are NOT final research truth yet.

After the full indicator suite is finalized, especially the remaining SMC detectors and foundation indicators, revisit:

1. `src/indicators/features/bos_context.py`
2. `src/indicators/features/choch_context.py`

Reason:

- some context columns depend on upstream detectors that are still provisional
- context usefulness should be re-evaluated once the full suite is calibrated
- scoring weights and proximity rules should be re-validated after final indicator refinement

