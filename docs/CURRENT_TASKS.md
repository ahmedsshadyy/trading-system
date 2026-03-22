Finalize the code of src/indicators by enhancing each function inside each file. When done, refactor the code while making sure it is consistent to be put in a cleaner way MVC.

1. Current file working on : SMC , init (probably persistent throughout)
2. Finished : 0
3. Need to work on : ta_core (make sure its alright), trend, momentum,regime,session,value,volatility,volume profile,volume. 
4. Then finalize with SMT as well, which also requires data fetching for several instruments
5. Enhance current strategies


Notes: What I noticed is a calibration issue, not an architectural failure.

In some weakening / neutral zones:

earlier bullish pressure remains alive

opposite signals do not erode it fast enough

so the net score stays positive longer than a human might expect

That usually comes from one or more of these:

1. Decay is slightly too slow

Current:

strength_decay_half_life_bars = 12

On H4, that is fairly sticky.

2. Opposite-side events may not hit hard enough

A weakening structure should sometimes “bleed off” prior pressure faster.

3. Positive carry from earlier strong swings is dominating small opposite evidence

This can actually be logically correct in many cases, but sometimes visually it lingers too long.


Revisit note:

BOS context and CHoCH context are implemented early so the execution plan can continue, but they are NOT final research truth yet.

After the full indicator suite is finalized, especially the remaining SMC detectors and foundation indicators, revisit:

1. `src/indicators/features/bos_context.py`
2. `src/indicators/features/choch_context.py`

Reason:

- some context columns depend on upstream detectors that are still provisional
- context usefulness should be re-evaluated once the full suite is calibrated
- scoring weights and proximity rules should be re-validated after final indicator refinement
