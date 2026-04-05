Range Boundary Interpretation Contract

Canonical meaning, human reading doctrine, plausibility rules, and downstream usage

Step 8 interpretability freeze

0. Purpose

This document freezes the interpretability contract for the range_boundaries subsystem.

Its purpose is to answer, explicitly and permanently:
- what a confirmed range boundary means
- what it does not mean
- how a human should read it on a chart
- how downstream modules are allowed to consume it
- what counts as a plausible confirmed range
- what counts as a bad or weak false positive
- what validator/chart review should look for

This document exists because statistical detector tuning alone is insufficient.
A range-boundary subsystem is only useful if its outputs correspond to a market object that is both:
1. causally well-defined, and
2. interpretable enough that a human or downstream strategy can say why the object matters.

This is therefore a semantic and usage contract, not a code patch.

---

1. Canonical meaning

1.1 What a confirmed range boundary is

A confirmed range boundary is a causally recognized local balance structure whose upper and lower edges are currently credible reference levels for later interaction analysis.

In plain language:
- the market has recently traded in a bounded two-sided way
- both edges have enough recent structural relevance to matter
- the upper and lower edges are worth tracking as live reference prices

A confirmed range is therefore a local balance envelope, not a general truth about the market.

1.2 What the subsystem outputs mean

When range_boundaries emits a confirmed range, it is saying:
- price recently formed a bounded local consolidation structure
- the structure is sufficiently coherent to expose a range_high and range_low
- those edges are now valid to use as boundary sources for later analysis
- future price interaction with these edges may be meaningful for:
- sweep logic
- breach / reclaim logic
- breakout context
- scanner context / conditioning

1.3 What the subsystem is not claiming

A confirmed range does not mean:
- price must remain inside the range
- breakout is unlikely
- this is a macro or high-timeframe range
- the boundaries are “true support/resistance”
- the range is a standalone entry signal
- the detector knows future persistence
- the range is a hidden-state truth model of market regime

The subsystem only claims:
- a local bounded balance object exists
- its edges are currently worth tracking

---

2. Primary role in the architecture

range_boundaries is not a trade signal generator.

Its primary role is to act as a:

2.1 Local balance context indicator

It tells downstream logic that price is currently or recently boxed enough that edge interactions matter.

2.2 Boundary source generator

It emits upper/lower boundary levels that may later be consumed by:
- sweeps
- breach / reclaim logic
- unified source normalization
- scanner context logic

2.3 Compression-state indicator

It marks a local compression / balance condition whose later resolution may matter.

2.4 Event-history object

It creates a lifecycle:
- candidate
- confirmed
- active
- pressured
- reclaimed
- accepted breakout / invalidated / superseded / expired

That lifecycle can matter as much as the static box itself.

---

3. Human reading doctrine

3.1 How a human should read the indicator

A human should read an active confirmed range as:

“Price has recently behaved as if these two edges matter as local balance boundaries. I should care about how price behaves relative to these edges.”

This means the human should look for:
- continued containment
- edge tests
- wick breaches
- close-based breaks
- reclaims
- sweeps
- later resolution

The correct mindset is:
- not “buy because a range exists”
- but “these boundaries are now meaningful local reference points”

3.2 What the chart object should feel like

A good confirmed range should feel like:
- a visually recognizable local box
- with two edges that appear structurally relevant
- not already obviously dead at the moment of confirmation
- worth monitoring afterward

If a box is geometrically neat but its edges do not feel worth tracking after confirmation, it is not a good output for this subsystem.

---

4. Canonical downstream usage doctrine

Downstream modules may consume confirmed range boundaries only in the following ways.

4.1 Allowed usage — source level consumption

Allowed:
- use range_high / range_low as live reference levels
- use the range object as one source family among others
- use lifecycle state, age, strength, and active-state outputs causally

4.2 Allowed usage — context conditioning

Allowed:
- use active range state as local-balance / compression context
- use boundary distance / inside-outside state as context features
- use range lifecycle transitions as contextual features

4.3 Forbidden usage — direct predictive truth claims

Forbidden:
- treating a range as proof that price will stay inside
- treating the box itself as an entry signal without a separate setup rule
- treating range confirmation as directional signal by itself
- using future persistence / later behavior inside canonical live-safe columns

---

5. Plausibility doctrine

This section defines what counts as a plausible confirmed range for this project.

A confirmed range is plausible only if most of the following are true.

5.1 Visual plausibility

A human looking at the local window should be able to say:
- yes, price was genuinely trading between these bounds
- yes, both upper and lower edges look real
- no, this is not just a tiny accidental micro-cluster
- no, this is not just a trend pause being over-labeled as a range

5.2 Boundary plausibility

The upper and lower lines should feel like:
- levels a human would naturally mark as local edges
- not arbitrary rolling extrema with no visual importance

5.3 Timing plausibility

At confirmation time, the box should not feel:
- already broken
- already obviously irrelevant
- already resolved in spirit

A plausible range should still feel alive enough at confirmation that monitoring the edges makes sense.

5.4 Downstream plausibility

A plausible confirmed range should make a human say:
- “If price touches or breaches these edges later, that interaction could matter.”

If the answer is mostly no, the range is not plausible for this system, even if it is mathematically tidy.

---

6. False-positive doctrine

The system must explicitly recognize that not every detected box is a good range-boundary output.

6.1 Weak false positive

A weak false positive is a box that is:
- visually real enough to exist,
- but not meaningful enough to justify downstream attention.

Example:
- a neat local mini-box that confirms and dies immediately

6.2 Strong false positive

A strong false positive is a box that:
- does not visually look like a meaningful local range,
- has edges that do not feel relevant,
- or looks like a trivial trend pause mislabeled as a range

6.3 Important doctrine

A detector can be causally valid and still emit weak false positives.
The interpretability contract exists specifically to filter out the mistake of calling every neat box a useful range source.

---

7. Types of acceptable outputs

This subsystem may emit multiple kinds of visually real ranges, but they must all satisfy the same core meaning: the edges are worth tracking.

Acceptable types include:

7.1 Durable consolidation range
- survives meaningfully after confirmation
- supports later edge interaction
- strong candidate for downstream source usage

7.2 Fragile but still meaningful local balance
- shorter-lived than ideal
- but still visually real
- still has edges worth tracking briefly

7.3 Transitional local compression
- a temporary bounded phase before resolution
- acceptable only if the edges still matter enough to monitor

Unacceptable outputs include:
- trivial one-bar / near-one-bar micro-boxes
- boxes whose edges never seem worth caring about
- boxes confirmed after they are already effectively dead

---

8. Indicator interpretation checklist

For every manually reviewed confirmed range, the reviewer should ask:

8.1 Was price actually boxed?
- Did the market truly behave as though it had upper/lower local bounds?

8.2 Do both edges matter?
- Would I naturally care about both the upper and lower levels on the chart?

8.3 Does confirmation feel timely?
- Was the box still alive when it confirmed?

8.4 Is this more than a trivial micro-pause?
- Does the object feel like a meaningful local balance structure rather than noise?

8.5 Would later edge interaction be worth monitoring?
- If price sweeps / breaks / reclaims one edge later, would that feel like meaningful behavior?

If several answers are “no,” the range is not a good indicator output.

---

9. Statistical plausibility doctrine

Numeric validation must support interpretability, not replace it.

The following numeric behaviors improve plausibility:
- longer survival after confirmation
- nontrivial bars-to-first-breach
- meaningful reclaim behavior
- healthier behavior in ranging contexts than trending contexts
- stable but not overly uniform width / quality distributions
- ranking that does not favor instant-fail boxes over durable ones

The following numeric behaviors reduce plausibility:
- immediate confirm-then-die behavior dominating top-ranked outputs
- one-bar or near-one-bar fragile boxes ranking as strongest
- active family becoming too sparse to matter
- active family becoming so dense that it marks every tiny pause
- viability / strength inversion where durable outputs score below fragile ones

---

10. Visual audit doctrine

The validator must support at least these chart-review buckets:

10.1 Durable plausible ranges

Boxes that are both visually real and downstream-meaningful.

10.2 Fragile but plausible ranges

Boxes that are real but short-lived.

10.3 Weak false positives

Boxes that exist but do not justify downstream attention.

10.4 Strong false positives

Boxes that should not have been promoted at all.

10.5 Missed plausible ranges

Visually obvious local balance structures the detector failed to capture.

This bucketed audit is mandatory because interpretability cannot be judged from aggregate summary stats alone.

---

11. Story-based reading doctrine

Each confirmed range should be readable as a small market-structure story.

The story should be expressible as:
- range candidate formed
- range confirmed after X bars
- width was Y ATR
- upper/lower touch quality was A/B
- confirm happened near center / near edge
- first breach occurred after N bars
- reclaimed M times
- final outcome was accepted breakout / invalidated / superseded / expired

If the story sounds trivial or uninteresting, the output may be mathematically valid but still not useful.

---

12. Interpretability tags for review

For manual audit or audit-only exports, ranges may be tagged as:
- durable_consolidation
- fragile_but_meaningful
- late_confirm_fragile
- micro_box_only
- visually_weak
- good_boundary_source_candidate
- missed_obvious_range

These are audit-only interpretability labels, not canonical live-safe outputs.

They exist to connect statistical tuning to human judgment.

---

13. Canonical doctrine for this project

The meaning of range_boundaries is now frozen as:

Range boundaries are not standalone trade signals. They are causal local-balance reference levels. A plausible confirmed range is one whose edges remain worth tracking after confirmation for sweep, breach, reclaim, or breakout context.

This is the definitive interpretation for Step 8.

---

14. Consequences for detector tuning

Because of this contract:

14.1 A neat box is not automatically a good output

Geometric tidiness alone is insufficient.

14.2 A slightly messy box may be better than a neat but useless one

If its edges are more meaningful and worth tracking, it may be the superior output.

14.3 Coverage and ranking must be judged through interpretability

Sparse but beautiful is not enough if the family becomes unusable.
Dense but noisy is not enough if the edges become meaningless.

14.4 Final tuning target

The subsystem should prefer:
- plausible,
- monitor-worthy,
- locally meaningful range boundaries

over:
- trivial,
- instantly irrelevant,
- tidy-but-useless micro-boxes

---

15. Freeze status

This document freezes:
- the canonical meaning of a confirmed range boundary
- the proper human reading of the indicator
- what counts as plausible
- what counts as a false positive
- how downstream modules are allowed to use the output
- how chart review should be conducted

Future detector changes must be evaluated against this interpretation contract, not just against aggregate summary metrics.

---

16. Final one-line operator doctrine

When looking at a confirmed active range, the correct question is:

“Are these two edges locally meaningful enough that I would care how price interacts with them next?”

If the answer is no, the output is not good enough for this system.

---
