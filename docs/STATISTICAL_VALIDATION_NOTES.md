# Statistical Validation Notes

## Purpose

This note is for later research and backtesting phases.

## Revisit Note

`bos_context.py` and `choch_context.py` are being implemented before the entire indicator suite is finalized.

That is acceptable for execution flow, but it means their outputs should currently be treated as provisional research scaffolding rather than final production-grade context truth.

After the remaining SMC and foundation indicators are finalized, revisit both context layers and re-check:
- proximity thresholds
- context feature semantics
- score weights
- forward-diagnostic usefulness
- interaction with finalized upstream detectors

In short:
- proceed with context implementation now
- re-validate context quality after the full indicator library is stabilized

Current BOS and CHoCH validation is mainly:
- visual
- numerical
- schema-driven

That is appropriate for the current stage of the project.
At this stage, the goal is to verify:
- causal timing
- correct event placement
- correct column semantics
- non-degenerate distributions
- visually plausible structure behavior

It is not yet the stage to claim predictive edge from BOS or CHoCH metadata alone.

## What Current Validation Can Prove

With the current validation approach, the useful statistics are mostly descriptive, not predictive.

They help catch:
- schema errors
- missing columns
- broken ranges or impossible values
- degenerate features
- threshold settings that are obviously too loose or too strict
- misleading visuals that appear correct at first glance but are numerically inconsistent

Examples:
- `bos_body_to_range` should stay in `[0, 1]`
- `bos_close_location` should stay in `[0, 1]`
- `bos_displacement_score` should be positive on valid BOS rows
- `bos_source_rank` should stay in `[1, 10]` on event rows
- boolean flags should remain binary and logically consistent
- event counts should not collapse to zero or explode unrealistically

This kind of validation is essential because it protects the semantics of the detector before any attempt is made to infer edge.

## What Current Validation Cannot Prove

At the BOS stage, there are no forward outcome labels attached to the event itself.

That means current BOS validation cannot yet prove:
- that a green/high-score BOS is more likely to continue
- that trend-aligned BOS is statistically better than non-aligned BOS
- that better `bos_source_rank` actually improves trading outcomes
- that `bos_displacement_score` has true predictive value rather than chart appeal

Without forward diagnostics or labels, BOS statistics remain structural diagnostics, not predictive evidence.

This distinction is important.
A feature can be mechanically elegant and visually convincing while still having weak or no predictive value in backtesting.

## When Statistical Validation Becomes Valuable

The real payoff begins after the context layers and forward diagnostics exist.

In particular, once BOS context is implemented with forward-looking research columns such as:
- `bos_hold_1`
- `bos_hold_2`
- `bos_hold_3`
- `bos_hold_5`
- `bos_failed_1`
- `bos_failed_2`
- `bos_failed_3`
- `bos_failed_5`
- `bos_retest_1`
- `bos_retest_3`
- `bos_retest_5`
- `bos_mfe_3_atr`
- `bos_mae_3_atr`
- `bos_mfe_5_atr`
- `bos_mae_5_atr`
- `bos_mfe_10_atr`
- `bos_mae_10_atr`

then BOS event quality can be tested against actual forward behavior.

At that point, the question changes from:

"Does this BOS look strong?"

to:

"Do stronger BOS events measurably behave better in forward price action?"

That is the correct quant question.

## Core Questions to Test Later

Once forward diagnostics exist, test whether:
- higher `bos_displacement_score` improves hold rates
- higher `bos_displacement_score` reduces failure rates
- higher `bos_source_rank` improves continuation quality
- higher `bos_source_prominence_atr` improves continuation quality
- trend-aligned BOS outperforms neutral or counter-trend BOS
- BOS events near wedges, sweeps, FVGs, OBs, or liquidity behave differently
- combinations of features outperform single features

The point is not only to inspect single columns.
The point is to determine whether BOS metadata contains signal that survives contact with forward outcomes.

## Recommended Statistical Validation Framework

When BOS context is ready, validate in layers.

### 1. Univariate bucket tests

Bucket event rows by variables such as:
- `bos_displacement_score`
- `bos_source_rank`
- `bos_source_prominence_atr`
- `bos_source_age`

Example bucket structure:
- low
- medium
- high

Or use quantile buckets:
- bottom 25%
- middle 50%
- top 25%

For each bucket, compare:
- `hold_3` rate
- `hold_5` rate
- `failed_3` rate
- `failed_5` rate
- mean `mfe_5_atr`
- mean `mae_5_atr`

This answers the first practical question:
- do better-looking BOS events actually perform better?

### 2. Group comparisons by regime/context

Split BOS events by context:
- trend-aligned vs non-aligned
- bull vs bear
- fresh source vs older source
- high-rank vs low-rank source
- after sweep vs not after sweep
- after displacement vs not after displacement
- near wedge vs not near wedge

Then compare the same forward diagnostics across groups.

This helps identify whether BOS quality is conditional rather than universal.

### 3. Joint feature interaction checks

Single-feature tests are useful, but many structural features only become meaningful in combination.

Examples:
- high `bos_displacement_score` + trend alignment
- high `bos_displacement_score` + after sweep
- high `bos_source_rank` + after displacement
- fresh source + trend alignment + nearby liquidity event

This matters because many false positives come from features that look strong in isolation but fail when context is poor.

### 4. Confidence intervals

Do not rely only on raw bucket means.

Use bootstrap confidence intervals for:
- hold rates
- failure rates
- mean MFE
- mean MAE
- differences between buckets

Reason:
- some buckets may have small sample sizes
- apparent differences may be noise
- bootstrap intervals provide a practical, robust uncertainty estimate without overly rigid assumptions

If the top displacement bucket has a higher `hold_3` rate than the bottom bucket, but confidence intervals overlap heavily, the apparent edge may not be reliable.

### 5. Stability across time

Do not test only on the full pooled sample.

Check whether observed relationships hold across:
- different date windows
- different volatility regimes
- different instruments later

If BOS quality metrics only work in one narrow time slice, they are not robust enough for deployment.

## Concrete Examples of Useful Future Tests

These are examples of the kind of statistical outputs worth producing later.

### Example A: Displacement score buckets

Question:
- do top-quartile BOS displacement events hold better than bottom-quartile events?

Compare:
- `hold_3`
- `failed_3`
- `mfe_5_atr`
- `mae_5_atr`

Desired outcome:
- stronger displacement -> higher hold rate, higher MFE, lower failure rate

### Example B: Source rank usefulness

Question:
- does a BOS breaking a more prominent recent swing produce better continuation than one breaking a weak swing?

Compare:
- rank `1-3`
- rank `4-7`
- rank `8-10`

This tests whether source ranking is meaningful or just decorative.

### Example C: Trend alignment

Question:
- do trend-aligned BOS events materially outperform neutral/counter-trend BOS events?

Compare:
- hold rates
- failure rates
- excursion

This becomes especially important before CHoCH and BOS are used together in strategy logic.

### Example D: Context interaction

Question:
- does a high-score BOS after a sweep behave differently from a high-score BOS without a sweep?

This can reveal whether raw break quality is insufficient without structural setup context.

## Why This Matters for the Project

The goal is not to produce pretty indicators.
The goal is to produce features that survive backtesting and improve downstream strategy/model quality.

Statistical validation matters because it helps answer:
- which BOS features are actually informative
- which BOS features are redundant
- which BOS features only work conditionally
- which BOS features are visually persuasive but statistically weak

That directly affects:
- scanner quality
- feature engineering quality
- strategy filtering quality
- model input quality
- eventual generalization in walk-forward testing

In short:
- visual validation tells you whether the detector behaves sensibly
- statistical validation tells you whether the detector metadata contains edge

Both are necessary, but they answer different questions.

## Anti-Leakage Reminder

Any predictive validation must respect time direction.

Rules:
- BOS columns themselves remain causal
- forward diagnostics are research-only
- those forward diagnostics must never enter live feature computation directly
- any model training using forward-derived labels must separate labels from causal inputs correctly

Do not let research convenience contaminate live semantics.

## Recommended Implementation Timing

Do not do full predictive statistical validation yet.

Do it after:
1. canonical BOS is finalized
2. canonical CHoCH is finalized
3. BOS context is implemented
4. forward diagnostics exist
5. validation scripts for context layers are stable

Only then will the sample contain enough structural and forward information to test whether BOS metadata has real signal.

## Working Principle

For now:
- visual validation is for semantic correctness
- numerical validation is for schema and sanity

Later:
- statistical validation is for predictive usefulness

That is the correct sequencing for a research-grade pipeline.
