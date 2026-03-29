# Regime Detection Notes — Deferred Advanced Research Methods

## Purpose

The current canonical regime layer is an observable, causal, live-safe context
classifier. Its job is to provide stable scanner and sweep context through the
frozen three-state regime ontology:

- `RANGING`
- `TRANSITIONAL`
- `TRENDING`

This canonical regime is not a hidden-state truth model. It is a practical,
auditable environment classifier built from observable live-safe inputs.

The methods in this file are deferred future research directions. They are not
part of the current canonical live-safe regime layer and must not contaminate
the scanner-safe regime contract prematurely.

## Current Doctrine

- The canonical regime is non-directional.
- The canonical regime is stabilized by design; stabilization is part of the
  regime semantics, not post-processing glue.
- The canonical regime remains the base context layer for scanner gating,
  sweep conditioning, downstream feature conditioning, and later
  regime-stratified evaluation.
- Richer regime taxonomies belong in derived layers on top of canonical regime,
  trend/bias, and volatility. They do not replace the base regime.

## Deferred Methods

### Rolling Hurst Exponent

Rolling Hurst exponent can help distinguish:

- persistent / trending behavior when `H` is materially above `0.5`
- mean-reverting behavior when `H` is materially below `0.5`
- ambiguous or noisy behavior around the middle

Practical notes:

- shorter windows such as `50-100` bars reduce lag but increase noise
- smoothing may help but must remain causal
- Hurst should be treated as a research or auxiliary regime feature, not as
  standalone regime truth

### Kalman Filter State Estimation

Kalman-style filters can estimate latent trend or state variables and may be
useful later for smoother regime-state estimation.

Practical notes:

- useful for denoising and latent-state tracking
- can still lag materially in changing environments
- should be treated as future research-grade state estimation, not as the
  current canonical live-safe regime

### Hidden Markov Models

HMMs are natural candidates for probabilistic latent-regime modeling.

Practical notes:

- attractive because they can infer hidden state probabilities
- can become fragile or overfit if the fitting/update doctrine is loose
- if explored later, they must have a strict live-safe fitting and update
  contract
- initial HMM outputs should be research-only and compared against the
  canonical regime rather than replacing it blindly

### Spectral / Dominant-Cycle Methods

Spectral tools such as Goertzel-style dominant-cycle estimation can help
distinguish cyclical from noisy environments.

Practical notes:

- Hurst asks persistence vs mean reversion
- spectral methods ask whether the environment has a stable dominant cycle
- stable dominant period can suggest cyclical regime conditions
- flat or unstable spectrum can suggest noisy or non-cyclical conditions
- a later combination of Hurst plus dominant-cycle stability may be useful

## Practical Caveat

Advanced regime-detection methods are promising but are explicitly deferred
because the immediate goal is a clean, auditable, live-safe regime layer for
sweep context and later strategy feature controllers. Hidden-state and
signal-processing regime methods will be evaluated later as research
enhancements, not allowed to contaminate the current canonical live-safe layer
prematurely.

## Deferred Enhanced Taxonomy

Also deferred from the canonical regime freeze:

- Bull Quiet
- Bull Volatile
- Bear Quiet
- Bear Volatile
- Sideways Quiet
- Sideways Volatile / Chop

These belong in a later derived layer built on top of:

- canonical regime
- `trend_state` / `trend_bias_state`
- volatility-state context

They are not part of the frozen base regime ontology.
