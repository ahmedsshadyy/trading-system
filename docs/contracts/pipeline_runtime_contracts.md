# Pipeline Runtime Contracts

This document freezes the runtime-facing contracts for the current indicator stack. The intent is to preserve causal semantics while allowing incremental execution, frontier rewrites, and cache-aware validation.

## Build Live Output Contract

- Entry point: `src.indicators.pipelines.build_live.build_live_indicators`
- Runtime entry point: `src.indicators.pipelines.build_live.run_live_pipeline`
- Input columns: `timestamp`, `open`, `high`, `low`, `close`, and canonical or normalizable volume input.
- Output contract: same row count and timestamp order as input, plus live-safe feature columns only.
- Causal rule: every output at row `t` must depend only on information available through row `t`.
- Persistence expectation: ordinary incremental runs may only rewrite the mutable frontier implied by the replay window.

## Build Research Output Contract

- Entry point: `src.indicators.pipelines.build_research.build_research_indicators`
- Runtime entry point: `src.indicators.pipelines.build_research.run_research_pipeline`
- Input columns: same as live.
- Output contract: same row count and timestamp order as input, plus research-only columns when explicitly enabled.
- Causal backbone rule: shared live columns must preserve parity with the live stack for identical inputs and config.
- Research-only rule: retrospective or exploratory columns must not leak into canonical live artifacts.

## Persistence Contract

- All pipeline-managed writes must go through `src.pipeline_runtime.artifact_store`.
- Writes use temp-path then atomic rename semantics.
- Metadata writes must occur after artifact writes succeed.
- Canonical datasets should be partitioned by symbol, timeframe, and monthly frontier by default.
- Closed historical partitions are immutable unless an explicit rebuild, migration, or bug-fix path is invoked.

## Cache And State Contract

- Metadata identity: `symbol`, `timeframe`, `pipeline`.
- Required metadata fields: `last_processed_ts`, `schema_version`, `feature_contract_version`, `input_fingerprint`, `config_fingerprint`, `engine_version`.
- Cache invalidation is content-aware. Fingerprints must include schema/config/upstream context, not only file mtimes.
- No-op runs are valid only when raw input fingerprint, config fingerprint, and versions all match and there are no new bars.

## Indicator Classification

### Class A: bounded-window/stateless rolling

- `atr`
- `ema`
- `adx`
- `rsi`
- `macd`
- `bb_width`
- `body_ratio`
- `rolling_atr_ratio`
- `volume_features`
- `prev_day_hl`
- `prev_week_hl`
- `round_number_flag`
- `intraday_context`
- `volume_profile`

Contract:
- Depends on a finite trailing window.
- Replay window can be bounded explicitly.
- State does not need to be checkpointed beyond the replay slice.

### Class B: stateful causal event engines

- `swings`
- `trend_state`
- `bos`
- `choch`
- `rsi_divergence`
- `fvg_stack`
- `displacement`
- `order_blocks`
- `ob_mitigation`
- `liquidity_sweeps`
- `equal_hl`
- `amd_engine`
- `anchored_vwap`
- live `regime`

Contract:
- Causal semantics are strict, but outputs may depend on active state carried across many bars.
- Incremental runs require bounded replay windows with conservative safety margins.
- Carried state must not be approximated silently.

### Class C: research-only / retrospective

- research `regime` treatment in orchestration
- any future labels or retrospective diagnostics added above the shared live backbone

Contract:
- May be incrementalized only if parity against full rebuild is proven.
- Must never contaminate live outputs or metadata identity.

## Replay Policy Manifest

Current conservative replay defaults are encoded in the pipeline stage specs:

- `atr`: 150 bars
- `ema`: 220 bars
- `swings`: `max(400, swing_window * 30)`
- `trend_state` / `bos` / `choch`: 400 bars
- `fvg_stack` / `ifvg` / `ob` / `ob_mitigation` / `sweeps` / `equal_hl` / `amd_engine`: 300 bars
- `volume_profile`: 140 bars

These are safety margins, not proofs. If a stage is tightened later, parity tests must demonstrate no semantic drift.

## DAG Runtime Contract

- Graph execution is now described through manifest-driven nodes under `src/dag_runtime`.
- A node contract must freeze:
  - node kind
  - semantic class
  - upstream dependencies
  - cache policy
  - replay/frontier policy where relevant
  - validation level
- Node fingerprints must include:
  - graph name
  - node name
  - schema version
  - feature contract version
  - engine version
  - runtime config
  - direct source-input fingerprints
  - explicit upstream node fingerprints
- Report nodes are terminal consumers. They must not be used as upstream dependencies for canonical compute nodes.
- Graph invalidation is dependency-driven. A node reruns only if:
  - its own fingerprint changes, or
  - it is explicitly invalidated, or
  - a downstream target depends on an upstream node whose fingerprint changed

## Built-In Graph Families

- `live_pipeline`
  - source raw frame
  - causal stage chain
  - target is the final live-safe feature frame
- `research_pipeline`
  - source raw frame
  - research stage chain
  - target is the final research feature frame
- `validate_range_boundaries`
  - source raw frame
  - context node
  - lightweight rung debug nodes
  - selected-rung node
  - selected full debug node
  - downstream analytics nodes
  - terminal report node(s)
- `validate_regime`
- `validate_trend_state`
- `validate_sr_levels`

The current DAG rollout intentionally preserves indicator and validator semantics by reusing existing compute helpers as node compute functions rather than rewriting the math.
