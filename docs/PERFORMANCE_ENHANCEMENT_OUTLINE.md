# Performance Enhancement Outline

This document is the reusable rollout outline for bringing an indicator or validator family up to the same performance standard now reached by `range_boundaries`.

The purpose is not to change logic. The purpose is to freeze orchestration, caching, invalidation, profiling, and command behavior so future logic work happens on top of a stable and fast validation surface.

## Core Rule

Every phase below must preserve:

- indicator math
- validation math
- event timing
- schema meaning
- report meaning
- selected-config / selected-rung decisions

No semantic changes are allowed during the performance pass.

## Step 1: Freeze the validator command contract

Make the validator wrapper the official interface.

Required wrapper shape:
- parse CLI args
- build `GraphRunContext`
- resolve wrapper `--target` to a DAG target or target bundle
- call `execute_graph(...)` or `explain_graph_run(...)`
- print:
  - validator name
  - graph family
  - wrapper target
  - resolved target
  - executed node closure
  - cache summary
  - profiler path
  - artifact paths when applicable

Required flags for a migrated validator:
- `--target`
- `--explain`
- `--force`
- `--invalidate-cache`
- `--cleanup-stale`
- `--max-artifact-age-days`

Optional flags when relevant:
- `--html`
- `--write-csv`
- `--full`
- `--tail-rows`
- `--date-from`

Acceptance gate:
- wrapper contract tests pass
- `--explain` does not execute compute

## Step 2: Convert orchestration into explicit DAG nodes

Move orchestration out of the script and into DAG node families.

Required node categories:
- source/context node
- compute nodes
- aggregate/summary nodes
- selection nodes if a choice is made
- report nodes
- top-level bundle nodes

Rules:
- report nodes must be terminal only
- diagnostic targets must not force unrelated report nodes
- report targets must not force unrelated diagnostic bundles
- nodes must materialize minimal reusable artifacts, not convenience copies of giant frames unless necessary

Acceptance gate:
- script no longer owns orchestration
- target map is explicit and documented

## Step 3: Add no-drift parity before optimization

Before any performance claim is accepted, lock parity.

Required parity surfaces:
- node parity vs legacy helper/direct path
- graph parity vs legacy validator path
- command-contract parity
- explain output correctness

Exactness rules:
- exact equality for discrete fields and decisions
- float tolerance only where already accepted historically

Acceptance gate:
- no-drift gate is green before moving on

## Step 4: Add target-based execution for real work

Split the validator into narrow user-facing targets so logic iteration can run only the needed closure.

Typical targets:
- selection / config choice
- selected-debug
- forensics
- geometry
- active-truth
- downstream usefulness
- charts
- csv
- full

Rules:
- every wrapper target must map to a known DAG target
- every target must have a documented meaning
- every target must have a minimal allowed closure

Acceptance gate:
- target-mode command-contract tests pass

## Step 5: Add profiler evidence at node level

Before optimizing, make runtime evidence machine-readable.

Required profiler output:
- total wall time
- executed node list
- cache hit/miss per node
- artifact writes per node
- bytes written when measurable
- node details for the heavy path

Implementation rule:
- profiler output must be stable JSON
- profiler data must come from the DAG runtime, not ad hoc print statements

Acceptance gate:
- profiler JSON is parseable and stable
- execution evidence can be asserted in tests

## Step 6: Decompose the heavy path into named sub-stages

Do not optimize a monolith blindly.

For the family’s known heavy path, add observational sub-stage profiling inside the compute function.

Examples of sub-stage profiling:
- raw collection
- feature enrichment
- audit enrichment
- summary build
- cache write

Rules:
- profiling must not alter outputs
- sub-stage names must be stable
- sub-stage timings must be emitted in machine-readable form

Acceptance gate:
- bottleneck can be named by node and by sub-stage

## Step 7: Harden real-data target closure

After the profiler exists, prove on real data that each target executes only its intended closure.

Required proof pattern:
- warm cache
- run one target
- assert against `executed_nodes`
- assert against profiler artifact records

Required checks:
- diagnostic targets avoid unrelated bundles
- report targets avoid upstream compute on warm cache
- unchanged rerun is mostly cache hits

Acceptance gate:
- target-mode integration tests pass on representative real data

## Step 8: Fix fingerprint scope and invalidation leaks

This is where many real performance bugs appear.

Typical bug:
- report-only flags such as `html=True` or `write_csv=True` accidentally invalidate upstream compute

Required rule:
- source and compute nodes fingerprint only the config they actually depend on
- report nodes fingerprint only report-relevant config
- bundle nodes fingerprint only the config relevant to their closure

Acceptance gate:
- report-only flags do not trigger upstream recompute

## Step 9: Close the performance freeze gate

A validator family is performance-frozen only when all of these are true:

- command contract is stable
- DAG orchestration is stable
- no-drift parity is green
- target closures are proven on real data
- unchanged rerun is mostly cache hits
- report-only flags do not invalidate upstream compute
- heavy path has named sub-stage evidence

Only after that should deeper algorithmic optimization begin.

## Step 10: Only then do algorithmic optimization

This is the final step, not the first.

Once the orchestration contract is frozen:
- optimize the true bottleneck sub-stage
- keep no-drift parity green during each optimization step
- keep the same wrapper targets and profiler evidence

Rules:
- optimize one heavy path at a time
- rerun no-drift gate after every optimization
- do not change target routing while doing algorithmic optimization unless the routing itself is the bug

## Replication Checklist For The Next Indicator Family

1. Freeze wrapper command contract
2. Move orchestration into DAG nodes
3. Add node and graph parity
4. Add `--target` and `--explain`
5. Add node-level profiler output
6. Add sub-stage profiling to the heavy path
7. Add real-data target-mode tests
8. Fix fingerprint-scope bugs
9. Mark performance freeze complete
10. Start algorithmic optimization only after freeze

## Range Boundaries As The Reference Pattern

`range_boundaries` is now the reference implementation for this outline.

What to copy from it:
- stable wrapper command contract
- explicit target map
- hard no-drift gate
- node-level profiler details
- sub-stage heavy-path profiling
- real-data target closure tests
- report-only invalidation discipline
- freeze decision recorded in docs and memory
