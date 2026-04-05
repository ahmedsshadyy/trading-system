# Range Boundaries Agent Handoff

This is the exact handoff for the agent responsible for `range_boundaries` logic, diagnostics, and analysis.

The performance/orchestration layer is now frozen. Logic work should continue on top of this contract, not around it.

## Current State

`range_boundaries` is:
- DAG-backed
- target-driven
- parity-gated
- performance-frozen at the orchestration/cache/invalidation layer

This means:
- wrapper commands are stable
- target closures are stable
- report-only flags no longer invalidate upstream compute
- no-drift gates are active

What is *not* frozen:
- the algorithmic internals of the heavy compute path
- deeper optimization inside `collect_range_boundary_debug_tables(...)`
- any future logic changes you intentionally make

## What Was Changed

### 1. The wrapper is now the official interface

Use:
- `scripts/validate_range_boundaries.py`

It is now a thin wrapper over DAG targets.

It supports:
- `--target`
- `--explain`
- `--force`
- `--invalidate-cache`
- `--cleanup-stale`
- `--max-artifact-age-days`
- `--html`
- `--write-csv`
- `--full`
- `--tail-rows`
- `--date-from`

### 2. Stable target contract now exists

Wrapper target map:

- `selection` -> `range_selection_bundle`
- `selected-debug` -> `range_selected_debug`
- `forensics` -> `range_forensics`
- `geometry` -> `range_geometry_audit`
- `active-truth` -> `range_active_truth_audit`
- `coverage` -> `range_coverage_regime_report`
- `ranking` -> `range_ranking_bundle`
- `downstream` -> `range_downstream_usefulness`
- `diagnostics` -> `range_diagnostics_bundle`
- `charts` -> `range_chart_bundle`
- `csv` -> `range_csv_bundle`
- `full` -> `range_validation_bundle`

### 3. Range-boundary graph bundles were formalized

Important bundles now in the DAG:
- `range_selection_bundle`
- `range_analysis_bundle`
- `range_chart_bundle`
- `range_csv_bundle`
- `range_validation_bundle`

These exist so you can run narrow areas without paying for the whole validator.

### 4. Retune behavior is gated

This is important:
- Step 8E-B must not run unless Step 8E-A fails to yield a valid rung

That rule is now part of both:
- parity expectations
- performance expectations

Do not break this without intentionally changing the contract and re-running parity.

### 5. Heavy rung path now has sub-stage profiling

`_run_debug_with_params(...)` now records:
- `debug_collect`
- `pressure_imbalance_legacy`
- `pressure_imbalance_v2`
- `contract_scores`
- `summary_build`

This is observational only.
It must not be treated as logic.

### 6. DAG profiler now carries machine-readable node details

Profiler output now carries:
- per-node total time
- cache hit/miss
- node artifact writes
- node `details`

For range-boundary rung nodes, details now include:
- rung label
- `skipped`
- sub-stage timings

For `range_selected_debug`, details also include:
- `selected_debug_cache_write`

### 7. A real invalidation bug was fixed

Before the freeze pass:
- toggling `--html` or `--write-csv` could invalidate upstream compute

This is now fixed.

Current rule:
- report-only flags must not bust upstream compute fingerprints

If you see upstream recompute after a report-only change, treat it as a regression.

## CLI Commands You Should Use

### Explain before running

Use this first to inspect closure and cache behavior:

```bash
poetry run python scripts/validate_range_boundaries.py --target selection --explain
```

### Selection work

Use when you are changing:
- rung selection logic
- recovery ladder logic
- retune behavior
- selection diagnostics

```bash
poetry run python scripts/validate_range_boundaries.py --target selection
```

### Selected debug work

Use when you need the selected debug frame and summary only:

```bash
poetry run python scripts/validate_range_boundaries.py --target selected-debug
```

### Geometry work

Use when changing geometry diagnostics or geometry review logic:

```bash
poetry run python scripts/validate_range_boundaries.py --target geometry
```

### Active-truth work

Use when changing frozen-vs-refresh or active-truth auditing:

```bash
poetry run python scripts/validate_range_boundaries.py --target active-truth
```

### Downstream work

Use when changing downstream usefulness logic:

```bash
poetry run python scripts/validate_range_boundaries.py --target downstream
```

### Forensics / ranking / diagnostics

```bash
poetry run python scripts/validate_range_boundaries.py --target forensics
poetry run python scripts/validate_range_boundaries.py --target ranking
poetry run python scripts/validate_range_boundaries.py --target diagnostics
```

### Chart-only regeneration

Use after compute cache is warm:

```bash
poetry run python scripts/validate_range_boundaries.py --target charts --html
```

Expected behavior:
- no upstream compute recompute on unchanged inputs

### CSV / memo-only regeneration

Use after compute cache is warm:

```bash
poetry run python scripts/validate_range_boundaries.py --target csv --write-csv
```

Expected behavior:
- no chart nodes
- no upstream compute recompute on unchanged inputs

### Full validation

Use for final full-surface validation:

```bash
poetry run python scripts/validate_range_boundaries.py --target full
```

## Hard Rules For The Logic Agent

### 1. Do not bypass the wrapper

For normal logic iteration, use the wrapper targets above.

Do not:
- manually rebuild the old orchestration path
- treat report generation as the default path

### 2. Do not treat profiling as logic

The profiler additions are observational only.

Do not:
- branch logic on profiler fields
- branch logic on cache-hit status

### 3. Do not widen fingerprints casually

If you add config that only affects reports, keep it report-scoped.

Do not let:
- `html`
- `write_csv`
- output-path-only options

invalidate upstream compute nodes.

### 4. Do not break the no-drift gates

If you intentionally change logic, you must expect parity changes.

If you are not intentionally changing logic, the following must remain unchanged:
- selected rung
- retune usage
- selected summary
- downstream summary
- diagnostics bundle
- report terminal-only behavior

## What Was Verified

The freeze pass verified:

- wrapper command contract behavior
- explain output correctness
- selected rung parity
- retune-used parity
- selected summary parity
- downstream summary parity
- diagnostics bundle parity
- real-data target closure for:
  - `selection`
  - `geometry`
  - `active-truth`
  - `downstream`
  - `charts --html`
  - `csv --write-csv`

Also verified:
- report-only flags no longer trigger upstream compute recompute

## What Is Safe To Work On Now

Safe:
- range-boundary logic itself
- diagnostics logic
- geometry logic
- active-truth logic
- downstream usefulness logic
- ranking logic
- algorithmic optimization inside the heavy compute path

Not safe to casually change without re-freezing:
- target map
- report-only invalidation rules
- wrapper contract
- retune gate contract

## If You Need The Next Performance Layer

The next performance layer is no longer orchestration.
It is algorithmic optimization inside the heavy compute path, mainly:

- `collect_range_boundary_debug_tables(...)`
- any repeated expensive internals inside rung evaluation

That work should happen with:
- `--target selection`
- `--target geometry`
- `--target active-truth`
- `--explain`

so you keep the iteration loop narrow.

## Paste-Ready Summary

`range_boundaries` is now DAG-backed, target-driven, parity-gated, and performance-frozen at the orchestration/cache/invalidation layer. Use `scripts/validate_range_boundaries.py` as the official interface. Stable targets are `selection`, `selected-debug`, `forensics`, `geometry`, `active-truth`, `coverage`, `ranking`, `downstream`, `diagnostics`, `charts`, `csv`, and `full`. Use `--explain` before runs. Step 8E-B retune must not run unless Step 8E-A fails. Report-only flags must not invalidate upstream compute. Sub-stage rung profiling now exists for `debug_collect`, `pressure_imbalance_legacy`, `pressure_imbalance_v2`, `contract_scores`, and `summary_build`. The orchestration layer is frozen; remaining performance work is algorithmic optimization inside the heavy compute path.`
