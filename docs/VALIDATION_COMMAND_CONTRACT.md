# Validation Command Contract

This document defines the stable command contract for validation commands in this codebase.

## Command Doctrine

- `scripts/validate_*.py` are the official user-facing validation commands.
- `scripts/run_graph.py` is the lower-level DAG runner for debugging, invalidation inspection, and power-user workflows.
- Every migrated validator must map its command surface onto explicit DAG targets or target bundles.
- No validator command may hide which DAG target it executed.

## Mandatory Flags For Migrated Validators

Every migrated validator must support:

- `--target`
- `--explain`
- `--force`
- `--invalidate-cache`
- `--cleanup-stale`
- `--max-artifact-age-days`

## Optional Flags By Validator Type

Validators may also support:

- `--html`
- `--write-csv`
- `--full`
- `--tail-rows`
- `--date-from`

These flags must not change semantic behavior. They only affect:

- target routing
- report generation
- cache invalidation behavior
- plot/report window selection

## Command Result Contract

Every migrated validator command must print:

- validator name
- graph family
- requested wrapper target
- resolved DAG target
- executed node closure
- cache hit/miss summary for executed nodes
- profiler summary path

If the command writes artifacts, it must also print:

- exact artifact paths
- whether artifacts were cache hits or rewritten

If the command is report-disabled, it must explicitly say so.

## Explain Contract

`--explain` must not execute node compute.

It must print, at minimum:

- wrapper target
- resolved DAG target
- node names in dependency order
- node kinds
- upstream dependencies
- cache-hit status
- rerun reason

Allowed rerun reasons:

- `cache-hit`
- `cache-miss`
- `force`
- `invalidate-cache`
- `invalidated-node`
- `source-input`

## Validation Command Classes

### Diagnostic commands

Purpose:
- inspect one area of a validator without paying for the full report bundle

Examples:
- selection
- selected-debug
- forensics
- geometry
- active-truth
- coverage
- ranking
- downstream
- diagnostics

Requirements:
- may not trigger unrelated report nodes
- may not trigger unrelated downstream bundles

### Report commands

Purpose:
- render HTML, CSV, JSON, or memo artifacts

Examples:
- charts
- csv

Requirements:
- report nodes remain terminal only
- report-only reruns must not invalidate upstream compute nodes

### Full validation commands

Purpose:
- execute the full validator bundle for final checks or release-grade validation

Example:
- full

Requirements:
- must use the top-level validation bundle target
- must print the full validation summary and artifact map

## Range Boundaries Target Contract

`scripts/validate_range_boundaries.py` must support these exact wrapper targets:

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

Behavior rules:

- `--html` only enables chart report nodes
- `--write-csv` only enables CSV and memo report nodes
- `--target charts` with `--html` runs only chart closure
- `--target csv` with `--write-csv` runs only CSV/memo closure
- `--target selection` must not compute post-selection analytics
- `--target geometry` must not compute unrelated downstream bundles

## Required Validation After Command-Surface Changes

If a validator wrapper changes, the implementer must run:

1. wrapper target contract tests
2. node parity tests for changed node families
3. graph parity tests for changed bundles
4. unchanged rerun cache-reuse checks
5. report-only rerun checks
6. profiler review showing the expected closure reduction

No wrapper command change is accepted without that set.

## Performance Freeze Checklist

Before a validator family is declared performance-frozen, the implementer must provide:

1. sub-stage profiler evidence for the known heavy path
2. real-data proof that each supported target executes only its intended closure
3. cache-reuse proof on unchanged rerun
4. report-only proof that report flags do not invalidate upstream compute
5. final no-drift parity proof against the legacy helper path

### Range Boundaries Freeze Commands

Use these commands as the canonical validation runbook for `range_boundaries`:

```bash
poetry run python scripts/validate_range_boundaries.py --target selection --explain
poetry run python scripts/validate_range_boundaries.py --target selection
poetry run python scripts/validate_range_boundaries.py --target geometry
poetry run python scripts/validate_range_boundaries.py --target active-truth
poetry run python scripts/validate_range_boundaries.py --target downstream
poetry run python scripts/validate_range_boundaries.py --target charts --html
poetry run python scripts/validate_range_boundaries.py --target csv --write-csv
poetry run python scripts/validate_range_boundaries.py --target full
```

Expected freeze-grade evidence:

- `selection` excludes post-selection analytics
- `geometry` excludes unrelated downstream bundles
- `active-truth` excludes unrelated downstream bundles
- `downstream` excludes chart and CSV bundles
- `charts --html` does not trigger upstream compute on warm cache
- `csv --write-csv` does not trigger chart nodes
- unchanged rerun is mostly cache hits
