# Persistence Audit

## Scope

This note inventories the remaining repository write paths after the runtime/incremental scaffold pass and documents the canonical persistence migration.

## Canonical Dataset Writes

Before this migration, there were no centralized canonical live/research feature dataset writers in the repo. The indicator pipelines produced in-memory dataframes, but canonical feature-store persistence was not routed through a shared artifact manager.

Canonical persistence is now defined as:

- live feature store: `data/features/live/{symbol}/{timeframe}/YYYY-MM.parquet`
- research feature store: `data/features/research/{symbol}/{timeframe}/YYYY-MM.parquet`
- metadata/state: `data/features/_state/{pipeline}/{symbol}/{timeframe}/metadata.json`

Canonical materialization entrypoints:

- `src/indicators/pipelines/build_live.py::materialize_live_features`
- `src/indicators/pipelines/build_research.py::materialize_research_features`

Behavior:

- ordinary no-op rerun: zero canonical writes
- ordinary incremental run: only frontier partition rewritten
- historical closed partitions: immutable during ordinary runs
- force rebuild: full partition rewrite allowed
- metadata update: only after successful partition writes

## Inventory Of Direct Write Paths

### Canonical live output

- Previous status: none centralized
- Current path: `src/indicators/pipelines/build_live.py::materialize_live_features`
- Artifact type: canonical live output
- Prior full-history rewrite behavior: not applicable because no centralized canonical store existed
- Target behavior: monthly partitioned frontier rewrite only

### Canonical research output

- Previous status: none centralized
- Current path: `src/indicators/pipelines/build_research.py::materialize_research_features`
- Artifact type: canonical research output
- Prior full-history rewrite behavior: not applicable because no centralized canonical store existed
- Current behavior:
  - unchanged rerun: no-op
  - changed input: full rebuild through the artifact manager, then partitioned write
- Reason:
  - the current research stack still includes retrospective columns such as research regime dwell/transition fields that are not frontier-safe yet
- Target future behavior: monthly frontier rewrite once retrospective research columns are explicitly split or state-carried safely

### Validation / report / debug outputs

Representative direct write paths intentionally left separate from canonical persistence:

- `src/validation/common/chart_core.py::save_figure_html`
- `scripts/validate_*`
- `scripts/visual_validation.py`
- `scripts/compare_swing_configs.py`
- `scripts/export_fvg_ifvg_forensics.py`
- `scripts/tune_displacement.py`
- `scripts/tune_equal_hl.py`
- `scripts/analyze_displacement_overlap.py`
- `scripts/audit_equal_hl_features.py`

Artifact type:

- HTML charts
- CSV debug tables
- JSON summaries
- forensic/research exports

Current behavior:

- many still rewrite opt-in debug/report artifacts directly
- this is acceptable because they are not canonical feature-store outputs

### Raw / ingestion outputs

- `scripts/fetch_data.py`
- `scripts/load_candles.py`

Artifact type:

- raw source parquet / database ingestion

These are outside the canonical feature-store migration scope.

## Partition Policy

- Partition key: UTC `timestamp` month formatted as `YYYY-MM`
- Immutable history: all partitions strictly earlier than the frontier partition selected by replay start
- Frontier: the month containing `replay_from_ts`, plus any later months covered by the recomputed suffix
- Merge rule: runtime first merges frozen history plus recomputed frontier in-memory, drops duplicate timestamps, sorts by timestamp, then persists only frontier-era partitions
- Recovery rule: partition files are written through temp-path plus atomic rename; metadata advances only after all partition writes succeed

## Intentionally Full-Rebuild-Only Paths

The following remain intentionally full or opt-in heavy:

- canonical research materialization on changed inputs
- validation/debug/report scripts that generate HTML, CSV, or JSON artifacts
- exploratory tuning/forensics exports

Reason:

- current canonical research output still includes retrospective fields that cannot yet be frontier-updated without semantic drift
- these are not canonical feature-store outputs
- they are user-invoked analysis artifacts, not operational live/research stores
- keeping them separate avoids contaminating the canonical persistence contract
