# Validation DAG Migration Status

## Current Wrapper Status

### Fully DAG-backed wrappers

These scripts now act as CLI compatibility wrappers over built-in DAG targets. They no longer own validation orchestration.

| Script | Graph Family | Primary Target | Notes |
| --- | --- | --- | --- |
| `scripts/validate_range_boundaries.py` | `validate_range_boundaries` | `range_validation_bundle` | Phase 1 migration complete and performance freeze complete. Summary, rung selection, downstream analytics, chart packs, CSV bundle, and memos route through DAG nodes. Sub-stage rung profiling, target-closure hardening, cache-scope fixes, and no-drift freeze gates are green. |
| `scripts/validate_regime.py` | `validate_regime` | `regime_validation_bundle` | Live/research context reuse preserved. HTML remains terminal-only. |
| `scripts/validate_trend_state.py` | `validate_trend_state` | `trend_state_validation_bundle` and `trend_state_minimal_overlay_context` | `--minimal` now targets the dedicated minimal-overlay node. |
| `scripts/validate_sr_levels.py` | `validate_sr_levels` | `sr_validation_bundle` | Summary-first default preserved. HTML remains terminal-only. |

### Legacy procedural validators still pending DAG migration

These scripts still own their orchestration path and remain the next migration surface.

| Script | Planned Graph Family | Priority |
| --- | --- | --- |
| `scripts/validate_structure_context.py` | `validate_structure_context` | High |
| `scripts/validate_swings.py` | `validate_swings` | High |
| `scripts/validate_bos.py` | `validate_bos` | High |
| `scripts/validate_bos_context.py` | `validate_bos_context` | High |
| `scripts/validate_choch.py` | `validate_choch` | High |
| `scripts/validate_choch_context.py` | `validate_choch_context` | High |
| `scripts/validate_volatility.py` | `validate_volatility` | Medium |
| `scripts/validate_volume_profile.py` | `validate_volume_profile` | Medium |
| `scripts/validate_volume.py` | `validate_volume` | Medium |
| `scripts/validate_fvg.py` | `validate_fvg` | Medium |
| `scripts/validate_equal_hl.py` | `validate_equal_hl` | Medium |
| `scripts/validate_displacement.py` | `validate_displacement` | Medium |
| `scripts/validate_avwap.py` | `validate_avwap` | Medium |
| `scripts/validate_session.py` | `validate_session` | Medium |
| `scripts/validate_sweeps.py` | `validate_sweeps` | Medium |
| `scripts/validate_wedges.py` | `validate_wedges` | Medium |
| `scripts/validate_fibonacci.py` | `validate_fibonacci` | Medium |
| `scripts/validate_ob.py` | `validate_ob` | Medium |
| `scripts/validate_amd.py` | `validate_amd` | Medium |
| `scripts/validate_indicators.py` | `validate_indicators` | Low |
| `scripts/validate_detectors.py` | `validate_detectors` | Low |

## Migration Conventions

- Existing `scripts/validate_*.py` entrypoints remain the stable user-facing interface.
- DAG nodes own compute, cache reuse, invalidation, and profiling.
- Wrapper scripts should only:
  - parse CLI args
  - build `GraphRunContext`
  - execute the correct DAG target
  - print summaries and artifact paths
- Report generation remains terminal-only.
- No validator migration may alter indicator semantics, event timing, or report meaning.
- Command behavior for migrated validators is governed by `docs/VALIDATION_COMMAND_CONTRACT.md`.

## Range Boundaries Command Targets

`scripts/validate_range_boundaries.py` now uses the following target map:

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

## Range Boundaries Freeze Status

Current status: `performance freeze complete`

Freeze evidence now in place:

- rung sub-stage profiling is emitted in DAG profiler details
- real-data target-mode tests prove minimal closure for:
  - `selection`
  - `geometry`
  - `active-truth`
  - `downstream`
  - `charts --html`
  - `csv --write-csv`
- report-only flags no longer invalidate upstream compute fingerprints
- no-drift parity and wrapper-contract suites remain green

## Next Recommended Order

1. `validate_structure_context.py`
2. `validate_swings.py`
3. `validate_bos.py`
4. `validate_bos_context.py`
5. `validate_choch.py`
6. `validate_choch_context.py`
7. `validate_volatility.py`
8. `validate_volume_profile.py`
9. Remaining medium-priority validators
