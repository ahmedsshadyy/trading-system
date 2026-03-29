# Session Validation Notes

## Purpose

This note freezes the final validation doctrine for the canonical session module in:

- `src/indicators/foundation/session.py`
- `src/validation/indicators/session.py`

It exists so later work on scanner features, LR inputs, and research tables does not silently drift session semantics.

## Frozen Semantics

### Session Identity

Session identity is assigned by **bar start timestamp** only.

Frozen mapping:

- `0` = `Asia`
- `1` = `London`
- `2` = `Overlap`
- `3` = `NY`
- `4` = `Dead`

### Open / Active Window Flags

Window flags use **bar-interval overlap semantics**, not bar-start semantics.

This is intentional.

Reason:

- on H1 and lower, start-vs-overlap usually matches
- on H4 and other coarse bars, bar-start semantics aliases away real overlap with London / NY windows

Frozen rule:

- `session_*` columns use bar-start semantics
- `is_london_open_window`, `is_ny_open_window`, `is_london_active_window`, `is_ny_active_window` use bar-overlap semantics

## Research Combo Codebook

The following research-only columns remain string-valued:

- `r_asia_london_direction_combo_final`
- `r_london_ny_direction_combo_final`
- `r_asia_ny_direction_combo_final`

### Direction codebook

- `-1` = down
- `0` = flat
- `1` = up

### Pair codebook

Interpret strings as `<left_session>_<right_session>`.

Examples:

- `-1_-1` = down_down
- `-1_0` = down_flat
- `-1_1` = down_up
- `0_-1` = flat_down
- `0_0` = flat_flat
- `0_1` = flat_up
- `1_-1` = up_down
- `1_0` = up_flat
- `1_1` = up_up

These columns are research-only and must remain prefixed with `r_`.

## Research Daily Triad

The session research layer also includes a completed-day triad:

- `r_asia_london_ny_direction_triple_final`
- `r_asia_london_ny_direction_triple_label`
- `r_asia_london_active_ny_direction_triple_final`
- `r_asia_london_active_ny_direction_triple_label`

### Meaning

This is the completed UTC-day structure descriptor:

- Asia final direction
- London final direction
- NY final direction

Examples:

- `-1_-1_-1`
- `-1_0_1`
- `1_1_1`

Human-readable label examples:

- `down_down_down`
- `down_flat_up`
- `up_up_up`

### Availability rule

This triad is **research-only**.

It must remain null until NY completes for that UTC day.

The current frozen implementation stamps it once on the **first row after NY completion** for that day.

That makes the triad a completed-day marker rather than a live intraday feature.

### State space

The triad has `3 x 3 x 3 = 27` valid states.

Validator requirements:

- no premature values before NY completion
- all observed values must be inside the 27-state space
- frequency counts must sum to total completed-day markers

## Alternate London-Active Triad

An additional separate research family is also frozen:

- `r_asia_london_active_ny_direction_triple_final`
- `r_asia_london_active_ny_direction_triple_label`

This does **not** replace the original triad.

It exists specifically to study the alternate interpretation where:

- Asia = `00:00–08:00`
- London-active = `08:00–17:00`
- NY = `17:00–22:00`

So:

- the original triad uses London as `08:00–13:00`
- the alternate triad uses London-active as `08:00–17:00`

This is the separate research family for the case where London is taken through overlap until NY starts.

Validator output for this family is:

- `direction_triple_active_london_distribution`

## Audit Boolean Definitions

The session validator emits:

- `annotation_safe`
- `detect_safe`
- `confirm_safe`
- `active_safe`
- `model_safe`
- `research_only_model_safe`

Frozen meanings:

- `annotation_safe=True` iff session exclusivity holds.
- `detect_safe=True` iff previous-session boundary checks pass for Asia, London, and NY.
- `confirm_safe=True` uses the same timing boundary doctrine as `detect_safe` for this module, because session completion becomes known only when the next session-group starts.
- `active_safe=True` iff all live-safe causal checks pass:
  - session exclusivity
  - dead-zone consistency
  - `bars_since_session_open` reset correctness
  - `session_high_so_far` monotonicity
  - `session_low_so_far` monotonicity
  - valid `session_progress_frac`
  - non-negative `bars_remaining_in_session`
  - no Asia final-field leakage before Asia completion
  - previous-session boundary checks
- `model_safe=True` iff `active_safe=True` and:
  - no `r_` columns exist in the live build
  - research/live parity passes for canonical non-`r_` columns
- `research_only_model_safe=False` by definition, because `r_` columns are not allowed in live-safe model inputs

## Why CI Tests Matter

The validator is a human-readable audit.

Pytest remains the hard CI gate.

The session module is not considered frozen unless both are true:

1. validator output is clean
2. pytest assertions cover the same causal claims

## Accepted Flat-State Frequency

Session direction currently uses the frozen deadband:

- up if `session_close_change_from_open_atr > +0.05`
- down if `session_close_change_from_open_atr < -0.05`
- flat otherwise

Observed accepted frequencies from the current validation run:

- H1: `2569 / 65935` flat rows, about `3.90%`
- H4: `1197 / 17983` flat rows, about `6.66%`

These frequencies are not treated as a defect under the current doctrine.

Do not change the deadband casually.

Any future change to the flat-state threshold is a research-schema change and must be treated as a deliberate semantic migration, not a tuning tweak.
