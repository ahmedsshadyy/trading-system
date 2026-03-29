# Working Memory

## Project State

- Repository: `trading-system`
- Current roadmap position: Phase 3 addendum, SMC layer.
- Finished and frozen upstream SMC work before this session pass:
  - FVG and related work complete.
  - Displacement frozen.
  - Equal H/L frozen.
- Current roadmap thread when this memory was written:
  - working through the sweeps roadmap
  - specifically Step 4: finalize session and calendar liquidity references
- Next expected order after sessions:
  - volume-related subjects
  - regime
  - support & resistance

## Core Doctrine

- The system must be live and causal.
- No future leakage is allowed in canonical outputs.
- Live-safe columns may only use information known by bar close at time `t`.
- Research-only columns are allowed, but they must be explicitly separated and never mixed into live-safe defaults.
- Swings are causal with retrace confirmation semantics:
  - a candidate can arise earlier and confirm later
  - structure outputs like BOS and CHoCH are still causal because confirmation is delayed rather than backfilled into live-safe logic
- Development, scanner, feature engineering, model training, and backtesting all need point-in-time-safe semantics.

## End-State Goal

The intended downstream system is:

1. a live scanner detects specific strategy setups from indicators and context
2. the setup is converted into a feature vector for logistic regression
3. the model outputs a probability score
4. if probability passes threshold, broader-view agents are activated
5. if the agents agree, the user is notified
6. the system remains human-in-the-loop and does not place trades autonomously

## Planning Context Used

These external planning documents were reviewed before implementing sessions:

- `docs/CURRENT_TASKS.md`
- `/Users/ahmadshady/Downloads/Session_Plan.txt`
- `/Users/ahmadshady/Downloads/SweepsPlan.md`
- `/Users/ahmadshady/Downloads/indicator_library_missing_improvements_master_addendum.docx`
- `/Users/ahmadshady/Downloads/00_master_outline (1).docx`
- `/Users/ahmadshady/Downloads/01_mvp1 (1).docx`

## Frozen Session Decisions

The canonical session module is now based on the session execution plan:

- UTC-only session logic
- public canonical builders:
  - `add_time_features(df)`
  - `add_session_features(df, atr_col="atr_14", include_research_only=False)`
- frozen session partition:
  - Asia: `00:00 <= t < 08:00`
  - London: `08:00 <= t < 13:00`
  - Overlap: `13:00 <= t < 17:00`
  - NY: `17:00 <= t < 22:00`
  - Dead: `22:00 <= t < 24:00`
- frozen codes:
  - `0 Asia`
  - `1 London`
  - `2 Overlap`
  - `3 NY`
  - `4 Dead`
- open windows:
  - London open: `08:00 <= t < 10:00`
  - NY open: `13:00 <= t < 15:00`
- scanner-active windows:
  - London active: `08:00 <= t < 17:00`
  - NY active: `13:00 <= t < 22:00`
- frozen semantic split:
  - `session_*` identity uses bar-start semantics
  - `is_*_open_window` and `is_*_active_window` use bar-interval overlap semantics
  - this was added to avoid H4 aliasing where coarse bars overlap a window without starting inside it
- session progress is inferred from the modal positive timestamp gap in minutes
- previous completed session summaries are carried forward only after the referenced session closes
- Asia range package is live-safe:
  - current Asia uses only `*_so_far`
  - completed Asia values only appear after Asia closes
- research-only columns are hard-gated behind `include_research_only=True` and use the `r_` prefix

## Session Implementation Completed

### Source files changed

- `src/indicators/foundation/session.py`
- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/indicators/foundation/__init__.py`
- `src/indicators/__init__.py`
- `src/validation/indicators/session.py`
- `scripts/validate_session.py`
- `tests/indicators/foundation/test_session.py`

### Canonical outputs now implemented

- base UTC time features
- canonical session identity and one-hot flags
- London/NY open-window flags
- London/NY active-window flags
- dead-zone flag
- session progress metrics
- current-session running-state metrics
- previous completed Asia/London/NY summaries and ATR-normalized distances
- Asia range live-safe package
- research-only final session fields and direction-combo fields
- research-only daily triad fields:
  - `r_asia_london_ny_direction_triple_final`
  - `r_asia_london_ny_direction_triple_label`
- session grouping now uses UTC day + session family boundaries rather than raw contiguous code runs
- completed-session carry-forward now advances on session-group transitions, which keeps holiday / weekend / early-close handling causal without waiting for a synthetic final bar

### Compatibility note

- a backward-compatible `add_session_classifier()` alias still exists inside `session.py`
- this is only for compatibility with older call sites while the repo transitions
- canonical downstream usage should move to `add_session_features()`

## Verification Completed

- Focused session tests added and passing.
- Legacy session-focused checks inside `tests/test_indicators.py` still pass.
- Weekend-gap regression coverage was added so Friday NY and Sunday NY cannot merge into one running session again.
- Validation artifacts generated:
  - `notebooks/foundation/session_validation_XAU_USD_H1.html`
  - `notebooks/foundation/session_validation_XAU_USD_H4.html`
- Research/live parity check for canonical non-`r_` columns passed in the validator.
- Validation summary now includes hard checks for:
  - session exclusivity
  - reset correctness
  - running high/low monotonicity
  - no Asia-final leakage during active Asia
  - previous-session boundary availability
  - no `r_` columns in live output
  - H4 window-semantics audit comparing bar-start counts vs overlap counts
- Validation summary also includes:
  - session-direction pair distributions
  - completed-day triad distribution
  - triad 27-state space check
  - triad no-premature-values check

## Triad Freeze

- The full Asia/London/NY day-structure triad is research-only.
- It is not a live-safe feature.
- It is stamped once on the first row after NY completes for that UTC day.
- Raw encoding remains string-based:
  - example: `-1_0_1`
- Human-readable label also exists:
  - example: `down_flat_up`

## Commands Run Successfully

- `poetry run pytest tests/indicators/foundation/test_session.py tests/test_indicators.py -k session -q`
- `poetry run python -m py_compile src/indicators/foundation/session.py src/validation/indicators/session.py scripts/validate_session.py src/indicators/pipelines/build_live.py src/indicators/pipelines/build_research.py src/indicators/__init__.py src/indicators/foundation/__init__.py`
- `poetry run python scripts/validate_session.py`

## Important Downstream Notes

- The scanner/model stack should consume the canonical session columns, not the legacy alias columns.
- Research/live parity is now explicit for session features.
- Session outputs are suitable for scanner filtering and LR-safe one-hot/continuous session features.
- The broader roadmap still requires later work on:
  - regime stabilization
  - support/resistance
  - unified liquidity source framework
  - final sweeps rebuild after prerequisites are frozen

# Step 5 Freeze: Volume Contract

## Files touched

- `src/indicators/foundation/volume.py`
- `src/indicators/foundation/volume_profile.py`
- `src/indicators/_helpers/schema.py`
- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/indicators/foundation/__init__.py`
- `src/indicators/__init__.py`
- `src/indicators/research/displacement_research.py`
- `src/validation/indicators/volume.py`
- `src/validation/indicators/volume_profile.py`
- `scripts/validate_volume.py`
- `scripts/validate_volume_profile.py`
- `tests/indicators/foundation/test_volume.py`
- `tests/indicators/foundation/test_volume_profile.py`
- `tests/test_schema_normalization.py`
- `tests/test_displacement_research.py`
- `src/indicators/smc/equal_hl.py`

## Canonical volume doctrine now frozen

- `volume` is the sole internal source of truth.
- Raw `tickVolume` is normalized once at schema level.
- `normalize_candle_schema()` now preserves canonical volume column position when converting from `tickVolume`.
- Downstream modules should consume `add_volume_features()` rather than recomputing local baselines.
- `build_live_indicators()` now uses `add_volume_features(..., include_research_only=False)`.
- `build_research_indicators()` now uses `add_volume_features(..., include_research_only=True)`.
- `displacement_research` no longer derives local `vol_ratio` fallback; if canonical volume features are absent, event-time volume ratio is left `NaN`.

## Public volume builder surface

- New canonical builder:
  - `add_volume_features(df, include_research_only=False, atr_col="atr_14")`
- Lower-level helpers now exist and are frozen:
  - `add_volume_baselines`
  - `add_volume_flags`
  - `add_effort_result_features`
  - `add_delta_proxy_features`
  - `add_vsa_features`
  - `add_wick_effort_features`
- Backward-compatible wrappers still exist:
  - `add_volume_ratio`
  - `add_key_volume_flags`
  - `add_candle_delta_proxy`
  - `add_vsa`
  - `add_wick_ratio`

## Canonical live-safe outputs now implemented

- base volume baselines:
  - `vol_sma_20`
  - `vol_med_20`
  - `vol_std_20`
  - `vol_zscore_20`
  - `vol_ratio`
  - `vol_ratio_med_20`
  - `vol_pct_rank_100`
- participation / abnormality flags:
  - `vol_above_avg`
  - `vol_below_avg`
  - `vol_above_1_5x`
  - `vol_above_2_0x`
  - `vol_below_0_8x`
  - `vol_extreme_pct90`
  - `vol_extreme_pct95`
- volume trend family:
  - `vol_slope_5`
  - `vol_slope_5_norm`
  - `vol_ema_5`
  - `vol_ema_20`
  - `vol_ema_ratio_5_20`
- candle result / spread primitives:
  - `bar_range`
  - `bar_range_atr`
  - `bar_body`
  - `bar_body_frac`
  - `close_pos_in_range`
  - `body_direction`
  - `true_spread_ratio`
- effort/result proxies:
  - `effort_vs_result`
  - `effort_vs_body`
  - `result_vs_effort`
  - `body_result_vs_effort`
- signed delta proxy family:
  - `close_strength`
  - `delta_proxy_raw`
  - `delta_proxy_norm`
  - `delta_proxy_body`
  - `candle_delta_proxy`
- VSA proxy family:
  - `vsa_absorption`
  - `vsa_directional`
  - `vsa_no_demand`
  - `vsa_no_supply`
  - `vsa_climactic_up`
  - `vsa_climactic_down`
  - `vsa_churn`
  - `vsa_effort_failure`
- wick / rejection / effort family:
  - `upper_wick_ratio`
  - `lower_wick_ratio`
  - `dominant_wick_ratio`
  - `wick_imbalance`
  - `upper_rejection_effort`
  - `lower_rejection_effort`
  - `wick_effort_imbalance`

## Research-only volume extras

- `r_vol_forward_1_return`
- `r_vol_forward_3_return`
- `r_vol_forward_5_return`
- `r_post_spike_followthrough_label`
- `r_post_spike_reversal_label`

## Frozen warmup / pathology behavior

- 20-bar family first becomes valid at row index `19`.
- 100-bar percentile-rank first becomes valid at row index `99`.
- 5-bar slope family first becomes valid at row index `4`.
- binary warmup rows remain `0`, not `NaN`.
- zero-volume paths produce no `inf`.
- zero-range paths force:
  - `bar_range = 0`
  - `bar_body_frac = 0`
  - `close_pos_in_range = NaN`
  - wick ratios = `0`
  - `close_strength = 0`
  - delta proxies = `0` when ratio inputs are otherwise valid

## Volume Profile contract now frozen

- `add_volume_profile()` signature is now:
  - `add_volume_profile(df, lookback=80, n_bins=50, atr_col="atr_14", mode="exact")`
- default `mode="exact"` now computes true rolling VP per row.
- current bar is excluded by contract:
  - `vp_*[t]` uses the trailing window ending at `t-1`
- explicit `mode="stepped"` still exists as an approximation option.
- canonical VP outputs now include:
  - `vp_poc`
  - `vp_vah`
  - `vp_val`
  - `vp_poc_distance_atr`
  - `vp_value_width`
  - `vp_value_width_atr`
  - `vp_inside_value_area`
  - `vp_above_vah`
  - `vp_below_val`
  - `vp_distance_to_vah_atr`
  - `vp_distance_to_val_atr`

## Validation / audit additions

- Added dedicated validators:
  - `src/validation/indicators/volume.py`
  - `src/validation/indicators/volume_profile.py`
- Added validator entry scripts:
  - `scripts/validate_volume.py`
  - `scripts/validate_volume_profile.py`
- Volume validator now audits:
  - required columns
  - no-inf status
  - zero-range handling
  - zero-volume handling
  - source parity (`volume` vs `tickVolume`)
  - live/research parity
  - no-label contamination
  - flag value counts
  - summary stats for key volume outputs
- VP validator now audits:
  - POC / VAH / VAL ordering
  - non-negative value width
  - source parity
  - current-bar exclusion
  - total allocated volume consistency
  - value-area coverage >= 70%

## Verification completed

- `poetry run pytest tests/indicators/foundation/test_volume.py tests/indicators/foundation/test_volume_profile.py tests/test_schema_normalization.py tests/test_displacement_research.py tests/test_indicators.py -q`
  - passed: `142 passed`
- `poetry run python -m py_compile src/indicators/foundation/volume.py src/indicators/foundation/volume_profile.py src/validation/indicators/volume.py src/validation/indicators/volume_profile.py scripts/validate_volume.py scripts/validate_volume_profile.py src/indicators/_helpers/schema.py src/indicators/smc/equal_hl.py`
- `poetry run python scripts/validate_volume.py`
  - source parity, live/research parity, no-inf, zero-range, zero-volume all reported `True`
  - artifacts written:
    - `notebooks/foundation/volume_validation_XAU_USD_H1.html`
    - `notebooks/foundation/volume_validation_XAU_USD_H4.html`
- `poetry run python scripts/validate_volume_profile.py`
  - source parity, current-bar exclusion, allocated-volume audit, value-area audit all reported `True`
  - validator now reports:
    - `mode`
    - explicit warmup contract (`expected_first_valid_row`, `observed_first_valid_row`, valid-row count)
    - full-row VP audits rather than sample-only checks
    - degenerate-window counts (`zero_volume_windows`, `flat_price_windows`, `single_bin_windows`, `nan_contaminated_windows`)
    - exact-vs-stepped comparison stats
    - distribution stats for value-area occupancy and ATR-normalized VP distances/width
  - artifacts written:
    - `notebooks/foundation/volume_profile_validation_XAU_USD_H1.html`
    - `notebooks/foundation/volume_profile_validation_XAU_USD_H4.html`

## Downstream note for sweeps

- The sweep feature controller can now rely on canonical volume context instead of local volume patching.
- Default sweep-relevant subset now available from the shared volume layer:
  - `vol_ratio`
  - `vol_pct_rank_100`
  - `vol_above_1_5x`
  - `vol_extreme_pct90`
  - `bar_range_atr`
  - `close_pos_in_range`
  - `effort_vs_result`
  - `result_vs_effort`
  - `candle_delta_proxy`
  - `upper_rejection_effort`
  - `lower_rejection_effort`
  - `vsa_absorption`
  - `vsa_directional`
  - `vsa_effort_failure`

# Step 6 Freeze: AVWAP Context Layer And Signed Tick-Pressure Expansion

## Files touched

- `src/indicators/foundation/value.py`
- `src/indicators/foundation/volume.py`
- `src/indicators/foundation/__init__.py`
- `src/indicators/__init__.py`
- `src/indicators/pipelines/build_research.py`
- `src/validation/indicators/volume.py`
- `tests/indicators/foundation/test_value.py`
- `tests/indicators/foundation/test_volume.py`
- `tests/test_schema_normalization.py`
- `tests/test_indicators.py`

## Canonical AVWAP doctrine now frozen

- AVWAP stays in `src/indicators/foundation/value.py` as a canonical value/reference context layer.
- AVWAP is explicitly a tick-volume-weighted anchored fair-value proxy built on canonical internal `volume`, not true exchange VWAP.
- Low-level math remains in:
  - `compute_anchored_vwap(df, anchor_idx, atr_col="atr_14")`
- Canonical anchored builder is now:
  - `add_anchored_vwap(...)`
- Explicit anchor metadata is now part of the frozen contract:
  - `avwap_anchor_class`
  - `avwap_anchor_label`
  - `avwap_anchor_idx`
  - `avwap_anchor_origin_idx`
  - `avwap_anchor_confirm_idx`
  - `avwap_anchor_live_from_idx`
- Allowed anchor classes are frozen:
  - `live_safe`
  - `retrospective`
  - `hybrid`
- Live activation semantics are explicit:
  - AVWAP math can start from the anchor origin bar
  - canonical AVWAP outputs are masked until `avwap_anchor_live_from_idx`
  - this keeps hybrid anchors causal when their origin is only known at confirm time
- `add_avwap_from_last_swing()` is no longer the canonical contract:
  - it remains only as an optional helper
  - preferred helper behavior uses swing confirm-bar activation with origin-bar anchoring and classifies the anchor as `hybrid`
  - fallback behavior uses origin-bar swing annotations only and classifies the anchor as `retrospective`

## Canonical AVWAP outputs now implemented

- `avwap`
- `avwap_std`
- `avwap_upper_1`
- `avwap_lower_1`
- `avwap_upper_2`
- `avwap_lower_2`
- `avwap_dev_sigma`
- `avwap_distance`
- `avwap_distance_atr`
- `avwap_distance_pct`
- `avwap_above`
- `avwap_below`
- `avwap_inside_1sigma`
- `avwap_outside_2sigma`
- `avwap_cross_up`
- `avwap_cross_down`
- `avwap_slope_5`
- `avwap_slope_20`
- `avwap_trend_state`
- `bars_since_anchor`

## Research-only AVWAP outputs

- gated behind `include_research_only=True`
- columns:
  - `r_avwap_forward_1_return`
  - `r_avwap_forward_3_return`
  - `r_avwap_forward_5_return`
  - `r_avwap_reversion_label`
  - `r_avwap_breakout_followthrough_label`
  - `r_avwap_touch_count_since_anchor`
  - `r_avwap_max_dev_sigma_since_anchor`
  - `r_avwap_min_dev_sigma_since_anchor`

## Signed tick-pressure doctrine now frozen

- Signed pressure remains in `src/indicators/foundation/volume.py`, not in `value.py`.
- These are directional pressure proxies from OHLC geometry plus canonical tick-volume-backed `volume`, not true delta, net volume, or real signed traded volume.
- Backward-compatible aliases remain available:
  - `delta_proxy_raw`
  - `delta_proxy_norm`
  - `delta_proxy_body`
  - `candle_delta_proxy`
- The richer canonical family now includes:
  - primitives:
    - `close_strength`
    - `body_strength`
    - `wick_bias`
  - pressure measures:
    - `signed_tick_pressure_raw`
    - `signed_tick_pressure_norm`
    - `signed_tick_pressure_body`
    - `signed_tick_pressure_wick`
    - `signed_tick_pressure_blend`
    - `signed_tick_pressure_z`
  - agreement / divergence / extremes:
    - `pressure_agrees_with_body`
    - `pressure_agrees_with_close_location`
    - `pressure_divergence_flag`
    - `pressure_extreme_pos`
    - `pressure_extreme_neg`
- Research-only pressure study columns now exist behind `r_` gating:
  - `r_pressure_forward_1_return`
  - `r_pressure_forward_3_return`
  - `r_pressure_forward_5_return`
  - `r_pressure_followthrough_label`
  - `r_pressure_reversal_label`

## Validation and regression hardening added

- Volume validator now audits the richer pressure family explicitly:
  - required pressure columns present
  - zero-range behavior keeps wick-bias and blended pressure neutral
  - pressure proxy consistency formulas match the frozen definitions
  - chart now includes the blended pressure and its z-score
- Added AVWAP regression coverage for:
  - tickVolume vs canonical volume parity
  - constant-price stability
  - live activation masking
  - non-`r_` live/research parity
  - last-swing helper confirm-bar activation semantics
  - position-based masking correctness when index labels are not a default range
- Added schema-normalization parity coverage for `add_anchored_vwap(...)`.

## Downstream doctrine reinforced

- Sweeps and later modules should consume canonical AVWAP columns through `add_anchored_vwap(...)` or scanner-owned explicit anchors.
- No downstream module should reimplement local AVWAP math.
- No downstream module should reimplement local pressure / delta fallback logic.

## Dedicated AVWAP validator added

- Added:
  - `src/validation/indicators/value.py`
  - `scripts/validate_avwap.py`
- Validator audits:
  - required AVWAP columns present
  - no-inf status
  - band ordering
  - non-negative std and zero-std deviation behavior
  - bars-since-anchor alignment
  - trend-state value domain
  - non-overlapping cross events
  - no AVWAP values before live activation
  - first active row equals `avwap_anchor_live_from_idx`
  - source parity for canonical `volume` vs raw `tickVolume`
  - live/research parity for non-`r_` columns
- Validation artifact path:
  - `notebooks/foundation/avwap_validation_XAU_USD_H1.html`
  - `notebooks/foundation/avwap_validation_XAU_USD_H4.html`

## Validator hardening follow-up

- AVWAP validator now supports multi-anchor-family audits in the same summary output.
- `scripts/validate_avwap.py` now audits:
  - validation-window-start live-safe anchor
  - confirmed swing hybrid anchor
  - BOS event live-safe anchor
  - CHOCH event live-safe anchor
  - sweep detect-to-confirm hybrid anchor
- Volume validator now reports warmup context from the full reference frame, not just the displayed slice:
  - `displayed_slice_start_row`
  - `displayed_slice_end_row`
  - `warmup_reference_frame_row_count`
  - `reported_slice_is_post_warmup_only`
  - first-valid-row diagnostics for the main warmup families
- Volume validation chart volume bars are now colored by the sign of `signed_tick_pressure_blend`:
  - green = bullish pressure proxy
  - red = bearish pressure proxy
  - gray = unavailable / neutral fallback

## Effort / VP hardening follow-up

- The original epsilon-based `effort_vs_result` style ratios were numerically unstable when `bar_range_atr` or `bar_body_frac` approached zero.
- Canonical live-safe effort/result features now use meaningful floors instead of `1e-12`:
  - `effective_range_atr_floor = max(bar_range_atr, 0.05)`
  - `effective_body_frac_floor = max(bar_body_frac, 0.05)`
  - `effective_vol_ratio_floor = max(vol_ratio, 0.10)`
- This hardens:
  - `effort_vs_result`
  - `effort_vs_body`
  - `result_vs_effort`
  - `body_result_vs_effort`
- The unstable raw variants are preserved for research-only diagnostics:
  - `r_effort_vs_result_raw`
  - `r_effort_vs_body_raw`
  - `r_result_vs_effort_raw`
  - `r_body_result_vs_effort_raw`
- Volume Profile doctrine is now explicit:
  - canonical mode is `exact`
  - `stepped` is supported only as `approximation_only`
  - `stepped` is not parity-equivalent to exact and should not be treated as interchangeable in live/model contracts

## Remaining-gap hardening

- Volume validation now surfaces pressure-family orthogonality diagnostics instead of only agreement counts:
  - blend vs `candle_delta_proxy` correlation
  - mean absolute blend-vs-norm gap
  - sign-disagreement rate
  - blend-near-zero rate
  - mean absolute body-component share
  - mean absolute wick-component share
- The body/wick component-share diagnostics now divide by `max(abs(signed_tick_pressure_blend), 0.10)` to avoid exploding when the blend is near zero.
- AVWAP family validation now selects the latest family event with a minimum active-row budget when possible, rather than always taking the very last event and ending up with trivial 2-3 row coverage.
- Sweep detect-to-confirm hybrid AVWAP semantics now have explicit synthetic regression coverage in tests, so the contract is proven even when the real validation sample contains no recent confirmed sweeps.
- `scripts/validate_avwap.py` now also includes a `synthetic_sweep_detect_to_confirm_hybrid` family audit so validator output always exercises sweep-hybrid activation semantics even when the real sample contains no confirmed sweeps.
- The real `sweep_detect_to_confirm_hybrid` family is still unavailable in the current validation sample.
- This is not treated as a math/activation failure. It is an explicit finishing item for the sweep layer:
  - when the sweep family is finalized, rerun AVWAP validation with real sweep events and replace the current “unavailable in sample” status with true family coverage.
- AVWAP family validation is no longer single-anchor-only:
  - `scripts/validate_avwap.py` now samples up to the last `5` anchors per family when possible
  - `src/validation/indicators/value.py` aggregates family audits across those samples
  - summaries now expose:
    - `sample_count`
    - aggregate `active_row_count`
    - `active_row_count_stats`
    - per-sample previews
- Canonical signed pressure blend was retuned to reduce over-dominance of close-location:
  - base blend now gives wick a larger share than before
  - when `dominant_wick_ratio >= 0.35`, the blend shifts into a rejection regime that heavily favors wick information
  - wick components opposing close-location get a multiplier boost
  - body components opposing close-location get a smaller multiplier boost
- After this retune, validator output moved materially away from the old near-identity with `candle_delta_proxy`:
  - `blend_vs_norm_correlation` dropped from roughly `0.998` to roughly `0.95`
  - sign disagreement rose from roughly `1.8%` to roughly `12.5%`
  - this is still coherent, but now body/wick components are refinements with visibly stronger influence rather than near-no-op additions

# Step 6 Freeze: Regime Context

## Files touched

- `src/indicators/foundation/regime.py`
- `src/indicators/foundation/ema.py`
- `src/indicators/foundation/volatility.py`
- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/validation/indicators/regime.py`
- `scripts/validate_regime.py`
- `tests/indicators/foundation/test_regime.py`
- `tests/test_indicators.py`
- `docs/SCANNER_NOTES.md`
- `docs/PHASE2_HANDOFF.md`
- `docs/REGIME_DETECTION_NOTES.md`
- `.codex/memory.md`

## Canonical regime doctrine now frozen

- Regime is a non-directional environment classifier, not a trend/bias surrogate.
- Frozen live-safe state space:
  - `0 = RANGING`
  - `1 = TRANSITIONAL`
  - `2 = TRENDING`
- Regime is score-based, not chained-boolean placeholder logic.
- Canonical stabilization now biases toward persistence in extreme states:
  - weak transitional evidence does not immediately pull `RANGING` / `TRENDING` out of place
  - leaving a forced transitional state for a new extreme requires repeated confirmation unless margin is decisive
- Stabilization is part of canonical regime semantics, not post-processing glue:
  - `regime` is the stabilized canonical output
  - `raw_regime` is audit-only and should not be used by scanner/sweeps as the primary contract
- Warmup is explicit:
  - before prerequisites are valid, `regime` is null and one-hot regime flags are `0`
  - no silent default to transitional during warmup
- `trend_confidence < 0` is treated as zero directional confidence, not as a permanently invalid row. This keeps regime live-safe and usable during neutral structure once other prerequisites are ready.

## Canonical regime inputs / upstream contract

- `add_bb_width()` now emits:
  - `bb_width_pct_rank_100` in `[0, 1]`
- `add_emas()` now emits:
  - `ema_{P}_slope` as raw 3-bar EMA delta
  - `ema_{P}_slope_atr` as ATR-normalized EMA slope
- `add_regime()` requires:
  - `adx_14`
  - `bb_width`
  - `bb_width_pct_rank_100`
  - `ema_20_slope`
  - `ema_20_slope_atr`
  - `trend_state`
  - `trend_confidence`
  - `hh_count`
  - `ll_count`
  - `atr_14`

## Regime outputs now implemented

- helper inputs:
  - `adx_strength`
  - `compression_score`
  - `ema_slope_strength`
  - `structure_continuity`
  - `trend_confidence_norm`
  - `neutral_structure_penalty`
  - `regime_input_ready`
- scores / confidence:
  - `trend_regime_score`
  - `range_regime_score`
  - `transition_regime_score`
  - `regime_confidence`
  - `regime_margin`
  - `regime_strength_bucket`
  - `regime_boundary_flag`
- categorical outputs:
  - `raw_regime`
  - `raw_regime_label`
  - `raw_regime_confidence`
  - `raw_regime_margin`
  - `regime`
  - `regime_label`
  - `regime_is_ranging`
  - `regime_is_transitional`
  - `regime_is_trending`
- transition / persistence:
  - `regime_prev`
  - `regime_changed`
  - `regime_enter_ranging`
  - `regime_enter_transitional`
  - `regime_enter_trending`
  - `bars_in_regime`
  - `regime_persistence_5`
  - `regime_persistence_20`
- interpretation:
  - `regime_trend_alignment`
  - `regime_bias_alignment`
  - `regime_stabilized_from_raw`
  - `regime_forced_transitional`
  - `regime_direct_extreme_jump`
  - `regime_context_caution`
- research-only:
  - `r_regime_forward_5_return_abs`
  - `r_regime_forward_10_return_abs`
  - `r_regime_realized_vol_10`
  - `r_regime_dwell_final`
  - `r_regime_transition_type`

## Regime validation now implemented

- Added dedicated validator:
  - `src/validation/indicators/regime.py`
  - `scripts/validate_regime.py`
- Validator reports:
  - warmup contract / first valid row
  - valid regime row count
  - regime value counts
  - regime change count
  - dwell statistics
  - trend/bias alignment rates
  - per-regime alignment rates
  - trend/bias confusion matrices
  - extreme-misalignment audit for `TRENDING + neutral trend_state` and `RANGING + directional trend_state`
  - current regime snapshot
  - extreme-misalignment profiles
  - transition matrix
  - flicker diagnostics
  - boundary diagnostics
  - caution-source breakdown and overlap counts
  - raw-vs-stabilized audit
  - downstream caution contract
  - synthetic fixture runtime summary
  - score/confidence/margin stats
  - live/research parity
  - no-label contamination
  - regime-by-session counts when session context exists
- Validation artifacts:
  - `notebooks/foundation/regime_validation_XAU_USD_H1.html`
  - `notebooks/foundation/regime_validation_XAU_USD_H4.html`

## Regime verification completed

- `poetry run pytest tests/indicators/foundation/test_regime.py tests/test_indicators.py -k "regime or build_all or build_live" -q`
- `poetry run python scripts/validate_regime.py`

## Reference-data validation snapshot

- H1:
  - first valid regime row: `118`
  - valid regime rows: `65817 / 65935`
  - counts:
    - `RANGING: 23840`
    - `TRANSITIONAL: 16336`
    - `TRENDING: 25641`
  - `regime_change_count: 9062`
  - `trend_alignment_rate_pct: 55.46`
  - `bias_alignment_rate_pct: 48.57`
  - `single_bar_segment_rate_pct: 16.98`
  - `two_bar_segment_rate_pct: 15.35`
- H4:
  - first valid regime row: `118`
  - valid regime rows: `17865 / 17983`
  - counts:
    - `RANGING: 6234`
    - `TRANSITIONAL: 4321`
    - `TRENDING: 7310`
  - `regime_change_count: 2410`
  - `trend_alignment_rate_pct: 54.90`
  - `bias_alignment_rate_pct: 48.14`
  - `single_bar_segment_rate_pct: 15.64`
  - `two_bar_segment_rate_pct: 15.89`

These distributions, diagnostics, and stabilized segment rates are strong enough to keep regime as canonical scanner/model context, with explicit caution handling through:
- `regime_boundary_flag`
- `regime_confidence`
- `bars_in_regime`
- `regime_context_caution`

## Step 6A hard freeze record

- The canonical regime core is now frozen as a three-state, non-directional
  environment classifier:
  - `RANGING`
  - `TRANSITIONAL`
  - `TRENDING`
- Stabilization is part of canonical regime semantics, not a temporary patch.
- `raw_regime*` remains audit-only and should not be promoted to scanner or
  sweep contract.
- The caution contract is intentionally left unchanged for this freeze:
  - `regime_boundary_flag == 1`
  - `regime_confidence < 0.60`
  - `bars_in_regime <= 2`
  - `regime_context_caution == 1`
- The current validator remains the freeze gate:
  - warmup checks
  - live/research parity
  - synthetic fixtures
  - stabilization audit
  - transition matrix
  - misalignment profiles
  - caution breakdown
  - current regime snapshot
- Advanced regime-detection research is deferred to
  `docs/REGIME_DETECTION_NOTES.md`:
  - Hurst
  - Kalman
  - HMM
  - spectral / dominant-cycle methods
- Scanner doctrine was updated in `docs/SCANNER_NOTES.md`:
  - use canonical stabilized `regime`
  - do not consume `raw_regime*` as scanner contract
  - treat degraded regime context through:
    - `regime_boundary_flag`
    - `regime_confidence < 0.60`
    - `bars_in_regime <= 2`
    - `regime_context_caution`
- Phase handoff was updated in `docs/PHASE2_HANDOFF.md`:
  - Step 6A finalizes the canonical regime core
  - only bugs or downstream semantic conflicts should change the regime core
- Enhanced directional-volatility regime taxonomy is explicitly deferred to a
  separate derived layer pass and is not part of the canonical freeze.
- After Step 6A, only bug fixes or downstream semantic conflicts should alter
  the regime core. Broader regime redesign is out of scope.

## Latest regime reference snapshot after trend hardening

- H1:
  - current regime snapshot:
    - `TRENDING`
    - confidence `0.7335`
    - margin `0.4669`
    - caution `0`
    - bars in regime `3`
  - counts:
    - `RANGING: 22397`
    - `TRANSITIONAL: 13527`
    - `TRENDING: 29893`
  - `regime_change_count: 6894`
  - `single_bar_segment_rate_pct: 11.97`
  - `two_bar_segment_rate_pct: 13.14`
  - `context_caution_rate_pct: 39.71`
  - extreme mismatch buckets:
    - `TRENDING + neutral trend_state: 10115 (33.84%)`
    - `RANGING + directional trend_state: 6711 (29.96%)`
- H4:
  - current regime snapshot:
    - `TRANSITIONAL`
    - confidence `0.6625`
    - margin `0.1751`
    - caution `1`
    - bars in regime `1`
  - counts:
    - `RANGING: 5888`
    - `TRANSITIONAL: 3310`
    - `TRENDING: 8667`
  - `regime_change_count: 1744`
  - `single_bar_segment_rate_pct: 10.72`
  - `two_bar_segment_rate_pct: 11.63`
  - `context_caution_rate_pct: 38.21`
  - extreme mismatch buckets:
    - `TRENDING + neutral trend_state: 3010 (34.73%)`
    - `RANGING + directional trend_state: 1904 (32.34%)`

Interpretation:
- trend hardening improved regime chattiness materially without touching the
  regime core
- mismatch did not disappear, which is acceptable because Step 6B was meant to
  harden trend semantics first rather than force regime/trend agreement

# Step 6B: Trend-State Hardening

## Files touched

- `src/indicators/structure/trend_state.py`
- `src/validation/indicators/trend_state.py`
- `scripts/validate_trend_state.py`
- `tests/test_indicators.py`
- `docs/PHASE2_HANDOFF.md`
- `.codex/memory.md`

## Canonical trend-state doctrine now tightened

- `trend_state` remains strict structure:
  - `-1 = BEARISH`
  - `0 = NEUTRAL`
  - `1 = BULLISH`
- `NEUTRAL` is treated as structurally unresolved at bar close.
- `trend_bias_state` remains inherited directional pressure and may disagree
  with `trend_state`, but never replaces it.
- `trend_strength_*` remains directional evidence magnitude.
- `trend_confidence` is now normalized to `[0, 1]` and is no longer the old
  integer side-effect ladder.
- Neutral rows may retain meaningful confidence in neutrality.
- Regime remained frozen during this pass. No regime-core ontology or caution
  changes were made.

## Trend outputs added / changed

- confidence components:
  - `trend_conf_structure_continuity`
  - `trend_conf_freshness`
  - `trend_conf_event_quality`
  - `trend_conf_persistence`
  - `trend_conf_contradiction_penalty`
  - `trend_conf_neutral_coherence`
  - `trend_confidence`
- bias lifecycle:
  - `trend_bias_inherited_flag`
  - `trend_bias_expired_flag`
  - `trend_bias_contradicted_flag`
- transition / dwell:
  - `trend_prev`
  - `trend_enter_bullish`
  - `trend_enter_bearish`
  - `trend_enter_neutral`
  - `bars_in_trend_state`
  - `trend_persistence_5`
  - `trend_persistence_20`
  - `trend_direct_opposite_flip`

## Trend semantics implemented

- `trend_state` remains the strict current structural assignment.
- `trend_bias_state` remains inherited directional pressure under neutral /
  weakening structure and never replaces `trend_state`.
- `trend_strength_raw` / `trend_strength_ema` remain the directional evidence
  magnitude family.
- `trend_confidence` is now decomposed and normalized:
  - directional rows score from structure continuity, freshness, event quality,
    persistence, and contradiction penalty
  - neutral rows score from neutral coherence, contradiction penalty, and low
    directional-pressure balance
- bias lifecycle is now explicit and auditable:
  - inherited
  - expired
  - contradicted
- full dwell semantics now exist for all states, including neutral:
  - `trend_prev`
  - enter flags
  - `bars_in_trend_state`
  - `trend_persistence_5`
  - `trend_persistence_20`
  - `trend_direct_opposite_flip`

## Trend validator hardening

- `scripts/validate_trend_state.py` now validates both `XAU_USD H1` and
  `XAU_USD H4`.
- Validation now uses full-frame summary plus recent plot slice.
- Transition window sampling was fixed so windows are no longer empty because
  of filtered-frame index misalignment.
- Trend validator now reports:
  - current trend snapshot
  - transition matrices
  - dwell diagnostics
  - confidence-by-state
  - strength-by-state
  - bias interaction
  - trend/regime interaction
  - semantic mismatch buckets
- Validation artifacts now include:
  - `notebooks/structure/trend_state_validation_XAU_USD_H1.html`
  - `notebooks/structure/trend_state_validation_XAU_USD_H4.html`

## Verification completed

- `poetry run pytest tests/test_indicators.py -k "trend_state or build_all or build_live or regime" -q`
- `poetry run pytest tests/indicators/foundation/test_regime.py tests/test_indicators.py -k "regime or build_all or build_live or trend_state" -q`
- `poetry run python -m py_compile src/indicators/structure/trend_state.py src/validation/indicators/trend_state.py scripts/validate_trend_state.py`
- `poetry run python scripts/validate_trend_state.py`
- `poetry run python scripts/validate_regime.py`

## Current read

- Trend is now more observable and auditable than before.
- Regime remained frozen during this pass.
- The dedicated trend validator is now a real semantic gate rather than a
  lightweight single-slice check.
- The old `trend_confidence` integer ladder contract is gone in favor of a
  normalized `[0, 1]` confidence score. Any future downstream consumer must
  treat it as continuous confidence, not categorical state metadata.
- The post-trend-hardening regime snapshot improved materially on chattiness:
  - H1 `regime_change_count: 6894`
  - H4 `regime_change_count: 1744`
  - H1 single-bar regime segments: `11.97%`
  - H4 single-bar regime segments: `10.72%`
- Alignment did not improve by forcing agreement. That is acceptable for this
  pass because the goal was to harden trend semantics first, not cosmetically
  optimize confusion matrices.

## Trend reference snapshot after hardening

- H1:
  - current trend snapshot:
    - `trend_state: BEARISH`
    - `trend_bias_state: BEARISH`
    - confidence `0.6140`
    - `trend_strength_ema: -0.3229`
    - `bars_in_trend_state: 3`
  - strict-state counts:
    - `BEARISH: 17032`
    - `NEUTRAL: 30755`
    - `BULLISH: 18148`
  - transition count: `6770`
  - direct opposite flips are rare:
    - `BULLISH -> BEARISH: 16`
    - `BEARISH -> BULLISH: 15`
  - dwell diagnostics:
    - single-bar segments: `4.05%`
    - two-bar segments: `8.08%`
  - confidence by state:
    - bearish: `0.6668`
    - neutral: `0.6460`
    - bullish: `0.6744`
- H4:
  - current trend snapshot:
    - `trend_state: NEUTRAL`
    - `trend_bias_state: NEUTRAL`
    - confidence `0.7380`
    - `trend_strength_ema: -0.0493`
    - `bars_in_trend_state: 5`
  - strict-state counts:
    - `BEARISH: 4682`
    - `NEUTRAL: 8159`
    - `BULLISH: 5142`
  - transition count: `1840`
  - direct opposite flips are rare:
    - `BULLISH -> BEARISH: 7`
    - `BEARISH -> BULLISH: 7`
  - dwell diagnostics:
    - single-bar segments: `3.75%`
    - two-bar segments: `6.57%`
  - confidence by state:
    - bearish: `0.6708`
    - neutral: `0.6467`
    - bullish: `0.6777`

Interpretation:
- neutral is still common, but it is now far more auditable
- confidence no longer collapses mechanically when state is neutral
- trend can now be compared to regime more fairly because dwell, transitions,
  and bias lifecycle are explicit

## Step 6C.1 confidence correction shipped

- `trend_confidence` remains on the corrected Step 6C.1 contract:
  - directional rows score from structure, freshness, event quality,
    persistence, and contradiction penalty
  - neutral rows use soft-compressed neutral coherence with directional-evidence
    penalties
- the confidence inflation regression was fixed and kept fixed:
  - H1 mean confidence:
    - bearish `0.6681`
    - neutral `0.3980`
    - bullish `0.6764`
  - H4 mean confidence:
    - bearish `0.6732`
    - neutral `0.3985`
    - bullish `0.6806`
- neutral confidence cap check remains healthy:
  - max neutral confidence `0.621785`
  - neutral mean remains well below directional means

## Step 6C.2 diagnostics shipped; state retune rejected

- Added new canonical trend diagnostics:
  - `trend_commit_gap`
  - `trend_commit_dominant_side`
  - `trend_commit_gap_persist_3`
  - `trend_bull_dominant_2_of_3`
  - `trend_bear_dominant_2_of_3`
- Trend validator now includes:
  - `neutral_in_trend_audit`
  - `directional_in_range_audit`
  - `neutral_with_directional_bias_audit`
  - `commit_gap_audit`
  - `neutral_age_audit`
- Also clarified strict vs broad neutral-directional-evidence buckets in the
  validator output.

Important outcome:
- a live state-doctrine retune was attempted in this pass
- it reduced `NEUTRAL` in trending environments, but materially worsened churn
  and single-bar segment rates
- that retune was explicitly backed out
- shipped state engine therefore remains the stable Step 6C.1 behavior, with
  richer audits and commit-gap diagnostics added on top

Stable post-backout reference:
- H1:
  - strict-state shares:
    - bearish `26.07%`
    - neutral `46.03%`
    - bullish `27.90%`
  - transition count: `6744`
  - single-bar segments: `4.20%`
  - two-bar segments: `8.17%`
  - `neutral_in_trend_rows: 8111`
  - `directional_in_range_rows: 6778`
- H4:
  - strict-state shares:
    - bearish `26.36%`
    - neutral `44.20%`
    - bullish `29.44%`
  - transition count: `1830`
  - single-bar segments: `4.15%`
  - two-bar segments: `6.61%`
  - `neutral_in_trend_rows: 2347`
  - `directional_in_range_rows: 1935`

Doctrine after this session:
- confidence semantics are fixed and should not be retuned again casually
- regime remains frozen
- the next trend pass, if attempted, should use the new audits first and only
  then propose a narrower state-assignment change

# Runtime / Incremental Refactor Session

## Mission executed in this session

The objective for this session was not generic performance work. It was to start refactoring the trading system toward:

- strictly causal execution
- incremental-by-default runtime behavior
- lower IO pressure
- cache-aware validation/reporting
- reproducible metadata-driven runs
- frozen semantics for future ML/live usage

The implementation in this pass focused on the runtime/orchestration layer around the existing indicator builders, without changing the direct indicator semantics.

## What I changed

### 1. Added a shared pipeline runtime package

Created a new package:

- `src/pipeline_runtime/__init__.py`
- `src/pipeline_runtime/fingerprint.py`
- `src/pipeline_runtime/metadata.py`
- `src/pipeline_runtime/cache.py`
- `src/pipeline_runtime/profiling.py`
- `src/pipeline_runtime/artifact_store.py`
- `src/pipeline_runtime/incremental.py`

Responsibilities now covered there:

- stable content/config fingerprinting
- metadata read/write with atomic replace
- cache/report fingerprint helpers
- run profiling with stage summaries and artifact summaries
- atomic parquet/json artifact writing
- temp artifact cleanup helper
- incremental plan resolution
- frontier slicing and recomputed-frontier merge logic

### 2. Refactored pipeline entrypoints into staged orchestration

Updated:

- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/indicators/pipelines/__init__.py`
- `src/indicators/__init__.py`

Key changes:

- preserved the existing direct builders:
  - `build_live_indicators()`
  - `build_research_indicators()`
- added new orchestration/runtime entrypoints:
  - `run_live_pipeline()`
  - `run_research_pipeline()`
- introduced explicit stage lists with replay policies and class tags
- wrapped stage execution with runtime profiling
- added incremental planning against:
  - metadata
  - input fingerprint
  - config fingerprint
  - schema version
  - feature contract version
- added no-op detection
- added frontier replay slicing
- added merge of immutable history plus recomputed frontier

Important: I did not replace the canonical direct builders with approximations. The direct builders still compute the full stack in dependency order, and the runtime wrappers call them on the replay slice.

## Stage classification and replay policy

I added the contract doc:

- `docs/contracts/pipeline_runtime_contracts.md`

This freezes:

- live output contract
- research output contract
- persistence contract
- cache/state contract
- stage classification into:
  - Class A bounded-window/stateless
  - Class B stateful causal engine
  - Class C research-only/retrospective

Replay windows were encoded conservatively inside the staged runtime specs. Examples:

- `atr`: 150 bars
- `ema`: 220 bars
- `swings`: `max(400, swing_window * 30)`
- `trend_state`, `bos`, `choch`: 400 bars
- `fvg/ifvg/ob/sweeps/equal_hl/amd`: 300 bars
- `volume_profile`: 140 bars

These are safety margins, not proof-minimized limits. They were intentionally conservative to preserve semantics.

## Validation/reporting changes

Added:

- `src/validation/common/reporting.py`

Updated:

- `src/validation/common/chart_core.py`
- `scripts/validate_volatility.py`
- `scripts/validate_volume_profile.py`
- `scripts/validate_indicators.py`

Behavioral changes:

- validation scripts no longer default to heavy artifact generation
- HTML generation is now opt-in via `--html`
- default validation reads a trailing window unless `--full` is passed
- report generation is cached by fingerprint
- unchanged validation inputs skip HTML regeneration unless `--force`
- chart writes now use temp path plus atomic rename
- logging is more structured and less spammy

Specific script changes:

### `scripts/validate_volatility.py`

- added `--html`
- added `--full`
- added `--tail-rows`
- added `--force`
- switched to summary-first behavior
- added report fingerprinting and no-op HTML skip

### `scripts/validate_volume_profile.py`

- added `--html`
- added `--full`
- added `--tail-rows`
- added `--force`
- switched to lightweight default behavior
- added report fingerprinting and no-op HTML skip

### `scripts/validate_indicators.py`

- added `--full`
- added `--limit`
- defaulted to trailing-window validation instead of implicit full-history run

## Baseline instrumentation / benchmark command

Added:

- `scripts/benchmark_pipeline_runtime.py`

Purpose:

- provide a baseline benchmark entrypoint for live/research runtime execution
- emit runtime summaries through the new profiler
- optionally write benchmark JSON summaries

This is the beginning of Phase 0 instrumentation, not the final benchmark suite.

## Tests added

Added:

- `tests/pipeline_runtime/test_runtime.py`
- `tests/test_pipeline_incremental.py`

Coverage added:

- metadata atomic round-trip
- atomic parquet write
- no-op incremental plan detection
- report cache metadata round-trip
- profiler summary content
- live runtime full path parity against direct builder
- live incremental path parity against full rebuild
- research runtime no-op behavior
- research runtime full path parity against direct builder

## Verification completed

Commands run successfully in this session:

- `poetry run pytest tests/pipeline_runtime/test_runtime.py tests/test_pipeline_incremental.py -q`
- `poetry run pytest tests/test_indicators.py -q`
- `python3 -m compileall src/pipeline_runtime src/indicators/pipelines scripts/benchmark_pipeline_runtime.py scripts/validate_volatility.py scripts/validate_volume_profile.py scripts/validate_indicators.py src/validation/common`

Results:

- new runtime/incremental tests: passed
- existing indicator suite: passed
- compile check: passed

Observed warning during tests:

- `src/indicators/smc/ifvg.py` emits pandas fragmentation `PerformanceWarning`

This warning was pre-existing behavior in the indicator implementation. I did not rewrite its internals in this pass because the session goal was runtime/orchestration and semantic safety first.

## Semantics preserved

Important constraints I followed:

- no threshold changes
- no event timing reinterpretation
- no column drops for speed
- no change to direct builder feature meanings
- no switch to timestamp-only caching
- no shortcut that weakens live causality

The runtime layer is additive around the existing indicator stack.

## What is finished vs unfinished

### Finished in this pass

- shared runtime helper package
- staged live/research orchestration wrappers
- profiling support
- metadata support
- content-aware fingerprinting
- no-op detection
- replay-slice execution path
- frontier merge helper
- contract documentation
- lightweight validation/report caching defaults
- new parity/runtime tests

### Not finished yet

The following are still pending for the broader master plan:

- migrate every canonical dataset write path onto the new artifact manager
- complete end-to-end partitioned monthly frontier persistence
- wire metadata/artifact store into all true production persistence paths
- expand parity tests to interrupted-write recovery and schema migration paths
- broaden validation-script refactors across the entire script surface
- produce before/after benchmark summaries from real project datasets

In other words: the runtime framework now exists, the pipeline wrappers use it, and the validation defaults are improved, but the repo is not yet fully migrated to frontier-only partition persistence everywhere.

## File list changed in this session

- `src/pipeline_runtime/__init__.py`
- `src/pipeline_runtime/fingerprint.py`
- `src/pipeline_runtime/metadata.py`
- `src/pipeline_runtime/cache.py`
- `src/pipeline_runtime/profiling.py`
- `src/pipeline_runtime/artifact_store.py`
- `src/pipeline_runtime/incremental.py`
- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/indicators/pipelines/__init__.py`
- `src/indicators/__init__.py`
- `docs/contracts/pipeline_runtime_contracts.md`
- `src/validation/common/reporting.py`
- `src/validation/common/chart_core.py`
- `scripts/validate_volatility.py`
- `scripts/validate_volume_profile.py`
- `scripts/validate_indicators.py`
- `scripts/benchmark_pipeline_runtime.py`
- `tests/pipeline_runtime/test_runtime.py`
- `tests/test_pipeline_incremental.py`

## Practical summary

If I needed to describe this session in one sentence:

I did not yet fully redesign all persistence in the repo, but I built the runtime foundation for deterministic incremental execution, added content-aware no-op/caching behavior, moved live/research pipelines onto explicit staged orchestration, reduced unnecessary validation writes, and proved parity against the existing builders with tests.

# Canonical Persistence Migration Session

## Mission executed in this session

This session completed the next focused step after the runtime scaffold:

- migrate canonical live/research feature-store persistence onto the shared artifact/runtime layer
- enforce partition-aware atomic writes
- make ordinary live runs frontier-only at persistence time
- prove no-op and incremental artifact behavior with end-to-end tests
- document what is still intentionally full-rebuild-only

The key distinction in this pass:

- compute semantics were kept frozen
- persistence behavior changed materially
- validation/debug/report outputs remain separate and are not treated as canonical feature-store outputs

## Write-path audit outcome

I audited the repo write paths and found:

- there were many direct validation/debug/report exports
- there were raw-data write paths
- there was no previous centralized canonical live/research feature-store writer

So the migration work here was not replacing one old canonical writer with another. It was:

1. introducing the canonical materialization path
2. making it the defined path for canonical feature datasets
3. leaving validation/debug/report outputs explicitly separate

Audit note added:

- `docs/contracts/persistence_audit.md`

That note documents:

- canonical live output path
- canonical research output path
- validation/debug/report paths
- raw ingestion outputs
- partition policy
- immutable history rule
- intentionally full-rebuild-only paths

## Canonical persistence paths now added

### Live canonical materialization

Added/extended:

- `src/indicators/pipelines/build_live.py`

New canonical entrypoint:

- `materialize_live_features(...)`

Behavior:

- reads existing metadata from:
  - `data/features/_state/build_live/{symbol}/{timeframe}/metadata.json`
- loads existing canonical live partitions when metadata exists
- computes through the runtime pipeline
- persists to:
  - `data/features/live/{symbol}/{timeframe}/YYYY-MM.parquet`
- uses atomic temp-write + rename
- updates metadata only after successful partition writes
- ordinary no-op rerun writes nothing
- ordinary append rewrites only the frontier partition
- historical partitions before the frontier month stay untouched

### Research canonical materialization

Added/extended:

- `src/indicators/pipelines/build_research.py`

New canonical entrypoint:

- `materialize_research_features(...)`

Behavior:

- reads metadata from:
  - `data/features/_state/build_research/{symbol}/{timeframe}/metadata.json`
- persists to:
  - `data/features/research/{symbol}/{timeframe}/YYYY-MM.parquet`
- uses atomic temp-write + rename
- no-op rerun writes nothing

Important current rule:

- canonical research materialization is still full-rebuild-on-change

Reason:

- current research output still contains retrospective fields that are not frontier-safe yet
- specifically, fields like research regime dwell/transition style outputs can change when future rows change
- allowing frontier-only research rewrites today would create semantic drift

So the current research persistence contract is:

- unchanged input: no-op
- changed input: full rebuild through artifact manager

This was an intentional safety decision, not an omission.

## Artifact store changes

Extended:

- `src/pipeline_runtime/artifact_store.py`
- `src/pipeline_runtime/__init__.py`

New functionality added:

- canonical dataset root helper
- partition path listing
- partitioned dataset loading
- partitioned dataset persistence
- full rebuild stale-partition cleanup

Current partition scheme:

- live:
  - `data/features/live/{symbol}/{timeframe}/YYYY-MM.parquet`
- research:
  - `data/features/research/{symbol}/{timeframe}/YYYY-MM.parquet`

Persistence rules now implemented:

- partition key is UTC month from `timestamp`
- writes are atomic
- duplicates are removed on `timestamp`
- rows are sorted by `timestamp`
- full rebuild can remove stale partitions no longer present in rebuilt output
- ordinary live incremental runs only rewrite partitions from the current frontier month onward

## Frontier vs replay separation

One important bug surfaced during testing:

- replay context and persistence frontier are not the same thing

Initial issue:

- the replay window could stretch into older months
- if the persistence layer rewrote from replay start, it would rewrite historical partitions

Fix implemented:

- recomputation can still load a wider replay context
- persistence rewrites only from the partition containing `metadata.last_processed_ts`
- the merge helper now trims recomputed rows to the canonical persistence frontier before merging

This is what finally enforced:

- immutable history remains untouched
- only the frontier month is rewritten during ordinary live runs

## Additional runtime behavior changes

### Live materialization

Live materialization now:

- uses existing metadata for no-op detection
- loads current canonical history when needed
- preserves historical partitions
- updates profiler artifact records for canonical partition writes

### Research materialization

Research materialization now:

- supports no-op detection
- uses the artifact manager for canonical writes
- intentionally rebuilds fully on changed input
- still stores metadata/content fingerprints so no-op detection remains content-aware

## Tests added in this persistence pass

Added:

- `tests/test_pipeline_persistence.py`

Coverage added:

1. no-op rerun does not rewrite historical partitions
2. one-bar append touches only the frontier partition
3. multi-bar append preserves parity with full rebuild for canonical research persistence
4. interrupted write does not advance metadata or corrupt canonical artifacts
5. historical immutable partitions are not rewritten in ordinary live runs

These tests are end-to-end persistence tests, not just unit tests of helper functions.

## Verification completed in this session

Commands run successfully:

- `python3 -m compileall src/pipeline_runtime src/indicators/pipelines scripts/benchmark_pipeline_runtime.py tests/test_pipeline_persistence.py`
- `poetry run pytest tests/test_pipeline_persistence.py tests/test_pipeline_incremental.py tests/pipeline_runtime/test_runtime.py -q`
- `poetry run pytest tests/test_indicators.py -q`

Results:

- persistence tests: passed
- previous runtime/incremental tests: passed
- broader indicator suite: passed

Observed warnings:

- `src/indicators/smc/ifvg.py` still emits pandas fragmentation `PerformanceWarning`
- pandas warns that `to_period("M")` drops timezone information before relocalization in the partition-frontier helper

These warnings do not indicate semantic failure in the tested paths.

## Benchmark / proof of artifact behavior

I also updated:

- `scripts/benchmark_pipeline_runtime.py`

New capability:

- `--persist`
- `--features-root`

This allows benchmarking canonical materialization behavior, not just in-memory pipeline execution.

Practical evidence now exists in two forms:

1. automated persistence tests proving:
   - no-op writes zero canonical artifacts
   - one-bar append touches only one frontier partition
2. benchmark script support for persisted runs and profiler artifact summaries

## Files changed in this session

- `src/pipeline_runtime/artifact_store.py`
- `src/pipeline_runtime/incremental.py`
- `src/pipeline_runtime/__init__.py`
- `src/indicators/pipelines/build_live.py`
- `src/indicators/pipelines/build_research.py`
- `src/indicators/pipelines/__init__.py`
- `src/indicators/__init__.py`
- `scripts/benchmark_pipeline_runtime.py`
- `docs/contracts/persistence_audit.md`
- `tests/test_pipeline_persistence.py`

## What is now finished

- canonical live persistence is now routed through the artifact manager
- canonical live writes are frontier-only and partition-aware
- metadata updates are atomic and post-write
- interrupted write path is covered
- no-op rerun behavior is covered
- partition immutability is covered
- canonical research persistence is now centralized and artifact-managed

## What is still intentionally not incremental

### Canonical research persistence on changed input

Status:

- intentionally full rebuild only for now

Reason:

- current research output includes retrospective fields that are not safe for frontier-only updates

### Validation/debug/report artifacts

Status:

- still separate direct outputs

Reason:

- they are not canonical feature-store outputs
- they remain opt-in operationally

## Practical summary of this session

If I had to compress this session into one sentence:

I completed the canonical persistence migration by introducing artifact-managed live/research materialization, made live persistence frontier-only and atomic, kept research persistence centralized but full-rebuild-on-change for semantic safety, added end-to-end persistence proofs, and documented the remaining intentionally non-incremental paths.
