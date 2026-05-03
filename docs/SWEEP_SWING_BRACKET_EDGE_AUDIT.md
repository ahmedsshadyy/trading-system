# Sweep / Swing Bracket Edge Audit (Steps 11T + 11U)

Research-only validation layer. Nothing in these audits mutates the
production sweep, swing, displacement, or unified-liquidity stages.

## Purpose

Answer one trading-direct question for confirmed sweeps and confirmed
causal swings:

> If we entered at signal confirmation, would TP hit before SL within
> the next 1–5 bars?

Step 11T probes a single symmetric ±0.5 ATR bracket. Step 11U extends to
four bracket profiles and six event universes that combine sweep, swing,
and displacement signals.

## Modules

- [src/validation/indicators/atr_bracket_first_hit.py](../src/validation/indicators/atr_bracket_first_hit.py)
  — Step 11T: per-event ATR bracket first-hit + 6 grouped tables.
- [src/validation/indicators/bracket_matrix_confluence.py](../src/validation/indicators/bracket_matrix_confluence.py)
  — Step 11U: bracket matrix × confluence audit + 13 grouped tables.
- Both wired into
  [scripts/validate_final_sweeps.py](../scripts/validate_final_sweeps.py).

## Causality contract

Same rules apply to both audits:

- **Swing entries anchor at the confirmation bar**, not the origin bar.
  The confirmation bar is itself a live signal — the
  `swing_*_confirm_flag` fires at confirm-bar close, exposing
  `entry_close = close[confirm_idx]`, `atr_ref = atr_14[confirm_idx]`.
  See [src/indicators/structure/swings.py:356-361](../src/indicators/structure/swings.py#L356-L361).
- **Sweep entries anchor at `sweep_confirm_idx`** (close of the bar
  where `sweep_flag = 1`).
- **Forward bars used only as research output**: outcomes read
  `confirm_idx + 1 .. confirm_idx + horizon`, never confirm_idx itself
  or earlier.
- **Same-bar TP/SL collisions** are explicitly tagged
  `ambiguous_same_bar` (Step 11T) / `ambiguous` (Step 11U). Never
  guessed.
- **Insufficient horizon** is distinct from `neither` — the audit only
  emits `insufficient_future` / `insufficient` when no hit was observed
  AND we did not have `horizon` bars to look at. If the bracket position
  closes early on a TP/SL/ambiguous hit, that outcome is locked in
  regardless of horizon length.
- **Step 11U swing-confluence** matches a sweep's `sweep_source_level`
  only against swings whose `swing_*_confirm_flag` fired at or before
  the sweep bar. Future swings are excluded — see
  [test_sweep_swing_confluence_does_not_use_future_swing](../tests/test_step11u_bracket_matrix_confluence.py#L162-L177).

## Live vs post-hoc — the swing-displacement nuance

The swing confirmation itself is **fully live and causal**. When
`swing_low_confirm_flag[t] = 1` you can enter at bar t close; the
bracket pricing has no future leakage.

What is **post-hoc is the universe filter**
`swing_displacement_confirmed`, not the swing. That filter is defined as
"confirmed swings whose next 1–3 bars contain a reversal-direction
displacement bar." At bar t you cannot yet know whether bars
t+1..t+3 will produce displacement — universe membership is only
decidable up to 3 bars later.

This matters because the displacement bar by definition has
`body_atr ≥ ~0.7`. If a bullish displacement lands at t+k for a
swing_low entered at t close, that bar's high almost-mechanically
reaches `entry + 1.0 ATR` — i.e. TP is hit on or before the displacement
bar. So "swing has reversal-direction displacement in 1–3 bars" is
nearly synonymous with "TP @ 1 ATR is reached in 1–3 bars." The 90.5%
win rate on `tp1p0_sl1p0` for `swing_displacement_confirmed` is closer
to a tautology than to a tradeable edge — it requires observing future
favorable displacement.

By contrast, `sweep_displacement_confirmed` is **genuinely live**. The
class is set inside the sweep stage at `sweep_confirm_idx` from the
`sweep_is_displacement_confirmed` flag in
[src/indicators/smc/sweeps/final_sweeps.py:367](../src/indicators/smc/sweeps/final_sweeps.py#L367) — all of its inputs are known
at the confirm bar.

| Universe                             | Filter is decidable at | Status               |
|--------------------------------------|------------------------|----------------------|
| `sweep_all`                          | sweep_confirm_idx      | live                 |
| `sweep_displacement_confirmed`       | sweep_confirm_idx      | **live, tradeable**  |
| `swing_all`                          | swing_confirm_idx      | live                 |
| `swing_displacement_confirmed`       | swing_confirm_idx + 3  | post-hoc filter      |
| `sweep_swing_confluence`             | sweep_confirm_idx      | live                 |
| `sweep_swing_displacement_confluence`| sweep_confirm_idx      | live                 |

## Headline findings — XAU_USD H4, h=5

### Step 11T (single ±0.5 ATR bracket)

```
swings_total: 3932
sweeps_total: 2609
swings_win_rate_ex_ambiguous_5:           0.4854
sweeps_win_rate_ex_ambiguous_5:           0.4977
displacement_confirmed_sweep_win_rate_5:  0.5994   (n=362 resolved)
tradeable_sweep_candidate_win_rate_5:     0.4869
best_sweep_class:    displacement_confirmed_sweep
best_sweep_family:   equal_low
best_swing_side:     swing_high
```

### Step 11U (4 bracket profiles × 6 universes)

Win rate ex-ambiguous, h=5:

```
universe                              count   tp0p5_sl0p5  tp1p0_sl1p0  tp1p0_sl0p5  tp0p5_sl1p0
sweep_all                              2609     0.4977       0.4894       0.3251       0.6708
sweep_displacement_confirmed            588     0.5792       0.6273       0.4745       0.7268
swing_all                              3932     0.4854       0.4968       0.3175       0.6638
swing_displacement_confirmed            170     0.7610       0.9053       0.7195       0.9112  ← post-hoc
sweep_swing_confluence                 2249     0.4960       0.4912       0.3209       0.6705
sweep_swing_displacement_confluence     504     0.5698       0.6205       0.4594       0.7244
```

`best_group_with_min_count_300 = sweep_displacement_confirmed`. The
`best_group_with_min_count_100` slot is `swing_displacement_confirmed`
but should be read as a research-curiosity rather than a candidate —
see post-hoc note above.

## Key conclusions

1. **Sweeps alone and swings alone are random** at 0.5 ATR (49.77%,
   48.54%).
2. **Displacement confirmation is the only robust live edge.**
   `sweep_displacement_confirmed`: 57.92–72.68% across the four
   profiles, count=588, fully causal.
3. **Swing confluence does not improve sweep quality.**
   `sweep_swing_confluence` (49.60%) ≈ `sweep_all` (49.77%).
   `sweep_swing_displacement_confluence` (56.98%) is *worse* than
   `sweep_displacement_confirmed` (57.92%) — once displacement is
   accounted for, the swing-confluence filter adds no incremental edge.
4. **Bracket profile choice is degenerate without R-multiple
   weighting.** `tp0p5_sl1p0` wins more often (67%) but at 0.5R while
   `tp1p0_sl0p5` wins 33% at 2R. Expectancy is similar on `sweep_all`.
   Symmetric `tp1p0_sl1p0` (1R win) is the cleanest comparison:
   `sweep_displacement_confirmed` is the only universe that clears 50%
   robustly with a live-decidable filter.
5. **The swing-displacement universe is post-hoc.** Its 90.5% win rate
   is nearly mechanical — the displacement bar's range is what fires
   TP. Cannot be traded as-is; could inform a different strategy
   ("enter at swing confirm, abandon if no displacement by t+3"), but
   that variant needs its own causal audit.

## Promotion recommendation

Promote `sweep_is_displacement_confirmed` (the existing
`displacement_confirmed_sweep` selectivity class) into the final
trading-candidate definition. Skip `sweep_swing_confluence`.

## Output files

Per (instrument, timeframe), under `notebooks/sweeps_v2/`:

**Step 11T (7 files):** `atr_bracket_first_hit_events_*`,
`atr_bracket_first_hit_by_entity_*`,
`atr_bracket_first_hit_by_sweep_class_*`,
`atr_bracket_first_hit_by_sweep_family_*`,
`atr_bracket_first_hit_by_swing_side_*`,
`atr_bracket_first_hit_by_regime_*`,
`atr_bracket_first_hit_by_session_*`.

**Step 11U (14 files):** `final_sweeps_step11u_events_*`,
`final_sweeps_step11u_by_confluence_type_*`,
`final_sweeps_step11u_by_entity_type_*`,
`final_sweeps_step11u_by_sweep_class_*`,
`final_sweeps_step11u_by_sweep_family_*`,
`final_sweeps_step11u_by_swing_side_*`,
`final_sweeps_step11u_by_regime_*`,
`final_sweeps_step11u_by_session_*`,
`final_sweeps_step11u_by_volume_confirmed_*`,
`final_sweeps_step11u_by_displacement_confirmed_*`,
`final_sweeps_step11u_by_swing_confluent_*`,
`final_sweeps_step11u_by_confluence_regime_*`,
`final_sweeps_step11u_by_confluence_session_*`,
`final_sweeps_step11u_by_confluence_volume_*`.

## Sample-size guards (Step 11U)

Grouped tables annotate every row with:

- `low_sample = True` when `count < 100`
- `low_resolution_{profile}_h5 = True` when `resolved_count_5 < 50`

`best_group_*` selectors skip low-sample / low-resolution groups
unless explicitly invoked through the `min_count_100` / `min_count_300`
variants in the headline summary.

## Tests

- [tests/test_atr_bracket_first_hit.py](../tests/test_atr_bracket_first_hit.py)
  — 11 cases covering causality, same-bar ambiguous, direction, lock-in.
- [tests/test_step11u_bracket_matrix_confluence.py](../tests/test_step11u_bracket_matrix_confluence.py)
  — 15 cases covering bracket directionality (4 profiles), confluence
  pool causality (no future swings), displacement direction matching,
  universe-overlap counts, low-sample/low-resolution flags.


  📌 TODO — Post-Indicator Phase: Displacement-Based Strategy Validation

Context

Preliminary validation indicates:

* Raw swings and sweeps ≈ no edge (≈ 50% winrate)
* Displacement-confirmed events show significant edge
* Best-performing groups:
    * sweep + displacement
    * swing + displacement (high winrate, lower sample)

Conclusion:
→ Displacement acts as the primary edge gate
→ Swings and sweeps act as context/location only

⸻

🚧 Deferred Work (To Be Done After Indicator Phase)

1) Robustness Validation

Run full validation across:

Instruments

* XAUUSD (baseline)
* EURUSD
* USDJPY
* USOIL

Timeframes

* H4
* H1
* M15

Time splits

* Train-like: 2014–2020
* Test-like: 2021–present

Regime segmentation

* Trending
* Ranging
* Transitional

Goal:
→ Ensure displacement edge is not instrument-specific or regime-fragile

⸻

2) Strategy Formalization

Define a minimal, fully causal rule:

Entry

* Condition: sweep + displacement_confirmed
* Entry timing: close of confirmation bar

Stop Loss

Candidates:

* Structure-based (beyond sweep)
* ATR-based
* Displacement-origin-based

Take Profit

Evaluate:

* 0.5R
* 1.0R
* asymmetric RR (based on prior results)

⸻

3) Performance Metrics

Move beyond winrate:

* Expectancy
* Equity curve
* Max drawdown
* Trade frequency
* Risk-adjusted return

⸻

4) Data Logging for ML (Preparation Only)

For each event, log:

* displacement_score
* regime
* ATR / volatility
* time of day
* sweep type
* proximity to S/R

Purpose:
→ Future supervised learning / ranking model

⸻

⚠️ Important Constraints

* All logic must remain strictly causal (no lookahead)
* Use only frozen indicator outputs (no redefining primitives)
* No additional SMC concepts to be introduced before validation

⸻

🎯 Objective

Validate whether displacement-based setups form a:

→ robust, generalizable trading edge

before promoting to production or ML layer
