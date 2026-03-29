---
name: EQH/EQL Doctrine Freeze
description: Frozen decisions for Equal H/L detector — role, config, selector, score v3, acceptance criteria, downstream contract
type: project
---

## Role (frozen)

EQH/EQL is a **ranked structural liquidity-source family**, not a BOS-like immediate trigger.
It behaves like a liquidity map / pool inventory. Detection is inherently delayed (~18 bars formation delay on H4 XAU/USD by confirmed-swing doctrine). Do not force it to behave like a fresh execution event.

## Canonical Config (evaluation, not yet frozen)

- `atr_tolerance`: 0.15 (being evaluated; 0.12 gives sharper ranking, 0.15 gives more inventory)
- `lookback_swings`: 50
- `min_touches`: 2
- `max_cluster_width_atr`: 0.5
- `max_cluster_span`: 120
- `max_active_age`: 200
- Config must be explicitly frozen before handoff to live pipeline.

## Score v3 (implemented)

Reference constants fixed for H4 realities:
- `TRADEABLE_AGE_REF = 200.0` (was 20 → created step function; most clusters scored 0)
- `TRADEABLE_FORM_REF = 60.0` (was 25 → too tight for mean 18-bar formation delay)
- `span_component` replaced by `structural_component` (span == formation_delay for 98% of 2-touch clusters; structural_score is age-independent and non-redundant)
- Touch formula changed: `(touch_count - 2) / 3.0` so 2-touch=0, 3-touch=0.33 (was 0.33 for all 2-touch)
- Default weights: 0.35 age + 0.25 structural + 0.20 formation_delay + 0.10 touch + 0.10 width

**Why:** Old constants caused score collapse to near-single-factor (0.94 weight on age step-function). v3 fixes the decay curves and removes the span/formation_delay redundancy.

## Selector (still provisional — not frozen)

Current: `structural_score` first, then `distance_atr`, then `touch_count`, then `width_atr`, then `detect_idx`, then `tradeable_live_score`, then `cluster_id`.

Open question: should live selection be structural-first, tradeable-score-first, or proximity-first?
Do not freeze this until v3 score calibration is validated on real data.

## Frozen Economic Acceptance Criteria (in code as constants)

Defined in `src/validation/indicators/equal_hl.py`:
- `EQHL_ACCEPT_MIN_RANK_CORR = 0.05` — overall Spearman rank correlation must be positive
- `EQHL_ACCEPT_MIN_TOP_QUARTILE_AVG_R = 0.0` — top-25% trades must be profitable
- `EQHL_ACCEPT_MIN_TOP_VS_BOTTOM_AVG_R = 0.05` — top-quartile must beat bottom-decile by ≥0.05R
- `EQHL_ACCEPT_MIN_WIN_RATE_GAP = 0.03` — top-quartile win-rate must beat bottom-decile by ≥3pp
- `EQHL_ACCEPT_SIDE_RANK_CORR_FLOOR = -0.05` — neither EQH nor EQL rank-corr may collapse below this
- `EQHL_ACCEPT_TOP_DECILE_IMPROVES_QUARTILE = True` — top-10% avg_r must be ≥ top-25% avg_r

**Why top-decile >= top-quartile matters:** The old v2 score had identical top-decile and top-quartile metrics because the score distribution was degenerate (step function). A good multi-factor score should produce a spread where the tighter top-10% cut outperforms the top-25%.

## Known Remaining Risks

- 2-touch dominance: ~98% of clusters are minimal 2-touch. Touch component has near-zero variance. Grid search should keep touch weight ≤ 0.10 (already constrained).
- Generalization: all calibration on XAU/USD H4. Score may be dataset-specific. Do not over-fit.
- Structural score std is narrow (0.048). It discriminates but not sharply. Do not expect it to carry the full ranking alone.

## Downstream Usage Contract (not frozen)

Open question: should downstream modules (sweeps, scanner) consume:
- `eqh_active` / `eql_active` (current-best only)?
- `eqh_rank1_active_*` / `eqh_rank2_active_*` (top 2 ranked)?
- A freshness-tiered subset?
Freeze this before sweeps integration.

## Event Freshness vs Snapshot Persistence (separate diagnostics)

- `formation_delay`: event-level — how many bars between first and last touch at cluster activation. Measures how slow the pool was born.
- `active_age` (snapshot-level): how long the cluster has been active in the dense export at any given bar. Measures persistence.
These have different optimization targets. Do not conflate. Do not penalize durable levels just for persisting.

## What is done and should NOT be redesigned

- Detector causality (confirmed swings only)
- Detect timing (active state begins at min_touches row)
- Sweep lifecycle (swept row = last export, deactivated next row)
- Event IDs (sequential, deterministic)
- Research/live separation
- EQHL_LIVE_COLUMNS constant (now exported from equal_hl.py)
