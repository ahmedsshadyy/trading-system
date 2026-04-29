# Scanner Implementation Notes

## Purpose
This document tracks gaps between what the indicator/detector library produces
and what the scanner needs to implement for each strategy. These are
confirmation patterns that require **cross-referencing multiple detectors**
at scanner time — they don't belong in individual detectors.

Review this document before implementing each strategy in Phase 3.

---

## FVG Confirmation Stack (Strategy 7: FVG Continuation)

The FVG detector identifies zones. The scanner must confirm:

1. **Liquidity sweep before FVG creation**
   - Check: was there a `sweep_high` or `sweep_low` within N candles (configurable, suggest 3-5)
     before the `fvg_bull` or `fvg_bear` fired?
   - Sweep before bullish FVG = higher conviction (stops were taken, then impulse created FVG)
   - Feature: `sweep_preceded_fvg` (binary)

2. **Volume on FVG creation candle**
   - Check: was `vol_above_avg` or `vol_above_1_5x` true on the middle candle of the FVG?
   - Feature: `fvg_creation_volume_ratio` (continuous)

3. **CHoCH after FVG tap (retest confirmation)**
   - When price returns to the FVG zone (`fvg_fill_pct > 0`), check if a `choch_bull` or
     `choch_bear` fires on or after the retest candle
   - This confirms trend reversal at the FVG zone
   - Feature: `choch_at_fvg_retest` (binary)

4. **HTF FVG → LTF entry**
   - Scanner detects FVG on 4H
   - Then loads 1H or 15M data and checks for: rejection candle from FVG zone,
     1H MACD turning, 1H volume above average
   - This is multi-timeframe logic in the scanner, not the detector

5. **Volume Profile confluence**
   - Check: does the FVG zone overlap with `vp_poc`, `vp_vah`, or `vp_val`?
   - Overlap = FVG at a high-volume node = institutional significance
   - Feature: `fvg_at_vp_level` (binary), `fvg_vp_distance_atr` (continuous)

6. **Retest quality**
   - Use `fvg_fill_pct` at the retest candle — optimal entry zone is 20-50% fill
   - Above 70% fill = zone is being absorbed, weaker setup
   - Feature: `fvg_fill_pct` already exists

---

## IFVG Confirmation Stack (Strategy 4: AMD + IFVG Retest)

1. **AMD cycle must be confirmed first**
   - `amd_phase` should show accumulation → manipulation sequence before IFVG forms
   - The manipulation candle creates the original FVG
   - Price reversing through it creates the IFVG
   - Scanner checks: was there an accumulation phase (10+ low-ATR candles) before?

2. **Daily bias alignment**
   - IFVG retest entry must align with Daily bias direction
   - Scanner loads Daily data and checks `trend_state` on Daily timeframe

3. **RSI divergence during manipulation**
   - Price makes new extreme during manipulation, RSI doesn't confirm
   - Check `rsi_div_bearish` or `rsi_div_bullish` near the manipulation extreme

4. **1H rejection from IFVG zone**
   - After 4H CHoCH, scanner loads 1H data
   - Checks for: candle wicking into IFVG zone with close back on distribution side
   - 1H RSI not overextended (below 55 bearish, above 45 bullish)
   - 1H volume on retest below average (weak participation into zone)

---

## OB Confirmation Stack (Deprecated For Strategy Use)

- BOS and CHoCH are superior to OB as primary structural signals.
- OB does not need to be used in any strategy in this repo.
- If OB remains available, it should be treated as optional execution research
  only, not as a required scanner or strategy component.
- Strategy work should prefer BOS / CHoCH directly instead of building a
  dedicated OB continuation strategy.

---

## Sweep Confirmation Stack (Strategy 5: Liquidity Sweep Reversal)

1. **RSI divergence at sweep**
   - `rsi_div_bearish` at sweep high, `rsi_div_bullish` at sweep low
   - Already computed — scanner just cross-references timing

2. **Volume below average on sweep candle**
   - `vol_below_avg` on the sweep candle = stop hunt, not genuine breakout
   - Already computed

3. **SMT divergence (deferred)**
   - XAU vs DXY, USOIL vs USDCAD
   - Requires correlated instrument data — implement when available

4. **Number of prior tests of swept level**
   - `equal_highs_count` or `equal_lows_count` at the swept level
   - More prior tests = larger stop cluster = more violent reversal

---

## BOS Confirmation Stack (Strategy 10: Grimes First Pullback)

1. **First pullback tracker**
   - After BOS fires, track whether this is the FIRST pullback since that BOS
   - If a prior pullback already occurred, this is Strategy 1 territory
   - Scanner must maintain state: `is_first_pullback_after_bos` (binary)

2. **BOS swing age minimum**
   - `bos_swing_age` should be ≥ 5 candles — not a trivially recent micro-swing
   - Already computed as a feature

3. **FVG from BOS candle**
   - Did the BOS candle leave an FVG? If yes, pullback to FVG is higher quality
   - Check temporal proximity of `bos_bull/bear` and `fvg_bull/bear`

---

## Volume Profile + BOS Stack (Strategy 11: VP Node BOS + Sweep)

1. **BOS at VP level**
   - Check: is the BOS candle's broken swing within 0.3 ATR of `vp_poc`, `vp_vah`, or `vp_val`?
   - Feature: `bos_at_vp_level` (binary)

2. **Volume on BOS candle ≥ 1.5x average**
   - Check `vol_above_1_5x` on the `bos_bull`/`bos_bear` candle
   - Already computed

3. **Subsequent sweep of broken VP level**
   - After BOS, price pulls back and sweeps below the broken level
   - Then closes back above — this is the entry trigger
   - Scanner must track this sequence explicitly

---

## Session Range Stack (Strategy 3: Session Range Sweep)

1. **Asian session H/L already computed** — `asian_high`, `asian_low`, `asian_range`

2. **Scanner must check sweep timing**
   - Sweep must occur within first 2 hours of London or NY open
   - Use `london_open` or `ny_open` flags + `hours_since_session`

3. **15M CHoCH after sweep**
   - Scanner loads 15M data after 4H sweep detected
   - Checks for CHoCH on 15M — displacement candle with body ≥ 60% of range

---

## General Cross-Detector Patterns

These apply across multiple strategies:

1. **Regime gating** — every strategy has a regime requirement. Check `regime` column
   before evaluating strategy conditions.
   Canonical doctrine:
   - the scanner-safe base regime is frozen to `RANGING / TRANSITIONAL / TRENDING`
   - the stabilized `regime` output is the only primary contract
   - `raw_regime*` fields are validator/audit outputs, not scanner-gating inputs
   Treat degraded regime context with caution:
   - `regime_boundary_flag == 1`
   - `regime_confidence < 0.60`
   - `bars_in_regime <= 2`
   - or the canonical convenience flag `regime_context_caution == 1`
   Derived richer regime labels, if added later, must sit on top of canonical
   regime rather than replacing it inside scanner logic.

2. **Daily bias** — most strategies require Daily trend alignment. Scanner loads
   Daily timeframe data and checks `trend_state` on Daily.

3. **Session filtering** — all strategies restricted to London (08:00-17:00 UTC)
   and NY (13:00-22:00 UTC). Use `session` column, reject if `is_dead_zone` == 1
   or `session` == 0 (Asian) unless Strategy 3.

4. **Macro event proximity** — "no Tier-1 macro event within 48 hours."
   Requires ForexFactory data (not yet implemented). Add as deferred feature.

---

## Key Level Scope Note

Exness-style "key levels that matter if broken" are not the same thing as
plain static support/resistance.

In this codebase, those should be modeled as a **composite breakout/reversal
layer** built on top of multiple inputs:
- HTF support/resistance
- prior session/day/week levels
- liquidity context
- structure context
- break/reclaim behavior

So `sr_levels` should remain the base S/R engine. The higher-conviction
"important if broken" interpretation belongs in scanner logic or in a future
derived layer that combines `sr_levels` with structure and liquidity, rather
than being treated as the direct meaning of raw `sr_levels` output.

`sr_range_proxy` is the separate answer to a different question:
"do the current S/R bands behave enough like a bounded range that downstream
logic can treat them as a range-style proxy?"

`key_levels` is the separate answer to the Exness-style question:
"which current S/R levels look important enough that a break/reclaim should
matter structurally?"

---

## Deferred Items

- SMT divergence (needs DXY, USDCAD data)
- Macro event calendar (needs ForexFactory scraper)
- CME futures volume proxy for order flow (MVP2+)

---

*Last updated: Phase 2 indicator validation, March 2026*
## Sweep + FVG/OB Confluence (cross-strategy)

When a sweep occurs at a level that also has an active FVG or unmitigated OB:
- Check: does `sweep_level_high` or `sweep_level_low` overlap with an active FVG zone or OB zone?
- This is the highest-conviction sweep setup — institutional level + imbalance zone + stop hunt
- Feature: `sweep_at_fvg` (binary), `sweep_at_ob` (binary)
- Applies to Strategy 5 primarily, but also S3 and S11


## Timing is important. 
The scanner should note, how things went in the Asian session. What are the critical numbers. Market behavior and structure in Asia. For example if Asia accumulates or expands, since this will make it more important to see if london manipulates or accumulates and then NY if it distributes. More generally, it should feed the agent with data that would allow it to tell me what is proper price entry, sizing and confirmations
