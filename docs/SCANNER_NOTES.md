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

## OB Confirmation Stack (Strategy 6: Order Block Continuation)

1. **First retest only**
   - `ob_first_retest` flag exists — scanner should only consider first retest
   - Second/third retests have diminishing quality

2. **Deceleration into OB**
   - ATR of candles approaching OB should be below ATR of impulse candles
   - Use `atr_ratio_rolling` or compute specifically for the approach

3. **FVG above/below OB**
   - The impulse that created the OB should also have left an FVG
   - Check: is there a `fvg_bull` or `fvg_bear` within 1-3 candles of the OB?
   - Feature: `ob_has_fvg` (binary)

4. **1H rejection candle from OB zone**
   - Load 1H data, check for hammer/engulfing at OB zone
   - 1H MACD turning in OB direction
   - 1H volume on rejection above average

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

2. **Daily bias** — most strategies require Daily trend alignment. Scanner loads
   Daily timeframe data and checks `trend_state` on Daily.

3. **Session filtering** — all strategies restricted to London (08:00-17:00 UTC)
   and NY (13:00-22:00 UTC). Use `session` column, reject if `is_dead_zone` == 1
   or `session` == 0 (Asian) unless Strategy 3.

4. **Macro event proximity** — "no Tier-1 macro event within 48 hours."
   Requires ForexFactory data (not yet implemented). Add as deferred feature.

---

## Deferred Items

- SMT divergence (needs DXY, USDCAD data)
- Macro event calendar (needs ForexFactory scraper)
- CME futures volume proxy for order flow (MVP2+)

---

*Last updated: Phase 2 indicator validation, March 2026*