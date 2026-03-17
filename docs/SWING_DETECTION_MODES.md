# Swing Detection: Training vs Live Deployment

## The Problem

The system has two swing detection approaches that produce different results.
The model must be trained and deployed on compatible detectors, or feature
values will mismatch and predictions degrade.

## The Two Detectors

### `add_swings(causal=True)` — Symmetric with Delayed Availability

- **How it works:** Uses a symmetric window (looks at `window` bars before AND
  after each bar) to identify pivot highs/lows. This produces clean, confirmed
  swing points. But `last_swing_high/low` is only made available `window` bars
  after the swing — simulating the delay you'd experience in real-time.
- **Swing quality:** High — the look-ahead confirms the pivot is real.
- **Availability timing:** Honest — no look-ahead leakage into downstream
  features (BOS, CHoCH, sweeps, etc.).
- **Limitation:** Cannot run on a live bar-by-bar feed because it requires
  future bars that don't exist yet.

### `add_swings_causal()` — Pure Causal (Backward-Only)

- **How it works:** Only uses past bars. A swing high at bar `i` means `high[i]`
  is the max of the last `window` bars. No look-ahead of any kind.
- **Swing quality:** Noisier — finds ~50% more swing points because every new
  local high/low is flagged immediately, even if the next bar goes higher.
- **Availability timing:** Immediate — the swing is known on the bar it occurs.
- **Advantage:** This is what actually runs in a live scanner.

## Comparison Data (XAU/USD H4, 18K candles)

| Metric | Symmetric (causal=True) | Pure Causal |
|---|---|---|
| Swing highs | 1,877 | 2,843 |
| Swing lows | 1,916 | 2,470 |
| BOS bull | 701 | 797 |
| BOS bear | 637 | 597 |
| CHoCH bull | 347 | 298 |
| CHoCH bear | 283 | 320 |
| Sweep high | 414 | 431 |
| Sweep low | 450 | 412 |
| Equal highs | 239 | 482 |
| Equal lows | 236 | 398 |

The swing points are different, which causes BOS/CHoCH to fire on different
candles against different levels, sweeps to reference different levels, and
equal H/L clusters to form differently.

## Which Detectors Are Affected

### Directly affected (read `last_swing_high/low` from add_swings):

- **BOS detector** (`add_bos`) — fires when close breaks `last_swing_high/low`.
  Different swing levels = different BOS candles.
- **CHoCH detector** (`add_choch`) — fires on first BOS against trend. Different
  BOS = different CHoCH.
- **Liquidity Sweep detector** (`add_liquidity_sweep`) — checks sweeps against
  `last_swing_high/low`. Different levels = different sweep detections.
- **Trend State Machine** (`add_trend_state`) — tracks HH/HL/LH/LL from
  `swing_high_price/swing_low_price`. Different swing points = different trend
  state transitions.

### Indirectly affected (read `swing_high_price/swing_low_price`):

- **Equal H/L detector** (`add_equal_hl`) — clusters swing prices. More swings
  = more clusters, but cluster levels also shift.
- **RSI Divergence** (`add_rsi_divergence`) — checks RSI at swing points.
  Different swing locations = divergence at different bars.

### NOT affected (have their own internal swing detection):

- **Order Block detector** (`add_ob`) — uses `_last_confirmed_swing_levels()`
  internally, independent of `add_swings()`.
- **FVG detector** (`add_fvg`) — uses its own internal confirmed pivots for
  optional BOS check.
- **IFVG classifier** (`add_ifvg`) — depends on FVG output, not swings.
- **Displacement candle** (`add_displacement_candle`) — pure ATR/body ratio.
- **AMD Phase** (`add_amd_phase`) — uses ATR percentile, not swings.
- **All standard indicators** (EMA, RSI, ATR, ADX, MACD, BB, volume) — no swing
  dependency.

## MVP1 Plan (Current)

Use `add_swings(causal=True)` for all historical backtesting, labeling, and
model training.

- Clean pivot points (symmetric detection)
- No look-ahead leakage (delayed availability)
- Consistent feature values across training set
- Can compare against TradingView with `causal=False` for validation

The `swing_mode` parameter in `build_all_indicators()` controls this:
- `swing_mode="symmetric"` → `add_swings(causal=False)` — for TV comparison
- `swing_mode="symmetric_causal"` → `add_swings(causal=True)` — for training
- `swing_mode="causal"` → `add_swings_causal()` — experimental comparison

**Default should be set to `"symmetric_causal"` for training.**

## MVP2 Plan (Live Deployment)

Two options, to be decided based on Phase 6 model performance:

### Option A: Retrain on Pure Causal Before Going Live

1. Switch to `swing_mode="causal"` in `build_all_indicators()`
2. Re-run the full pipeline: indicators → scanner → labeling → feature matrix
3. Retrain the model on causal-derived features
4. Deploy with `add_swings_causal()` in the live scanner
5. Training and deployment match perfectly

**Trade-off:** Noisier swings mean noisier features. Model may have lower
precision. But no train/deploy mismatch.

### Option B: Delayed Symmetric in Live (Hybrid)

1. Keep the model trained on `causal=True` (symmetric + delayed)
2. In the live scanner, maintain a rolling buffer of recent H4 candles
3. After each new candle, recompute `add_swings(causal=True)` on the buffer
4. Swings are confirmed `window` bars late — live signals are delayed by
   `window × 4 hours` (12 hours for window=3)
5. Training and deployment match perfectly

**Trade-off:** Live signals arrive 12 hours after the swing is formed. For a
system targeting 2-3 trades/day on H4, this delay may be acceptable — the
scanner runs on H4 candle close anyway, and the swing needs confirmation.

### Recommendation

Test both options in MVP2 paper trading phase (60 days). Compare:
- Signal frequency
- Feature value alignment (log live features vs training distribution)
- Model confidence scores on live signals

## Key Principle

> Train on what you deploy with. If the feature distributions seen in
> production differ from training, the model's calibrated probabilities
> become unreliable. This is the most common source of silent model
> degradation in live trading systems.

---

*Created during Phase 2 indicator validation, March 2026*
*Review before MVP2 live deployment*