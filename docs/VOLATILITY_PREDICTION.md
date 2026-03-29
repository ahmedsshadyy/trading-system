# VOLATILITY_PREDICTION

## Status

**Phase status:** Deferred to **Phase 6 MVP1**  
**Current canonical decision:** The existing volatility layer is **frozen for now** and accepted as the live-safe baseline.

This means the current implementation remains the canonical volatility context layer for downstream work in the current phase, including:

- ATR-based normalization and context
- BB-width compression / expansion context
- realized volatility family
- Parkinson range volatility
- fixed-parameter recursive GARCH(1,1)
- filtered stochastic volatility (SV) layer
- bridge ratios and volatility-state flags

The current layer is considered:

- **causal**
- **live-safe**
- **research-usable**
- **sufficiently stable for current downstream dependencies**

but **not fully finalized as the ultimate volatility modeling stack**.

---

## Why frozen now

The current validation results indicate that the volatility stack is structurally sound:

- warmup behavior is explicit
- no `inf` contamination is present
- parity checks pass
- GARCH readiness is explicitly defined and live-safe
- realized volatility and TV historical volatility match extremely closely after scale alignment
- cross-family volatility measures are numerically coherent
- bridge ratios are stable and non-pathological
- filtered SV is smoother than GARCH and behaves as a distinct auxiliary volatility series

Accordingly, the remaining issues are **not core correctness blockers**. They are primarily:

- statistical tail-modeling refinements
- comparative model diagnostics
- enhanced regime integration
- cross-family interpretation hardening

These are better handled in a later dedicated phase rather than blocking current scanner / trend / regime / S&R work.

---

## Frozen canonical doctrine for current phase

### 1. ATR remains canonical
ATR remains the primary downstream-safe volatility backbone for:

- distance normalization
- sweep size normalization
- displacement scaling
- AVWAP / VP distance scaling
- stop/target scale context
- broad risk-state context

### 2. GARCH remains canonical model-based volatility
The current canonical GARCH implementation is accepted as:

- **fixed-parameter**
- **recursive**
- **live-safe**
- **initialized from the first 500 valid return rows**
- **first live-safe row = 501**

It is accepted as a **practical conditional-volatility estimate**, not as a perfect tail model.

### 3. Filtered SV remains auxiliary
Filtered SV remains accepted as:

- a **live-safe filtered-only latent-volatility proxy**
- smoother than GARCH
- an auxiliary context layer
- not yet a mandatory downstream dependency

### 4. TradingView comparison conclusion
The TradingView “GARCH Volatility Estimation - The Quant Science” script was reviewed and treated as a **benchmark sanity reference**, not as the source of truth.

Conclusion:

- TradingView historical volatility aligns very closely with our scaled realized volatility implementation
- the TradingView “GARCH” proxy is **not** the same object as our canonical return-based GARCH
- mismatch between TV GARCH proxy and canonical GARCH is therefore expected and is **not treated as an implementation bug**

---

## Known limitations accepted for current phase

The following issues are acknowledged and explicitly accepted for deferral:

### A. Heavy Gaussian GARCH residual tails
The current Gaussian GARCH residual diagnostics show heavy tails relative to ideal standardized behavior.

This is treated as a **known modeling limitation**, not a current implementation failure.

Implication:
- canonical GARCH is acceptable for current use
- tail-fit refinement is deferred

### B. Student-t comparison not yet trustworthy enough
The current Student-t comparison block is not yet considered reliable enough for doctrinal sign-off.

Implication:
- do not use current Student-t comparison as evidence for replacing canonical Gaussian GARCH
- revisit in Phase 6 MVP1

### C. No regime-conditioned volatility audit yet
Volatility has not yet been fully analyzed conditional on canonical regime / trend context.

Implication:
- no final “quiet vs volatile” enhanced regime mapping is frozen yet
- this remains deferred

### D. No final event-response doctrine yet
Shock-response diagnostics exist, but their downstream doctrinal interpretation is not yet fully frozen.

Implication:
- ATR / RV / GARCH / SV role split is directionally clear
- but the final policy for enhanced volatility-state classification remains deferred

### E. SV usefulness is not fully proven yet
Filtered SV has been shown to be smoother and non-redundant, but its full incremental value over GARCH is not yet fully established.

Implication:
- keep SV available
- do not yet promote it to primary downstream volatility driver

---

## Deferred to Phase 6 MVP1

The following tasks are explicitly deferred to **Phase 6 MVP1**.

### 1. GARCH tail-model refinement
- audit and fix Student-t residual comparison
- compare Gaussian vs Student-t GARCH under a properly standardized residual framework
- decide whether canonical GARCH remains Gaussian or gains a research-approved Student-t companion

### 2. Regime-conditioned volatility audit
Add full volatility-by-regime diagnostics for:

- ATR percentile
- RV20
- GARCH volatility
- filtered SV
- volatility-state flags

This is required before freezing any enhanced regime axis such as:

- Bull Quiet
- Bull Volatile
- Bear Quiet
- Bear Volatile
- Sideways Quiet
- Sideways Volatile / Chop

### 3. Event-based shock-response doctrine
Perform a dedicated event study comparing:

- ATR
- RV20
- GARCH
- filtered SV

around major absolute-return shock events at:
- t-1
- t
- t+1
- t+3
- t+5

Then freeze the doctrinal role of each family in shock-state interpretation.

### 4. SV vs GARCH disagreement analysis
Study disagreement windows between filtered SV and GARCH, including:
- regime context
- session context
- return magnitude
- persistence of disagreement
- high-vol state interpretation

### 5. Final bridge-ratio doctrine
Freeze the semantic interpretation and downstream meaning of:
- `garch_to_atr_ratio`
- `garch_to_rv20_ratio`
- `sv_to_atr_ratio`
- `sv_to_garch_ratio`
- `atr_vs_rv20_ratio`
- `parkinson_vs_rv20_ratio`

### 6. Final volatility family role split
Freeze the final doctrinal assignment of roles across:
- ATR
- RV / Parkinson / other realized-vol estimators
- GARCH
- filtered SV

especially for:
- enhanced regime
- volatility-state labeling
- scanner gating
- ML feature subsets

---

## Current downstream usage policy

Until Phase 6 MVP1, downstream work should assume the following:

### Safe to use now
- ATR family
- BB-width family
- RV family
- Parkinson range volatility
- canonical GARCH volatility
- filtered SV as optional auxiliary context
- bridge ratios
- existing volatility-state flags

### Not yet frozen for final semantic use
- Student-t GARCH conclusions
- quiet vs volatile enhanced regime mapping
- volatility-conditioned regime taxonomy
- canonical conflict resolution between GARCH and SV when they disagree

---

## Practical guidance for current phases

Until Phase 6 MVP1:

- continue using ATR as the main normalization backbone
- continue using canonical GARCH as the primary model-based volatility estimate
- treat filtered SV as auxiliary context, not primary doctrinal truth
- do not block current development on remaining volatility refinements
- document any downstream dependency on unresolved volatility semantics explicitly

---

## Re-entry criteria for Phase 6 MVP1

Re-open the volatility layer in Phase 6 MVP1 only when all of the following are addressed:

1. Student-t comparison is statistically meaningful and correctly standardized  
2. volatility-by-regime diagnostics are implemented  
3. shock-response doctrine is frozen  
4. SV vs GARCH disagreement windows are explained  
5. enhanced volatility-state mapping is ready to be frozen  
6. final role split across ATR / RV / GARCH / SV is documented and accepted

---

## Final current decision

**Decision:** Freeze the current volatility layer now.  
**Rationale:** Current implementation is good enough for present downstream work. Remaining issues are important, but they are refinement and doctrine tasks better handled in a dedicated future phase rather than blocking current progress.