# PINN Optimization Quick Reference

**Current Latency:** 1600ms | **Target:** <1000ms | **Status:** ✅ Achievable with P0 fixes

---

## TL;DR - Root Causes

1. ❌ **Model instantiated TWICE per API call** (call + put) → 1000ms wasted
2. ❌ **Nested GradientTapes for Greeks** → 300ms overhead
3. ❌ **No caching** → Full initialization cost on every request
4. ✅ **Weights ARE loaded correctly** (157KB valid trained model)

---

## Critical Fixes (P0 - Must Have)

### Fix #1: Model Caching (-1000ms)

**Problem:** Fresh OptionPricingPINN instantiation on every API call

**Solution:**
```python
_pinn_model_cache: Optional[OptionPricingPINN] = None

async def get_cached_pinn_model():
    global _pinn_model_cache
    if _pinn_model_cache is None:
        _pinn_model_cache = OptionPricingPINN(...)
    return _pinn_model_cache

# In get_pinn_prediction():
pinn = await get_cached_pinn_model()  # 0ms after first call
```

**Expected gain:** -1000ms
**Risk:** Low
**Priority:** P0 - Implement first

---

### Fix #2: Optimized Greek Computation (-250ms)

**Problem:** Nested persistent GradientTapes

**Current (300ms):**
```python
with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape1:
        V = self.model(x)
```

**Solution (50ms):**
```python
@tf.function  # Compile to graph
def compute_greeks_fast(model, x):
    with tf.GradientTape() as tape:  # Non-persistent
        tape.watch(x)
        V = model(x, training=False)
    dV_dx = tape.gradient(V, x)
    delta = dV_dx[0, 0]
    theta = -dV_dx[0, 1]

    # Gamma via finite difference (faster than second autodiff)
    h = 0.01
    V_plus = model(tf.constant([[S+h, tau, K]]))
    V_minus = model(tf.constant([[S-h, tau, K]]))
    gamma = (V_plus - 2*V + V_minus) / h**2

    return V, delta, gamma, theta
```

**Expected gain:** -250ms
**Risk:** Low
**Priority:** P0

---

### Fix #3: Eliminate Dual Prediction (-550ms)

**Problem:** Separate put model instantiation for directional signal

**Current:**
```python
pinn = OptionPricingPINN(option_type='call')  # 500ms
pinn_put = OptionPricingPINN(option_type='put')  # 500ms more!
```

**Solution:** Use call-put parity formula
```python
# Call-Put Parity: P = C - (S - K*e^(-r*τ))
call_premium = pinn.predict(S, K, tau)['price']
put_premium = call_premium - (S - K * np.exp(-r * tau))

# No second model needed!
```

**Expected gain:** -550ms
**Risk:** Medium - requires validation (European options only)
**Priority:** P0 - but validate directional signal accuracy first

---

## Latency Breakdown

| Component | Current | After P0.1 | After P0.2 | After P0.3 | Target |
|-----------|---------|-----------|-----------|-----------|--------|
| **Call model instantiation** | 500ms | 0ms | 0ms | 0ms | 0ms |
| **Put model instantiation** | 500ms | 500ms | 500ms | 0ms | 0ms |
| **Weight loading** | 100ms | 0ms | 0ms | 0ms | 0ms |
| **Greek computation** | 300ms | 300ms | 50ms | 50ms | <100ms |
| **Forward passes** | 100ms | 100ms | 100ms | 50ms | <100ms |
| **Overhead** | 100ms | 100ms | 100ms | 100ms | <100ms |
| **TOTAL** | **1600ms** | **1000ms** | **750ms** | **200ms** | **<1000ms** ✅ |

---

## Implementation Checklist

### Phase 1 (Week 1) - Critical Path

- [ ] **P0.1:** Implement model caching
  - [ ] Create global `_pinn_model_cache` singleton
  - [ ] Add thread-safe async lock
  - [ ] Add warm-up prediction on first initialization
  - [ ] Test: Verify 0ms instantiation after first call

- [ ] **P0.2:** Optimize Greek computation
  - [ ] Create `compute_greeks_fast()` with single tape
  - [ ] Add `@tf.function` decorator for graph compilation
  - [ ] Use finite differences for gamma
  - [ ] Add fallback to price-only if Greeks fail
  - [ ] Test: Verify Greeks match analytical BS within 1%

- [ ] **P0.3:** Remove dual prediction
  - [ ] Replace put model with call-put parity formula
  - [ ] Update directional signal calculation
  - [ ] Add A/B test flag: `USE_CALL_PUT_PARITY` (default: True)
  - [ ] Test: Validate directional accuracy on historical data
  - [ ] Monitor: Track signal quality in production

### Phase 2 (Week 2) - Validation

- [ ] **End-to-end latency test**
  - [ ] Measure `/api/forecast/all` response time
  - [ ] Target: <1000ms p95, <500ms p50
  - [ ] Compare directional signal accuracy: before vs after

- [ ] **Regression testing**
  - [ ] Verify price predictions match pre-optimization
  - [ ] Validate Greeks accuracy (Delta, Gamma, Theta)
  - [ ] Check directional bias fix still works

---

## Optional Improvements (P1/P2)

### P1: Further Optimizations

1. **Swish activation** (-5% training time, +2-5% accuracy)
   - Replace `tanh` → `swish` in hidden layers
   - Requires retraining model

2. **Model quantization** (INT8: -75% size, +2x inference speed)
   - Use TFLite converter
   - For edge deployment only

3. **Batch prediction API** (10x speedup for portfolios)
   - Single batched forward pass for N symbols

### P2: Accuracy Improvements

1. **Enhanced training**
   - Increase samples: 10K → 50K
   - Early stopping + LR scheduling
   - Wider tau range: 0.01-2.0 years

2. **Physics weight tuning**
   - Grid search: [1.0, 5.0, 10.0, 20.0, 50.0]
   - Select based on validation MSE + PDE residual

---

## Risk Assessment

| Fix | Risk | Mitigation | Priority |
|-----|------|------------|----------|
| **Model caching** | Low | Standard pattern, add cache invalidation | P0 |
| **Greek optimization** | Low | tf.function is production-tested | P0 |
| **Dual prediction removal** | **Medium** | **Validate on test set, add A/B flag** | **P0** |

**Key Risk:** P0.3 (call-put parity)
- Assumes European options (no early exercise)
- American options violate parity → may affect directional signal
- **Mitigation:** A/B test, fallback to dual model if signal degrades

---

## Success Metrics

### Latency Targets ✅

- **Total latency:** 1600ms → <500ms (67% reduction)
- **Model instantiation:** 1000ms → 0ms (100% reduction)
- **Greek computation:** 300ms → 50ms (83% reduction)

### Accuracy Targets

- **Price MAE:** <$1.00 → <$0.50 (current validation passed)
- **Greek accuracy:** ~90% → >95% (vs analytical Black-Scholes)
- **PDE residual:** <0.01 → <0.001 (better physics constraint satisfaction)

---

## Code Locations

| Component | File | Line |
|-----------|------|------|
| **PINN model class** | `src/ml/physics_informed/general_pinn.py` | 339-535 |
| **API integration** | `src/api/ml_integration_helpers.py` | 703-856 |
| **Unified routes** | `src/api/unified_routes.py` | 316-330 |
| **Training script** | `scripts/train_pinn_model.py` | 24-98 |
| **Weight file** | `models/pinn/option_pricing/model.weights.h5` | 157KB |

---

## Debugging Tips

### Verify weights are loaded:
```python
pinn = OptionPricingPINN()
result = pinn.predict(S=100.0, K=100.0, tau=1.0)
bs_price = pinn.black_scholes_price(S=100.0, K=100.0, tau=1.0)
error = abs(result['price'] - bs_price)
if error > 5.0:
    logger.warning(f"PINN weights may not be loaded! Error: ${error:.2f}")
```

### Profile latency:
```python
import time

t0 = time.time()
pinn = OptionPricingPINN()  # Measure instantiation
t1 = time.time()
result = pinn.predict(S=100.0, K=100.0, tau=1.0)  # Measure inference
t2 = time.time()

print(f"Instantiation: {(t1-t0)*1000:.0f}ms")
print(f"Inference: {(t2-t1)*1000:.0f}ms")
```

### Check Greek accuracy:
```python
# Compare PINN Greeks vs Black-Scholes analytical
from scipy.stats import norm

# Analytical Delta
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
bs_delta = norm.cdf(d1)

pinn_delta = pinn.predict(S, K, tau)['delta']
error = abs(pinn_delta - bs_delta)
assert error < 0.01, f"Delta error too high: {error:.4f}"
```

---

## Next Steps

1. **Read full architecture review:** `PINN_ARCHITECTURE_REVIEW.md`
2. **Implement P0 fixes** in order: caching → Greeks → dual prediction
3. **Test after each fix** to isolate performance gains
4. **Validate directional signal** before deploying P0.3
5. **Monitor production metrics** for regression

---

**Questions?** Refer to detailed analysis in `PINN_ARCHITECTURE_REVIEW.md`
