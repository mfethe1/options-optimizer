# ML REMEDIATION EXECUTION PLAN
**Agent Orchestration Strategy for Production Readiness**

**Document Version:** 1.0
**Created:** 2025-11-09
**Orchestrator:** Agent Orchestration Specialist
**Status:** EXECUTION APPROVED - Scenario A (FULL FIX)
**Timeline:** 8-12 weeks (Starting Week 1 immediately)

---

## EXECUTIVE DASHBOARD

### Current State
- **Grade:** D- (85% catastrophic failure risk)
- **Test Status:** 288/299 passing (96.3%)
- **P0 Showstoppers:** 7 critical issues
- **P1 Major Issues:** 5 concerns
- **P2 Technical Debt:** 11 items

### Target State
- **Grade:** A- (70%+ directional accuracy)
- **Test Coverage:** >95% with integration tests
- **Production Risk:** <5% failure probability
- **SLA Compliance:** <500ms API latency (p95)

### Week 1 Objectives (P0 Emergency Fixes)
1. Fix data leakage (augmentation timing)
2. Fix look-ahead bias (feature engineering)
3. Add file locking (race conditions)
4. Train PINN model (1000+ epochs)
5. Optimize PINN latency (<1000ms)
6. Add TensorFlow error handling
7. Implement MambaModel.build()

---

## PHASE 1: WEEK 1 EMERGENCY FIXES

### Parallel Execution Strategy

**Critical Path Analysis:**
- **Longest Sequence:** PINN training (16-20 hours) → PINN optimization (8-10 hours) = 28 hours
- **Independent Tasks:** Data leakage, look-ahead bias, race conditions, error handling, build() method
- **Optimal Parallelization:** Launch 5-7 agents simultaneously

### Task Assignments (Parallel Launch)

#### AGENT 1: Data Leakage Fix (expert-code-writer)
**Task ID:** P0-1
**Priority:** CRITICAL
**Estimated Time:** 6-8 hours
**Dependencies:** None (can start immediately)
**Blocking:** Model retraining (Phase 2)

**Files to Modify:**
- `scripts/train_mamba_models.py` (lines 486-494, 534-542)
- `src/ml/state_space/data_preprocessing.py` (augment_data method)

**Deliverables:**
1. Move augmentation AFTER train/val split
2. Implement proper temporal split (80/20)
3. Add test: `test_augmentation_after_split()`
4. Verify validation metrics are uncontaminated

**Success Criteria:**
- Augmentation only applied to training data
- Validation set has NO augmented samples
- New test passes: `test_no_data_leakage_in_validation()`
- Training metrics change (validation loss likely increases 20-40%)

**Acceptance Test:**
```python
def test_augmentation_after_split():
    X, y = create_sequences(prices)
    X_train, X_val, y_train, y_val = split_sequences(X, y, split=0.8)
    X_aug, y_aug = augment_data(X_train, y_train)

    # Validation set unchanged
    assert len(X_val) == int(len(X) * 0.2)
    assert np.array_equal(X_val, X[int(len(X)*0.8):])

    # Training set augmented
    assert len(X_aug) > len(X_train)
```

---

#### AGENT 2: Look-Ahead Bias Fix (expert-code-writer)
**Task ID:** P0-2
**Priority:** CRITICAL
**Estimated Time:** 8-10 hours
**Dependencies:** None (can start immediately)
**Blocking:** Model retraining (Phase 2)

**Files to Modify:**
- `src/ml/state_space/data_preprocessing.py` (lines 120-165)
- Specifically: `extract_features()` method

**Problem Areas:**
```python
# LINE 153-154: USES FUTURE DATA
sma_5 / (np.mean(price_history) + 1e-8),  # np.mean() uses all prices!
sma_60 / (np.mean(price_history) + 1e-8),
```

**Required Changes:**
1. Replace `np.mean(price_history)` with expanding window mean
2. Ensure all features use ONLY past data up to current timestep
3. Implement `expanding_mean()` helper function
4. Validate temporal integrity for all 25+ features

**Deliverables:**
1. New helper: `expanding_mean(data) -> rolling mean using only past values`
2. Fix all feature normalization to use expanding statistics
3. Add test: `test_no_future_data_in_features()`
4. Document: FEATURE_TEMPORAL_INTEGRITY.md

**Success Criteria:**
- All features computed using only t-N to t data (no t+1)
- Test validates: features[i] computed using only data[:i+1]
- Production accuracy drop expected (model was cheating with future data)

**Acceptance Test:**
```python
def test_no_future_data_in_features():
    prices = np.array([100, 101, 99, 102, 98])

    # Extract features at each timestep
    for i in range(1, len(prices)):
        features_at_t = extract_features(prices[:i+1])

        # Features should NOT change when future data added
        features_at_t_plus_1 = extract_features(prices[:i+2])
        assert np.allclose(features_at_t[:-1], features_at_t_plus_1[:-1])
```

---

#### AGENT 3: Race Condition Fix (expert-code-writer)
**Task ID:** P0-3
**Priority:** CRITICAL
**Estimated Time:** 10-12 hours
**Dependencies:** None (can start immediately)
**Blocking:** Parallel training reliability

**Files to Modify:**
- `scripts/train_mamba_models.py` (lines 450-480, 560-570, 656-671)
- New file: `src/utils/file_locking.py`

**Problem Areas:**
```python
# LINE 560-562: CONCURRENT WRITES
weights_path = os.path.join(save_dir, 'weights', f'{symbol}.weights.h5')
os.makedirs(os.path.dirname(weights_path), exist_ok=True)
model.save_weights(weights_path)  # NO LOCKING!
```

**Required Changes:**
1. Implement cross-platform file locking (fcntl on Unix, msvcrt on Windows)
2. Create atomic file write helper (write to temp, then rename)
3. Add per-symbol subdirectories to reduce contention
4. Implement retry logic with exponential backoff

**Deliverables:**
1. New module: `src/utils/file_locking.py`
   - `FileLock` context manager (cross-platform)
   - `atomic_write(path, data)` helper
   - `safe_makedirs(path)` with locking
2. Update all file writes in training script
3. Add test: `test_parallel_training_no_corruption()`
4. Document: FILE_LOCKING_STRATEGY.md

**Success Criteria:**
- 10 parallel training runs complete without file corruption
- No partial writes or malformed JSON
- Test validates: concurrent writes to same directory succeed

**Acceptance Test:**
```python
def test_parallel_training_no_corruption():
    async def train_batch():
        symbols = ['AAPL', 'MSFT', 'GOOGL'] * 5  # 15 symbols
        results = await asyncio.gather(
            *[train_single_symbol(sym, save_dir) for sym in symbols],
            return_exceptions=True
        )

        # All succeed
        assert all(not isinstance(r, Exception) for r in results)

        # All files intact
        for sym in set(symbols):
            assert os.path.exists(f'{save_dir}/weights/{sym}.weights.h5')
            # Verify file is valid HDF5 (not corrupted)
            with h5py.File(f'{save_dir}/weights/{sym}.weights.h5', 'r') as f:
                assert len(f.keys()) > 0
```

---

#### AGENT 4: PINN Training (ml-neural-network-architect + expert-code-writer)
**Task ID:** P0-4
**Priority:** CRITICAL
**Estimated Time:** 16-20 hours (longest task)
**Dependencies:** None (can start immediately)
**Blocking:** PINN optimization (P0-5), production deployment

**Files to Modify:**
- `src/ml/physics_informed/general_pinn.py` (add training loop)
- New file: `scripts/train_pinn_models.py`
- New file: `tests/test_pinn_training.py`

**Problem:**
- PINN model currently uses RANDOM WEIGHTS (never trained)
- Call-Put Parity error: 10-15% (should be <2%)
- Predictions are pure noise

**Training Strategy:**
1. **Data Generation:** Synthetic option data with Black-Scholes pricing
2. **Physics Loss:** PDE residual + boundary conditions + Call-Put Parity
3. **Training Epochs:** 1000-2000 (until physics loss < 0.01)
4. **Validation:** 30-day walk-forward on real option data

**Deliverables:**
1. PINN training script with physics-informed loss
2. Trained model weights: `models/pinn/option_pricing/weights.h5`
3. Training metrics dashboard
4. Validation report: PINN_TRAINING_REPORT.md
5. Tests: `test_pinn_call_put_parity()`, `test_pinn_greeks_accuracy()`

**Success Criteria:**
- Call-Put Parity error: <2% (currently 10-15%)
- Physics loss: <0.01 (PDE residual)
- Delta accuracy: within 5% of Black-Scholes
- Gamma accuracy: within 10% of Black-Scholes

**Acceptance Test:**
```python
def test_pinn_call_put_parity():
    pinn = OptionPricingPINN(option_type='call', r=0.05, sigma=0.2, physics_weight=10.0)
    pinn.load_weights('models/pinn/option_pricing/weights.h5')

    S, K, tau = 100.0, 100.0, 0.25

    call_price = pinn.predict(S=S, K=K, tau=tau)['price']
    pinn.option_type = 'put'
    put_price = pinn.predict(S=S, K=K, tau=tau)['price']

    # Call-Put Parity: C - P = S - K*e^(-r*τ)
    actual_diff = call_price - put_price
    theoretical_diff = S - K * np.exp(-0.05 * tau)
    parity_error = abs(actual_diff - theoretical_diff) / S

    assert parity_error < 0.02, f"Parity error {parity_error:.3f} exceeds 2%"
```

---

#### AGENT 5: PINN Latency Optimization (expert-code-writer)
**Task ID:** P0-5
**Priority:** CRITICAL
**Estimated Time:** 8-10 hours
**Dependencies:** PINN training (P0-4) must complete first
**Blocking:** Production SLA compliance

**Files to Modify:**
- `src/api/ml_integration_helpers.py` (lines 752-818)
- New file: `src/ml/physics_informed/pinn_cache.py`

**Problem:**
- Current latency: 1600ms (2x slowdown from "fix")
- SLA target: <500ms (p95)
- Root cause: Creates new model instance for EVERY prediction

**Current Code (BAD):**
```python
# LINE 757-764: CREATES NEW MODEL EVERY TIME
pinn_call = OptionPricingPINN(option_type='call', r=r, sigma=sigma, physics_weight=10.0)
pinn_put = OptionPricingPINN(option_type='put', r=r, sigma=sigma, physics_weight=10.0)
```

**Optimization Strategy:**
1. **Model Caching:** Singleton pattern for PINN instances
2. **Batch Prediction:** Vectorize multiple predictions
3. **Put Price Caching:** Cache put prices by (K, tau)
4. **Model Reuse:** Single model with option_type parameter

**Deliverables:**
1. PINN cache manager: `src/ml/physics_informed/pinn_cache.py`
2. Batch prediction API: `batch_predict(options_list)`
3. Performance test: `test_pinn_latency_under_1s()`
4. Benchmark report: PINN_OPTIMIZATION_REPORT.md

**Success Criteria:**
- Latency: <1000ms for single prediction (p95)
- Latency: <500ms with caching (p95)
- Batch throughput: >50 predictions/sec
- Memory usage: <200 MB (vs 500 MB before)

**Acceptance Test:**
```python
def test_pinn_latency_under_1s():
    pinn_cache = PINNCache()

    start = time.time()
    for _ in range(100):
        result = pinn_cache.predict_with_cache(
            S=100, K=100, tau=0.25, r=0.05, sigma=0.2, option_type='call'
        )
    elapsed = time.time() - start

    avg_latency = elapsed / 100
    assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}s exceeds 1s"

    # With cache (subsequent calls)
    start = time.time()
    for _ in range(100):
        result = pinn_cache.predict_with_cache(S=100, K=100, tau=0.25, r=0.05, sigma=0.2, option_type='call')
    elapsed = time.time() - start

    avg_latency_cached = elapsed / 100
    assert avg_latency_cached < 0.5, f"Cached latency {avg_latency_cached:.3f}s exceeds 500ms"
```

---

#### AGENT 6: TensorFlow Error Handling (expert-code-writer)
**Task ID:** P0-6
**Priority:** CRITICAL
**Estimated Time:** 10-12 hours
**Dependencies:** None (can start immediately)
**Blocking:** Production reliability

**Files to Modify:**
- `scripts/train_mamba_models.py` (lines 467-468, 534-542)
- `src/ml/state_space/mamba_model.py` (training methods)
- `src/ml/physics_informed/general_pinn.py` (training methods)
- All ML training scripts

**Problem:**
- NO ERROR HANDLING around TensorFlow operations
- GPU OOM crashes lose all training progress
- NaN loss causes infinite loops
- CUDA errors crash silently

**Current Code (BAD):**
```python
# LINE 534-542: NO TRY/CATCH
history = model.fit(X, y, epochs=epochs, batch_size=batch_size, ...)
# Crashes on GPU OOM, NaN loss, CUDA errors
```

**Required Error Handling:**
1. **GPU OOM:** Retry with smaller batch size (32 → 16 → 8)
2. **CUDA Errors:** Fallback to CPU with warning
3. **NaN Loss:** Early stopping, log gradients, restart with lower LR
4. **KeyboardInterrupt:** Save progress, cleanup resources
5. **Disk Full:** Warn user, cleanup old checkpoints

**Deliverables:**
1. New module: `src/utils/training_error_handler.py`
   - `with safe_training() context manager`
   - `handle_gpu_oom(exception, state)` retry logic
   - `handle_nan_loss(model, history)` recovery
2. Update all training scripts with error handling
3. Add tests: `test_gpu_oom_recovery()`, `test_nan_loss_handling()`
4. Document: ERROR_HANDLING_GUIDE.md

**Success Criteria:**
- GPU OOM triggers automatic batch size reduction
- NaN loss triggers early stopping (not infinite loop)
- Keyboard interrupt saves partial progress
- All errors logged with actionable messages

**Acceptance Test:**
```python
def test_gpu_oom_recovery():
    # Simulate GPU OOM
    with patch('tensorflow.keras.Model.fit', side_effect=tf.errors.ResourceExhaustedError):
        handler = TrainingErrorHandler(initial_batch_size=32)

        result = handler.train_with_recovery(model, X, y)

        # Should retry with batch_size=16, then 8
        assert handler.current_batch_size == 8
        assert result['recovery_attempts'] == 2

def test_nan_loss_handling():
    # Simulate NaN loss
    def fit_with_nan(*args, **kwargs):
        history = MagicMock()
        history.history = {'loss': [0.5, 0.3, np.nan, np.nan]}
        return history

    with patch('tensorflow.keras.Model.fit', side_effect=fit_with_nan):
        handler = TrainingErrorHandler()
        result = handler.train_with_recovery(model, X, y)

        # Should detect NaN and stop early
        assert result['stopped_early'] == True
        assert 'NaN loss detected' in result['stop_reason']
```

---

#### AGENT 7: MambaModel.build() Implementation (expert-code-writer)
**Task ID:** P0-7
**Priority:** CRITICAL
**Estimated Time:** 6-8 hours
**Dependencies:** None (can start immediately)
**Blocking:** Model serialization reliability

**Files to Modify:**
- `src/ml/state_space/mamba_model.py` (add build() method)

**Problem:**
- Keras warning: "layer does not have a build() method implemented"
- Model weights not properly initialized
- Weight loading/saving may fail silently
- Transfer learning won't work

**Current Warning:**
```
WARNING: `build()` was called on layer 'mamba_model', however the layer does not have
a `build()` method implemented and it looks like it has unbuilt state.
```

**Required Changes:**
1. Implement proper `build(input_shape)` method
2. Initialize all layer weights in build()
3. Call `super().build(input_shape)`
4. Set `self.built = True` only after initialization

**Deliverables:**
1. Proper `MambaModel.build()` implementation
2. Test: `test_mamba_model_build_proper()`
3. Verify no Keras warnings during training
4. Document: MAMBA_ARCHITECTURE.md

**Success Criteria:**
- No Keras warnings during model.fit()
- Model properly serializes/deserializes
- Transfer learning works (load partial weights)

**Acceptance Test:**
```python
def test_mamba_model_build_proper():
    config = MambaConfig()
    model = MambaModel(config)

    # Should NOT be built yet
    assert not model.built

    # Call build explicitly
    model.build((None, 60, 25))

    # Should be built now
    assert model.built

    # Verify all layers initialized
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            assert layer.kernel is not None

    # Train without warnings (capture stderr)
    import io, sys
    stderr_capture = io.StringIO()
    sys.stderr = stderr_capture

    model.fit(X_dummy, y_dummy, epochs=1, verbose=0)

    sys.stderr = sys.__stderr__
    warnings = stderr_capture.getvalue()

    assert 'build()' not in warnings, f"Keras warnings still present: {warnings}"
```

---

## EXECUTION TIMELINE - WEEK 1

### Day 1 (Monday): Launch & Setup
**Morning:**
- Launch Agents 1, 2, 3, 6, 7 in parallel (independent tasks)
- Agent briefings with detailed specifications
- Setup monitoring dashboards

**Afternoon:**
- Monitor progress, resolve blockers
- Agents report initial findings
- Prepare PINN training data

**Evening Status:**
- 5 agents actively working
- No blockers expected (independent tasks)

---

### Day 2 (Tuesday): First Results
**Morning:**
- Agent 7 (MambaModel.build) completes (6-8 hours)
- Agent 1 (Data leakage) completes (6-8 hours)
- Launch Agent 4 (PINN training) - longest task

**Afternoon:**
- Agent 2 (Look-ahead bias) completes (8-10 hours)
- Code reviews for completed tasks
- Integration testing begins

**Evening Status:**
- 3 tasks complete (build, data leakage, look-ahead bias)
- PINN training in progress (16-20 hours)
- 3 agents still working (race conditions, error handling, PINN)

---

### Day 3 (Wednesday): Mid-Week Checkpoint
**Morning:**
- Agent 3 (Race conditions) completes (10-12 hours)
- Agent 6 (Error handling) completes (10-12 hours)
- Agent 4 (PINN training) 50% complete

**Afternoon:**
- Integration testing for completed tasks
- Resolve any merge conflicts
- Monitor PINN training progress

**Evening Status:**
- 5 tasks complete
- PINN training 75% complete
- No blockers

---

### Day 4 (Thursday): PINN Training Complete
**Morning:**
- Agent 4 (PINN training) completes (16-20 hours total)
- Begin PINN validation tests
- Launch Agent 5 (PINN optimization)

**Afternoon:**
- PINN optimization work begins
- Full integration test run
- Bug fixes from integration testing

**Evening Status:**
- 6 tasks complete
- PINN optimization in progress (8-10 hours)
- Integration tests passing

---

### Day 5 (Friday): Wrap-Up & Validation
**Morning:**
- Agent 5 (PINN optimization) completes (8-10 hours)
- Full test suite run (all 299+ tests)
- Performance benchmarks

**Afternoon:**
- Final code reviews
- Documentation updates
- Week 1 completion report

**Evening Status:**
- ALL 7 P0 TASKS COMPLETE
- Tests passing: target >290/299
- Ready for Phase 2 (Model Retraining)

---

## COORDINATION & MONITORING

### Daily Standup (9 AM EST)
**Agenda:**
1. Agent status updates (each 2 min)
2. Blockers and dependencies
3. Resource allocation
4. Risk assessment

**Attendees:** All active agents + orchestrator

---

### Progress Tracking Dashboard

**Metrics:**
- Tasks completed / total (target: 7/7 by Friday)
- Test pass rate (target: >95%)
- Code coverage (target: >90%)
- Integration health (green/yellow/red)

**Daily Reports:**
- Morning: Overnight progress
- Evening: Day summary + next day plan

---

### Risk Mitigation

**Identified Risks:**
1. **PINN training takes longer than 20 hours**
   - Mitigation: Reduce epochs to 500, accept 5% parity error
   - Fallback: Defer PINN to Week 2, focus on other fixes

2. **Race condition testing requires multi-core machine**
   - Mitigation: Use GitHub Actions runner with 8 cores
   - Fallback: Test with 4 parallel workers instead of 15

3. **Integration conflicts between agents**
   - Mitigation: Assign unique file ownership, minimal overlaps
   - Fallback: Sequential merges with manual conflict resolution

4. **TensorFlow version incompatibilities**
   - Mitigation: Lock TensorFlow==2.16.1 in requirements
   - Fallback: Test with TF 2.15 if issues arise

---

## SUCCESS CRITERIA - WEEK 1

### Must-Have (Go/No-Go)
- [ ] All 7 P0 issues fixed
- [ ] No new test failures introduced
- [ ] Integration tests pass
- [ ] Documentation updated

### Should-Have (Nice-to-Have)
- [ ] Performance benchmarks showing improvements
- [ ] PINN parity error <2%
- [ ] PINN latency <1000ms
- [ ] No Keras warnings

### Quality Gates
- [ ] Code review approval from 2+ reviewers
- [ ] Test coverage >90%
- [ ] No P0/P1 bugs introduced
- [ ] Backward compatibility maintained

---

## PHASE 2 PREVIEW (WEEK 2)

**Objective:** Retrain all models with fixed pipeline

**Dependencies:** Week 1 P0 fixes MUST be complete

**Tasks:**
1. Retrain MAMBA models (50 symbols)
2. Validate walk-forward performance
3. Get true baseline metrics (expect 20-40% accuracy drop due to leakage fix)
4. Identify models to keep vs retrain

**Expected Outcome:** Know true model performance without data leakage

---

## DELIVERABLES - WEEK 1

### Code Changes
1. Fixed training pipeline (data leakage, look-ahead bias)
2. File locking implementation
3. Trained PINN weights
4. PINN optimization cache
5. TensorFlow error handling
6. MambaModel.build() implementation

### Tests
1. `test_augmentation_after_split()`
2. `test_no_future_data_in_features()`
3. `test_parallel_training_no_corruption()`
4. `test_pinn_call_put_parity()`
5. `test_pinn_latency_under_1s()`
6. `test_gpu_oom_recovery()`
7. `test_mamba_model_build_proper()`

### Documentation
1. FEATURE_TEMPORAL_INTEGRITY.md
2. FILE_LOCKING_STRATEGY.md
3. PINN_TRAINING_REPORT.md
4. PINN_OPTIMIZATION_REPORT.md
5. ERROR_HANDLING_GUIDE.md
6. MAMBA_ARCHITECTURE.md
7. WEEK_1_COMPLETION_REPORT.md

---

## APPROVAL & SIGN-OFF

**Prepared by:** Agent Orchestration Specialist
**Reviewed by:** [Stakeholder]
**Approved by:** [Technical Lead]

**Approval Date:** 2025-11-09
**Execution Start:** 2025-11-09 (Immediately)

---

**Next Steps:**
1. Agent briefings (1 hour)
2. Launch parallel execution (5 agents)
3. Daily standups at 9 AM EST
4. Friday: Week 1 completion report

**LET'S FIX THIS SYSTEM AND BUILD SOMETHING WORLD-CLASS.**
