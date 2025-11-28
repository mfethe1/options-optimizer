# Multi-Horizon Forecasting Playbook (Post–Phase 1)

This playbook is for the **next coding agent** to carry the multi-horizon forecasting work from Phase 2 onward.

## 0. Current State After Phase 1

### Backend contract (as of this commit)

**Unified endpoint:** `POST /api/unified/forecast/all?symbol=SPY&time_range=1D`

Response structure (relevant fields):

- `timeline: List[Point]`
  - Each `Point` is **pure market data** (no model overlay):
    - `timestamp: ISO-8601 string`
    - `time: string` (formatted date/HH:MM)
    - `actual, open, high, low, volume`
- `predictions: Dict[str, List[PredictionPoint]]`
  - `model_id -> [ { time: string, predicted: float }, ... ]`
  - `time` values are **future dates** derived from the last `timeline` timestamp using trading-day logic.
- `metadata.models[model_id]`
  - Raw model metadata with *no* `prediction` / `upper_bound` / `lower_bound` keys (those are stripped).

### Model-level contract (Mamba is already multi-horizon)

`src/api/ml_integration_helpers.py::get_mamba_prediction` now returns:

- `prediction: float` (primary 30d point; backward compatible)
- `multi_horizon: { '1d', '5d', '10d', '30d' }`
- `horizons: [1, 5, 10, 30]`
- `trajectory: [p1d, p5d, p10d, p30d]`
- `confidence, status, timestamp, model, ...`

The fallback path also returns `horizons` + `trajectory`.

### UnifiedPredictionService multi-horizon plumbing

In `src/api/unified_routes.py`:

- `align_time_series(...)` now **only** builds OHLCV timeline and caches it (no predictions overlay at any point).
- `_compute_forecast_timestamps(base_timestamp, horizons_days)`
  - Uses NYSE calendar via `pandas_market_calendars` **if available**; otherwise skips weekends.
- `_extract_prediction_series(timeline, predictions)`
  - Uses the **last** timeline point as `base_timestamp`.
  - For each model:
    1. Prefer `horizons` + `trajectory` if present.
    2. Else, derive from `multi_horizon` dict like `{"1d": ..., "5d": ...}`.
    3. Else, fall back to repeating scalar `prediction` over `[1, 5, 10, 30]`.
  - Maps horizons to forecast timestamps via `_compute_forecast_timestamps`.
  - Returns future-only series: `{ time: 'YYYY-MM-DD', predicted: float }`.

Frontend `UnifiedAnalysisEnhanced` already expects `data.predictions[modelId]` as arrays of `{ time, predicted }` and will plot these series directly.

**Tests:**

- `py -m pytest tests/ -k "unified" -v` currently passes.

---

## 1. Phase 2 – Epidemic + TFT Multi-Horizon Integration

Goal: Make Epidemic Volatility (and TFT/advanced forecasting, if wired) emit **true multi-horizon trajectories** instead of a single scalar.

### 1.1. Epidemic Volatility

**Files to inspect:**
- `src/api/unified_routes.py::UnifiedPredictionService.get_epidemic_prediction`
- `src/ml/bio_financial/epidemic_volatility.py`

**Target contract for `get_epidemic_prediction` (additive, backward compatible):**

- Keep existing fields (VIX prediction, bounds, price-converted `prediction`, etc.).
- Add:
  - `horizons: List[int]` – e.g. `[1, 5, 10, 30]` **trading days ahead**.
  - `trajectory: List[float]` – predicted **price** (not VIX) at those horizons.

**Recommended steps:**

1. Inside `get_epidemic_prediction`, after computing `price_conversion`:
   - Build horizon set `[1, 5, 10, 30]`.
   - Use the underlying VIX model or a simple term-structure assumption to derive VIX at each horizon.
   - Convert each VIX horizon into a price forecast using the same `vix_to_price_change` logic.
   - Populate `horizons` + `trajectory` in the returned dict.
2. For the fallback (mock) path, do the same but based on `mock_predicted_vix`.
3. Verify via direct curl:
   - `POST /api/unified/forecast/all?symbol=SPY&time_range=1D`
   - Confirm `predictions.epidemic` has ~4 points with future `time`s.

### 1.2. TFT / AdvancedForecastService

If/when TFT is routed into unified predictions (via `advanced_forecast_service.py`):

1. Expose TFT’s native multi-horizon outputs as `horizons` + `trajectory` in whatever helper is used (e.g., `get_tft_prediction`).
2. Ensure the helper returns the same contract as Mamba (scalar `prediction`, plus `horizons` + `trajectory`).
3. Wire this into `UnifiedPredictionService.get_all_predictions` under a model id like `"tft"`.

**Validation:**

- Add/extend a backend test (e.g., in `tests/test_ml_integration_p0_fix.py` or a new file) to assert:
  - `len(response["predictions"]["epidemic"]) >= 2` and times are **future**.
  - If TFT integrated: `"tft" in response["predictions"]` with multiple forecast points.

---

## 2. Phase 3 – PINN & Mamba Refinement

### 2.1. Mamba sanity checks

Even though Mamba already exposes multi-horizon data, you should:

1. Confirm `predictor.predict(...)` returns reasonable per-horizon values (not all identical).
2. Optionally normalize horizons to **trading days** if Mamba is using calendar days.
3. Add tests that inspect `metadata.models["mamba"]["trajectory"]` for monotonicity or reasonable variance.

### 2.2. PINN multi-horizon extension

**Files:**
- `src/api/ml_integration_helpers.py::get_pinn_prediction`

Target: extend PINN to output `horizons` + `trajectory` of **underlying price forecasts**, even if internally it works in option space.

Suggested approach:

1. Define canonical horizons `[1, 5, 10, 30]`.
2. Use delta/vega structure or a simplified transformation to estimate underlying price at those horizons based on implied vol and risk-free rate.
3. Return:
   - `prediction`: primary horizon (e.g., 30d price or 10d price – pick one and document).
   - `horizons` + `trajectory` as described above.
   - Keep existing option Greeks and metadata.

**Tests:**

- New tests should assert that PINN’s `trajectory` is present and non-degenerate.

---

## 3. Phase 4 – GNN + Ensemble Upgrade

### 3.1. GNN multi-horizon outputs

**Files:**
- `src/api/ml_integration_helpers.py::get_gnn_prediction`
- `src/ml/graph_neural_network/...` (model implementation)

Goal: move from single-step return prediction to multi-horizon price or return paths.

Options:

1. **Model-level change** – retrain GNN to emit `H`-step-ahead returns.
2. **Post-processing** – for an initial iteration, estimate future trajectory from predicted return and historical volatility (e.g., geometric Brownian motion with correlation-aware adjustments).

Whichever you choose, ensure `get_gnn_prediction` returns `horizons` + `trajectory` with the usual contract.

### 3.2. Ensemble multi-horizon aggregation

Once multiple base models expose `trajectory`:

1. Implement an ensemble helper (if not already present) that:
   - Aligns horizon sets across models (e.g., intersection of `[1, 5, 10, 30]`).
   - Applies confidence-weighted averaging per horizon.
2. Return ensemble as another model in `predictions` (e.g., key `"ensemble"`).
3. Ensure `metadata.models["ensemble"]` includes per-model weights and agreement metrics.

**Tests:**

- Assert that ensemble predictions differ from individual models and that removing a low-confidence model changes the trajectory.

---

## 4. Frontend: Visual & UX Enhancements (Optional Phase)

While backend work proceeds, a future agent can:

1. Update `UnifiedAnalysisEnhanced` to visually distinguish **historical** vs **forecast** segments (e.g., dashed lines for future, vertical now-marker).
2. Add a small legend showing horizons (1D, 5D, 10D, 30D) when hovering over any prediction line.
3. Optionally surface `metadata.models[modelId].trajectory` in a side panel as a small numeric table.

---

## 5. Operational Checklist for Future Changes

Before shipping any new phase:

1. **Backend tests:**
   - `py -m pytest tests/ -k "unified" -v`
2. **Spot-check API:**
   - `curl -X POST "http://localhost:8000/api/unified/forecast/all?symbol=SPY&time_range=1D"`
   - Verify `predictions` has multiple future points per upgraded model.
3. **Frontend sanity:**
   - Navigate to Unified Analysis and confirm non-flat, forward-extending forecast lines.

If any model still only returns a scalar `prediction` without `horizons`/`trajectory`, `_extract_prediction_series` will keep it functional by repeating the scalar over `[1, 5, 10, 30]`, but you should treat that as **technical debt** to be eliminated in subsequent phases.

