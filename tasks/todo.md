# Rivière des Prairies — Task Backlog

## Flood Detection Reorientation (current focus)

Context: The model's real goal is to predict upcoming flow increases concerning to residents
(2500 m³/s = concern threshold, 3000 m³/s = near flood). RMSE optimization on a test set
with no flood events (2024–2026 max: 2269 m³/s) does not measure what matters. Full
analysis in memory/project_flood_detection_reorientation.md.

### Step 1 — Build flood detection evaluation framework
- [x] Write `src/evaluate_flood.py`:
  - Load flow history back to 1978
  - For each horizon t+1..t+5, compute precision/recall on flow > 2500 m³/s threshold
  - Also report lead time: how many days before threshold crossing did we first predict it?
  - Run against the current production model on 2017, 2019, 2023 (the flood years)
  - This becomes the baseline to beat before any model gets promoted
- [x] Baseline results (MSE walk-forward):
  - Precision: 0.95–1.00 across all horizons (almost no false alarms)
  - Recall: 0.935 (t+1) → 0.656 (t+5) — collapses at longer horizons
  - 2017 worst: recall=0.552 at t+3, 0.103 at t+5
  - Lead time: 2017 and 2019 onset events had ZERO advance warning (model only
    detected flood once it was already happening); 2023 had 2–5 day warnings

### Step 2 — Quantile regression model (alpha=0.85)
- [x] In `src/model.py`, add a `train_quantile(X, y, alpha=0.85)` variant
- [x] Train and save as `models/lgbm_forecast_quantile.pkl`
- [x] Evaluate with the flood detection framework from Step 1
- [x] Result: clear improvement over MSE baseline
  - Recall t+3: 0.828 → 0.925  |  t+5: 0.656 → 0.806
  - Precision stays high: 0.915 at t+3 and t+5 (was 0.951/0.953)
  - Lead time: 2017 onset 0 days → 5-day warning; 2019 onset 0 → 2-day warning
  - Verdict: quantile model advances to Step 3 (event-focused tuning)

### Step 3 — Event-focused hyperparameter tuning
- [x] Modify `src/tune_hyperparams.py` to support event-focused CV (`--mode quantile`)
  - CV metric: pinball loss on days where current flow > 1500 m³/s
  - CV folds: [2020, 2021, 2022, 2023] — 2017/2019 held out
- [x] Run 30-trial quantile tuning → `models/lgbm_forecast_quantile_tuned.pkl`
- [x] Evaluate with flood detection framework
- [x] Results vs untuned quantile:
  - Recall t+1: 0.978 → 0.989  |  t+3: 0.925 → 0.946  |  t+5: 0.806 → 0.839
  - Precision stays ≥ 0.876 at all horizons
  - Lead time: 2017 first onset regressed (5d → 1d); 2019 onset held at 2d

### Step 4 — Evaluate and promote
- [ ] Compare on flood years (2017, 2019, 2023):
  - Current production model (ext10, MSE)
  - Quantile model (untuned)
  - Quantile model (CV-tuned, event-focused)
- [ ] Define promotion criteria: recall on 2500+ threshold at t+3 must improve; precision
  must not drop below X% (TBD based on results)
- [ ] Promote winning model to production + SCP to VM

### Step 5 — Update forecast output for residents
- [ ] Add flood risk signal to `docs/forecast.json`:
  - Flag when any predicted horizon exceeds 2500 m³/s
  - Flag when any predicted horizon exceeds 3000 m³/s
- [ ] Consider adding warning band to forecast charts (forecast.png, forecast_30d.png)
  - Horizontal line at 2500 m³/s and 3000 m³/s on flow panel

---

## Completed

- [x] Weather forecast features extended to t+10 (ext10) — promoted to production
- [x] Walk-forward CV infrastructure (model.py `walk_forward_cv()`)
- [x] Hyperparameter tuning with 5-fold CV (tune_hyperparams.py) — evaluated, not promoted
      (reason: RMSE objective is wrong for flood detection goal)
- [x] Parallel tuning with ProcessPoolExecutor (8 workers)
- [x] Fix empty live feed crash (DatetimeIndex bug in _live_daily)
- [x] Establish Mac=train, VM=run workflow
