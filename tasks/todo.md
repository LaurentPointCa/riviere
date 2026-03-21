# Rivière des Prairies — Task Backlog

## Flood Detection Reorientation (current focus)

Context: The model's real goal is to predict upcoming flow increases concerning to residents
(2500 m³/s = concern threshold, 3000 m³/s = near flood). RMSE optimization on a test set
with no flood events (2024–2026 max: 2269 m³/s) does not measure what matters. Full
analysis in memory/project_flood_detection_reorientation.md.

### Step 1 — Build flood detection evaluation framework
- [ ] Write `src/evaluate_flood.py`:
  - Load flow history back to 1978
  - For each horizon t+1..t+5, compute precision/recall on flow > 2500 m³/s threshold
  - Also report lead time: how many days before threshold crossing did we first predict it?
  - Run against the current production model on 2017, 2019, 2023 (the flood years)
  - This becomes the baseline to beat before any model gets promoted

### Step 2 — Quantile regression model (alpha=0.85)
- [ ] In `src/model.py`, add a `train_quantile(X, y, alpha=0.85)` variant:
  - Uses `objective='quantile'`, `alpha=alpha` in LGBMRegressor
  - Everything else identical (same seasonal split, same features)
- [ ] Train and save as `models/lgbm_forecast_quantile.pkl`
- [ ] Evaluate with the flood detection framework from Step 1
- [ ] Compare: does quantile model catch more threshold crossings with acceptable precision?

### Step 3 — Event-focused hyperparameter tuning
- [ ] Modify `src/tune_hyperparams.py` to support event-focused CV:
  - CV metric: pinball loss on days where flow > 1500 m³/s (approaching threshold)
  - Use quantile pinball loss to align with quantile training objective
  - CV folds: exclude 2017 and 2019 (held out for evaluation), use 2020–2023
- [ ] Run tuning on the quantile model
- [ ] Evaluate with flood detection framework

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
