"""
Horizon-specific hyperparameter tuning for the Des Prairies forecast model.

Two modes:

  --mode mse  (default)
    Objective : 5-fold walk-forward CV mean RMSE (folds 2019–2023)
    Output    : models/lgbm_forecast_tuned.pkl

  --mode quantile
    Objective : 4-fold walk-forward CV mean pinball loss on event days
                (days where actual flow > 1500 m³/s, folds 2020–2023)
                2017 and 2019 are held out — never used in CV or training
    Training  : LightGBM quantile loss at alpha=0.85
    Output    : models/lgbm_forecast_quantile_tuned.pkl

Search space per target × season:
  num_leaves        : [15, 31, 63, 127]
  min_child_samples : [20, 30, 50, 100]
  reg_lambda        : log-uniform [0.1, 10.0]
  reg_alpha         : log-uniform [0.01, 1.0]

n_estimators and learning_rate are fixed (500 / 0.05).

Usage:
    python src/tune_hyperparams.py                        # MSE mode, 30 trials
    python src/tune_hyperparams.py --mode quantile        # event-focused quantile
    python src/tune_hyperparams.py --mode quantile --trials 20
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import (
    evaluate, save_model, load_model,
    time_split, _season_mask,
    COLD_MONTHS, WARM_MONTHS, TARGET_COLS, LGBM_PARAMS,
)
from lightgbm import LGBMRegressor

TUNED_PATH              = Path("models/lgbm_forecast_tuned.pkl")
QUANTILE_TUNED_PATH     = Path("models/lgbm_forecast_quantile_tuned.pkl")
EXT10_PATH              = Path("models/lgbm_forecast_ext10.pkl")
PARAMS_PATH             = Path("models/best_params.json")
QUANTILE_PARAMS_PATH    = Path("models/best_params_quantile.json")

# MSE mode: 5 folds including flood years (objective is RMSE on all days)
CV_FOLD_YEARS      = [2019, 2020, 2021, 2022, 2023]

# Quantile mode: 4 folds, 2017 and 2019 held out for final flood evaluation
CV_FOLD_YEARS_SAFE = [2020, 2021, 2022, 2023]

QUANTILE_ALPHA   = 0.85
EVENT_THRESHOLD  = 1500.0   # m³/s — approaching flood; CV metric filtered to these days


def _cv_rmse(target: str, X: pd.DataFrame, y: pd.DataFrame,
             months: set, params: dict) -> float:
    """Mean RMSE across walk-forward CV folds for one target+season+params."""
    fold_rmses = []
    for fold_year in CV_FOLD_YEARS:
        cutoff     = pd.Timestamp(f"{fold_year - 1}-12-31")
        year_start = pd.Timestamp(f"{fold_year}-01-01")
        year_end   = pd.Timestamp(f"{fold_year}-12-31")

        tr_mask  = (X.index <= cutoff)     & _season_mask(X.index, months)
        val_mask = (X.index >= year_start) & (X.index <= year_end) & _season_mask(X.index, months)

        X_tr,  y_tr  = X[tr_mask],  y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_tr) < 30 or len(X_val) == 0:
            continue

        # n_jobs=1: caller may run many workers in parallel; avoid thread contention
        m = LGBMRegressor(**{**LGBM_PARAMS, **params, "n_jobs": 1})
        m.fit(X_tr, y_tr[target])
        preds = m.predict(X_val)
        fold_rmses.append(float(np.sqrt(np.mean((y_val[target].values - preds) ** 2))))

    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean quantile/pinball loss at level alpha."""
    err = y_true - y_pred
    return float(np.where(err >= 0, alpha * err, (alpha - 1) * err).mean())


def _cv_event_pinball(target: str, X: pd.DataFrame, y: pd.DataFrame,
                      months: set, params: dict,
                      alpha: float = QUANTILE_ALPHA,
                      event_threshold: float = EVENT_THRESHOLD) -> float:
    """
    Mean pinball loss on event days across walk-forward CV folds.

    'Event days' = rows where the actual future flow (y[target]) > event_threshold.
    Trains with quantile objective so the CV metric aligns with training loss.
    Uses CV_FOLD_YEARS_SAFE (excludes 2017/2019, kept for final flood evaluation).
    """
    fold_losses = []
    qparams = {**LGBM_PARAMS, **params,
               "objective": "quantile", "alpha": alpha, "metric": "quantile",
               "n_jobs": 1}

    for fold_year in CV_FOLD_YEARS_SAFE:
        cutoff     = pd.Timestamp(f"{fold_year - 1}-12-31")
        year_start = pd.Timestamp(f"{fold_year}-01-01")
        year_end   = pd.Timestamp(f"{fold_year}-12-31")

        tr_mask  = (X.index <= cutoff)     & _season_mask(X.index, months)
        val_mask = (X.index >= year_start) & (X.index <= year_end) & _season_mask(X.index, months)

        X_tr,  y_tr  = X[tr_mask],  y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_tr) < 30 or len(X_val) == 0:
            continue

        m = LGBMRegressor(**qparams)
        m.fit(X_tr, y_tr[target])
        preds  = m.predict(X_val)
        y_true = y_val[target].values

        event_mask = X_val["flow_m3s"].values > event_threshold
        if event_mask.sum() == 0:
            continue  # no approaching-flood days in this fold

        fold_losses.append(_pinball_loss(y_true[event_mask], preds[event_mask], alpha))

    return float(np.mean(fold_losses)) if fold_losses else float("inf")


def _tune_target_worker(args: tuple) -> tuple:
    """Top-level worker (must be module-level for ProcessPoolExecutor pickling)."""
    target, X, y, months, n_trials, alpha = args
    params   = _tune_target(target, X, y, months, n_trials, alpha)
    if alpha is None:
        cv_base = _cv_rmse(target, X, y, months, {})
        cv_best = _cv_rmse(target, X, y, months, params)
    else:
        cv_base = _cv_event_pinball(target, X, y, months, {}, alpha)
        cv_best = _cv_event_pinball(target, X, y, months, params, alpha)
    return target, params, cv_base, cv_best


def _tune_target(
    target: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    months: set,
    n_trials: int,
    alpha: float | None = None,
) -> dict:
    """Run an optuna study for one target. alpha=None → RMSE; float → event pinball."""

    def objective(trial):
        params = {
            "num_leaves":        trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 30, 50, 100]),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 1.0,  log=True),
        }
        if alpha is None:
            return _cv_rmse(target, X, y, months, params)
        return _cv_event_pinball(target, X, y, months, params, alpha)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_season(
    season: str,
    months: set,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_trials: int,
    alpha: float | None = None,
) -> dict:
    """Tune all 10 targets for one season in parallel. Returns {target: best_params}."""
    label     = "Nov–May" if season == "cold" else "Jun–Oct"
    n_workers = min(len(TARGET_COLS), os.cpu_count() or 4)
    if alpha is None:
        cv_desc = (f"{len(CV_FOLD_YEARS)}-fold CV RMSE  "
                   f"(folds {CV_FOLD_YEARS[0]}–{CV_FOLD_YEARS[-1]})")
    else:
        cv_desc = (f"{len(CV_FOLD_YEARS_SAFE)}-fold event-pinball  "
                   f"(folds {CV_FOLD_YEARS_SAFE[0]}–{CV_FOLD_YEARS_SAFE[-1]}, "
                   f"α={alpha}, threshold>{EVENT_THRESHOLD:.0f} m³/s)")
    print(f"\n── {season.upper()} ({label})  {cv_desc}  parallel={n_workers} workers")

    worker_args = [(t, X, y, months, n_trials, alpha) for t in TARGET_COLS]
    collected = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_tune_target_worker, args): args[0]
                   for args in worker_args}
        for future in as_completed(futures):
            target, params, cv_base, cv_best = future.result()
            collected[target] = (params, cv_base, cv_best)
            metric = "RMSE" if alpha is None else "pinball"
            pct    = (cv_best - cv_base) / cv_base * 100 if cv_base > 0 else 0.0
            print(f"   {target:<12} CV {metric} {cv_base:.4f} → {cv_best:.4f}  "
                  f"({pct:+.1f}%)  leaves={params['num_leaves']}  "
                  f"mcs={params['min_child_samples']}  "
                  f"λ={params['reg_lambda']:.2f}  α={params['reg_alpha']:.3f}")

    return {t: collected[t][0] for t in TARGET_COLS}


def retrain_with_best_params(
    months: set,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_params: dict,
    alpha: float | None = None,
) -> dict:
    """Retrain all 10 targets on the full training set using best params.
    alpha=None → MSE; float → quantile at that alpha level."""
    mask = _season_mask(X_train.index, months)
    X_s, y_s = X_train[mask], y_train[mask]
    extra = ({} if alpha is None
             else {"objective": "quantile", "alpha": alpha, "metric": "quantile"})
    models = {}
    for target in TARGET_COLS:
        params = {**LGBM_PARAMS, **extra, **best_params[target]}
        m = LGBMRegressor(**params)
        m.fit(X_s, y_s[target])
        models[target] = m
    return models


def compare_on_test(X_test, y_test, baseline_seasonal, tuned_seasonal) -> dict:
    """Evaluate both model sets on the test set. Returns results dict."""
    results = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        mask = _season_mask(X_test.index, months)
        X_s, y_s = X_test[mask], y_test[mask]
        if len(X_s) == 0:
            continue

        b_models = baseline_seasonal[season]
        t_models = tuned_seasonal[season]

        b_features = b_models[TARGET_COLS[0]].feature_name_
        b_metrics  = evaluate(b_models, X_s[b_features], y_s)
        t_metrics  = evaluate(t_models, X_s, y_s)

        results[season] = {
            "label":    "Nov–May" if season == "cold" else "Jun–Oct",
            "n":        len(X_s),
            "baseline": b_metrics,
            "tuned":    t_metrics,
        }
    return results


def print_comparison(results: dict) -> None:
    for season, res in results.items():
        print(f"\n{'═'*72}")
        print(f"  {season.upper()} ({res['label']})  —  {res['n']} test rows")
        print(f"{'═'*72}")
        print(f"  {'Target':<12}  {'Ext-10 RMSE':>12}  {'Tuned RMSE':>11}  {'Delta':>8}  {'Δ%':>7}")
        print(f"  {'-'*12}  {'-'*12}  {'-'*11}  {'-'*8}  {'-'*7}")
        for target in TARGET_COLS:
            b = res["baseline"][target]["rmse"]
            t = res["tuned"][target]["rmse"]
            delta = t - b
            pct   = delta / b * 100
            flag  = " ✓" if delta < 0 else ("  " if abs(pct) < 1.0 else " ✗")
            print(f"  {target:<12}  {b:>12.3f}   {t:>11.3f}  {delta:>+8.3f}  {pct:>+6.1f}%{flag}")

    print("\n── MAE by horizon (flow, averaged across seasons) ─────────────────")
    print(f"  {'Horizon':<10}  {'Ext-10 MAE':>12}  {'Tuned MAE':>11}  {'Δ%':>7}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*7}")
    for h in range(1, 6):
        b_maes, t_maes = [], []
        for res in results.values():
            b_maes.append(res["baseline"][f"flow_t{h}"]["mae"])
            t_maes.append(res["tuned"][f"flow_t{h}"]["mae"])
        b = np.mean(b_maes); t = np.mean(t_maes)
        pct = (t - b) / b * 100
        print(f"  flow t+{h:<4}  {b:>12.1f}   {t:>11.1f}   {pct:>+6.1f}%")


def plot_comparison(results: dict, best_params: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Ext-10 Baseline vs CV-Tuned Model\nTest set 2024–2026", fontsize=13)

    panels = [
        ("cold", "flow",  "Flow RMSE — Cold season (m³/s)",  axes[0, 0]),
        ("cold", "level", "Level RMSE — Cold season (m)",    axes[0, 1]),
        ("warm", "flow",  "Flow RMSE — Warm season (m³/s)",  axes[1, 0]),
        ("warm", "level", "Level RMSE — Warm season (m)",    axes[1, 1]),
    ]
    x = np.arange(1, 6)
    w = 0.35

    for season, var, title, ax in panels:
        if season not in results:
            ax.set_visible(False)
            continue
        res = results[season]
        b = [res["baseline"][f"{var}_t{h}"]["rmse"] for h in range(1, 6)]
        t = [res["tuned"][f"{var}_t{h}"]["rmse"]    for h in range(1, 6)]
        ax.bar(x - w/2, b, w, label="Ext-10 baseline", color="#4a9fd4", alpha=0.85)
        ax.bar(x + w/2, t, w, label="CV-Tuned",        color="#22c55e", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Horizon (days)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in range(1, 6)])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path("docs/backtest_tuned_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved → {out}")

    print("\n── Best params per target (cold season) ────────────────────────────")
    print(f"  {'Target':<12}  {'num_leaves':>10}  {'min_child':>10}  {'reg_λ':>8}  {'reg_α':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for target in TARGET_COLS:
        p = best_params["cold"].get(target, {})
        print(f"  {target:<12}  {p.get('num_leaves','?'):>10}  "
              f"{p.get('min_child_samples','?'):>10}  "
              f"{p.get('reg_lambda', 0):>8.2f}  "
              f"{p.get('reg_alpha', 0):>8.3f}")

    print("\n── Best params per target (warm season) ────────────────────────────")
    print(f"  {'Target':<12}  {'num_leaves':>10}  {'min_child':>10}  {'reg_λ':>8}  {'reg_α':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for target in TARGET_COLS:
        p = best_params["warm"].get(target, {})
        print(f"  {target:<12}  {p.get('num_leaves','?'):>10}  "
              f"{p.get('min_child_samples','?'):>10}  "
              f"{p.get('reg_lambda', 0):>8.2f}  "
              f"{p.get('reg_alpha', 0):>8.3f}")


def main(n_trials: int = 30, mode: str = "mse") -> None:
    alpha      = QUANTILE_ALPHA if mode == "quantile" else None
    out_path   = QUANTILE_TUNED_PATH if mode == "quantile" else TUNED_PATH
    save_path  = QUANTILE_PARAMS_PATH if mode == "quantile" else PARAMS_PATH

    print(f"Loading dataset...")
    X, y = build_dataset()
    print(f"  X: {X.shape}  |  {X.index[0].date()} → {X.index[-1].date()}")

    # Final train / test split (same as ext10 baseline)
    X_train, X_test, y_train, y_test = time_split(X, y, test_years=2)
    print(f"  Train (final): {X_train.index[0].date()} → {X_train.index[-1].date()}"
          f"  ({len(X_train):,} rows)")
    print(f"  Test:          {X_test.index[0].date()} → {X_test.index[-1].date()}"
          f"  ({len(X_test):,} rows)")

    # ── Tuning phase ──────────────────────────────────────────────────────
    mode_label = ("event-focused quantile pinball" if mode == "quantile"
                  else "MSE RMSE")
    print(f"\n{'═'*60}")
    print(f"  Tuning phase  ({n_trials} trials per target per season)")
    print(f"  Mode: {mode_label}")
    print(f"{'═'*60}")

    best_params = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        best_params[season] = tune_season(season, months, X_train, y_train,
                                          n_trials, alpha=alpha)

    # Save best params
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(best_params, indent=2))
    print(f"\nBest params saved → {save_path}")

    # ── Retrain with best params on full train set ────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Retraining on full train set with best params")
    print(f"{'═'*60}")

    tuned_seasonal = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        label = "Nov–May" if season == "cold" else "Jun–Oct"
        print(f"\n── {season.upper()} ({label})...")
        tuned_seasonal[season] = retrain_with_best_params(
            months, X_train, y_train, best_params[season], alpha=alpha
        )
        print("   done")

    save_model(tuned_seasonal, out_path)

    # ── Test-set comparison (RMSE, informational only) ────────────────────
    if EXT10_PATH.exists():
        print(f"\n{'═'*60}")
        print(f"  Test-set RMSE comparison (informational — not the flood metric)")
        print(f"{'═'*60}")
        baseline_seasonal = load_model(EXT10_PATH)
        results = compare_on_test(X_test, y_test, baseline_seasonal, tuned_seasonal)
        print_comparison(results)
        plot_comparison(results, best_params)

    print(f"\nDone. Run evaluate_flood.py to compare flood detection performance.")
    if mode == "quantile":
        print(f"  python src/evaluate_flood.py --alpha {QUANTILE_ALPHA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per target per season (default: 30)")
    parser.add_argument("--mode", choices=["mse", "quantile"], default="mse",
                        help="mse: RMSE CV (default); quantile: event-focused pinball CV")
    args = parser.parse_args()
    main(n_trials=args.trials, mode=args.mode)
