"""
Horizon-specific hyperparameter tuning for the Des Prairies forecast model.

Tuning objective: 5-fold walk-forward CV mean RMSE (folds 2019–2023).
  Fold Y: train on data ≤ Dec 31 of Y-1, evaluate on year Y (season-filtered)

Splits:
  CV folds   : 2019, 2020, 2021, 2022, 2023  ← optuna objective
  final-train: 1978-01-01 → 2024-03-16       ← same as ext10 baseline, best params applied
  test       : 2024-03-17 → 2026-03-16       ← untouched, final comparison only

Search space per target × season:
  num_leaves        : [15, 31, 63, 127]
  min_child_samples : [20, 30, 50, 100]
  reg_lambda        : log-uniform [0.1, 10.0]
  reg_alpha         : log-uniform [0.01, 1.0]

n_estimators and learning_rate are fixed (500 / 0.05).

Usage:
    python src/tune_hyperparams.py              # tune + retrain + compare
    python src/tune_hyperparams.py --trials 20  # fewer trials for quick test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import (
    train, evaluate, save_model, load_model,
    time_split, _season_mask,
    COLD_MONTHS, WARM_MONTHS, TARGET_COLS, LGBM_PARAMS,
)
from lightgbm import LGBMRegressor

TUNED_PATH  = Path("models/lgbm_forecast_tuned.pkl")
EXT10_PATH  = Path("models/lgbm_forecast_ext10.pkl")
PARAMS_PATH = Path("models/best_params.json")

CV_FOLD_YEARS = [2019, 2020, 2021, 2022, 2023]


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

        m = LGBMRegressor(**{**LGBM_PARAMS, **params})
        m.fit(X_tr, y_tr[target])
        preds = m.predict(X_val)
        fold_rmses.append(float(np.sqrt(np.mean((y_val[target].values - preds) ** 2))))

    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


def _tune_target(
    target: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    months: set,
    n_trials: int,
) -> dict:
    """Run an optuna study for one target using 5-fold CV. Returns best params dict."""

    def objective(trial):
        params = {
            "num_leaves":        trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 30, 50, 100]),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.01, 1.0,  log=True),
        }
        return _cv_rmse(target, X, y, months, params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_season(
    season: str,
    months: set,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_trials: int,
) -> dict:
    """Tune all 10 targets for one season. Returns {target: best_params}."""
    label = "Nov–May" if season == "cold" else "Jun–Oct"

    n_cv = sum(
        1 for fold_year in CV_FOLD_YEARS
        if (_season_mask(X.index, months) & (X.index >= pd.Timestamp(f"{fold_year}-01-01"))
            & (X.index <= pd.Timestamp(f"{fold_year}-12-31"))).sum() > 0
    )
    print(f"\n── {season.upper()} ({label})  {n_cv}-fold CV  ({len(CV_FOLD_YEARS)} folds: "
          f"{CV_FOLD_YEARS[0]}–{CV_FOLD_YEARS[-1]})")

    best = {}
    for target in TARGET_COLS:
        print(f"   {target}...", end=" ", flush=True)
        params = _tune_target(target, X, y, months, n_trials)
        best[target] = params
        cv_rmse_base = _cv_rmse(target, X, y, months, {})
        cv_rmse_best = _cv_rmse(target, X, y, months, params)
        print(f"CV RMSE {cv_rmse_base:.2f} → {cv_rmse_best:.2f}  "
              f"({(cv_rmse_best - cv_rmse_base) / cv_rmse_base * 100:+.1f}%)  "
              f"leaves={params['num_leaves']}  mcs={params['min_child_samples']}  "
              f"λ={params['reg_lambda']:.2f}  α={params['reg_alpha']:.3f}")

    return best


def retrain_with_best_params(
    season: str,
    months: set,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_params: dict,
) -> dict:
    """Retrain all 10 targets on the full training set using best params."""
    mask = _season_mask(X_train.index, months)
    X_s, y_s = X_train[mask], y_train[mask]
    models = {}
    for target in TARGET_COLS:
        params = {**LGBM_PARAMS, **best_params[target]}
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


def main(n_trials: int = 30) -> None:
    print("Loading dataset...")
    X, y = build_dataset()
    print(f"  X: {X.shape}  |  {X.index[0].date()} → {X.index[-1].date()}")

    # Final train / test split (same as ext10 baseline)
    X_train, X_test, y_train, y_test = time_split(X, y, test_years=2)
    print(f"  Train (final): {X_train.index[0].date()} → {X_train.index[-1].date()}"
          f"  ({len(X_train):,} rows)")
    print(f"  Test:          {X_test.index[0].date()} → {X_test.index[-1].date()}"
          f"  ({len(X_test):,} rows)")

    # ── Tuning phase ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Tuning phase  ({n_trials} trials per target per season)")
    print(f"  CV folds: {CV_FOLD_YEARS}")
    print(f"{'═'*60}")

    best_params = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        best_params[season] = tune_season(season, months, X_train, y_train, n_trials)

    # Save best params
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    print(f"\nBest params saved → {PARAMS_PATH}")

    # ── Retrain with best params on full train set ────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Retraining on full train set with best params")
    print(f"{'═'*60}")

    tuned_seasonal = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        label = "Nov–May" if season == "cold" else "Jun–Oct"
        print(f"\n── {season.upper()} ({label})...")
        tuned_seasonal[season] = retrain_with_best_params(
            season, months, X_train, y_train, best_params[season]
        )
        print("   done")

    save_model(tuned_seasonal, TUNED_PATH)

    # ── Test-set comparison ───────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Test-set comparison (ext-10 baseline vs CV-tuned)")
    print(f"{'═'*60}")

    baseline_seasonal = load_model(EXT10_PATH)
    results = compare_on_test(X_test, y_test, baseline_seasonal, tuned_seasonal)
    print_comparison(results)
    plot_comparison(results, best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per target per season (default: 30)")
    args = parser.parse_args()
    main(n_trials=args.trials)
