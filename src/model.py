"""
LightGBM forecast model for station 043301 — Des Prairies.

Strategy: two seasonal model sets (cold: Nov–May, warm: Jun–Oct),
each with one LGBMRegressor per target horizon (10 models × 2 seasons = 20 total).
The saved pickle has the structure:
    {"cold": {target: model, ...}, "warm": {target: model, ...}}

Public API:
    train(X, y)                        -> dict[str, LGBMRegressor]
    evaluate(models, X, y)             -> dict[str, dict[str, float]]
    save_model(models, path)
    load_model(path)                   -> dict[str, LGBMRegressor]  (single set)
    load_seasonal_models(path)         -> dict[str, dict]           (both sets)
    season_for(date)                   -> "cold" | "warm"
    feature_importances(models, ...)   -> pd.DataFrame
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset


# ── Hyperparameters ──────────────────────────────────────────────────────────

LGBM_PARAMS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 30,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}

TARGET_COLS = (
    [f"flow_t{h}"  for h in range(1, 6)] +
    [f"level_t{h}" for h in range(1, 6)]
)

# Nov 1 – May 31: snowpack accumulation + spring freshet
# Jun 1 – Oct 31: post-freshet low-flow, rain-dominated
COLD_MONTHS = {11, 12, 1, 2, 3, 4, 5}
WARM_MONTHS = {6, 7, 8, 9, 10}


# ── Season helpers ───────────────────────────────────────────────────────────

def season_for(date: pd.Timestamp) -> str:
    """Return 'cold' (Nov–May) or 'warm' (Jun–Oct) for a given date."""
    return "cold" if date.month in COLD_MONTHS else "warm"


def _season_mask(index: pd.DatetimeIndex, months: set) -> np.ndarray:
    return np.array([d.month in months for d in index])


# ── Public API ───────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    """
    Train one LGBMRegressor per target column.

    Parameters
    ----------
    X : feature matrix (NaNs handled natively by LightGBM)
    y : target matrix; must contain the 10 columns in TARGET_COLS

    Returns
    -------
    dict[str, LGBMRegressor] keyed by target column name
    """
    models = {}
    for target in TARGET_COLS:
        print(f"  Training {target}...", end=" ", flush=True)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y[target])
        models[target] = model
        print("done")
    return models


def train_quantile(X: pd.DataFrame, y: pd.DataFrame, alpha: float = 0.85) -> dict:
    """
    Train one quantile LGBMRegressor per target column.

    Uses objective='quantile' with the given alpha, biasing predictions upward
    under uncertainty. alpha=0.85 penalises under-prediction 85% of the time,
    which helps catch threshold crossings without generating excess false alarms.

    Everything else is identical to train() (same features, same n_estimators, etc.).
    """
    params = {**LGBM_PARAMS, "objective": "quantile", "alpha": alpha,
              "metric": "quantile"}
    models = {}
    for target in TARGET_COLS:
        print(f"  Training {target} (quantile α={alpha})...", end=" ", flush=True)
        model = LGBMRegressor(**params)
        model.fit(X, y[target])
        models[target] = model
        print("done")
    return models


def evaluate(models: dict, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """
    Compute RMSE and MAE per target on a held-out set.

    Returns
    -------
    dict[str, dict[str, float]]
        e.g. {"flow_t1": {"rmse": 45.2, "mae": 31.0}, ...}
    """
    metrics = {}
    for target, model in models.items():
        y_pred = model.predict(X_test)
        y_true = y_test[target].values
        residuals = y_true - y_pred
        metrics[target] = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae":  float(np.mean(np.abs(residuals))),
        }
    return metrics


def save_model(models: dict, path: str | Path) -> None:
    """Serialize a models dict (single or seasonal) to disk with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved → {path}")


def load_model(path: str | Path) -> dict:
    """Load a models dict previously saved with save_model."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def load_seasonal_models(path: str | Path) -> dict:
    """
    Load the seasonal model file and return {"cold": {...}, "warm": {...}}.
    Raises KeyError if the file does not contain seasonal models.
    """
    data = load_model(path)
    if "cold" not in data or "warm" not in data:
        raise KeyError(f"{path} does not contain seasonal models ('cold'/'warm' keys).")
    return data


def feature_importances(
    models: dict,
    feature_names: list[str],
    top_n: int = 15,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Build a DataFrame of feature importances (by gain) across all models.

    Returns
    -------
    pd.DataFrame
        Rows = features, columns = targets + "mean_gain".
        Sorted by mean_gain descending, limited to top_n rows.
    """
    records = {
        target: model.booster_.feature_importance(importance_type=importance_type).astype(float)
        for target, model in models.items()
    }
    df = pd.DataFrame(records, index=feature_names)
    df["mean_gain"] = df[list(models.keys())].mean(axis=1)
    return df.sort_values("mean_gain", ascending=False).head(top_n)


# ── Train / test split ───────────────────────────────────────────────────────

def time_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: last `test_years` calendar years become the test set.

    Returns (X_train, X_test, y_train, y_test).
    """
    cutoff = X.index[-1] - pd.DateOffset(years=test_years)
    train_mask = X.index <= cutoff
    test_mask  = X.index >  cutoff
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ── Walk-forward cross-validation ────────────────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    first_test_year: int = 2019,
    last_test_year: int | None = None,
) -> pd.DataFrame:
    """
    Walk-forward cross-validation with expanding training window.

    For each test year Y in [first_test_year, last_test_year]:
      - Train on all data up to Dec 31 of Y-1
      - Evaluate on all available data in year Y
      - Repeat for cold and warm seasons separately

    Parameters
    ----------
    first_test_year : first calendar year used as test fold (default 2019)
    last_test_year  : last calendar year used as test fold (default: last full
                      year in X)

    Returns
    -------
    pd.DataFrame
        Rows = targets, columns = (season, metric) MultiIndex.
        Summary rows show mean and std RMSE across folds.
    """
    if last_test_year is None:
        last_year = X.index[-1].year
        # Only use full years as test folds
        last_test_year = last_year if X.index[-1].month == 12 else last_year - 1

    fold_years = list(range(first_test_year, last_test_year + 1))
    print(f"\nWalk-forward CV: {len(fold_years)} folds "
          f"({first_test_year}–{last_test_year})\n")

    # Accumulate per-fold RMSE: fold_results[season][target] = [rmse, ...]
    fold_results: dict[str, dict[str, list]] = {
        "cold": {t: [] for t in TARGET_COLS},
        "warm": {t: [] for t in TARGET_COLS},
    }

    for fold_year in fold_years:
        cutoff    = pd.Timestamp(f"{fold_year - 1}-12-31")
        year_start = pd.Timestamp(f"{fold_year}-01-01")
        year_end   = pd.Timestamp(f"{fold_year}-12-31")

        train_mask = X.index <= cutoff
        test_mask  = (X.index >= year_start) & (X.index <= year_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f"  [{fold_year}] skipped (insufficient data)")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        fold_rmse = {}
        for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
            s_tr = _season_mask(X_tr.index, months)
            s_te = _season_mask(X_te.index, months)

            if s_tr.sum() < 30 or s_te.sum() == 0:
                continue

            models = train(X_tr[s_tr], y_tr[s_tr])
            metrics = evaluate(models, X_te[s_te], y_te[s_te])

            for target, m in metrics.items():
                fold_results[season][target].append(m["rmse"])

            fold_rmse[season] = {t: round(metrics[t]["rmse"], 2) for t in ["flow_t1", "level_t1"]}

        print(f"  [{fold_year}]  "
              f"cold  flow_t1={fold_rmse.get('cold', {}).get('flow_t1', '-'):>6}  "
              f"level_t1={fold_rmse.get('cold', {}).get('level_t1', '-'):>6}  |  "
              f"warm  flow_t1={fold_rmse.get('warm', {}).get('flow_t1', '-'):>6}  "
              f"level_t1={fold_rmse.get('warm', {}).get('level_t1', '-'):>6}")

    # Summarise
    print(f"\n{'═'*65}")
    print("  Walk-forward CV summary")
    print(f"{'═'*65}")
    for season in ("cold", "warm"):
        label = "Nov–May" if season == "cold" else "Jun–Oct"
        print(f"\n── {season.upper()} ({label}) ────────────────────────────────────")
        rows = []
        for target in TARGET_COLS:
            vals = fold_results[season][target]
            if vals:
                rows.append({
                    "target": target,
                    "mean_RMSE": round(np.mean(vals), 2),
                    "std_RMSE":  round(np.std(vals),  2),
                    "min_RMSE":  round(np.min(vals),  2),
                    "max_RMSE":  round(np.max(vals),  2),
                    "n_folds":   len(vals),
                })
        df = pd.DataFrame(rows).set_index("target")
        print(df.to_string())

    return df


# ── Display helpers ──────────────────────────────────────────────────────────

def _print_metrics(metrics: dict, y_test: pd.DataFrame, X_test: pd.DataFrame,
                   label: str = "") -> None:
    """Print RMSE / MAE table with persistence-model skill score."""
    flow_today  = X_test["flow_m3s"].values
    level_today = X_test["level_m"].values

    rows = []
    for target, m in metrics.items():
        naive_vals = flow_today if target.startswith("flow") else level_today
        naive_rmse = float(np.sqrt(np.mean((y_test[target].values - naive_vals) ** 2)))
        rows.append({
            "target":     target,
            "RMSE":       m["rmse"],
            "MAE":        m["mae"],
            "naive_RMSE": naive_rmse,
            "skill":      round(1.0 - m["rmse"] / naive_rmse, 3) if naive_rmse > 0 else float("nan"),
        })

    df = pd.DataFrame(rows).set_index("target")
    header = f" [{label}]" if label else ""

    print(f"\n── Flow targets (m³/s){header} ──────────────────────────────────")
    flow_rows = [t for t in df.index if t.startswith("flow")]
    print(df.loc[flow_rows].round({"RMSE": 1, "MAE": 1, "naive_RMSE": 1, "skill": 3}).to_string())

    print(f"\n── Level targets (m){header} ────────────────────────────────────")
    level_rows = [t for t in df.index if t.startswith("level")]
    print(df.loc[level_rows].round({"RMSE": 4, "MAE": 4, "naive_RMSE": 4, "skill": 3}).to_string())


def _print_importances(models: dict, feature_names: list[str], top_n: int = 15,
                       label: str = "") -> None:
    imp_df = feature_importances(models, feature_names, top_n=top_n)
    display_cols = [c for c in ["mean_gain", "flow_t1", "level_t1"] if c in imp_df.columns]
    header = f" [{label}]" if label else ""
    print(f"\n── Top {top_n} features by average gain{header} ─────────────────")
    print(imp_df[display_cols].round(1).to_string())


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the Des Prairies forecast model.")
    parser.add_argument("--cv", action="store_true",
                        help="Run walk-forward cross-validation instead of training")
    parser.add_argument("--first-test-year", type=int, default=2019,
                        help="First test year for CV (default: 2019)")
    parser.add_argument("--last-test-year",  type=int, default=None,
                        help="Last test year for CV (default: last full year in dataset)")
    parser.add_argument("--quantile", type=float, default=None, metavar="ALPHA",
                        help="Train quantile model at ALPHA (e.g. 0.85) and save as "
                             "models/lgbm_forecast_quantile.pkl")
    args = parser.parse_args()

    # 1. Build dataset
    print("Loading dataset...")
    X, y = build_dataset()
    print(f"  X: {X.shape}  |  y: {y.shape}")
    print(f"  Date range: {X.index[0].date()} → {X.index[-1].date()}")

    if args.cv:
        walk_forward_cv(X, y,
                        first_test_year=args.first_test_year,
                        last_test_year=args.last_test_year)
        sys.exit(0)

    # 2. Chronological train / test split
    X_train, X_test, y_train, y_test = time_split(X, y, test_years=2)
    print(f"\nTrain: {len(X_train):,} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Test:  {len(X_test):,} rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    train_fn   = (lambda X, y: train_quantile(X, y, alpha=args.quantile)
                  if args.quantile else train)
    out_path   = ("models/lgbm_forecast_quantile.pkl" if args.quantile
                  else "models/lgbm_forecast.pkl")
    label_sfx  = f" (quantile α={args.quantile})" if args.quantile else ""

    seasonal_models = {}
    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        label = "Nov–May" if season == "cold" else "Jun–Oct"
        train_mask = _season_mask(X_train.index, months)
        test_mask  = _season_mask(X_test.index,  months)

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_te, y_te = X_test[test_mask],   y_test[test_mask]

        print(f"\n{'═'*60}")
        print(f"  Season: {season.upper()} ({label}){label_sfx}")
        print(f"  Train rows: {len(X_tr):,}  |  Test rows: {len(X_te):,}")
        print(f"{'═'*60}")

        print(f"\nTraining 10 LGBMRegressor models ({season}){label_sfx}...")
        models = train_fn(X_tr, y_tr)

        print(f"\nEvaluating on {season} test set...")
        metrics = evaluate(models, X_te, y_te)
        _print_metrics(metrics, y_te, X_te, label=label)
        _print_importances(models, list(X.columns), top_n=10, label=label)

        seasonal_models[season] = models

    # Save both season model sets in one file
    save_model(seasonal_models, out_path)
