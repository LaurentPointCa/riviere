"""
LightGBM forecast model for station 043301 — Des Prairies.

Strategy: one LGBMRegressor per target horizon (10 models total),
stored in a dict keyed by target name.

Public API:
    train(X, y)            -> dict[str, LGBMRegressor]
    evaluate(models, X, y) -> dict[str, dict[str, float]]
    save_model(models, path)
    load_model(path)       -> dict[str, LGBMRegressor]
    feature_importances(models, feature_names, top_n) -> pd.DataFrame
"""

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
    """Serialize the models dict to disk with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved → {path}")


def load_model(path: str | Path) -> dict:
    """Load a models dict previously saved with save_model."""
    with open(Path(path), "rb") as f:
        return pickle.load(f)


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


# ── Display helpers ──────────────────────────────────────────────────────────

def _print_metrics(metrics: dict, y_test: pd.DataFrame, X_test: pd.DataFrame) -> None:
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
            "skill":      round(1.0 - m["rmse"] / naive_rmse, 3),
        })

    df = pd.DataFrame(rows).set_index("target")

    print("\n── Flow targets (m³/s) ──────────────────────────────────────────")
    flow_rows = [t for t in df.index if t.startswith("flow")]
    print(df.loc[flow_rows].round({"RMSE": 1, "MAE": 1, "naive_RMSE": 1, "skill": 3}).to_string())

    print("\n── Level targets (m) ────────────────────────────────────────────")
    level_rows = [t for t in df.index if t.startswith("level")]
    print(df.loc[level_rows].round({"RMSE": 4, "MAE": 4, "naive_RMSE": 4, "skill": 3}).to_string())


def _print_importances(models: dict, feature_names: list[str], top_n: int = 15) -> None:
    imp_df = feature_importances(models, feature_names, top_n=top_n)
    display_cols = [c for c in ["mean_gain", "flow_t1", "level_t1"] if c in imp_df.columns]
    print(f"\n── Top {top_n} features by average gain ─────────────────────────")
    print(imp_df[display_cols].round(1).to_string())


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Build dataset
    print("Loading dataset...")
    X, y = build_dataset()
    print(f"  X: {X.shape}  |  y: {y.shape}")
    print(f"  Date range: {X.index[0].date()} → {X.index[-1].date()}")

    # 2. Chronological train / test split
    X_train, X_test, y_train, y_test = time_split(X, y, test_years=2)
    print(f"\nTrain: {len(X_train):,} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"Test:  {len(X_test):,} rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    # 3. Train
    print("\nTraining 10 LGBMRegressor models...")
    models = train(X_train, y_train)

    # 4. Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate(models, X_test, y_test)
    _print_metrics(metrics, y_test, X_test)

    # 5. Save
    save_model(models, "models/lgbm_forecast.pkl")

    # 6. Feature importances
    _print_importances(models, list(X.columns), top_n=15)
