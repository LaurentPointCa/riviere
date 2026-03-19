"""
Experiment: do 39_RDP09 and 01_RDP11 improve the model?

Builds two datasets from the *unmodified* pipeline:
  - baseline  : standard build_dataset()
  - augmented : baseline + rdp09_level_m + rdp11_level_m with lags & rolling stats

Trains seasonal models on both and prints side-by-side RMSE / skill.
Nothing in src/ is modified.

Usage:
    .venv/bin/python scripts/experiment_rdp_stations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_dataset
from load_data import load_rdp09_level, load_rdp11_level
from model import (
    COLD_MONTHS,
    WARM_MONTHS,
    TARGET_COLS,
    LGBM_PARAMS,
    train,
    evaluate,
    time_split,
    feature_importances,
    _season_mask,
)
from lightgbm import LGBMRegressor

# ── Lag / rolling config (mirrors features.py) ───────────────────────────────

LAG_DAYS = [1, 2, 3, 4, 5, 7, 14, 30]


def _add_rdp_features(X: pd.DataFrame, rdp09: pd.DataFrame, rdp11: pd.DataFrame) -> pd.DataFrame:
    """
    Join rdp09_level_m and rdp11_level_m onto X, then add lags and rolling stats.

    Data only goes back to Sep/Oct 2020; rows before that are NaN
    (LightGBM handles missing values natively via split direction).
    """
    extra = X.join(rdp09, how="left").join(rdp11, how="left")

    new_cols = {}
    for var in ("rdp09_level_m", "rdp11_level_m"):
        col = extra[var]

        # Lags
        for n in LAG_DAYS:
            new_cols[f"{var}_lag{n}"] = col.shift(n)

        # Rolling means
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d"), (30, "30d")]:
            new_cols[f"{var}_roll_mean_{suffix}"] = col.rolling(window, min_periods=1).mean()

        # Rolling max
        for window, suffix in [(3, "3d"), (7, "7d"), (14, "14d")]:
            new_cols[f"{var}_roll_max_{suffix}"] = col.rolling(window, min_periods=1).max()

    return pd.concat([extra, pd.DataFrame(new_cols, index=extra.index)], axis=1)


# ── Metrics display ───────────────────────────────────────────────────────────

def _print_comparison(base_metrics: dict, aug_metrics: dict,
                      y_test: pd.DataFrame, X_test: pd.DataFrame,
                      season_label: str) -> None:
    flow_today  = X_test["flow_m3s"].values
    level_today = X_test["level_m"].values

    print(f"\n{'═'*72}")
    print(f"  {season_label}")
    print(f"{'═'*72}")
    print(f"{'target':<12}  {'base RMSE':>10}  {'aug RMSE':>10}  {'Δ RMSE':>8}  "
          f"{'base skill':>10}  {'aug skill':>10}  {'Δ skill':>8}")
    print(f"{'─'*72}")

    for target in TARGET_COLS:
        naive_vals = flow_today if target.startswith("flow") else level_today
        naive_rmse = float(np.sqrt(np.mean((y_test[target].values - naive_vals) ** 2)))

        b = base_metrics[target]
        a = aug_metrics[target]

        b_skill = 1.0 - b["rmse"] / naive_rmse if naive_rmse > 0 else float("nan")
        a_skill = 1.0 - a["rmse"] / naive_rmse if naive_rmse > 0 else float("nan")
        delta_rmse  = a["rmse"]  - b["rmse"]
        delta_skill = a_skill - b_skill

        # Positive skill delta = improvement; negative RMSE delta = improvement
        rmse_marker  = " ✓" if delta_rmse  < -0.5 else (" ✗" if delta_rmse  > 0.5 else "  ")
        skill_marker = " ✓" if delta_skill > 0.002 else (" ✗" if delta_skill < -0.002 else "  ")

        is_flow = target.startswith("flow")

        if is_flow:
            line = (f"{target:<12}  "
                    f"{b['rmse']:>10.1f}  {a['rmse']:>10.1f}  "
                    f"{delta_rmse:>+8.1f}{rmse_marker}  "
                    f"{b_skill:>10.3f}  {a_skill:>10.3f}  "
                    f"{delta_skill:>+8.3f}{skill_marker}")
        else:
            line = (f"{target:<12}  "
                    f"{b['rmse']:>10.4f}  {a['rmse']:>10.4f}  "
                    f"{delta_rmse:>+8.4f}{rmse_marker}  "
                    f"{b_skill:>10.3f}  {a_skill:>10.3f}  "
                    f"{delta_skill:>+8.3f}{skill_marker}")

        print(line)


def _print_new_feature_importances(aug_models: dict, feature_names: list[str]) -> None:
    """Show gain-based importance only for the new RDP features."""
    rdp_features = [f for f in feature_names
                    if f.startswith("rdp09_") or f.startswith("rdp11_")]
    if not rdp_features:
        return

    records = {}
    for target, model in aug_models.items():
        imp = model.booster_.feature_importance(importance_type="gain").astype(float)
        imp_dict = dict(zip(feature_names, imp))
        records[target] = {f: imp_dict[f] for f in rdp_features}

    df = pd.DataFrame(records, index=rdp_features)
    df["mean_gain"] = df[list(aug_models.keys())].mean(axis=1)
    df = df.sort_values("mean_gain", ascending=False)

    # Express as % of total gain so it's comparable across runs
    total_gain = sum(
        model.booster_.feature_importance(importance_type="gain").sum()
        for model in aug_models.values()
    ) / len(aug_models)
    df["gain_pct"] = (df["mean_gain"] / total_gain * 100).round(2)

    display = df[["mean_gain", "gain_pct"]].head(20)
    print("\n── New RDP feature importances (gain) ────────────────────────────")
    print(display.round(1).to_string())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("  Experiment: 39_RDP09 + 01_RDP11 feature contribution")
    print("=" * 72)

    # 1. Build baseline dataset (production pipeline, untouched)
    print("\n[1/4] Building baseline dataset...")
    X_base, y = build_dataset()
    print(f"      X_base: {X_base.shape}  |  y: {y.shape}")
    print(f"      Date range: {X_base.index[0].date()} → {X_base.index[-1].date()}")

    # 2. Load new station data and augment X
    print("\n[2/4] Loading new station data...")
    rdp09 = load_rdp09_level()   # rdp09_level_m, daily
    rdp11 = load_rdp11_level()   # rdp11_level_m, daily
    print(f"      39_RDP09: {len(rdp09):,} days  "
          f"({rdp09.index.min().date()} → {rdp09.index.max().date()})")
    print(f"      01_RDP11: {len(rdp11):,} days  "
          f"({rdp11.index.min().date()} → {rdp11.index.max().date()})")

    X_aug = _add_rdp_features(X_base, rdp09, rdp11)
    new_feature_count = X_aug.shape[1] - X_base.shape[1]
    print(f"      X_aug: {X_aug.shape}  (+{new_feature_count} new features)")

    # How many training rows actually have the new data?
    has_rdp = X_aug["rdp09_level_m"].notna()
    print(f"      Rows with RDP data: {has_rdp.sum():,} / {len(X_aug):,} "
          f"({has_rdp.mean()*100:.1f}%)")

    # 3. Chronological train / test split (same cutoff for both)
    print("\n[3/4] Splitting train / test (last 2 years = test)...")
    X_tr_b, X_te_b, y_train, y_test = time_split(X_base, y, test_years=2)
    X_tr_a, X_te_a, _,      _       = time_split(X_aug,  y, test_years=2)
    print(f"      Train: {len(X_tr_b):,} rows  "
          f"({X_tr_b.index[0].date()} → {X_tr_b.index[-1].date()})")
    print(f"      Test:  {len(X_te_b):,} rows  "
          f"({X_te_b.index[0].date()} → {X_te_b.index[-1].date()})")

    # 4. Train + evaluate for each season
    print("\n[4/4] Training & evaluating seasonal models...")
    for season, months, label in [
        ("cold", COLD_MONTHS, "COLD season — Nov–May"),
        ("warm", WARM_MONTHS, "WARM season — Jun–Oct"),
    ]:
        tr_mask_b = _season_mask(X_tr_b.index, months)
        te_mask_b = _season_mask(X_te_b.index, months)
        tr_mask_a = _season_mask(X_tr_a.index, months)
        te_mask_a = _season_mask(X_te_a.index, months)

        Xtrb, ytrb = X_tr_b[tr_mask_b], y_train[tr_mask_b]
        Xteb, yteb = X_te_b[te_mask_b], y_test[te_mask_b]
        Xtra, ytra = X_tr_a[tr_mask_a], y_train[tr_mask_a]
        Xtea, ytea = X_te_a[te_mask_a], y_test[te_mask_a]

        print(f"\n  [{season.upper()}]  train: {tr_mask_b.sum():,} rows  |  "
              f"test: {te_mask_b.sum():,} rows")

        print(f"    baseline  — training...", end=" ", flush=True)
        base_models = train(Xtrb, ytrb)
        base_metrics = evaluate(base_models, Xteb, yteb)
        print("evaluating... done")

        print(f"    augmented — training...", end=" ", flush=True)
        aug_models = train(Xtra, ytra)
        aug_metrics = evaluate(aug_models, Xtea, ytea)
        print("evaluating... done")

        _print_comparison(base_metrics, aug_metrics, yteb, Xteb, label)
        _print_new_feature_importances(aug_models, list(X_aug.columns))

    print(f"\n{'═'*72}")
    print("  Done. No production files were modified.")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
