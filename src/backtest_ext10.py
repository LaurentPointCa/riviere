"""
Backtest comparison: baseline model (t+1..t+5 forecast features)
vs extended model (t+1..t+10 forecast features).

Both models are evaluated on the same 2-year chronological test set
using ERA5 perfect-forecast proxy for weather features (same as training).

Usage:
    python src/backtest_ext10.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from features import build_dataset
from model import load_model, time_split, evaluate, season_for, COLD_MONTHS, WARM_MONTHS, TARGET_COLS

BASELINE_PATH = Path("models/lgbm_forecast.pkl")
EXT10_PATH    = Path("models/lgbm_forecast_ext10.pkl")


def _season_mask(index: pd.DatetimeIndex, months: set) -> np.ndarray:
    return np.array([d.month in months for d in index])


def run_backtest():
    print("Loading dataset (extended features)...")
    X, y = build_dataset()
    print(f"  X: {X.shape}  |  date range: {X.index[0].date()} → {X.index[-1].date()}")

    _, X_test, _, y_test = time_split(X, y, test_years=2)
    print(f"  Test set: {X_test.index[0].date()} → {X_test.index[-1].date()} ({len(X_test):,} rows)")

    # Load models
    baseline_seasonal = load_model(BASELINE_PATH)
    ext10_seasonal    = load_model(EXT10_PATH)

    results = {}

    for season, months in [("cold", COLD_MONTHS), ("warm", WARM_MONTHS)]:
        label = "Nov–May" if season == "cold" else "Jun–Oct"
        mask  = _season_mask(X_test.index, months)
        X_s   = X_test[mask]
        y_s   = y_test[mask]

        if len(X_s) == 0:
            continue

        baseline_models = baseline_seasonal[season]
        ext10_models    = ext10_seasonal[season]

        # Baseline: restrict X to the features it was trained on
        baseline_features = baseline_models[TARGET_COLS[0]].feature_name_
        X_base = X_s[baseline_features]

        baseline_metrics = evaluate(baseline_models, X_base, y_s)
        ext10_metrics    = evaluate(ext10_models,    X_s,    y_s)

        results[season] = {
            "label":   label,
            "n":       len(X_s),
            "baseline": baseline_metrics,
            "ext10":    ext10_metrics,
        }

    # ── Print comparison table ──────────────────────────────────────────────
    print()
    for season, res in results.items():
        print(f"{'═'*72}")
        print(f"  {season.upper()} ({res['label']})  —  {res['n']} test rows")
        print(f"{'═'*72}")
        print(f"  {'Target':<12}  {'Baseline RMSE':>14}  {'Ext-10 RMSE':>12}  {'Delta':>8}  {'Δ%':>7}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*8}  {'-'*7}")
        for target in TARGET_COLS:
            b = res["baseline"][target]["rmse"]
            e = res["ext10"][target]["rmse"]
            delta = e - b
            pct   = delta / b * 100
            flag  = " ✓" if delta < 0 else ("  " if abs(delta) < 0.5 else " ✗")
            unit  = "m³/s" if target.startswith("flow") else "m   "
            print(f"  {target:<12}  {b:>12.2f}{unit[0]}  {e:>10.2f}{unit[0]}  {delta:>+8.2f}  {pct:>+6.1f}%{flag}")
        print()

    # ── MAE comparison by horizon ───────────────────────────────────────────
    print("\n── MAE by horizon (averaged across seasons) ───────────────────────────")
    print(f"  {'Horizon':<10}  {'Baseline MAE':>14}  {'Ext-10 MAE':>12}  {'Δ%':>7}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*7}")
    for h in range(1, 6):
        b_maes, e_maes = [], []
        for season, res in results.items():
            b_maes.append(res["baseline"][f"flow_t{h}"]["mae"])
            e_maes.append(res["ext10"][f"flow_t{h}"]["mae"])
        b = np.mean(b_maes); e = np.mean(e_maes)
        pct = (e - b) / b * 100
        print(f"  flow t+{h:<4}  {b:>12.1f}   {e:>10.1f}   {pct:>+6.1f}%")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Baseline vs Extended Forecast (t+1..t+10)\nTest set 2024–2026", fontsize=13)

    metrics_pairs = [
        ("cold", "flow",  "Flow RMSE — Cold season (m³/s)",  axes[0, 0]),
        ("cold", "level", "Level RMSE — Cold season (m)",    axes[0, 1]),
        ("warm", "flow",  "Flow RMSE — Warm season (m³/s)",  axes[1, 0]),
        ("warm", "level", "Level RMSE — Warm season (m)",    axes[1, 1]),
    ]

    x = np.arange(1, 6)
    width = 0.35

    for season, var, title, ax in metrics_pairs:
        if season not in results:
            ax.set_visible(False)
            continue
        res = results[season]
        b_rmse = [res["baseline"][f"{var}_t{h}"]["rmse"] for h in range(1, 6)]
        e_rmse = [res["ext10"][f"{var}_t{h}"]["rmse"]    for h in range(1, 6)]

        ax.bar(x - width/2, b_rmse, width, label="Baseline (t+1..t+5)", color="#4a9fd4", alpha=0.85)
        ax.bar(x + width/2, e_rmse, width, label="Ext-10 (t+1..t+10)",  color="#f97316", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Horizon (days)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{h}" for h in range(1, 6)])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path("docs/backtest_ext10_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved → {out}")


if __name__ == "__main__":
    run_backtest()
