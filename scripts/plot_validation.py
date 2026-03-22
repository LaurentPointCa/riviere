"""
Validation plot: observed flow/level vs all forecast horizons (t+1..t+5).
Reads from docs/forecast_history.json. Saves to docs/forecast_validation.png.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from load_data import load_flow, load_level

MODEL_START    = "2026-03-14"   # first day of current model
MODEL_SWITCH   = "2026-03-22"   # date quantile production model replaced MSE
HISTORY_PATH   = Path(__file__).parent.parent / "docs" / "forecast_history.json"
OUT_PATH       = Path(__file__).parent.parent / "docs" / "forecast_validation.png"

# ── Load forecast history ─────────────────────────────────────────────────────
with open(HISTORY_PATH) as f:
    history = json.load(f)

# Build per-horizon DataFrames: index = predicted date
pred = {h: [] for h in range(1, 6)}
for entry in history:
    anchor = entry["anchor_date"]
    if anchor < MODEL_START:
        continue
    for fc in entry["forecast"]:
        h = fc["day"]
        if h in pred:
            pred[h].append({
                "date":  pd.Timestamp(fc["date"]),
                "flow":  fc["flow_m3s"],
                "level": fc["level_m"],
            })

pred_df = {}
for h in range(1, 6):
    if pred[h]:
        df = pd.DataFrame(pred[h]).drop_duplicates("date").set_index("date").sort_index()
        pred_df[h] = df

# ── Load observations ─────────────────────────────────────────────────────────
flow_obs  = load_flow().loc[MODEL_START:,  "flow_m3s"]
level_obs = load_level().loc[MODEL_START:, "level_m"]

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS  = {1: "#E91E63", 2: "#FF6D00", 3: "#FFC107", 4: "#7C4DFF", 5: "#00BCD4"}
MARKERS = {1: "o",       2: "s",       3: "^",       4: "D",       5: "P"}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    f"Station 043301 — Observed vs Forecast by Horizon (from {MODEL_START})",
    fontsize=13, fontweight="bold",
)

for ax, obs, col, ylabel in [
    (ax1, flow_obs,  "flow",  "Flow (m³/s)"),
    (ax2, level_obs, "level", "Level (m)"),
]:
    ax.plot(obs.index, obs.values, "-", color="#1565C0", lw=2.5, zorder=10, label="Observed")
    ax.scatter(obs.index, obs.values, color="#1565C0", s=35, zorder=11)

    for h in range(1, 6):
        df = pred_df.get(h)
        if df is None or df.empty:
            continue
        ax.plot(df.index, df[col], "--", color=COLORS[h], lw=1.3, alpha=0.75, zorder=5)
        ax.scatter(df.index, df[col],
                   marker=MARKERS[h], color=COLORS[h], s=55, zorder=6, label=f"t+{h}")

    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, ncol=3, framealpha=0.9)

    # Model switch annotation
    switch_ts = pd.Timestamp(MODEL_SWITCH)
    ax.axvline(switch_ts, color="#555", lw=1.2, ls="--", zorder=4, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ymid = ymin + (ymax - ymin) * 0.97
    ax.text(switch_ts - pd.Timedelta(hours=12), ymid, "MSE",
            ha="right", va="top", fontsize=8, color="#555", style="italic")
    ax.text(switch_ts + pd.Timedelta(hours=12), ymid, "Quantile (prod)",
            ha="left",  va="top", fontsize=8, color="#555", style="italic")

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
