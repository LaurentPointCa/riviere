#!/bin/bash
set -e
cd ~/riviere

echo "=== $(date) ==="

# Refresh CEHQ historical files (043301 flow/level, 043108 upstream level)
echo "Refreshing CEHQ historical cache..."
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from load_data import load_flow, load_level, load_upstream_level
load_flow(cache=False)
load_level(cache=False)
load_upstream_level(cache=False)
EOF

# Refresh climate cache; flow/level extend automatically from live feed
echo "Refreshing climate cache..."
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from load_climate import load_climate
load_climate(cache=False)
EOF

# Refresh CGM upstream cache (39_RDP09, 01_RDP11, 11_LDM01) — accumulates daily
echo "Refreshing CGM upstream station cache..."
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from load_cgm import load_cgm_history
load_cgm_history(cache=False)
EOF

# Run forecast — also updates docs/forecast_sample.png
.venv/bin/python src/predict.py

# Generate validation plots (observed vs forecast horizons) — 5-day + 10-day
.venv/bin/python scripts/plot_validation.py
.venv/bin/python scripts/plot_validation_10d.py

# Commit and push only from the VM — on Mac, forecasts are for local validation only.
if [[ "$(hostname -s)" == "riviere" ]]; then
    # Sync with remote first so commits made elsewhere (e.g. Mac) don't block the push.
    # Stash any auto-refreshed cache files so rebase is clean, then restore them.
    STASHED=0
    if ! git diff --quiet; then
        git stash push -u -m "daily-forecast-autostash" >/dev/null
        STASHED=1
    fi
    git fetch origin master
    git rebase origin/master
    if [[ $STASHED -eq 1 ]]; then
        git stash pop >/dev/null
    fi

    git add docs/forecast.png docs/forecast_30d.png docs/forecast.json docs/forecast_history.json \
            docs/forecast_mse.png docs/forecast_mse_30d.png docs/forecast_mse.json docs/forecast_mse_history.json \
            docs/forecast_10d_30d.png docs/forecast_10d.json docs/forecast_10d_history.json \
            docs/forecast_validation.png docs/forecast_10d_validation.png
    if ! git diff --cached --quiet; then
        git commit -m "chore: daily forecast $(date +%Y-%m-%d)"
        git push
        echo "Chart and forecast JSON committed and pushed."
    else
        echo "Forecast unchanged, nothing to push."
    fi
else
    echo "Mac: forecast complete (local only, not committed)."
fi
