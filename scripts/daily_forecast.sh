#!/bin/bash
set -e
cd ~/riviere

echo "=== $(date) ==="

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

# Generate validation plot (observed vs all forecast horizons)
.venv/bin/python scripts/plot_validation.py

# Commit and push the updated chart and forecast JSON if changed
git add docs/forecast.png docs/forecast_30d.png docs/forecast.json docs/forecast_history.json \
        docs/forecast_mse.png docs/forecast_mse_30d.png docs/forecast_mse.json docs/forecast_mse_history.json \
        docs/forecast_validation.png
if ! git diff --cached --quiet; then
    git commit -m "chore: daily forecast $(date +%Y-%m-%d)"
    git pull --rebase origin master
    git push
    echo "Chart and forecast JSON committed and pushed."
else
    echo "Forecast unchanged, nothing to push."
fi
