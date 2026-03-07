#!/bin/bash
set -e
cd /Users/laurentchouinard/claude/riviere

echo "=== $(date) ==="

# Refresh climate cache; flow/level extend automatically from live feed
echo "Refreshing climate cache..."
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from load_climate import load_climate
load_climate(cache=False)
EOF

# Run forecast — also updates docs/forecast_sample.png
.venv/bin/python src/predict.py

# Commit and push the updated chart and forecast JSON if changed
git add docs/forecast.png docs/forecast.json
if ! git diff --cached --quiet; then
    git commit -m "chore: daily forecast $(date +%Y-%m-%d)"
    git push
    echo "Chart and forecast JSON committed and pushed."
else
    echo "Forecast unchanged, nothing to push."
fi
