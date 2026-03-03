#!/bin/bash
set -e
cd /Users/laurentchouinard/claude/riviere

echo "=== $(date) ==="

# Refresh data cache (flow, level, climate)
echo "Refreshing data cache..."
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
from load_data import load_flow, load_level
from load_climate import load_climate
load_flow(cache=False)
load_level(cache=False)
load_climate(cache=False)
EOF

# Run forecast — also updates docs/forecast_sample.png
.venv/bin/python src/predict.py

# Commit and push the updated chart if it changed
git add docs/forecast_sample.png
if ! git diff --cached --quiet; then
    git commit -m "chore: daily forecast $(date +%Y-%m-%d)"
    git push
    echo "Chart committed and pushed."
else
    echo "Chart unchanged, nothing to push."
fi
