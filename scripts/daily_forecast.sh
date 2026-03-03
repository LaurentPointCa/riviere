#!/bin/bash
set -e
cd /Users/laurentchouinard/claude/riviere

echo "=== $(date) ==="

# Run forecast — also refreshes docs/forecast_sample.png
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
