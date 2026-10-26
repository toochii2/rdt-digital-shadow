#!/usr/bin/env bash
# Reproduces every result and figure in the paper from scratch.
# Runtime is roughly 4 to 6 hours on a 2-core machine; the modes are independent and can run in parallel.
set -euo pipefail
cd "$(dirname "$0")/code"
for mode in main nonlinear shared lowprev lofo stability robust containment; do
  echo "== $mode"; python3 experiments.py "$mode"
done
python3 analysis2.py > ../results/v2_analysis.txt
for mode in main lofo stress_mild stress_shared stress_nonlinear2; do
  echo "== r2 $mode"; python3 experiments_r2.py "$mode"
done
python3 analysis_r2.py > ../results/r2_analysis.txt
python3 figures_r2.py
echo "Done. Tables and statistics: results/; figures: figures/"
