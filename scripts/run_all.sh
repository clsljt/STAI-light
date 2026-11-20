#!/bin/bash
# All datasets, three seeds, one configuration; then the mean over seeds.
# Usage: bash scripts/run_all.sh [gpu]
GPU=${1:-0}
for data in MSL PSM SMAP SMD SWAT GECCO Genesis Creditcard; do
  for seed in 2026 2027 2028; do
    python -u run.py --data "$data" --seed "$seed" --gpu "$GPU" --config ./configs/stai_lite.json
  done
done
python scripts/summarize.py --save_path ./results
