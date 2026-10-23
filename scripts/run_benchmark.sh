#!/usr/bin/env bash
# Train and evaluate one Laya checkpoint per dataset preset, sequentially, on CPU.
#
#   scripts/run_benchmark.sh [data_root] [runs_dir] [dataset ...]
#
# Budgets are per dataset: small tables train on all their rows, large ones on a sample.
# Each run writes runs/<dataset>-v1/ (checkpoint, schema.json, train_run.json) and
# docs/results/<dataset>-v1.json (held-out evaluation against the baselines).
set -euo pipefail
root="${1:-data}"
runs="${2:-data/runs}"
mkdir -p "$runs"
if [ "$#" -gt 2 ]; then shift 2; datasets=("$@"); else datasets=(mushroom votes breast-cancer bank car); fi
py="${PYTHON:-python}"

budget() {  # dataset -> "max_train_rows views_per_row"
  case "$1" in
    mushroom)      echo "1200 1" ;;
    votes)         echo "348 2" ;;
    breast-cancer) echo "229 3" ;;
    bank)          echo "1200 1" ;;
    car)           echo "1000 2" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

for ds in "${datasets[@]}"; do
  read -r rows views <<<"$(budget "$ds")"
  out="$runs/$ds-v1"
  echo "== $ds: train (rows<=$rows, views=$views)"
  "$py" -m laya_impute.train --dataset "$ds" --root "$root" --out "$out" \
      --max-train-rows "$rows" --views-per-row "$views" --epochs 3 \
      --micro-batch 8 --grad-accum 4 --seed 0 --device cpu > "$out.log" 2>&1
  echo "== $ds: evaluate"
  "$py" -m laya_impute.evaluate --checkpoint "$out" \
      --train-csv "$out/prepared/train.csv" --test-csv "$out/prepared/test.csv" \
      --n-rows 500 --max-train-rows "$rows" --out "docs/results/$ds-v1.json" > "$out.eval.log" 2>&1
  echo "== $ds: done"
done
