#!/usr/bin/env bash
# Download the raw archives for every dataset preset (laya_impute/datasets.py) into the data root.
# Sources: UCI Machine Learning Repository static zips, CC BY 4.0. Run from anywhere.
#
#   scripts/fetch_datasets.sh [data_root]      # default data_root: data (relative to the repo root)
#
# Paths follow the presets: adult lives in <root>/adult/, the other five in <root>/datasets/<name>/.
# Idempotent: a dataset whose files are all present is skipped, and unzip never overwrites a file.
# Exits non-zero if any expected file is missing after the run.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
root="${1:-data}"

uci() { echo "https://archive.ics.uci.edu/static/public/$1/$2.zip"; }

# have <file>...: true when every file exists and is non-empty.
have() {
  local f
  for f in "$@"; do [ -s "$f" ] || return 1; done
}

# get <dir> <url> <file>...: unless the files exist, download the zip into <dir>, extract without
# overwriting, and delete the zip.
get() {
  local dir="$1" url="$2"
  shift 2
  if have "$@"; then
    echo "skip: ${*} already present"
    return 0
  fi
  local zip="$dir/$(basename "$url")"
  mkdir -p "$dir"
  curl -fsSL -o "$zip" "$url"
  unzip -n -q "$zip" -d "$dir"
  rm -f "$zip"
}

get "$root/adult" "$(uci 2 adult)" \
  "$root/adult/adult.data" "$root/adult/adult.test"

get "$root/datasets/mushroom" "$(uci 73 mushroom)" \
  "$root/datasets/mushroom/agaricus-lepiota.data"

get "$root/datasets/votes" "$(uci 105 congressional+voting+records)" \
  "$root/datasets/votes/house-votes-84.data"

get "$root/datasets/breast-cancer" "$(uci 14 breast+cancer)" \
  "$root/datasets/breast-cancer/breast-cancer.data"

# The bank archive holds bank.zip, which holds bank-full.csv: unzip twice.
bank="$root/datasets/bank"
if ! have "$bank/bank-full.csv"; then
  mkdir -p "$bank"
  bank_url="$(uci 222 bank+marketing)"
  outer="$bank/$(basename "$bank_url")"
  curl -fsSL -o "$outer" "$bank_url"
  unzip -n -q "$outer" -d "$bank"
  unzip -n -q "$bank/bank.zip" -d "$bank"
  rm -f "$outer"
fi

get "$root/datasets/car" "$(uci 19 car+evaluation)" \
  "$root/datasets/car/car.data"

expected=(
  "$root/adult/adult.data"
  "$root/adult/adult.test"
  "$root/datasets/mushroom/agaricus-lepiota.data"
  "$root/datasets/votes/house-votes-84.data"
  "$root/datasets/breast-cancer/breast-cancer.data"
  "$root/datasets/bank/bank-full.csv"
  "$root/datasets/car/car.data"
)
status=0
for f in "${expected[@]}"; do
  if [ ! -s "$f" ]; then
    echo "missing: $f" >&2
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  exit 1
fi
echo "ok: ${#expected[@]} dataset files present under $root"
