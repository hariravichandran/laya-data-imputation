#!/usr/bin/env bash
# Download UCI Adult (CC BY 4.0) into data/adult/. Run from the repository root.
set -euo pipefail
dest="${1:-data/adult}"
mkdir -p "$dest"
url="https://archive.ics.uci.edu/static/public/2/adult.zip"
curl -fsSL -o "$dest/adult.zip" "$url"
unzip -o -q "$dest/adult.zip" -d "$dest"
rm -f "$dest/adult.zip"
for f in adult.data adult.test; do
  test -s "$dest/$f" || { echo "missing $dest/$f" >&2; exit 1; }
done
echo "ok: $(wc -l < "$dest/adult.data") training rows, $(wc -l < "$dest/adult.test") test lines in $dest"
