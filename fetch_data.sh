#!/usr/bin/env bash
# Download SkillCorner open data (20 A-League 2024/25 matches) into data_raw/opendata.
# Tracking files are Git-LFS objects: they are fetched from GitHub's media endpoint and verified against the
# SHA-256 recorded in the repository. Works on Linux and macOS. Run from the repository root.
set -euo pipefail
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
mkdir -p data_raw && cd data_raw
[ -d opendata ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/SkillCorner/opendata.git
cd opendata
n=0
for d in data/matches/*/; do
  id=$(basename "$d"); f="${d}${id}_tracking_extrapolated.jsonl"
  expected=$(git show HEAD:"$f" | awk '/^oid/{split($2,a,":");print a[2]}')
  [ -n "$expected" ] || { echo "no LFS pointer for $f"; exit 1; }
  if [ ! -f "$f" ] || [ "$(sha256 "$f")" != "$expected" ]; then
    curl -fsSL --retry 5 --retry-delay 3 -o "$f" "https://media.githubusercontent.com/media/SkillCorner/opendata/master/$f"
    [ "$(sha256 "$f")" = "$expected" ] || { echo "checksum mismatch: $f"; exit 1; }
  fi
  n=$((n+1)); echo "verified $id"
done
echo "all $n tracking files verified"
