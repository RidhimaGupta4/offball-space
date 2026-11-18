#!/usr/bin/env bash
# Full pipeline, end to end (about 90 min on 2 CPU cores: run features are rebuilt once per held-out
# match, twice - primary and the alpha = 0 sensitivity - and penalties are tuned by nested CV).
# Run from the repository root after ./fetch_data.sh.
set -euo pipefail
PY="${PYTHON:-python3}"   # override with PYTHON=/path/to/python
export SKILLCORNER_DATA="${SKILLCORNER_DATA:-$(pwd)/data_raw/opendata/data}"
mkdir -p data results figures explorer docs
"$PY" src/ingest.py "$SKILLCORNER_DATA" data/tracking
cd src
"$PY" build_passes.py
"$PY" fit_pitch_control.py
"$PY" build_runs.py
ALPHA0=1 "$PY" build_runs.py
"$PY" validate_data_assumptions.py
"$PY" validate_runs.py
"$PY" positive_control.py
"$PY" descriptive.py
"$PY" figures.py
"$PY" export_explorer.py
cd ..
"$PY" explorer/build.py
"$PY" -m pytest -q tests
if command -v node >/dev/null 2>&1; then node tests/test_explorer_parity.mjs; else echo "node not found: skipped explorer parity test"; fi
