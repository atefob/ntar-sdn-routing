#!/usr/bin/env bash
# Reproduce every table and figure of the NTAR paper.
set -euo pipefail
cd "$(dirname "$0")"
python ntar_sim.py --out results --check
python experiments2.py --out results
python make_tables.py
python make_figures.py
echo "Done: results/ and figures/ are up to date."
