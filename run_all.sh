#!/usr/bin/env bash
# Reproduce every table and figure of the NTAR paper.
set -euo pipefail
cd "$(dirname "$0")"
python ntar_sim.py --out results --check      # Tables 1-4
python experiments2.py --out results          # Tables 5-6, Fig. 4
python experiments3.py --out results          # Table 9
python table7_decoupled.py                    # Table 7 (independent link-cost draws)
python energy_matched_baseline.py             # Table 10 (held-out energy-matched baseline)
python ntar_state_tie_audit.py                # tie audit of the representative run (Section 4.1)
# Table 8 measures wall-clock time, so its values depend on the machine. The published
# timings are in results/table8_timing.json; to re-measure them, uncomment:
# python table8_timing.py
python make_tables.py                         # results/tables.tex (Tables 1-10)
python make_figures.py                        # figures/
echo "Done: results/ and figures/ are up to date."
