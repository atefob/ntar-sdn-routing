# NTAR: Normalized Threat-Adaptive Routing for SDN

Simulation code for the paper:

> A. A. Obeidat, T. A. Al-Qablan, S. Thuneibat, and M. S. Alkoffash, "NTAR: A Normalized Threat-Adaptive Routing Framework for Energy–Security Trade-offs in SDN," manuscript, 2026.

NTAR is a centralized SDN routing controller that re-weights a multi-objective path cost (energy, latency, path novelty) according to a runtime threat signal. It favors efficient paths in normal operation and diversifies paths (moving target defense) while a threat is reported. NTAR-R is a randomized variant that selects uniformly among near-optimal paths.

This repository reproduces every table and figure in the paper from a single command.

## Quick start

```bash
git clone https://github.com/atefob/ntar-sdn-routing.git
cd ntar-sdn-routing
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` takes about two minutes on a laptop and writes:

- `results/results.json`: main experiments (Tables 1–4).
- `results/results2.json`: extended experiments (Tables 5–8).
- `results/tables.tex`: all eight tables as LaTeX, generated from the JSON files.
- `figures/`: Figs. 2–4 as PDF and PNG.

## Repository layout

| File | Purpose |
|---|---|
| `ntar_sim.py` | Main simulator. Its routing engine reproduces the original baseline script step for step. It generates the representative run, the benchmark, the cost-form ablation, the threat-weight sweep and the generalization study. Use `--check` to verify the benchmark against the published Table 2. |
| `experiments2.py` | Extended experiments: NTAR-R, frequency/Markov/policy-aware adversaries, periodic MTD baselines, imperfect threat signals, decoupled energy/latency costs, computational scaling, topology-level statistics, and the Proposition 1 counterexample. |
| `make_tables.py` | Builds `results/tables.tex` from the JSON results, so no number is transcribed by hand. |
| `make_figures.py` | Builds Figs. 2–4 from the JSON results. |
| `original/code_com_baseline.py` | The authors' original baseline script, kept for reference. |

## Model summary

**Topology.** The main topology is a six-node Erdős–Rényi graph (p = 0.6, seed 42), routing from node 0 to node 5. Candidate paths have at most five nodes (four hops); there are 13 candidates. Node 3 fails at step 8 and recovers at step 12, and a threat is reported during steps 15–19.

**Cost.** NTAR scores every candidate path *p* with

C_norm(p) = w1·E(p)/E_max + w2·(L(p)/100)/L_max − w3/(rep(p)+1)

where *rep(p)* is how often *p* was used before. The weights are (0.4, 0.4, 0.2) in normal operation and (0.2, 0.2, 0.6) under threat.

**Original cost.** The original cost also added an unweighted repetition penalty 0.5·rep(p). That penalty makes path selection independent of the weights (Proposition 1 in the paper).

**NTAR-R.** NTAR-R picks uniformly among the paths whose cost is within τ = 0.1 of the minimum.

**Tie-breaking.** Candidates are evaluated in NetworkX's depth-first enumeration order, and the first minimum wins. Two exact cost ties occur in the representative run; Section IV-A of the paper describes them.

## Reproducing individual results

```bash
python ntar_sim.py --out results --check   # Tables 1-4, verifies Table 2 of the original script
python experiments2.py --out results       # Tables 5-8
python make_tables.py                      # results/tables.tex
python make_figures.py                     # figures/
```

## Limitations

These are discrete-time NetworkX simulations, not packet-level or emulated SDN experiments.

- **Threat signal:** it is an input to the controller; Table 6 studies how imperfections in it affect the results.
- **Adversaries:** they are passive next-path predictors.
- **Scaling:** Table 8 reports the cost of the candidate search, not end-to-end controller latency.

See Section V of the paper for details.

## Requirements

Python 3.10 or newer, with NetworkX, NumPy, SciPy and Matplotlib (see `requirements.txt`). The results were produced with Python 3.13, NetworkX 3.6.1, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2.

The candidate order, and therefore tie-breaking, depends on NetworkX's path enumeration. Pin the NetworkX version if you need bit-identical path sequences.

The `results/` folder holds the exact output used in the paper. Rerunning `run_all.sh` reproduces every simulated number exactly. The only values that change between runs are the wall-clock timings in Table 8, which depend on the machine and its load.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT License. See `LICENSE`.
