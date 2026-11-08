"""Held-out energy-matched random-routing baseline for NTAR-R.

Run from the ntar-sdn-routing project root (next to experiments2.py):
    .\\.venv\\Scripts\\python.exe energy_matched_baseline.py

What it does:
  * Uses the existing 30 topologies for CALIBRATION ONLY.
  * Generates 30 disjoint, held-out topologies using different graph seeds.
  * Fits a single Gibbs/softmax cost-bias parameter beta on calibration data
    to match the mean TOTAL-HORIZON energy of NTAR-R (tau=0.1).
  * Locks beta before assessing the held-out topologies.
  * Reports topology-level paired 95% CIs and Wilcoxon p-values.
  * Writes NEW files only. Does not modify prior results or core simulator.

Policy: probability(path p | topology) proportional to exp(-beta * E(p)/Emax).
It is a threat-INDEPENDENT, energy-aware stochastic baseline. It does not use
path history, threat weights or the NTAR novelty term.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import networkx as nx
import numpy as np

import experiments2 as E
import ntar_sim as S

SIZES = (10, 15, 20)
THREAT_START = 15
THREAT_LENGTH = 10
HORIZON = 40
METRICS = ("energy", "energy_nom", "energy_thr", "latency", "f1", "f3", "m1", "m3", "pa1", "pa3", "h_avail")
METHODS = ("Fixed-Weight", "Random", "NTAR-R", "Energy-Matched Random")


def heldout_topologies(per_size: int = 10):
    """Apply experiments2.random_topologies inclusion criteria with DISJOINT seeds."""
    out = []
    for n in SIZES:
        made = 0
        tseed = 1_000_000 + 1_000 * n
        trials = 0
        while made < per_size:
            tseed += 1
            trials += 1
            if trials > 100_000:
                raise RuntimeError(f"Could not generate held-out topologies of size {n}")
            g = nx.erdos_renyi_graph(n, 3.6 / (n - 1), seed=tseed)
            if not nx.is_connected(g):
                continue
            lengths = dict(nx.all_pairs_shortest_path_length(g))
            pairs = [(a, b) for a in g for b in g if a < b and lengths[a][b] == 3]
            if not pairs:
                continue
            src, dst = pairs[tseed % len(pairs)]
            fail = nx.shortest_path(g, src, dst)[1]
            after = g.copy()
            after.remove_node(fail)
            if not nx.has_path(after, src, dst) or not S.candidates(after, src, dst):
                continue
            if len(S.candidates(g, src, dst)) < 4:
                continue
            out.append({"g": g, "src": src, "dst": dst, "fail": fail,
                        "n": n, "seed": tseed})
            made += 1
    return out


def build_engine(tp, model: str, seed: int):
    return E.Engine(tp["g"], tp["src"], tp["dst"], tp["fail"],
                    model=model, cost_seed=seed)


def stable_probabilities(energies, emax, beta: float):
    """Energy-only Gibbs distribution; subtract min for numerical stability."""
    costs = np.asarray(energies, dtype=float)
    if len(costs) == 0 or not np.all(np.isfinite(costs)) or emax <= 0:
        raise ValueError("Invalid candidate energy costs")
    normalized = costs / float(emax)
    logits = -beta * (normalized - float(normalized.min()))
    weights = np.exp(logits)
    return weights / weights.sum()


def candidate_states(engine):
    """Precompute candidate arrays for intact and failed topology."""
    intact = S.candidates(engine.g0, engine.src, engine.dst)
    failed_graph = engine.g0.copy()
    failed_graph.remove_node(engine.fail_node)
    failed = S.candidates(failed_graph, engine.src, engine.dst)
    if not intact or not failed:
        raise RuntimeError("A topology has no candidate paths in an event state")

    def pack(paths):
        ee, ll = [], []
        for p in paths:
            e, l = E.path_cost_terms(p, engine.e, engine.l, engine.model)
            ee.append(float(e))
            ll.append(float(l))
        return {"paths": tuple(tuple(p) for p in paths),
                "energy": np.asarray(ee, dtype=float),
                "latency": np.asarray(ll, dtype=float)}

    return {"intact": pack(intact), "failed": pack(failed)}


def analytic_mean_energy(states, emax: float, beta: float, down):
    """Expected total-horizon energy for a threat-independent policy."""
    values = {}
    for state_name, d in states.items():
        probs = stable_probabilities(d["energy"], emax, beta)
        values[state_name] = float(np.dot(probs, d["energy"]))
    return float(np.mean([values["failed" if down[t] else "intact"]
                          for t in range(1, HORIZON + 1)]))


def calibrate_beta(train_engines, training_targets, down):
    """Fit beta ONLY using calibration topologies and the NTAR-R target."""
    prepared = [(candidate_states(eng), float(eng.emax)) for eng in train_engines]
    target = float(np.mean(training_targets))

    def expectation(beta):
        return float(np.mean([analytic_mean_energy(st, mx, beta, down)
                              for st, mx in prepared]))

    uniform = expectation(0.0)
    if target > uniform + 1e-9:
        raise RuntimeError(f"Cannot match NTAR-R {target:.6f} J: uniform baseline {uniform:.6f} J is lower")
    hi = 1.0
    while expectation(hi) > target and hi < 100_000.0:
        hi *= 2.0
    floor = expectation(hi)
    if floor > target + 1e-8:
        raise RuntimeError(f"Cannot reach target {target:.6f} J even with beta={hi}: {floor:.6f} J")
    lo = 0.0
    for _ in range(75):
        mid = 0.5 * (lo + hi)
        if expectation(mid) > target:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    return {"beta": float(beta), "target_ntar_r_energy": target,
            "uniform_expected_energy": uniform,
            "matched_expected_energy": expectation(beta),
            "large_beta_energy": floor}


def simulate_energy_matched(engine, T, true_thr, down, seed: int, beta: float, states=None):
    """Match E.simulate's adversary timing, path history and metrics conventions."""
    rng = np.random.default_rng(seed)
    states = candidate_states(engine) if states is None else states
    history = []
    transitions = defaultdict(Counter)
    records = []
    for t in range(1, T + 1):
        state = states["failed" if down[t] else "intact"]
        paths = state["paths"]
        probs = stable_probabilities(state["energy"], engine.emax, beta)

        # Frequency and first-order Markov adversaries act BEFORE the choice.
        freq = Counter(history)
        f_rank = sorted(freq, key=lambda p: (-freq[p], history.index(p)))
        if history and transitions[history[-1]]:
            nxt = transitions[history[-1]]
            m_rank = sorted(nxt, key=lambda p: (-nxt[p], history.index(p)))
            m_rank += [p for p in f_rank if p not in m_rank]
        else:
            m_rank = f_rank

        # Policy-aware attacker knows the Gibbs distribution, beta, energies,
        # topology and eligible paths, but does NOT see the new random draw.
        sorted_prob = np.sort(probs)[::-1]
        pa1 = float(sorted_prob[0])
        pa3 = float(sorted_prob[:3].sum())

        picked_index = int(rng.choice(len(paths), p=probs))
        chosen = paths[picked_index]
        records.append({
            "t": t, "thr": int(true_thr[t]), "reported": int(true_thr[t]),
            "path": chosen,
            "energy": float(state["energy"][picked_index]),
            "latency": float(state["latency"][picked_index]),
            "f1": int(bool(f_rank) and f_rank[0] == chosen),
            "f3": int(chosen in f_rank[:3]),
            "m1": int(bool(m_rank) and m_rank[0] == chosen),
            "m3": int(chosen in m_rank[:3]),
            "pa1": pa1, "pa3": pa3,
            "guessable": int(bool(history)),
        })
        if history:
            transitions[history[-1]][chosen] += 1
        history.append(chosen)
    return records


def average_seed_metrics(records):
    return {k: float(np.mean([r[k] for r in records])) for k in METRICS}


def summary_for_topologies(rows):
    return {k: {"mean": E.ci95([row[k] for row in rows])[0],
                "ci95_half_width": E.ci95([row[k] for row in rows])[1]}
            for k in METRICS}


def holm_adjust(pvals):
    """Holm family-wise-error correction for pre-specified tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    largest = 0.0
    for i, (key, p) in enumerate(items):
        largest = max(largest, min(1.0, (m-i)*p))
        adjusted[key] = largest
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hop", "indep", "anti"), default="hop",
                        help="Main hop-cost model by default; independent/anti are sensitivity checks")
    parser.add_argument("--seeds", type=int, default=5, help="Routing seeds per topology (default 5)")
    parser.add_argument("--out", default="results/energy_matched_baseline.json")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")
    if args.seeds > 100:
        parser.error("--seeds above 100 is not supported")

    train = E.random_topologies()  # Original 30; do not use for final evaluation
    test = heldout_topologies(per_size=10)  # Separate, NEW 30 topologies
    assert len(train) == len(test) == 30
    assert not set(tp["seed"] for tp in train) & set(tp["seed"] for tp in test)
    true_thr, down = E.schedule(HORIZON, THREAT_START, THREAT_LENGTH)

    train_engines = [build_engine(tp, args.model, 3_000_000 + k)
                     for k, tp in enumerate(train)]
    train_target_per_topology = []
    for eng in train_engines:
        sample_metrics = []
        for seed in range(args.seeds):
            run = E.simulate(eng, "ntar", HORIZON, true_thr, down,
                             seed=seed, tau=E.TAU_MAIN)
            sample_metrics.append(E.metrics(run, eng.n_avail))
        train_target_per_topology.append(float(np.mean([m["energy"] for m in sample_metrics])))

    calibration = calibrate_beta(train_engines, train_target_per_topology, down)
    beta = calibration["beta"]
    print("Energy-Matched Baseline | held-out evaluation | cost model:", args.model)
    print(f"Calibration: 30 training topologies, NTAR-R mean energy = {calibration['target_ntar_r_energy']:.6f} J")
    print(f"Softmax beta = {beta:.6f}; analytical training baseline = {calibration['matched_expected_energy']:.6f} J")
    print(f"Held-out: 30 NEW topologies (10 per size), {args.seeds} routing seeds each")

    per_topology = {lab: [] for lab in METHODS}
    test_meta = []
    records_csv = []
    for k, tp in enumerate(test):
        eng = build_engine(tp, args.model, 4_000_000 + k)
        states = candidate_states(eng)
        methods_runs = defaultdict(list)
        for seed in range(args.seeds):
            for lab, method, kwargs in (
                ("Fixed-Weight", "fixed", {}),
                ("Random", "random", {}),
                ("NTAR-R", "ntar", {"tau": E.TAU_MAIN}),
            ):
                rec = E.simulate(eng, method, HORIZON, true_thr, down,
                                 seed=seed, **kwargs)
                methods_runs[lab].append(E.metrics(rec, eng.n_avail))
            rec = simulate_energy_matched(eng, HORIZON, true_thr, down,
                                          seed=seed, beta=beta, states=states)
            methods_runs["Energy-Matched Random"].append(E.metrics(rec, eng.n_avail))

        for lab in METHODS:
            averaged = average_seed_metrics(methods_runs[lab])
            per_topology[lab].append(averaged)
            records_csv.append({"topology_index": k, "graph_seed": tp["seed"],
                                "nodes": tp["n"], "method": lab, **averaged})
        test_meta.append({"graph_seed": tp["seed"], "nodes": tp["n"],
                          "candidate_paths": len(states["intact"]["paths"]),
                          "candidate_paths_during_failure": len(states["failed"]["paths"]),
                          "src": tp["src"], "dst": tp["dst"], "fail": tp["fail"]})

    summary = {lab: summary_for_topologies(per_topology[lab]) for lab in METHODS}
    comp = {}
    for competitor in ("Energy-Matched Random", "Random", "Fixed-Weight"):
        diffs = {}
        for metric in METRICS:
            delta = [a[metric] - b[metric] for a, b in
                     zip(per_topology["NTAR-R"], per_topology[competitor])]
            dm, ci = E.ci95(delta)
            diffs[metric] = {"mean_difference": dm,
                             "ci95_half_width": ci,
                             "wilcoxon_p": E.wilcoxon_p(delta)}
        primary = ("energy", "m1", "pa1")
        corrected = holm_adjust({k: diffs[k]["wilcoxon_p"] for k in primary})
        for metric in primary:
            diffs[metric]["holm_adjusted_p_primary_family"] = corrected[metric]
        comp[competitor] = diffs

    base_e = summary["Energy-Matched Random"]["energy"]["mean"]
    ntar_e = summary["NTAR-R"]["energy"]["mean"]
    relative_gap = (ntar_e - base_e) / ntar_e * 100 if ntar_e else float("nan")

    output = {
        "design": {
            "purpose": "Energy-matched stochastic routing baseline, no test-set retuning",
            "graph_family": "Connected Erdos-Renyi, expected degree 3.6, path count >=4, survivable failure",
            "training_topologies": 30, "heldout_topologies": 30,
            "sizes": list(SIZES), "new_test_seed_range": "1_000_000 + 1000*n + offset",
            "routing_seeds_per_topology": args.seeds,
            "time_steps": HORIZON, "threat_steps": f"{THREAT_START}-{THREAT_START+THREAT_LENGTH-1}",
            "cost_model": args.model, "beta_fit_target": "NTAR-R mean total-horizon energy on TRAINING topologies",
            "beta_fit_method": "Bisection of analytically expected softmax energy on training topologies",
            "policy": "P(p) proportional to exp(-beta * E(p)/Emax)",
            "policy_uses_threat_or_history": False,
            "test_recalibration": False,
            "sampling_unit_for_inference": "topology (seeds averaged within topology)",
            "primary_confirmatory_metrics": ["energy", "m1", "pa1"],
            "primary_multiple_testing": "Holm within each NTAR-R vs competitor comparison",
            "policy_aware_metric": "Bayes-optimal expected hit rate given full selection distribution",
        },
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "networkx": nx.__version__,
                        "code_commit": git_commit()},
        "calibration": calibration,
        "test_topology_metadata": test_meta,
        "heldout_method_summary": summary,
        "heldout_ntar_r_minus_competitor": comp,
        "heldout_energy_gap_ntar_r_vs_matched_percent": relative_gap,
        "heldout_per_topology_metrics": per_topology,
        "interpretation_note": ("A lower policy-aware hit rate is better, but a match on training energy "
                                "does not guarantee exact matching on new networks. "
                                "If held-out energy differs materially, do not retune using the held-out set."),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("topology_index", "graph_seed", "nodes", "method") + METRICS)
        writer.writeheader()
        writer.writerows(records_csv)

    print("\nHELD-OUT RESULTS (mean +/- 95% CI over 30 independent topologies):")
    print(f"{'Method':22s} {'Energy J':>16s} {'Markov top-1':>17s} {'Policy-aware top-1':>21s}")
    for lab in METHODS:
        s = summary[lab]
        def fm(k):
            return f"{s[k]['mean']:.3f}+/-{s[k]['ci95_half_width']:.3f}"
        print(f"{lab:22s} {fm('energy'):>16s} {fm('m1'):>17s} {fm('pa1'):>21s}")

    print("\nPAIRED DIFFERENCE: NTAR-R minus Energy-Matched Random")
    for metric in ("energy", "latency", "m1", "pa1", "f1"):
        d = comp["Energy-Matched Random"][metric]
        correction = (f", Holm p={d['holm_adjusted_p_primary_family']:.5g}"
                      if "holm_adjusted_p_primary_family" in d else "")
        print(f"  {metric:8s} delta={d['mean_difference']:+.5f} "
              f"+/- {d['ci95_half_width']:.5f} (95% CI half-width), "
              f"Wilcoxon p={d['wilcoxon_p']:.5g}{correction}")
    print(f"Held-out energy mismatch relative to NTAR-R: {relative_gap:+.2f}%")
    if abs(relative_gap) > 3.0:
        print("CAUTION: >3% held-out energy mismatch. Report this; DO NOT retune beta on test networks.")
    print("Wrote", output_path, "and", csv_path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
