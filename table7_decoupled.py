"""Table 7: decoupled energy and latency link costs.

Three independent link-cost draws per topology and five routing seeds per draw.
Run from the repository root:  python table7_decoupled.py
Writes results/table7_decoupled.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import experiments2 as E
import ntar_sim as S

N_COST_DRAWS = 3
N_ROUTING_SEEDS = 5
METHODS = {
    "Random": ("random", {}),
    "Fixed-Weight": ("fixed", {}),
    "NTAR": ("ntar", {}),
    "NTAR-R": ("ntar", {"tau": E.TAU_MAIN}),
}
METRICS = ("energy", "latency", "f1", "m1", "pa1")


def run_table7():
    topologies = E.random_topologies()
    true_thr, down = E.schedule(40, 15, 10)
    output = {
        "design": {
            "independent_topologies": len(topologies),
            "link_cost_draws_per_topology": N_COST_DRAWS,
            "routing_seeds_per_draw": N_ROUTING_SEEDS,
            "simulation_horizon": 40,
            "threat_window": "15-24 (10 steps)",
            "link_cost_seed_formula": "1000000 + 100 * topology_index + draw_index",
            "aggregation": "Average draws and routing seeds within each topology; 95% CI across 30 topologies",
            "note": "All routing methods share exactly the same link costs for each topology and cost draw.",
        },
        "models": {},
    }
    for model in ("indep", "anti"):
        per_topology = defaultdict(list)
        change_rates = []
        distinct_draw_check = 0
        for k, tp in enumerate(topologies):
            records = defaultdict(list)
            weight_changes = []
            seen_cost_vectors = set()
            for draw in range(N_COST_DRAWS):
                cost_seed = 1_000_000 + 100 * k + draw
                engine = E.Engine(tp["g"], tp["src"], tp["dst"], tp["fail"],
                                  model=model, cost_seed=cost_seed)
                cost_vector = tuple(sorted((tuple(sorted(link)), round(val, 14))
                                           for link, val in engine.e.items()))
                if cost_vector in seen_cost_vectors:
                    raise AssertionError(f"Duplicate energy cost draw: model={model} topology={k} draw={draw}")
                seen_cost_vectors.add(cost_vector)
                candidates = S.candidates(tp["g"], tp["src"], tp["dst"])
                best_e = min(engine.ranked(candidates, (0.7, 0.1, 0.2), {}))[2]
                best_l = min(engine.ranked(candidates, (0.1, 0.7, 0.2), {}))[2]
                weight_changes.append(int(best_e != best_l))
                for label, (method, options) in METHODS.items():
                    for route_seed in range(N_ROUTING_SEEDS):
                        seq = E.simulate(engine, method, 40, true_thr, down,
                                         seed=route_seed, **options)
                        records[label].append(E.metrics(seq, engine.n_avail))
            distinct_draw_check += len(seen_cost_vectors)
            change_rates.append(float(np.mean(weight_changes)))
            for label, samples in records.items():
                per_topology[label].append({metric: float(np.mean([s[metric] for s in samples]))
                                            for metric in METRICS})
        summary = {}
        for label, rows in per_topology.items():
            summary[label] = {
                metric: dict(mean=E.ci95([r[metric] for r in rows])[0],
                             ci95_half_width=E.ci95([r[metric] for r in rows])[1])
                for metric in METRICS
            }
        paired = {}
        for competitor in ("Random", "Fixed-Weight"):
            paired[competitor] = {}
            for metric in METRICS:
                diffs = [a[metric] - b[metric] for a, b in
                         zip(per_topology["NTAR-R"], per_topology[competitor])]
                mean, half_width = E.ci95(diffs)
                paired[competitor][metric] = dict(
                    mean_difference=mean, ci95_half_width=half_width,
                    wilcoxon_p=E.wilcoxon_p(diffs))
        output["models"][model] = {
            "method_summary": summary,
            "paired_ntar_r_minus_competitor": paired,
            "weight_change_fraction_by_topology": change_rates,
            "mean_fraction_of_draws_changing_path": float(np.mean(change_rates)),
            "topologies_with_any_weight_change": sum(x > 0 for x in change_rates),
            "verified_distinct_link_cost_draws": distinct_draw_check,
            "per_topology": dict(per_topology),
        }
        print(f"\n{model.upper()} | {len(topologies)} topologies, "
              f"{distinct_draw_check} independent link-cost draws")
        print("Method         Energy(J)         Latency(ms)       Freq top-1       Markov top-1     Policy-aware top-1")
        for label in METHODS:
            s = summary[label]
            print(f"{label:14s} " + " ".join(
                f"{s[m]['mean']:.3f}±{s[m]['ci95_half_width']:.3f}".ljust(17)
                for m in METRICS))
        print("Topologies with at least one weight-dependent route change:",
              output["models"][model]["topologies_with_any_weight_change"])
    output_path = Path("results/table7_decoupled.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nWrote", output_path)


if __name__ == "__main__":
    run_table7()
