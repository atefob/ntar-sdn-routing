"""Reproducible, hop-cap-matched candidate-search benchmark for NTAR Table 8.

Place next to experiments2.py and ntar_sim.py and run from the project root:
    python table8_revision.py

Does not overwrite results2.json, tables.tex, or any baseline artifacts.
Writes results/table8_revision.json and results/table8_revision.csv.

Measures candidate generation PLUS route scoring/selection for:
  - exhaustive DFS, all simple paths of <= 4 hops
  - Yen/shortest_simple_paths, all simple paths of <= 4 hops (set-matched control)
  - Yen/shortest_simple_paths, first k=16 simple paths of <= 4 hops
Timing excludes graph generation, engine construction, and SDN rule installation.
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np

import experiments2 as E
import ntar_sim as S

MAX_HOPS = S.MAX_NODES - 1
K_BOUND = 16
N_AND_DEGREE = [(n, degree) for n in (20, 50, 100, 200) for degree in (3.6, 6.0)]


def yen_with_hop_limit(graph, source, destination, max_hops=MAX_HOPS, k=None):
    """Yield only paths satisfying the *same* hop cap as DFS.

    shortest_simple_paths is nondecreasing in hop count on an unweighted graph,
    so it is safe to stop after the first path that exceeds max_hops.
    """
    paths = []
    for path in nx.shortest_simple_paths(graph, source, destination):
        if len(path) - 1 > max_hops:
            break
        paths.append(path)
        if k is not None and len(paths) >= k:
            break
    return paths


def find_topologies(n, degree, n_graphs):
    """Recreate experiments2.py's random-graph and distance-3 selection."""
    made = 0
    seed = 50_000 + n * 10 + int(degree)
    attempts = 0
    while made < n_graphs:
        attempts += 1
        if attempts > 50_000:
            raise RuntimeError(f"Could not sample {n_graphs} graphs for n={n}, degree={degree}")
        seed += 1
        graph = nx.erdos_renyi_graph(n, degree / (n - 1), seed=seed)
        if not nx.is_connected(graph):
            continue
        rng = np.random.default_rng(seed)
        nodes = list(graph)
        # Equivalent to the original distance-3 endpoint condition, but bounded.
        chosen = None
        for _ in range(10_000):
            src, dst = rng.choice(nodes, 2, replace=False)
            if nx.shortest_path_length(graph, int(src), int(dst)) == 3:
                chosen = int(src), int(dst)
                break
        if chosen is None:
            continue
        made += 1
        yield seed, graph, chosen[0], chosen[1]


def score_best(engine, paths):
    if not paths:
        raise AssertionError("No valid candidate paths")
    # This uses the exact engine/scoring implementation of experiments2.py.
    return min(engine.ranked(paths, E.W_THR, {}))[2]


def time_one_action(action, repeats, batches):
    """Report the median batch time per decision (milliseconds)."""
    for _ in range(2):
        action()  # warmup, excluded from measurements
    ms_per_decision = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(repeats):
            action()
        ms_per_decision.append((time.perf_counter() - start) * 1000.0 / repeats)
    return float(statistics.median(ms_per_decision))


def mean_ci(values):
    mean, half = E.ci95(values)
    return {"mean": float(mean), "ci95_half_width": float(half)}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(graphs_per_config=5, repeats=20, batches=3):
    if MAX_HOPS != 4:
        raise RuntimeError(f"Unexpected hop cap {MAX_HOPS}, expected 4; inspect ntar_sim.py")
    if graphs_per_config < 2 or repeats < 1 or batches < 1:
        raise ValueError("Use at least 2 graphs per configuration, 1 repeat and 1 batch")

    out = {
        "design": {
            "graph_generator": "Erdos-Renyi G(n, degree/(n-1)); connected graphs",
            "source_destination": "3-hop shortest path, same selection as experiments2.py",
            "graph_sizes": [20, 50, 100, 200],
            "mean_degrees": [3.6, 6.0],
            "graphs_per_configuration": graphs_per_config,
            "max_candidate_hops_all_methods": MAX_HOPS,
            "yen_k": K_BOUND,
            "timing_repeats_per_batch": repeats,
            "timing_batches_per_graph": batches,
            "timing_per_graph": "median of batches, millisecond per decision",
            "confidence_interval_unit": "independent graph (95% t CI across graph-level medians)",
            "timed_operations": "candidate generation + NTAR engine path cost and argmin",
            "not_timed": "graph generation, engine setup, flow-rule installation, network latency",
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "git_commit": git_commit(),
        },
        "by_configuration": [],
        "per_graph": [],
    }
    methods = {
        "dfs_all": lambda g, s, d: S.candidates(g, s, d),
        "yen_all": lambda g, s, d: yen_with_hop_limit(g, s, d),
        "yen16": lambda g, s, d: yen_with_hop_limit(g, s, d, k=K_BOUND),
    }
    print("Table 8 revision | all methods <= 4 hops | generation + scoring")
    print("Nodes  Degree   All paths       Yen k=16        DFS ms            Yen-all ms        Yen16 ms")
    for n, degree in N_AND_DEGREE:
        rows = []
        for top_seed, graph, src, dst in find_topologies(n, degree, graphs_per_config):
            engine = E.Engine(graph, src, dst, fail_node=-1, model="hop")
            computed = {name: fn(graph, src, dst) for name, fn in methods.items()}
            full_set = {tuple(x) for x in computed["dfs_all"]}
            yen_set = {tuple(x) for x in computed["yen_all"]}
            if len(full_set) != len(computed["dfs_all"]):
                raise AssertionError("Duplicate DFS paths")
            if full_set != yen_set:
                raise AssertionError(f"DFS/Yen set mismatch on graph seed={top_seed}")
            if len(computed["yen16"]) != min(K_BOUND, len(full_set)):
                raise AssertionError("Incorrect bounded Yen candidate count")
            if any(len(p) - 1 > MAX_HOPS for pp in computed.values() for p in pp):
                raise AssertionError("Hop-limit violation")
            actions = {
                name: (lambda f=fn: score_best(engine, f(graph, src, dst)))
                for name, fn in methods.items()
            }
            measured = {name: time_one_action(action, repeats, batches)
                        for name, action in actions.items()}
            row = {
                "nodes": n,
                "degree": degree,
                "topology_seed": top_seed,
                "source": src,
                "destination": dst,
                "all_candidates": len(full_set),
                "yen16_candidates": len(computed["yen16"]),
                "was_truncated": len(computed["yen16"]) < len(full_set),
                "ms_dfs_all": measured["dfs_all"],
                "ms_yen_all": measured["yen_all"],
                "ms_yen16": measured["yen16"],
            }
            rows.append(row)
            out["per_graph"].append(row)
        summary = {
            "nodes": n,
            "degree": degree,
            "all_candidates": mean_ci([row["all_candidates"] for row in rows]),
            "yen16_candidates": mean_ci([row["yen16_candidates"] for row in rows]),
            "truncated_graphs": sum(row["was_truncated"] for row in rows),
            "ms_dfs_all": mean_ci([row["ms_dfs_all"] for row in rows]),
            "ms_yen_all": mean_ci([row["ms_yen_all"] for row in rows]),
            "ms_yen16": mean_ci([row["ms_yen16"] for row in rows]),
            "candidate_set_equal_dfs_vs_yen_all": True,
        }
        out["by_configuration"].append(summary)
        def fmt(metric, decimals=2):
            data = summary[metric]
            return f"{data['mean']:.{decimals}f}±{data['ci95_half_width']:.{decimals}f}"
        print(f"{n:5d} {degree:7.1f}  {fmt('all_candidates'):15s} "
              f"{fmt('yen16_candidates'):15s} {fmt('ms_dfs_all', 3):17s} "
              f"{fmt('ms_yen_all', 3):17s} {fmt('ms_yen16', 3)}")

    target = Path("results")
    target.mkdir(exist_ok=True, parents=True)
    json_file = target / "table8_revision.json"
    csv_file = target / "table8_revision.csv"
    json_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out["per_graph"][0]))
        writer.writeheader()
        writer.writerows(out["per_graph"])
    print("\nChecked: DFS and complete Yen candidate SETS match for every graph.")
    print("Wrote", json_file, "and", csv_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Controlled 4-hop candidate search benchmark for Table 8")
    ap.add_argument("--graphs", type=int, default=5, help="Independent graphs per configuration")
    ap.add_argument("--repeats", type=int, default=20, help="Timing repetitions in each batch")
    ap.add_argument("--batches", type=int, default=3, help="Batches per graph")
    args = ap.parse_args()
    run(args.graphs, args.repeats, args.batches)
