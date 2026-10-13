"""
NTAR: single-source simulator for every table and figure in the revised manuscript.

The routing engine reproduces the authors' original baseline script
(code_com_baseline) step for step: same topology, same candidate-path order,
same tie-breaking (first minimum in networkx enumeration order), same RNG call
order and the same uniform cost noise. With --check it verifies that the
20-run benchmark matches the original script's Table 2 to four decimals.

On top of that engine it adds what the revision needs: the available-path
entropy, a passive-adversary hit rate, the four-way cost-form ablation, a
threat-weight sweep and 30 random topologies.

Run:  python ntar_sim.py --out results [--check]
"""
import argparse
import json
import math
import os
import random
from collections import Counter

import networkx as nx
import numpy as np

# ---------------------------------------------------------------- configuration
N_NODES, P_EDGE, TOPO_SEED = 6, 0.6, 42
SRC, DST = 0, 5
MAX_NODES = 5                    # candidate paths have <= 5 nodes, i.e. <= 4 hops
H_NORM = 5                       # normalization constant used in Eq. (4): 5 hops
E_MAX = H_NORM * 0.05            # 0.25
L_MAX = H_NORM * 10.0 / 100.0    # 0.50
HORIZON, N_RUNS = 25, 20
W_NOMINAL, W_THREAT = (0.4, 0.4, 0.2), (0.2, 0.2, 0.6)
EVENTS = {8: ("fail", 3), 12: ("restore", 3), 15: ("threat", 1), 20: ("threat", 0)}
FAIL_NODE = 3
PHASES = [("Normal I", 1, 7), ("Fault & Recovery", 8, 14),
          ("Threat Escalation", 15, 19), ("Normal II", 20, 25)]


# ---------------------------------------------------------------- environment
class Env:
    """Mirrors SimulatedSDNNetwork in the original script."""

    def __init__(self, graph, n_nodes):
        self.original = graph
        self.graph = graph.copy()
        self.n = n_nodes
        self.energy = np.random.uniform(0.7, 1.0, size=n_nodes)   # consumes RNG as the original
        self.history = []

    def fail(self, v):
        if v in self.graph:
            self.graph.remove_node(v)
            self.energy[v] = 0.0

    def restore(self, v):
        if v not in self.graph:
            self.graph.add_node(v)
            for a, b in self.original.edges:
                if v in (a, b) and a in self.graph and b in self.graph:
                    self.graph.add_edge(a, b)
            self.energy[v] = 0.8

    def measure(self, path):
        hops = len(path) - 1
        e = hops * 0.05 + np.random.uniform(0.01, 0.03)
        l = hops * 10.0 + np.random.uniform(1.0, 5.0)
        self.history.append(tuple(path))
        return e, l

    def consume(self, path):
        for v in path:
            if v < len(self.energy):
                self.energy[v] = max(0.1, self.energy[v] - 0.02)


def build_topology(n=N_NODES, p=P_EDGE, seed=TOPO_SEED):
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    while not nx.is_connected(g):
        seed += 1
        g = nx.erdos_renyi_graph(n, p, seed=seed)
    return g, seed


def candidates(graph, s, d):
    if s not in graph or d not in graph:
        return []
    # cutoff prunes the DFS without changing the order of the kept paths
    return [p for p in nx.all_simple_paths(graph, s, d, cutoff=MAX_NODES - 1) if len(p) <= MAX_NODES]


# ---------------------------------------------------------------- cost functions
def cost(path, w, history, form):
    """form: 'eq3'        original cost with unweighted repetition penalty R(p)
             'eq3_noR'    unnormalized, R(p) removed      (isolates R(p))
             'eq5_withR'  normalized, R(p) kept           (isolates normalization)
             'eq5'        normalized, R(p) removed        (NTAR, Eq. (5))"""
    hops = len(path) - 1
    e, l = hops * 0.05, hops * 10.0 / 100.0
    rep = history.count(tuple(path))
    b = 1.0 / (rep + 1)
    if form.startswith("eq5"):
        e, l = e / E_MAX, l / L_MAX
    c = w[0] * e + w[1] * l - w[2] * b
    if form in ("eq3", "eq5_withR"):
        c += rep * 0.5
    return c


def argmin_first(paths, key):
    best, best_c = None, float("inf")
    for p in paths:                       # strict '<': first minimum wins, as in the original
        c = key(p)
        if c < best_c:
            best, best_c = p, c
    return best


# ---------------------------------------------------------------- metrics
def entropy_bits(counts):
    tot = sum(counts.values())
    return -sum((c / tot) * math.log2(c / tot + 1e-6) for c in counts.values())


def h_obs(counts):
    k = len(counts)
    return 0.0 if k <= 1 else entropy_bits(counts) / math.log2(k)


def h_avail(counts, n_avail):
    return entropy_bits(counts) / math.log2(n_avail)


# ---------------------------------------------------------------- one run
def run(graph, method, seed, form="eq5", src=SRC, dst=DST, fail_node=FAIL_NODE,
        w_threat=None, n_nodes=None):
    w_threat = w_threat or W_THREAT
    random.seed(seed)
    np.random.seed(seed)
    env = Env(graph, n_nodes or graph.number_of_nodes())
    n_avail = len(candidates(graph, src, dst))
    events = {8: ("fail", fail_node), 12: ("restore", fail_node), 15: ("threat", 1), 20: ("threat", 0)}
    thr, trace = 0, []
    for step in range(1, HORIZON + 1):
        if step in events:
            kind, v = events[step]
            if kind == "fail":
                env.fail(v)
            elif kind == "restore":
                env.restore(v)
            else:
                thr = v
        w = w_threat if thr else W_NOMINAL

        counts = Counter(env.history)
        guess = None
        if counts:
            top = max(counts.values())
            guess = next(p for p in env.history if counts[p] == top)   # earliest-seen among most frequent

        cands = candidates(env.graph, src, dst)
        if not cands:
            continue
        if method == "static":
            path = nx.shortest_path(env.graph, src, dst)
        elif method == "random":
            path = random.choice(cands)
        elif method == "fixed":
            path = argmin_first(cands, lambda p: cost(p, W_NOMINAL, env.history, form))
        elif method == "ntar":
            path = argmin_first(cands, lambda p: cost(p, w, env.history, form))
        else:
            raise ValueError(method)

        e, l = env.measure(path)
        counts = Counter(env.history)
        ho, ha = h_obs(counts), h_avail(counts, n_avail)
        reward = w[0] * (-e) + w[1] * (-l / 100.0) + w[2] * ho
        env.consume(path)
        trace.append(dict(step=step, threat=thr, node_down=int(fail_node not in env.graph),
                          path="-".join(map(str, path)), hops=len(path) - 1, energy=e, latency=l,
                          h_obs=ho, h_avail=ha, reward=reward,
                          hit=int(guess == tuple(path)) if guess else 0,
                          guessable=int(guess is not None)))
    return trace


def summarize(trace):
    thr = [t for t in trace if t["threat"]]
    pre = {t["path"] for t in trace if t["step"] < 15}
    return dict(
        energy=np.mean([t["energy"] for t in trace]),
        latency=np.mean([t["latency"] for t in trace]),
        h_obs=np.mean([t["h_obs"] for t in trace]),        # per-step mean, as in the original Table 2
        h_avail=np.mean([t["h_avail"] for t in trace]),
        h_avail_final=trace[-1]["h_avail"],
        reward=np.mean([t["reward"] for t in trace]),
        unique=len({t["path"] for t in trace}),
        unique_threat=len({t["path"] for t in thr}),
        new_in_threat=len({t["path"] for t in thr} - pre),
        adv_hit_threat=np.mean([t["hit"] for t in thr]) if thr else float("nan"),
    )


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


# ---------------------------------------------------------------- main
ORIGINAL_TABLE2 = {   # output of the authors' original script (np.std, ddof = 0)
    "static": dict(energy=0.1195, latency=22.9657, h_obs=0.0000, reward=-0.1256, unique=1.0),
    "random": dict(energy=0.1813, latency=35.3257, h_obs=0.9120, reward=0.0663, unique=11.1),
    "fixed": dict(energy=0.1295, latency=24.9657, h_obs=0.8085, reward=0.0897, unique=7.0),
    "ntar": dict(energy=0.1455, latency=28.1657, h_obs=0.8390, reward=0.0900, unique=11.0),
}


def main(out, check=False):
    os.makedirs(out, exist_ok=True)
    g, used_seed = build_topology()
    n_avail = len(candidates(g, SRC, DST))
    meta = dict(topology_seed=used_seed, edges=sorted(g.edges()), n_paths_available=n_avail,
                max_hops=MAX_NODES - 1, runs=N_RUNS, horizon=HORIZON,
                noise="energy +U(0.01,0.03) J, latency +U(1,5) ms per transmission")

    bench, traces = {}, {}
    for m in ("static", "random", "fixed", "ntar"):
        rows = []
        for s in range(N_RUNS):
            tr = run(g, m, s)
            rows.append(summarize(tr))
            if s == 0:
                traces[m] = tr
        bench[m] = rows

    keys = ["energy", "latency", "h_obs", "h_avail", "h_avail_final", "reward",
            "unique", "unique_threat", "new_in_threat", "adv_hit_threat"]
    table2 = {m: {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
                  for k in keys} for m, rows in bench.items()}   # ddof = 0, as in the original
    table3 = {m: {k: welch([r[k] for r in bench["ntar"]], [r[k] for r in bench[m]])
                  for k in ("energy", "latency", "reward")} for m in ("static", "random", "fixed")}

    if check:
        for m, ref in ORIGINAL_TABLE2.items():
            for k, v in ref.items():
                got = table2[m][k][0]
                tol = 0.051 if k == "unique" else 6e-5      # 'unique' was printed with one decimal
                assert abs(got - v) < tol, (m, k, got, v)
        print("check: benchmark reproduces the original Table 2")

    meta["ntar_path_sequences_distinct_across_seeds"] = len(
        {tuple(t["path"] for t in run(g, "ntar", s)) for s in range(N_RUNS)})

    tr = traces["ntar"]
    table1 = []
    for name, a, b in PHASES:
        ph = [t for t in tr if a <= t["step"] <= b]
        table1.append(dict(phase=name, steps=f"{a}-{b}",
                           energy=float(np.mean([t["energy"] for t in ph])),
                           latency=float(np.mean([t["latency"] for t in ph])),
                           h_avail_end=ph[-1]["h_avail"], h_obs_end=ph[-1]["h_obs"],
                           reward=float(np.mean([t["reward"] for t in ph])),
                           distinct=len({t["path"] for t in ph}), length=len(ph)))

    ablation = {}
    for form in ("eq3", "eq3_noR", "eq5_withR", "eq5"):
        tf, ta = run(g, "fixed", 0, form), run(g, "ntar", 0, form)
        sf, sa = summarize(tf), summarize(ta)
        diff = [x["step"] for x, y in zip(tf, ta) if x["path"] != y["path"]]
        ablation[form] = dict(steps_where_adaptive_differs=len(diff),
                              first_diff_step=diff[0] if diff else None,
                              unique_fixed=sf["unique"], unique_adaptive=sa["unique"],
                              energy_fixed=sf["energy"], energy_adaptive=sa["energy"],
                              adv_hit_threat_fixed=sf["adv_hit_threat"],
                              adv_hit_threat_adaptive=sa["adv_hit_threat"],
                              seq_fixed=[x["path"] for x in tf], seq_adaptive=[x["path"] for x in ta])

    sweep = []
    for w3 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        s = summarize(run(g, "ntar", 0, w_threat=((1 - w3) / 2, (1 - w3) / 2, w3)))
        sweep.append(dict(w3=w3, energy=s["energy"], unique=s["unique"],
                          new_in_threat=s["new_in_threat"], adv_hit_threat=s["adv_hit_threat"]))

    gen = []
    for n in (10, 15, 20):
        made, tseed = 0, 1000 * n
        while made < 10:
            tseed += 1
            gg = nx.erdos_renyi_graph(n, 3.6 / (n - 1), seed=tseed)
            if not nx.is_connected(gg):
                continue
            d = dict(nx.all_pairs_shortest_path_length(gg))
            pairs = [(a, b) for a in gg for b in gg if a < b and d[a][b] == 3]
            if not pairs:
                continue
            a, b = pairs[tseed % len(pairs)]
            fn = nx.shortest_path(gg, a, b)[1]
            h = gg.copy()
            h.remove_node(fn)
            if not nx.has_path(h, a, b) or not candidates(h, a, b):
                continue
            npaths = len(candidates(gg, a, b))
            if npaths < 4:
                continue
            made += 1
            row = dict(n=n, topo_seed=tseed, src=a, dst=b, fail=fn, n_paths=npaths)
            for m in ("random", "fixed", "ntar"):
                ss = [summarize(run(gg, m, s, src=a, dst=b, fail_node=fn)) for s in range(5)]
                for k in ("energy", "unique", "new_in_threat", "adv_hit_threat"):
                    row[f"{m}_{k}"] = float(np.mean([x[k] for x in ss]))
            for form in ("eq3", "eq3_noR", "eq5_withR", "eq5"):
                tf = run(gg, "fixed", 0, form, a, b, fn)
                ta = run(gg, "ntar", 0, form, a, b, fn)
                row[f"differs_{form}"] = sum(x["path"] != y["path"] for x, y in zip(tf, ta))
                row[f"energy_nominal_{form}"] = float(np.mean([x["energy"] for x in tf if not x["threat"]]))
            gen.append(row)

    res = dict(meta=meta, table1=table1, table2=table2, table3=table3, ablation=ablation,
               sweep=sweep, generalization=gen, trace_ntar=traces["ntar"],
               trace_fixed=traces["fixed"], trace_static=traces["static"],
               trace_random=traces["random"])
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    main(a.out, a.check)
