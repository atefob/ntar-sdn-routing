"""
Second-round experiments for the NTAR revision (responses to the external review).

Uses a general engine that shares NTAR's cost function, candidate generation and
tie-breaking with ntar_sim.py, and adds:
  E1  randomized NTAR (NTAR-R): uniform choice among tau-near-optimal paths
  E2  three adversaries (frequency, first-order Markov, policy-aware) + top-k hit rate,
      threat windows of 5, 10 and 20 steps
  E3  imperfect threat signal: false alarms, missed detections, detection delay, flapping
  E4  decoupled per-link energy and latency (independent and anti-correlated)
  E5  periodic MTD baseline (threat-independent route mutation every M steps)
  E6  scalability: candidate-set size and per-decision time on 50/100/200-node graphs
  E7  topology-level paired statistics (95% CI, Wilcoxon signed-rank)
  E8  counterexample to an unconditional reading of Proposition 1 (small rho)

Run:  python experiments2.py --out results
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from scipy import stats

import ntar_sim as S

W_NOM, W_THR = S.W_NOMINAL, S.W_THREAT
H_N = S.H_NORM


# ---------------------------------------------------------------- generic engine
def link_costs(graph, model, seed):
    """Per-link energy (J) and latency (ms).
    'hop'      : 0.05 J and 10 ms per hop (the paper's main model)
    'indep'    : energy ~ U(0.02, 0.08), latency ~ U(4, 16), independent per link
    'anti'     : latency = 20 - 200*energy (+ small noise): low-energy links are slow"""
    rng = np.random.default_rng(10_000 + seed)
    e, l = {}, {}
    for a, b in graph.edges:
        if model == "hop":
            ee, ll = 0.05, 10.0
        elif model == "indep":
            ee, ll = rng.uniform(0.02, 0.08), rng.uniform(4.0, 16.0)
        elif model == "anti":
            ee = rng.uniform(0.02, 0.08)
            ll = max(1.0, 20.0 - 200.0 * ee + rng.normal(0, 0.5))
        else:
            raise ValueError(model)
        e[frozenset((a, b))], l[frozenset((a, b))] = ee, ll
    return e, l


def path_cost_terms(p, e, l, model="indep"):
    if model == "hop":                      # identical floating-point form to the original script
        h = len(p) - 1
        return h * 0.05, h * 10.0
    edges = [frozenset((p[i], p[i + 1])) for i in range(len(p) - 1)]
    return sum(e[x] for x in edges), sum(l[x] for x in edges)


class Engine:
    def __init__(self, graph, src, dst, fail_node, model="hop", cost_seed=0):
        self.g0, self.src, self.dst, self.fail_node = graph, src, dst, fail_node
        self.e, self.l = link_costs(graph, model, cost_seed)
        self.model = model
        cands = S.candidates(graph, src, dst)
        self.n_avail = len(cands)
        if model == "hop":
            self.emax, self.lmax = H_N * 0.05, H_N * 10.0 / 100.0
        else:   # maximum attainable over the intact candidate set
            self.emax = max(path_cost_terms(p, self.e, self.l, model)[0] for p in cands)
            self.lmax = max(path_cost_terms(p, self.e, self.l, model)[1] for p in cands) / 100.0

    def cost(self, p, w, rep, rho=0.0, normalize=True):
        ee, ll = path_cost_terms(p, self.e, self.l, self.model)
        ll /= 100.0
        if normalize:
            ee, ll = ee / self.emax, ll / self.lmax
        r = rep.get(tuple(p), 0)
        c = w[0] * ee + w[1] * ll - w[2] * (1.0 / (r + 1))   # same float form as the original
        return c + rho * r if rho else c

    def ranked(self, cands, w, rep, rho=0.0, normalize=True):
        """Candidates with costs, stable in enumeration order."""
        return [(self.cost(p, w, rep, rho, normalize), i, p) for i, p in enumerate(cands)]


def schedule(T, threat_on, threat_len, fail_step=8, restore_step=12):
    true_thr = np.zeros(T + 1, dtype=int)
    true_thr[threat_on:threat_on + threat_len] = 1
    down = np.zeros(T + 1, dtype=int)
    down[fail_step:restore_step] = 1
    return true_thr, down


def corrupt_signal(true_thr, rng, p_fa=0.0, p_miss=0.0, delay=0, p_flip=0.0):
    T = len(true_thr) - 1
    rep = np.zeros_like(true_thr)
    for t in range(1, T + 1):
        src = true_thr[t - delay] if t - delay >= 1 else 0
        r = src
        if src == 1 and rng.random() < p_miss:
            r = 0
        if src == 0 and rng.random() < p_fa:
            r = 1
        if rng.random() < p_flip:
            r = 1 - r
        rep[t] = r
    return rep


def simulate(eng, method, T, true_thr, down, reported=None, seed=0, tau=0.0,
             period=1, rho=0.0, normalize=True, w_thr=None):
    """method: 'static' | 'random' | 'fixed' | 'ntar' | 'mtd'
    tau > 0 turns 'ntar'/'fixed' into the randomized variant (uniform over paths whose
    cost is within tau of the minimum). Returns per-step records with what each
    adversary predicted."""
    w_thr = w_thr or W_THR
    reported = true_thr if reported is None else reported
    rng = np.random.default_rng(seed)
    rep = Counter()
    hist = []
    trans = defaultdict(Counter)
    current = None
    out = []
    g = eng.g0.copy()          # mutated in place on failure/restoration, as in the original script
    for t in range(1, T + 1):
        if down[t] and not down[t - 1] and eng.fail_node in g:
            g.remove_node(eng.fail_node)
        if not down[t] and t > 1 and down[t - 1] and eng.fail_node not in g:
            g.add_node(eng.fail_node)
            for a, b in eng.g0.edges:
                if eng.fail_node in (a, b) and a in g and b in g:
                    g.add_edge(a, b)
        cands = S.candidates(g, eng.src, eng.dst)
        w = w_thr if reported[t] else W_NOM
        if method in ("fixed",):
            w = W_NOM
        # ---- adversary predictions (made before the controller acts) ----------
        freq = Counter(hist)
        f_rank = sorted(freq, key=lambda p: (-freq[p], hist.index(p)))
        if hist and trans[hist[-1]]:
            nxt = trans[hist[-1]]
            m_rank = sorted(nxt, key=lambda p: (-nxt[p], hist.index(p)))
            m_rank += [p for p in f_rank if p not in m_rank]
        else:
            m_rank = f_rank
        # policy-aware: knows the cost function, the weights in use and the history
        if method in ("fixed", "ntar"):
            ranked = sorted(eng.ranked(cands, w, rep, rho, normalize))
            pa_rank = [tuple(x[2]) for x in ranked]
        else:
            pa_rank = []
        # ---- controller ----------------------------------------------------------
        pa1 = pa3 = None          # Bayes-optimal success probability of the policy-aware adversary
        if method == "static":
            p = tuple(nx.shortest_path(g, eng.src, eng.dst))
            pa1 = pa3 = 1.0
        elif method == "random":
            p = tuple(cands[rng.integers(len(cands))])
            pa1, pa3 = 1.0 / len(cands), min(3, len(cands)) / len(cands)
        elif method == "mtd":
            valid = [tuple(c) for c in cands]
            if current not in valid or (t - 1) % period == 0:
                pool = [c for c in valid if c != current] or valid
                current = pool[rng.integers(len(pool))]
                pa1, pa3 = 1.0 / len(pool), min(3, len(pool)) / len(pool)
            else:
                pa1 = pa3 = 1.0   # the adversary knows the period and the current path
            p = current
        else:
            ranked = eng.ranked(cands, w, rep, rho, normalize)
            cmin = min(x[0] for x in ranked)
            if tau > 0:
                near = [x[2] for x in ranked if x[0] <= cmin + tau + 1e-12]
                p = tuple(near[rng.integers(len(near))])
            else:
                p = tuple(next(x[2] for x in ranked if x[0] == cmin))
            if tau > 0:
                k = len(near)
                pa1, pa3 = 1.0 / k, min(3, k) / k
            else:
                pa1 = float(pa_rank[0] == p)
                pa3 = float(p in pa_rank[:3])
        ee, ll = path_cost_terms(p, eng.e, eng.l, eng.model)
        out.append(dict(t=t, thr=int(true_thr[t]), reported=int(reported[t]), path=p,
                        energy=ee, latency=ll,
                        f1=int(bool(f_rank) and f_rank[0] == p),
                        f3=int(p in f_rank[:3]),
                        m1=int(bool(m_rank) and m_rank[0] == p),
                        m3=int(p in m_rank[:3]),
                        pa1=pa1, pa3=pa3,
                        guessable=int(bool(hist))))
        if hist:
            trans[hist[-1]][p] += 1
        hist.append(p)
        rep[p] += 1
    return out


def metrics(rec, n_avail):
    thr = [r for r in rec if r["thr"]]
    nom = [r for r in rec if not r["thr"]]
    cnt = Counter(r["path"] for r in rec)
    tot = sum(cnt.values())
    H = -sum(c / tot * math.log2(c / tot) for c in cnt.values())
    m = lambda rows, k: float(np.mean([r[k] for r in rows])) if rows else float("nan")
    return dict(energy=m(rec, "energy"), energy_nom=m(nom, "energy"), energy_thr=m(thr, "energy"),
                latency=m(rec, "latency"), unique=len(cnt), h_avail=H / math.log2(n_avail),
                f1=m(thr, "f1"), f3=m(thr, "f3"), m1=m(thr, "m1"), m3=m(thr, "m3"),
                pa1=m(thr, "pa1"), pa3=m(thr, "pa3"))


# ---------------------------------------------------------------- topology sets
def six_node():
    g, _ = S.build_topology()
    return dict(g=g, src=0, dst=5, fail=3, n=6)


def random_topologies(sizes=(10, 15, 20), per=10):
    out = []
    for n in sizes:
        made, tseed = 0, 1000 * n
        while made < per:
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
            if not nx.has_path(h, a, b) or not S.candidates(h, a, b):
                continue
            if len(S.candidates(gg, a, b)) < 4:
                continue
            made += 1
            out.append(dict(g=gg, src=a, dst=b, fail=fn, n=n, seed=tseed))
    return out


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return m, 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return float(m), float(h)


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------- experiments
TAUS = (0.05, 0.10, 0.20)
TAU_MAIN = 0.10
METHODS = [("static", {}), ("random", {}), ("mtd", dict(period=1)), ("mtd", dict(period=3)),
           ("fixed", {}), ("ntar", {}), ("ntar", dict(tau=TAU_MAIN))]


def label(m, kw):
    if m == "mtd":
        return f"mtd_M{kw['period']}"
    if m == "ntar" and kw.get("tau"):
        return f"ntar_r{kw['tau']}"
    return m


def run_set(topos, T, on, L, seeds, methods=METHODS, model="hop", signal=None):
    """Average per topology over seeds; returns {label: [per-topology metric dicts]}"""
    res = defaultdict(list)
    for k, tp in enumerate(topos):
        eng = Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], model, cost_seed=k)
        true_thr, down = schedule(T, on, L)
        for m, kw in methods:
            per = []
            for s in seeds:
                rep_sig = None
                if signal:
                    rep_sig = corrupt_signal(true_thr, np.random.default_rng(777 + s), **signal)
                per.append(metrics(simulate(eng, m, T, true_thr, down, rep_sig, seed=s, **kw), eng.n_avail))
            res[label(m, kw)].append({key: float(np.mean([p[key] for p in per])) for key in per[0]})
    return res


def summarize_set(res):
    return {lab: {k: ci95([r[k] for r in rows]) for k in rows[0]} for lab, rows in res.items()}


def main(out):
    t0 = time.time()
    six = six_node()
    rand30 = random_topologies()
    R = {}

    # ---- E1/E2/E5: methods x adversaries, 6-node (20 seeds) and 30 topologies (5 seeds)
    for L in (5, 10, 20):
        T = 40 if L < 20 else 60
        on = 15 if L < 20 else 21
        r6 = run_set([six], T, on, L, range(20))
        R[f"adv6_L{L}"] = {lab: rows[0] for lab, rows in r6.items()}
        rr = run_set(rand30, T, on, L, range(5))
        R[f"adv30_L{L}"] = summarize_set(rr)
        if L == 10:
            # paired topology-level statistics, NTAR-R vs fixed and vs random/mtd
            pairs = {}
            for other in ("fixed", "random", "mtd_M1", "mtd_M3", "ntar"):
                for k in ("energy", "energy_nom", "f1", "m1", "pa1"):
                    d = [a[k] - b[k] for a, b in zip(rr[f"ntar_r{TAU_MAIN}"], rr[other])]
                    pairs[f"{other}:{k}"] = dict(mean_ci=ci95(d), p=wilcoxon_p(d))
            R["paired30_L10"] = pairs
            R["per_topology_L10"] = {lab: rows for lab, rows in rr.items()}
    # tau sweep (6-node and 30 topologies, L = 10)
    sweep = {}
    for tau in (0.0,) + TAUS:
        meth = [("ntar", dict(tau=tau) if tau else {})]
        r6 = run_set([six], 40, 15, 10, range(20), meth)
        rr = run_set(rand30, 40, 15, 10, range(5), meth)
        lab = list(r6)[0]
        sweep[str(tau)] = dict(six=r6[lab][0], rand30=summarize_set(rr)[lab])
    R["tau_sweep"] = sweep

    # ---- E3: imperfect threat signal (30 topologies, L = 10, 5 seeds)
    scen = {"perfect": {}, "fa_0.1": dict(p_fa=0.1), "fa_0.3": dict(p_fa=0.3),
            "miss_0.3": dict(p_miss=0.3), "miss_0.6": dict(p_miss=0.6),
            "delay_2": dict(delay=2), "delay_5": dict(delay=5), "flip_0.1": dict(p_flip=0.1)}
    meth = [("fixed", {}), ("ntar", {}), ("ntar", dict(tau=TAU_MAIN))]
    sig = {}
    for name, kw in scen.items():
        rr = run_set(rand30, 40, 15, 10, range(5), meth, signal=kw)
        sig[name] = summarize_set(rr)
    R["signal"] = sig

    # ---- E4: decoupled energy/latency (6-node with 20 link-cost draws, and 30 topologies)
    dec = {}
    for model in ("indep", "anti"):
        rows = defaultdict(list)
        topos6 = [six] * 20     # 20 independent link-cost draws on the 6-node graph
        r6 = run_set(topos6, 40, 15, 10, [0], meth + [("random", {})], model=model)
        rr = run_set(rand30, 40, 15, 10, range(3), meth + [("random", {})], model=model)
        dec[model] = dict(six=summarize_set(r6), rand30=summarize_set(rr))
        # independent effect of w1 vs w2 with w1+w2 fixed: does the selected path change?
        changed, tot = 0, 0
        e_sh, l_sh = [], []
        for k, tp in enumerate(rand30):
            eng = Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], model, cost_seed=k)
            cands = S.candidates(tp["g"], tp["src"], tp["dst"])
            pick = lambda w: min(eng.ranked(cands, w, {}))[2]
            pe, pl = pick((0.7, 0.1, 0.2)), pick((0.1, 0.7, 0.2))
            tot += 1
            changed += int(pe != pl)
            ee1, ll1 = path_cost_terms(pe, eng.e, eng.l, model)
            ee2, ll2 = path_cost_terms(pl, eng.e, eng.l, model)
            e_sh.append(ee1 - ee2)
            l_sh.append(ll1 - ll2)
        dec[model]["w1_vs_w2"] = dict(changed=changed, total=tot,
                                      d_energy=ci95(e_sh), d_latency=ci95(l_sh))
    R["decoupled"] = dec

    # ---- E8: counterexample for Proposition 1 (rho below the efficiency spread)
    ce = {}
    for rho in (0.5, 0.2, 0.1, 0.05, 0.02):
        n_diff = 0
        for tp in [six] + rand30:
            eng = Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], "hop")
            true_thr, down = schedule(25, 15, 5)
            a = simulate(eng, "fixed", 25, true_thr, down, rho=rho, normalize=False)
            b = simulate(eng, "ntar", 25, true_thr, down, rho=rho, normalize=False)
            n_diff += int(any(x["path"] != y["path"] for x, y in zip(a, b)))
        ce[str(rho)] = dict(topologies_where_adaptation_changes_routing=n_diff, of=1 + len(rand30))
    R["prop1_counterexample"] = ce

    # ---- E6: scalability of the candidate search
    sc = []
    for n in (20, 50, 100, 200):
        for deg in (3.6, 6.0):
            times, ncand, tyen = [], [], []
            made, seed = 0, 50_000 + n * 10 + int(deg)
            while made < 5:
                seed += 1
                gg = nx.erdos_renyi_graph(n, deg / (n - 1), seed=seed)
                if not nx.is_connected(gg):
                    continue
                d = dict(nx.all_pairs_shortest_path_length(gg)) if n <= 100 else None
                rng = np.random.default_rng(seed)
                nodes = list(gg)
                while True:
                    a, b = rng.choice(nodes, 2, replace=False)
                    if nx.shortest_path_length(gg, a, b) == 3:
                        break
                made += 1
                eng = Engine(gg, int(a), int(b), -1, "hop")
                rep = {}
                t1 = time.perf_counter()
                for _ in range(20):
                    cands = S.candidates(gg, int(a), int(b))
                    min(eng.ranked(cands, W_THR, rep))
                times.append((time.perf_counter() - t1) / 20 * 1e3)
                ncand.append(len(cands))
                t1 = time.perf_counter()
                for _ in range(20):
                    from itertools import islice
                    ks = list(islice(nx.shortest_simple_paths(gg, int(a), int(b)), 16))
                    min(eng.ranked(ks, W_THR, rep))
                tyen.append((time.perf_counter() - t1) / 20 * 1e3)
            sc.append(dict(n=n, degree=deg, candidates=ci95(ncand), ms_enum=ci95(times),
                           ms_yen16=ci95(tyen)))
    R["scalability"] = sc

    R["meta"] = dict(tau_main=TAU_MAIN, runtime_s=time.time() - t0,
                     note="adv*: hit rates over threat-window steps; f=frequency, m=Markov, pa=policy-aware; 1=top-1, 3=top-3")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results2.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
