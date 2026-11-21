"""
Additional baselines for the NTAR manuscript (Table 11) and route churn.

Same protocol as Table 5: the 30 random topologies of experiments2.random_topologies(),
40 steps, threat reported during steps 15-24, node failure at steps 8-11, five routing
seeds per topology, hop-cost model. Adds three baselines:

  pareto   Pareto-random. At each step the candidates are scored on three objectives to be
           minimised: energy E(p), latency L(p) and the number of previous uses rep(p).
           The controller picks uniformly among the non-dominated (Pareto-optimal) paths.
           It ignores the threat signal.
  tgmtd    Threat-gated MTD (adaptive MTD). Outside a reported threat it routes like
           Fixed-Weight (deterministic minimiser of the nominal cost); during a reported
           threat it switches to a uniformly random different path at every step.
  qsoft    The same trained Q-table with a Boltzmann (softmax) policy whose temperature is
           calibrated by bisection so that its mean energy matches NTAR-R on these topologies.
  qlearn   Tabular Q-learning. State = (reported threat flag, previous path); action =
           candidate path; reward = -C_norm(p) with the weight profile of the current
           threat state, i.e. the same objective NTAR minimises. The Q-table is trained
           for N_TRAIN episodes per topology (threat window placed at a random step in
           each episode, epsilon decaying from 0.5 to 0.05) and then evaluated with
           epsilon = 0.05 and learning switched off.

It also reports, for every method:
  churn      fraction of steps (t >= 2) on which the route differs from the previous
             step, i.e. how often the controller must reinstall flow rules;
  flowmods   mean number of switches whose flow entry must be (re)written per step,
             counted as the nodes of the new path that are not on the same path segment
             as before (0 when the route is unchanged);
  pareto_ok  for the deterministic NTAR/Fixed-Weight controllers, the fraction of
             decisions that lie on the Pareto set of (E, L, rep).

The simulate() below reproduces experiments2.simulate() for the existing methods; this is
checked at start-up (assert_consistency) so the new rows are comparable with Table 5.

Run:  python experiments4.py --out results      (writes results/results4.json)
"""
import argparse
import warnings
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import networkx as nx

import ntar_sim as S
import experiments2 as E

T, ON, L = 40, 15, 10
SEEDS = range(5)
TAU = E.TAU_MAIN
N_TRAIN = 4000                # Q-learning training episodes per topology
N_TRAIN_SENS = (100, 400, 1500)   # shorter training, reported as a sensitivity check
ALPHA, GAMMA = 0.1, 0.9
EPS_START, EPS_END, EPS_EVAL = 0.5, 0.05, 0.05


# ---------------------------------------------------------------- helpers
def pareto_set(paths, eng, rep):
    """Indices of non-dominated paths for the objectives (E, L, rep), all minimised."""
    obj = []
    for p in paths:
        ee, ll = E.path_cost_terms(p, eng.e, eng.l, eng.model)
        obj.append((ee, ll, rep.get(tuple(p), 0)))
    keep = []
    for i, a in enumerate(obj):
        dominated = any(all(b[k] <= a[k] for k in range(3)) and any(b[k] < a[k] for k in range(3))
                        for j, b in enumerate(obj) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def flowmods(prev, new):
    """Switches whose forwarding entry changes when the route moves from prev to new:
    every node of new (except the destination) whose next hop differs from prev."""
    if prev is None or prev == new:
        return 0
    nh_prev = {prev[i]: prev[i + 1] for i in range(len(prev) - 1)}
    return sum(1 for i in range(len(new) - 1) if nh_prev.get(new[i]) != new[i + 1])


class GraphStates:
    """Candidate sets for the intact and the failed graph (the only two states that occur)."""
    def __init__(self, eng):
        self.eng = eng
        self.intact = [tuple(p) for p in S.candidates(eng.g0, eng.src, eng.dst)]
        h = eng.g0.copy()
        h.remove_node(eng.fail_node)
        self.failed = [tuple(p) for p in S.candidates(h, eng.src, eng.dst)]
        # after restoration experiments2.simulate re-adds the node and its edges to the
        # mutated graph, which changes the adjacency order and hence the DFS order
        r = h.copy()
        r.add_node(eng.fail_node)
        for a, b in eng.g0.edges:
            if eng.fail_node in (a, b) and a in r and b in r:
                r.add_edge(a, b)
        self.restored = [tuple(p) for p in S.candidates(r, eng.src, eng.dst)]
        self.sp_intact = tuple(nx.shortest_path(eng.g0, eng.src, eng.dst))
        self.sp_failed = tuple(nx.shortest_path(h, eng.src, eng.dst))

    def cands(self, is_down, after_failure=False):
        if is_down:
            return self.failed
        return self.restored if after_failure else self.intact


# ---------------------------------------------------------------- Q-learning
class QAgent:
    def __init__(self):
        self.q = defaultdict(float)

    def key(self, thr, prev):
        return (int(thr), prev)

    def best(self, s, cands):
        vals = [self.q[(s, a)] for a in cands]
        m = max(vals)
        return cands[vals.index(m)]          # first maximum in enumeration order

    def act(self, s, cands, eps, rng):
        if rng.random() < eps:
            return cands[rng.integers(len(cands))]
        return self.best(s, cands)


def reward(eng, p, thr, rep):
    w = E.W_THR if thr else E.W_NOM
    return -eng.cost(list(p), w, rep)


def train_q(eng, gs, seed):
    rng = np.random.default_rng(50_000 + seed)
    agent = QAgent()
    for ep in range(N_TRAIN):
        eps = EPS_START + (EPS_END - EPS_START) * ep / max(1, N_TRAIN - 1)
        on = int(rng.integers(2, T - L))                 # threat window at a random position
        true_thr, down = E.schedule(T, on, L)
        rep = Counter()
        prev = None
        s = agent.key(0, None)
        for t in range(1, T + 1):
            cands = gs.cands(down[t], t >= 12)
            s = agent.key(true_thr[t], prev if prev in cands else None)
            a = agent.act(s, cands, eps, rng)
            r = reward(eng, a, true_thr[t], rep)
            rep[a] += 1
            if t < T:
                nc = gs.cands(down[t + 1], t + 1 >= 12)
                s2 = agent.key(true_thr[t + 1], a if a in nc else None)
                target = r + GAMMA * max(agent.q[(s2, b)] for b in nc)
            else:
                target = r
            agent.q[(s, a)] += ALPHA * (target - agent.q[(s, a)])
            prev = a
    return agent


# ---------------------------------------------------------------- simulation
def simulate(eng, gs, method, true_thr, down, seed=0, agent=None, temp=None):
    """Same adversary bookkeeping as experiments2.simulate (perfect threat signal)."""
    rng = np.random.default_rng(seed)
    rep = Counter()
    hist = []
    trans = defaultdict(Counter)
    current = None
    prev = None
    out = []
    for t in range(1, T + 1):
        cands = list(gs.cands(down[t], t >= 12))
        thr = int(true_thr[t])
        w = E.W_THR if thr else E.W_NOM
        if method in ("fixed",):
            w = E.W_NOM
        freq = Counter(hist)
        f_rank = sorted(freq, key=lambda p: (-freq[p], hist.index(p)))
        if hist and trans[hist[-1]]:
            nxt = trans[hist[-1]]
            m_rank = sorted(nxt, key=lambda p: (-nxt[p], hist.index(p)))
            m_rank += [p for p in f_rank if p not in m_rank]
        else:
            m_rank = f_rank
        pareto_ok = None
        if method == "static":
            p = gs.sp_failed if down[t] else gs.sp_intact
            pa1 = 1.0
        elif method == "random":
            p = cands[rng.integers(len(cands))]
            pa1 = 1.0 / len(cands)
        elif method == "mtd":
            if current not in cands or (t - 1) % 1 == 0:
                pool = [c for c in cands if c != current] or cands
                current = pool[rng.integers(len(pool))]
                pa1 = 1.0 / len(pool)
            p = current
        elif method in ("fixed", "ntar", "ntar_r"):
            ranked = eng.ranked([list(c) for c in cands], w, rep)
            cmin = min(x[0] for x in ranked)
            if method == "ntar_r":
                near = [tuple(x[2]) for x in ranked if x[0] <= cmin + TAU + 1e-12]
                p = near[rng.integers(len(near))]
                pa1 = 1.0 / len(near)
            else:
                p = tuple(next(x[2] for x in ranked if x[0] == cmin))
                pa1 = 1.0                                   # deterministic and policy known
                pareto_ok = int(cands.index(p) in pareto_set(cands, eng, rep))
        elif method == "pareto":
            ps = pareto_set(cands, eng, rep)
            p = cands[ps[rng.integers(len(ps))]]
            pa1 = 1.0 / len(ps)
        elif method == "tgmtd":
            if thr:
                pool = [c for c in cands if c != prev] or cands
                p = pool[rng.integers(len(pool))]
                pa1 = 1.0 / len(pool)
            else:
                ranked = eng.ranked([list(c) for c in cands], E.W_NOM, rep)
                cmin = min(x[0] for x in ranked)
                p = tuple(next(x[2] for x in ranked if x[0] == cmin))
                pa1 = 1.0
        elif method == "qlearn":
            s = agent.key(thr, prev if prev in cands else None)
            g = agent.best(s, cands)
            p = agent.act(s, cands, EPS_EVAL, rng)
            # Bayes-optimal guess of an adversary that knows the Q-table and epsilon
            pa1 = (1 - EPS_EVAL) + EPS_EVAL / len(cands)
        elif method == "qsoft":
            # Boltzmann (softmax) policy over the learned Q-values
            s = agent.key(thr, prev if prev in cands else None)
            qv = np.array([agent.q[(s, a)] for a in cands])
            z = np.exp((qv - qv.max()) / temp)
            pr = z / z.sum()
            p = cands[rng.choice(len(cands), p=pr)]
            pa1 = float(pr.max())                         # adversary knows Q-table and temperature
        else:
            raise ValueError(method)
        ee, ll = E.path_cost_terms(p, eng.e, eng.l, eng.model)
        out.append(dict(t=t, thr=thr, path=p, energy=ee, latency=ll,
                        f1=int(bool(f_rank) and f_rank[0] == p),
                        m1=int(bool(m_rank) and m_rank[0] == p),
                        pa1=pa1, changed=None if prev is None else int(p != prev),
                        fm=flowmods(prev, p), pareto_ok=pareto_ok))
        if hist:
            trans[hist[-1]][p] += 1
        hist.append(p)
        rep[p] += 1
        prev = p
    return out


def metrics(rec):
    thr = [r for r in rec if r["thr"]]
    nom = [r for r in rec if not r["thr"]]
    m = lambda rows, k: float(np.mean([r[k] for r in rows if r[k] is not None])) \
        if any(r[k] is not None for r in rows) else float("nan")
    return dict(energy=m(rec, "energy"), energy_nom=m(nom, "energy"), energy_thr=m(thr, "energy"),
                f1=m(thr, "f1"), m1=m(thr, "m1"), pa1=m(thr, "pa1"),
                churn=m(rec, "changed"), churn_nom=m(nom, "changed"), churn_thr=m(thr, "changed"),
                flowmods=m(rec, "fm"), pareto_ok=m(rec, "pareto_ok"))


METHODS = ["fixed", "random", "mtd", "ntar", "ntar_r", "pareto", "tgmtd", "qlearn"]
LABELS = {"mtd": "mtd_M1", "ntar_r": f"ntar_r{TAU}"}


def assert_consistency(topos):
    """The existing methods must give the same per-step paths as experiments2.simulate."""
    old = {"fixed": ("fixed", {}), "random": ("random", {}), "mtd": ("mtd", dict(period=1)),
           "ntar": ("ntar", {}), "ntar_r": ("ntar", dict(tau=TAU))}
    for k, tp in enumerate(topos):
        eng = E.Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], "hop", cost_seed=k)
        gs = GraphStates(eng)
        true_thr, down = E.schedule(T, ON, L)
        for m, (om, kw) in old.items():
            for s in SEEDS:
                a = [r["path"] for r in simulate(eng, gs, m, true_thr, down, seed=s)]
                b = [tuple(r["path"]) for r in E.simulate(eng, om, T, true_thr, down, seed=s, **kw)]
                assert a == b, (k, m, s)


def main(out):
    t0 = time.time()
    topos = E.random_topologies()
    assert_consistency(topos)
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    per = defaultdict(list)
    qtrain = []
    agents = []
    for k, tp in enumerate(topos):
        eng = E.Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], "hop", cost_seed=k)
        gs = GraphStates(eng)
        true_thr, down = E.schedule(T, ON, L)
        agent = train_q(eng, gs, seed=k)
        agents.append((eng, gs, true_thr, down, agent))
        qtrain.append(len(agent.q))
        for m in METHODS:
            rows = [metrics(simulate(eng, gs, m, true_thr, down, seed=s, agent=agent)) for s in SEEDS]
            per[LABELS.get(m, m)].append({key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]})
    # share of Fixed-Weight route changes that are between paths of equal length
    same = total = 0
    for eng, gs, true_thr, down, agent in agents:
        for s in SEEDS:
            rec = simulate(eng, gs, "fixed", true_thr, down, seed=s)
            for x, y in zip(rec, rec[1:]):
                if x["path"] != y["path"]:
                    total += 1
                    same += int(len(x["path"]) == len(y["path"]))
    fixed_equal_len = same / total
    # Boltzmann Q-learning, temperature calibrated to NTAR-R's mean energy
    target = float(np.mean([r["energy"] for r in per[f"ntar_r{TAU}"]]))

    def soft_rows(temp):
        out_rows = []
        for eng, gs, true_thr, down, agent in agents:
            rows = [metrics(simulate(eng, gs, "qsoft", true_thr, down, seed=s, agent=agent, temp=temp))
                    for s in SEEDS]
            out_rows.append({key: float(np.nanmean([r[key] for r in rows])) for key in rows[0]})
        return out_rows

    lo, hi = -4.0, 1.0                                      # log10 temperature bracket
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        e = float(np.mean([r["energy"] for r in soft_rows(10 ** mid)]))
        if e < target:
            lo = mid
        else:
            hi = mid
    temp_star = 10 ** (0.5 * (lo + hi))
    per["qsoft"] = soft_rows(temp_star)
    temp_sweep = {}
    for tt in (0.01, 0.03, 0.1, 0.3):
        rr = soft_rows(tt)
        temp_sweep[str(tt)] = {key: E.ci95([r[key] for r in rr]) for key in ("energy", "m1", "pa1")}
    # Q-learning sensitivity to the training length
    global N_TRAIN
    sens = {}
    n_main = N_TRAIN
    for n in N_TRAIN_SENS:
        N_TRAIN = n
        rows_n = []
        for k, tp in enumerate(topos):
            eng = E.Engine(tp["g"], tp["src"], tp["dst"], tp["fail"], "hop", cost_seed=k)
            gs = GraphStates(eng)
            true_thr, down = E.schedule(T, ON, L)
            agent = train_q(eng, gs, seed=k)
            rows = [metrics(simulate(eng, gs, "qlearn", true_thr, down, seed=s, agent=agent)) for s in SEEDS]
            rows_n.append({key: float(np.mean([r[key] for r in rows])) for key in ("energy", "m1", "pa1")})
        sens[str(n)] = {key: E.ci95([r[key] for r in rows_n]) for key in ("energy", "m1", "pa1")}
    N_TRAIN = n_main
    sens[str(n_main)] = {key: E.ci95([r[key] for r in per["qlearn"]]) for key in ("energy", "m1", "pa1")}
    summary = {lab: {k: E.ci95([r[k] for r in rows]) for k in rows[0]
                     if not all(np.isnan(r[k]) for r in rows)} for lab, rows in per.items()}
    ref = f"ntar_r{TAU}"
    paired = {}
    for other in ("pareto", "tgmtd", "qlearn", "qsoft", "fixed", "random", "mtd_M1"):
        for key in ("energy", "m1", "pa1", "churn"):
            d = [a[key] - b[key] for a, b in zip(per[ref], per[other])]
            paired[f"{other}:{key}"] = dict(mean_ci=E.ci95(d), p=E.wilcoxon_p(d))
    R = dict(summary=summary, paired_vs_ntar_r=paired, qlearn_training_sensitivity=sens,
             qsoft=dict(temperature=temp_star, energy_target=target, temperature_sweep=temp_sweep,
                        calibration="bisection on log10 temperature, same 30 topologies"),
             fixed_route_changes_between_equal_length_paths=fixed_equal_len,
             per_topology=per,
             meta=dict(T=T, threat_on=ON, threat_len=L, seeds=list(SEEDS), tau=TAU,
                       n_train=N_TRAIN, alpha=ALPHA, gamma=GAMMA, eps_start=EPS_START,
                       eps_end=EPS_END, eps_eval=EPS_EVAL, q_states_actions_mean=float(np.mean(qtrain)),
                       consistency_check="existing methods reproduce experiments2.simulate paths",
                       runtime_s=time.time() - t0))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results4.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
