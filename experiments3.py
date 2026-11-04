"""
Third-round experiment: does truncating the candidate set to the k shortest paths
(Yen's algorithm, as recommended for deployment) change the results?

Runs the 10-step-threat-window protocol of experiments2.py on the 30 random
topologies with four candidate generators:
  full_enum : all simple paths of <= 4 hops, NetworkX DFS order (main results)
  full_yen  : the same set, in Yen order (isolates the tie-breaking order)
  yen16     : the 16 shortest simple paths of <= 4 hops (Yen)
  yen8      : the 8 shortest simple paths of <= 4 hops (Yen)

Run:  python experiments3.py --out results   ->  results/results3.json
"""
import argparse
import json
import os
from itertools import islice

import networkx as nx

import experiments2 as X
import ntar_sim as S

ORIG_CANDIDATES = S.candidates


def yen_candidates(k):
    def cands(graph, s, d):
        if s not in graph or d not in graph or not nx.has_path(graph, s, d):
            return []
        out = []
        for p in nx.shortest_simple_paths(graph, s, d):   # Yen's algorithm, hop count
            if len(p) > S.MAX_NODES:
                break                                      # paths come in nondecreasing length
            out.append(p)
            if k and len(out) >= k:
                break
        return out
    return cands


METHODS = [("random", {}), ("mtd", dict(period=1)), ("fixed", {}), ("ntar", {}),
           ("ntar", dict(tau=X.TAU_MAIN))]


def main(out):
    topos = X.random_topologies()                     # built with the original generator
    sizes = [len(ORIG_CANDIDATES(t["g"], t["src"], t["dst"])) for t in topos]
    res = dict(candidate_set_sizes=sizes)
    for name, gen in (("full_enum", ORIG_CANDIDATES), ("full_yen", yen_candidates(None)),
                      ("yen16", yen_candidates(16)), ("yen8", yen_candidates(8))):
        S.candidates = gen
        try:
            r = X.run_set(topos, 40, 15, 10, range(5), METHODS)
        finally:
            S.candidates = ORIG_CANDIDATES
        res[name] = X.summarize_set(r)
        res[name + "_truncated_topologies"] = sum(
            len(gen(t["g"], t["src"], t["dst"])) < n for t, n in zip(topos, sizes))
        print(name, "done")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results3.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    main(ap.parse_args().out)
