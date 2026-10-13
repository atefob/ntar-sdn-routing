import random
from collections import Counter
import numpy as np
import networkx as nx

# ----------------------------------------------------------------------
# Environment (identical logic to code1.docx, minus the no-op GNN calls)
# ----------------------------------------------------------------------

class SimulatedSDNNetwork:
    def __init__(self, num_nodes=6, topology_seed=42):
        self.num_nodes = num_nodes
        self.graph = nx.erdos_renyi_graph(n=num_nodes, p=0.6, seed=topology_seed)
        while not nx.is_connected(self.graph):
            topology_seed += 1
            self.graph = nx.erdos_renyi_graph(n=num_nodes, p=0.6, seed=topology_seed)
        self.original_graph = self.graph.copy()
        self.path_history = []
        self.reset_environment()

    def reset_environment(self):
        self.node_energy = np.random.uniform(0.7, 1.0, size=self.num_nodes)
        self.path_history = []
        return self.graph, self.node_energy

    def simulate_node_failure(self, node_id):
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            self.node_energy[node_id] = 0.0

    def restore_node(self, node_id):
        if node_id not in self.graph.nodes:
            self.graph.add_node(node_id)
            for u, v in self.original_graph.edges:
                if u == node_id or v == node_id:
                    if u in self.graph.nodes and v in self.graph.nodes:
                        self.graph.add_edge(u, v)
            self.node_energy[node_id] = 0.8

    def get_network_metrics(self, path):
        hops = len(path) - 1
        energy_spent = hops * 0.05 + np.random.uniform(0.01, 0.03)
        latency = hops * 10.0 + np.random.uniform(1.0, 5.0)

        self.path_history.append(tuple(path))
        path_counts = Counter(self.path_history)
        total_paths = len(self.path_history)

        entropy = 0.0
        for count in path_counts.values():
            p = count / total_paths
            if p > 0:
                entropy -= p * np.log2(p + 1e-6)
        max_entropy = np.log2(len(path_counts)) if len(path_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        return energy_spent, latency, normalized_entropy

    def consume_energy(self, path):
        for node in path:
            if node < len(self.node_energy):
                self.node_energy[node] = max(0.1, self.node_energy[node] - 0.02)


# ----------------------------------------------------------------------
# True threat-aware weights, shared across ALL methods for reward scoring
# (this is the paper's actual objective function -- used to fairly judge
# every routing strategy against the same utility, regardless of how each
# strategy picks its path)
# ----------------------------------------------------------------------

def true_weights(threat_level):
    if threat_level == 1:
        return 0.2, 0.2, 0.6
    return 0.4, 0.4, 0.2


def calculate_reward(w1, w2, w3, energy, latency, entropy):
    return (w1 * (-energy)) + (w2 * (-latency / 100.0)) + (w3 * entropy)


# ----------------------------------------------------------------------
# Method 1 (PROPOSED): threat-adaptive MADRL cost-based selection
# ----------------------------------------------------------------------

MAX_HOPS = 5
E_MAX = MAX_HOPS * 0.05          # max possible energy term
L_MAX = MAX_HOPS * 10.0 / 100.0  # max possible normalized latency term


def select_proposed(graph, source, target, path_history, w1, w2, w3, mode="original"):
    """
    mode="original"  : exact cost function from code1.docx (repetition_penalty
                        unweighted, entropy_bonus on raw 0-1 scale vs raw
                        energy/latency on a much smaller scale -> w3 barely
                        changes which path wins).
    mode="normalized": CORRECTED cost function. Energy and latency are
                        normalized to [0,1] against their max possible value
                        (5-hop path) so all three terms (energy, latency,
                        diversity) live on the same [0,1] scale. Now w1/w2/w3
                        genuinely control the trade-off, since they're
                        weighting comparable quantities instead of a large
                        entropy term against a much smaller energy term.
                        The separate unweighted repetition_penalty is
                        removed -- diversity preference is carried entirely
                        by the (now properly weighted) entropy_bonus term.
    """
    try:
        all_paths = [p for p in nx.all_simple_paths(graph, source, target) if len(p) <= MAX_HOPS]
    except nx.NetworkXNoPath:
        return None
    if not all_paths:
        return None

    best_path, min_cost = None, float("inf")
    for path in all_paths:
        hops = len(path) - 1
        energy = hops * 0.05
        latency = hops * 10.0
        path_tuple = tuple(path)
        rep_count = path_history.count(path_tuple)
        entropy_bonus = 1.0 / (rep_count + 1)

        if mode == "normalized":
            norm_energy = energy / E_MAX
            norm_latency = (latency / 100.0) / L_MAX
            cost = (w1 * norm_energy) + (w2 * norm_latency) - (w3 * entropy_bonus)
        else:
            repetition_penalty = rep_count * 0.5
            cost = (w1 * energy) + (w2 * (latency / 100.0)) - (w3 * entropy_bonus) + repetition_penalty

        if cost < min_cost:
            min_cost, best_path = cost, path
    return best_path


# ----------------------------------------------------------------------
# Baseline A: Static shortest-path routing (Dijkstra, hop-count weight)
# ----------------------------------------------------------------------

def select_static_shortest(graph, source, target, path_history):
    try:
        return nx.shortest_path(graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ----------------------------------------------------------------------
# Baseline B: Naive random path selection (bounded by the same <=5 hop cap)
# ----------------------------------------------------------------------

def select_random(graph, source, target, path_history):
    try:
        all_paths = [p for p in nx.all_simple_paths(graph, source, target) if len(p) <= 5]
    except nx.NetworkXNoPath:
        return None
    if not all_paths:
        return None
    return random.choice(all_paths)


# ----------------------------------------------------------------------
# Baseline C: Fixed-weight MADRL (same cost function, weights NEVER switch
# on threat alert -- isolates the value of threat-adaptive weighting)
# ----------------------------------------------------------------------

def select_fixed_weight(graph, source, target, path_history, mode="normalized"):
    w1, w2, w3 = 0.4, 0.4, 0.2  # frozen, ignores threat_level entirely
    return select_proposed(graph, source, target, path_history, w1, w2, w3, mode=mode)


# ----------------------------------------------------------------------
# Simulation driver -- identical event schedule for every method/run
# ----------------------------------------------------------------------

EVENTS = {8: ("fail", 3), 12: ("restore", 3), 15: ("threat", 1), 20: ("threat", 0)}
TOTAL_STEPS = 25
SOURCE, TARGET = 0, 5


def run_single(method_name, run_seed, topology_seed=42, num_nodes=6):
    random.seed(run_seed)
    np.random.seed(run_seed)

    env = SimulatedSDNNetwork(num_nodes=num_nodes, topology_seed=topology_seed)
    threat_level = 0

    energies, latencies, entropies, rewards = [], [], [], []

    for step in range(1, TOTAL_STEPS + 1):
        if step in EVENTS:
            kind, val = EVENTS[step]
            if kind == "fail":
                env.simulate_node_failure(val)
            elif kind == "restore":
                env.restore_node(val)
            elif kind == "threat":
                threat_level = val

        w1, w2, w3 = true_weights(threat_level)

        if method_name == "Proposed (Adaptive MADRL) - original":
            path = select_proposed(env.graph, SOURCE, TARGET, env.path_history, w1, w2, w3, mode="original")
        elif method_name == "Proposed (Adaptive MADRL) - corrected":
            path = select_proposed(env.graph, SOURCE, TARGET, env.path_history, w1, w2, w3, mode="normalized")
        elif method_name == "Static Shortest-Path":
            path = select_static_shortest(env.graph, SOURCE, TARGET, env.path_history)
        elif method_name == "Random Path Selection":
            path = select_random(env.graph, SOURCE, TARGET, env.path_history)
        elif method_name == "Fixed-Weight MADRL - corrected":
            path = select_fixed_weight(env.graph, SOURCE, TARGET, env.path_history, mode="normalized")
        else:
            raise ValueError(method_name)

        if path is None:
            continue  # no route available this step (e.g. mid-failure disconnection)

        energy, latency, entropy = env.get_network_metrics(path)
        reward = calculate_reward(w1, w2, w3, energy, latency, entropy)

        energies.append(energy)
        latencies.append(latency)
        entropies.append(entropy)
        rewards.append(reward)

        env.consume_energy(path)

    unique_paths = len(set(env.path_history))
    return {
        "energy": np.mean(energies),
        "latency": np.mean(latencies),
        "entropy": np.mean(entropies),
        "reward": np.mean(rewards),
        "unique_paths": unique_paths,
    }


def run_experiment(n_runs=20):
    methods = [
        "Proposed (Adaptive MADRL) - original",
        "Proposed (Adaptive MADRL) - corrected",
        "Static Shortest-Path",
        "Random Path Selection",
        "Fixed-Weight MADRL - corrected",
    ]

    results = {m: {"energy": [], "latency": [], "entropy": [], "reward": [], "unique_paths": []} for m in methods}

    for run in range(n_runs):
        for m in methods:
            r = run_single(m, run_seed=run, topology_seed=42)
            for k in r:
                results[m][k].append(r[k])

    return results


def print_table(results, n_runs):
    metrics = ["energy", "latency", "entropy", "reward", "unique_paths"]
    labels = {
        "energy": "Energy (J)",
        "latency": "Latency (ms)",
        "entropy": "Norm. Entropy",
        "reward": "Reward",
        "unique_paths": "Unique Paths",
    }

    col_w = 26
    header = f"{'Method':<28}" + "".join(f"{labels[m]:^{col_w}}" for m in metrics)
    print(f"\n=== Baseline Comparison (mean ± std over {n_runs} runs, {TOTAL_STEPS} steps) ===\n")
    print(header)
    print("-" * len(header))
    for method, vals in results.items():
        row = f"{method:<28}"
        for m in metrics:
            arr = np.array(vals[m])
            mean, std = arr.mean(), arr.std()
            if m == "unique_paths":
                row += f"{f'{mean:.1f} ± {std:.1f}':^{col_w}}"
            else:
                row += f"{f'{mean:.4f} ± {std:.4f}':^{col_w}}"
        print(row)
    print()


if __name__ == "__main__":
    N_RUNS = 20
    results = run_experiment(n_runs=N_RUNS)
    print_table(results, N_RUNS)

    # Save as CSV too
    import csv
    with open("baseline_comparison_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Energy_mean", "Energy_std", "Latency_mean", "Latency_std",
                          "Entropy_mean", "Entropy_std", "Reward_mean", "Reward_std",
                          "UniquePaths_mean", "UniquePaths_std"])
        for method, vals in results.items():
            e = np.array(vals["energy"]); l = np.array(vals["latency"])
            h = np.array(vals["entropy"]); r = np.array(vals["reward"]); u = np.array(vals["unique_paths"])
            writer.writerow([method, e.mean(), e.std(), l.mean(), l.std(),
                              h.mean(), h.std(), r.mean(), r.std(), u.mean(), u.std()])
    print("Saved: baseline_comparison_results.csv")