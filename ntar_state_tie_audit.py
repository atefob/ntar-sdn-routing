r"""Non-invasive audit of NTAR energy state and deterministic tie-breaking.

Place next to ntar_sim.py and experiments2.py, then run:
    .\.venv\Scripts\python.exe ntar_state_tie_audit.py

Outputs results/ntar_state_tie_audit.json. Existing simulation, figures,
and Tables 1-9 are not modified.
"""

import inspect
import json
import math
from pathlib import Path

import numpy as np

import ntar_sim as S
import experiments2 as E


OUTPUT = Path('results/ntar_state_tie_audit.json')
TOL = 1e-12  # diagnostic near-tie tolerance, NOT a changed routing rule


def route_from_trace(entry):
    return tuple(int(x) for x in entry['path'].split('-'))


def replay_ties(graph, method, trace):
    """Replay the actual selected route history with the original cost code."""
    np.random.seed(0)
    env = S.Env(graph, graph.number_of_nodes())
    threat = 0
    records = []
    for row in trace:
        t = int(row['step'])
        if t == 8:
            env.fail(S.FAIL_NODE)
        elif t == 12:
            env.restore(S.FAIL_NODE)
        elif t == 15:
            threat = 1
        elif t == 20:
            threat = 0
        weights = S.W_NOMINAL if method == 'fixed' or not threat else S.W_THREAT
        candidates = S.candidates(env.graph, S.SRC, S.DST)
        costs = [float(S.cost(p, weights, env.history, 'eq5')) for p in candidates]
        minimum = min(costs)
        first_min = tuple(candidates[costs.index(minimum)])
        picked = route_from_trace(row)
        exact = [tuple(p) for p, c in zip(candidates, costs) if c == minimum]
        near = [tuple(p) for p, c in zip(candidates, costs) if math.isclose(c, minimum, abs_tol=TOL, rel_tol=0)]
        picked_cost = float(S.cost(picked, weights, env.history, 'eq5'))
        records.append({
            'step': t,
            'selected': list(picked),
            'selected_cost': picked_cost,
            'min_cost': minimum,
            'first_exact_minimum': list(first_min),
            'matches_exact_first_minimum': picked == first_min,
            'number_exact_minima': len(exact),
            'exact_minima': [list(p) for p in exact],
            'number_near_minima': len(near),
            'near_minima': [list(p) for p in near],
            'selected_is_near_minimum': math.isclose(picked_cost, minimum, abs_tol=TOL, rel_tol=0),
            'selected_hops': len(picked) - 1,
        })
        env.history.append(picked)
        env.consume(picked)  # the original simulated energy-state update
    return records


def run_audit():
    graph, graph_seed = S.build_topology()
    traces = {method: S.run(graph, method, seed=0) for method in ('fixed', 'ntar')}
    tie_records = {method: replay_ties(graph, method, traces[method]) for method in traces}
    mismatches = [
        {'step': a['step'], 'ntar': a['path'], 'fixed': b['path']}
        for a, b in zip(traces['ntar'], traces['fixed'])
        if a['path'] != b['path']
    ]

    # Check the main simulated residual-energy update, without modifying inputs.
    np.random.seed(900)
    env = S.Env(graph, graph.number_of_nodes())
    sample_path = tuple(S.candidates(graph, S.SRC, S.DST)[0])
    before = env.energy.copy()
    env.consume(sample_path)
    after = env.energy.copy()
    energy_update_correct = all(
        np.isclose(float(after[v]), max(0.1, float(before[v]) - 0.02))
        for v in sample_path
    ) and all(np.isclose(float(after[v]), float(before[v]))
              for v in graph if v not in sample_path)
    # No residual-energy state is passed to S.cost; check effect with same cost input.
    original_cost = float(S.cost(sample_path, S.W_NOMINAL, [], 'eq5'))
    cost_after_consumption = float(S.cost(sample_path, S.W_NOMINAL, [], 'eq5'))

    # Independent cross-implementation comparison of original 25-step protocol.
    engine = E.Engine(graph, S.SRC, S.DST, S.FAIL_NODE, model='hop', cost_seed=0)
    true_thr, down = E.schedule(25, 15, 5)
    crosschecks = {}
    for method in ('fixed', 'ntar'):
        extended = E.simulate(engine, method, 25, true_thr, down, seed=0)
        different_steps = [
            int(orig['step']) for orig, ext in zip(traces[method], extended)
            if route_from_trace(orig) != tuple(ext['path'])
        ]
        crosschecks[method] = {
            'steps_compared': min(len(traces[method]), len(extended)),
            'different_steps': different_steps,
        }

    code_checks = {
        'main_has_residual_energy_update': '.consume(' in inspect.getsource(S.run),
        'extended_simulate_calls_env_consume': '.consume(' in inspect.getsource(E.simulate),
        'extended_engine_cost_uses_residual_energy_state': 'residual' in inspect.getsource(E.Engine.cost).lower()
            or 'self.energy' in inspect.getsource(E.Engine.cost),
        'original_cost_accepts_no_residual_energy_argument': list(inspect.signature(S.cost).parameters) ==
            ['path', 'w', 'history', 'form'],
    }
    result = {
        'purpose': 'Audit only. Does not change original code, algorithm, or results.',
        'graph_seed': graph_seed,
        'near_tie_absolute_tolerance': TOL,
        'main_energy_update_correct_on_example_path': bool(energy_update_correct),
        'example_energy_before': [float(before[v]) for v in sample_path],
        'example_energy_after': [float(after[v]) for v in sample_path],
        'example_cost_before_state_update': original_cost,
        'example_cost_after_state_update': cost_after_consumption,
        'code_checks': code_checks,
        'original_vs_extended_path_comparison': crosschecks,
        'ntar_vs_fixed_different_decision_steps': mismatches,
        'steps_8_13_20': {
            method: [r for r in records if r['step'] in (8, 13, 20)]
            for method, records in tie_records.items()
        },
        'all_step_tie_details': tie_records,
    }

    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('NTAR audit | no source files or existing results changed')
    print(f'Original Env.consume update works on sample path: {energy_update_correct}')
    print(f'Original cost changes after only a residual-energy update: {original_cost != cost_after_consumption}')
    print(f'Extended simulator calls Env.consume: {code_checks["extended_simulate_calls_env_consume"]}')
    print(f'Extended Engine.cost reads residual energy: {code_checks["extended_engine_cost_uses_residual_energy_state"]}')
    for method in ('fixed', 'ntar'):
        print(f'{method}: original vs extended, differing steps = {crosschecks[method]["different_steps"]}')
        for row in result['steps_8_13_20'][method]:
            print(f'  step {row["step"]:02d}: path={row["selected"]} '
                  f'exact_minima={row["number_exact_minima"]} '
                  f'near_minima={row["number_near_minima"]} '
                  f'first_min_match={row["matches_exact_first_minimum"]}')
    print('NTAR vs Fixed differing decisions:', [r['step'] for r in mismatches])
    print('Replay all paths match the original first-minimum choice:',
          all(r['matches_exact_first_minimum'] for rows in tie_records.values() for r in rows))
    print('Wrote', OUTPUT)


if __name__ == '__main__':
    run_audit()
