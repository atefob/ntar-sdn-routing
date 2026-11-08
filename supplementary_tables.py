#!/usr/bin/env python3
"""Build NTAR Tables 7, 8 and 10 from their JSON results and insert them into
results/tables.tex, which make_tables.py has already written with Tables 1-6 and 9.

Numbers are always read from results/table7_decoupled.json, results/table8_timing.json and
results/energy_matched_baseline.json; the script stops if a required file or field is
missing and never substitutes hard-coded values.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

MARKER = re.compile(r"(?m)^% ---------------- TABLE (\d+) ----------------[ \t]*$")
MODELS = (("indep", "Independent"), ("anti", "Anti-correlated"))
METHODS = ("Random", "Fixed-Weight", "NTAR", "NTAR-R")
SCALING_COLUMNS = ("all_candidates", "yen16_candidates", "ms_dfs_all",
                   "ms_yen_all", "ms_yen16")
MATCHED_METHODS = ("Fixed-Weight", "Random", "NTAR-R", "Energy-Matched Random")


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment output: {path}; run its experiment first")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pair(item: dict, precision: int = 3) -> str:
    mean = float(item["mean"])
    half = float(item["ci95_half_width"])
    if not (math.isfinite(mean) and math.isfinite(half) and half >= 0):
        raise ValueError(f"Invalid mean or CI half-width: {item}")
    # Use exactly the JSON mean/95% CI, with the precision used in the manuscript.
    if abs(mean) < 0.5 * 10**(-precision):
        mean = 0.0
    if abs(half) < 0.5 * 10**(-precision):
        half = 0.0
    return f"${mean:.{precision}f}\\pm{half:.{precision}f}$"


def _latex_table(number: int, caption: str, label: str, colspec: str,
                 header: str, rows: list[str]) -> str:
    if not rows:
        raise ValueError(f"No data for Table {number}")
    return "\n".join([
        f"% ---------------- TABLE {number} ----------------",
        r"\begin{table*}[!t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ])


def _table7(data: dict) -> str:
    d = data["design"]
    expected = {"independent_topologies": 30,
                "link_cost_draws_per_topology": 3,
                "routing_seeds_per_draw": 5}
    for key, value in expected.items():
        if int(d[key]) != value:
            raise ValueError(f"Table 7 design changed: {key}={d[key]}, expected {value}; update the caption")
    rows = []
    for mkey, model_name in MODELS:
        summary = data["models"][mkey]["method_summary"]
        for index, method in enumerate(METHODS):
            metrics = summary[method]
            cells = [_pair(metrics[key]) for key in ("energy", "latency", "f1", "m1", "pa1")]
            rows.append(f"{model_name if index == 0 else ''} & {method} & "
                        + " & ".join(cells) + r" \\")
        if mkey == "indep":
            rows.append(r"\midrule")
    return _latex_table(
        7,
        r"Decoupled link costs: 30 independent topologies, three independent link-cost draws "
        r"per topology and five routing seeds per draw. Mean $\pm$ 95\% confidence "
        r"half-width across independent topologies. Independent: "
        r"$e\sim U(0.02,0.08)$~J and $l\sim U(4,16)$~ms; anti-correlated: "
        r"$l=\max(1,20-200e+\mathcal{N}(0,0.5))$~ms.",
        "tab:decoupled", "llccccc",
        r"Cost model & Method & Energy (J) & Latency (ms) & Freq. top-1 & Markov top-1 & Policy-aware top-1 \\",
        rows,
    )


def _table8(data: dict) -> str:
    d = data["design"]
    if int(d["graphs_per_configuration"]) != 5 or int(d["max_candidate_hops_all_methods"]) != 4 or int(d["yen_k"]) != 16:
        raise ValueError("Table 8 design changed (graphs=5, hops=4, k=16 expected); update the caption")
    summaries = data["by_configuration"]
    rows = []
    seen = set()
    for item in sorted(summaries, key=lambda s: (int(s["nodes"]), float(s["degree"]))):
        n, degree = int(item["nodes"]), float(item["degree"])
        if (n, degree) in seen:
            raise ValueError(f"Duplicate scaling configuration {(n, degree)}")
        seen.add((n, degree))
        if item.get("candidate_set_equal_dfs_vs_yen_all") is not True:
            raise ValueError(f"DFS/Yen candidate sets differ for {(n, degree)}")
        values = [_pair(item[k], precision=2 if k in SCALING_COLUMNS[:2] else 3)
                  for k in SCALING_COLUMNS]
        rows.append(f"{n} & {degree:.1f} & " + " & ".join(values) + r" \\")
    if len(rows) != 8:
        raise ValueError(f"Expected eight Table 8 configurations, got {len(rows)}")
    return _latex_table(
        8,
        r"Four-hop-matched candidate generation plus route scoring "
        r"(Python/NetworkX, five graphs per size/density; mean $\pm$ 95\% "
        r"confidence half-width across graphs). Complete DFS and Yen yield the same "
        r"candidate path sets; Yen-16 returns at most 16 eligible paths. "
        r"Timings exclude graph construction and SDN control-plane communication.",
        "tab:scale", "ccccccc",
        r"$|V|$ & Mean degree & All paths & Yen-16 paths & DFS all (ms) & Yen all (ms) & Yen-16 (ms) \\",
        rows,
    )


def _table10(data: dict) -> str:
    stats = data["heldout_method_summary"]
    rows = []
    for method in MATCHED_METHODS:
        values = [_pair(stats[method][key]) for key in ("energy", "m1", "pa1")]
        rows.append(f"{method} & " + " & ".join(values) + r" \\")
    design = data["design"]
    if design.get("test_recalibration") is not False:
        raise ValueError("Table 10 source does not confirm absence of test-set recalibration")
    beta = float(data["calibration"]["beta"])
    if not math.isfinite(beta):
        raise ValueError("Invalid calibration beta")
    train, test = int(design["training_topologies"]), int(design["heldout_topologies"])
    if (train, test) != (30, 30) or int(design["routing_seeds_per_topology"]) != 5:
        raise ValueError("Table 10 design changed (30/30 topologies, 5 seeds expected); update the caption")
    return _latex_table(
        10,
        rf"Energy-matched stochastic routing baseline on {test} held-out topologies "
        rf"(five routing seeds per topology; mean $\pm$ 95\% confidence half-width "
        rf"across topologies). Gibbs random baseline was calibrated on a separate "
        rf"set of {train} training topologies ($\beta={beta:.6f}$) and fixed for held-out testing. "
        r"Lower adversary top-1 hit rates are better.",
        "tab:energymatched", "lccc",
        r"Method & Energy (J) & Markov top-1 & Policy-aware top-1 \\",
        rows,
    )


def add_supplementary_tables(results_dir: str | Path = "results") -> None:
    """Insert Tables 7, 8 and 10 into a freshly generated tables.tex."""
    root = Path(results_dir)
    t7 = _table7(_load(root / "table7_decoupled.json"))
    t8 = _table8(_load(root / "table8_timing.json"))
    t10 = _table10(_load(root / "energy_matched_baseline.json"))

    output = root / "tables.tex"
    if not output.is_file():
        raise FileNotFoundError(f"Missing {output}; run make_tables.py first")
    text = output.read_text(encoding="utf-8")
    starts = list(MARKER.finditer(text))
    if not starts:
        raise ValueError(f"No numbered table markers found in {output}")
    blocks = {}
    for idx, marker in enumerate(starts):
        number = int(marker.group(1))
        if number in blocks:
            raise ValueError(f"Duplicate TABLE {number} marker in {output}")
        end = starts[idx+1].start() if idx+1 < len(starts) else len(text)
        blocks[number] = text[marker.start():end].rstrip()
    if not {1, 2, 3, 4, 5, 6, 9} <= set(blocks):
        raise ValueError(f"Unexpected existing table markers: {sorted(blocks)}")
    blocks[7] = t7.rstrip()
    blocks[8] = t8.rstrip()
    blocks[10] = t10.rstrip()
    prefix = text[:starts[0].start()]
    combined = prefix + "\n\n".join(blocks[i] for i in range(1, 11)) + "\n"
    # Validate all 10 markers before atomic replacement.
    found = [int(x) for x in MARKER.findall(combined)]
    if found != list(range(1, 11)):
        raise AssertionError(f"Output table sequence invalid: {found}")
    temp = output.with_suffix(".tex.tmp")
    temp.write_text(combined, encoding="utf-8")
    temp.replace(output)
    print("wrote results/tables.tex: Tables 1-10")


if __name__ == "__main__":
    add_supplementary_tables()
