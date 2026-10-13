"""LaTeX tables for the revised NTAR manuscript, generated from results/results.json
(ntar_sim.py) and results/results2.json (experiments2.py), so that no number is
transcribed by hand. Writes results/tables.tex; each table is preceded by a marker comment
'% ---------------- TABLE <n> ----------------'."""
import json
import math

import numpy as np
from scipy import stats

res = json.load(open("results/results.json"))
R2 = json.load(open("results/results2.json"))
out = []


def pm(m, s, d):
    m = 0.0 if abs(m) < 5e-4 else m
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def ci(v, d):          # v = [mean, half-width]
    return pm(v[0], v[1], d)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return [float(x.mean()), 0.0]
    return [float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))]


def wil(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return float("nan")
    return float(stats.wilcoxon(d).pvalue)


def ptxt(p):
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def table(num, env, caption, label, colspec, header, rows, width="\\textwidth", note=None):
    body = "\n".join(rows)
    note_tex = f"\\\\[3pt]\n\\parbox{{{width}}}{{\\footnotesize {note}}}\n" if note else ""
    out.append(f"""% ---------------- TABLE {num} ----------------
\\begin{{{env}}}[!t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering
\\footnotesize
\\resizebox{{{width}}}{{!}}{{\\begin{{tabular}}{{{colspec}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
{note_tex}\\end{{{env}}}
""")


# ---------------------------------------------------------------- Table 1: per-phase
rows = [f"{r['phase'].replace('&', chr(92) + '&')} ({r['steps'].replace('-', '--')}) & {r['energy']:.4f} & "
        f"{r['latency']:.2f} & {r['h_avail_end']:.3f} & {r['reward']:.4f} & {r['distinct']} / {r['length']} \\\\"
        for r in res["table1"]]
table(1, "table", "Per-phase breakdown of the representative NTAR run (seed 0). Entropy values are cumulative at the end of each phase.",
      "tab:phase", "lccccc",
      r"Phase (steps) & Energy (J) & Latency (ms) & $H_{avail}$ & Reward & Distinct paths \\",
      rows, width="\\columnwidth")

# ---------------------------------------------------------------- Table 2: benchmark
NAMES = {"static": "Static Shortest-Path", "random": "Random Path Selection",
         "fixed": "Fixed-Weight (ablation)", "ntar": "NTAR (deterministic)"}
t2 = res["table2"]
rows = []
for m in ("static", "random", "fixed", "ntar"):
    v = t2[m]
    rows.append(f"{NAMES[m]} & {pm(*v['energy'], 4)} & {pm(*v['latency'], 2)} & {pm(*v['h_avail'], 3)} & "
                f"{pm(*v['h_obs'], 3)} & {pm(*v['reward'], 4)} & {pm(*v['unique'], 1)} & {pm(*v['adv_hit_threat'], 2)} \\\\")
table(2, "table*",
      r"Comparative performance on the six-node topology (mean $\pm$ std over $N=20$ runs, 25 steps). Entropy columns are per-step averages: "
      r"$H_{avail}$ is normalized by the 13 available paths (Eq.~(\ref{eq:havail})) and $H_{obs}$ is Eq.~(\ref{eq2}). "
      r"The hit rate is that of the frequency-based observer over the five threat-window steps; Section~\ref{sec:results}-\ref{sec:adv} evaluates stronger adversaries.",
      "tab:bench", "lccccccc",
      r"Method & Energy (J) & Latency (ms) & $H_{avail}$ & $H_{obs}$ & Reward & Unique paths & Freq.\ hit rate (threat) \\",
      rows)

# ---------------------------------------------------------------- Table 3: cost-form ablation
ab, gen = res["ablation"], res["generalization"]
LBL = {"eq3": r"Original cost, Eq.~(\ref{eq3}) (with $R(p)$)",
       "eq5_withR": r"Normalized, with $R(p)$",
       "eq3_noR": r"Unnormalized, without $R(p)$",
       "eq5": r"NTAR cost, Eq.~(\ref{eq5})"}
rows = []
for f in ("eq3", "eq5_withR", "eq3_noR", "eq5"):
    a = ab[f]
    nd = sum(x["differs_" + f] > 0 for x in gen)
    ne = np.mean([x["energy_nominal_" + f] for x in gen])
    rows.append(f"{LBL[f]} & {a['steps_where_adaptive_differs']} / 25 & {a['unique_fixed']} / {a['unique_adaptive']} & "
                f"{a['energy_fixed']:.4f} & {nd} / {len(gen)} & {ne:.4f} \\\\")
table(3, "table*",
      r"Cost-form ablation. Each row changes one ingredient of the cost function and compares the threat-adaptive controller with its fixed-weight twin. "
      r"Columns 2--4: six-node topology (seed 0); ``steps differing'' counts steps at which the two controllers select different paths. "
      r"Columns 5--6: 30 random topologies of 10--20 nodes, with nominal energy averaged over the threat-free steps of the fixed-weight run.",
      "tab:costform", "lccccc",
      r" & \multicolumn{3}{c}{Six-node topology} & \multicolumn{2}{c}{30 random topologies} \\" "\n"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
      r"Cost form & Steps differing & Unique paths (fixed / adapt.) & Energy, fixed (J) & Topologies differing & Nominal energy (J) \\",
      rows)

# ---------------------------------------------------------------- Table 4: generalization with CIs
rows = []
for n in (10, 15, 20):
    tr = [x for x in gen if x["n"] == n]
    c = lambda k, d: ci(ci95([x[k] for x in tr]), d)
    d_e = [x["ntar_energy"] - x["fixed_energy"] for x in tr]
    d_h = [x["ntar_adv_hit_threat"] - x["fixed_adv_hit_threat"] for x in tr]
    rows.append(f"{n} & {np.mean([x['n_paths'] for x in tr]):.1f} & {c('random_energy', 3)} & {c('fixed_energy', 3)} & "
                f"{c('ntar_energy', 3)} & {c('fixed_adv_hit_threat', 2)} & {c('ntar_adv_hit_threat', 2)} & "
                f"{ptxt(wil(d_e))} & {ptxt(wil(d_h))} \\\\")
table(4, "table*",
      r"Generalization to random topologies, original 25-step protocol. Each topology is averaged over 5 seeds; values are means $\pm$ 95\% confidence "
      r"half-widths across the 10 topologies of each size, which are the independent samples. The last two columns give Wilcoxon signed-rank "
      r"$p$-values for the paired NTAR$-$Fixed differences in energy and in the frequency-observer hit rate.",
      "tab:gen", "cc ccc cc cc",
      r" & & \multicolumn{3}{c}{Energy (J)} & \multicolumn{2}{c}{Freq.\ hit rate (threat)} & \multicolumn{2}{c}{Paired $p$, NTAR vs.\ Fixed} \\" "\n"
      r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}" "\n"
      r"$|V|$ & Paths & Random & Fixed & NTAR & Fixed & NTAR & Energy & Hit rate \\",
      rows)

# ---------------------------------------------------------------- Table 5: adversaries (30 topologies, L = 10)
LAB = [("static", "Static Shortest-Path"), ("random", "Random Path Selection"),
       ("mtd_M1", "Periodic MTD, $M=1$"), ("mtd_M3", "Periodic MTD, $M=3$"),
       ("fixed", "Fixed-Weight"), ("ntar", "NTAR (deterministic)"),
       ("ntar_r0.1", r"NTAR-R ($\tau=0.1$)")]
A = R2["adv30_L10"]
rows = [f"{nm} & {ci(A[k]['energy'], 3)} & {ci(A[k]['f1'], 2)} & {ci(A[k]['f3'], 2)} & {ci(A[k]['m1'], 2)} & "
        f"{ci(A[k]['m3'], 2)} & {ci(A[k]['pa1'], 2)} & {ci(A[k]['pa3'], 2)} \\\\" for k, nm in LAB]
table(5, "table*",
      r"Hit rates of three adversaries over a 10-step threat window (steps 15--24 of 40) on the 30 random topologies "
      r"(mean $\pm$ 95\% CI across topologies; 5 seeds per topology). Top-1 is the probability that the adversary's single guess is the selected path; "
      r"top-3 that the selected path is among its three guesses. Policy-aware hit rates are Bayes-optimal expected values given the policy, its state and the weights in use.",
      "tab:adv", "lccccccc",
      r" & & \multicolumn{2}{c}{Frequency} & \multicolumn{2}{c}{Markov} & \multicolumn{2}{c}{Policy-aware} \\" "\n"
      r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
      r"Method & Energy (J) & Top-1 & Top-3 & Top-1 & Top-3 & Top-1 & Top-3 \\",
      rows)

# ---------------------------------------------------------------- Table 6: imperfect threat signal
SC = [("perfect", "Perfect detection"), ("fa_0.1", "False alarms, $p_{fa}=0.1$"), ("fa_0.3", "False alarms, $p_{fa}=0.3$"),
      ("miss_0.3", "Missed detection, $p_{m}=0.3$"), ("miss_0.6", "Missed detection, $p_{m}=0.6$"),
      ("delay_2", "Detection delay, 2 steps"), ("delay_5", "Detection delay, 5 steps"), ("flip_0.1", "Flapping signal, $p_{f}=0.1$")]
sig = R2["signal"]
rows = []
for k, nm in SC:
    a, b, f = sig[k]["ntar"], sig[k]["ntar_r0.1"], sig[k]["fixed"]
    rows.append(f"{nm} & {a['energy_nom'][0]:.3f} & {a['f1'][0]:.2f} & {a['m1'][0]:.2f} & "
                f"{b['energy_nom'][0]:.3f} & {b['f1'][0]:.2f} & {b['m1'][0]:.2f} & {b['pa1'][0]:.2f} \\\\")
f = sig["perfect"]["fixed"]
rows.append(r"\midrule" "\n" f"Fixed-Weight (reference) & {f['energy_nom'][0]:.3f} & {f['f1'][0]:.2f} & {f['m1'][0]:.2f} & & & & \\\\")
table(6, "table*",
      r"Sensitivity to an imperfect threat signal (30 random topologies, 10-step threat window, means across topologies). "
      r"Nominal energy is measured on steps without a true threat, so it includes the cost of false alarms; hit rates are measured on the true threat window. "
      r"The policy-aware hit rate of deterministic NTAR is 1.00 in every scenario.",
      "tab:signal", "lccccccc",
      r" & \multicolumn{3}{c}{NTAR (deterministic)} & \multicolumn{4}{c}{NTAR-R ($\tau=0.1$)} \\" "\n"
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}" "\n"
      r"Threat signal & Nominal energy (J) & Freq.\ top-1 & Markov top-1 & Nominal energy (J) & Freq.\ top-1 & Markov top-1 & Policy-aware top-1 \\",
      rows)

# ---------------------------------------------------------------- Table 7: decoupled costs
D = R2["decoupled"]
rows = []
for model, nm in (("indep", "Independent"), ("anti", "Anti-correlated")):
    for k, mn in (("random", "Random"), ("fixed", "Fixed-Weight"), ("ntar", "NTAR"), ("ntar_r0.1", r"NTAR-R")):
        v = D[model]["rand30"][k]
        rows.append(f"{nm if k == 'random' else ''} & {mn} & {ci(v['energy'], 3)} & {ci(v['latency'], 1)} & "
                    f"{ci(v['f1'], 2)} & {ci(v['m1'], 2)} & {ci(v['pa1'], 2)} \\\\")
    if model == "indep":
        rows.append(r"\midrule")
table(7, "table*",
      r"Decoupled energy and latency (30 random topologies, 10-step threat window, 3 link-cost draws per topology; mean $\pm$ 95\% CI across topologies). "
      r"Independent: per-link energy $\sim U(0.02,0.08)$~J and latency $\sim U(4,16)$~ms drawn independently. "
      r"Anti-correlated: latency $=20-200\,e+\mathcal{N}(0,0.5)$~ms, so low-energy links are slow.",
      "tab:decoupled", "llccccc",
      r"Cost model & Method & Energy (J) & Latency (ms) & Freq.\ top-1 & Markov top-1 & Policy-aware top-1 \\",
      rows)

# ---------------------------------------------------------------- Table 8: scalability
rows = []
for s in R2["scalability"]:
    rows.append(f"{s['n']} & {s['degree']:.1f} & {ci(s['candidates'], 1)} & {ci(s['ms_enum'], 2)} & {ci(s['ms_yen16'], 2)} \\\\")
table(8, "table",
      r"Per-decision cost of the candidate search (Python/NetworkX, one CPU core; 5 graphs per row, source and destination three hops apart, "
      r"mean $\pm$ 95\% CI). Enumeration lists all simple paths of at most four hops; Yen lists the 16 shortest simple paths.",
      "tab:scale", "ccccc",
      r"$|V|$ & Mean degree & Candidates & Enumeration (ms) & Yen, $k=16$ (ms) \\",
      rows, width="\\columnwidth")

open("results/tables.tex", "w").write("\n".join(out))
print("wrote", len(out), "tables")
