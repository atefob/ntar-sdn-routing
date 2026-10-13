"""Figures for the revised NTAR manuscript, all drawn from results/results.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.json"))
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def shade(ax):
    ax.axvspan(7.5, 11.5, color="#d9d8d2", alpha=0.6, lw=0)
    ax.axvspan(14.5, 19.5, color="#fbd9cb", alpha=0.7, lw=0)


# ---------------------------------------------------------------- Fig. 2
tn, tf = res["trace_ntar"], res["trace_fixed"]
steps = [t["step"] for t in tn]
fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)

ax = axs[0]
shade(ax)
ax.step(steps, [t["hops"] for t in tf], where="mid", color=BLUE, lw=2, label="Fixed-weight")
ax.step(steps, [t["hops"] + 0.08 for t in tn], where="mid", color=ORANGE, lw=2, label="NTAR")
ax.set_ylabel("Hops of selected path")
ax.set_yticks([2, 3, 4])
ax.set_ylim(1.6, 4.5)
ax.text(9.5, 4.3, "node 3 down", ha="center", va="top", fontsize=7, color=MUTED)
ax.text(17, 4.3, "threat = 1", ha="center", va="top", fontsize=7, color=MUTED)
ax.legend(loc="lower right", fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.0))
ax.set_title("(a)", loc="left", fontsize=8, color=INK)

ax = axs[1]
shade(ax)
ax.plot(steps, [t["h_avail"] for t in tf], color=BLUE, lw=2)
ax.plot(steps, [t["h_avail"] for t in tn], color=ORANGE, lw=2)
ax.plot(steps, [t["h_obs"] for t in tn], color=ORANGE, lw=1.2, ls=(0, (3, 2)))
ax.text(3.2, 1.0, r"NTAR, Eq. (2) $H_{obs}$ (dashed)", fontsize=7, color=MUTED, va="bottom")
ax.set_ylabel("Normalized entropy")
ax.set_xlabel("Time step")
ax.set_ylim(0, 1.12)
ax.set_xticks([1, 5, 8, 12, 15, 20, 25])
ax.set_title("(b)", loc="left", fontsize=8, color=INK)
fig.tight_layout(h_pad=0.6)
fig.savefig("figures/fig2_trace.pdf")
fig.savefig("figures/fig2_trace.png", dpi=300)

# ---------------------------------------------------------------- Fig. 3
gen = res["generalization"]
forms = ["eq3", "eq5_withR", "eq3_noR", "eq5"]
labels = ["Original cost\nwith $R(p)$", "Normalized\nwith $R(p)$", "Unnormalized\nno $R(p)$", "NTAR cost"]
pct = [100 * np.mean([g[f"differs_{f}"] > 0 for g in gen]) for f in forms]
en = [np.mean([g[f"energy_nominal_{f}"] for g in gen]) for f in forms]
cols = [GRAY, GRAY, BLUE, ORANGE]

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, vals, ylab, fmt in ((axs[0], pct, "Topologies where threat\nweights change routing (%)", "{:.0f}%"),
                            (axs[1], en, "Mean energy per step,\nnominal phases (J)", "{:.3f}")):
    x = np.arange(len(forms))
    bars = ax.bar(x, vals, width=0.6, color=cols, edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom",
                fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylab)
    ax.grid(axis="x", visible=False)
axs[0].set_ylim(0, 105)
axs[1].set_ylim(0, max(en) * 1.18)
axs[0].set_title("(a) Weight sensitivity", loc="left", fontsize=8)
axs[1].set_title("(b) Efficiency of the nominal profile", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig3_costform_ablation.pdf")
fig.savefig("figures/fig3_costform_ablation.png", dpi=300)

# ---------------------------------------------------------------- Fig. 4: adversaries vs window length
R2 = json.load(open("results/results2.json"))
series = [("fixed", "Fixed-Weight", BLUE, "o"), ("ntar", "NTAR (deterministic)", ORANGE, "s"),
          ("ntar_r0.1", r"NTAR-R ($\tau=0.1$)", AQUA, "D"), ("random", "Random", GRAY, "^"),
          ("mtd_M3", "Periodic MTD, $M=3$", "#e87ba4", "v")]
Ls = [5, 10, 20]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, (key, title) in zip(axs, (("f1", "(a) Frequency adversary"), ("m1", "(b) Markov adversary"),
                                  ("pa1", "(c) Policy-aware adversary"))):
    for lab, name, col, mk in series:
        y = [R2[f"adv30_L{L}"][lab][key][0] for L in Ls]
        e = [R2[f"adv30_L{L}"][lab][key][1] for L in Ls]
        ax.errorbar(Ls, y, yerr=e, color=col, marker=mk, ms=4, lw=1.6, capsize=2, label=name)
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xticks(Ls)
    ax.set_xlabel("Threat-window length (steps)")
    ax.set_ylim(-0.03, 1.05)
axs[0].set_ylabel("Top-1 hit rate (threat window)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("figures/fig4_adversaries.pdf")
fig.savefig("figures/fig4_adversaries.png", dpi=300)
print("ok")
