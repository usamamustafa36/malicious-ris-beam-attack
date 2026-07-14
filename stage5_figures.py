"""Stage 5: publication-quality figures from saved artifacts.

Figures are authored at true IEEE column width (see figstyle) so on-page text is
readable. The results table (table_main.tex) is curated separately (it merges rows
from several stages) and is intentionally NOT regenerated here.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(FIG, exist_ok=True)
A = bd.ART
s2 = json.load(open(os.path.join(A, "stage2_attacks.json")))
s3 = json.load(open(os.path.join(A, "stage3_ris.json")))
s4 = json.load(open(os.path.join(A, "stage4_defense.json")))
fs.use_style(); C = fs.COLORS


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


# ---- Fig 1: white-box feature-space baselines vs epsilon ----
eps = s2["eps"]
f1 = [x["top1"] * 100 for x in s2["fgsm"]]; p1 = [x["top1"] * 100 for x in s2["pgd"]]
fse = [x["se_ratio"] * 100 for x in s2["fgsm"]]; pse = [x["se_ratio"] * 100 for x in s2["pgd"]]
fig, ax = fs.fig_2panel()
for a, (yf, yp, cw, ylab) in zip(ax, [(f1, p1, s2["cw"]["top1"] * 100, "Top-1 accuracy (%)"),
                                      (fse, pse, s2["cw"]["se_ratio"] * 100, "SE ratio (%)")]):
    a.plot(eps, yf, "o-", color=C["fgsm"], label="FGSM")
    a.plot(eps, yp, "s-", color=C["pgd"], label="PGD")
    a.axhline(cw, ls="--", color=C["cw"], lw=1.3, label="C&W ($c{=}1$)")
    a.set_xlabel(r"budget $\epsilon$"); a.set_ylabel(ylab); fs.despine(a)
fs.tag(ax[0], "(a)"); fs.tag(ax[1], "(b)")
ax[0].legend(loc="upper right")
save(fig, "fig_baselines.png")

# ---- Fig 2: malicious RIS vs aperture size M ----
M = s3["M_sweep"]
o1 = [x["top1"] * 100 for x in s3["opt"]]; r1 = [x["top1"] * 100 for x in s3["rand"]]
ose = [x["se_ratio"] * 100 for x in s3["opt"]]; rse = [x["se_ratio"] * 100 for x in s3["rand"]]
fig, ax = fs.fig_2panel()
for a, (yo, yr, cl, ylab) in zip(ax, [(o1, r1, s3["clean"]["top1"] * 100, "Top-1 accuracy (%)"),
                                      (ose, rse, s3["clean"]["se_ratio"] * 100, "SE ratio (%)")]):
    a.plot(M, yo, "o-", color=C["ris"], label="malicious RIS")
    a.plot(M, yr, "^--", color=C["rand"], label="random RIS")
    a.axhline(cl, ls=":", color=C["clean"], lw=1.2, label="no RIS")
    a.set_xscale("log"); a.set_xticks(M); a.set_xticklabels([str(m) for m in M])
    a.set_xlabel("RIS elements $M$"); a.set_ylabel(ylab); fs.despine(a)
fs.tag(ax[0], "(a)"); fs.tag(ax[1], "(b)")
ax[0].legend(loc="lower left")
save(fig, "fig_ris_sweep.png")

# ---- Fig 3: hardware-constrained b-bit phases (M=128) ----
bb = s3["bbit"]; order = ["1", "2", "3", "None"]; labels = ["1-bit", "2-bit", "3-bit", "cont."]
vals = [bb[k]["top1"] * 100 for k in order]; vse = [bb[k]["se_ratio"] * 100 for k in order]
fig, ax = fs.fig_1panel(); x = np.arange(len(order)); w = 0.38
ax.bar(x - w / 2, vals, w, color=C["ris"], label="Top-1 acc.")
ax.bar(x + w / 2, vse, w, color=C["se"], label="SE ratio")
ax.axhline(s3["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS (top-1)")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False); fs.despine(ax); ax.legend(loc="upper right", ncol=1)
save(fig, "fig_bbit.png")

# ---- Fig 4: defense (undefended vs defended) ----
conds = ["no RIS", "RIS $M{=}64$", "RIS $M{=}128$"]
und = [s4["clean"]["undefended"]["top1"] * 100] + [x["top1"] * 100 for x in s4["attack_Msweep"]["undefended"]]
dfd = [s4["clean"]["defended"]["top1"] * 100] + [x["top1"] * 100 for x in s4["attack_Msweep"]["defended"]]
fig, ax = fs.fig_1panel(); x = np.arange(len(conds)); w = 0.38
ax.bar(x - w / 2, und, w, color=C["undef"], label="undefended")
ax.bar(x + w / 2, dfd, w, color=C["def"], label="RIS-adv. training")
for i, (u, dd) in enumerate(zip(und, dfd)):
    ax.text(i - w / 2, u + 1.5, f"{u:.0f}", ha="center", fontsize=8)
    ax.text(i + w / 2, dd + 1.5, f"{dd:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False); fs.despine(ax); ax.legend(loc="upper right")
save(fig, "fig_defense.png")

print("Saved fig_baselines, fig_ris_sweep, fig_bbit, fig_defense")
